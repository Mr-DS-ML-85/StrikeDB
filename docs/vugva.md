# VUGVA — Virtual Unified GPU VRAM Architecture

## Overview

VUGVA provides software-defined memory tiering that moves data transparently across GPU VRAM, system RAM, and NVMe. Designed for large-scale vector search where the index exceeds GPU memory.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  VugvaVmt                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Chunk 0  │  │ Chunk 1  │  │ Chunk N  │       │
│  │ Tier=VRAM│  │ Tier=RAM │  │ Tier=NVMe│       │
│  │ Hot      │  │ Warm     │  │ Cold     │       │
│  └──────────┘  └──────────┘  └──────────┘       │
├──────────────────────────────────────────────────┤
│  LruEvictor          │  LookaheadTracker          │
│  VRAM→RAM eviction   │  Predict upcoming chunks  │
│  Watermark-based     │  Stride detection         │
├──────────────────────────────────────────────────┤
│  Tier 1: GPU VRAM    ~1µs    Limited (8GB)       │
│  Tier 2: System RAM  ~100µs  Large (32GB)        │
│  Tier 3: NVMe SSD    ~10ms   Unlimited           │
└──────────────────────────────────────────────────┘
```

## API

```rust
use vugva::{VugvaConfig, VugvaVmt, Chunk, Tier, LruEvictor, LookaheadTracker};

// Create memory manager
let mut vmt = VugvaVmt::new(VugvaConfig::default());

// Insert chunks into VRAM
vmt.insert(Chunk::new(0, Tier::Vram, vec![0u8; 256_000])); // 256KB chunk
vmt.insert(Chunk::new(1, Tier::Vram, vec![0u8; 256_000]));

// Check if eviction needed
if vmt.needs_eviction() {
    let mut evictor = LruEvictor::new();
    evictor.evict_to_target(&mut vmt);
}

// Promote from RAM to VRAM
let data = vmt.promote_to_vram(chunk_id);

// Prefetch upcoming chunks
let mut pf = LookaheadTracker::new(256, 16);
pf.record_access(chunk_id);
let predictions = pf.predict(); // chunk IDs to prefetch
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| chunk_size | 256 KB | Fixed chunk size |
| vram_capacity | 6 GB | Max VRAM (leave 2GB for model) |
| ram_capacity | 32 GB | Max system RAM |
| nvme_capacity | 0 (unlimited) | Max NVMe storage |
| prefetch_window | 16 | Chunks to predict ahead |
| lru_high_watermark | 0.85 | Evict when VRAM > 85% |
| lru_low_watermark | 0.70 | Target after eviction |

## Performance

### Control plane (metadata only)

These are page-table and policy operations — no tensor data moves, which is why
the throughputs are so large. They measure the *bookkeeping*, not the transfer.

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Insert (1000 chunks) | 35.5 µs/batch | 28B chunks/s |
| Lookup (10000) | 74.8 µs/batch | 134B lookups/s |
| Eviction cycle | 1.2 µs/chunk | 85K cycles/s |
| Prefetch prediction | 68.7 ns | 14.5M predicts/s |

### Data plane (actual transfers, RTX 4060 / PCIe 4.0 ×8)

Verified by `cargo test --test hardware` in the `vugva` crate:

| Path | Status | Evidence |
|---|---|---|
| DRAM (T1) → VRAM (T0) | working, page-locked | `paper_dram_tier_promotes_at_dma_speed` — the pool asserts it is genuinely pinned; an unpinned pool silently degrades to a synchronous staged copy at roughly half the rate |
| VRAM → DRAM (demote) | working | `paper_demote_writes_back_and_reclaims_vram` — byte-exact writeback, device block recycled |
| NVMe (T2) → DRAM → VRAM | working | `paper_ssd_tier_promotes_cold_pages`, `paper_ssd_tier_round_trips_written_bytes` — 2 MiB written, spilled, promoted, byte-exact against a 4 MiB DRAM pool |
| Larger-than-DRAM corpus | working | a corpus 1.5× the DRAM budget streams to NVMe and pages back through bounded staging |

The bandwidth figure is deliberately not quoted against the paper's 28–58 GB/s:
that range is server-class hardware, and this box is a single-socket desktop on
PCIe 4.0 ×8 whose practical ceiling is ~13 GB/s. The pinned path essentially
reaches it. Claiming the paper's number on this hardware would be dishonest.

### Not yet implemented

- **Look-Ahead prefetch and the DMA descriptor ring are scaffolding.** Promotion
  works but is synchronous — there is no overlap of transport with compute, so
  the paper's §3.2 latency hiding is design intent, not shipped behaviour.
- **End-to-end larger-than-VRAM search is unbenchmarked.** The memory layer is
  covered by the tests above, but `search_ef` takes the device path only under
  `Turbo`, and `Turbo` uses a plain VRAM allocation rather than the tier. So no
  benchmark yet exercises tiering under query load.

## Use Cases

1. **Vector search > VRAM**: Graph index exceeds GPU memory, tiered across RAM/NVMe
2. **Multi-GPU clusters**: Route data between GPUs without CPU involvement
3. **Streaming ingestion**: New vectors enter RAM, hot vectors promoted to VRAM
4. **Cold data archival**: Frequently unused graph nodes evicted to NVMe
