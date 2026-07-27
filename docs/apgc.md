# APGC — Adaptive Precision Graph Construction

## Overview

APGC builds kNN graphs using mixed-precision distance computation, adapting precision to node importance. Combined with KV-aware search pruning from OpusEdge and memory tiering from VUGVA.

## Precision Hierarchy

```
Node Importance (top → bottom):
┌─────────────────────────────────────┐
│  Seed nodes (1%)     → FP32 (4B)   │  Maximum precision
│  High-importance (9%) → BF16 (2B)   │  Wide exponent range
│  Majority (80%)      → FP16/BF16   │  Tensor Core accelerated
│  Near-outlier (9%)   → FP8 (1B)    │  4× memory reduction
│  Outlier (1%)        → INT8 (1B)   │  Maximum compression
└─────────────────────────────────────┘
Overall: 50% memory savings vs FP32-only
```

## Algorithm

```
Input: V[N][D], precision thresholds, GPU capabilities
Output: kNN graph G with mixed-precision edges

Phase 1: Seed Initialization
  seeds = random_sample(V, 1%)
  G_seed = build_knn(seeds, k, FP32)  // Full precision for critical nodes

Phase 2: Precision Assignment
  for each node v in V:
    if v is seed:        precision = FP32
    elif v < 90th %ile:  precision = BF16 (or FP16 on V100)
    elif v < 99th %ile:  precision = FP8 (or FP16 on V100)
    else:                precision = INT8

Phase 3: Graph Expansion
  for each node v in V \ seeds:
    distances[v][*] = compute_distance(v, V, precision[v])
    G.add_edges(v, top_k(distances[v]))
```

## KV-Aware Search Pruning

Uses LLM attention patterns to guide graph traversal:

1. **Attention Scoring**: OpusEdge provides per-token importance via Δ signal
2. **Region Weighting**: High-attention graph nodes are expanded at FP32 precision
3. **Precision Selection**: Low-attention nodes use INT8 or are skipped entirely
4. **Beam Pruning**: Reduce search beam from N to K based on attention scores

```
Search Pruning Results (8192 candidates):
  Input:  8192 candidates
  Output: 32 pruned candidates (99.6% reduction)
  Latency: 68.8 µs (14.5K prunes/s)
```

## GPU Integration

### Kernel Loading (GPU-Aware)

APGC only compiles CUDA kernels for formats supported by the detected GPU:

```rust
let caps = GpuCaps::detect();
// On V100: only compile FP32, FP16, INT8 kernels
// On Ada: compile all 5 formats (FP32, FP16, BF16, FP8, INT8)

for prec in caps.supported_precisions() {
    compile_distance_kernel(prec);  // Only load what's needed
}
```

### Batch Distance Computation

GPU-accelerated distance computation using Tensor Cores:

| Format | Tensor Cores | Throughput (RTX 4060) |
|--------|-------------|----------------------|
| FP32 | No | 1× (baseline) |
| BF16 | Yes (sm_80+) | 2× |
| FP16 | Yes (sm_70+) | 2× |
| FP8 | Yes (sm_89+) | 4× |
| INT8 | Yes (sm_70+) | 4× |

## Memory Budget

| Component | VRAM (1M × 384d) | RAM | Total |
|-----------|-------------------|-----|-------|
| FP32 graph only | 3.6 GB | 0 | 3.6 GB |
| APGC mixed-precision | 1.8 GB | 0 | 1.8 GB |
| APGC + VUGVA tiered | 0.6 GB | 1.2 GB | 1.8 GB |

## Measured performance

Real embeddings, 100k × 384-d, RTX 4060 (8 GB) + Ryzen 7700 (16 threads).
Reproduce with `dbstrike-bench --gpu-bench <vectors.fbin>` (~25 s).

| mode | build | vec/s | Recall@10 | QPS (1t) | QPS (16t) | QPS (batch 256) |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only | 7.16 s | 13,964 | **0.999** | 2,900 | 23,675 | 2,832 |
| Turbo | **2.57 s** | **38,916** | 0.994 | **7,607** | **86,641** | **35,198** |
| Hybrid (VUGVA) | 2.85 s | 35,105 | 0.994 | 8,288 | 80,133 | 27,747 |

Turbo vs CPU-only: build **2.79×** · 1t **2.62×** · 16t **3.66×** · batch **12.43×**

**What this shows, including what it does not.**

*Build is where the GPU wins.* 2.57 s against 7.16 s — **2.79×** — at a cost of
0.005 recall. Phase 1 (pivot assignment) and the NN-descent refine both run on
device; the CPU only wires edges.

*Only the batch column is GPU search: 12.4×.* 35,198 QPS against the CPU's
2,832. A graph query is one CUDA block, so a lone query leaves 23 of 24 SMs idle
— and 16 client threads is still only 16 blocks, which is why raising thread
count never rescued it. Single queries are therefore routed to the CPU in every
mode; `DBSTRIKE_GPU_SINGLE=1` forces the device path and reproduces the 0.61×
regression. Ruled out as causes: the coalescer (`GPU_COALESCE=0` moves it ~1%)
and a shared-memory cliff (7–12 KB used against the 48 KB limit).

*The 1t and 16t columns measure graph quality, not hardware.* Every row runs the
same CPU search code there, and Turbo still wins 2.62× and 3.66× because the
APGC graph — flat, degree-54, locality-ordered — is cheaper to walk than a
CPU-built M=32 hierarchy. A CPU-only deployment keeps most of that gain provided
the index was built on a GPU.

*Hybrid now takes the device path for batches too.* `search_many` was gated on
`Turbo` alone, so the mode whose whole purpose is serving a corpus through VUGVA
uploaded that corpus and never read it. Both GPU modes now batch on device.

*The tiering is not stressed here.* 36 MB fits VRAM, so nothing demotes or
spills. T2 has unit and hardware coverage (a corpus 1.5× the DRAM budget writes
to NVMe and promotes back byte-exact) but no end-to-end benchmark.

Not yet measured: 1M × 384-d and 1M × 768-d in any GPU mode.

## CAGRA Comparison

APGC **replaces** CAGRA rather than extending it; the rows below are design
differences, not benchmark results. Only the table above is measured.

| Feature | CAGRA | APGC |
|---------|-------|------|
| Precision | FP32 only | FP32/BF16/FP16/FP8/INT8 |
| Graph construction | NN-Descent | Mixed-precision NN-Descent |
| Search guidance | Query-independent | KV-aware (Δ signal) |
| Memory scaling | VRAM only | VRAM + RAM + NVMe (VUGVA) |
| GPU format detection | None | Auto-detect per GPU generation |
| Dynamic updates | Static rebuild | Streaming via VUGVA |
