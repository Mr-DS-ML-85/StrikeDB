# APGC — GPU-Built, CPU-Served ANN Index

## Overview

APGC constructs a kNN graph entirely on the GPU in INT8, then serves queries from
the CPU. Index residence is managed by VUGVA's VRAM→DRAM→NVMe tiering.

The name is historical: "Adaptive Precision Graph Construction" described an
earlier design. What shipped adapts precision differently — see below — and the
paper (`research/APGC_Paper.tex`) is titled for what the system actually does.

## Precision: INT8 throughout, exact where it matters

Construction runs entirely in INT8 via `dp4a`. That is not a compromise, it is
the fast path — per instruction on `sm_89`:

| instruction | multiply-accumulates |
|---|---:|
| `dp4a` (INT8) | **4** |
| `half2` FMA (FP16) | 2 |
| FP32 FMA | 1 |

So a precision hierarchy assigning FP16 to the *majority* of distance work —
which an earlier design did — would roughly **halve** build throughput. INT8
everywhere is both simpler and faster.

Precision is instead spent where quantization error changes the answer:

- **Candidate rerank** — the graph is walked in INT8, and the retrieved
  candidates are rescored against exact FP32 vectors, fused into the same kernel
  launch. Touches O(k) vectors rather than the O(ef·degree) the walk visits.
- **Entry nodes** — every descent starts from the same few nodes, so error there
  propagates into every query while error at a leaf affects one.

The principle: **precision follows error propagation, not node rank.**

## Construction pipeline

Three phases, all INT8:

1. **Pivot assignment** — `min(n/50, 2048)` pivots by strided sampling; each node
   records its nearest and second-nearest pivot. The cap matters: a pivot set
   proportional to n makes this O(n²).
2. **Locality ordering** — sort by (pivot, 2nd pivot, distance) and wire each
   node to its neighbours in that 1-D order. **Zero distance computations.** Its
   job is a non-degenerate starting point plus cache-friendly node numbering.
3. **GPU NN-descent** — 8–12 passes; candidates from own list ∪ reverse list ∪
   forward/reverse joins, exact INT8 top-k retained, double-buffered on device.

## Search pruning (SelKV / Delta-AR)

Adjacency lists are sorted by node **hubness** — `delta = 0.1 + 0.9·indeg/max_indeg`,
i.e. normalized graph in-degree.

Two things worth stating plainly:

- This is a **graph-degree heuristic, not an LLM attention signal.** An earlier
  design routed OpusEdge Δ scores here; the shipped code does not.
- **Both gates are effectively off by default.** SelKV's threshold at
  `selkv_ratio=0.9` sits at the minimum possible δ, so nothing is pruned;
  Delta-AR (`delta_ar_k=0`) reads the full adjacency. Enabling either trades
  recall for negligible latency, which is why they are off.

The surviving effect is the δ-ordering of adjacency, which changes candidate
visit order.

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

| mode | build | vec/s | Recall@10 | QPS (1t) | QPS (16t) | QPS (batch) |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only | 7.31 s | 13,683 | **0.999** | 2,781 | **27,894** | 2,860 |
| Turbo | **2.46 s** | **40,723** | 0.994 | 8,590 | 72,044 | 22,116 |
| Hybrid (VUGVA) | 2.70 s | 37,081 | 0.994 | 7,576 | 76,983 | 18,542 |

All search columns at beam 128 so every mode does equal work per query.
Build **2.98×** (~3.5× steady-state; NVRTC compile is inside the timer).
Batched GPU is **7.73× one CPU core but 0.79× the 16-thread CPU** — on this
hardware a saturated CPU beats the device at search.

**What this shows, including what it does not.**

*Build is where the GPU wins.* 2.46 s against 7.31 s — **2.98×**, and ~3.5×
steady-state since NVRTC compilation (~0.39 s) is charged inside the timer — at
a cost of 0.005 recall. Phase 1 (pivot assignment) and the NN-descent refine both run on
device; the CPU only wires edges.

*Batched GPU search does not beat a saturated CPU.* 22,116 QPS against 2,860 on
one core (7.73×) but 27,894 across sixteen (0.79×). An earlier revision claimed
12.4×; that compared an unequal beam (`search_many` dropped `ef` to 64 against
the CPU's 128) to a single-threaded baseline, and is retracted. Batch recall is
also unverified — recall is measured through `search_ef` only.

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
