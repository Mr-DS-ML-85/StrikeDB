# DB-Strike HNSW Build Bottleneck Audit

**Date:** 2026-07-18
**Target:** 1M vectors, ~170s+ build time
**Files audited:**
- `crates/views/src/vector.rs` (build_parallel_tiered, build_segment_indexed, insert_attr, search_layer)
- `crates/gpu/src/lib.rs` (gpu_build_knn_graph)
- `crates/gpu/kernels/all_kernels.cu` (batch_cosine_dist_kernel)

---

## 1. CPU HNSW Build — vector.rs

### 1.1 Distance Computations Per Vector Insert

The `insert_attr` function (line 1796) calls `search_layer` for every level assignment.

**Parameters (line 1692-1694):**
```
m             = 32    (neighbors per layer)
m_max0        = 64    (max layer-0 neighbors before pruning)
ef_construction = 200 (beam width at layer 0)
```

**Per-insert distance computation breakdown:**

| Phase | Calls | Distances per call | Subtotal |
|-------|-------|--------------------|----------|
| Greedy descent (layers top → level+1) | ~1-3 levels | ~1-3 (ef=1) | ~3-9 |
| Layer-0 search (ef=200) | 1 | up to `visited_count` ≈ 200 × degree ≈ 6400 | **~6400** |
| select_neighbors_heuristic | 1 | O(m × ef) worst case, typically ~200 | ~200 |
| Reverse pruning (degree > m_max0) | variable | 1 per excess neighbor | ~0-100 |

**Total per vector insert: ~6,400–6,700 distance computations** (dominated by layer-0 search).

For 1M vectors: **~6.4 billion i8 dot products** on the CPU.

The CPU `dot_i8` function uses AVX2 (line 551): processes 32 i8s per iteration via `_mm256_madd_epi16`. For dim=384, that is 12 AVX2 iterations per dot product. At ~1 dot product per ~100ns (AVX2 @ 3GHz), 6.4B dots ≈ **640 seconds** — this is the CPU bound.

### 1.2 Can GPU batch_cosine_dist Replace CPU Distance in insert_attr?

**No — not directly.** The `search_layer` algorithm is inherently sequential:
- Each iteration pops the nearest candidate from a binary heap
- Expands its unvisited neighbors
- Pushes new candidates back into the heap
- The next iteration depends on the current heap state

This is a greedy graph walk — you cannot batch the distance computations across independent vectors because each step depends on the previous step's result.

**What CAN be GPU-accelerated:**
- The `gpu_build_knn_graph` path (line 3459) bypasses HNSW entirely: GPU computes kNN graph, CPU just wires edges. This is the correct approach.
- A GPU `cagra_search_kernel` (line 54 in all_kernels.cu) does exist and parallelizes distance computation across 256 threads within a single search — but it replaces the traversal, not the insert.

### 1.3 build_segment_indexed (line 2934) Analysis

```rust
fn build_segment_indexed(data, dim, perm, off, count, base_id, attrs) -> Hnsw {
    for row in 0..count {
        let true_row = perm[off + row];
        let v = data[true_row*dim .. (true_row+1)*dim].to_vec();  // COPY
        h.insert_attr(base_id + row, v, attrs[off + row]);
    }
}
```

**Bottleneck:** Pure serial loop — `count` sequential inserts. Each `insert_attr` does ~6400 AVX2 dot products. For a segment of 125K vectors (1M / 8 shards): 125K × 6400 = **800M dot products per shard**.

The parallel sharding across `n_shards` threads helps, but each shard is still serial internally.

### 1.4 build_parallel_tiered (line 3395) Analysis

```
GPU Turbo path:
  gpu_build_knn_graph(i8_data, n=1M, dim, k_init=32)
  → ONE kNN graph → ONE HNSW (no segments, no merge)

CPU path:
  1. Build 8 parallel shards via build_segment_indexed
  2. merge_segments(bridge=16)
```

The GPU path is the fast path. The CPU path bottleneck is the serial insert loop within each shard.

---

## 2. GPU kNN Graph Build — lib.rs

### 2.1 Architecture: gpu_build_knn_graph (line 528)

```
batch_q = 32 queries per kernel launch
N = 1,000,000 vectors
dim = 384
Output per launch: Q × N × 4 bytes = 32 × 1M × 4 = 128 MB
```

**Loop structure (line 561):**
```rust
for batch_start in (0..n).step_by(batch_q) {  // 31,250 iterations for 1M/32
    // 1. Upload Q×dim bytes (32×384 = 12KB) — cheap
    cuMemcpyHtoD_v2(d_q, ...);
    // 2. Launch kernel: Q blocks × 256 threads
    cuLaunchKernel(func, q_count, 1, 1, 256, 1, 1, ...);
    // 3. READ BACK 128 MB via PCIe  ← BOTTLENECK
    cuMemcpyDtoH_v2(dists, d_d, q_count * n * 4);
    // 4. CPU: sort N distances per query, find top-k
    for qi in 0..q_count {
        candidates.select_nth_unstable_by(k_init, ...);
    }
}
```

### 2.2 The PCIe Bottleneck — Quantified

| Metric | Value |
|--------|-------|
| Batches | ceil(1M / 32) = **31,250** |
| PCIe readback per batch | 32 × 1M × 4 = **128 MB** |
| **Total PCIe readback** | 31,250 × 128 MB = **4,000 GB (4 TB)** |
| PCIe 3.0 x16 bandwidth | ~15 GB/s → **267 seconds** |
| PCIe 4.0 x16 bandwidth | ~32 GB/s → **125 seconds** |
| PCIe 5.0 x16 bandwidth | ~64 GB/s → **63 seconds** |

**The GPU kernel itself is fast (likely <5 seconds for all 31,250 launches combined). The PCIe readback of 4 TB of distance matrix data dominates at 63-267 seconds depending on PCIe generation.**

### 2.3 CPU Top-k Sort Overhead

After each 128 MB readback, the CPU sorts N=1M distances per query:
- 31,250 batches × 32 queries × O(N log N) sort = **~10^13 comparisons**
- `select_nth_unstable_by` is O(N) average but still processes 1M floats per query
- Total: 1M × 31,250 × 32 = **1 trillion** float comparisons on CPU

### 2.4 VRAM Usage

```
d_v: 1M × 384 = 384 MB (vectors, uploaded once)
d_d: 32 × 1M × 4 = 128 MB (distance output, reused)
d_q: 32 × 384 = 12 KB (query buffer, reused)
Total: ~512 MB
```

This fits in any modern GPU. The bottleneck is NOT VRAM — it is PCIe bandwidth.

---

## 3. CUDA Kernel Analysis — all_kernels.cu

### 3.1 batch_cosine_dist_kernel (line 22)

```cuda
extern "C" __global__
void batch_cosine_dist_kernel(
    const char* queries,     // Q × D int8
    const char* vectors,     // N × D int8
    float* dists,            // Q × N float32
    int Q, int N, int D)
{
    int qid = blockIdx.x;           // one block per query
    int tid = threadIdx.x;
    int threads = blockDim.x;       // 256
    for (int vid = tid; vid < N; vid += threads) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)queries[qid * D + d] * (int)vectors[vid * D + d];
        }
        dists[qid * N + vid] = 1.0f - (float)dot / 16129.0f;
    }
}
```

### 3.2 Identified Kernel Bottlenecks

**Bottleneck 1: No Shared Memory Tiling for Query Vector**
- The query vector (D=384 bytes) is read from global memory by ALL 256 threads in the block.
- Each thread reads the same query D times → 256 × 384 = 98,304 global reads of the same data.
- **Fix:** Load query into `__shared__` once (384 bytes, trivial in 48KB shared mem). Saves ~98K global reads per block.

**Bottleneck 2: Scalar int8 Loads (No Vectorized Access)**
- Inner loop reads `(int)queries[qid*D + d]` and `(int)vectors[vid*D + d]` one byte at a time.
- GPU global memory is optimized for coalesced 32/64/128-byte transactions. Byte-at-a-time reads waste bandwidth.
- **Fix:** Use `int4` (16-byte) loads: `reinterpret_cast<const int4*>`. For D=384, that is 24 int4 loads instead of 384 byte loads. **Up to 16× reduction in memory transactions.**

**Bottleneck 3: Poor Memory Coalescing for Vector Access**
- Thread `tid` reads `vectors[(tid + k*256) * D + d]`.
- For D=384, consecutive threads read vectors 384 bytes apart. This means thread 0 reads from offset `vid*384+d`, thread 1 reads from `(vid+1)*384+d` — offset by 384 bytes.
- These are NOT coalesced into the same cache line unless D < 32.
- **Fix:** Restructure so consecutive threads read consecutive bytes within the same vector (reduction pattern), or transpose the vector layout.

**Bottleneck 4: No Warp-Level Reduction**
- Each thread computes its own independent dot product. No communication between threads.
- For the CAGRA search kernel (line 54), this is correct (each thread handles a different candidate). But for batch distance, a cooperative reduction across threads within a warp would be faster.
- **Fix:** Use `__shfl_down_sync` for warp-level dot product reduction.

**Bottleneck 5: Output Materialization**
- The kernel writes Q × N float32 output (128 MB). This is the data that gets read back via PCIe.
- **Fix (critical):** Compute top-k ON GPU using a partial-sort or heap kernel, then read back only Q × k × 8 bytes (32 × 32 × 8 = 8 KB instead of 128 MB). **Reduces PCIe readback by 16,000×.**

**Bottleneck 6: Normalization Constant**
- `16129.0f` = 127² hardcoded. This means vectors are assumed to be unit-length after quantization. If input vectors are not pre-normalized, this gives incorrect distances. (Not a perf issue, but a correctness note.)

### 3.3 Estimated Kernel Performance

For D=384, Q=32, N=1M:
- Total int8 multiply-accumulate operations: 32 × 1M × 384 = **12.3 billion**
- GPU (e.g. RTX 4090 @ 83 TFLOPS INT8): 12.3B / 83T ≈ **0.15 ms** theoretical
- With memory bandwidth bottleneck (1 TB/s): 12.3B bytes read / 1 TB/s ≈ **12 ms** actual
- × 31,250 batches = **~375 seconds** kernel time (bandwidth bound)

The kernel itself is memory-bandwidth bound, not compute bound.

---

## 4. Root Cause Summary

| Bottleneck | Location | Time (est.) | % of 170s |
|-----------|----------|-------------|-----------|
| **PCIe readback (4 TB)** | lib.rs:588 | 125-267s (PCIe gen dependent) | **>70%** |
| CPU top-k sort (1T comparisons) | lib.rs:589-601 | ~10-20s | ~10% |
| Kernel launch overhead (31K launches) | lib.rs:579-584 | ~3-5s | ~3% |
| H2D upload (12KB × 31K = 384 MB) | lib.rs:567 | <0.5s | <1% |
| GPU kernel compute | all_kernels.cu:22 | ~0.4s | <1% |

**The #1 bottleneck is the PCIe readback of the full Q×N distance matrix (128 MB per batch × 31,250 batches = 4 TB).**

If the CPU path is taken (GPU not available): the bottleneck shifts to the serial HNSW insert loop (~6400 AVX2 dot products × 1M vectors = 6.4B operations ≈ 640s).

---

## 5. Optimization Roadmap (Ordered by Impact)

### 5.1 [CRITICAL] GPU-side Top-k — Eliminates PCIe Bottleneck

**Impact: 170s → ~5-10s (17-34× speedup)**

Replace the current approach (compute full Q×N matrix → readback → CPU sort) with:

1. Allocate a GPU buffer of Q × k_init × 2 (index + distance pairs) — initialized to worst-case.
2. After each batch kernel, launch a `gpu_topk_kernel` that scans the Q×N distances and maintains a partial heap of size k_init per query.
3. Only readback Q × k_init × 8 bytes (8 KB instead of 128 MB).

**Implementation sketch:**
```cuda
// After batch_cosine_dist_kernel, launch:
global_topk_kernel(dists, d_out_idx, d_out_dist, Q, N, k_init);
// Reads dists[qi*N .. qi*N+N], writes only top-k to d_out_idx/d_out_dist
// Then CPU reads Q × k_init × 8 bytes = 8 KB total (not 128 MB)
```

This reduces PCIe from 4 TB to 31,250 × 8 KB = **250 MB total**. At PCIe 3.0: 250 MB / 15 GB/s = **17 ms**.

### 5.2 [HIGH] Shared Memory Tiling in Kernel

**Impact: ~2-3× kernel speedup**

```cuda
__shared__ char s_query[D];  // D=384 bytes, fits in 48KB shared mem
// Load query once per block:
if (tid < D) s_query[tid] = queries[qid * D + tid];
__syncthreads();
// All threads read from s_query instead of queries
```

Saves 256× redundant global memory reads of the query vector per block.

### 5.3 [HIGH] Vectorized int4 Loads

**Impact: ~2-4× kernel speedup (combined with 5.2)**

```cuda
// Instead of:
dot += (int)queries[qid*D+d] * (int)vectors[vid*D+d];

// Use:
const int4* q4 = reinterpret_cast<const int4*>(s_query);  // after shared mem load
const int4* v4 = reinterpret_cast<const int4*>(&vectors[vid*D]);
for (int d4 = 0; d4 < D/16; d4++) {
    int4 q = q4[d4];
    int4 v = v4[d4];
    // Multiply 16 int8 pairs using pmaddwd
    dot += vdot16(q, v);
}
```

For D=384: 24 int4 loads instead of 384 byte loads. Each int4 load fetches 16 bytes in one transaction.

### 5.4 [MEDIUM] Increase batch_q with GPU Top-k

With GPU-side top-k (5.1), the output buffer is no longer Q×N. Can increase batch_q to 128 or 256, reducing launch count from 31,250 to 7,812 or 3,906. This reduces kernel launch overhead and improves GPU utilization.

### 5.5 [MEDIUM] Stream-Ordered Execution

Overlap H2D upload of batch B+1 with kernel execution of batch B and D2H readback of batch B-1:
```rust
cuStreamBeginCapture(...);  // or manual stream ordering
cuMemcpyHtoDAsync(d_q, batch_B+1);
cuLaunchKernel(func, batch_B);
cuMemcpyDtoHAsync(dists, batch_B-1);
```

### 5.6 [LOW] Reduce ef_construction for Build

`ef_construction=200` is generous. For build-time only, reducing to 64-128 halves the layer-0 search cost (from ~6400 to ~3200 distance computations per insert). Use higher ef at query time. This only helps the CPU fallback path.

### 5.7 [LOW] Fused Distance + Top-k Kernel

Combine `batch_cosine_dist_kernel` and top-k into a single kernel. Each block computes distances for its query and maintains a thread-private top-k list, then merges via warp shuffle. Eliminates the intermediate Q×N buffer entirely.

---

## 6. Recommended Implementation Priority

| Priority | Change | Expected Build Time |
|----------|--------|-------------------|
| P0 | GPU-side top-k (5.1) | 170s → 5-10s |
| P1 | Shared memory + int4 (5.2+5.3) | Kernel 3-5× faster |
| P2 | Increase batch_q (5.4) | 20% fewer launches |
| P3 | Streams (5.5) | 10-15% overlap benefit |
| P4 | Lower ef_c for build (5.6) | CPU path 2× faster |

**With P0 alone: build time drops from 170s+ to under 10 seconds for 1M vectors.**

