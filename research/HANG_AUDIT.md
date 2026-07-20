# GPU Turbo Benchmark Hang Audit — 1M×384d

## Executive Summary

The benchmark does not deadlock or infinite-loop. It hangs because `topk_select_kernel`
is an **O(k²·N) single-thread algorithm** running on 1 GPU thread for N=1,000,000.
Estimated time to complete one batch: **~4.5 hours**. Total for 1M vectors: **effectively infinite**.

---

## Bottleneck #1 (CRITICAL): `topk_select_kernel` — single-thread O(k²·N)

**File**: `crates/gpu/kernels/all_kernels.cu:162-189`

```cuda
// Line 170: ONLY thread 0 runs — block launched with 1 thread
if (threadIdx.x != 0) return;

// O(k × N × k) = O(k²·N) per query
for (int i = 0; i < k; i++) {          // k=32 iterations
    for (int j = 0; j < N; j++) {      // N=1,000,000 per iteration
        // ... check duplicate against all i previously found
        for (int t = 0; t < i; t++) {  // up to k=32 checks
```

**Launch config** (`lib.rs:590`):
```rust
cuLaunchKernel(func_topk, q_count as u32, 1, 1,  // grid: ≤32 blocks
               1, 1, 1,                            // block: 1 thread(!)
```

### Why it hangs — arithmetic

| Parameter | Value |
|-----------|-------|
| N (vectors) | 1,000,000 |
| k (top-k) | 32 |
| batch_q | 32 |
| Batches | ⌈1M / 32⌉ = 31,250 |

Per query:
- Outer loop: k = 32 iterations
- Inner scan: N = 1,000,000 candidates
- Duplicate check: avg k/2 = 16 comparisons
- Total: **32 × 1,000,000 × 16 = 512 billion comparisons**

On a single GPU thread at effective ~1 GHz: **~512 seconds per query**.

Per batch (32 queries): **~16,384 seconds ≈ 4.5 hours**.

Total (31,250 batches): **~142 million hours** — the benchmark will never finish.

### Root cause

The kernel was designed for small N (CAGRA builds with N=1K-10K). At N=1M, the O(k²·N)
single-thread approach becomes catastrophic. The 256-thread block is wasted — only thread 0 runs.

---

## Bottleneck #2 (MODERATE): `batch_cosine_dist_kernel` — 384-dim sequential dot product

**File**: `crates/gpu/kernels/all_kernels.cu:22-39`

```cuda
// Each thread processes N/256 ≈ 3906 vectors
// Each vector: 384 multiply-adds in a scalar loop
for (int d = 0; d < D; d++) {   // D=384, no unrolling
    dot += (int)queries[qid * D + d] * (int)vectors[vid * D + d];
}
```

With Q=32 blocks × 256 threads = 8,192 active threads processing 1M vectors.
Each thread handles ~3,906 vectors × 384 dims = ~1.5M scalar multiply-adds.

This takes ~100-500ms per batch — significant over 31,250 batches (~50 minutes total),
but it's the **secondary** bottleneck, not the hang.

---

## Bottleneck #3 (MINOR): No duplicate check needed in topk

**File**: `crates/gpu/kernels/all_kernels.cu:179-184`

The inner `found` check (`for t in 0..i`) is unnecessary and doubles the constant factor.
For k=32 and N=1M, distances are almost certainly unique (floating-point noise), so the
check is wasted work. Even if fixed, the O(k·N) single-thread approach is still too slow.

---

## Deadlock / Infinite Loop Check

**File**: `crates/views/src/vector.rs:3395-3529` (`build_parallel_tiered`)

- **No deadlock**: GPU path is single-threaded, no lock contention.
- **No infinite loop**: the `for batch_start in (0..n).step_by(batch_q)` loop
  in `lib.rs:565` terminates correctly (step_by(32) over 0..1M).
- **CPU fallback** (vector.rs:3473-3488) is O(N²·dim) — also very slow for 1M
  but won't infinite-loop.
- The `unwrap_or_else` fallback (line 3471) will fire if `gpu_build_knn_graph`
  returns `None` (e.g., VRAM OOM), but when it returns `Some`, the topk bottleneck
  kicks in and the process appears hung.

---

## Fix Recommendations

### Fix 1: Parallel top-k with shared memory (replaces current kernel)

Replace the single-thread selection sort with a proper parallel algorithm:

```cuda
// APPROACH: Partial sort via bitonic network in shared memory
// 1. Each thread scans N/256 elements, keeps local top-k (registers)
// 2. Merge all 256 local top-k lists via bitonic sort in shared memory
// 3. Write final top-k to global memory

extern "C" __global__
void topk_select_kernel(
    const float* __restrict__ dists,
    int* __restrict__ out_idx,
    int Q, int N, int k)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;  // 256

    // Phase 1: each thread finds local top-k from its strip
    float local_best[32];     // register file
    int local_idx[32];
    for (int i = 0; i < k; i++) { local_best[i] = 1e30f; local_idx[i] = -1; }

    for (int j = tid; j < N; j += threads) {
        float d = dists[qid * N + j];
        if (d < local_best[k-1]) {
            local_best[k-1] = d;
            local_idx[k-1] = j;
            // insertion sort (k=32, fast)
            for (int s = k-2; s >= 0; s--) {
                if (local_best[s] <= local_best[s+1]) break;
                float td = local_best[s]; local_best[s] = local_best[s+1]; local_best[s+1] = td;
                int ti = local_idx[s]; local_idx[s] = local_idx[s+1]; local_idx[s+1] = ti;
            }
        }
    }

    // Phase 2: merge 256 local top-k via shared memory
    extern __shared__ char smem[];
    float* s_dists = (float*)smem;         // 256 * 32 * 4 = 32KB
    int* s_indices = (int*)(smem + 256*32*sizeof(float));

    for (int i = 0; i < k; i++) {
        s_dists[tid * k + i] = local_best[i];
        s_indices[tid * k + i] = local_idx[i];
    }
    __syncthreads();

    // Thread 0 does final selection from 256*k candidates
    if (tid == 0) {
        // Find k smallest from 256*32=8192 candidates
        for (int i = 0; i < k; i++) {
            float best = 1e30f;
            int best_pos = -1;
            for (int j = 0; j < threads * k; j++) {
                if (s_dists[j] < best) {
                    best = s_dists[j];
                    best_pos = j;
                }
            }
            out_idx[qid * k + i] = s_indices[best_pos];
            s_dists[best_pos] = 1e30f;  // mark as used
        }
    }
}
```

**Launch config**: `cuLaunchKernel(func_topk, q_count, 1, 1, 256, 1, 1, smem, ...)`

This reduces per-query work from O(k²·N) on 1 thread to O(N/256 × k + k²×256) on 256 threads.
For N=1M, k=32: from 512B comparisons (1 thread) to ~4M comparisons (256 threads) = **~128,000x speedup**.

### Fix 2: Increase batch size (easy, no kernel change)

Change `lib.rs:544`:
```rust
// FROM:
let batch_q = 32.min(n);
// TO:
let batch_q = 128.min(n);  // reduces iterations 4x, VRAM cost: 128*1M*4 = 512MB
```

This alone reduces total iterations from 31,250 to 7,812 — still too slow, but helps.

### Fix 3: Vectorized dot product in batch_cosine_dist

Use `int4` (128-bit) loads for the 384-dim dot product:
```cuda
for (int d = 0; d < D; d += 4) {
    int4 q4 = *reinterpret_cast<const int4*>(&queries[qid * D + d]);
    int4 v4 = *reinterpret_cast<const int4*>(&vectors[vid * D + d]);
    dot += q4.x*v4.x + q4.y*v4.y + q4.z*v4.z + q4.w*v4.w;
}
```

4x throughput improvement on the distance kernel.

---

## Priority Order

1. **Fix topk_select_kernel** — this is the hang. Without this, nothing else matters.
2. **Increase batch_q** — quick win while working on fix 1.
3. **Vectorize dot product** — optimization after the hang is fixed.

---

## Files Involved

| File | Lines | Issue |
|------|-------|-------|
| `crates/gpu/kernels/all_kernels.cu` | 162-189 | O(k²·N) single-thread topk (THE hang) |
| `crates/gpu/src/lib.rs` | 544 | batch_q=32 too small |
| `crates/gpu/src/lib.rs` | 590 | topk launched with 1 thread/block |
| `crates/gpu/src/lib.rs` | 565-605 | Main loop — runs 31,250 batches |
| `crates/gpu/kernels/all_kernels.cu` | 22-39 | Scalar dot product (secondary) |
| `crates/views/src/vector.rs` | 3395-3529 | No deadlock, but calls the broken GPU path |
