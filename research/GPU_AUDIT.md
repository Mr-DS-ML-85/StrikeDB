# GPU Audit Report — crates/gpu

**Files audited:**
- `crates/gpu/src/lib.rs` (926 lines) — Rust CUDA driver FFI + public API
- `crates/gpu/kernels/all_kernels.cu` (164 lines) — CUDA kernels

---

## 1. Memory Bugs (Double-Free, Leaks, Buffer Overflows)

### CRITICAL: GPU memory leak in `gpu_cosine_dist` on kernel failure (lib.rs:447)

When `cuLaunchKernel` fails, the function returns `None` without freeing `d_q`, `d_v`, or `d_d`. All three device buffers leak.

```rust
// lib.rs:427-453
let mut d_q = 0u64; let mut d_v = 0u64; let mut d_d = 0u64;
let r = cuMemAlloc_v2(&mut d_q, q_bytes); if r != 0 { return None; }  // d_q allocated
let r = cuMemAlloc_v2(&mut d_v, v_bytes); if r != 0 { return None; }  // d_q leaked if fails
let r = cuMemAlloc_v2(&mut d_d, d_bytes); if r != 0 { return None; }  // d_q, d_v leaked if fails
// ... kernel launch ...
if r != 0 {
    // BUG: d_q, d_v, d_d all leaked here
    return None;
}
```

Compare with `gpu_batch_cosine_dist` (line 496-504) which correctly frees all three on error. Inconsistency suggests this is a bug, not intentional.

**Fix:** Add `cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);` before the `return None` at line 453.

### CRITICAL: GPU memory leak in `gpu_build_knn_graph` batch loop on kernel failure (lib.rs:574)

```rust
// lib.rs:574
if r != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_d); return None; }
// BUG: d_v is NOT freed — the vectors buffer leaked
```

The vectors device buffer `d_v` (allocated at line 529) is never freed on this error path.

### CRITICAL: GPU memory leak in `gpu_cosine_dist` partial alloc failure (lib.rs:428-430)

```rust
let r = cuMemAlloc_v2(&mut d_q, q_bytes); if r != 0 { return None; }
let r = cuMemAlloc_v2(&mut d_v, v_bytes); if r != 0 { return None; }  // d_q leaks
let r = cuMemAlloc_v2(&mut d_d, d_bytes); if r != 0 { return None; }  // d_q, d_v leak
```

If `d_v` allocation fails, `d_q` leaks. If `d_d` allocation fails, both `d_q` and `d_v` leak.

### MEDIUM: GPU memory leak in `gpu_batch_cosine_dist` partial alloc failure (lib.rs:476-478)

Same pattern as above — sequential allocs without cleanup on intermediate failure.

### CRITICAL: `gpu_matmul` has no error checking on allocations (lib.rs:613)

```rust
cuMemAlloc_v2(&mut d_a, a_bytes);  // return value IGNORED
cuMemAlloc_v2(&mut d_b, b_bytes);  // return value IGNORED
cuMemAlloc_v2(&mut d_c, c_bytes);  // return value IGNORED
```

If any allocation fails, the function continues with invalid device pointers, leading to undefined behavior (kernel reads garbage, potential GPU crash). If `d_b` or `d_c` fail, the previously allocated buffers also leak.

### CRITICAL: Use-after-free via `gpu_unload` + `gpu_init` (lib.rs:641-650, 202-220)

`gpu_unload` destroys the CUDA context via `cuCtxDestroy_v2`, then sets `GPU_ENABLED = false`. However, `OnceLock` cannot be re-initialized — `GPU_STATE` retains the old `GpuState` with the destroyed context. On a subsequent `gpu_init`:

1. `GPU_ENABLED` is `false` → enters the init path
2. `GPU_STATE.get_or_init(...)` returns the OLD state (OnceLock already set)
3. `state.available` is still `true` → sets `GPU_ENABLED = true`
4. All subsequent GPU calls use the destroyed CUDA context → **use-after-free / GPU crash**

**Fix:** Either don't destroy the context in `gpu_unload` (just disable), or replace `OnceLock` with a `Mutex<Option<GpuState>>` that can be reset.

### MEDIUM: `gpu_unload` doesn't unload CUDA module (lib.rs:641-650)

`cuModuleUnload` is never called before `cuCtxDestroy_v2`. The module handle leaks.

---

## 2. CUDA API Misuse

### CRITICAL: `gpu_matmul` doesn't hold GPU_ACCESS lock (lib.rs:606-638)

`gpu_matmul` calls `gpu_ensure_kernel("matmul")` then `ensure_ctx()`, but never acquires `GPU_ACCESS`. All CUDA driver calls (cuMemAlloc, cuMemcpy, cuLaunchKernel, cuCtxSynchronize) proceed without the lock. The codebase comments explicitly state: "CUDA driver API is NOT thread-safe. All CUDA calls MUST go through this lock." This violates that invariant.

Other public functions (`gpu_cosine_dist`, `gpu_batch_cosine_dist`, `gpu_build_knn_graph`, `gpu_build_index`, `gpu_search`) correctly use `gpu_lock_and_ensure()`.

### MEDIUM: Hardcoded `compute_86` architecture (lib.rs:144)

```rust
let arch = CString::new("--gpu-architecture=compute_86").unwrap();
```

This fails on GPUs with different compute capabilities:
- Volta/Turing: compute_70/75
- Ampere A100: compute_80
- Hopper H100: compute_90
- Ada Lovelace: compute_89

**Fix:** Detect the GPU's compute capability via `cuDeviceGetAttribute` and pass the correct architecture string, or use `--gpu-architecture=compute_XX` with the detected value.

### LOW: Unchecked CUDA API return values throughout

- `cuMemcpyHtoD_v2` returns not checked in `gpu_cosine_dist` (lines 431-432), `gpu_batch_cosine_dist` (lines 479-480), `gpu_matmul` (lines 614-615), `gpu_build_knn_graph` (lines 530, 559)
- `cuMemcpyDtoH_v2` returns not checked (lines 459, 507, 580, 634, 807-808)
- `cuCtxSynchronize` return not checked in `gpu_cosine_dist` (line 455) — actually checked at 456. But not checked in `gpu_batch_cosine_dist` (line 505), `gpu_build_knn_graph` (line 575), `gpu_matmul` (line 632)

---

## 3. Performance Issues

### MEDIUM: `gpu_matmul` uses naive kernel with no shared memory tiling (all_kernels.cu:156-163)

```c
for (int k = 0; k < K; k++) s += (int)A[r * K + k] * (int)B[k * N + c];
```

Each thread reads entire rows of A and columns of B from global memory. For large K, this causes massive redundant global memory traffic. A tiled approach using shared memory (e.g., 16×16 tiles) would be 5-10x faster by exploiting data reuse.

### LOW: `dists` buffer reallocated every batch iteration in kNN build (lib.rs:579)

```rust
for batch_start in (0..n).step_by(batch_q) {
    let mut dists = vec![0.0f32; q_count * n];  // Allocated every iteration
```

For n=1M, batch_q=32, this allocates/frees 128MB per iteration. Should be pre-allocated once before the loop.

### LOW: Block size 256 may underutilize for large N in `batch_cosine_dist` (lib.rs:481)

With `threads = 256`, each block processes all N vectors for one query. For N > 256, threads iterate over vectors in a strided loop (line 32 of kernel). This is fine but could benefit from a 2D grid (queries × vector-tiles) for better occupancy.

### INFO: No kernel fusion or vectorized loads

The cosine distance kernels load individual `char` values. Using `int4` (128-bit) loads when D is a multiple of 16 would reduce memory transactions by 16x.

---

## 4. Kernel Correctness

### CRITICAL: Race condition in `cagra_search_kernel` — `nc` is thread-local (all_kernels.cu:89-103)

```c
int nc = 0;                    // ← thread-local register variable
if (tid == 0) {
    // ... only thread 0 writes nc ...
    cand_idx[nc++] = nbr;
}
__syncthreads();                // ← synchronizes shared memory, NOT registers
if (nc == 0) break;            // ← all threads read their OWN nc (still 0)
```

`nc` is a thread-local variable, not in shared memory. `__syncthreads()` only guarantees shared memory consistency. After the barrier:
- Thread 0 reads its own `nc` (correct value, could be > 0) → continues loop
- Threads 1-255 read their own `nc` (still 0) → break out of loop

On the next iteration's `__syncthreads()` (line 102 or 115), only thread 0 participates — the other 255 threads have already exited the loop. This causes **undefined behavior** (some threads in a block reaching `__syncthreads()` while others don't).

**Fix:** Declare `nc` in shared memory:
```c
__shared__ int smem_nc;
if (tid == 0) smem_nc = 0;
__syncthreads();
// ... thread 0 writes to smem_nc ...
__syncthreads();
if (smem_nc == 0) break;
```

### LOW: Dot product overflow for large dimensions (all_kernels.cu:13-18, 33-37, 109-113)

```c
int dot = 0;
for (int d = 0; d < D; d++) {
    dot += (int)query[d] * (int)vectors[vid * D + d];
}
```

INT8 range is [-128, 127]. Max single product: 127 × 127 = 16,129. Max accumulated dot for D dimensions: 16,129 × D. For int32 (max ~2.1B), this overflows at D > 131,072. Current dimensions are likely safe (128-4096), but there's no guard for pathological inputs.

### LOW: `cagra_search_kernel` doesn't handle k > itopk (all_kernels.cu:138-146)

The output loop writes `k` results, but the topk buffer only holds `itopk` entries. If `k > itopk`, the extra entries get `-1` / `2.0f` sentinel values. This is intentional (the Rust caller should set `k <= itopk`), but there's no assertion or documentation enforcing this constraint.

---

## 5. `gpu_build_knn_graph` Batch Loop Leak Analysis

**Pre-loop allocations:**
| Buffer | Allocation (line) | Size |
|--------|-------------------|------|
| `d_v` (vectors) | line 529 | `n * dim` bytes |
| `d_d` (distances) | line 539 | `batch_q * n * 4` bytes |
| `d_q` (query batch) | line 543 | `batch_q * dim` bytes |

**Loop body (line 553-594):** Reuses persistent buffers `d_q` and `d_d` — no alloc/free per iteration. This is correct and efficient.

**Error path in loop (line 574):**
```rust
if r != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_d); return None; }
// BUG: d_v leaked — should also cuMemFree_v2(d_v)
```

**Post-loop cleanup (line 599-601):**
```rust
cuMemFree_v2(d_v);
cuMemFree_v2(d_q);
cuMemFree_v2(d_d);
```
All three freed on success. Correct.

**Normal-path summary:** The batch loop is clean — no leaks in the happy path. The only leak is on kernel launch failure inside the loop.

---

## Summary Table

| Severity | Category | Location | Issue |
|----------|----------|----------|-------|
| CRITICAL | Leak | lib.rs:447-453 | `gpu_cosine_dist` leaks 3 GPU buffers on kernel failure |
| CRITICAL | Leak | lib.rs:574 | `gpu_build_knn_graph` leaks `d_v` on kernel failure |
| CRITICAL | Leak | lib.rs:428-430 | `gpu_cosine_dist` partial alloc leak |
| CRITICAL | Leak | lib.rs:613 | `gpu_matmul` no error checking, potential leak + UB |
| CRITICAL | UAF | lib.rs:641-650 | `gpu_unload` + re-`gpu_init` uses destroyed context |
| CRITICAL | Race | all_kernels.cu:89-103 | `nc` thread-local → divergence past `__syncthreads()` |
| CRITICAL | Thread safety | lib.rs:606-638 | `gpu_matmul` skips `GPU_ACCESS` lock |
| MEDIUM | Correctness | lib.rs:144 | Hardcoded `compute_86` fails on non-Ampere GPUs |
| MEDIUM | Leak | lib.rs:641-650 | `gpu_unload` doesn't call `cuModuleUnload` |
| MEDIUM | Performance | all_kernels.cu:156-163 | Naive matmul, no shared memory tiling |
| LOW | Leak | lib.rs:579 | `dists` buffer reallocated every iteration |
| LOW | Overflow | all_kernels.cu:13 | Dot product overflow for D > 131K |
| LOW | Robustness | all_kernels.cu:138 | No guard for k > itopk |
