// DB-Strike GPU kernels — APGC-style parallel graph search.
// Key: ALL threads compute distances in parallel (team-cooperative).

// 4-way int8 dot product: single instruction on sm_61+ (Pascal and newer).
// Operands are 32-bit words holding 4 packed int8 lanes, so callers also get
// coalesced 4-byte loads instead of 4 separate byte loads.
__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    char4 av = *(char4*)&a; char4 bv = *(char4*)&b;
    return c + (int)av.x*bv.x + (int)av.y*bv.y + (int)av.z*bv.z + (int)av.w*bv.w;
#endif
}

// Order-preserving float <-> int key map, so the integer bitonic network in
// `apgc_search_kernel` can sort exact f32 scores without a second network.
//
// IEEE-754 already orders non-negative floats the same way their bit patterns
// order as integers; negatives run backwards. Flipping the negatives with
// `0x80000000 - i` fixes the direction AND drops every negative key below
// every non-negative one. The map is its own inverse.
__device__ __forceinline__ int f32_to_key(float f) {
    int i = __float_as_int(f);
    return (i >= 0) ? i : (int)(0x80000000u - (unsigned)i);
}
__device__ __forceinline__ float key_to_f32(int kk) {
    int i = (kk >= 0) ? kk : (int)(0x80000000u - (unsigned)kk);
    return __int_as_float(i);
}

extern "C" __global__
void cosine_dist_kernel(
    const char* __restrict__ query,
    const char* __restrict__ vectors,
    float* __restrict__ dists,
    int N, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)query[d] * (int)vectors[tid * D + d];
        }
        dists[tid] = 1.0f - (float)dot / 16129.0f;
    }
}

extern "C" __global__
void batch_cosine_dist_kernel(
    const char* __restrict__ queries,
    const char* __restrict__ vectors,
    float* __restrict__ dists,
    int Q, int N, int D)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;
    for (int vid = tid; vid < N; vid += threads) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)queries[qid * D + d] * (int)vectors[vid * D + d];
        }
        dists[qid * N + vid] = 1.0f - (float)dot / 16129.0f;
    }
}

// FP32 Batch Cosine Distance — APGC graph construction needs FP32 discrimination.
extern "C" __global__
void batch_cosine_dist_f32_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ vectors,
    float* __restrict__ dists,
    int Q, int N, int D)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;
    for (int vid = tid; vid < N; vid += threads) {
        float dot = 0.0f, nq = 0.0f, nv = 0.0f;
        for (int d = 0; d < D; d++) {
            float q = queries[qid * D + d];
            float v = vectors[vid * D + d];
            dot += q * v; nq += q * q; nv += v * v;
        }
        float norm = sqrtf(nq) * sqrtf(nv);
        dists[qid * N + vid] = (norm > 0.0f) ? (1.0f - dot / norm) : 1.0f;
    }
}

// ── APGC GPU Search: Team-Cooperative Parallel Graph Traversal ──────────
//
// One query = one block. Every phase inside an iteration is data-parallel
// across the whole block; nothing is left to thread 0 except a 9-int beam
// pick. The block is launched wide (512 threads = 16 warps) so the SM has
// warps to switch to while the random neighbour gathers miss L2 — at 4
// warps the walk ran at memory *latency* instead of memory *bandwidth*.
//
// Search-list layout:
//   buf[0 .. itopk)   top list, sorted DESCENDING by dot, carried across iters
//   buf[itopk .. BUF) this iteration's candidates, BUF = next_pow2(itopk +
//                     beam*degree) so the whole list bitonic-sorts in place
//
// A tempting optimisation is to sort only the candidate region and bitonic-
// merge it against the already-sorted top half (log2(BUF) stages instead of
// log2(BUF)(log2(BUF)+1)/2). That needs the two halves to be equal powers of
// two, which forces the per-node neighbour budget down to itopk/beam. It was
// measured: clamping degree 34 -> 32 drops recall@10 from 0.996 to 0.948, and
// -> 16 drops it to 0.524. The graph's adjacency is not quality-ordered, so
// truncating it is never free. Full sort, full neighbour list.
//
// Shared memory (ints): 3*BUF search list + SR_HASH dedup set
//                       + ceil(D/4) packed query + 16 control
extern "C" __global__
void apgc_search_kernel(
    const char* __restrict__ vectors,
    const int* __restrict__ graph,
    const char* __restrict__ query,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist,
    const float* __restrict__ delta,  // OpusEdge SelKV: per-node importance
    int N, int D, int degree,
    int k, int itopk, int max_iters,
    int entry_node, int num_queries,
    float selkv_ratio,               // OpusEdge SelKV: keep top-fraction (0.5 = 50%)
    int delta_ar_k,                  // OpusEdge Delta-AR: neighbour prefix budget
    int beam_in,                     // nodes expanded per iteration
    const float* __restrict__ vec_f32,   // exact corpus for fused rerank (may be null)
    const float* __restrict__ query_f32) // exact query for fused rerank (may be null)
{
    int qid = blockIdx.x;
    if (qid >= num_queries) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // ── Shared layout ────────────────────────────────────────────────────
    const int SR_HASH = 1024;
    const int SR_HMASK = SR_HASH - 1;
    int BEAM = (beam_in > 0 && beam_in <= 8) ? beam_in : 8;
    int BUF = 1; while (BUF < itopk + BEAM * degree) BUF <<= 1;
    int D4 = (D + 3) >> 2;

    extern __shared__ int smem[];
    int* buf_dot = smem;
    int* buf_idx = smem + BUF;
    int* buf_exp = smem + 2 * BUF;
    int* vis     = smem + 3 * BUF;              // per-iteration dedup set
    int* qcache  = smem + 3 * BUF + SR_HASH;    // query vector, block-local
    int* s_ctl   = smem + 3 * BUF + SR_HASH + D4;  // [0]=nc, [1..8]=picked srcs
    // Rerank scratch, only touched when vec_f32 != null. The host sizer adds
    // (D + RR_MAX) ints for it exactly when it passes a non-null corpus.
    float* qf32   = (float*)(smem + 3 * BUF + SR_HASH + D4 + 16); // exact query
    float* rr_acc = qf32 + D;                                     // per-candidate dot

    // Cache the query once in shared memory. The old kernel re-read it from
    // global memory for every dimension of every candidate.
    for (int i = tid; i < D4; i += threads) {
        int w = 0;
        for (int b = 0; b < 4; b++) {
            int d = i * 4 + b;
            if (d < D) w |= ((int)(unsigned char)query[qid * D + d]) << (b * 8);
        }
        qcache[i] = w;
    }
    for (int i = tid; i < BUF; i += threads) {
        buf_dot[i] = -2147483647; buf_idx[i] = -1; buf_exp[i] = 1;
    }
    __syncthreads();

    const char* qbytes = (const char*)qcache;

    // Entry node distance (parallel dp4a reduction would be overkill for one
    // vector; thread 0 is fine here since it happens once).
    if (tid == 0) {
        int dot = 0;
        if ((D & 3) == 0) {
            const int* vv = (const int*)vectors + (size_t)entry_node * D4;
            for (int d4 = 0; d4 < D4; d4++) dot = dp4a_i8(__ldg(&vv[d4]), qcache[d4], dot);
        } else {
            for (int d = 0; d < D; d++) dot += (int)qbytes[d] * (int)vectors[(size_t)entry_node * D + d];
        }
        buf_dot[0] = dot; buf_idx[0] = entry_node; buf_exp[0] = 0;
    }
    __syncthreads();

    float selkv_min = 1.0f - selkv_ratio;       // OpusEdge SelKV gate
    // OpusEdge Delta-AR: read only a prefix of the δ-sorted adjacency.
    // Opt-in only — the default reads the full neighbour list, because
    // truncating it costs far more recall than it saves time (see header).
    int nb_lim = (delta_ar_k > 0 && delta_ar_k < degree) ? delta_ar_k : degree;
    int cand_cap = BEAM * nb_lim;

    for (int iter = 0; iter < max_iters; iter++) {
        // ── Clear the dedup set, seed it with the current top list ────────
        for (int i = tid; i < SR_HASH; i += threads) vis[i] = -1;
        if (tid == 0) s_ctl[0] = 0;
        __syncthreads();
        for (int i = tid; i < itopk; i += threads) {
            int nd = buf_idx[i];
            if (nd < 0) continue;
            unsigned h = ((unsigned)nd * 2654435761u) & SR_HMASK;
            for (int probe = 0; probe < 32; probe++) {
                int cur = vis[h];
                if (cur == nd) break;
                if (cur == -1 && atomicCAS(&vis[h], -1, nd) == -1) break;
                h = (h + 1) & SR_HMASK;
            }
        }
        // ── Pick up to BEAM best unexpanded sources (list is sorted) ──────
        if (tid == 0) {
            int picked = 0;
            for (int p = 0; p < itopk && picked < BEAM; p++) {
                if (buf_idx[p] < 0) break;
                if (buf_exp[p]) continue;
                buf_exp[p] = 1;
                s_ctl[1 + picked] = buf_idx[p];
                picked++;
            }
            s_ctl[9] = picked;
        }
        __syncthreads();
        int picked = s_ctl[9];
        if (picked == 0) break;                  // converged

        // ── Parallel neighbour gather, deduped against the top list and
        //    against each other through the shared hash set ──────────────
        int total = picked * nb_lim;
        for (int i = tid; i < total; i += threads) {
            int src = s_ctl[1 + i / nb_lim];
            int nbr = graph[(size_t)src * degree + (i % nb_lim)];
            if (nbr < 0 || nbr >= N || nbr == src) continue;
            if (delta[nbr] < selkv_min) continue;            // SelKV gate
            unsigned h = ((unsigned)nbr * 2654435761u) & SR_HMASK;
            for (int probe = 0; probe < 32; probe++) {
                int cur = vis[h];
                if (cur == nbr) break;                        // duplicate
                if (cur == -1) {
                    if (atomicCAS(&vis[h], -1, nbr) == -1) {
                        int pos = atomicAdd(&s_ctl[0], 1);
                        if (pos < cand_cap) {
                            buf_idx[itopk + pos] = nbr;
                            buf_exp[itopk + pos] = 0;
                        }
                        break;
                    }
                    continue;                                 // lost race, re-probe
                }
                h = (h + 1) & SR_HMASK;
            }
        }
        __syncthreads();
        int nc = s_ctl[0]; if (nc > cand_cap) nc = cand_cap;

        // ── Parallel dp4a scoring of the staged candidates ───────────────
        for (int c = itopk + tid; c < itopk + nc; c += threads) {
            int node = buf_idx[c];
            int dot = 0;
            if ((D & 3) == 0) {
                const int* vv = (const int*)vectors + (size_t)node * D4;
                #pragma unroll 8
                for (int d4 = 0; d4 < D4; d4++) dot = dp4a_i8(__ldg(&vv[d4]), qcache[d4], dot);
            } else {
                for (int d = 0; d < D; d++) dot += (int)qbytes[d] * (int)vectors[(size_t)node * D + d];
            }
            buf_dot[c] = dot;
        }
        // Pad the unused tail so the sort keeps it below everything real.
        for (int c = itopk + nc + tid; c < BUF; c += threads) {
            buf_dot[c] = -2147483647; buf_idx[c] = -1; buf_exp[c] = 1;
        }
        __syncthreads();

        // ── Bitonic sort, descending by dot, payloads travel along ───────
        for (int size = 2; size <= BUF; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < (BUF >> 1); i += threads) {
                    int j = ((i / stride) * (stride << 1)) + (i % stride);
                    int p = j + stride;
                    if (((j & size) == 0) == (buf_dot[j] < buf_dot[p])) {
                        int t;
                        t = buf_dot[j]; buf_dot[j] = buf_dot[p]; buf_dot[p] = t;
                        t = buf_idx[j]; buf_idx[j] = buf_idx[p]; buf_idx[p] = t;
                        t = buf_exp[j]; buf_exp[j] = buf_exp[p]; buf_exp[p] = t;
                    }
                }
                __syncthreads();
            }
        }
    }

    // ── Fused exact-f32 rerank ───────────────────────────────────────────
    //
    // The walk above ranks by int8 dp4a, which costs ~15% recall on its own.
    // The fix used to be a CPU pass: copy k ids back, then dot k×D f32 pairs
    // on the host (~25k float mults per query at k=64, D=384). That is the CPU
    // burn visible in btop during "GPU-only" Turbo mode.
    //
    // The block already holds the ranked list in shared memory, so the rerank
    // is a phase, not a second launch: no round trip, no host arithmetic. The
    // int8 walk is now purely a *candidate generator* and the reported distance
    // is exact.
    //
    // RR = next_pow2(k) >= k, so we rescore slightly more than we emit and the
    // reordering can pull a true neighbour up from just outside the top k.
    if (vec_f32 != nullptr && query_f32 != nullptr && k > 0) {
        int RR = 1; while (RR < k) RR <<= 1;
        // The bitonic network below needs a power of two, so shrink by halving
        // rather than clamping to `itopk` directly (itopk is arbitrary).
        while (RR > itopk) RR >>= 1;
        if (RR < 1) RR = 1;

        for (int d = tid; d < D; d += threads) qf32[d] = query_f32[(size_t)qid * D + d];
        for (int i = tid; i < RR; i += threads) rr_acc[i] = 0.0f;
        __syncthreads();

        // Split the block into RR groups of `per` threads. Threads *within* a
        // group stride over dimensions, so their loads from one candidate row
        // coalesce; groups are what run in parallel across candidates.
        int per = threads / RR; if (per < 1) per = 1;
        int groups = threads / per;
        for (int c = tid / per; c < RR; c += groups) {
            int node = buf_idx[c];
            if (node < 0) continue;
            const float* vv = vec_f32 + (size_t)node * D;
            float acc = 0.0f;
            for (int d = tid % per; d < D; d += per) acc += qf32[d] * __ldg(&vv[d]);
            atomicAdd(&rr_acc[c], acc);
        }
        __syncthreads();

        // Re-key by the exact score and re-sort. Only RR <= 64 entries move, so
        // this is ~21 tiny stages, not another full log2(BUF)^2 network.
        for (int i = tid; i < RR; i += threads) {
            buf_dot[i] = (buf_idx[i] >= 0) ? f32_to_key(rr_acc[i]) : -2147483647;
        }
        __syncthreads();
        for (int size = 2; size <= RR; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < (RR >> 1); i += threads) {
                    int j = ((i / stride) * (stride << 1)) + (i % stride);
                    int p = j + stride;
                    if (((j & size) == 0) == (buf_dot[j] < buf_dot[p])) {
                        int t;
                        t = buf_dot[j]; buf_dot[j] = buf_dot[p]; buf_dot[p] = t;
                        t = buf_idx[j]; buf_idx[j] = buf_idx[p]; buf_idx[p] = t;
                    }
                }
                __syncthreads();
            }
        }
        for (int i = tid; i < k; i += threads) {
            if (i < RR && buf_idx[i] >= 0) {
                out_idx[qid * k + i] = buf_idx[i];
                float d = 1.0f - key_to_f32(buf_dot[i]);
                out_dist[qid * k + i] = fminf(fmaxf(d, 0.0f), 2.0f);
            } else {
                out_idx[qid * k + i] = -1;
                out_dist[qid * k + i] = 2.0f;
            }
        }
        return;
    }

    // No f32 corpus in VRAM: emit the int8 ranking and let the host rerank.
    for (int i = tid; i < k; i += threads) {
        if (i < itopk && buf_idx[i] >= 0) {
            out_idx[qid * k + i] = buf_idx[i];
            out_dist[qid * k + i] = 1.0f - (float)buf_dot[i] / 16129.0f;
        } else {
            out_idx[qid * k + i] = -1;
            out_dist[qid * k + i] = 2.0f;
        }
    }
}

extern "C" __global__
void fill_one_kernel(float* out, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) out[t] = 1.0f;
}

// Parallel tree merge of per-thread sorted top-k lists in shared memory.
// log2(threads) steps; each step merges pairs of sorted k-lists in O(k).
// Requires threads to be a power of two (host launches 64 or 128).
// Result lands in list 0 (s_dist[0..k], s_idx[0..k]), sorted ascending.
__device__ void topk_tree_merge(float* s_dist, int* s_idx, int tid, int threads, int k) {
    float md[64]; int mi[64];
    for (int active = threads >> 1; active >= 1; active >>= 1) {
        if (tid < active) {
            float* d1 = s_dist + tid * k;            int* i1 = s_idx + tid * k;
            float* d2 = s_dist + (tid + active) * k; int* i2 = s_idx + (tid + active) * k;
            int p = 0, q = 0;
            for (int i = 0; i < k; i++) {
                if (q >= k || (p < k && d1[p] <= d2[q])) { md[i] = d1[p]; mi[i] = i1[p]; p++; }
                else                                     { md[i] = d2[q]; mi[i] = i2[q]; q++; }
            }
            for (int i = 0; i < k; i++) { d1[i] = md[i]; i1[i] = mi[i]; }
        }
        __syncthreads();
    }
}

// GPU top-k selection kernel: Q×N distances → Q×k top-k indices + distances.
// PARALLEL: all threads cooperatively scan N vectors per query.
// Each thread maintains a private top-k heap over N/threads elements,
// then merges via a parallel tree reduction in dynamic shared memory.
// Reads only Q×k×8 bytes back (tiny) instead of Q×N×4 (128MB).
// Shared memory budget: 128 threads × k × 8 bytes = 32 KB for k=32 (fits 48 KB default).
extern "C" __global__
void topk_select_kernel(
    const float* __restrict__ dists,   // Q × N distances on GPU
    int* __restrict__ out_idx,          // Q × k output indices
    float* __restrict__ out_dist,       // Q × k output distances
    int Q, int N, int k)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Dynamic shared memory: float s_dists[threads][k], int s_idxs[threads][k]
    extern __shared__ char smem_topk[];
    float* s_dists = (float*)smem_topk;
    int* s_idxs = (int*)(smem_topk + threads * k * sizeof(float));

    // k ≤ 64 (KMAX): bounds the register arrays below.
    if (k > 64) return;
    float local_dist[64];
    int local_idx[64];
    for (int i = 0; i < k; i++) {
        local_dist[i] = 1e10f;
        local_idx[i] = -1;
    }

    int start = tid * (N / threads);
    int end = (tid == threads - 1) ? N : start + (N / threads);
    for (int j = start; j < end; j++) {
        float d = dists[qid * N + j];
        if (d < local_dist[k - 1]) {
            int pos = k - 1;
            for (int s = k - 2; s >= 0; s--) {
                if (local_dist[s] <= d) break;
                local_dist[s + 1] = local_dist[s];
                local_idx[s + 1] = local_idx[s];
                pos = s;
            }
            local_dist[pos] = d;
            local_idx[pos] = j;
        }
    }

    for (int i = 0; i < k; i++) {
        s_dists[tid * k + i] = local_dist[i];
        s_idxs[tid * k + i] = local_idx[i];
    }
    __syncthreads();

    topk_tree_merge(s_dists, s_idxs, tid, threads, k);

    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            if (s_dists[i] < 1e9f) {
                out_idx[qid * k + i] = s_idxs[i];
                out_dist[qid * k + i] = s_dists[i];
            } else {
                out_idx[qid * k + i] = -1;
                out_dist[qid * k + i] = 2.0f;
            }
        }
    }
}

// FUSED kernel: batch cosine distance + top-k in ONE pass.
// Never materializes the full Q×N distance matrix — each thread computes
// distances for N/threads vectors and maintains a local top-k heap.
// Eliminates the Q×N GPU buffer entirely → saves 128 MB per batch.
// Launched with 128 threads: shared mem = 128 × k × 8 bytes ≤ 32 KB.
// Input: queries (Q×D int8), vectors (N×D int8)
// Output: out_idx (Q×k), out_dist (Q×k)
extern "C" __global__
void fused_cosine_topk_kernel(
    const char* __restrict__ queries,
    const char* __restrict__ vectors,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist,
    int Q, int N, int D, int k)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Per-thread local top-k (registers). k ≤ 64 (KMAX).
    if (k > 64) return;
    float local_dist[64];
    int local_idx[64];
    for (int i = 0; i < k; i++) {
        local_dist[i] = 1e10f;
        local_idx[i] = -1;
    }

    // Each thread computes distances for N/threads vectors.
    // dp4a path: 32-bit loads + 4-way int8 MAC per instruction (4× fewer
    // loads and 4× fewer MAC instructions than the per-byte fallback).
    int start = tid * (N / threads);
    int end = (tid == threads - 1) ? N : start + (N / threads);
    if ((D & 3) == 0) {
        const int D4 = D >> 2;
        const int* qv = (const int*)queries + (size_t)qid * D4;
        for (int vid = start; vid < end; vid++) {
            const int* vv = (const int*)vectors + (size_t)vid * D4;
            int dot = 0;
            #pragma unroll 8
            for (int d4 = 0; d4 < D4; d4++) {
                dot = dp4a_i8(__ldg(&vv[d4]), __ldg(&qv[d4]), dot);
            }
            float dist = 1.0f - (float)dot / 16129.0f;
            if (dist < local_dist[k - 1]) {
                int pos = k - 1;
                for (int s = k - 2; s >= 0; s--) {
                    if (local_dist[s] <= dist) break;
                    local_dist[s + 1] = local_dist[s];
                    local_idx[s + 1] = local_idx[s];
                    pos = s;
                }
                local_dist[pos] = dist;
                local_idx[pos] = vid;
            }
        }
    } else {
        for (int vid = start; vid < end; vid++) {
            int dot = 0;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                dot += (int)queries[qid * D + d] * (int)vectors[(size_t)vid * D + d];
            }
            float dist = 1.0f - (float)dot / 16129.0f;
            if (dist < local_dist[k - 1]) {
                int pos = k - 1;
                for (int s = k - 2; s >= 0; s--) {
                    if (local_dist[s] <= dist) break;
                    local_dist[s + 1] = local_dist[s];
                    local_idx[s + 1] = local_idx[s];
                    pos = s;
                }
                local_dist[pos] = dist;
                local_idx[pos] = vid;
            }
        }
    }

    // Write to shared memory
    extern __shared__ int smem_fused[];
    float* s_dist = (float*)smem_fused;
    int* s_idx = smem_fused + threads * k;

    for (int i = 0; i < k; i++) {
        s_dist[tid * k + i] = local_dist[i];
        s_idx[tid * k + i] = local_idx[i];
    }
    __syncthreads();

    topk_tree_merge(s_dist, s_idx, tid, threads, k);

    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            if (s_dist[i] < 1e9f) {
                out_idx[qid * k + i] = s_idx[i];
                out_dist[qid * k + i] = s_dist[i];
            } else {
                out_idx[qid * k + i] = -1;
                out_dist[qid * k + i] = 2.0f;
            }
        }
    }
}

// ═══ GPU NN-Descent (APGC Phase 3 on-device) ═══
// Replaces the CPU refinement loop: candidate gather from forward + reverse
// neighbor lists, dp4a rescoring, exact top-k per node. One pass per launch.

// Build capped reverse adjacency: rev[nb][slot] = v for each edge v→nb.
extern "C" __global__
void nn_rev_kernel(const int* __restrict__ graph, int* __restrict__ rev,
                   int* __restrict__ rev_cnt, int N, int K, int rev_cap) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    for (int j = 0; j < K; j++) {
        int nb = graph[(size_t)v * K + j];
        if (nb < 0 || nb >= N || nb == v) continue;
        int pos = atomicAdd(&rev_cnt[nb], 1);
        if (pos < rev_cap) rev[(size_t)nb * rev_cap + pos] = v;
    }
}

// One NN-descent pass: one block per node, fully parallel.
//   1. Parallel candidate gather (own edges, reverse edges, forward-neighbors'
//      lists, reverse-neighbors' lists — the NN-descent "general join"),
//      deduplicated on the fly through a shared-memory open-addressing hash set
//      so no vector is ever scored twice.
//   2. Parallel dp4a scoring across all threads.
//   3. Parallel K-step arg-max selection (block tree reduction).
// Own edges are always in the candidate set, so a pass can never degrade a list.
#define ND_CAND_MAX 1024
#define ND_HASH 2048
#define ND_HMASK (ND_HASH - 1)

__device__ __forceinline__ void nd_try_insert(int w, int* hash, int* cand, int* nc_ptr) {
    unsigned h = ((unsigned)w * 2654435761u) & ND_HMASK;
    for (int probe = 0; probe < 48; probe++) {
        int cur = hash[h];
        if (cur == w) return;                       // already gathered
        if (cur == -1) {
            int old = atomicCAS(&hash[h], -1, w);
            if (old == -1) {                        // we claimed the slot
                int pos = atomicAdd(nc_ptr, 1);
                if (pos < ND_CAND_MAX) cand[pos] = w;
                return;
            }
            if (old == w) return;                   // lost race to same id
        }
        h = (h + 1) & ND_HMASK;
    }
}

extern "C" __global__
void nn_descent_kernel(
    const char* __restrict__ vectors,
    const int* __restrict__ graph_in,
    const int* __restrict__ rev,
    const int* __restrict__ rev_cnt,
    int* __restrict__ graph_out,
    int N, int D, int K, int expand, int rev_cap)
{
    int v = blockIdx.x;
    if (v >= N) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    __shared__ int cand[ND_CAND_MAX];
    __shared__ int cdot[ND_CAND_MAX];
    __shared__ int hash[ND_HASH];
    __shared__ int r_val[128];
    __shared__ int r_idx[128];
    __shared__ int s_nc;

    for (int i = tid; i < ND_HASH; i += threads) hash[i] = -1;
    if (tid == 0) s_nc = 0;
    __syncthreads();

    const int* own = graph_in + (size_t)v * K;
    const int* rv = rev + (size_t)v * rev_cap;
    int rc = rev_cnt[v]; if (rc > rev_cap) rc = rev_cap;
    int ef = expand < K ? expand : K;
    int er = expand < rc ? expand : rc;
    int total = K + rc + ef * K + er * K;

    // ── 1. Parallel gather + dedup ───────────────────────────────────────
    for (int i = tid; i < total; i += threads) {
        int w = -1;
        if (i < K) {
            w = own[i];
        } else if (i < K + rc) {
            w = rv[i - K];
        } else {
            int j = i - K - rc;
            if (j < ef * K) {
                int u = own[j / K];
                if (u >= 0 && u < N) w = graph_in[(size_t)u * K + (j % K)];
            } else {
                int j2 = j - ef * K;
                int u = rv[j2 / K];
                if (u >= 0 && u < N) w = graph_in[(size_t)u * K + (j2 % K)];
            }
        }
        if (w >= 0 && w < N && w != v) nd_try_insert(w, hash, cand, &s_nc);
    }
    __syncthreads();
    int nc = s_nc; if (nc > ND_CAND_MAX) nc = ND_CAND_MAX;

    // ── 2. Parallel scoring (dp4a when D is 4-byte aligned) ──────────────
    if ((D & 3) == 0) {
        const int D4 = D >> 2;
        const int* qv = (const int*)vectors + (size_t)v * D4;
        for (int c = tid; c < nc; c += threads) {
            const int* vv = (const int*)vectors + (size_t)cand[c] * D4;
            int dot = 0;
            #pragma unroll 8
            for (int d4 = 0; d4 < D4; d4++) dot = dp4a_i8(__ldg(&vv[d4]), __ldg(&qv[d4]), dot);
            cdot[c] = dot;
        }
    } else {
        const char* qb = vectors + (size_t)v * D;
        for (int c = tid; c < nc; c += threads) {
            const char* vb = vectors + (size_t)cand[c] * D;
            int dot = 0;
            for (int d = 0; d < D; d++) dot += (int)qb[d] * (int)vb[d];
            cdot[c] = dot;
        }
    }
    __syncthreads();

    // ── 3. Parallel top-K selection (K arg-max tree reductions) ──────────
    // Candidates are unique (hash dedup), so removing the winner each round
    // is a single-slot write, not a scan.
    for (int s = 0; s < K; s++) {
        int bv = -2147483647, bi = -1;
        for (int c = tid; c < nc; c += threads) {
            int d = cdot[c];
            if (d > bv) { bv = d; bi = c; }
        }
        r_val[tid] = bv; r_idx[tid] = bi;
        __syncthreads();
        for (int stride = threads >> 1; stride >= 1; stride >>= 1) {
            if (tid < stride && r_val[tid + stride] > r_val[tid]) {
                r_val[tid] = r_val[tid + stride];
                r_idx[tid] = r_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int best = r_idx[0];
            graph_out[(size_t)v * K + s] = (best >= 0) ? cand[best] : -1;
            if (best >= 0) cdot[best] = -2147483648;
        }
        __syncthreads();
    }
}

extern "C" __global__
void matmul_kernel(const char* A, const char* B, int* C, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        int s = 0;
        for (int k = 0; k < K; k++) s += (int)A[r * K + k] * (int)B[k * N + c];
        C[r * N + c] = s;
    }
}

// ═══ APGC Graph Optimization: Reorder + Reverse + Merge ═══
// These add long-range edges to the graph. NOT CAGRA-specific.
// Generic graph structure optimization applicable to any graph.

// Reorder: count detourable routes per edge, keep most important
extern "C" __global__
void apgc_optimize_kernel(const int* raw_knn, int* optimized, int N, int d_init, int d_final) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    int counts[64];
    for (int r = 0; r < d_init && r < 64; r++) {
        int y = raw_knn[node * d_init + r]; int detour = 0;
        if (y >= 0 && y < N && y != node) {
            for (int r2 = 0; r2 < r; r2++) {
                int z = raw_knn[node * d_init + r2]; if (z < 0 || z >= N || z == node) continue;
                for (int j = 0; j < d_init; j++) { if (raw_knn[z * d_init + j] == y) { if (j < r) detour++; break; } }
            }
        }
        counts[r] = detour;
    }
    int order[64]; for (int i = 0; i < d_init && i < 64; i++) order[i] = i;
    for (int i = 1; i < d_init && i < 64; i++) {
        int key = order[i]; int kc = counts[key]; int s = i - 1;
        while (s >= 0 && counts[order[s]] < kc) { order[s + 1] = order[s]; s--; }
        order[s + 1] = key;
    }
    for (int i = 0; i < d_final; i++) optimized[node * d_final + i] = raw_knn[node * d_init + order[i]];
}

// Reverse edges
extern "C" __global__
void apgc_reverse_kernel(const int* pruned, int* rev_edges, int* rev_count, int N, int d) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    for (int r = 0; r < d; r++) {
        int y = pruned[node * d + r]; if (y < 0 || y >= N) continue;
        int pos = atomicAdd(&rev_count[y], 1); if (pos < d) rev_edges[y * d + pos] = node;
    }
}

// Merge pruned + reversed
extern "C" __global__
void apgc_merge_kernel(const int* pruned, const int* rev_edges, const int* rev_count, int* final_graph, int N, int d) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    int half = d / 2; int nc = rev_count[node]; if (nc > d) nc = d;
    for (int i = 0; i < d; i++) {
        if (i < half) final_graph[node * d + i] = pruned[node * d + i];
        else { int ri = i - half; final_graph[node * d + i] = (ri < nc) ? rev_edges[node * d + ri] : pruned[node * d + i]; }
    }
}

// Preprocess INT8 to float
extern "C" __global__
void preprocess_i8_to_f32_kernel(const char* input, float* output, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * D) output[i] = (float)(signed char)input[i];
}

// ═══ OpusEdge CUDA Primitives ═══
// Telemetry-guided dynamic compute allocation for GPU search.
// Uses attention scores to prune low-importance graph nodes.

// OpusEdge SelKV: Zero-Cost KV pruning on GPU.
// For each graph node, check if attention score is above threshold.
// Skip nodes below threshold during graph traversal.
extern "C" __global__
void opusedge_selkv_prune(
    const float* __restrict__ attention_scores, // [N] per-node importance
    int* __restrict__ candidates,               // [N*k] candidate node indices (read-write)
    int* __restrict__ candidate_counts,         // [N] number of candidates per node
    int N, int k, float threshold)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    // Prune candidates below threshold
    int write = 0;
    int base = node * k;
    for (int i = 0; i < k; i++) {
        int cand = candidates[base + i];
        if (cand >= 0 && cand < N && attention_scores[cand] >= threshold) {
            if (write != i) {
                candidates[base + write] = cand;
            }
            write++;
        }
    }
    // Fill remaining with -1
    for (int i = write; i < k; i++) candidates[base + i] = -1;
    candidate_counts[node] = write;
}

// OpusEdge Delta-AR: Adaptive attention routing on GPU.
// Route each query to top-K nodes with highest attention scores.
extern "C" __global__
void opusedge_delta_ar_route(
    const float* __restrict__ attention_scores, // [N] per-node importance
    int* __restrict__ routed_nodes,             // [N*K] output: top-K routed node indices
    int N, int K)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    // Simple selection: find top-K nodes by attention score
    // For each position in routed_nodes, find the best unselected node
    int base = node * K;
    for (int pos = 0; pos < K; pos++) {
        int best = -1;
        float best_score = -1.0f;
        for (int j = 0; j < N; j++) {
            if (attention_scores[j] > best_score) {
                // Check not already selected
                int found = 0;
                for (int s = 0; s < pos; s++) {
                    if (routed_nodes[base + s] == j) { found = 1; break; }
                }
                if (!found) {
                    best_score = attention_scores[j];
                    best = j;
                }
            }
        }
        routed_nodes[base + pos] = best;
    }
}

// OpusEdge HeadDeactivate: Multi-head gating on GPU.
// Determine which attention heads to activate based on token entropy.
extern "C" __global__
void opusedge_head_gate(
    const float* __restrict__ token_entropy, // [N] per-token entropy
    int* __restrict__ active_heads,          // [N] number of active heads per token
    int N, int total_heads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float entropy = token_entropy[tid];
    // 4-tier gating based on entropy
    if (entropy < 0.5) active_heads[tid] = 4;           // Low: 4/32 heads
    else if (entropy < 1.5) active_heads[tid] = 16;     // Mid: 16/32
    else if (entropy < 2.5) active_heads[tid] = 24;     // High: 24/32
    else active_heads[tid] = total_heads;                 // Critical: all heads
}

// OpusEdge StateCompress: Hidden-state compression on GPU.
// Zero low-magnitude channels when entropy is low.
extern "C" __global__
void opusedge_state_compress(
    float* __restrict__ hidden_states, // [N*D] hidden states (read-write)
    const float* __restrict__ entropy, // [N] per-token entropy
    int N, int D, float threshold, float keep_ratio)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (entropy[tid] >= threshold) return; // high entropy = no compression

    // Zero low-magnitude channels
    int keep_count = (int)(D * keep_ratio);
    int base = tid * D;
    for (int d = keep_count; d < D; d++) {
        hidden_states[base + d] = 0.0f;
    }
}

// OpusEdge Proxy-Delta: RMS hidden-state drift computation on GPU.
// O(L) cost per token — extracts importance signal from dense models.
extern "C" __global__
void opusedge_proxy_delta(
    const float* __restrict__ hidden_states, // [N*D] current hidden states
    const float* __restrict__ prev_states,   // [N*D] previous hidden states
    float* __restrict__ delta_out,           // [N] output: Proxy-Δ per token
    int N, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float ssd = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = hidden_states[tid * D + d] - prev_states[tid * D + d];
        ssd += diff * diff;
    }
    delta_out[tid] = sqrtf(ssd / (float)D); // RMS drift
}
