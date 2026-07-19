// DB-Strike GPU kernels — CAGRA-style GPU-native search.
// Flat CSR graph + INT8 vectors on GPU. One CUDA block per query.
// Thread 0 orchestrates, all threads compute distances.

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

// ── CAGRA GPU Search Kernel ──────────────────────────────────────────────
// One block per query. Iterative beam search on GPU.
//
// Shared memory layout (all ints):
//   topk[2*itopk]  — alternating (dist, idx) pairs, sorted by dist desc
//   scratch[2*max_cand] — candidate (dist, idx) pairs
//
// Args: vectors[N*D i8], graph[N*degree i32], queries[Q*D i8],
//        out_idx[Q*k], out_dist[Q*k float],
//        N, D, degree, k, itopk, max_iters, Q, entry_node
extern "C" __global__
void cagra_search_kernel(
    const char* __restrict__ vectors,
    const int* __restrict__ graph,
    const char* __restrict__ queries,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist,
    int N, int D, int degree,
    int k, int itopk, int max_iters,
    int num_queries, int entry_node
) {
    int qid = blockIdx.x;
    if (qid >= num_queries) return;
    int tid = threadIdx.x;

    extern __shared__ int smem[];
    // topk: pairs of (dist, idx), sorted by dist descending
    // We store as two separate arrays for simplicity
    int* topk_dist = smem;                     // [itopk]
    int* topk_idx  = smem + itopk;             // [itopk]
    // scratch for merge: candidates from neighbor expansion
    // max candidates = degree (search_width=1, so 1 node × degree neighbors)
    int* tmp_dist  = smem + 2 * itopk;         // [degree]
    int* tmp_idx   = smem + 2 * itopk + degree; // [degree]

    // ── Initialize topk with entry node ──
    if (tid == 0) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)queries[qid * D + d] * (int)vectors[entry_node * D + d];
        }
        topk_dist[0] = dot;
        topk_idx[0] = entry_node;
        for (int i = 1; i < itopk; i++) {
            topk_dist[i] = -2147483647;
            topk_idx[i] = -1;
        }
    }
    __syncthreads();

    // ── Iterative beam search ──
    for (int iter = 0; iter < max_iters; iter++) {
        // Phase A: Thread 0 collects neighbors of top-1 node.
        int nc = 0;
        if (tid == 0) {
            int src = topk_idx[0];
            if (src >= 0 && src < N) {
                for (int nb = 0; nb < degree && nc < degree; nb++) {
                    int neighbor = graph[src * degree + nb];
                    if (neighbor < 0 || neighbor >= N) continue;
                    // Skip if already in topk
                    int dup = 0;
                    for (int t = 0; t < itopk; t++) {
                        if (topk_idx[t] == neighbor) { dup = 1; break; }
                    }
                    if (!dup) {
                        tmp_idx[nc] = neighbor;
                        nc++;
                    }
                }
            }
        }
        __syncthreads();
        if (nc == 0) break;

        // Phase B: All threads compute distances to candidates.
        for (int c = tid; c < nc; c += blockDim.x) {
            int node = tmp_idx[c];
            int dot = 0;
            for (int d = 0; d < D; d++) {
                dot += (int)queries[qid * D + d] * (int)vectors[node * D + d];
            }
            tmp_dist[c] = dot;
        }
        __syncthreads();

        // Phase C: Thread 0 merges topk with candidates, keep top-itopk.
        if (tid == 0) {
            // Simple approach: try inserting each candidate into topk
            for (int c = 0; c < nc; c++) {
                int c_dot = tmp_dist[c];
                int c_node = tmp_idx[c];
                // Find position in topk (sorted descending by dot)
                // If c_dot > worst in topk, insert and evict worst
                if (c_dot > topk_dist[itopk - 1]) {
                    // Shift everything after insertion point
                    int pos = itopk - 1;
                    for (int t = itopk - 2; t >= 0; t--) {
                        if (topk_dist[t] >= c_dot) break;
                        topk_dist[t + 1] = topk_dist[t];
                        topk_idx[t + 1] = topk_idx[t];
                        pos = t;
                    }
                    topk_dist[pos] = c_dot;
                    topk_idx[pos] = c_node;
                }
            }
        }
        __syncthreads();
    }

    // ── Write output top-k ──
    for (int i = tid; i < k; i += blockDim.x) {
        if (i < itopk && topk_idx[i] >= 0) {
            out_idx[qid * k + i] = topk_idx[i];
            out_dist[qid * k + i] = 1.0f - (float)topk_dist[i] / 16129.0f;
        } else {
            out_idx[qid * k + i] = -1;
            out_dist[qid * k + i] = 2.0f;
        }
    }
}

extern "C" __global__
void fill_one_kernel(float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) out[tid] = 1.0f;
}

extern "C" __global__
void matmul_kernel(
    const char* A, const char* B, int* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; k++)
            sum += (int)A[row * K + k] * (int)B[k * N + col];
        C[row * N + col] = sum;
    }
}
