// DB-Strike GPU kernels — CAGRA-style parallel graph search.
// Key: ALL threads compute distances in parallel (team-cooperative).

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

// ── CAGRA GPU Search: Team-Cooperative Parallel Graph Traversal ──────────
//
// 256 threads per query. Distance computation PARALLELIZED across threads.
// This is what makes GPU faster than CPU: 256 threads compute 256
// different distances simultaneously, while CPU computes one at a time.
//
// Algorithm per iteration:
//   1. Thread 0 reads neighbors from graph → shared memory
//   2. ALL threads compute distances (one distance per thread, parallel)
//   3. Thread 0 merges into sorted topk
//
// Shared memory: topk[2*itopk] + cand_idx[8*degree] + cand_dot[8*degree]
extern "C" __global__
void cagra_search_kernel(
    const char* __restrict__ vectors,
    const int* __restrict__ graph,
    const char* __restrict__ query,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist,
    int N, int D, int degree,
    int k, int itopk, int max_iters,
    int entry_node, int num_queries)
{
    int qid = blockIdx.x;
    if (qid >= num_queries) return;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ int smem[];
    int* topk_dot = smem;
    int* topk_idx = smem + itopk;
    int* cand_idx = smem + 2 * itopk;
    int* cand_dot = smem + 2 * itopk + 8 * degree;
    int* smem_nc = smem + 2 * itopk + 16 * degree; // dedicated slot for nc count

    // Init topk with entry node
    if (tid == 0) {
        int dot = 0;
        for (int d = 0; d < D; d++) dot += (int)query[qid * D + d] * (int)vectors[entry_node * D + d];
        topk_dot[0] = dot;
        topk_idx[0] = entry_node;
        for (int i = 1; i < itopk; i++) { topk_dot[i] = -2147483647; topk_idx[i] = -1; }
    }
    __syncthreads();

    int BEAM = 4;

    for (int iter = 0; iter < max_iters; iter++) {
        // Phase 1: Thread 0 reads neighbors, stores count at shared smem[0]
        if (tid == 0) {
            int nc = 0;
            for (int p = 0; p < BEAM && p < itopk; p++) {
                int src = topk_idx[p];
                if (src < 0 || src >= N) continue;
                for (int nb = 0; nb < degree && nc < 8 * degree; nb++) {
                    int nbr = graph[src * degree + nb];
                    if (nbr >= 0 && nbr < N) {
                        cand_idx[nc++] = nbr;
                    }
                }
            }
            smem_nc[0] = nc;
        }
        __syncthreads();
        int nc = smem_nc[0];
        if (nc == 0) break;

        // Phase 2: ALL threads compute distances in PARALLEL
        // Each thread computes dot product for one candidate vector
        for (int c = tid; c < nc; c += threads) {
            int node = cand_idx[c];
            int dot = 0;
            for (int d = 0; d < D; d++) {
                dot += (int)query[qid * D + d] * (int)vectors[node * D + d];
            }
            cand_dot[c] = dot;
        }
        __syncthreads();

        // Phase 3: Thread 0 merges into sorted topk
        if (tid == 0) {
            for (int c = 0; c < nc; c++) {
                int dot = cand_dot[c];
                if (dot > topk_dot[itopk - 1]) {
                    int pos = itopk - 1;
                    for (int s = itopk - 2; s >= 0; s--) {
                        if (topk_dot[s] >= dot) break;
                        topk_dot[s + 1] = topk_dot[s];
                        topk_idx[s + 1] = topk_idx[s];
                        pos = s;
                    }
                    topk_dot[pos] = dot;
                    topk_idx[pos] = cand_idx[c];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < k; i += threads) {
        if (i < itopk && topk_idx[i] >= 0) {
            out_idx[qid * k + i] = topk_idx[i];
            out_dist[qid * k + i] = 1.0f - (float)topk_dot[i] / 16129.0f;
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
