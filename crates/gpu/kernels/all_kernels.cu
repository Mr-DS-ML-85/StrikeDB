// DB-Strike GPU kernels — real HNSW graph search, not brute-force.
// One CUDA block per query. Thread 0 does graph traversal.

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

// ── CAGRA GPU Search: Real Graph Traversal with Beam Search ──────────────
// One block per query. Thread 0 does the search.
// Expands from TOP-P nodes per iteration (beam width), not just 1.
// This is the real CAGRA algorithm — same as CPU HNSW but on GPU.
//
// Shared memory: topk[2*itopk] = (dot, idx) pairs sorted descending.
//
// Args: vectors, graph, query, out_idx, out_dist,
//        N, D, degree, k, itopk, max_iters, entry_node, num_queries
extern "C" __global__
void cagra_search_kernel(
    const char* __restrict__ vectors,
    const int* __restrict__ graph,
    const char* __restrict__ query,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist,
    int N, int D, int degree,
    int k, int itopk, int max_iters,
    int entry_node, int num_queries
) {
    int qid = blockIdx.x;
    if (qid >= num_queries) return;
    int tid = threadIdx.x;

    extern __shared__ int smem[];
    int* topk_dot = smem;
    int* topk_idx = smem + itopk;

    // Init topk with entry node
    if (tid == 0) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)query[qid * D + d] * (int)vectors[entry_node * D + d];
        }
        topk_dot[0] = dot;
        topk_idx[0] = entry_node;
        for (int i = 1; i < itopk; i++) {
            topk_dot[i] = -2147483647;
            topk_idx[i] = -1;
        }
    }
    __syncthreads();

    // Iterative beam search — expand from TOP-P candidates per iteration
    int P = 4; // beam width: expand from top-4 nodes each iteration
    for (int iter = 0; iter < max_iters; iter++) {
        if (tid == 0) {
            // For each of the top-P nodes, visit ALL their neighbors
            for (int p = 0; p < P && p < itopk; p++) {
                int src = topk_idx[p];
                if (src < 0 || src >= N) continue;

                for (int nb = 0; nb < degree; nb++) {
                    int neighbor = graph[src * degree + nb];
                    if (neighbor < 0 || neighbor >= N) continue;

                    // Skip if already in topk
                    int dup = 0;
                    for (int t = 0; t < itopk; t++) {
                        if (topk_idx[t] == neighbor) { dup = 1; break; }
                    }
                    if (dup) continue;

                    // Compute dot product
                    int dot = 0;
                    for (int d = 0; d < D; d++) {
                        dot += (int)query[qid * D + d] * (int)vectors[neighbor * D + d];
                    }

                    // Insert into topk if better than worst
                    if (dot > topk_dot[itopk - 1]) {
                        int pos = itopk - 1;
                        for (int s = itopk - 2; s >= 0; s--) {
                            if (topk_dot[s] >= dot) break;
                            topk_dot[s + 1] = topk_dot[s];
                            topk_idx[s + 1] = topk_idx[s];
                            pos = s;
                        }
                        topk_dot[pos] = dot;
                        topk_idx[pos] = neighbor;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output top-k
    for (int i = tid; i < k; i += blockDim.x) {
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
