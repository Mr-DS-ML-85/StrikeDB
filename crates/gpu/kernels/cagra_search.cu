// CAGRA-style GPU search kernel for DB-Strike.
// Batch wavefront graph traversal with team-cooperative INT8 distance.
//
// Data layout (flat CSR):
//   graph:  N × degree × u32  (adjacency lists, row-major)
//   vectors: N × dim × i8    (INT8 vectors, row-major)
//   query:  dim × i8          (single query, quantized)
//
// One CUDA block = one query. Threads within a block cooperate
// on distance computation (team-cooperative reduction).

extern "C" __global__
void cagra_search_kernel(
    const int* __restrict__ query,      // [dim] query vector (int8)
    const char* __restrict__ vectors,   // [N * dim] database vectors
    const int* __restrict__ graph,      // [N * degree] flat adjacency
    int* __restrict__ out_topk_idx,     // [num_queries * k] output indices
    int* __restrict__ out_topk_dist,    // [num_queries * k] output distances
    int N,                               // number of vectors
    int dim,                             // vector dimension
    int degree,                          // graph out-degree
    int k,                               // top-k to return
    int itopk,                           // intermediate top-k (>= k)
    int max_iters,                       // max search iterations
    int num_queries,                     // batch size
    int block_id                         // which query this block handles
) {
    if (block_id >= num_queries) return;

    int tid = threadIdx.x;
    int team_size = blockDim.x;

    // Per-query state in shared memory (declared via extern for flexibility).
    // itopk entries: (dist, idx) pairs for the internal top-M list.
    extern __shared__ int smem[];
    // Layout: [0..itopk-1] = topk_dist, [itopk..2*itopk-1] = topk_idx,
    // [2*itopk..3*itopk-1] = candidates
    int* topk_dist = smem;
    int* topk_idx = smem + itopk;
    int* candidates = smem + 2 * itopk;

    // Step 0: Initialize top-M with sentinel (dist=INT_MAX, idx=-1).
    for (int i = tid; i < itopk; i += team_size) {
        topk_dist[i] = 2147483647; // INT_MAX
        topk_idx[i] = -1;
    }
    __syncthreads();

    // Step 1: Seed with random starting nodes (use block_id as seed offset).
    // Each thread picks one random seed.
    int num_seeds = (degree < itopk) ? degree : itopk;
    if (tid < num_seeds) {
        // Simple hash-based pseudo-random based on query + iteration
        int seed = ((block_id * 997 + tid * 131) % N);
        candidates[tid] = seed;
    }
    __syncthreads();

    // Main search loop
    for (int iter = 0; iter < max_iters; iter++) {
        // Step 2: Compute distances for all candidates (team-cooperative).
        int num_cands = (iter == 0) ? num_seeds : num_seeds;

        for (int c = tid; c < num_cands; c += team_size) {
            int node = candidates[c];
            if (node < 0 || node >= N) continue;

            // INT8 cosine distance: dot = sum(query[d] * vector[d])
            // dist = 1 - dot / (127*127)
            int dot = 0;
            for (int d = tid; d < dim; d += team_size) {
                dot += (int)query[d] * (int)vectors[node * dim + d];
            }

            // Warp-level reduction
            for (int offset = team_size / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }

            if (tid == 0) {
                topk_dist[c] = -dot; // negative for max-heap (closest = largest negative)
                topk_idx[c] = node;
            }
        }
        __syncthreads();

        // Step 3: Parallel merge-sort to find top-itopk from combined candidates.
        // Simple insertion sort for small arrays (itopk <= 128).
        if (tid == 0) {
            // Count valid entries
            int total = num_cands;
            // Sort combined topk + candidates by distance (descending = closest first)
            // Using simple selection sort — itopk is small (64-128)
            for (int i = 0; i < itopk - 1; i++) {
                int max_idx = i;
                for (int j = i + 1; j < itopk; j++) {
                    if (topk_dist[j] > topk_dist[max_idx]) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    int td = topk_dist[i]; topk_dist[i] = topk_dist[max_idx]; topk_dist[max_idx] = td;
                    int ti = topk_idx[i]; topk_idx[i] = topk_idx[max_idx]; topk_idx[max_idx] = ti;
                }
            }
        }
        __syncthreads();

        // Step 4: Expand frontier — read neighbors of top-p nodes.
        int p = 1; // search_width = 1
        int new_cands = 0;
        for (int src = 0; src < p && src < itopk; src++) {
            int src_idx = topk_idx[src];
            if (src_idx < 0 || src_idx >= N) continue;
            for (int nb = tid; nb < degree; nb += team_size) {
                int neighbor = graph[src_idx * degree + nb];
                if (neighbor >= 0 && neighbor < N) {
                    int slot = (src * degree + nb) % itopk;
                    candidates[slot] = neighbor;
                }
            }
        }
        num_seeds = (p * degree < itopk) ? p * degree : itopk;
        __syncthreads();
    }

    // Final: extract top-k from top-itopk.
    for (int i = tid; i < k; i += team_size) {
        out_topk_idx[block_id * k + i] = topk_idx[i];
        out_topk_dist[block_id * k + i] = -topk_dist[i]; // convert back to positive
    }
}
