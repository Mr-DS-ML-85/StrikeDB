// INT8 matrix multiply kernel for bridge distance computation.
// Computes C[M][N] = A[M][K] × B[K][N] where A is int8, B is int8.
// Each thread computes one element of C using int32 accumulation.
// No cuBLAS — pure CUDA kernel, compiled via NVRTC at runtime.

extern "C" __global__
void int8_matmul_kernel(
    const int8_t* __restrict__ A,  // [M x K] row-major
    const int8_t* __restrict__ B,  // [K x N] row-major
    int32_t* __restrict__ C,       // [M x N] row-major
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Batch INT8 matmul: computes distance from query to all vectors.
// A = query [1 x D], B = vectors [N x D]^T, C = distances [1 x N].
// For cosine distance on unit vectors: dist = 1 - dot_i8 / (127*127).
extern "C" __global__
void int8_cosine_dist_kernel(
    const int8_t* __restrict__ query,   // [D]
    const int8_t* __restrict__ vectors, // [N x D] row-major
    float* __restrict__ dists,          // [N]
    int N, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int32_t dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int32_t)query[d] * (int32_t)vectors[tid * D + d];
        }
        dists[tid] = 1.0f - (float)dot / 16129.0f; // 127*127
    }
}

// Batch cosine distance: compute distances from multiple queries to all vectors.
// queries [Q x D], vectors [N x D], dists [Q x N].
extern "C" __global__
void batch_cosine_dist_kernel(
    const int8_t* __restrict__ queries,  // [Q x D]
    const int8_t* __restrict__ vectors,  // [N x D]
    float* __restrict__ dists,           // [Q x N]
    int Q, int N, int D)
{
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (q < Q && n < N) {
        int32_t dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int32_t)queries[q * D + d] * (int32_t)vectors[n * D + d];
        }
        dists[q * N + n] = 1.0f - (float)dot / 16129.0f;
    }
}

// Parallel HNSW bridge: for each node, find nearest entry in other segments.
// nodes_i8 [total x D], entry_i8 [K x D], entry_idx [K], offsets [K], n_per [K].
// Output: edges [total x 2] — (ga, gb) pairs for bridge connections.
extern "C" __global__
void hnsw_bridge_kernel(
    const int8_t* __restrict__ nodes_i8,   // [total x D]
    const int8_t* __restrict__ entry_i8,    // [K x D]
    const int* __restrict__ entry_idx,      // [K]
    const int* __restrict__ offsets,        // [K]
    const int* __restrict__ n_per,          // [K]
    int* __restrict__ edge_from,            // [total]
    int* __restrict__ edge_to,              // [total]
    int total, int K, int D)
{
    int ga = blockIdx.x * blockDim.x + threadIdx.x;
    if (ga >= total) return;

    // Find own segment.
    int own = 0;
    for (int s = 0; s < K; s++) {
        if (ga >= offsets[s]) own = s;
    }

    // Find nearest OTHER segment entry.
    float best_d = 1e30f;
    int best_b = 0;
    const int8_t* q = nodes_i8 + ga * D;
    for (int b = 0; b < K; b++) {
        if (b == own) continue;
        int32_t dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int32_t)q[d] * (int32_t)entry_i8[b * D + d];
        }
        float d = 1.0f - (float)dot / 16129.0f;
        if (d < best_d) { best_d = d; best_b = b; }
    }

    edge_from[ga] = ga;
    edge_to[ga] = entry_idx[best_b];
}
