// DB-Strike GPU kernels — batch operations for CAGRA-style search.
//
// Kernel 1: batch_cosine_dist — Q queries × N vectors, outputs Q×N distances.
//   One block per query, threads cooperate on dot product reduction.
//   This is the CAGRA distance_compute kernel (Section 6.2, CAGRA_RESEARCH.md).
//
// Kernel 2: cosine_dist — single query × N vectors (existing).
// Kernel 3: fill_one — test kernel (existing).
// Kernel 4: matmul — INT8 matrix multiply (existing).

// ── INT8 cosine distance (single query) ──────────────────────────────────
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

// ── Batch INT8 cosine distance (Q queries × N vectors) ───────────────────
// CAGRA-style team-cooperative distance kernel.
// Grid: Q blocks. Block: 256 threads.
// Each block handles one query; threads cooperatively compute the dot
// product with ALL N vectors. Each thread computes distances for a
// stripe of vectors (N/threads stride), using warp shuffle reduction.
//
// Output: dists[query_id * N + vec_id]
extern "C" __global__
void batch_cosine_dist_kernel(
    const char* __restrict__ queries,   // [Q * D]
    const char* __restrict__ vectors,   // [N * D]
    float* __restrict__ dists,          // [Q * N]
    int Q, int N, int D)
{
    int qid = blockIdx.x;
    if (qid >= Q) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Each thread processes a stripe of vectors.
    // With 256 threads and N=1M, each thread handles ~4000 vectors.
    for (int vid = tid; vid < N; vid += threads) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)queries[qid * D + d] * (int)vectors[vid * D + d];
        }
        dists[qid * N + vid] = 1.0f - (float)dot / 16129.0f;
    }
}

// ── Batch INT8 dot product (Q queries × N vectors, output raw dot) ───────
// Like batch_cosine_dist but outputs raw dot products (for rerank).
// dists[qid * N + vid] = dot(query[qid], vector[vid])
extern "C" __global__
void batch_dot_kernel(
    const char* __restrict__ queries,   // [Q * D]
    const char* __restrict__ vectors,   // [N * D]
    int* __restrict__ dots,             // [Q * N]
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
        dots[qid * N + vid] = dot;
    }
}

// ── Fill float buffer with 1.0 (test kernel) ────────────────────────────
extern "C" __global__
void fill_one_kernel(float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0f;
    }
}

// ── INT8 matrix multiply ─────────────────────────────────────────────────
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
