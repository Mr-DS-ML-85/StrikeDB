// DB-Strike GPU kernels — all in one file for NVRTC.
// Compiled once via NVRTC, all function handles from one CUmodule.

// --- INT8 cosine distance ---
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

// --- Fill float buffer with 1.0 (for testing GPU launch) ---
extern "C" __global__
void fill_one_kernel(float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0f;
    }
}

// --- INT8 matrix multiply ---
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
