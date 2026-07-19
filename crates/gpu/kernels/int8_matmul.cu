// INT8 matrix multiply + cosine distance kernels for DB-Strike.
// Pure CUDA, no cuBLAS, compiled via NVRTC at runtime.
// char is signed in CUDA by default (matches Rust i8).

// Trivial kernel for testing launch mechanism.
extern "C" __global__
void noop_kernel(int* x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *x = 42;
    }
}

extern "C" __global__
void cosine_dist_kernel(
    const char* __restrict__ query,   // [D]
    const char* __restrict__ vectors, // [N x D] row-major
    float* __restrict__ dists,        // [N]
    int N, int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int dot = 0;
        for (int d = 0; d < D; d++) {
            dot += (int)query[d] * (int)vectors[tid * D + d];
        }
        dists[tid] = 1.0f - (float)dot / 16129.0f; // 127*127
    }
}

extern "C" __global__
void matmul_kernel(
    const char* __restrict__ A,  // [M x K]
    const char* __restrict__ B,  // [K x N]
    int* __restrict__ C,         // [M x N]
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int)A[row * K + k] * (int)B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
