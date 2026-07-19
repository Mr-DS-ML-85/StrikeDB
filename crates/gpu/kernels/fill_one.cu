// Simple kernel: fill output with 1.0 for each element.
// Used to verify GPU launch mechanism works.
extern "C" __global__
void fill_one_kernel(float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0f;
    }
}
