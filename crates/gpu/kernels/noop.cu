// Trivial kernel for testing GPU launch mechanism.
extern "C" __global__
void noop_kernel(int* x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *x = 42;
    }
}
