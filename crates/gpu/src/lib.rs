//! GPU acceleration for DB-Strike via NVRTC (NVIDIA Runtime Compilation).
//! No cuBLAS, no cuDNN — just raw CUDA kernels compiled at runtime.
//!
//! Provides:
//! - INT8 matrix multiply for bridge distance computation
//! - Batch cosine distance for vector search
//! - Parallel HNSW bridge connections

use std::ffi::CString;

// NVRTC FFI bindings (minimal, only what we need).
type NvrtcProgram = *mut std::ffi::c_void;
type NvrtcResult = i32;

const NVRTC_SUCCESS: NvrtcResult = 0;

extern "C" {
    fn nvrtcCreateProgram(prog: *mut NvrtcProgram, src: *const i8, name: *const i8,
                          numHeaders: i32, headers: *const *const i8,
                          includeNames: *const *const i8) -> NvrtcResult;
    fn nvrtcCompileProgram(prog: NvrtcProgram, numOptions: i32,
                           options: *const *const i8) -> NvrtcResult;
    fn nvrtcGetPTXSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut i8) -> NvrtcResult;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> NvrtcResult;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut i8) -> NvrtcResult;

    // CUDA driver API
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate_v2(context: *mut *mut std::ffi::c_void, flags: u32, device: i32) -> i32;
    fn cuModuleLoadDataEx(module: *mut *mut std::ffi::c_void, image: *const i8,
                          numOptions: u32, options: *const u32,
                          optionValues: *mut *mut std::ffi::c_void) -> i32;
    fn cuModuleGetFunction(function: *mut *mut std::ffi::c_void,
                           hmod: *mut std::ffi::c_void,
                           name: *const i8) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemcpyHtoD_v2(dstDevice: u64, srcHost: *const std::ffi::c_void, byteCount: usize) -> i32;
    fn cuMemcpyDtoH_v2(dstHost: *mut std::ffi::c_void, srcDevice: u64, byteCount: usize) -> i32;
    fn cuLaunchKernel(f: *mut std::ffi::c_void,
                      gridDimX: u32, gridDimY: u32, gridDimZ: u32,
                      blockDimX: u32, blockDimY: u32, blockDimZ: u32,
                      sharedMemBytes: u32,
                      hStream: *mut std::ffi::c_void,
                      kernelParams: *mut *mut std::ffi::c_void,
                      extra: *mut *mut std::ffi::c_void) -> i32;
    fn cuCtxSynchronize() -> i32;
}

const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;

/// GPU context — holds the NVRTC-compiled module and CUDA context.
pub struct GpuContext {
    module: *mut std::ffi::c_void,
    _ctx: *mut std::ffi::c_void,
}

unsafe impl Send for GpuContext {}

impl GpuContext {
    /// Initialize GPU and compile CUDA kernels via NVRTC.
    pub fn new() -> Option<Self> {
        unsafe {
            if cuInit(0) != 0 { return None; }
            let mut device = 0i32;
            if cuDeviceGet(&mut device, 0) != 0 { return None; }
            let mut ctx = std::ptr::null_mut();
            if cuCtxCreate_v2(&mut ctx, CU_CTX_SCHED_BLOCKING_SYNC, device) != 0 { return None; }

            // Compile kernels via NVRTC.
            let kernel_src = include_str!("../kernels/int8_matmul.cu");
            let src_cstr = CString::new(kernel_src).ok()?;
            let name_cstr = CString::new("int8_matmul").ok()?;

            let mut prog: NvrtcProgram = std::ptr::null_mut();
            let ret = nvrtcCreateProgram(&mut prog, src_cstr.as_ptr(), name_cstr.as_ptr(),
                                          0, std::ptr::null(), std::ptr::null());
            if ret != NVRTC_SUCCESS { return None; }

            let arch = CString::new("--gpu-architecture=compute_86").ok()?;
            let opts = [arch.as_ptr()];
            let ret = nvrtcCompileProgram(prog, 1, opts.as_ptr());
            if ret != NVRTC_SUCCESS {
                let mut log_size = 0usize;
                nvrtcGetProgramLogSize(prog, &mut log_size);
                let mut log = vec![0i8; log_size];
                nvrtcGetProgramLog(prog, log.as_mut_ptr());
                let log_bytes: &[u8] = std::slice::from_raw_parts(log.as_ptr() as *const u8, log.len());
                eprintln!("[GPU] NVRTC compile error: {}", String::from_utf8_lossy(log_bytes));
                nvrtcDestroyProgram(&mut prog);
                return None;
            }

            let mut ptx_size = 0usize;
            nvrtcGetPTXSize(prog, &mut ptx_size);
            let mut ptx = vec![0i8; ptx_size];
            nvrtcGetPTX(prog, ptx.as_mut_ptr());
            nvrtcDestroyProgram(&mut prog);

            let mut module = std::ptr::null_mut();
            let ret = cuModuleLoadDataEx(&mut module, ptx.as_ptr(), 0,
                                          std::ptr::null(), std::ptr::null_mut());
            if ret != 0 { return None; }

            eprintln!("[GPU] NVRTC kernels compiled successfully (RTX 4060)");
            Some(Self { module, _ctx: ctx })
        }
    }

    /// Get a kernel function by name.
    fn get_kernel(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let cname = CString::new(name).ok()?;
        let mut func = std::ptr::null_mut();
        unsafe {
            if cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) == 0 {
                Some(func)
            } else {
                None
            }
        }
    }

    /// INT8 cosine distance: compute distances from query to all vectors on GPU.
    /// Returns Vec<f32> of distances.
    pub fn int8_cosine_dist(&self, query: &[i8], vectors: &[i8], n: usize, dim: usize) -> Option<Vec<f32>> {
        let func = self.get_kernel("int8_cosine_dist_kernel")?;
        unsafe {
            let q_bytes = query.len();
            let v_bytes = vectors.len() * std::mem::size_of::<i8>();
            let d_bytes = n * std::mem::size_of::<f32>();

            let mut d_query = 0u64;
            let mut d_vectors = 0u64;
            let mut d_dists = 0u64;
            cuMemAlloc_v2(&mut d_query, q_bytes);
            cuMemAlloc_v2(&mut d_vectors, v_bytes);
            cuMemAlloc_v2(&mut d_dists, d_bytes);

            cuMemcpyHtoD_v2(d_query, query.as_ptr() as *const std::ffi::c_void, q_bytes);
            cuMemcpyHtoD_v2(d_vectors, vectors.as_ptr() as *const std::ffi::c_void, v_bytes);

            let threads = 256u32;
            let blocks = ((n as u32 + threads - 1) / threads, 1u32, 1u32);
            let mut args = [d_query as *mut std::ffi::c_void,
                           d_vectors as *mut std::ffi::c_void,
                           d_dists as *mut std::ffi::c_void,
                           n as *mut std::ffi::c_void,
                           dim as *mut std::ffi::c_void];
            cuLaunchKernel(func, blocks.0, blocks.1, blocks.2,
                          threads, 1, 1, 0, std::ptr::null_mut(),
                          args.as_mut_ptr(), std::ptr::null_mut());
            cuCtxSynchronize();

            let mut dists = vec![0.0f32; n];
            cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_dists, d_bytes);

            cuMemFree_v2(d_query);
            cuMemFree_v2(d_vectors);
            cuMemFree_v2(d_dists);

            Some(dists)
        }
    }

    /// INT8 matmul: A[M x K] × B[K x N] on GPU.
    pub fn int8_matmul(&self, a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
        let func = self.get_kernel("int8_matmul_kernel")?;
        unsafe {
            let a_bytes = m * k;
            let b_bytes = k * n;
            let c_bytes = m * n * 4;

            let mut d_a = 0u64;
            let mut d_b = 0u64;
            let mut d_c = 0u64;
            cuMemAlloc_v2(&mut d_a, a_bytes);
            cuMemAlloc_v2(&mut d_b, b_bytes);
            cuMemAlloc_v2(&mut d_c, c_bytes);

            cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const std::ffi::c_void, a_bytes);
            cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const std::ffi::c_void, b_bytes);

            let threads = 16u32;
            let blocks = (
                ((n as u32 + threads - 1) / threads),
                ((m as u32 + threads - 1) / threads),
                1u32,
            );
            let mut args = [d_a as *mut std::ffi::c_void,
                           d_b as *mut std::ffi::c_void,
                           d_c as *mut std::ffi::c_void,
                           m as *mut std::ffi::c_void,
                           n as *mut std::ffi::c_void,
                           k as *mut std::ffi::c_void];
            cuLaunchKernel(func, blocks.0, blocks.1, blocks.2,
                          threads, threads, 1, 0, std::ptr::null_mut(),
                          args.as_mut_ptr(), std::ptr::null_mut());
            cuCtxSynchronize();

            let mut c = vec![0i32; m * n];
            cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut std::ffi::c_void, d_c, c_bytes);

            cuMemFree_v2(d_a);
            cuMemFree_v2(d_b);
            cuMemFree_v2(d_c);

            Some(c)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        if let Some(gpu) = GpuContext::new() {
            // Test INT8 matmul
            let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
            let b: Vec<i8> = vec![1, 0, 0, 1, 0, 1];
            let c = gpu.int8_matmul(&a, &b, 2, 3, 2).unwrap();
            assert_eq!(c, vec![1, 4, 4, 10]); // [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]... wait
            // Actually: A=[1,2,3;4,5,6], B=[1,0;0,1;0,1]
            // C[0,0] = 1*1+2*0+3*0 = 1
            // C[0,1] = 1*0+2*1+3*1 = 5
            // C[1,0] = 4*1+5*0+6*0 = 4
            // C[1,1] = 4*0+5*1+6*1 = 11
            // Hmm, B is [K x N] = [3 x 2] row-major = [1,0, 0,1, 0,1]
            // C[0,0] = 1*1+2*0+3*0 = 1
            // C[0,1] = 1*0+2*1+3*1 = 5
            // C[1,0] = 4*1+5*0+6*0 = 4
            // C[1,1] = 4*0+5*1+6*1 = 11
            // So c should be [1, 5, 4, 11]
            assert_eq!(c, vec![1, 5, 4, 11]);
        }
    }
}
