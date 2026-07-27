//! NVRTC runtime CUDA kernel compilation.
//!
//! VUGVA uses NVRTC to compile minimal CUDA kernels at runtime:
//! - `memcpy_peer`: P2P memcpy that respects peer access flags.
//! - `tier_promote`: DRAM→VRAM bulk copy for tier promotion.
//!
//! No pre-compiled `.ptx` files — everything is compiled on first use.

use crate::ffi::cuda::*;
use crate::ffi::nvrtc::*;
use crate::{check_cu, check_nvrtc, Result, VugvaError};

// ============================================================================
// Compiled kernel handle
// ============================================================================

/// A compiled and loaded NVRTC kernel module.
pub struct CompiledKernel {
    module: CUmodule,
    function: CUfunction,
    _program: nvrtcProgram,
}

impl CompiledKernel {
    /// Get the raw function handle for `cuLaunchKernel`.
    pub fn function(&self) -> CUfunction {
        self.function
    }

    /// Get the raw module handle.
    pub fn module(&self) -> CUmodule {
        self.module
    }
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            nvrtcDestroyProgram(&mut self._program);
        }
    }
}

// ============================================================================
// NVRTC compilation
// ============================================================================

/// Compile a CUDA source string to PTX and load it into a CUDA module.
///
/// Uses NVRTC (Runtime Compilation) to compile at runtime without
/// requiring a host `nvcc` compiler.
fn compile_and_load(
    source: &str,
    kernel_name: &str,
    arch: &str, // e.g. "sm_89"
) -> Result<CompiledKernel> {
    let c_source = std::ffi::CString::new(source).unwrap();
    // NVRTC stamps this name onto every diagnostic it emits. It used to be the
    // constant "vugva_kernel", so a syntax error in *any* kernel reported the
    // same filename and gave no clue which source string failed. Naming it after
    // the entry point makes the compile log self-identifying.
    let c_name = std::ffi::CString::new(format!("{kernel_name}.cu")).unwrap();
    let c_kernel = std::ffi::CString::new(kernel_name).unwrap();

    let mut program = nvrtcProgram(std::ptr::null_mut());

    // Create program
    unsafe {
        check_nvrtc(
            "nvrtcCreateProgram",
            nvrtcCreateProgram(
                &mut program,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
        )?;
    }

    // Compile options
    let arch_flag = std::ffi::CString::new(format!("--gpu-architecture={arch}")).unwrap();
    let mut options = vec![arch_flag.as_ptr()];
    let compile_res =
        unsafe { nvrtcCompileProgram(program, options.len() as i32, options.as_mut_ptr()) };

    if compile_res != NVRTC_SUCCESS {
        // Get compile log for debugging
        let mut log_size: usize = 0;
        unsafe {
            nvrtcGetProgramLogSize(program, &mut log_size);
        }
        let mut log = vec![0u8; log_size];
        unsafe {
            nvrtcGetProgramLog(program, log.as_mut_ptr() as *mut i8);
        }
        let log_str = String::from_utf8_lossy(&log);
        eprintln!("NVRTC compile error:\n{log_str}");
        return Err(VugvaError::NvrtcError {
            fn_name: "nvrtcCompileProgram",
            code: compile_res,
        });
    }

    // Get PTX
    let mut ptx_size: usize = 0;
    unsafe {
        check_nvrtc("nvrtcGetPTXSize", nvrtcGetPTXSize(program, &mut ptx_size))?;
    }
    let mut ptx = vec![0u8; ptx_size];
    unsafe {
        check_nvrtc(
            "nvrtcGetPTX",
            nvrtcGetPTX(program, ptx.as_mut_ptr() as *mut i8),
        )?;
    }

    // Load into CUDA module
    let mut module = CUmodule(std::ptr::null_mut());
    unsafe {
        check_cu(
            "cuModuleLoadData",
            cuModuleLoadData(&mut module, ptx.as_ptr() as *const std::ffi::c_void),
        )?;
    }

    // Get function handle
    let mut function = CUfunction(std::ptr::null_mut());
    unsafe {
        check_cu(
            "cuModuleGetFunction",
            cuModuleGetFunction(&mut function, module, c_kernel.as_ptr()),
        )?;
    }

    Ok(CompiledKernel {
        module,
        function,
        _program: program,
    })
}

/// Auto-detect the best SM architecture for the current GPU.
pub fn detect_sm_arch(gpu_ordinal: i32) -> Result<String> {
    let dev = CUdevice(gpu_ordinal);
    let (mut major, mut minor) = (0i32, 0i32);
    unsafe {
        check_cu(
            "cuDeviceComputeCapability",
            cuDeviceComputeCapability(&mut major, &mut minor, dev),
        )?;
    }
    Ok(format!("sm_{major}{minor}"))
}

// ============================================================================
// Built-in kernels
// ============================================================================

/// CUDA source for the peer-to-peer memcpy kernel.
const MEMCPY_PEER_SRC: &str = r#"
extern "C" __global__ void memcpy_peer(
    const char* __restrict__ src,
    char* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
"#;

/// CUDA source for the tier-promotion bulk copy kernel.
const TIER_PROMOTE_SRC: &str = r#"
extern "C" __global__ void tier_promote(
    const char* __restrict__ src,
    char* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
"#;

/// Compile and return the `memcpy_peer` kernel.
pub fn compile_memcpy_peer(gpu_ordinal: i32) -> Result<CompiledKernel> {
    let arch = detect_sm_arch(gpu_ordinal)?;
    compile_and_load(MEMCPY_PEER_SRC, "memcpy_peer", &arch)
}

/// Compile and return the `tier_promote` kernel.
pub fn compile_tier_promote(gpu_ordinal: i32) -> Result<CompiledKernel> {
    let arch = detect_sm_arch(gpu_ordinal)?;
    compile_and_load(TIER_PROMOTE_SRC, "tier_promote", &arch)
}

/// Launch a kernel with the given grid/block dimensions and arguments.
///
/// # Safety
///
/// - `kernel` must be a valid compiled kernel.
/// - `args` must contain the correct number and types of arguments
///   matching the kernel signature.
/// - `stream` must be a valid CUDA stream (or null for default stream).
pub unsafe fn launch_kernel(
    kernel: &CompiledKernel,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem: u32,
    stream: CUstream,
    args: &[*mut std::ffi::c_void],
) -> Result<()> {
    check_cu(
        "cuLaunchKernel",
        cuLaunchKernel(
            kernel.function(),
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem,
            stream,
            args.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut(),
        ),
    )
}
