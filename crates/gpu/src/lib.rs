//! GPU acceleration for DB-Strike via NVRTC (NVIDIA Runtime Compilation).
//! Zero dependencies — pure CUDA kernels compiled at runtime.
//!
//! Architecture — GPU/CPU Hybrid Tiered:
//! - Auto-detect NVIDIA GPU on startup (cuInit + cuDeviceGet)
//! - If data fits in VRAM → GPU-only path (fastest)
//! - If data exceeds VRAM → tiered: GPU (hot int8) + RAM (f32 rerank) + NVMe (cold)
//! - If no GPU → CPU-only path (graceful fallback)
//! - Lazy kernel loading via RESP: GPU.LOAD <kernel>, GPU.INFO, GPU.UNLOAD
//!
//! Kernels (loaded on-demand):
//! - `cosine_dist` — INT8 cosine distance for vector search
//! - `matmul` — INT8 matrix multiply for bridge distances

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

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
                      sharedMemBytes: u32, hStream: *mut std::ffi::c_void,
                      kernelParams: *mut *mut std::ffi::c_void,
                      extra: *mut *mut std::ffi::c_void) -> i32;
    fn cuCtxSynchronize() -> i32;
    fn cuCtxDestroy_v2(context: *mut std::ffi::c_void) -> i32;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> i32;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    fn cuGetErrorString(error: i32, str: *mut *const i8) -> i32;
}

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;

/// Compiled kernel handle.
struct CompiledKernel {
    name: String,
    function: *mut std::ffi::c_void,
}

/// GPU state — holds CUDA context + compiled kernels.
pub struct GpuState {
    ctx: *mut std::ffi::c_void,
    module: *mut std::ffi::c_void,
    kernels: Vec<CompiledKernel>,
    #[allow(dead_code)]
    available: bool,
    vram_total: usize,
    vram_free: usize,
}

unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

/// Global GPU state — lazy init on first use.
static GPU_STATE: std::sync::OnceLock<GpuState> = std::sync::OnceLock::new();
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

impl GpuState {
    /// Try to initialize CUDA context, compile all kernels, detect VRAM.
    fn init_ctx() -> Option<Self> {
        unsafe {
            if cuInit(0) != 0 { return None; }
            let mut device = 0i32;
            if cuDeviceGet(&mut device, 0) != 0 { return None; }
            let mut ctx = std::ptr::null_mut();
            if cuCtxCreate_v2(&mut ctx, CU_CTX_SCHED_BLOCKING_SYNC, device) != 0 { return None; }

            // Detect VRAM.
            let mut vram_total: usize = 0;
            let mut vram_free: usize = 0;
            cuMemGetInfo_v2(&mut vram_free, &mut vram_total);
            let mp_count = { let mut v = 0i32; cuDeviceGetAttribute(&mut v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device); v };
            eprintln!("[GPU] NVIDIA GPU detected: {} MB VRAM ({} free), {} SMs", vram_total / 1024 / 1024, vram_free / 1024 / 1024, mp_count);

            let mut state = Self {
                ctx, module: std::ptr::null_mut(), kernels: Vec::new(),
                available: true, vram_total, vram_free,
            };

            // Compile kernel sources via NVRTC (each source = one kernel).
            eprintln!("[GPU] Compiling CUDA kernels via NVRTC...");
            let kernel_srcs = [("cosine_dist", COSINE_DIST_SRC), ("fill_one", FILL_ONE_SRC)];
            for (name, src) in &kernel_srcs {
                match state.compile_kernel(name, src) {
                    Some(_) => eprintln!("[GPU] Kernel '{}' compiled", name),
                    None => eprintln!("[GPU] Kernel '{}' FAILED", name),
                }
            }
            Some(state)
        }
    }

    /// Compile a kernel from CUDA source.
    fn compile_kernel(&mut self, name: &str, source: &str) -> Option<*mut std::ffi::c_void> {
        unsafe {
            let src_cstr = CString::new(source).ok()?;
            let name_cstr = CString::new(name).ok()?;
            let mut prog: NvrtcProgram = std::ptr::null_mut();
            let ret = nvrtcCreateProgram(&mut prog, src_cstr.as_ptr(), name_cstr.as_ptr(),
                                          0, std::ptr::null(), std::ptr::null());
            if ret != NVRTC_SUCCESS { return None; }

            let arch = CString::new("--gpu-architecture=compute_86").ok()?;
            let opts = [arch.as_ptr()];
            let ret = nvrtcCompileProgram(prog, opts.len() as i32, opts.as_ptr());
            if ret != NVRTC_SUCCESS {
                let mut log_size = 0usize;
                nvrtcGetProgramLogSize(prog, &mut log_size);
                let mut log = vec![0i8; log_size];
                nvrtcGetProgramLog(prog, log.as_mut_ptr());
                let log_bytes: &[u8] = std::slice::from_raw_parts(log.as_ptr() as *const u8, log.len());
                eprintln!("[GPU] Compile error '{}': {}", name, String::from_utf8_lossy(log_bytes));
                nvrtcDestroyProgram(&mut prog);
                return None;
            }

            let mut ptx_size = 0usize;
            nvrtcGetPTXSize(prog, &mut ptx_size);
            let mut ptx = vec![0i8; ptx_size];
            nvrtcGetPTX(prog, ptx.as_mut_ptr());
            nvrtcDestroyProgram(&mut prog);

            // Load PTX into module (or create new module).
            if self.module.is_null() {
                let ret = cuModuleLoadDataEx(&mut self.module, ptx.as_ptr(), 0,
                                              std::ptr::null(), std::ptr::null_mut());
                if ret != 0 { return None; }
            }

            // Get function handle.
            let cname = CString::new(format!("{}_kernel", name)).ok()?;
            let mut func = std::ptr::null_mut();
            if cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) == 0 {
                self.kernels.push(CompiledKernel { name: name.to_string(), function: func });
                Some(func)
            } else {
                None
            }
        }
    }

    /// Get a compiled kernel by name.
    fn get_kernel(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.kernels.iter().find(|k| k.name == name).map(|k| k.function)
    }
}

// ── Kernel source code (lazy-loaded per RESP command) ──────────────────────

const COSINE_DIST_SRC: &str = include_str!("../kernels/cosine_dist.cu");
const FILL_ONE_SRC: &str = include_str!("../kernels/fill_one.cu");
#[allow(dead_code)]
const MATMUL_SRC: &str = include_str!("../kernels/int8_matmul.cu");

// ── Public API ──────────────────────────────────────────────────────────────

/// Initialize GPU (lazy). Returns true if CUDA is available.
pub fn gpu_init() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed) || {
        let state = GpuState::init_ctx();
        if let Some(s) = state {
            let _ = GPU_STATE.set(s);
            GPU_ENABLED.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

/// Check if GPU is available.
pub fn gpu_available() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

/// Load a kernel by name (RESP: GPU.LOAD <name>).
/// Kernels are compiled at init; this just checks availability.
pub fn gpu_load_kernel(name: &str) -> bool {
    if !gpu_init() { return false; }
    GPU_STATE.get().map(|s| s.get_kernel(name).is_some()).unwrap_or(false)
}

/// Get GPU info (RESP: GPU.INFO).
pub fn gpu_info() -> Vec<(&'static str, String)> {
    let mut info = Vec::new();
    info.push(("available", gpu_available().to_string()));
    if let Some(state) = GPU_STATE.get() {
        info.push(("vram_total_mb", (state.vram_total / 1024 / 1024).to_string()));
        info.push(("vram_free_mb", (state.vram_free / 1024 / 1024).to_string()));
        info.push(("kernels_loaded", state.kernels.len().to_string()));
        for k in &state.kernels {
            info.push(("kernel", k.name.clone()));
        }
    }
    info
}

/// Check if `n` vectors of `dim` dimensions fit in GPU VRAM.
/// Returns (fits, vram_needed_bytes, vram_free_bytes).
pub fn gpu_check_capacity(n: usize, dim: usize) -> (bool, usize, usize) {
    if let Some(state) = GPU_STATE.get() {
        let needed = n * dim; // int8 vectors
        (needed <= state.vram_free, needed, state.vram_free)
    } else {
        (false, 0, 0)
    }
}

/// Auto-tier: decide GPU vs CPU+RAM based on data size vs VRAM.
/// Returns the optimal strategy.
pub fn gpu_tier_strategy(n: usize, dim: usize) -> GpuTier {
    if !gpu_available() {
        return GpuTier::CpuOnly;
    }
    let (fits, needed, free) = gpu_check_capacity(n, dim);
    if fits {
        GpuTier::GpuOnly
    } else if needed <= free * 3 {
        // Data 3x VRAM — tiered GPU+RAM
        GpuTier::GpuPlusRam
    } else {
        // Data >3x VRAM — GPU+RAM+CPU
        GpuTier::GpuRamCpu
    }
}

/// GPU tier strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuTier {
    /// No GPU — CPU only.
    CpuOnly,
    /// Data fits in VRAM — GPU only (fastest).
    GpuOnly,
    /// Data exceeds VRAM — GPU (hot) + RAM (warm).
    GpuPlusRam,
    /// Data >> VRAM — GPU (hot) + RAM (warm) + CPU (cold).
    GpuRamCpu,
}

/// INT8 cosine distance on GPU. Returns None if GPU unavailable or kernel not loaded.
pub fn gpu_cosine_dist(query: &[i8], vectors: &[i8], n: usize, dim: usize) -> Option<Vec<f32>> {
    let func = GPU_STATE.get()?.get_kernel("cosine_dist")?;
    unsafe {
        let q_bytes = query.len();
        let v_bytes = vectors.len();
        let d_bytes = n * 4;

        let mut d_q = 0u64; let mut d_v = 0u64; let mut d_d = 0u64;
        let r = cuMemAlloc_v2(&mut d_q, q_bytes); if r != 0 { return None; }
        let r = cuMemAlloc_v2(&mut d_v, v_bytes); if r != 0 { return None; }
        let r = cuMemAlloc_v2(&mut d_d, d_bytes); if r != 0 { return None; }
        cuMemcpyHtoD_v2(d_q, query.as_ptr() as *const std::ffi::c_void, q_bytes);
        cuMemcpyHtoD_v2(d_v, vectors.as_ptr() as *const std::ffi::c_void, v_bytes);

        let threads = 256u32;
        let blocks = (n as u32 + threads - 1) / threads;
        let n_val = n as i32;
        let dim_val = dim as i32;
        let mut arg0 = d_q as *mut std::ffi::c_void;
        let mut arg1 = d_v as *mut std::ffi::c_void;
        let mut arg2 = d_d as *mut std::ffi::c_void;
        let mut arg3 = &n_val as *const i32 as *mut std::ffi::c_void;
        let mut arg4 = &dim_val as *const i32 as *mut std::ffi::c_void;
        let mut args = [&mut arg0, &mut arg1, &mut arg2, &mut arg3, &mut arg4];
        let r = cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0,
                      std::ptr::null_mut(), args.as_mut_ptr() as *mut *mut std::ffi::c_void,
                      std::ptr::null_mut());
        if r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] cuLaunchKernel error {}: {}", r, msg);
            return None;
        }
        let sync_r = cuCtxSynchronize();
        if sync_r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(sync_r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] cuCtxSynchronize error {}: {}", sync_r, msg);
            return None;
        }

        let mut dists = vec![0.0f32; n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, d_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
        Some(dists)
    }
}

/// INT8 matmul on GPU. Returns None if GPU unavailable.
pub fn gpu_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
    let func = GPU_STATE.get()?.get_kernel("matmul")?;
    unsafe {
        let a_bytes = m * k; let b_bytes = k * n; let c_bytes = m * n * 4;
        let mut d_a = 0u64; let mut d_b = 0u64; let mut d_c = 0u64;
        cuMemAlloc_v2(&mut d_a, a_bytes); cuMemAlloc_v2(&mut d_b, b_bytes); cuMemAlloc_v2(&mut d_c, c_bytes);
        cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const std::ffi::c_void, a_bytes);
        cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const std::ffi::c_void, b_bytes);
        let threads = 16u32;
        let bx = (n as u32 + threads - 1) / threads;
        let by = (m as u32 + threads - 1) / threads;
        let mut args = [d_a as *mut std::ffi::c_void, d_b as *mut std::ffi::c_void,
                       d_c as *mut std::ffi::c_void, m as *mut std::ffi::c_void,
                       n as *mut std::ffi::c_void, k as *mut std::ffi::c_void];
        cuLaunchKernel(func, bx, by, 1, threads, threads, 1, 0,
                      std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut());
        cuCtxSynchronize();
        let mut c = vec![0i32; m * n];
        cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut std::ffi::c_void, d_c, c_bytes);
        cuMemFree_v2(d_a); cuMemFree_v2(d_b); cuMemFree_v2(d_c);
        Some(c)
    }
}

/// Unload all kernels and destroy GPU context (RESP: GPU.UNLOAD).
pub fn gpu_unload() {
    GPU_ENABLED.store(false, Ordering::Relaxed);
    if let Some(state) = GPU_STATE.get() {
        unsafe {
            if !state.module.is_null() {
                // Module destroyed with context.
            }
            if !state.ctx.is_null() {
                cuCtxDestroy_v2(state.ctx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        #[test]
    fn test_gpu_init() {
        if !gpu_init() {
            println!("[GPU] No CUDA available (expected in CI)");
            return;
        }
        println!("[GPU] CUDA initialized");
        let info = gpu_info();
        println!("[GPU] info: {:?}", info);
        // Test memory alloc/free
        unsafe {
            let mut d = 0u64;
            let r = super::cuMemAlloc_v2(&mut d, 1024);
            assert_eq!(r, 0, "GPU mem alloc failed");
            super::cuMemFree_v2(d);
            println!("[GPU] mem alloc/free: OK");
        }
        // Test fill_one kernel — compile inline to verify GPU launch works
        unsafe {
            // Compile fill_one kernel inline
            let src = CString::new(FILL_ONE_SRC).unwrap();
            let pname = CString::new("fill_one_test").unwrap();
            let mut prog: super::NvrtcProgram = std::ptr::null_mut();
            super::nvrtcCreateProgram(&mut prog, src.as_ptr(), pname.as_ptr(), 0, std::ptr::null(), std::ptr::null());
            let arch = CString::new("--gpu-architecture=compute_86").unwrap();
            let opts = [arch.as_ptr()];
            super::nvrtcCompileProgram(prog, 1, opts.as_ptr());
            let mut ptx_size = 0usize;
            super::nvrtcGetPTXSize(prog, &mut ptx_size);
            let mut ptx = vec![0i8; ptx_size];
            super::nvrtcGetPTX(prog, ptx.as_mut_ptr());
            super::nvrtcDestroyProgram(&mut prog);
            let mut module = std::ptr::null_mut();
            super::cuModuleLoadDataEx(&mut module, ptx.as_ptr(), 0, std::ptr::null(), std::ptr::null_mut());
            let cname = CString::new("fill_one_kernel").unwrap();
            let mut func = std::ptr::null_mut();
            super::cuModuleGetFunction(&mut func, module, cname.as_ptr());

            // Launch
            let n: i32 = 16;
            let mut d = 0u64;
            super::cuMemAlloc_v2(&mut d, (n as usize) * 4);
            let mut arg0 = d as *mut std::ffi::c_void;
            let mut arg1 = n as *mut std::ffi::c_void;
            let mut args = [&mut arg0, &mut arg1];
            let r = super::cuLaunchKernel(func, 1, 1, 1, 16, 1, 1, 0,
                std::ptr::null_mut(), args.as_mut_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut());
            super::cuCtxSynchronize();
            let mut result = vec![0.0f32; n as usize];
            super::cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut std::ffi::c_void, d, (n as usize) * 4);
            super::cuMemFree_v2(d);
            println!("[GPU] fill_one: launch_ret={} all_ones={}", r, result.iter().all(|&x| x == 1.0));
            assert_eq!(r, 0, "cuLaunchKernel failed");
            assert!(result.iter().all(|&x| x == 1.0), "kernel did not fill with 1.0");
            println!("[GPU] Kernel execution VERIFIED — GPU works!");
        }
        println!("[GPU] All checks passed");
    }
}