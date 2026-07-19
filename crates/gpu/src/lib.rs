//! GPU acceleration for DB-Strike via NVRTC (NVIDIA Runtime Compilation).
//! Zero dependencies — pure CUDA kernels compiled at runtime.
//!
//! Architecture — GPU/CPU Hybrid Tiered:
//! - Auto-detect NVIDIA GPU on startup (cuInit + cuDeviceGet)
//! - If data fits in VRAM → GPU-only path (fastest)
//! - If data exceeds VRAM → tiered: GPU (hot int8) + RAM (f32 rerank) + NVMe (cold)
//! - If no GPU → CPU-only path (graceful fallback)
//! - Kernels disabled by default, loaded on-demand:
//!   - RESP: GPU.LOAD <kernel>, GPU.INFO, GPU.UNLOAD
//!   - Auto: first call to gpu_cosine_dist/gpu_matmul triggers lazy load
//!
//! Kernels (loaded on-demand):
//! - `cosine_dist` — INT8 cosine distance for vector search (VSEARCH, VADD, bridge)
//! - `matmul` — INT8 matrix multiply for VADDBATCH PAR

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

type NvrtcProgram = *mut std::ffi::c_void;

extern "C" {
    fn nvrtcCreateProgram(prog: *mut NvrtcProgram, src: *const i8, name: *const i8,
                          numHeaders: i32, headers: *const *const i8,
                          includeNames: *const *const i8) -> i32;
    fn nvrtcCompileProgram(prog: NvrtcProgram, numOptions: i32,
                           options: *const *const i8) -> i32;
    fn nvrtcGetPTXSize(prog: NvrtcProgram, size: *mut usize) -> i32;
    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut i8) -> i32;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> i32;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, size: *mut usize) -> i32;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut i8) -> i32;
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
    fn cuCtxSetCurrent(ctx: *mut std::ffi::c_void) -> i32;
}

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;

/// Compiled kernel handle.
struct CompiledKernel {
    name: String,
    function: *mut std::ffi::c_void,
}

/// GPU state — holds CUDA context + compiled kernels.
/// Kernels are compiled lazily on first demand, NOT at init.
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
static KERNELS_COMPILED: AtomicBool = AtomicBool::new(false);
static KERNEL_LOCK: Mutex<()> = Mutex::new(());

impl GpuState {
    /// Initialize CUDA context + detect VRAM. NO kernel compilation.
    fn init_ctx() -> Option<Self> {
        unsafe {
            if cuInit(0) != 0 { return None; }
            let mut device = 0i32;
            if cuDeviceGet(&mut device, 0) != 0 { return None; }
            let mut ctx = std::ptr::null_mut();
            if cuCtxCreate_v2(&mut ctx, CU_CTX_SCHED_BLOCKING_SYNC, device) != 0 { return None; }

            let mut vram_total: usize = 0;
            let mut vram_free: usize = 0;
            cuMemGetInfo_v2(&mut vram_free, &mut vram_total);
            let mp_count = { let mut v = 0i32; cuDeviceGetAttribute(&mut v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device); v };
            eprintln!("[GPU] NVIDIA GPU detected: {} MB VRAM ({} free), {} SMs",
                vram_total / 1024 / 1024, vram_free / 1024 / 1024, mp_count);

            Some(Self {
                ctx, module: std::ptr::null_mut(), kernels: Vec::new(),
                available: true, vram_total, vram_free,
            })
        }
    }

    /// Lazy-compile kernels. Called on first kernel use (not at init).
    /// Compiles all kernels from one source file into one module.
    /// Thread-safe: only compiles once via KERNELS_COMPILED flag.
    fn ensure_kernels(&mut self) {
        if KERNELS_COMPILED.load(Ordering::Acquire) { return; }
        if !self.module.is_null() { return; }
        let _guard = KERNEL_LOCK.lock().unwrap();
        // Double-check after acquiring lock.
        if KERNELS_COMPILED.load(Ordering::Acquire) { return; }
        if !self.module.is_null() { return; }
        if !ensure_ctx() { return; }
        unsafe {
            eprintln!("[GPU] Compiling CUDA kernels via NVRTC (lazy)...");
            let src_cstr = CString::new(KERNEL_SRC).unwrap();
            let name_cstr = CString::new("dbstrike_kernels").unwrap();
            let mut prog: NvrtcProgram = std::ptr::null_mut();
            nvrtcCreateProgram(&mut prog, src_cstr.as_ptr(), name_cstr.as_ptr(), 0,
                               std::ptr::null(), std::ptr::null());
            let arch = CString::new("--gpu-architecture=compute_86").unwrap();
            let opts = [arch.as_ptr()];
            let ret = nvrtcCompileProgram(prog, 1, opts.as_ptr());
            if ret != 0 {
                let mut log_size = 0usize;
                nvrtcGetProgramLogSize(prog, &mut log_size);
                if log_size > 1 {
                    let mut log = vec![0i8; log_size];
                    nvrtcGetProgramLog(prog, log.as_mut_ptr());
                    eprintln!("[GPU] NVRTC compile error:\n{}",
                        String::from_utf8_lossy(&log.iter().map(|&c| c as u8).collect::<Vec<_>>()));
                }
                nvrtcDestroyProgram(&mut prog);
                return;
            }
            let mut ptx_size = 0usize;
            nvrtcGetPTXSize(prog, &mut ptx_size);
            let mut ptx = vec![0i8; ptx_size];
            nvrtcGetPTX(prog, ptx.as_mut_ptr());
            nvrtcDestroyProgram(&mut prog);
            eprintln!("[GPU] PTX compiled ({} bytes)", ptx_size);

            let ret = cuModuleLoadDataEx(&mut self.module, ptx.as_ptr(), 0,
                                          std::ptr::null(), std::ptr::null_mut());
            if ret != 0 {
                eprintln!("[GPU] cuModuleLoadDataEx failed: {}", ret);
                return;
            }
            // Get ALL function handles from the loaded module.
            for name in &["cosine_dist_kernel", "matmul_kernel"] {
                let cname = CString::new(*name).unwrap();
                let mut func = std::ptr::null_mut();
                if cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) == 0 {
                    let short_name = name.replace("_kernel", "");
                    self.kernels.push(CompiledKernel { name: short_name.clone(), function: func });
                }
            }
            eprintln!("[GPU] {} kernels loaded", self.kernels.len());
            KERNELS_COMPILED.store(true, Ordering::Release);
        }
    }

    /// Get a compiled kernel by name. Returns None if not yet loaded.
    fn get_kernel(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.kernels.iter().find(|k| k.name == name).map(|k| k.function)
    }
}

// ── Kernel source code (compiled once, lazily) ────────────────────────────

const KERNEL_SRC: &str = include_str!("../kernels/all_kernels.cu");

// ── Public API ──────────────────────────────────────────────────────────────

/// Initialize GPU — detect CUDA, create context. Kernels NOT compiled yet.
pub fn gpu_init() -> bool {
    if GPU_ENABLED.load(Ordering::Relaxed) { return true; }
    // Use get_or_init to prevent double-init race.
    let state = GPU_STATE.get_or_init(|| {
        GpuState::init_ctx().unwrap_or_else(|| {
            // Sentinel: create a dummy state so get_or_init doesn't retry.
            // We check `available` before using it.
            GpuState { ctx: std::ptr::null_mut(), module: std::ptr::null_mut(),
                       kernels: Vec::new(), available: false, vram_total: 0, vram_free: 0 }
        })
    });
    if state.available {
        GPU_ENABLED.store(true, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Check if GPU is available.
pub fn gpu_available() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

/// Load a kernel category (RESP: GPU.LOAD <name>).
/// "cosine_dist" loads the cosine distance kernel.
/// "matmul" loads the matrix multiply kernel.
/// "all" loads everything.
/// First call compiles PTX; subsequent calls are no-ops.
pub fn gpu_load_kernel(name: &str) -> bool {
    if !gpu_init() { return false; }
    let state = match GPU_STATE.get() {
        Some(s) => s,
        None => return false,
    };
    // Check if already loaded
    if state.get_kernel(name).is_some() { return true; }
    // Need to compile — force via raw pointer (safe: GpuState is Sync, single-threaded init).
    let state_ptr = state as *const GpuState as *mut GpuState;
    unsafe { (*state_ptr).ensure_kernels(); }
    GPU_STATE.get().map(|s| s.get_kernel(name).is_some()).unwrap_or(false)
}

/// Auto-load a specific kernel if not already loaded.
/// Called internally before gpu_cosine_dist / gpu_matmul.
fn gpu_ensure_kernel(name: &str) -> bool {
    if !gpu_init() { return false; }
    let state = GPU_STATE.get().unwrap();
    if state.get_kernel(name).is_some() { return true; }
    let state_ptr = state as *const GpuState as *mut GpuState;
    unsafe { (*state_ptr).ensure_kernels(); }
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
pub fn gpu_check_capacity(n: usize, dim: usize) -> (bool, usize, usize) {
    if let Some(state) = GPU_STATE.get() {
        let needed = n * dim;
        (needed <= state.vram_free, needed, state.vram_free)
    } else {
        (false, 0, 0)
    }
}

/// Auto-tier: decide GPU vs CPU+RAM based on data size vs VRAM.
pub fn gpu_tier_strategy(n: usize, dim: usize) -> GpuTier {
    if !gpu_available() {
        return GpuTier::CpuOnly;
    }
    let (fits, needed, free) = gpu_check_capacity(n, dim);
    if fits {
        GpuTier::GpuOnly
    } else if needed <= free * 3 {
        GpuTier::GpuPlusRam
    } else {
        GpuTier::GpuRamCpu
    }
}

/// GPU tier strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuTier {
    CpuOnly,
    GpuOnly,
    GpuPlusRam,
    GpuRamCpu,
}

/// Ensure current thread has the CUDA context active.
/// CUDA driver API contexts are thread-local; must be set per-thread.
fn ensure_ctx() -> bool {
    if let Some(state) = GPU_STATE.get() {
        unsafe { cuCtxSetCurrent(state.ctx) == 0 }
    } else {
        false
    }
}

/// INT8 cosine distance on GPU. Auto-loads kernel if needed.
/// Returns None if GPU unavailable.
pub fn gpu_cosine_dist(query: &[i8], vectors: &[i8], n: usize, dim: usize) -> Option<Vec<f32>> {
    if !gpu_ensure_kernel("cosine_dist") { return None; }
    let func = GPU_STATE.get()?.get_kernel("cosine_dist")?;
    if !ensure_ctx() { return None; }
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
        let mut arg3 = n_val as *mut std::ffi::c_void;
        let mut arg4 = dim_val as *mut std::ffi::c_void;
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
        if sync_r != 0 { return None; }

        let mut dists = vec![0.0f32; n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, d_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
        Some(dists)
    }
}

/// INT8 matmul on GPU. Auto-loads kernel if needed.
pub fn gpu_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
    if !gpu_ensure_kernel("matmul") { return None; }
    let func = GPU_STATE.get()?.get_kernel("matmul")?;
    if !ensure_ctx() { return None; }
    unsafe {
        let a_bytes = m * k; let b_bytes = k * n; let c_bytes = m * n * 4;
        let mut d_a = 0u64; let mut d_b = 0u64; let mut d_c = 0u64;
        cuMemAlloc_v2(&mut d_a, a_bytes); cuMemAlloc_v2(&mut d_b, b_bytes); cuMemAlloc_v2(&mut d_c, c_bytes);
        cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const std::ffi::c_void, a_bytes);
        cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const std::ffi::c_void, b_bytes);
        let threads = 16u32;
        let bx = (n as u32 + threads - 1) / threads;
        let by = (m as u32 + threads - 1) / threads;
        let m_val = m as i32;
        let n_val = n as i32;
        let k_val = k as i32;
        let mut arg0 = d_a as *mut std::ffi::c_void;
        let mut arg1 = d_b as *mut std::ffi::c_void;
        let mut arg2 = d_c as *mut std::ffi::c_void;
        let mut arg3 = m_val as *mut std::ffi::c_void;
        let mut arg4 = n_val as *mut std::ffi::c_void;
        let mut arg5 = k_val as *mut std::ffi::c_void;
        let mut args = [&mut arg0, &mut arg1, &mut arg2, &mut arg3, &mut arg4, &mut arg5];
        cuLaunchKernel(func, bx, by, 1, threads, threads, 1, 0,
                      std::ptr::null_mut(), args.as_mut_ptr() as *mut *mut std::ffi::c_void,
                      std::ptr::null_mut());
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
        // At this point: 0 kernels loaded (lazy)
        let state = GPU_STATE.get().unwrap();
        assert!(state.kernels.is_empty(), "kernels should be empty before explicit load");

        // Test memory alloc/free
        unsafe {
            let mut d = 0u64;
            let r = cuMemAlloc_v2(&mut d, 1024);
            assert_eq!(r, 0, "GPU mem alloc failed");
            cuMemFree_v2(d);
            println!("[GPU] mem alloc/free: OK");
        }

        // Test lazy kernel loading — first cosine_dist call triggers compile
        let dists = gpu_cosine_dist(
            &[127, 0, 0],
            &[127, 0, 0,  0, 127, 0,  0, 0, 127],
            3, 3,
        );
        match dists {
            Some(d) => {
                println!("[GPU] cosine_dist (lazy): {:?}", d);
                assert!(d[0] < 0.01, "identical should be ~0, got {}", d[0]);
                assert!(d[1] > 0.99, "orthogonal should be ~1, got {}", d[1]);
                assert!(d[2] > 0.99, "orthogonal should be ~1, got {}", d[2]);
                println!("[GPU] cosine_dist VERIFIED — lazy load works!");
            }
            None => panic!("gpu_cosine_dist returned None"),
        }

        println!("[GPU] All checks passed — lazy kernel loading works!");
    }

    #[test]
    fn test_gpu_load_explicit() {
        if !gpu_init() { return; }
        // Explicit load via GPU.LOAD
        let loaded = gpu_load_kernel("cosine_dist");
        println!("[GPU] GPU.LOAD cosine_dist: {}", loaded);
        assert!(loaded, "gpu_load_kernel should succeed");
        let loaded2 = gpu_load_kernel("matmul");
        println!("[GPU] GPU.LOAD matmul: {}", loaded2);
        assert!(loaded2, "gpu_load_kernel matmul should succeed");
    }
}
