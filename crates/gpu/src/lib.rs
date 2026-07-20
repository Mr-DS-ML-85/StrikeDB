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

pub mod vugva;

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
    fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    fn cuCtxSetCurrent(ctx: *mut std::ffi::c_void) -> i32;
}

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;
// VUGVA: CUDA uses this flag for managed memory — GPU can read RAM directly
// without cuMemcpyHtoD. The CUDA driver handles page migration transparently.
// GPU accesses host pointers; if page is not in VRAM, a page fault triggers
// async DMA transfer. This is the software RDMA the VUGVA paper describes.
const CU_MEM_ATTACH_GLOBAL: u32 = 0x01;

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
static KERNELS_ATTEMPTED: AtomicBool = AtomicBool::new(false);
static KERNEL_LOCK: Mutex<()> = Mutex::new(());
// CUDA driver API is NOT thread-safe. All CUDA calls MUST go through this lock.
static GPU_ACCESS: Mutex<()> = Mutex::new(());

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
        if KERNELS_ATTEMPTED.load(Ordering::Acquire) { return; } // Don't retry after failure
        let _guard = KERNEL_LOCK.lock().unwrap();
        if KERNELS_COMPILED.load(Ordering::Acquire) { return; }
        if KERNELS_ATTEMPTED.load(Ordering::Acquire) { return; }
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
            for name in &["cosine_dist_kernel", "batch_cosine_dist_kernel", "cagra_search_kernel", "matmul_kernel"] {
                let cname = CString::new(*name).unwrap();
                let mut func = std::ptr::null_mut();
                if cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) == 0 {
                    let short_name = name.replace("_kernel", "");
                    self.kernels.push(CompiledKernel { name: short_name.clone(), function: func });
                }
            }
            eprintln!("[GPU] {} kernels loaded", self.kernels.len());
            KERNELS_ATTEMPTED.store(true, Ordering::Release);
            if !self.module.is_null() {
                KERNELS_COMPILED.store(true, Ordering::Release);
            }
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
    let mode = gpu_get_mode();
    info.push(("mode", format!("{:?}", mode)));
    info.push(("gpu_available", gpu_available().to_string()));
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
pub fn gpu_tier_strategy(n: usize, dim: usize) -> ComputeMode {
    gpu_auto_mode(n, dim)
}

/// Compute mode — controls how DB-Strike routes work across GPU/RAM/CPU.
///
/// ```text
/// ┌─────────────────────────────────────────┐
/// │         GPU AUTO-DETECTION              │
/// │  cuInit + cuDeviceGet + cuMemGetInfo    │
/// ├─────────────────────────────────────────┤
/// │  Data + Embeddings ≤ VRAM?             │
/// │    YES → TURBO  (GPU only, fastest)    │
/// │    NO  → HYBRID (GPU hot + RAM + CPU)  │
/// │         Split by shard                  │
/// │         GPU handles hot shard           │
/// │         RAM handles warm shard          │
/// │         CPU handles cold shard          │
/// ├─────────────────────────────────────────┤
/// │  No GPU? → CPU_ONLY (pure CPU path)    │
/// └─────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    /// Full GPU — all distance computation + graph traversal on GPU.
    /// Fastest. Requires NVIDIA GPU and data fits in VRAM.
    Turbo,
    /// GPU + RAM + CPU — auto-offload when VRAM insufficient.
    /// Hot data on GPU, warm in RAM, cold on CPU/disk.
    /// Same pattern as StrikeDB's tiered memory (RAM → NVMe → object store).
    Hybrid,
    /// Pure CPU — no GPU available or user explicitly disabled GPU.
    CpuOnly,
}

static COMPUTE_MODE: Mutex<ComputeMode> = Mutex::new(ComputeMode::CpuOnly);

/// Get current compute mode.
pub fn gpu_get_mode() -> ComputeMode {
    *COMPUTE_MODE.lock().unwrap()
}

/// Set compute mode explicitly (RESP: GPU.MODE turbo|hybrid|cpu).
pub fn gpu_set_mode(mode: ComputeMode) {
    *COMPUTE_MODE.lock().unwrap() = mode;
    eprintln!("[GPU] Compute mode set to {:?}", mode);
}

/// Auto-detect optimal mode based on GPU availability and data size.
/// Call once after init with the dataset dimensions.
pub fn gpu_auto_mode(n: usize, dim: usize) -> ComputeMode {
    if !gpu_available() {
        return ComputeMode::CpuOnly;
    }
    let (_, _, vram_free) = gpu_check_capacity(n, dim);
    // Peak VRAM: vectors + kNN build distance matrix + graph + 500MB overhead
    let shards = 16usize;
    let shard_n = (n + shards - 1) / shards;
    let build_peak = shard_n * dim + 1024 * shard_n * 4 + shard_n * 80 * 4 + 500 * 1024 * 1024;
    let data_only = n * dim + n * 80 * 4;

    if build_peak <= vram_free {
        let mode = ComputeMode::Turbo;
        gpu_set_mode(mode);
        mode
    } else if data_only <= vram_free {
        let mode = ComputeMode::Hybrid;
        gpu_set_mode(mode);
        mode
    } else {
        let mode = ComputeMode::Hybrid;
        gpu_set_mode(mode);
        mode
    }
}

/// Check if data of given size should use GPU path in current mode.
pub fn gpu_should_use_gpu(n: usize, dim: usize) -> bool {
    let mode = gpu_get_mode();
    match mode {
        ComputeMode::Turbo => true,
        ComputeMode::Hybrid => {
            // Use GPU if data fits in VRAM; otherwise CPU fallback for this chunk
            let (fits, _, _) = gpu_check_capacity(n, dim);
            fits
        }
        ComputeMode::CpuOnly => false,
    }
}

/// Legacy alias.
pub type GpuTier = ComputeMode;
#[allow(non_upper_case_globals)]
impl ComputeMode {
    pub const CpuOnly: Self = ComputeMode::CpuOnly;
    pub const GpuOnly: Self = ComputeMode::Turbo;
    pub const GpuPlusRam: Self = ComputeMode::Hybrid;
    pub const GpuRamCpu: Self = ComputeMode::Hybrid;
}

/// Ensure current thread has the CUDA context active AND holds the GPU lock.
/// CUDA driver API contexts are thread-local; must be set per-thread.
/// The GPU_ACCESS lock serializes all CUDA operations to prevent crashes.
fn ensure_ctx_locked() -> Option<std::sync::MutexGuard<'static, ()>> {
    let guard = GPU_ACCESS.lock().ok()?;
    if let Some(state) = GPU_STATE.get() {
        unsafe {
            if cuCtxSetCurrent(state.ctx) != 0 {
                return None;
            }
        }
        Some(guard)
    } else {
        None
    }
}

/// Ensure current thread has the CUDA context active (no lock).
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
    let _guard = ensure_ctx_locked()?;
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

/// Batch INT8 cosine distance: Q queries × N vectors → Q×N distances.
/// This is the CAGRA distance kernel — Q blocks, one per query.
/// Returns None if GPU unavailable.
pub fn gpu_batch_cosine_dist(queries: &[i8], vectors: &[i8], q: usize, n: usize, dim: usize) -> Option<Vec<f32>> {
    if !gpu_ensure_kernel("batch_cosine_dist") { return None; }
    let func = GPU_STATE.get()?.get_kernel("batch_cosine_dist")?;
    let _guard = ensure_ctx_locked()?;
    unsafe {
        let q_bytes = q * dim;
        let v_bytes = n * dim;
        let out_bytes = q * n * 4;
        let mut d_q = 0u64; let mut d_v = 0u64; let mut d_d = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_bytes) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_d, out_bytes) != 0 { return None; }
        cuMemcpyHtoD_v2(d_q, queries.as_ptr() as *const std::ffi::c_void, q_bytes);
        cuMemcpyHtoD_v2(d_v, vectors.as_ptr() as *const std::ffi::c_void, v_bytes);
        let threads = 256u32;
        let q_val = q as i32;
        let n_val = n as i32;
        let dim_val = dim as i32;
        let mut arg0 = d_q as *mut std::ffi::c_void;
        let mut arg1 = d_v as *mut std::ffi::c_void;
        let mut arg2 = d_d as *mut std::ffi::c_void;
        let mut arg3 = q_val as *mut std::ffi::c_void;
        let mut arg4 = n_val as *mut std::ffi::c_void;
        let mut arg5 = dim_val as *mut std::ffi::c_void;
        let mut args = [&mut arg0, &mut arg1, &mut arg2, &mut arg3, &mut arg4, &mut arg5];
        // Grid: Q blocks (one per query), Block: 256 threads
        let r = cuLaunchKernel(func, q as u32, 1, 1, threads, 1, 1, 0,
            std::ptr::null_mut(), args.as_mut_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] batch_cosine_dist error {}: {}", r, msg);
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
            return None;
        }
        cuCtxSynchronize();
        let mut dists = vec![0.0f32; q * n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, out_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
        Some(dists)
    }
}


/// GPU-accelerated kNN graph construction (CAGRA build kernel).
/// Uploads all vectors to GPU, then computes batch distances for each
/// query against the full dataset. Processes Q queries at a time to
/// stay within VRAM. Returns a flat kNN graph: n × k_init indices.
///
/// This is the CAGRA Phase 1 build — 2.2-27x faster than CPU HNSW.
pub fn gpu_build_knn_graph(vectors_i8: &[i8], n: usize, dim: usize, k_init: usize) -> Option<Vec<Vec<usize>>> {
    if !gpu_ensure_kernel("batch_cosine_dist") { return None; }
    let _guard = ensure_ctx_locked()?;
    unsafe { _gpu_build_knn_graph_impl(vectors_i8, n, dim, k_init) }
}

unsafe fn _gpu_build_knn_graph_impl(vectors_i8: &[i8], n: usize, dim: usize, k_init: usize) -> Option<Vec<Vec<usize>>> {
    // Upload all vectors to GPU once
    let v_bytes = n * dim;
    let mut d_v = 0u64;
    if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { return None; }
    cuMemcpyHtoD_v2(d_v, vectors_i8.as_ptr() as *const std::ffi::c_void, v_bytes);

    let func = GPU_STATE.get()?.get_kernel("batch_cosine_dist")?;

    // Adaptive batch size: fit output in available VRAM.
    // Output = batch_q * n * 4 bytes. Cap at 500MB to leave headroom.
    let max_output_bytes = 500 * 1024 * 1024; // 500MB
    let batch_q = (max_output_bytes / (n * 4).max(1)).max(1).min(n).min(512);
    let mut knn_graph: Vec<Vec<usize>> = vec![vec![0usize; k_init]; n];
    let mut q_buf: Vec<i8> = vec![0i8; batch_q * dim];
    let mut d_d = 0u64;
    if cuMemAlloc_v2(&mut d_d, batch_q * n * 4) != 0 {
        cuMemFree_v2(d_v);
        return None;
    }
    let threads = 256u32;

    eprintln!("[GPU] Building kNN graph: {n} vectors, dim={dim}, k_init={k_init}, batch={batch_q}");

    for batch_start in (0..n).step_by(batch_q) {
        let batch_end = (batch_start + batch_q).min(n);
        let q_count = batch_end - batch_start;

        // Copy query batch to host buffer
        for i in 0..q_count {
            let src = (batch_start + i) * dim;
            let dst = i * dim;
            q_buf[dst..dst + dim].copy_from_slice(&vectors_i8[src..src + dim]);
        }

        // Upload queries
        let q_bytes = q_count * dim;
        let mut d_q = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_bytes) != 0 { continue; }
        cuMemcpyHtoD_v2(d_q, q_buf.as_ptr() as *const std::ffi::c_void, q_bytes);

        // Launch batch_cosine_dist: Q queries × N vectors
        let mut arg0 = d_q as *mut std::ffi::c_void;
        let mut arg1 = d_v as *mut std::ffi::c_void;
        let mut arg2 = d_d as *mut std::ffi::c_void;
        let mut arg3 = q_count as *mut std::ffi::c_void;
        let mut arg4 = n as *mut std::ffi::c_void;
        let mut arg5 = dim as *mut std::ffi::c_void;
        let mut args = [&mut arg0, &mut arg1, &mut arg2, &mut arg3, &mut arg4, &mut arg5];

        let r = cuLaunchKernel(func, q_count as u32, 1, 1, threads, 1, 1, 0,
            std::ptr::null_mut(), args.as_mut_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 { cuMemFree_v2(d_q); continue; }
        cuCtxSynchronize();

        // Read back distances and find top-k per query
        let mut dists = vec![0.0f32; q_count * n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, q_count * n * 4);
        cuMemFree_v2(d_q);

        // For each query in batch, find k_init nearest (excluding self)
        for qi in 0..q_count {
            let global_q = batch_start + qi;
            let row = &dists[qi * n..(qi + 1) * n];
            // Find k_init smallest distances (simple selection)
            let mut candidates: Vec<(f32, usize)> = row.iter().enumerate()
                .filter(|&(i, _)| i != global_q)
                .map(|(i, &d)| (d, i))
                .collect();
            // Partial sort — just find top-k smallest
            candidates.select_nth_unstable_by(k_init, |a, b| a.0.partial_cmp(&b.0).unwrap());
            candidates.truncate(k_init);
            knn_graph[global_q] = candidates.into_iter().map(|(_, i)| i).collect();
        }
    }

    cuMemFree_v2(d_v);
    cuMemFree_v2(d_d);
    eprintln!("[GPU] kNN graph built: {n} × {k_init}");
    Some(knn_graph)
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

// ── GPU-resident index for CAGRA-style search ────────────────────────────
// Vectors + graph live on GPU. Repeated searches avoid re-upload.

/// GPU-resident index. Holds device pointers for vectors and graph.
/// Created once via `gpu_build_index`, then used for all searches.
pub struct GpuIndex {
    pub d_vectors: u64,
    pub d_graph: u64,
    pub n: usize,
    pub dim: usize,
    pub degree: usize,
}

impl GpuIndex {
    /// Free GPU memory.
    pub fn free(&self) {
        unsafe {
            cuMemFree_v2(self.d_vectors);
            cuMemFree_v2(self.d_graph);
        }
    }
}

/// Upload INT8 vectors + flat CSR graph to GPU. Returns a GpuIndex handle.
/// Call once after building the graph. All subsequent searches use the cached GPU data.
/// Build GPU index. Mode-aware memory allocation:
/// - Turbo: vectors in VRAM (fast access, cuMemAlloc + copy)
/// - Hybrid: vectors in unified memory (GPU reads from RAM via page faults)
pub fn gpu_build_index(vectors_i8: &[i8], graph_flat: &[i32], n: usize, dim: usize, degree: usize) -> Option<GpuIndex> {
    if !gpu_ensure_kernel("batch_cosine_dist") { return None; }
    let _guard = ensure_ctx_locked()?;
    let mode = gpu_get_mode();
    unsafe {
        let v_bytes = n * dim;
        let g_bytes = n * degree * 4;
        let mut d_v = 0u64;
        let mut d_g = 0u64;
        match mode {
            ComputeMode::Turbo => {
                // Turbo: vectors IN VRAM for fast kernel access.
                // Full copy once, then kernel reads at VRAM speed (~1TB/s).
                if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { return None; }
                cuMemcpyHtoD_v2(d_v, vectors_i8.as_ptr() as *const std::ffi::c_void, v_bytes);
                eprintln!("[GPU] Turbo: vectors in VRAM ({:.0} MB)", v_bytes as f64 / 1024.0 / 1024.0);
            }
            _ => {
                // Hybrid/CPU: unified memory — GPU reads from RAM.
                // VUGVA-style: CUDA page migrator pulls hot pages on demand.
                if cuMemAllocManaged(&mut d_v, v_bytes, CU_MEM_ATTACH_GLOBAL) != 0 {
                    // Fallback: regular alloc + copy
                    if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { return None; }
                    cuMemcpyHtoD_v2(d_v, vectors_i8.as_ptr() as *const std::ffi::c_void, v_bytes);
                    eprintln!("[GPU] Hybrid: vectors in VRAM (fallback, unified alloc failed)");
                } else {
                    std::ptr::copy_nonoverlapping(vectors_i8.as_ptr(), d_v as *mut i8, v_bytes);
                    eprintln!("[GPU] Hybrid: vectors in unified memory ({:.0} MB, GPU reads from RAM)", v_bytes as f64 / 1024.0 / 1024.0);
                }
            }
        }
        // Graph always in VRAM (small: degree × 4 bytes per node)
        if cuMemAlloc_v2(&mut d_g, g_bytes) != 0 { return None; }
        cuMemcpyHtoD_v2(d_g, graph_flat.as_ptr() as *const std::ffi::c_void, g_bytes);
        eprintln!("[GPU] Index: {} vecs × {}d, degree={}", n, dim, degree);
        Some(GpuIndex { d_vectors: d_v, d_graph: d_g, n, dim, degree })
    }
}

/// CAGRA-style GPU search: batch of INT8 queries → top-k results.
/// The entire graph traversal runs on GPU — no CPU graph walks.
/// `entry_node` is the starting point for graph traversal (0 = default random).
pub fn gpu_search(
    index: &GpuIndex,
    queries_i8: &[i8],
    num_queries: usize,
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry_node: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    let func = GPU_STATE.get()?.get_kernel("cagra_search")?;
    let _guard = ensure_ctx_locked()?;
    let dim = index.dim;
    let n = index.n;
    let degree = index.degree;
    unsafe {
        // Upload queries
        let q_bytes = num_queries * dim;
        let mut d_q = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_bytes) != 0 { return None; }
        cuMemcpyHtoD_v2(d_q, queries_i8.as_ptr() as *const std::ffi::c_void, q_bytes);

        // Allocate output
        let out_count = num_queries * k;
        let mut d_idx = 0u64;
        let mut d_dist = 0u64;
        if cuMemAlloc_v2(&mut d_idx, out_count * 4) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_dist, out_count * 4) != 0 { return None; }

        // Kernel args: vectors, graph, queries, out_idx, out_dist,
        //   N, dim, degree, k, itopk, max_iters, num_queries
        let mut arg0 = index.d_vectors as *mut std::ffi::c_void;
        let mut arg1 = index.d_graph as *mut std::ffi::c_void;
        let mut arg2 = d_q as *mut std::ffi::c_void;
        let mut arg3 = d_idx as *mut std::ffi::c_void;
        let mut arg4 = d_dist as *mut std::ffi::c_void;
        let mut arg5 = n as *mut std::ffi::c_void;
        let mut arg6 = dim as *mut std::ffi::c_void;
        let mut arg7 = degree as *mut std::ffi::c_void;
        let mut arg8 = k as *mut std::ffi::c_void;
        let mut arg9 = itopk as *mut std::ffi::c_void;
        let mut arg10 = max_iters as *mut std::ffi::c_void;
        let mut arg11 = entry_node as *mut std::ffi::c_void;
        let mut arg12 = num_queries as *mut std::ffi::c_void;
        let mut args = [
            &mut arg0, &mut arg1, &mut arg2, &mut arg3, &mut arg4,
            &mut arg5, &mut arg6, &mut arg7, &mut arg8, &mut arg9,
            &mut arg10, &mut arg11, &mut arg12,
        ];

        let threads = 256u32;
        // Shared memory: topk_dot + topk_idx + cand_idx[8*degree] + cand_dot[8*degree]
        let smem = ((2 * itopk + 16 * degree) * 4) as u32;

        let r = cuLaunchKernel(func,
            num_queries as u32, 1, 1,   // Grid: one block per query
            threads, 1, 1,               // Block: 256 threads
            smem,                        // Shared memory
            std::ptr::null_mut(),
            args.as_mut_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] cagra_search error {}: {}", r, msg);
            cuMemFree_v2(d_q); cuMemFree_v2(d_idx); cuMemFree_v2(d_dist);
            return None;
        }
        cuCtxSynchronize();

        // Read results
        let mut indices = vec![0i32; out_count];
        let mut distances = vec![0.0f32; out_count];
        cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, d_idx, out_count * 4);
        cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, d_dist, out_count * 4);

        cuMemFree_v2(d_q);
        cuMemFree_v2(d_idx);
        cuMemFree_v2(d_dist);

        Some((indices, distances))
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
            ensure_ctx();
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

    #[test]
    fn test_batch_cosine_dist() {
        if !gpu_init() { return; }
        // 2 queries × 3 vectors, dim=3
        let queries: Vec<i8> = vec![
            127, 0, 0,
            0, 127, 0,
        ];
        let vectors: Vec<i8> = vec![
            127, 0, 0,
            0, 127, 0,
            0, 0, 127,
        ];
        let dists = gpu_batch_cosine_dist(&queries, &vectors, 2, 3, 3).expect("batch_cosine_dist failed");
        println!("[GPU] batch_cosine_dist: {:?}", dists);
        assert!(dists[0] < 0.01, "q0-v0 should be ~0, got {}", dists[0]);
        assert!(dists[1] > 0.99, "q0-v1 should be ~1, got {}", dists[1]);
        assert!(dists[2] > 0.99, "q0-v2 should be ~1, got {}", dists[2]);
        assert!(dists[3] > 0.99, "q1-v0 should be ~1, got {}", dists[3]);
        assert!(dists[4] < 0.01, "q1-v1 should be ~0, got {}", dists[4]);
        assert!(dists[5] > 0.99, "q1-v2 should be ~1, got {}", dists[5]);
        println!("[GPU] batch_cosine_dist VERIFIED!");
    }

    #[test]
    fn test_cagra_gpu_search() {
        if !gpu_init() { return; }
        // 6 vectors of dim=3, degree=2 graph
        // Graph: 0→{1,2}, 1→{0,3}, 2→{0,4}, 3→{1,5}, 4→{2,5}, 5→{3,4}
        let vectors: Vec<i8> = vec![
            127, 0, 0,   // v0
            100, 50, 0,  // v1 (close to v0)
            100, 0, 50,  // v2 (close to v0)
            0, 127, 0,   // v3
            0, 0, 127,   // v4
            0, 80, 80,   // v5
        ];
        let graph: Vec<i32> = vec![
            1, 2,  // v0→{v1,v2}
            0, 3,  // v1→{v0,v3}
            0, 4,  // v2→{v0,v4}
            1, 5,  // v3→{v1,v5}
            2, 5,  // v4→{v2,v5}
            3, 4,  // v5→{v3,v4}
        ];
        let idx = gpu_build_index(&vectors, &graph, 6, 3, 2).expect("gpu_build_index failed");
        // Query: same as v0, should find v0 as nearest
        let queries: Vec<i8> = vec![127, 0, 0];
        let (indices, distances) = gpu_search(&idx, &queries, 1, 3, 8, 20, 0).expect("gpu_search failed");
        println!("[GPU] cagra_search: indices={:?} dists={:?}", indices, distances);
        // v0 should be the closest (dist ~0)
        assert!(distances[0] < 0.1, "v0 should be closest, got dist={}", distances[0]);
        assert_eq!(indices[0], 0, "first result should be v0");
        idx.free();
        println!("[GPU] cagra_search VERIFIED — full GPU graph traversal works!");
    }
}
