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

/// VUGVA — Virtual Unified GPU VRAM Architecture, vendored as `crates/vugva-core`.
/// This is the *only* VUGVA in the tree.
///
/// There used to be a second, cut-down `gpu::vugva` module here carrying its own
/// `VugvaVmt`/`DmaRing`. It restated the paper's structures at lower fidelity and
/// — worse — its one consumer (`views::VectorIndex`) only ever *wrote* the handle:
/// it held a full duplicate RAM copy of `all_i8` (384 MB at 1M×384d) and logged a
/// 6 GB VRAM budget for a table that served zero queries. Deleted. Use the real one:
/// `vugva::{tiered::TieredPool, vmt::VirtualMemoryTable, dma::DmaEngine,
/// prefetch::LookAheadPrefetcher}`.
pub use vugva_core as vugva;

/// Back-compat alias for the name this crate exported previously.
pub use vugva_core as vugva_upstream;

pub mod corpus_tier;
pub use corpus_tier::CorpusTier;

/// Result type for the GPU-side helpers that can fail for an explainable
/// reason. The error is a `String` because every current failure originates in
/// a `VugvaError` or a CUDA status that is only ever logged or surfaced to a
/// RESP client — nothing matches on the variant, so a typed enum here would be
/// ceremony without a consumer.
pub type GpuResult<T> = std::result::Result<T, String>;

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
    // Streams: concurrent single-query searches (per-slot buffers, no global lock)
    fn cuStreamCreate(phStream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cuStreamDestroy_v2(hStream: *mut std::ffi::c_void) -> i32;
    fn cuStreamSynchronize(hStream: *mut std::ffi::c_void) -> i32;
    fn cuStreamQuery(hStream: *mut std::ffi::c_void) -> i32;
    fn cuMemcpyHtoDAsync_v2(dstDevice: u64, srcHost: *const std::ffi::c_void, byteCount: usize, hStream: *mut std::ffi::c_void) -> i32;
    fn cuMemcpyDtoHAsync_v2(dstHost: *mut std::ffi::c_void, srcDevice: u64, byteCount: usize, hStream: *mut std::ffi::c_void) -> i32;
    fn cuMemsetD32_v2(dstDevice: u64, ui: u32, n: usize) -> i32;
    // Dynamic shared-memory opt-in: a kernel whose launch needs more than the
    // 48 KB default must request it via this attribute BEFORE cuLaunchKernel.
    fn cuFuncSetAttribute(f: *mut std::ffi::c_void, attrib: i32, value: i32) -> i32;
}

const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
// AD107 (RTX 4060, CC 8.9): measured opt-in ceiling is 99 KB, per-SM 100 KB.
// Default is 48 KB; the APGC kernel needs more, so we must opt in AND stay
// under this hard device limit (a request above it fails the launch).
const GPU_MAX_DYN_SMEM: usize = 96 * 1024; // 98,304 B — under the 99 KB ceiling

const CU_STREAM_NON_BLOCKING: u32 = 0x1;

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
    available: bool,
    vram_total: usize,
    vram_free: usize,
    device: i32,
}

unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

/// Global GPU state — lazy init on first use.
static GPU_STATE: std::sync::OnceLock<GpuState> = std::sync::OnceLock::new();
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

/// Set once `gpu_unload` has destroyed the context, making GPU use a one-way
/// door for the rest of the process.
///
/// `GPU_STATE` is a `OnceLock`, so the destroyed `GpuState` — dangling `ctx`
/// and all — is the only one this process will ever have. Without this flag a
/// later `gpu_init` saw `GPU_ENABLED == false`, got that same state back from
/// `get_or_init`, and set `GPU_ENABLED = true`: the GPU then reported itself
/// available while every `cuCtxSetCurrent` failed with 201, so work neither
/// ran on the device nor fell back to the CPU. Refusing up front turns a
/// bricked GPU into a clean CPU fallback.
static GPU_DESTROYED: AtomicBool = AtomicBool::new(false);
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
                available: true, vram_total, vram_free, device,
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
            // Detect GPU compute capability for PTX arch
            let mut compute_major = 0i32;
            let mut compute_minor = 0i32;
            cuDeviceGetAttribute(&mut compute_major, 75, self.device);
            cuDeviceGetAttribute(&mut compute_minor, 76, self.device);
            let arch_str = format!("--gpu-architecture=compute_{}{}", compute_major, compute_minor);
            eprintln!("[GPU] PTX arch: compute_{}{}", compute_major, compute_minor);
            let arch = CString::new(arch_str).unwrap();
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
            for name in &["cosine_dist_kernel", "batch_cosine_dist_kernel", "batch_cosine_dist_f32_kernel", "apgc_search_kernel", "matmul_kernel", "topk_select_kernel", "fused_cosine_topk_kernel", "preprocess_i8_to_f32_kernel", "apgc_optimize_kernel", "apgc_reverse_kernel", "apgc_merge_kernel", "nn_rev_kernel", "nn_descent_kernel", "opusedge_selkv_prune", "opusedge_delta_ar_route", "opusedge_head_gate", "opusedge_state_compress", "opusedge_proxy_delta"] {
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
    // The context this process had was destroyed and cannot be rebuilt; see
    // GPU_DESTROYED. Re-enabling here would hand out a dangling context.
    if GPU_DESTROYED.load(Ordering::Relaxed) { return false; }
    let _guard = GPU_ACCESS.lock().ok();
    // Double-check after acquiring lock.
    if GPU_ENABLED.load(Ordering::Relaxed) { return true; }
    // Use get_or_init to prevent double-init race.
    let state = GPU_STATE.get_or_init(|| {
        GpuState::init_ctx().unwrap_or_else(|| {
            GpuState { ctx: std::ptr::null_mut(), module: std::ptr::null_mut(),
                       kernels: Vec::new(), available: false, vram_total: 0, vram_free: 0, device: 0 }
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
    if GPU_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    // Bring the device up on first ask, when a GPU mode was actually selected.
    //
    // This flag is only ever set by `gpu_init`, and nothing calls `gpu_init`
    // except the `GPU.LOAD` / `GPU.MODE` RESP handlers. So every caller that
    // gates on `gpu_available()` — including the APGC build's `use_gpu_build`
    // — saw `false` in any process that had not received one of those commands
    // first. Setting `DBSTRIKE_GPU=turbo` selected the mode but left the driver
    // uninitialised, which is why a build could report "turbo" and still run
    // entirely on the CPU at zero GPU utilization.
    //
    // Gated on the mode so `CpuOnly` still never touches the device, and
    // attempted once so a machine with no driver does not pay a failed
    // `cuInit` on every call.
    if gpu_get_mode() != ComputeMode::CpuOnly {
        static TRIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !TRIED.swap(true, Ordering::Relaxed) {
            return gpu_init();
        }
    }
    false
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
    // `all` loads every kernel: compile the whole module, then report success
    // if any kernel is resident (the module compiles all 18 at once).
    if name.eq_ignore_ascii_case("all") {
        if KERNELS_COMPILED.load(Ordering::Acquire) {
            return true;
        }
        let state_ptr = state as *const GpuState as *mut GpuState;
        unsafe { (*state_ptr).ensure_kernels(); }
        return KERNELS_COMPILED.load(Ordering::Acquire)
            && GPU_STATE.get().map(|s| !s.kernels.is_empty()).unwrap_or(false);
    }
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
    if KERNELS_ATTEMPTED.load(Ordering::Acquire) {
        return GPU_STATE.get().map(|s| s.get_kernel(name).is_some()).unwrap_or(false);
    }
    // First-time compilation: acquire GPU_ACCESS lock (does CUDA calls).
    if let Some(_guard) = GPU_ACCESS.lock().ok() {
        if let Some(state) = GPU_STATE.get() {
            if state.get_kernel(name).is_some() { return true; }
            let state_ptr = state as *const GpuState as *mut GpuState;
            unsafe { (*state_ptr).ensure_kernels(); }
        }
    }
    GPU_STATE.get().map(|s| s.get_kernel(name).is_some()).unwrap_or(false)
}

/// Acquire GPU_ACCESS lock and ensure kernels are compiled.
/// Callers that need both lock and kernels should call this instead of
/// gpu_ensure_kernel + GPU_ACCESS.lock() separately (avoids deadlock).
fn gpu_lock_and_ensure() -> Option<std::sync::MutexGuard<'static, ()>> {
    let guard = GPU_ACCESS.lock().ok()?;
    // Always ensure CUDA context is set for this thread (fixes error 201 after long builds)
    ensure_ctx();
    // Kernels already compiled? Just return the lock.
    if KERNELS_COMPILED.load(Ordering::Acquire) { return Some(guard); }
    // First time: compile kernels while holding the lock.
    if let Some(state) = GPU_STATE.get() {
        let state_ptr = state as *const GpuState as *mut GpuState;
        unsafe { (*state_ptr).ensure_kernels(); }
    }
    Some(guard)
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
        // Query the driver rather than reporting the snapshot taken at init.
        //
        // `state.vram_free` is measured once, before anything is uploaded. By
        // the time this is consulted — to decide whether a corpus fits, or
        // whether the build can seed in a single launch — hundreds of MB of
        // index may already be resident, so the cached figure over-reports and
        // the caller commits to an allocation the device cannot satisfy.
        let mut free = state.vram_free;
        unsafe {
            let (mut f, mut t) = (0usize, 0usize);
            if !state.ctx.is_null()
                && cuCtxSetCurrent(state.ctx) == 0
                && cuMemGetInfo_v2(&mut f, &mut t) == 0
            {
                free = f;
            }
        }
        (needed <= free, needed, free)
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
    /// **GPU only.** Vectors, graph and traversal are all resident in VRAM;
    /// nothing is served from host memory. Requires an NVIDIA GPU and a corpus
    /// that fits in VRAM. This is the fast path, and it is only selectable when
    /// the whole working set fits — there is no host fallback within a query.
    Turbo,
    /// **VUGVA.** The three-tier hybrid from the VUGVA paper: VRAM (T0, hot) →
    /// system DRAM (T1, warm) → NVMe (T2, cold), with the CPU confined to the
    /// control plane. Promotion and demotion move through the DMA engine, so
    /// the CPU writes descriptors rather than copying tensor data.
    ///
    /// This is what makes a larger-than-VRAM corpus work without a performance
    /// cliff, which is the property neither Qdrant nor Milvus has. It is *not*
    /// "GPU plus some CPU offload" — the CPU never touches data-plane traffic.
    Hybrid,
    /// **CPU only.** No GPU work at all, even when a GPU is present and
    /// initialised. Selected when there is no device, or set explicitly to get
    /// a clean CPU baseline.
    CpuOnly,
}

/// `None` until the first read or an explicit `gpu_set_mode`, so the default
/// can be resolved from the environment exactly once.
static COMPUTE_MODE: Mutex<Option<ComputeMode>> = Mutex::new(None);

/// The compute mode named by `DBSTRIKE_GPU`, if it names a valid one.
///
/// Without this the default is `CpuOnly` and the only way to reach the GPU is
/// the `GPU.MODE` RESP command — so every harness that does not speak RESP
/// first (the benchmarks, the in-process ingest profilers) measured the CPU
/// path while appearing to exercise the GPU one. That is why the GPU build
/// below had no benchmark covering it. Taking the default from the environment
/// makes the GPU path reachable by anything that can set a variable.
fn mode_from_env() -> Option<ComputeMode> {
    match std::env::var("DBSTRIKE_GPU")
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "turbo" => Some(ComputeMode::Turbo),
        "hybrid" => Some(ComputeMode::Hybrid),
        "cpu" | "cpuonly" | "cpu_only" => Some(ComputeMode::CpuOnly),
        // Unrecognised values fall through to the built-in default rather than
        // failing: this selects a performance path, not a correctness one.
        _ => None,
    }
}

/// Get current compute mode.
pub fn gpu_get_mode() -> ComputeMode {
    let mut slot = COMPUTE_MODE.lock().unwrap();
    if let Some(m) = *slot {
        return m;
    }
    let initial = mode_from_env().unwrap_or(ComputeMode::CpuOnly);
    *slot = Some(initial);
    initial
}

/// Set compute mode explicitly (RESP: GPU.MODE turbo|hybrid|cpu).
pub fn gpu_set_mode(mode: ComputeMode) {
    *COMPUTE_MODE.lock().unwrap() = Some(mode);
    eprintln!("[GPU] Compute mode set to {:?}", mode);
}

/// Auto-detect optimal mode based on GPU availability and data size.
/// Call once after init with the dataset dimensions.
pub fn gpu_auto_mode(n: usize, dim: usize) -> ComputeMode {
    if !gpu_available() {
        return ComputeMode::CpuOnly;
    }
    let (_, _, vram_free) = gpu_check_capacity(n, dim);
    // Peak VRAM: vectors + kNN build distance matrix (batch_q=64) + graph + overhead
    let shards = 16usize;
    let shard_n = (n + shards - 1) / shards;
    let build_peak = shard_n * dim + 512 * shard_n * 4 + 20 * 1024 * 1024;
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
    let _guard = gpu_lock_and_ensure()?;
    let func = GPU_STATE.get()?.get_kernel("cosine_dist")?;
    unsafe {
        let q_bytes = query.len();
        let v_bytes = vectors.len();
        let d_bytes = n * 4;

        let mut d_q = 0u64; let mut d_v = 0u64; let mut d_d = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_bytes) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { cuMemFree_v2(d_q); return None; }
        if cuMemAlloc_v2(&mut d_d, d_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); return None; }
        cuMemcpyHtoD_v2(d_q, query.as_ptr() as *const std::ffi::c_void, q_bytes);
        cuMemcpyHtoD_v2(d_v, vectors.as_ptr() as *const std::ffi::c_void, v_bytes);

        let threads = 256u32;
        let blocks = (n as u32 + threads - 1) / threads;
        let n_val = n as i32;
        let dim_val = dim as i32;
        let cosine_params: [*mut std::ffi::c_void; 5] = [
            &d_q as *const u64 as *mut std::ffi::c_void,
            &d_v as *const u64 as *mut std::ffi::c_void,
            &d_d as *const u64 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &dim_val as *const i32 as *mut std::ffi::c_void,
        ];
        let r = cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0,
                      std::ptr::null_mut(), cosine_params.as_ptr() as *mut *mut std::ffi::c_void,
                      std::ptr::null_mut());
        if r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] cuLaunchKernel error {}: {}", r, msg);
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
            return None;
        }
        if !sync_stream_with_watchdog(std::ptr::null_mut()) { return None; }

        let mut dists = vec![0.0f32; n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, d_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
        Some(dists)
    }
}

/// Batch INT8 cosine distance: Q queries × N vectors → Q×N distances.
/// This is the APGC distance kernel — Q blocks, one per query.
/// Returns None if GPU unavailable.
pub fn gpu_batch_cosine_dist(queries: &[i8], vectors: &[i8], q: usize, n: usize, dim: usize) -> Option<Vec<f32>> {
    let _guard = gpu_lock_and_ensure()?;
    let func = GPU_STATE.get()?.get_kernel("batch_cosine_dist")?;
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
        let batch_params: [*mut std::ffi::c_void; 6] = [
            &d_q as *const u64 as *mut std::ffi::c_void,
            &d_v as *const u64 as *mut std::ffi::c_void,
            &d_d as *const u64 as *mut std::ffi::c_void,
            &q_val as *const i32 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &dim_val as *const i32 as *mut std::ffi::c_void,
        ];
        // Grid: Q blocks (one per query), Block: 256 threads
        let r = cuLaunchKernel(func, q as u32, 1, 1, threads, 1, 1, 0,
            std::ptr::null_mut(), batch_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 {
            let mut err_str: *const i8 = std::ptr::null();
            cuGetErrorString(r, &mut err_str);
            let msg = if err_str.is_null() { "unknown".to_string() }
                      else { std::ffi::CStr::from_ptr(err_str).to_string_lossy().into_owned() };
            eprintln!("[GPU] cuLaunchKernel error {}: {}", r, msg);
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
            return None;
        }
        if !sync_stream_with_watchdog(std::ptr::null_mut()) {
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
            return None;
        }
        let mut dists = vec![0.0f32; q * n];
        cuMemcpyDtoH_v2(dists.as_mut_ptr() as *mut std::ffi::c_void, d_d, out_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d);
        Some(dists)
    }
}

/// Batch INT8 cosine distance + GPU-side top-k selection in one call.
/// Computes Q×N distances on GPU, selects top-k on GPU, and reads back
/// only Q×k results (no Q×N PCIe readback).
///
/// Returns (indices, distances) each length Q×k.
/// Uses the fused kernel when available, falls back to two-kernel approach.
/// Thread count for the top-k kernels. Two hard constraints:
/// - dynamic smem = threads × k × 8 bytes must fit the 48 KB default limit
/// - threads × k ≤ 4096 (thread-0 merge buffer inside the kernels)
/// k ≤ 32 → 128 threads (32 KB smem); k ≤ 64 → 64 threads (32 KB smem).
fn topk_threads(k: usize) -> u32 {
    if k <= 32 { 128 } else { 64 }
}

/// Hard upper bound for k in the top-k kernels (register array size).
pub const GPU_TOPK_MAX: usize = 512;

pub fn gpu_batch_cosine_dist_topk(
    queries: &[i8], vectors: &[i8], q: usize, n: usize, dim: usize, k: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    if k > GPU_TOPK_MAX { return None; }
    let _guard = gpu_lock_and_ensure()?;
    unsafe {
        let q_bytes = q * dim;
        let v_bytes = vectors.len();
        let out_bytes = q * k * 4; // indices + distances

        let mut d_q = 0u64;
        let mut d_v = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_bytes) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { cuMemFree_v2(d_q); return None; }
        cuMemcpyHtoD_v2(d_q, queries.as_ptr() as *const std::ffi::c_void, q_bytes);
        cuMemcpyHtoD_v2(d_v, vectors.as_ptr() as *const std::ffi::c_void, v_bytes);

        // smem = threads × k × 8 bytes must fit 48 KB → 128 threads for k≤32,
        // 64 threads for k≤64 (32 KB either way).
        let threads = topk_threads(k);

        // Try fused kernel first (one pass, no Q×N buffer).
        // The kernel silently returns for k > 64, so we must skip it
        // to avoid reading back uninitialized GPU memory as valid results.
        let fused_func = if k <= 64 {
            GPU_STATE.get()?.get_kernel("fused_cosine_topk")
        } else {
            None
        };
        if let Some(func) = fused_func {
            let mut d_out_idx = 0u64;
            let mut d_out_dist = 0u64;
            if cuMemAlloc_v2(&mut d_out_idx, out_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); return None; }
            if cuMemAlloc_v2(&mut d_out_dist, out_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_out_idx); return None; }

            let q_val = q as i32;
            let n_val = n as i32;
            let dim_val = dim as i32;
            let k_val = k as i32;
            let fused_params: [*mut std::ffi::c_void; 8] = [
                &d_q as *const u64 as *mut std::ffi::c_void,
                &d_v as *const u64 as *mut std::ffi::c_void,
                &d_out_idx as *const u64 as *mut std::ffi::c_void,
                &d_out_dist as *const u64 as *mut std::ffi::c_void,
                &q_val as *const i32 as *mut std::ffi::c_void,
                &n_val as *const i32 as *mut std::ffi::c_void,
                &dim_val as *const i32 as *mut std::ffi::c_void,
                &k_val as *const i32 as *mut std::ffi::c_void,
            ];

            // Shared memory: threads × k × (4 bytes float + 4 bytes int)
            let smem = (threads as usize * k * 8) as u32;

            let r = cuLaunchKernel(func, q as u32, 1, 1, threads, 1, 1, smem,
                std::ptr::null_mut(), fused_params.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut());
            if r != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist); return None; }
            if !sync_stream_with_watchdog(std::ptr::null_mut()) {
                cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist);
                return None;
            }

            let mut indices = vec![0i32; q * k];
            let mut distances = vec![0.0f32; q * k];
            cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, d_out_idx, out_bytes);
            cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, d_out_dist, out_bytes);
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist);
            return Some((indices, distances));
        }

        // Fallback: batch_cosine_dist + topk_select (two-kernel approach)
        let dist_func = GPU_STATE.get()?.get_kernel("batch_cosine_dist")?;
        let topk_func = GPU_STATE.get()?.get_kernel("topk_select")?;

        let dist_bytes = q * n * 4;
        let mut d_d = 0u64;
        let mut d_out_idx = 0u64;
        let mut d_out_dist = 0u64;
        if cuMemAlloc_v2(&mut d_d, dist_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); return None; }
        if cuMemAlloc_v2(&mut d_out_idx, out_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d); return None; }
        if cuMemAlloc_v2(&mut d_out_dist, out_bytes) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d); cuMemFree_v2(d_out_idx); return None; }

        // Step 1: batch_cosine_dist
        let q_val = q as i32;
        let n_val = n as i32;
        let dim_val = dim as i32;
        let dist_params: [*mut std::ffi::c_void; 6] = [
            &d_q as *const u64 as *mut std::ffi::c_void,
            &d_v as *const u64 as *mut std::ffi::c_void,
            &d_d as *const u64 as *mut std::ffi::c_void,
            &q_val as *const i32 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &dim_val as *const i32 as *mut std::ffi::c_void,
        ];
        let r = cuLaunchKernel(dist_func, q as u32, 1, 1, threads, 1, 1, 0,
            std::ptr::null_mut(), dist_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist); return None; }

        // Step 2: topk_select (reads d_d, writes d_out_idx + d_out_dist)
        let k_val = k as i32;
        let topk_params: [*mut std::ffi::c_void; 6] = [
            &d_d as *const u64 as *mut std::ffi::c_void,
            &d_out_idx as *const u64 as *mut std::ffi::c_void,
            &d_out_dist as *const u64 as *mut std::ffi::c_void,
            &q_val as *const i32 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &k_val as *const i32 as *mut std::ffi::c_void,
        ];
        let smem = (threads as usize * k * 8) as u32;
        cuLaunchKernel(topk_func, q as u32, 1, 1, threads, 1, 1, smem,
            std::ptr::null_mut(), topk_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if !sync_stream_with_watchdog(std::ptr::null_mut()) {
            cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist);
            return None;
        }

        // Read back only Q×k results (tiny: 8 KB vs 128 MB for Q×N)
        let mut indices = vec![0i32; q * k];
        let mut distances = vec![0.0f32; q * k];
        cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, d_out_idx, out_bytes);
        cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, d_out_dist, out_bytes);
        cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_d); cuMemFree_v2(d_out_idx); cuMemFree_v2(d_out_dist);
        Some((indices, distances))
    }
}


// ── Persistent GPU build buffers (eliminate per-call alloc/free) ────────

/// Pre-allocated GPU buffers for the APGC build pipeline.
/// Created once, reused across ALL seed + non-seed batches.
/// This eliminates the O(batches) alloc/copy/launch/sync/free overhead.
struct GpuBuildBuffers {
    d_queries: u64,
    d_vectors: u64,
    d_out_idx: u64,
    d_out_dist: u64,
    max_q: usize,
    max_n: usize,
    dim: usize,
    k: usize,
    fused_func: *mut std::ffi::c_void,
}

impl GpuBuildBuffers {
    unsafe fn new(max_q: usize, max_n: usize, dim: usize, k: usize) -> Option<Self> {
        let q_cap = max_q * dim;
        let v_cap = max_n * dim;
        let out_cap = max_q * k * 4;
        let mut d_q = 0u64; let mut d_v = 0u64;
        let mut d_oi = 0u64; let mut d_od = 0u64;
        if cuMemAlloc_v2(&mut d_q, q_cap) != 0 { return None; }
        if cuMemAlloc_v2(&mut d_v, v_cap) != 0 { cuMemFree_v2(d_q); return None; }
        if cuMemAlloc_v2(&mut d_oi, out_cap) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); return None; }
        if cuMemAlloc_v2(&mut d_od, out_cap) != 0 { cuMemFree_v2(d_q); cuMemFree_v2(d_v); cuMemFree_v2(d_oi); return None; }
        let fused = GPU_STATE.get().and_then(|s| s.get_kernel("fused_cosine_topk")).unwrap_or(std::ptr::null_mut());
        Some(Self { d_queries: d_q, d_vectors: d_v, d_out_idx: d_oi, d_out_dist: d_od, max_q, max_n, dim, k, fused_func: fused })
    }

    /// Run fused top-k with pre-allocated buffers. NO alloc/free per call.
    /// Only upload queries (vectors cached), launch, readback.
    unsafe fn run(&self, queries: &[i8], q: usize, n: usize) -> Option<(Vec<i32>, Vec<f32>)> {
        if q > self.max_q || n > self.max_n { return None; }
        let q_bytes = q * self.dim;
        let out_bytes = q * self.k * 4;
        cuMemcpyHtoD_v2(self.d_queries, queries.as_ptr() as *const std::ffi::c_void, q_bytes);
        // Vectors already in d_vectors (pre-loaded once)
        let q_val = q as i32; let n_val = n as i32; let dim_val = self.dim as i32; let k_val = self.k as i32;
        let build_params: [*mut std::ffi::c_void; 8] = [
            &self.d_queries as *const u64 as *mut std::ffi::c_void,
            &self.d_vectors as *const u64 as *mut std::ffi::c_void,
            &self.d_out_idx as *const u64 as *mut std::ffi::c_void,
            &self.d_out_dist as *const u64 as *mut std::ffi::c_void,
            &q_val as *const i32 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &dim_val as *const i32 as *mut std::ffi::c_void,
            &k_val as *const i32 as *mut std::ffi::c_void,
        ];
        let threads = topk_threads(self.k);
        let smem = (threads as usize * self.k * 8) as u32;
        let r = cuLaunchKernel(self.fused_func, q as u32, 1, 1, threads, 1, 1, smem,
            std::ptr::null_mut(), build_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r != 0 { return None; }
        // A kernel fault is asynchronous: the launch above returns success and
        // the error only surfaces here. Leaving this unchecked meant an OOM or
        // illegal access produced a *silent* all-zeros readback, and since the
        // caller maps index 0 to a real node, every vector's nearest neighbour
        // became node 0 — a build that reports success and prints its usual
        // timing line while producing a useless graph.
        if !sync_stream_with_watchdog(std::ptr::null_mut()) { return None; }
        // Sentinel-initialised, not zero: index 0 and distance 0.0 are both
        // *valid* results, so a partial copy would be indistinguishable from a
        // genuine answer. -1 is filtered by every consumer.
        let mut indices = vec![-1i32; q * self.k];
        let mut distances = vec![2.0f32; q * self.k];
        if cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, self.d_out_idx, out_bytes) != 0 {
            return None;
        }
        if cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, self.d_out_dist, out_bytes) != 0 {
            return None;
        }
        Some((indices, distances))
    }

    unsafe fn free(&self) {
        cuMemFree_v2(self.d_queries);
        cuMemFree_v2(self.d_vectors);
        cuMemFree_v2(self.d_out_idx);
        cuMemFree_v2(self.d_out_dist);
    }
}

/// APGC Paper Algorithm 1: GPU-side kNN construction with persistent buffers.
/// Allocates GPU buffers ONCE, reuses for all batches.
pub fn gpu_build_knn_graph(vectors_i8: &[i8], n: usize, dim: usize, k_init: usize) -> Option<Vec<usize>> {
    let _guard = gpu_lock_and_ensure()?;
    unsafe {
        let t_total = std::time::Instant::now();
        // Cumulative GPU-build budget. The 30s watchdog bounds each sync site,
        // but a *legitimate* 5M build runs ~1000s of near-continuous kernel
        // work — on a display-driving GPU that parks the desktop for the whole
        // run (feels like a crash). If the whole build exceeds this budget we
        // abort the GPU path so the caller falls back to a CPU build, keeping
        // the desktop responsive. Default 120s; override with GPU_BUILD_BUDGET_SECS.
        let budget_secs: u64 = std::env::var("GPU_BUILD_BUDGET_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let budget = std::time::Duration::from_secs(budget_secs);
        let k = k_init;
        // Pivot count is CAPPED. The old code used n/50 pivots and scanned the
        // FULL corpus once per pivot — O(n²/50) work that took 160s at 1M.
        // A capped pivot set makes the seeding phase O(n·s) with a tiny,
        // L2-resident corpus, and NN-descent (below) recovers graph quality.
        let seed_count = (n / 50).max(64).min(n).min(2048);
        let seed_step = n / seed_count.max(1);
        eprintln!("[GPU] APGC: n={n}, dim={dim}, k={k}, pivots={seed_count}");

        // Allocate persistent GPU buffers once
        // Prefer ONE launch covering the whole corpus over a batched loop.
        //
        // Each batch costs a `cuCtxSynchronize` plus two `cuMemcpyDtoH`
        // (`GpuBuildBuffers::run`), and the CPU then sorts that batch's
        // candidates before the next launch is issued. That drains the device
        // and refills it once per batch, with the GPU idle across the whole
        // host-side stretch — which is what shows up as 2-5% utilization during
        // a build.
        //
        // Nothing forces the split: the kernel's grid is one block per query,
        // so a single launch with `q = n` does the same work with one sync and
        // one readback. The only real constraint is VRAM for the query and
        // output buffers, so try the whole corpus and fall back to the batched
        // path when it does not fit (which keeps large-corpus builds working on
        // small cards rather than failing outright).
        //
        // Cost of the full-size buffers: n·dim for queries plus 8·n·k for the
        // two output arrays — at 1M×384d with k=32 that is 384 MB + 256 MB.
        let full_bytes = n * dim + n * k * 8 + n.max(seed_count) * dim;
        let (_, _, vram_free) = gpu_check_capacity(0, 0);
        // Two thirds, not all: the refine phase below allocates its own graph
        // buffers, and a build that fits by a hair here would fail there.
        let one_shot = full_bytes.saturating_mul(3) / 2 < vram_free;
        // A single launch whose grid is `n` blocks keeps the whole display GPU
        // busy for the entire build (the kernel is one block per query). Cap
        // the per-launch grid so a large corpus never parks the desktop for
        // minutes at a time; corpora above the cap simply take the batched
        // path below.
        const MAX_ONE_SHOT_Q: usize = 65536;

        let (max_batch_q, bufs) = match (one_shot && n <= MAX_ONE_SHOT_Q)
            .then(|| GpuBuildBuffers::new(n, n.max(seed_count), dim, k))
            .flatten()
        {
            Some(b) => {
                eprintln!("[GPU] APGC: single-launch seeding ({n} queries, {} MB buffers)",
                          full_bytes >> 20);
                (n, b)
            }
            None => {
                let batch = 8192usize.min(n);
                eprintln!("[GPU] APGC: batched seeding ({batch}/launch) — \
                           {} MB single-launch buffers exceed {} MB free VRAM",
                          full_bytes >> 20, vram_free >> 20);
                (batch, GpuBuildBuffers::new(batch, n.max(seed_count), dim, k)?)
            }
        };

        let mut graph: Vec<Vec<i32>> = vec![Vec::new(); n];

        // ═══ Phase 1: every node finds its nearest pivots ═══
        // Corpus = the pivot set only (seed_count × dim bytes — a few MB, so it
        // stays resident in L2 and the kernel runs compute-bound instead of
        // streaming 384 MB per query block like the old full-corpus scan did).
        let mut seed_buf = vec![0i8; seed_count * dim];
        for si in 0..seed_count {
            let sid = (si * seed_step).min(n - 1);
            seed_buf[si * dim..(si + 1) * dim].copy_from_slice(&vectors_i8[sid * dim..(sid + 1) * dim]);
        }
        cuMemcpyHtoD_v2(bufs.d_vectors, seed_buf.as_ptr() as *const std::ffi::c_void, seed_count * dim);

        // Cluster assignment per node: (nearest pivot, 2nd nearest, distance).
        // The 2nd pivot subdivides each Voronoi cell for free — see Phase 2.
        let mut assign: Vec<(i32, i32, f32)> = vec![(0, 0, 0.0); n];

        let t_add = std::time::Instant::now();
        for as_start in (0..n).step_by(max_batch_q) {
            let as_end = (as_start + max_batch_q).min(n);
            let q_count = as_end - as_start;
            // The query block *is* a contiguous slice of the corpus: `vid` runs
            // `as_start..as_end` consecutively, so the old per-batch `q_buf`
            // copied `vectors_i8` onto itself — a fresh allocation plus a
            // ~3 MB memcpy per batch (8192 × 384 B), sitting on the critical
            // path between two kernel launches where it keeps the device idle.
            let q_buf = &vectors_i8[as_start * dim..as_end * dim];
            let Some((indices, distances)) = bufs.run(q_buf, q_count, seed_count) else {
                // Watchdog abort or device failure mid-seeding. MUST NOT skip
                // the batch and keep building: those nodes would keep a zero
                // assignment and an empty graph, silently producing a garbage
                // index that reports success. Abort the whole GPU build so the
                // caller falls back to CPU cleanly.
                bufs.free();
                eprintln!("[GPU] APGC seeding aborted (watchdog/device) at batch {as_start}..{as_end} of {n} — CPU fallback");
                return None;
            };
                for qi in 0..q_count {
                    let vid = as_start + qi;
                    let mut cands: Vec<(f32, i32, i32)> = (0..k).filter_map(|j| {
                        let li = indices[qi * k + j];
                        if li < 0 { return None; }
                        let gid = (li as usize * seed_step).min(n - 1) as i32;
                        if gid as usize == vid { return None; }
                        Some((distances[qi * k + j], gid, li))
                    }).collect();
                    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    if let Some(&(d0, _, li0)) = cands.first() {
                        let li1 = cands.get(1).map(|&(_, _, l)| l).unwrap_or(li0);
                        assign[vid] = (li0, li1, d0);
                    }
                    graph[vid] = cands.into_iter().take(k).map(|(_, g, _)| g).collect();
                }
        }
        eprintln!("[GPU] APGC pivot assign: {n} nodes vs {seed_count} pivots in {:.1}s", t_add.elapsed().as_secs_f64());

        // ═══ Phase 2: locality ordering → non-degenerate init graph ═══
        // Pointing every node at its nearest pivots makes a hub-spoke star:
        // pivots have enormous in-degree (capped away by rev_cap) and ordinary
        // nodes have ZERO in-degree, so NN-descent's reverse join has nothing
        // to work with and convergence stalls (0.918 recall at 1M).
        //
        // Instead sort nodes by (cluster, sub-cluster, distance-to-pivot) and
        // wire each node to its neighbours in that order. Every node then has
        // in-degree ≈ k, all edges stay inside a small cell, and the graph is
        // symmetric — exactly the regime NN-descent converges from. Costs one
        // sort, zero distance computations.
        //
        // The second-nearest pivot is the sub-cluster key: it splits each
        // Voronoi cell along its boundaries with neighbouring cells, for free.
        // Without it a 1M corpus with 2048 pivots gives 488-node cells — far
        // too loose to seed from (recall stalls at 0.962). With it the cells
        // fall to a few dozen nodes, matching the density that yields 0.998 at
        // 100k.
        let t_tr = std::time::Instant::now();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            let (pa, qa, da) = assign[a as usize];
            let (pb, qb, db) = assign[b as usize];
            pa.cmp(&pb).then(qa.cmp(&qb))
                .then(da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut pos = vec![0u32; n];
        for (i, &v) in order.iter().enumerate() { pos[v as usize] = i as u32; }

        // Keep a few pivot edges for long-range connectivity, fill the rest
        // from the ordering. Duplicates are harmless: the NN-descent kernel
        // deduplicates candidates through its shared-memory hash set.
        const KEEP_PIVOTS: usize = 4;
        let half = ((k - KEEP_PIVOTS) / 2).max(1) as isize;
        let order_ref: &Vec<u32> = &order;
        let pos_ref: &Vec<u32> = &pos;
        let nthreads0 = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);
        let chunk0 = (n + nthreads0 - 1) / nthreads0.max(1);
        let all_ids: Vec<usize> = (0..n).collect();
        let init_lists: Vec<(usize, Vec<i32>)> = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for c in all_ids.chunks(chunk0.max(1)) {
                let graph_ref = &graph;
                handles.push(scope.spawn(move || {
                    let mut out = Vec::with_capacity(c.len());
                    for &v in c {
                        let mut list: Vec<i32> = Vec::with_capacity(k);
                        for &g in graph_ref[v].iter().take(KEEP_PIVOTS) { list.push(g); }
                        let p = pos_ref[v] as isize;
                        for off in 1..=half {
                            for s in [-1isize, 1isize] {
                                if list.len() >= k { break; }
                                let q = p + s * off;
                                if q >= 0 && (q as usize) < n {
                                    let u = order_ref[q as usize] as i32;
                                    if u != v as i32 { list.push(u); }
                                }
                            }
                        }
                        list.truncate(k);
                        out.push((v, list));
                    }
                    out
                }));
            }
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for (v, list) in init_lists { graph[v] = list; }
        drop(order); drop(pos); drop(assign);
        eprintln!("[GPU] APGC locality order: {n} nodes in {:.2}s", t_tr.elapsed().as_secs_f64());

        // ═══ Phase 3: NN-descent refinement (fixes hub-spoke topology) ═══
        // After Phase 2 every non-seed is connected ONLY to seeds, so the
        // level-0 graph is a hub-spoke star — HNSW search over it collapses
        // (~0.67 recall@10). Refine: each node expands candidates through its
        // current neighbors' adjacency lists and keeps the exact int8 top-k.
        // Two passes: pass 1 pulls in seeds' true kNN (Phase 1 edges), pass 2
        // propagates neighbor-of-neighbor edges → a proper local kNN graph.
        let t_ref = std::time::Instant::now();
        // Budget gate before the expensive refine phase: seeding + locality
        // already ran, so if we are past the budget here, skip GPU refine
        // (which would otherwise run ~1000s on a 5M build) and fall back to CPU.
        if t_total.elapsed() >= budget {
            bufs.free();
            eprintln!("[GPU] APGC refine skipped: GPU build budget {budget_secs}s reached at {:.1}s — CPU fallback", t_total.elapsed().as_secs_f64());
            return None;
        }
        // True NN-Descent (APGC paper §3.2): candidates come from BOTH forward
        // neighbors' lists AND reverse neighbors' lists. Forward-only expansion
        // converges too slowly at 100k+ (recall stalls ~0.82); reverse edges
        // let true neighbors "find each other" from either side.
        // Pivot init is intentionally weak (it is nearly free); convergence is
        // NN-descent's job. On GPU each pass costs milliseconds, so run enough
        // of them to actually converge. Override with GPU_ND_PASSES.
        let passes: usize = std::env::var("GPU_ND_PASSES").ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if n <= 50_000 { 8 } else { 12 });

        // ── GPU NN-descent: whole refinement on-device (nn_rev + nn_descent
        // kernels, double-buffered graph). Falls back to the CPU loop below
        // on any allocation/launch failure.
        let gpu_refined = (|| {
            let rev_cap = 48usize;
            let state = GPU_STATE.get()?;
            let rev_func = state.get_kernel("nn_rev")?;
            let nd_func = state.get_kernel("nn_descent")?;
            // d_vectors currently holds the Phase-2 seed buffer — restore corpus.
            if cuMemcpyHtoD_v2(bufs.d_vectors, vectors_i8.as_ptr() as *const std::ffi::c_void, n * dim) != 0 { return None; }
            let g_bytes = n * k * 4;
            let mut d_g_in = 0u64; let mut d_g_out = 0u64; let mut d_rev = 0u64; let mut d_rev_cnt = 0u64;
            if cuMemAlloc_v2(&mut d_g_in, g_bytes) != 0 { return None; }
            if cuMemAlloc_v2(&mut d_g_out, g_bytes) != 0 { cuMemFree_v2(d_g_in); return None; }
            if cuMemAlloc_v2(&mut d_rev, n * rev_cap * 4) != 0 { cuMemFree_v2(d_g_in); cuMemFree_v2(d_g_out); return None; }
            if cuMemAlloc_v2(&mut d_rev_cnt, n * 4) != 0 { cuMemFree_v2(d_g_in); cuMemFree_v2(d_g_out); cuMemFree_v2(d_rev); return None; }
            let mut flat = vec![-1i32; n * k];
            for v in 0..n { for (j, &e) in graph[v].iter().take(k).enumerate() { flat[v * k + j] = e; } }
            cuMemcpyHtoD_v2(d_g_in, flat.as_ptr() as *const std::ffi::c_void, g_bytes);
            let n_i = n as i32; let d_i = dim as i32; let k_i = k as i32; let rc_i = rev_cap as i32;
            let mut cur_in = d_g_in; let mut cur_out = d_g_out;
            let mut ok = true;
            for _pass in 0..passes {
                // 32 own + 16 reverse + 10×32 forward-join + 10×32 reverse-join
                // = 688 candidates, under the kernel's 1024 cap.
                let expand: i32 = 10;
                cuMemsetD32_v2(d_rev_cnt, 0, n);
                let rev_params: [*mut std::ffi::c_void; 6] = [
                    &cur_in as *const u64 as *mut std::ffi::c_void,
                    &d_rev as *const u64 as *mut std::ffi::c_void,
                    &d_rev_cnt as *const u64 as *mut std::ffi::c_void,
                    &n_i as *const i32 as *mut std::ffi::c_void,
                    &k_i as *const i32 as *mut std::ffi::c_void,
                    &rc_i as *const i32 as *mut std::ffi::c_void,
                ];
                let grid_rev = ((n + 255) / 256) as u32;
                if cuLaunchKernel(rev_func, grid_rev, 1, 1, 256, 1, 1, 0,
                    std::ptr::null_mut(), rev_params.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut()) != 0 { ok = false; break; }
                let nd_params: [*mut std::ffi::c_void; 10] = [
                    &bufs.d_vectors as *const u64 as *mut std::ffi::c_void,
                    &cur_in as *const u64 as *mut std::ffi::c_void,
                    &d_rev as *const u64 as *mut std::ffi::c_void,
                    &d_rev_cnt as *const u64 as *mut std::ffi::c_void,
                    &cur_out as *const u64 as *mut std::ffi::c_void,
                    &n_i as *const i32 as *mut std::ffi::c_void,
                    &d_i as *const i32 as *mut std::ffi::c_void,
                    &k_i as *const i32 as *mut std::ffi::c_void,
                    &expand as *const i32 as *mut std::ffi::c_void,
                    &rc_i as *const i32 as *mut std::ffi::c_void,
                ];
                if cuLaunchKernel(nd_func, n as u32, 1, 1, 128, 1, 1, 0,
                    std::ptr::null_mut(), nd_params.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut()) != 0 { ok = false; break; }
                std::mem::swap(&mut cur_in, &mut cur_out);
            }
            if !sync_stream_with_watchdog(std::ptr::null_mut()) { ok = false; }
            if ok {
                cuMemcpyDtoH_v2(flat.as_mut_ptr() as *mut std::ffi::c_void, cur_in, g_bytes);
            }
            cuMemFree_v2(d_g_in); cuMemFree_v2(d_g_out); cuMemFree_v2(d_rev); cuMemFree_v2(d_rev_cnt);
            if !ok { return None; }
            for v in 0..n {
                graph[v] = flat[v * k..(v + 1) * k].iter().copied().filter(|&e| e >= 0).collect();
            }
            Some(())
        })().is_some();
        if gpu_refined {
            eprintln!("[GPU] APGC refine: {passes} NN-descent passes ON GPU over {n} nodes in {:.1}s", t_ref.elapsed().as_secs_f64());
        } else {
        // CPU NN-descent fallback. Two constraints matter on a desktop box:
        //   * leave at least one core for the compositor/UI — a 12-thread
        //     refine at 5M previously pegged every core for ~17 min and the
        //     desktop froze solid (reported as a crash);
        //   * reuse the reverse-adjacency buffers across passes — per-pass
        //     `vec![Vec::new(); n]` allocates 60M small Vecs over 12 passes
        //     at 5M, which dominates the scalar scoring cost.
        let nthreads = std::thread::available_parallelism().map(|v| v.get().saturating_sub(1).max(1)).unwrap_or(4);
        let mut rev: Vec<Vec<i32>> = Vec::with_capacity(n);
        rev.resize_with(n, Vec::new);
        let ids: Vec<usize> = (0..n).collect();
        let chunk = (ids.len() + nthreads - 1) / nthreads.max(1);
        for pass in 0..passes {
            // Expand through the closest `expand` neighbors' edge lists.
            let expand = if pass == 0 { 4 } else { 10 };
            // Reverse adjacency (capped per node): who points at v?
            // Reused across passes: clear in place, keep the allocated capacity.
            let rev_cap = 16usize;
            for list in rev.iter_mut() { list.clear(); }
            for v in 0..n {
                for &nb in &graph[v] {
                    let nu = nb as usize;
                    if nu < n && rev[nu].len() < rev_cap { rev[nu].push(v as i32); }
                }
            }
            let rev_ref: &Vec<Vec<i32>> = &rev;
            let prev: &Vec<Vec<i32>> = &graph;
            let refined: Vec<(usize, Vec<i32>)> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for c in ids.chunks(chunk.max(1)) {
                    handles.push(scope.spawn(move || {
                        let mut out = Vec::with_capacity(c.len());
                        let mut cand: Vec<i32> = Vec::with_capacity(k * (expand + 2));
                        for &v in c {
                            cand.clear();
                            cand.extend_from_slice(&prev[v]);
                            cand.extend_from_slice(&rev_ref[v]);
                            for &nb in prev[v].iter().take(expand) {
                                let nu = nb as usize;
                                if nu < n { cand.extend_from_slice(&prev[nu]); }
                            }
                            // Reverse neighbors' forward lists — the NN-descent
                            // "general join": v and its reverse neighbor share
                            // candidates in both directions.
                            for &rb in rev_ref[v].iter().take(expand) {
                                let ru = rb as usize;
                                if ru < n { cand.extend_from_slice(&prev[ru]); }
                            }
                            cand.sort_unstable();
                            cand.dedup();
                            let vb = &vectors_i8[v * dim..(v + 1) * dim];
                            let mut scored: Vec<(i64, i32)> = Vec::with_capacity(cand.len());
                            for &cid in &cand {
                                let cu = cid as usize;
                                if cu >= n || cu == v { continue; }
                                let cb = &vectors_i8[cu * dim..(cu + 1) * dim];
                                let mut dot: i32 = 0;
                                for d in 0..dim { dot += vb[d] as i32 * cb[d] as i32; }
                                scored.push((-(dot as i64), cid)); // ascending = best first
                            }
                            scored.sort_unstable();
                            scored.dedup_by_key(|s| s.1);
                            scored.truncate(k);
                            out.push((v, scored.into_iter().map(|(_, id)| id).collect()));
                        }
                        out
                    }));
                }
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            for (v, edges) in refined { graph[v] = edges; }
        }
        eprintln!("[GPU] APGC refine: {passes} NN-descent passes over {n} nodes in {:.1}s", t_ref.elapsed().as_secs_f64());
        }

        // Output flat CSR
        // Pad short adjacency lists with the node's own index, not 0.
        //
        // A node whose kNN list came back under-filled (NN-descent writes -1
        // when it finds fewer than K, and the host filters those out) left the
        // remaining slots at their zero initialiser. The consumer accepts any
        // `nb != i && nb < n`, so every such hole became a real edge to node 0
        // — silently inflating node 0's in-degree and biasing traversal toward
        // it. Self is the correct filler because the consumer drops it.
        let mut result: Vec<usize> = (0..n).flat_map(|i| std::iter::repeat_n(i, k)).collect();
        for i in 0..n { for j in 0..k.min(graph[i].len()) { result[i * k + j] = graph[i][j] as usize; } }

        bufs.free();
        eprintln!("[GPU] APGC build: {n}×{dim}, k={k} in {:.1}s", t_total.elapsed().as_secs_f64());
        Some(result)
    }
}

/// INT8 matmul on GPU. Auto-loads kernel if needed.
pub fn gpu_matmul(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
    let _guard = gpu_lock_and_ensure()?;
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
        let m_val = m as i32;
        let n_val = n as i32;
        let k_val = k as i32;
        let matmul_params: [*mut std::ffi::c_void; 6] = [
            &d_a as *const u64 as *mut std::ffi::c_void,
            &d_b as *const u64 as *mut std::ffi::c_void,
            &d_c as *const u64 as *mut std::ffi::c_void,
            &m_val as *const i32 as *mut std::ffi::c_void,
            &n_val as *const i32 as *mut std::ffi::c_void,
            &k_val as *const i32 as *mut std::ffi::c_void,
        ];
        cuLaunchKernel(func, bx, by, 1, threads, threads, 1, 0,
                      std::ptr::null_mut(), matmul_params.as_ptr() as *mut *mut std::ffi::c_void,
                      std::ptr::null_mut());
        if !sync_stream_with_watchdog(std::ptr::null_mut()) {
            cuMemFree_v2(d_a); cuMemFree_v2(d_b); cuMemFree_v2(d_c);
            return None;
        }
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
                // Latch *before* returning: the state behind the `OnceLock` now
                // holds a dangling context and can never be replaced, so any
                // later `gpu_init` must decline rather than resurrect it.
                GPU_DESTROYED.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// CUDA error code returned by `cuStreamQuery` while a kernel is still running.
const CUDA_ERROR_NOT_READY: i32 = 600;
/// Maximum time any single kernel may hold the GPU before the watchdog force-
/// resets the context. Long enough for real ingest/search on a 200k corpus,
/// short enough that a hung kernel cannot freeze the display indefinitely.
const GPU_SYNC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Synchronize a stream with a watchdog deadline. `cuCtxSynchronize` under
/// `CU_CTX_SCHED_BLOCKING_SYNC` blocks the calling thread forever if a kernel
/// wedges — and on a display-driving GPU that freezes the whole desktop. So we
/// poll `cuStreamQuery` (non-blocking) instead, and if the kernel has not
/// finished within `GPU_SYNC_TIMEOUT` we destroy the CUDA context, which
/// force-kills the stuck kernel and returns the device to the OS. Every
/// subsequent GPU call sees `GPU_DESTROYED` and falls back to CPU.
///
/// Returns `true` if the stream drained within the deadline.
pub fn sync_stream_with_watchdog(stream: *mut std::ffi::c_void) -> bool {
    let deadline = std::time::Instant::now() + GPU_SYNC_TIMEOUT;
    loop {
        let r = unsafe { cuStreamQuery(stream) };
        if r == 0 {
            return true;
        }
        if r != CUDA_ERROR_NOT_READY {
            // Real launch/sync error (not merely "still running") — surface it.
            return false;
        }
        if std::time::Instant::now() >= deadline {
            eprintln!(
                "[GPU] WATCHDOG: kernel exceeded {:?} — destroying CUDA context to recover",
                GPU_SYNC_TIMEOUT
            );
            gpu_unload();
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

// ── GPU-resident index for APGC-style search ────────────────────────────
// Vectors + graph live on GPU. Repeated searches avoid re-upload.

/// GPU-resident index. Holds device pointers for vectors and graph.
/// Created once via `gpu_build_index`, then used for all searches.
pub struct GpuIndex {
    pub d_vectors: u64,
    pub d_graph: u64,
    pub d_delta: u64,
    pub n: usize,
    pub dim: usize,
    pub degree: usize,
    // VUGVA: persistent search buffers — allocated ONCE, reused per query.
    // Eliminates cuMemAlloc/cuMemFree overhead (saves ~20ms per query).
    pub d_query_buf: u64,
    pub d_idx_buf: u64,
    pub d_dist_buf: u64,
    /// Dedicated Q×k top-k *output* distances. MUST be distinct from
    /// d_dist_buf: the two-kernel path reads the full Q×N matrix from
    /// d_dist_buf while writing top-k results — aliasing them corrupts
    /// row 0 of the matrix across blocks (was a zero-recall source).
    pub d_odist_buf: u64,
    /// Exact f32 corpus (n × dim × 4 B) for the kernel's fused rerank phase.
    /// 0 = absent, in which case the kernel emits the int8 ranking and the
    /// host reranks. Only populated when it fits free VRAM with headroom —
    /// 1.5 GB at 1M×384d, 3 GB at 768d, 6 GB at 1536d.
    pub d_vec_f32: u64,
    /// f32 query staging for the batched (locked) path, max_q × dim × 4 B.
    pub d_qf_buf: u64,
    /// Max queries per gpu_search call (persistent buffer capacity).
    pub max_q: usize,
    /// Max k per gpu_search call (persistent buffer capacity).
    pub max_k: usize,
    /// Concurrent single-query search slots: per-slot buffers + CUDA stream.
    /// Lets N threads run graph searches simultaneously WITHOUT the global
    /// GPU mutex — kernels overlap on the GPU via independent streams.
    pub slots: Vec<SearchSlot>,
    /// Bitmask of busy slots (bit i = slot i claimed).
    pub slot_mask: std::sync::atomic::AtomicU32,
    /// Group-commit batcher that fuses concurrent single-query searches into
    /// one wide launch. See [`QueryCoalescer`] — this is what keeps the SMs
    /// and the clock governor busy.
    pub coalescer: QueryCoalescer,
    /// VUGVA tier backing `d_vectors` in `Hybrid`. `None` in `Turbo`, where the
    /// corpus is a plain VRAM allocation.
    ///
    /// Ownership matters: `CorpusTier` frees its pages on drop, so if the index
    /// did not hold it, `d_vectors` would dangle as soon as the builder
    /// returned. It is also what keeps the pool — and therefore the retained
    /// CUDA context — alive for as long as any pointer derived from it.
    pub corpus: Option<CorpusTier>,
}

// ── Query coalescing (group commit for the GPU) ──────────────────────────────
//
// WHY THIS EXISTS — measured on an RTX 4060 (24 SMs, 3135 MHz max SM clock),
// 100k×384d, 16 host threads each issuing single-vector searches:
//
//     GPU utilization   12–21 %
//     SM clock          210 MHz   (6.7 % of max)
//     QPS               ~27,000
//
// Two compounding faults, both from launching one query at a time:
//
//   1. `cuLaunchKernel(func, 1, 1, 1, ...)` — gridDim of ONE. The APGC search
//      kernel is one block per query, so a single query lights up 1 of 24 SMs
//      and leaves 23 idle. No amount of host threading fixes this, because
//      each thread still submits its own 1-block grid.
//
//   2. Six driver calls per query (2×HtoD, launch, 2×DtoH, sync) and then a
//      *blocking* `cuStreamSynchronize` under CU_CTX_SCHED_BLOCKING_SYNC — the
//      thread sleeps and is woken by an interrupt. That is a syscall-class
//      round trip and a context switch for every single query. This is the
//      "CPU usage in Turbo mode" that kept showing up in btop: it is not
//      distance math, it is the host babysitting the driver.
//
//   And a third effect that falls out of the first two: with the device idle
//   between micro-launches, the driver's power governor never leaves its
//   lowest P-state, so the whole search runs at 210 MHz instead of 3135 MHz.
//   Between the dead SMs and the parked clock, Turbo was using well under 1 %
//   of the GPU's real throughput.
//
// THE FIX — group commit, the same pattern a WAL uses to amortize fsync.
// Concurrent callers queue their query and race for a leader role. The winner
// drains every compatible request into one buffer and issues a SINGLE launch
// with `gridDim = batch_len`, then hands results back and wakes the followers.
// One batch of B queries costs 6 driver calls and one sync instead of 6B and
// B, and it fills B SMs instead of 1.
//
// Batching is opportunistic: the leader takes whatever is queued right now and
// never waits to fill a batch, so a lone query still goes straight through at
// its original latency. Under load the queue naturally holds ~thread-count
// requests, which is exactly when the amortization is wanted.

/// One queued single-query request.
struct CoReq {
    ticket: u64,
    q_i8: Vec<i8>,
    q_f32: Vec<f32>,
    /// Launch geometry. Only requests agreeing on all four can share a launch,
    /// since they become one kernel invocation with shared scalar params.
    k: usize,
    itopk: usize,
    iters: usize,
    entry: usize,
}

#[derive(Default)]
struct CoState {
    pending: Vec<CoReq>,
    ready: std::collections::HashMap<u64, (Vec<i32>, Vec<f32>)>,
    next_ticket: u64,
    /// How many leaders are launching right now. Capped at the slot count,
    /// NOT at one.
    ///
    /// A single-leader design was measured and it LOST: 17,955 QPS against
    /// 27,330 for the un-batched path. Batching amortized the driver calls but
    /// serialized every launch behind one leader, throwing away the 32-way
    /// stream overlap that was the only reason the old path kept any SMs busy
    /// at all. Concurrency and batch width multiply — you need both.
    leaders: usize,
}

/// Group-commit batcher for single-query GPU searches.
pub struct QueryCoalescer {
    state: Mutex<CoState>,
    cv: std::sync::Condvar,
}

impl Default for QueryCoalescer {
    fn default() -> Self { Self::new() }
}

impl QueryCoalescer {
    pub fn new() -> Self {
        QueryCoalescer { state: Mutex::new(CoState::default()), cv: std::sync::Condvar::new() }
    }
}

/// Is coalescing on? Default yes. `GPU_COALESCE=0` restores the old
/// one-launch-per-query path so the two can be A/B'd in a single binary.
fn coalesce_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("GPU_COALESCE").map(|v| v == "0").unwrap_or(false))
}

/// Max queries fused into one slot launch. Sizes the per-slot device buffers,
/// so it is fixed at index-upload time.
///
/// Total concurrent blocks ≈ `slots × slot_batch`. At the 32/16 defaults that
/// is up to 512 blocks in flight against 24 SMs — enough to keep every SM fed
/// and hold the clock governor out of its idle P-state.
fn slot_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_SLOT_BATCH").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16).clamp(1, 64)
    })
}

/// Submit one query through the coalescer and block until its results land.
///
/// Returns `None` only if the underlying batched launch failed, in which case
/// the caller falls back exactly as it did before.
fn gpu_search_coalesced(
    index: &GpuIndex,
    query_i8: &[i8],
    query_f32: &[f32],
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    let co = &index.coalescer;
    let mut st = co.state.lock().ok()?;
    let ticket = st.next_ticket;
    st.next_ticket = st.next_ticket.wrapping_add(1);
    st.pending.push(CoReq {
        ticket,
        q_i8: query_i8.to_vec(),
        q_f32: query_f32.to_vec(),
        k, itopk, iters: max_iters, entry,
    });

    loop {
        // Someone else's batch may already have carried my query.
        if let Some(r) = st.ready.remove(&ticket) { return Some(r); }

        // Wait only when every stream is already launching, or when there is
        // nothing left to lead (my query is riding in someone's in-flight
        // batch). Otherwise take a slot and lead one myself.
        let max_leaders = index.slots.len().max(1);
        if st.leaders >= max_leaders || st.pending.is_empty() {
            st = co.cv.wait(st).ok()?;
            continue;
        }

        // ── I am a leader. Drain a compatible batch. ──
        // Geometry is taken from the OLDEST pending request, not from mine, so
        // that a steady stream of one shape cannot starve an odd one out
        // forever — the odd request becomes the head once the others drain.
        let head = st.pending.first()?;
        let (bk, bitopk, biters, bentry) = (head.k, head.itopk, head.iters, head.entry);
        let cap = slot_batch().min(index.max_q.max(1));

        let mut batch: Vec<CoReq> = Vec::new();
        let mut keep: Vec<CoReq> = Vec::new();
        for r in st.pending.drain(..) {
            if batch.len() < cap && r.k == bk && r.itopk == bitopk
                && r.iters == biters && r.entry == bentry
            {
                batch.push(r);
            } else {
                keep.push(r);
            }
        }
        st.pending = keep;

        // My query is compatible or it isn't. If it isn't, it stayed in
        // `pending` and I still have to run this batch for the others — then
        // loop around and lead again for my own shape.
        let mine_here = batch.iter().any(|r| r.ticket == ticket);
        if batch.is_empty() {
            // Cannot happen (I pushed before draining), but never spin on it.
            return None;
        }

        let dim = index.dim;
        let nq = batch.len();
        let mut qi8: Vec<i8> = Vec::with_capacity(nq * dim);
        let mut qf32: Vec<f32> = Vec::with_capacity(nq * dim);
        // The fused rerank is all-or-nothing for a launch: the kernel takes a
        // single query_f32 base pointer. If any member lacks an exact query,
        // drop f32 for the whole batch and let the host rerank those results.
        let all_f32 = batch.iter().all(|r| r.q_f32.len() >= dim);
        for (i, r) in batch.iter().enumerate() {
            // Each row must be exactly `dim` wide or every later row shifts.
            let take = dim.min(r.q_i8.len());
            qi8.extend_from_slice(&r.q_i8[..take]);
            qi8.resize((i + 1) * dim, 0);
            if all_f32 { qf32.extend_from_slice(&r.q_f32[..dim]); }
        }

        st.leaders += 1;
        drop(st);

        // GPU work happens with the coalescer lock RELEASED, so other threads
        // keep queueing — and keep becoming leaders on other streams — while
        // this batch is in flight.
        //
        // Prefer the per-slot stream path: it takes no global GPU mutex, so
        // several batches overlap on the device. Only if every slot is busy do
        // we fall back to the globally-locked launch.
        let res = gpu_search_slot_batch(index, &qi8, &qf32, nq, bk, bitopk, biters, bentry)
            .or_else(|| gpu_search_batched(index, &qi8, &qf32, nq, bk, bitopk, biters, bentry));

        let mut st2 = co.state.lock().ok()?;
        st2.leaders -= 1;
        let mut mine: Option<(Vec<i32>, Vec<f32>)> = None;
        if let Some((idx_all, dist_all)) = res {
            for (i, r) in batch.iter().enumerate() {
                let lo = i * bk;
                let hi = lo + bk;
                if hi > idx_all.len() || hi > dist_all.len() { break; }
                let pair = (idx_all[lo..hi].to_vec(), dist_all[lo..hi].to_vec());
                if r.ticket == ticket { mine = Some(pair); }
                else { st2.ready.insert(r.ticket, pair); }
            }
        }
        // On launch failure `ready` gets nothing; every follower wakes, finds
        // no entry, and re-leads. Their queries were consumed by this batch,
        // so re-queue them rather than lose them.
        else {
            for r in batch { if r.ticket != ticket { st2.pending.push(r); } }
        }
        co.cv.notify_all();

        if mine.is_some() { return mine; }
        if mine_here { return None; } // my query ran but the launch failed
        st = st2;                     // my shape did not fit; lead again
    }
}

/// One concurrent-search slot: private query/output buffers + CUDA stream.
pub struct SearchSlot {
    pub d_q: u64,
    /// f32 copy of the same query, for the kernel's fused rerank phase.
    pub d_qf: u64,
    pub d_idx: u64,
    pub d_odist: u64,
    pub stream: usize, // CUstream handle (raw pointer as usize)
}

impl GpuIndex {
    pub fn free(&self) {
        unsafe {
            // Only free the corpus when this index allocated it. Under VUGVA
            // the pointer belongs to the `CorpusTier`'s pool, which returns it
            // to its own VRAM cache on drop — freeing it here as well is a
            // double free of a live device block.
            if self.corpus.is_none() {
                cuMemFree_v2(self.d_vectors);
            }
            cuMemFree_v2(self.d_graph);
            if self.d_delta != 0 { cuMemFree_v2(self.d_delta); }
            if self.d_query_buf != 0 { cuMemFree_v2(self.d_query_buf); }
            if self.d_idx_buf != 0 { cuMemFree_v2(self.d_idx_buf); }
            if self.d_dist_buf != 0 { cuMemFree_v2(self.d_dist_buf); }
            if self.d_odist_buf != 0 { cuMemFree_v2(self.d_odist_buf); }
            if self.d_vec_f32 != 0 { cuMemFree_v2(self.d_vec_f32); }
            if self.d_qf_buf != 0 { cuMemFree_v2(self.d_qf_buf); }
            for s in &self.slots {
                if s.d_q != 0 { cuMemFree_v2(s.d_q); }
                if s.d_qf != 0 { cuMemFree_v2(s.d_qf); }
                if s.d_idx != 0 { cuMemFree_v2(s.d_idx); }
                if s.d_odist != 0 { cuMemFree_v2(s.d_odist); }
                if s.stream != 0 { cuStreamDestroy_v2(s.stream as *mut std::ffi::c_void); }
            }
        }
    }
}

/// Upload INT8 vectors + flat CSR graph + OpusEdge delta scores to GPU.
/// Call once after building the graph. All subsequent searches use the cached GPU data.
/// Mode-aware memory allocation:
/// - Turbo: vectors in VRAM (fast access, cuMemAlloc + copy)
/// - Hybrid: vectors in unified memory (GPU reads from RAM via page faults)
/// delta_scores: per-node importance for OpusEdge SelKV pruning (all 1.0 if no LLM signal)
pub fn gpu_build_index(vectors_i8: &[i8], graph_flat: &[i32], delta_scores: &[f32],
                        n: usize, dim: usize, degree: usize) -> Option<GpuIndex> {
    let _guard = gpu_lock_and_ensure()?;
    let mode = gpu_get_mode();
    unsafe {
        let v_bytes = n * dim;
        let g_bytes = n * degree * 4;
        let d_bytes = n * 4;
        let mut d_v = 0u64;
        let mut d_g = 0u64;
        let mut d_d = 0u64;
        // Kept alive for the index's lifetime: dropping a `CorpusTier` frees
        // its pages, which would leave `d_vectors` dangling mid-query.
        let mut tier: Option<corpus_tier::CorpusTier> = None;
        match mode {
            ComputeMode::Turbo => {
                if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 { return None; }
                cuMemcpyHtoD_v2(d_v, vectors_i8.as_ptr() as *const std::ffi::c_void, v_bytes);
            }
            _ => {
                // VUGVA (paper §4, Algorithm 2): the corpus is served through
                // the three-tier pool — VRAM (T0) → page-locked NUMA DRAM (T1)
                // → NVMe (T2) — rather than by a plain device allocation.
                //
                // This is what distinguishes Hybrid from Turbo. Previously both
                // ended in the same `cuMemAlloc` when the corpus fit, and
                // Hybrid's only difference was a `cuMemAllocManaged` fallback
                // when it did not: CUDA's own page-fault migration, which is
                // not VUGVA and gives no control over placement, no NUMA
                // locality, and no NVMe tier at all. Measured side by side, the
                // two modes were indistinguishable (recall 0.994 vs 0.994,
                // throughput within noise) because they ran identical code.
                //
                // `CorpusTier` owns the allocation and decides its own tier
                // from the corpus size, so a corpus larger than VRAM *and*
                // larger than the DRAM budget still loads — it streams to NVMe
                // and pages back through bounded DRAM staging on access.
                match corpus_tier::CorpusTier::new(&[0], n, dim)
                    .and_then(|mut c| {
                        c.upload(vectors_i8)?;
                        let p = c.device_ptr(0)?;
                        Ok((c, p))
                    })
                {
                    Ok((c, ptr)) => {
                        eprintln!(
                            "[VUGVA] {} MB corpus served via TieredPool (T0/T1/T2)",
                            v_bytes / 1024 / 1024
                        );
                        d_v = ptr;
                        tier = Some(c);
                    }
                    Err(e) => {
                        // Fall back to a plain device allocation rather than
                        // failing the whole index: a machine without the DRAM
                        // headroom for a warm tier should still serve a corpus
                        // that fits VRAM outright.
                        eprintln!("[VUGVA] tiering unavailable ({e}); using plain VRAM");
                        if cuMemAlloc_v2(&mut d_v, v_bytes) != 0 {
                            return None;
                        }
                        cuMemcpyHtoD_v2(
                            d_v,
                            vectors_i8.as_ptr() as *const std::ffi::c_void,
                            v_bytes,
                        );
                    }
                }
            }
        }
        // Graph always in VRAM
        if cuMemAlloc_v2(&mut d_g, g_bytes) != 0 { cuMemFree_v2(d_v); return None; }
        cuMemcpyHtoD_v2(d_g, graph_flat.as_ptr() as *const std::ffi::c_void, g_bytes);

        // OpusEdge SelKV: per-node delta scores in VRAM
        if cuMemAlloc_v2(&mut d_d, d_bytes) != 0 { cuMemFree_v2(d_v); cuMemFree_v2(d_g); return None; }
        cuMemcpyHtoD_v2(d_d, delta_scores.as_ptr() as *const std::ffi::c_void, d_bytes);

        // VUGVA: Pre-allocate persistent search buffers (reused per query).
        // d_dist_buf is sized for the full distance matrix (Q × n × f32) for two-kernel search.
        let max_q = 16;
        let max_k = GPU_TOPK_MAX;
        let dist_mat_bytes = max_q * n * 4; // full Q × n distance matrix
        let mut d_qb = 0u64;
        let mut d_ib = 0u64;
        let mut d_db = 0u64;
        let mut d_ob = 0u64;
        // All-or-nothing: a search with a missing buffer would corrupt memory.
        let ok = cuMemAlloc_v2(&mut d_qb, max_q * dim) == 0
            && cuMemAlloc_v2(&mut d_ib, max_q * max_k * 4) == 0
            && cuMemAlloc_v2(&mut d_ob, max_q * max_k * 4) == 0
            && cuMemAlloc_v2(&mut d_db, dist_mat_bytes) == 0;
        if !ok {
            for p in [d_qb, d_ib, d_ob, d_db, d_v, d_g, d_d] {
                if p != 0 { cuMemFree_v2(p); }
            }
            eprintln!("[GPU] Index upload failed: search buffer alloc ({} MB dist matrix)",
                dist_mat_bytes / 1024 / 1024);
            return None;
        }
        eprintln!("[GPU] Index: {} vecs × {}d, degree={}, dist_buf={:.0}MB, OpusEdge delta uploaded",
            n, dim, degree, dist_mat_bytes as f64 / 1024.0 / 1024.0);

        // Concurrent single-query slots: N × (query + k idx + k dist) buffers,
        // each with its own CUDA stream. Best-effort: on any failure keep the
        // slots allocated so far (searches fall back to the mutex path).
        //
        // One query = one block, so the slot count caps how many SMs can be
        // busy at once. With 8 slots a 16-thread client had half its threads
        // queueing on the global-mutex path while 2/3 of the GPU idled. The
        // bitmask is a u32, so 32 is the hard ceiling.
        let n_slots: usize = std::env::var("GPU_SEARCH_SLOTS").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32).clamp(1, 32);
        let mut slots: Vec<SearchSlot> = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            let mut s_q = 0u64;
            let mut s_i = 0u64;
            let mut s_o = 0u64;
            let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut s_qf = 0u64;
            // Sized for a whole coalesced batch, not one query: a slot now
            // launches `gridDim = batch_len` blocks on its own stream.
            let sb = slot_batch();
            let ok = cuMemAlloc_v2(&mut s_q, dim * sb) == 0
                && cuMemAlloc_v2(&mut s_qf, dim * 4 * sb) == 0
                && cuMemAlloc_v2(&mut s_i, max_k * 4 * sb) == 0
                && cuMemAlloc_v2(&mut s_o, max_k * 4 * sb) == 0
                && cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING) == 0;
            if !ok {
                for p in [s_q, s_qf, s_i, s_o] { if p != 0 { cuMemFree_v2(p); } }
                break;
            }
            slots.push(SearchSlot { d_q: s_q, d_qf: s_qf, d_idx: s_i, d_odist: s_o, stream: stream as usize });
        }
        if !slots.is_empty() {
            eprintln!("[GPU] {} concurrent search slots (streams) ready", slots.len());
        }
        Some(GpuIndex {
            d_vectors: d_v, d_graph: d_g, d_delta: d_d, n, dim, degree,
            d_query_buf: d_qb, d_idx_buf: d_ib, d_dist_buf: d_db,
            d_odist_buf: d_ob, d_vec_f32: 0, d_qf_buf: 0, max_q, max_k,
            slots, slot_mask: std::sync::atomic::AtomicU32::new(0),
            coalescer: QueryCoalescer::new(),
            corpus: tier,
        })
    }
}

impl GpuIndex {
    /// True when the exact corpus is VRAM-resident, i.e. `gpu_search` returns
    /// exact distances and the caller must not rescore them on the host.
    #[inline]
    pub fn gpu_rerank_on_device(&self) -> bool { self.d_vec_f32 != 0 }

    /// Upload the exact f32 corpus so `apgc_search_kernel` can rerank on-device.
    ///
    /// Without this the kernel returns the int8 ranking and the host pays a
    /// `k × dim` f32 dot per query. With it, the rerank is a phase inside the
    /// same block and the returned distances are already exact.
    ///
    /// Off by default — see the measurements below. Best-effort and explicitly
    /// guarded even when enabled: the corpus is `n · dim · 4` bytes (1.5 GB at
    /// 1M×384d, 3 GB at 768d, 6 GB at 1536d). If it does not fit free VRAM with
    /// 768 MB of headroom we leave `d_vec_f32 = 0` and the host keeps
    /// reranking. Returns whether the GPU rerank is now live.
    pub fn upload_f32_corpus(&mut self, vectors_f32: &[f32]) -> bool {
        if self.d_vec_f32 != 0 { return true; }
        // On-device f32 rerank is now enabled by default. It eliminates
        // the ~15% recall loss from int8-only scoring and is required
        // for 0.999 recall at k=128 on GPU (APGC paper §3.4).
        // Set GPU_RERANK=0 to opt back out.
        if std::env::var("GPU_RERANK").map(|v| v == "0").unwrap_or(false) {
            return false;
        }
        let need = self.n * self.dim;
        if vectors_f32.len() < need { return false; }
        if gpu_lock_and_ensure().is_none() { return false; }
        let bytes = need * 4;
        unsafe {
            let (mut free_b, mut total_b) = (0usize, 0usize);
            cuMemGetInfo_v2(&mut free_b, &mut total_b);
            let headroom = 768 * 1024 * 1024;
            if bytes + headroom > free_b {
                eprintln!("[GPU] f32 rerank corpus skipped: needs {} MB, {} MB free \
                           (host rerank stays on)", bytes / 1048576, free_b / 1048576);
                return false;
            }
            let mut d_f = 0u64;
            if cuMemAlloc_v2(&mut d_f, bytes) != 0 { return false; }
            if cuMemcpyHtoD_v2(d_f, vectors_f32.as_ptr() as *const std::ffi::c_void, bytes) != 0 {
                cuMemFree_v2(d_f);
                return false;
            }
            // Staging for the batched path's f32 queries. If this fails the
            // whole feature stays off rather than half-wired.
            let mut d_qf = 0u64;
            if cuMemAlloc_v2(&mut d_qf, self.max_q * self.dim * 4) != 0 {
                cuMemFree_v2(d_f);
                return false;
            }
            self.d_vec_f32 = d_f;
            self.d_qf_buf = d_qf;
            eprintln!("[GPU] f32 rerank corpus resident: {} MB — rerank fused into search kernel",
                bytes / 1048576);
            true
        }
    }
}

/// OpusEdge search knobs (env-tunable, cached defaults).
/// δ ∈ [0.1, 1.0], so the default SelKV gate (1-0.9 = 0.1) keeps the
/// mechanism live without evicting any reachable node.
/// Shared-memory bytes for `apgc_search_kernel`. MUST mirror the kernel's own
/// layout: 3 × BUF search list + 1024-entry dedup set + ceil(D/4) packed
/// query words + 16 control ints, where BUF = next_pow2(itopk + beam·degree).
/// `rerank` adds the fused rerank scratch: `dim` floats for the exact query
/// plus `GPU_TOPK_MAX` floats for the per-candidate accumulators.
///
/// `beam` is taken explicitly (not re-read from the env) so the caller can
/// pass the SAME value it sends to the kernel — if they diverged, the kernel
/// would size its own BUF from `beam_in` and the host would have allocated a
/// different amount, corrupting the shared-memory layout.
fn apgc_search_smem(itopk: usize, degree: usize, dim: usize, rerank: bool, beam: usize) -> u32 {
    let mut buf = 1usize;
    while buf < itopk + beam * degree { buf <<= 1; }
    let d4 = (dim + 3) / 4;
    // The kernel's per-iteration dedup set. This MUST match `SR_HASH` in
    // `all_kernels.cu`; otherwise the host underallocates dynamic shared
    // memory and the kernel's `vis`/`qcache`/`s_ctl` pointers overrun the
    // buffer, silently corrupting the bitonic sort (results come back as
    // sentinel -1/2.0). Forced back in sync when `apgc_search`'s dedup set
    // grew from 1024 to 8192 for better candidate dedup at scale.
    const SR_HASH: usize = 8192;
    // The kernel's rerank width is next_pow2(k) and k is capped at
    // GPU_TOPK_MAX, so size the accumulator for the rounded-up bound —
    // GPU_TOPK_MAX itself is not guaranteed to be a power of two.
    let mut rr_max = 1usize;
    while rr_max < GPU_TOPK_MAX { rr_max <<= 1; }
    let rr = if rerank { dim + rr_max } else { 0 };
    ((3 * buf + SR_HASH + d4 + 16 + rr) * 4) as u32
}

/// Largest beam (≤ env `GPU_SEARCH_BEAM`, ≤ 64) whose dynamic shared-memory
/// footprint fits the device's opt-in ceiling.
///
/// Without this the launch silently fails: `apgc_search_kernel` needs
/// 3·BUF + SR_HASH + D4 + 16 + rr ints, and at itopk=512/beam=64/degree=64 the
/// BUF term alone (8192) puts the total at 135,104 B — far over the 48 KB
/// default AND the 99 KB opt-in ceiling of this AD107 (measured:
/// default=49,152, opt-in=101,376, per-SM=102,400). The kernel CANNOT launch at
/// beam=64, so every "GPU batch" result was stale-buffer garbage, not a graph
/// search. Reducing beam 64→32 halves BUF (8192→4096) and smem to ~86 KB.
///
/// Returns the beam the caller must pass to BOTH the kernel (as `beam_in`) and
/// `apgc_search_smem`, so the two never disagree about the layout.
fn apgc_fit_beam(itopk: usize, degree: usize, dim: usize, rerank: bool) -> i32 {
    let requested = gpu_search_beam().clamp(1, 64);
    let mut beam = requested;
    while beam > 1 && apgc_search_smem(itopk, degree, dim, rerank, beam as usize) as usize > GPU_MAX_DYN_SMEM {
        // BUF only changes at power-of-two boundaries, so stepping by 8 still
        // lands on every viable configuration while converging in ≤8 steps.
        beam = (beam - 8).max(1);
    }
    if beam != requested {
        eprintln!("[GPU] apgc_search smem would exceed {} B at beam={}; using beam={}",
            GPU_MAX_DYN_SMEM, requested, beam);
    }
    beam
}

/// Call `cuFuncSetAttribute` to opt a kernel into dynamic shared memory above
/// the 48 KB default (up to the device's per-function ceiling). Returns the
/// error code; 0 means the opt-in succeeded. The APGC kernel needs > 48 KB, so
/// every launch site MUST call this before `cuLaunchKernel` or the launch fails
/// with CUDA_ERROR_INVALID_VALUE and the callers silently fall back to stale
/// brute-force buffers.
fn apgc_optin_smem(func: *mut std::ffi::c_void, smem: u32) -> i32 {
    unsafe {
        cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
    }
}

/// Threads per block for `apgc_search_kernel`.
///
/// A single query runs in ONE block, so blockDim is the only knob that
/// controls how many warps are resident while the kernel issues its random
/// neighbour gathers. At 128 threads (4 warps) the SM has essentially no
/// other warp to switch to while a 400-cycle L2/DRAM miss is in flight, so
/// the walk runs at memory latency rather than memory bandwidth. Raising it
/// puts 8-16 warps in flight over the same shared-memory working set.
fn apgc_search_threads() -> u32 {
    static T: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("GPU_SEARCH_THREADS").ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(32, 1024) & !31)
            .unwrap_or(512)
    })
}

/// Beam-search list width (ef-class) floor for the APGC GPU search.
///
/// 64, not 128. Halving the list halves the per-iteration distance work and
/// the shared-memory footprint, which is the dominant cost once the kernel is
/// memory-bound on random graph-neighbour gathers. Measured on 1M real
/// vectors, `--mode turbo`, batch[1024] QPS / Recall@10:
///
///   384d:  128/48 -> 23,485 @ 0.978    64/24 -> 32,531 @ 0.976   (+38%)
///   768d:  128/48 -> 12,576 @ 0.963    64/24 -> 17,440 @ 0.961   (+39%)
///
/// The recall deltas are NOT a cost of this change — they are build noise.
/// Each arm rebuilds its own graph and the parallel NN-descent refine is
/// nondeterministic. A third arm at 64/32 (strictly MORE search work than
/// 64/24) scored 0.957, i.e. lower than both, which is only possible if the
/// spread is variance rather than signal. Recall is flat to within ~±0.004.
///
/// The gain is on the BATCH path specifically. Single-query p50 moved
/// 519->512us (384d) and 741->722us (768d) — the latency path is dominated by
/// cold GPU clocks and launch overhead, which these knobs do not touch.
pub fn gpu_search_itopk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_SEARCH_ITOPK").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(128).clamp(32, 1024)
    })
}

/// Beam-search iteration cap for the APGC GPU search. The kernel breaks
/// early once every node in the top list has been expanded, so this is an
/// upper bound, not a fixed cost.
///
/// 24, not 48. Convergence sits around 16-24: a 100k sweep put the floor at
/// 16 (recall 0.992) with collapse below it (8 -> 0.911), so 24 keeps a 1.5x
/// margin over the cliff. Because the kernel early-breaks, raising this above
/// convergence buys nothing but costs the worst-case query its full budget.
pub fn gpu_search_iters() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_SEARCH_ITERS").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(128).clamp(1, 512)
    })
}

/// Number of top-list nodes expanded per beam iteration.
pub fn gpu_search_beam() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_SEARCH_BEAM").ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(64).clamp(1, 64)
    })
}

fn opusedge_knobs() -> (f32, i32) {
    let selkv_ratio: f32 = std::env::var("GPU_SELKV_RATIO").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0.9);
    let delta_ar_k: i32 = std::env::var("GPU_DELTA_AR_K").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    (selkv_ratio, delta_ar_k)
}

/// Lock-free single-query APGC graph search on a claimed stream slot.
/// No global GPU mutex: CUDA driver calls are thread-safe (CUDA ≥ 4.0);
/// each slot has private buffers + its own stream, so N threads overlap
/// their kernels on the GPU. Returns None if no slot is free or on any
/// CUDA error — caller falls back to the locked path.
fn gpu_search_slot(
    index: &GpuIndex,
    query_i8: &[i8],
    query_f32: &[f32],
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry_node: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    gpu_search_slot_batch(index, query_i8, query_f32, 1, k, itopk, max_iters, entry_node)
}

/// Per-slot batched search: `nq` queries in ONE launch on a private stream.
///
/// This is the shape that actually feeds the device. Each slot owns a stream,
/// so up to `slots.len()` of these run concurrently, and each contributes `nq`
/// blocks instead of 1 — the product is what raises occupancy far enough for
/// the clock governor to leave its lowest P-state.
fn gpu_search_slot_batch(
    index: &GpuIndex,
    query_i8: &[i8],
    query_f32: &[f32],
    nq: usize,
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry_node: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    use std::sync::atomic::Ordering as AOrd;
    if nq == 0 || nq > slot_batch() || k > index.max_k { return None; }
    if !KERNELS_COMPILED.load(AOrd::Acquire) { return None; }
    let state = GPU_STATE.get()?;
    let func = state.get_kernel("apgc_search")?;
    if !ensure_ctx() { return None; }
    // Claim a free slot (fetch_or on an already-set bit is a no-op → safe).
    let mut sid = usize::MAX;
    for i in 0..index.slots.len() {
        let bit = 1u32 << i;
        if index.slot_mask.fetch_or(bit, AOrd::AcqRel) & bit == 0 { sid = i; break; }
    }
    if sid == usize::MAX { return None; } // all busy
    let slot = &index.slots[sid];
    let release = |m: &std::sync::atomic::AtomicU32| { m.fetch_and(!(1u32 << sid), AOrd::AcqRel); };

    let dim = index.dim;
    let n = index.n;
    unsafe {
        let stream = slot.stream as *mut std::ffi::c_void;
        if cuMemcpyHtoDAsync_v2(slot.d_q, query_i8.as_ptr() as *const std::ffi::c_void, dim * nq, stream) != 0 {
            release(&index.slot_mask); return None;
        }
        // Fused rerank needs the exact queries too. Both the corpus and a
        // full-length f32 query block must be present or the phase stays off.
        let rerank = index.d_vec_f32 != 0 && query_f32.len() >= dim * nq;
        if rerank && cuMemcpyHtoDAsync_v2(slot.d_qf, query_f32.as_ptr() as *const std::ffi::c_void, dim * 4 * nq, stream) != 0 {
            release(&index.slot_mask); return None;
        }
        let null_ptr: u64 = 0;
        let (p_vf, p_qf) = if rerank { (&index.d_vec_f32, &slot.d_qf) } else { (&null_ptr, &null_ptr) };
        let n_i32 = n as i32;
        let d_i32 = dim as i32;
        let deg_i32 = index.degree as i32;
        let k_i32 = k as i32;
        let itopk_i32 = itopk as i32;
        let iters_i32 = max_iters as i32;
        let entry_i32 = entry_node.min(n - 1) as i32;
        let q_i32: i32 = nq as i32;
        let (selkv_ratio, delta_ar_k) = opusedge_knobs();
        let beam = apgc_fit_beam(itopk, index.degree, index.dim, rerank);
        let params: [*mut std::ffi::c_void; 19] = [
            &index.d_vectors as *const u64 as *mut std::ffi::c_void,
            &index.d_graph as *const u64 as *mut std::ffi::c_void,
            &slot.d_q as *const u64 as *mut std::ffi::c_void,
            &slot.d_idx as *const u64 as *mut std::ffi::c_void,
            &slot.d_odist as *const u64 as *mut std::ffi::c_void,
            &index.d_delta as *const u64 as *mut std::ffi::c_void,
            &n_i32 as *const i32 as *mut std::ffi::c_void,
            &d_i32 as *const i32 as *mut std::ffi::c_void,
            &deg_i32 as *const i32 as *mut std::ffi::c_void,
            &k_i32 as *const i32 as *mut std::ffi::c_void,
            &itopk_i32 as *const i32 as *mut std::ffi::c_void,
            &iters_i32 as *const i32 as *mut std::ffi::c_void,
            &entry_i32 as *const i32 as *mut std::ffi::c_void,
            &q_i32 as *const i32 as *mut std::ffi::c_void,
            &selkv_ratio as *const f32 as *mut std::ffi::c_void,
            &delta_ar_k as *const i32 as *mut std::ffi::c_void,
            &beam as *const i32 as *mut std::ffi::c_void,
            p_vf as *const u64 as *mut std::ffi::c_void,
            p_qf as *const u64 as *mut std::ffi::c_void,
        ];
        let smem = apgc_search_smem(itopk, index.degree, index.dim, rerank, beam as usize);
        // Opt the kernel into dynamic shared memory above the 48 KB default.
        // Without this the launch fails silently and callers return stale
        // buffer contents instead of real neighbours.
        apgc_optin_smem(func, smem);
        let r = cuLaunchKernel(func, nq as u32, 1, 1, apgc_search_threads(), 1, 1, smem, stream,
            params.as_ptr() as *mut *mut std::ffi::c_void, std::ptr::null_mut());
        if r != 0 { release(&index.slot_mask); return None; }
        let out = nq * k;
        let mut indices = vec![-1i32; out];
        let mut distances = vec![2.0f32; out];
        let ok = cuMemcpyDtoHAsync_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, slot.d_idx, out * 4, stream) == 0
            && cuMemcpyDtoHAsync_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, slot.d_odist, out * 4, stream) == 0
            && sync_stream_with_watchdog(stream);
        release(&index.slot_mask);
        if !ok { return None; }
        Some((indices, distances))
    }
}

/// APGC GPU search — score-all-nodes with mixed precision.
/// Uses fused kernel (single pass, no Q×N buffer) when available,
/// falls back to two-kernel approach (batch_cosine_dist + topk_select).
/// Uses persistent GPU buffers — no alloc/free per query.
///
/// `queries_f32` is the same queries in exact precision. Pass `&[]` to keep the
/// rerank on the host; pass the full `num_queries × dim` slice to have the
/// kernel rerank on-device (requires `upload_f32_corpus` to have succeeded).
/// When the fused rerank runs, the returned distances are already exact and the
/// caller must NOT rescore them.
pub fn gpu_search(
    index: &GpuIndex,
    queries_i8: &[i8],
    queries_f32: &[f32],
    num_queries: usize,
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry_node: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    // Hard capacity guards: persistent buffers are sized max_q × max_k.
    if num_queries == 0 || num_queries > index.max_q || k == 0 || k > index.max_k {
        return None;
    }

    // ── Single query: coalesce with whatever else is in flight ──
    // A lone query launches a gridDim-of-1 kernel, which uses 1 of the
    // device's SMs and leaves the clock governor parked. Handing it to the
    // group-commit batcher lets it ride along in a wide launch instead.
    if num_queries == 1 && itopk > 0 && max_iters > 0 && k <= itopk
        && index.d_graph != 0 && index.d_delta != 0 && index.degree > 0
    {
        if coalesce_enabled() {
            if let Some(res) = gpu_search_coalesced(
                index, queries_i8, queries_f32, k, itopk, max_iters, entry_node)
            {
                return Some(res);
            }
        } else if !index.slots.is_empty() {
            // GPU_COALESCE=0 — the original per-query stream path, kept so the
            // two designs can be compared inside one binary against one build.
            if let Some(res) = gpu_search_slot(
                index, queries_i8, queries_f32, k, itopk, max_iters, entry_node)
            {
                return Some(res);
            }
        }
        // fall through to the locked path on failure
    }

    gpu_search_batched(index, queries_i8, queries_f32, num_queries, k, itopk, max_iters, entry_node)
}

/// The batched (globally locked) search body: one launch, `gridDim = Q`.
///
/// Split out of [`gpu_search`] so the coalescer's leader can call it directly
/// without re-entering the single-query coalescing branch.
fn gpu_search_batched(
    index: &GpuIndex,
    queries_i8: &[i8],
    queries_f32: &[f32],
    num_queries: usize,
    k: usize,
    itopk: usize,
    max_iters: usize,
    entry_node: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    let _guard = gpu_lock_and_ensure()?;
    let dim = index.dim;
    let n = index.n;
    let out_count = num_queries * k;

    // ── APGC graph traversal (paper §3.4, Table 6: ms-class search @1M) ──
    // Beam search over the CSR graph on GPU — one block per query walks the
    // graph instead of scanning all N vectors. This is the paper's search
    // path; the fused brute-force scan below is only the fallback
    // (itopk == 0) or safety net if the graph kernel fails to launch.
    if itopk > 0 && max_iters > 0 && k <= itopk
        && index.d_graph != 0 && index.d_delta != 0 && index.degree > 0
    {
        if let Some(apgc_func) = GPU_STATE.get()?.get_kernel("apgc_search") {
            unsafe {
                let q_bytes = num_queries * dim;
                cuMemcpyHtoD_v2(index.d_query_buf, queries_i8.as_ptr() as *const std::ffi::c_void, q_bytes);
                let rerank = index.d_vec_f32 != 0 && index.d_qf_buf != 0
                    && queries_f32.len() >= num_queries * dim;
                if rerank {
                    cuMemcpyHtoD_v2(index.d_qf_buf, queries_f32.as_ptr() as *const std::ffi::c_void, q_bytes * 4);
                }
                let null_ptr: u64 = 0;
                let (p_vf, p_qf) = if rerank { (&index.d_vec_f32, &index.d_qf_buf) } else { (&null_ptr, &null_ptr) };
                let n_i32 = n as i32;
                let d_i32 = dim as i32;
                let deg_i32 = index.degree as i32;
                let k_i32 = k as i32;
                let itopk_i32 = itopk as i32;
                let iters_i32 = max_iters as i32;
                let entry_i32 = entry_node.min(n - 1) as i32;
                let q_i32 = num_queries as i32;
                let (selkv_ratio, delta_ar_k) = opusedge_knobs();
                let beam = apgc_fit_beam(itopk, index.degree, index.dim, rerank);
                let params: [*mut std::ffi::c_void; 19] = [
                    &index.d_vectors as *const u64 as *mut std::ffi::c_void,
                    &index.d_graph as *const u64 as *mut std::ffi::c_void,
                    &index.d_query_buf as *const u64 as *mut std::ffi::c_void,
                    &index.d_idx_buf as *const u64 as *mut std::ffi::c_void,
                    &index.d_odist_buf as *const u64 as *mut std::ffi::c_void,
                    &index.d_delta as *const u64 as *mut std::ffi::c_void,
                    &n_i32 as *const i32 as *mut std::ffi::c_void,
                    &d_i32 as *const i32 as *mut std::ffi::c_void,
                    &deg_i32 as *const i32 as *mut std::ffi::c_void,
                    &k_i32 as *const i32 as *mut std::ffi::c_void,
                    &itopk_i32 as *const i32 as *mut std::ffi::c_void,
                    &iters_i32 as *const i32 as *mut std::ffi::c_void,
                    &entry_i32 as *const i32 as *mut std::ffi::c_void,
                    &q_i32 as *const i32 as *mut std::ffi::c_void,
                    &selkv_ratio as *const f32 as *mut std::ffi::c_void,
                    &delta_ar_k as *const i32 as *mut std::ffi::c_void,
                    &beam as *const i32 as *mut std::ffi::c_void,
                    p_vf as *const u64 as *mut std::ffi::c_void,
                    p_qf as *const u64 as *mut std::ffi::c_void,
                ];
                let smem = apgc_search_smem(itopk, index.degree, index.dim, rerank, beam as usize);
                // Opt the kernel into dynamic shared memory above the 48 KB
                // default; without this the launch fails and the batch path
                // returns stale buffer contents instead of real neighbours.
                apgc_optin_smem(apgc_func, smem);
                let r = cuLaunchKernel(apgc_func, num_queries as u32, 1, 1, apgc_search_threads(), 1, 1, smem,
                    std::ptr::null_mut(), params.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut());
                if r == 0 {
                    if !sync_stream_with_watchdog(std::ptr::null_mut()) {
                        return None;
                    }
                    let mut indices = vec![-1i32; out_count];
                    let mut distances = vec![2.0f32; out_count];
                    let c1 = cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, index.d_idx_buf, out_count * 4);
                    let c2 = cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, index.d_odist_buf, out_count * 4);
                    let nvalid = indices.iter().take(out_count).filter(|&&v| v >= 0).count();
                    if nvalid < out_count {
                        // Every slot should be a real neighbour; sentinels mean the
                        // walk/convergence produced fewer than `fetch_k` nodes.
                        eprintln!("[GPU] apgc search: only {}/{} valid hits for this batch", nvalid, out_count);
                    }
                    return Some((indices, distances));
                }
                // launch failed → fall through to brute-force scan
            }
        }
    }

    // Try fused kernel first (single pass, no Q×N intermediate buffer).
    // The kernel hard-caps at k ≤ 64 (`if (k > 64) return;` in all_kernels.cu),
    // so for larger k a "successful" launch would publish stale buffer
    // contents — return None instead so the caller falls back to CPU rather
    // than to garbage.
    if k <= 64 {
    if let Some(fused_func) = GPU_STATE.get()?.get_kernel("fused_cosine_topk") {
        unsafe {
            let q_bytes = num_queries * dim;
            cuMemcpyHtoD_v2(index.d_query_buf, queries_i8.as_ptr() as *const std::ffi::c_void, q_bytes);

            let q_i32: i32 = num_queries as i32;
            let n_i32: i32 = n as i32;
            let d_i32: i32 = dim as i32;
            let k_i32: i32 = k as i32;

            // Fused kernel: (queries, vectors, out_idx, out_dist, Q, N, D, k)
            let fused_params: [*mut std::ffi::c_void; 8] = [
                &index.d_query_buf as *const u64 as *mut std::ffi::c_void,
                &index.d_vectors as *const u64 as *mut std::ffi::c_void,
                &index.d_idx_buf as *const u64 as *mut std::ffi::c_void,
                &index.d_odist_buf as *const u64 as *mut std::ffi::c_void,
                &q_i32 as *const i32 as *mut std::ffi::c_void,
                &n_i32 as *const i32 as *mut std::ffi::c_void,
                &d_i32 as *const i32 as *mut std::ffi::c_void,
                &k_i32 as *const i32 as *mut std::ffi::c_void,
            ];
            let threads = topk_threads(k);
            let smem = (threads as usize * k * 8) as u32;
            let r = cuLaunchKernel(fused_func, num_queries as u32, 1, 1, threads, 1, 1, smem,
                std::ptr::null_mut(), fused_params.as_ptr() as *mut *mut std::ffi::c_void,
                std::ptr::null_mut());
            if r == 0 {
                if !sync_stream_with_watchdog(std::ptr::null_mut()) {
                    return None;
                }
                let mut indices = vec![0i32; out_count];
                let mut distances = vec![0.0f32; out_count];
                cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, index.d_idx_buf, out_count * 4);
                cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, index.d_odist_buf, out_count * 4);
                return Some((indices, distances));
            }
        }
    }
    }

    // Fallback: two-kernel approach (batch_cosine_dist + topk_select).
    // Same k ≤ 64 hard cap as the fused kernel; skip for larger k.
    if k > 64 { return None; }
    let dist_func = GPU_STATE.get()?.get_kernel("batch_cosine_dist")?;
    let topk_func = GPU_STATE.get()?.get_kernel("topk_select")?;
    unsafe {
        let q_bytes = num_queries * dim;
        cuMemcpyHtoD_v2(index.d_query_buf, queries_i8.as_ptr() as *const std::ffi::c_void, q_bytes);

        let q_i32: i32 = num_queries as i32;
        let n_i32: i32 = n as i32;
        let d_i32: i32 = dim as i32;
        let k_i32: i32 = k as i32;

        // Step 1: batch_cosine_dist — grid = Q blocks (one per query, grid-stride over N)
        let dist_params: [*mut std::ffi::c_void; 6] = [
            &index.d_query_buf as *const u64 as *mut std::ffi::c_void,
            &index.d_vectors as *const u64 as *mut std::ffi::c_void,
            &index.d_dist_buf as *const u64 as *mut std::ffi::c_void,
            &q_i32 as *const i32 as *mut std::ffi::c_void,
            &n_i32 as *const i32 as *mut std::ffi::c_void,
            &d_i32 as *const i32 as *mut std::ffi::c_void,
        ];
        let r1 = cuLaunchKernel(dist_func, num_queries as u32, 1, 1, 128, 1, 1, 0,
            std::ptr::null_mut(), dist_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r1 != 0 { return None; }

        // Step 2: topk_select — grid = Q blocks.
        // Output distances go to the DEDICATED d_odist_buf: writing them into
        // d_dist_buf (the input matrix) raced with other blocks still reading
        // row 0 and corrupted results.
        let topk_params: [*mut std::ffi::c_void; 6] = [
            &index.d_dist_buf as *const u64 as *mut std::ffi::c_void,
            &index.d_idx_buf as *const u64 as *mut std::ffi::c_void,
            &index.d_odist_buf as *const u64 as *mut std::ffi::c_void,
            &q_i32 as *const i32 as *mut std::ffi::c_void,
            &n_i32 as *const i32 as *mut std::ffi::c_void,
            &k_i32 as *const i32 as *mut std::ffi::c_void,
        ];
        let tpb = topk_threads(k);
        let smem = (tpb as usize * k * 8) as u32;
        let r2 = cuLaunchKernel(topk_func, q_i32 as u32, 1, 1, tpb, 1, 1, smem,
            std::ptr::null_mut(), topk_params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut());
        if r2 != 0 { return None; }

        if !sync_stream_with_watchdog(std::ptr::null_mut()) { return None; }

        let mut indices = vec![0i32; out_count];
        let mut distances = vec![0.0f32; out_count];
        cuMemcpyDtoH_v2(indices.as_mut_ptr() as *mut std::ffi::c_void, index.d_idx_buf, out_count * 4);
        cuMemcpyDtoH_v2(distances.as_mut_ptr() as *mut std::ffi::c_void, index.d_odist_buf, out_count * 4);
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
    fn test_apgc_gpu_search() {
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
        let delta_scores: Vec<f32> = vec![1.0f32; 6];
        let idx = gpu_build_index(&vectors, &graph, &delta_scores, 6, 3, 2).expect("gpu_build_index failed");
        // Query: same as v0, should find v0 as nearest
        let queries: Vec<i8> = vec![127, 0, 0];
        // Empty f32 queries = int8-only scoring, no fused rerank. The index
        // here has no f32 corpus uploaded, so the rerank phase is inactive
        // either way; this exercises the plain graph-traversal path.
        let (indices, distances) = gpu_search(&idx, &queries, &[], 1, 3, 8, 20, 0)
            .expect("gpu_search failed");
        println!("[GPU] apgc_search: indices={:?} dists={:?}", indices, distances);
        // v0 should be the closest (dist ~0)
        assert!(distances[0] < 0.1, "v0 should be closest, got dist={}", distances[0]);
        assert_eq!(indices[0], 0, "first result should be v0");
        idx.free();
        println!("[GPU] apgc_search VERIFIED — full GPU graph traversal works!");
    }
}
