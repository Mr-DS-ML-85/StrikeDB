//! Raw FFI declarations for the **CUDA Driver API** subset used by VUGVA.
//!
//! Every entry point is resolved at runtime with `dlsym` through
//! [`super::cuda_module`]. Nothing here is linked at build time.
//!
//! That is a deliberate change from the original implementation, which
//! declared these in an `extern` block under `#[link(name = "cuda")]` while
//! `ffi/mod.rs` simultaneously carried a full `dlopen` loader that nothing
//! called. The two approaches contradicted each other, and the linker won: the
//! crate grew a hard `DT_NEEDED` on `libcuda.so`, so a binary embedding VUGVA
//! refused to *start* on a machine without an NVIDIA driver — even if it never
//! touched a GPU. For a library that is supposed to degrade to CPU, that is
//! fatal, and for a database that embeds it, worse.
//!
//! With late binding the cost of a missing driver is a `CUDA_ERROR_NOT_FOUND`
//! from the first call, which callers already handle. Each wrapper resolves its
//! symbol once into a `OnceLock` and thereafter costs an atomic load and an
//! indirect call.

use std::ffi::{c_char, c_int, c_void};

// ============================================================================
// Opaque types (mirror cuda.h)
// ============================================================================

/// `CUdevice` — integer device handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUdevice(pub i32);

/// `CUcontext` — pointer-sized context handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUcontext(pub *mut c_void);

/// `CUmodule` — loadable module.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUmodule(pub *mut c_void);

/// `CUfunction` — handle to a kernel function.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUfunction(pub *mut c_void);

/// `CUstream` — stream handle; null = default stream.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUstream(pub *mut c_void);

/// `CUdeviceptr` — pointer to device memory.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUdeviceptr(pub u64);

/// `CUevent` — CUDA event handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CUevent(pub *mut c_void);

/// `CUresult` — return code from the driver API.
pub type CUresult = c_int;
/// `CUDA_SUCCESS`
pub const CUDA_SUCCESS: c_int = 0;
/// `CUresult::CUDA_ERROR_INVALID_VALUE`
pub const CUDA_ERROR_INVALID_VALUE: c_int = 1;
/// `CUresult::CUDA_ERROR_OUT_OF_MEMORY`
pub const CUDA_ERROR_OUT_OF_MEMORY: c_int = 2;
/// `CUresult::CUDA_ERROR_NOT_INITIALIZED`
pub const CUDA_ERROR_NOT_INITIALIZED: c_int = 3;
/// `CUresult::CUDA_ERROR_NO_DEVICE`
pub const CUDA_ERROR_NO_DEVICE: c_int = 100;
/// `CUresult::CUDA_ERROR_NOT_READY` — async work still outstanding.
pub const CUDA_ERROR_NOT_READY: c_int = 600;
/// `CUresult::CUDA_ERROR_NOT_FOUND` — also returned by this module when the
/// driver is absent or does not export the requested symbol.
pub const CUDA_ERROR_NOT_FOUND: c_int = 500;
/// `CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` — the mapping this call
/// would install is already in place. Idempotent callers should treat it as
/// success: with a *primary* context, another library in the same process may
/// legitimately have enabled the same pair first.
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: c_int = 704;
/// `CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED` — the inverse, returned when
/// disabling a mapping that was never enabled.
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: c_int = 705;

// ============================================================================
// Device attribute IDs (subset from CUdevice_attribute enum)
// ============================================================================

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: c_int = 31;
/// `CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: c_int = 33;
/// `CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`
pub const CUDEVICE_ATTRIBUTE_PCI_DEVICE_ID: c_int = 34;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: c_int = 62;
/// `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`
pub const CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: c_int = 81;
/// `CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`
pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: c_int = 50;
/// `CU_DEVICE_ATTRIBUTE_INTEGRATED` — 1 when the GPU shares system memory.
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;

// ============================================================================
// Stream / event flags
// ============================================================================

/// `CU_STREAM_DEFAULT` — default stream.
pub const CU_STREAM_DEFAULT: u32 = 0x00;
/// `CU_STREAM_NON_BLOCKING` — stream does not synchronize with null stream.
pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;

/// `CU_EVENT_DEFAULT`
pub const CU_EVENT_DEFAULT: u32 = 0x00;
/// `CU_EVENT_BLOCKING_SYNC` — `cuEventSynchronize` sleeps instead of spinning.
pub const CU_EVENT_BLOCKING_SYNC: u32 = 0x01;
/// `CU_EVENT_DISABLE_TIMING` — cheaper event; cannot be used for elapsed time.
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x02;

// ============================================================================
// Memory flags
// ============================================================================

/// `CU_MEMHOSTALLOC_PORTABLE` — memory is portable across CUDA contexts.
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
/// `CU_MEMHOSTALLOC_DEVICEMAP` — map allocation into device address space.
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
/// `CU_MEMHOSTALLOC_WRITECOMBINED` — write-combined memory (faster host writes).
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 0x04;

/// `CU_MEMHOSTREGISTER_PORTABLE`
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;
/// `CU_MEMHOSTREGISTER_DEVICEMAP`
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
/// `CU_MEMHOSTREGISTER_IOMEMORY` — register mapped I/O memory.
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 0x04;

// ============================================================================
// Late-bound entry points
// ============================================================================

/// Declare CUDA driver entry points that are resolved lazily via `dlsym`.
///
/// Generates, for each signature, a `pub unsafe fn` with the identical shape to
/// the C prototype, so call sites read exactly as they would against a linked
/// `extern` block. The symbol address is cached in a per-function `OnceLock`;
/// a driver that is missing or too old to export the symbol yields
/// `CUDA_ERROR_NOT_FOUND` rather than a link failure or a crash.
macro_rules! cuda_api {
    ($(
        $(#[$attr:meta])*
        fn $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> CUresult;
    )*) => {
        $(
            $(#[$attr])*
            ///
            /// Resolved from `libcuda.so.1` on first call. Returns
            /// `CUDA_ERROR_NOT_FOUND` if the driver is absent.
            ///
            /// # Safety
            ///
            /// Same contract as the underlying CUDA Driver API entry point:
            /// pointers must be valid and correctly sized, and handles must
            /// belong to the current context.
            #[inline]
            #[allow(non_snake_case)]
            // The arity is the C prototype's, not a design choice: `cuLaunchKernel`
            // takes 11 parameters. Grouping them into a struct here would mean this
            // binding no longer reads like the entry point it forwards to, which is
            // the one property a raw FFI shim has to keep.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $name( $($arg: $ty),* ) -> CUresult {
                type Fp = unsafe extern "C" fn( $($ty),* ) -> CUresult;
                static SLOT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                // 0 is the "unresolved" sentinel: a valid symbol never lives at
                // address 0, so we do not need a second Option discriminant.
                let addr = *SLOT.get_or_init(|| super::cuda_sym_addr(stringify!($name)));
                if addr == 0 {
                    return CUDA_ERROR_NOT_FOUND;
                }
                // SAFETY: `addr` came from `dlsym` on libcuda for this exact
                // symbol name, and `Fp` mirrors the documented prototype.
                let f: Fp = std::mem::transmute(addr);
                f( $($arg),* )
            }
        )*
    };
}

cuda_api! {
    // ---- Initialization & device query ----

    /// `cuInit` — initialize the driver API. Must precede every other call.
    fn cuInit(flags: u32) -> CUresult;
    /// `cuDeviceGetCount` — number of CUDA-capable devices.
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    /// `cuDeviceGet` — handle for the device at `ordinal`.
    fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
    /// `cuDeviceGetName` — device name into a caller-provided buffer.
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    /// `cuDeviceComputeCapability` — compute capability major/minor.
    fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult;
    /// `cuDeviceTotalMem_v2` — total VRAM on the device.
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    /// `cuDeviceGetAttribute` — query one `CUdevice_attribute`.
    fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;

    // ---- Context: primary (preferred) ----

    /// `cuDevicePrimaryCtxRetain` — retain the device's primary context,
    /// creating it on first use. Reference-counted and shared with the CUDA
    /// runtime, which is what makes it the right handle for a library: two
    /// components retaining the same device cooperate instead of each paying
    /// for a private context.
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    /// `cuDevicePrimaryCtxRelease_v2` — drop one reference to the primary context.
    fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;

    // ---- Context: explicit ----

    /// `cuCtxCreate_v2` — create a *private* context. Prefer
    /// `cuDevicePrimaryCtxRetain` unless isolation is genuinely required.
    fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
    /// `cuCtxDestroy_v2` — destroy a context created by `cuCtxCreate_v2`.
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    /// `cuCtxGetCurrent` — context bound to the calling thread.
    fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    /// `cuCtxSetCurrent` — bind a context to the calling thread.
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    /// `cuCtxPushCurrent_v2` — push a context onto the thread's stack.
    fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    /// `cuCtxPopCurrent_v2` — pop the thread's current context.
    fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult;
    /// `cuCtxSynchronize` — block until all work in the context completes.
    fn cuCtxSynchronize() -> CUresult;
    /// `cuCtxGetDevice` — device backing the current context.
    fn cuCtxGetDevice(dev: *mut CUdevice) -> CUresult;

    // ---- Peer access ----

    /// `cuDeviceCanAccessPeer` — whether `dev` can address `peerDev` directly.
    fn cuDeviceCanAccessPeer(canAccessPeer: *mut c_int, dev: CUdevice, peerDev: CUdevice) -> CUresult;
    /// `cuCtxEnablePeerAccess` — enable P2P from the current context.
    fn cuCtxEnablePeerAccess(peerCtx: CUcontext, flags: u32) -> CUresult;
    /// `cuCtxDisablePeerAccess` — disable P2P from the current context.
    fn cuCtxDisablePeerAccess(peerCtx: CUcontext) -> CUresult;

    // ---- Memory: device allocation ----

    /// `cuMemAlloc_v2` — allocate device memory.
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    /// `cuMemFree_v2` — free device memory.
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    /// `cuMemAllocManaged` — allocate unified (migratable) memory.
    fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: u32) -> CUresult;

    // ---- Memory: host pinned allocation ----

    /// `cuMemHostAlloc` — page-locked host memory, DMA-addressable by the GPU.
    fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: u32) -> CUresult;
    /// `cuMemFreeHost` — free memory from `cuMemHostAlloc`.
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    /// `cuMemHostRegister` — page-lock an existing host range in place.
    fn cuMemHostRegister(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult;
    /// `cuMemHostUnregister` — undo `cuMemHostRegister`.
    fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
    /// `cuMemHostGetDevicePointer_v2` — device address of a mapped host range.
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: u32) -> CUresult;

    // ---- Memory: copies (synchronous) ----

    /// `cuMemcpyHtoD_v2` — host → device, blocking.
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytecount: usize) -> CUresult;
    /// `cuMemcpyDtoH_v2` — device → host, blocking.
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytecount: usize) -> CUresult;
    /// `cuMemcpyDtoD_v2` — device → device, blocking.
    fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytecount: usize) -> CUresult;

    // ---- Memory: copies (asynchronous) ----

    /// `cuMemcpyHtoDAsync_v2` — host → device on a stream. Only truly
    /// asynchronous when the host side is page-locked.
    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytecount: usize, stream: CUstream) -> CUresult;
    /// `cuMemcpyDtoHAsync_v2` — device → host on a stream.
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, bytecount: usize, stream: CUstream) -> CUresult;
    /// `cuMemcpyDtoDAsync_v2` — device → device on a stream.
    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, bytecount: usize, stream: CUstream) -> CUresult;
    /// `cuMemcpyPeerAsync` — GPU → GPU across contexts, without host staging.
    fn cuMemcpyPeerAsync(dst: CUdeviceptr, dst_ctx: CUcontext, src: CUdeviceptr, src_ctx: CUcontext, bytecount: usize, stream: CUstream) -> CUresult;

    // ---- Memory: query & fill ----

    /// `cuMemGetInfo_v2` — free and total VRAM in the current context.
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    /// `cuMemGetAddressRange` — base and size of the allocation holding `dptr`.
    fn cuMemGetAddressRange(pbase: *mut CUdeviceptr, pbytesize: *mut usize, dptr: CUdeviceptr) -> CUresult;
    /// `cuMemsetD32_v2` — fill `n` 32-bit words.
    fn cuMemsetD32_v2(dptr: CUdeviceptr, value: u32, n: usize) -> CUresult;
    /// `cuMemsetD8_v2` — fill `n` bytes.
    fn cuMemsetD8_v2(dptr: CUdeviceptr, value: u8, n: usize) -> CUresult;

    // ---- Streams ----

    /// `cuStreamCreate` — create a stream.
    fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    /// `cuStreamCreateWithPriority` — create a stream at a given priority.
    fn cuStreamCreateWithPriority(stream: *mut CUstream, flags: u32, priority: c_int) -> CUresult;
    /// `cuStreamDestroy_v2` — destroy a stream.
    fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    /// `cuStreamSynchronize` — block until the stream drains.
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    /// `cuStreamQuery` — `CUDA_SUCCESS` if drained, `CUDA_ERROR_NOT_READY` otherwise.
    fn cuStreamQuery(stream: CUstream) -> CUresult;
    /// `cuStreamGetPriority` — priority of an existing stream.
    fn cuStreamGetPriority(stream: CUstream, priority: *mut c_int) -> CUresult;
    /// `cuStreamWaitEvent` — make a stream wait on an event recorded elsewhere.
    /// This is how a transfer stream and a compute stream are ordered without
    /// dragging the host into the loop.
    fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: u32) -> CUresult;
    /// `cuCtxGetStreamPriorityRange` — the device's greatest/least priorities.
    fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult;

    // ---- Events ----

    /// `cuEventCreate` — create an event.
    fn cuEventCreate(event: *mut CUevent, flags: u32) -> CUresult;
    /// `cuEventDestroy_v2` — destroy an event.
    fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    /// `cuEventRecord` — record an event into a stream.
    fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    /// `cuEventSynchronize` — block until the event fires.
    fn cuEventSynchronize(event: CUevent) -> CUresult;
    /// `cuEventQuery` — `CUDA_SUCCESS` if fired, `CUDA_ERROR_NOT_READY` otherwise.
    fn cuEventQuery(event: CUevent) -> CUresult;
    /// `cuEventElapsedTime` — milliseconds between two recorded events.
    fn cuEventElapsedTime(millis: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    // ---- Modules & kernel launch ----

    /// `cuModuleLoadData` — load a PTX/cubin image from memory.
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    /// `cuModuleLoadDataEx` — load an image with JIT options.
    fn cuModuleLoadDataEx(module: *mut CUmodule, image: *const c_void, numOptions: c_int, options: *mut *mut c_void, optionValues: *mut *mut c_void) -> CUresult;
    /// `cuModuleGetFunction` — look up a kernel by name.
    fn cuModuleGetFunction(hfunc: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;
    /// `cuModuleGetGlobal_v2` — address and size of a `__device__` global.
    fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const c_char) -> CUresult;
    /// `cuModuleUnload` — unload a module.
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    /// `cuLaunchKernel` — launch a kernel on a stream.
    fn cuLaunchKernel(
        func: CUfunction,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

// ============================================================================
// Error strings
// ============================================================================

/// Human-readable name for a `CUresult`, e.g. `"CUDA_ERROR_OUT_OF_MEMORY"`.
///
/// Asks the driver first via `cuGetErrorName`, so the text tracks whatever
/// driver is actually loaded rather than a table that rots in this file. Falls
/// back to the numeric code when the driver is absent — which is itself the
/// answer, since a missing driver is the reason most callers land here.
pub fn error_name(code: CUresult) -> String {
    if let Some(s) = driver_error_string("cuGetErrorName", code) {
        return s;
    }
    match code {
        CUDA_SUCCESS => "CUDA_SUCCESS".to_string(),
        CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND (no driver, or symbol missing)".to_string(),
        other => format!("CUresult({other})"),
    }
}

/// Human-readable description for a `CUresult`, e.g. `"out of memory"`.
pub fn error_string(code: CUresult) -> String {
    driver_error_string("cuGetErrorString", code).unwrap_or_else(|| error_name(code))
}

/// Shared body of [`error_name`] and [`error_string`]: both driver entry points
/// have the signature `CUresult f(CUresult, const char**)`.
fn driver_error_string(symbol: &str, code: CUresult) -> Option<String> {
    type Fp = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
    let addr = super::cuda_sym_addr(symbol);
    if addr == 0 {
        return None;
    }
    let mut s: *const c_char = std::ptr::null();
    // SAFETY: `addr` is the driver's own entry point for `symbol`, whose
    // prototype `Fp` mirrors. The driver writes a pointer to a static,
    // NUL-terminated string it owns; we only read it.
    unsafe {
        let f: Fp = std::mem::transmute(addr);
        if f(code, &mut s) != CUDA_SUCCESS || s.is_null() {
            return None;
        }
        Some(std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned())
    }
}

// ============================================================================
// Convenience impls
// ============================================================================

impl CUdevice {
    /// Integer device ordinal.
    pub fn ordinal(self) -> i32 {
        self.0
    }
}

impl CUdeviceptr {
    /// Null device pointer.
    pub const NULL: Self = CUdeviceptr(0);
    /// `true` if the device pointer is zero (unallocated).
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl CUstream {
    /// Default stream handle.
    pub const NULL: Self = CUstream(std::ptr::null_mut());

    /// `true` if this is the default (null) stream.
    pub fn is_default(self) -> bool {
        self.0.is_null()
    }
}

impl CUcontext {
    /// Null context handle.
    pub const NULL: Self = CUcontext(std::ptr::null_mut());
    /// `true` if no context is bound.
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

// SAFETY: CUDA contexts, streams, events and modules are driver-side handles
// that may legally be used from any host thread; the driver serialises access
// internally. Sending them between threads is exactly what the stream pool and
// the background sweep need, and the raw pointer inside is opaque — we never
// dereference it on the Rust side.
unsafe impl Send for CUcontext {}
unsafe impl Sync for CUcontext {}
unsafe impl Send for CUstream {}
unsafe impl Sync for CUstream {}
unsafe impl Send for CUevent {}
unsafe impl Sync for CUevent {}
unsafe impl Send for CUmodule {}
unsafe impl Sync for CUmodule {}
unsafe impl Send for CUfunction {}
unsafe impl Sync for CUfunction {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of late binding: calling into the API on a machine with
    /// no driver must return an error, not abort the process. This test is the
    /// one that would have caught the `#[link(name = "cuda")]` regression,
    /// because it passes either way *at runtime* — but with a hard link the
    /// test binary could not have started at all on a driverless box.
    #[test]
    fn missing_symbol_degrades_to_error() {
        // SAFETY: cuInit's only argument is a flags word; 0 is always valid.
        let r = unsafe { cuInit(0) };
        assert!(
            r == CUDA_SUCCESS || r == CUDA_ERROR_NOT_FOUND || r == CUDA_ERROR_NO_DEVICE,
            "cuInit must succeed or report a clean error, got {r}"
        );
    }

    #[test]
    fn error_name_is_never_empty() {
        assert!(!error_name(CUDA_ERROR_OUT_OF_MEMORY).is_empty());
        assert!(!error_name(CUDA_ERROR_NOT_FOUND).is_empty());
        // An implausible code must still round-trip to something printable
        // rather than panicking or yielding "".
        assert!(!error_name(31337).is_empty());
    }

    #[test]
    fn null_handles_report_null() {
        assert!(CUdeviceptr::NULL.is_null());
        assert!(CUstream::NULL.is_default());
        assert!(CUcontext::NULL.is_null());
    }
}
