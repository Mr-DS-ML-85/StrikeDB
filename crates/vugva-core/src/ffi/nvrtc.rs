//! Raw FFI bindings to **NVRTC** — the runtime CUDA kernel compiler.
//!
//! NVRTC lets us take a CUDA source string at runtime, compile it to PTX
//! or cubin (driver-loadable image), and launch kernels without a `.cu`
//! compile step in our build system. VUGVA uses this to compile a minimal
//! memcpy_peer kernel that respects peer-to-peer access flags without
//! relying on the CUDA Runtime's cudart library.
//!
//! # Late binding
//!
//! Every entry point here is resolved with `dlsym` on first call, exactly as in
//! [`super::cuda`]. The original implementation declared these in a bare
//! `extern "C"` block with no `#[link]` attribute at all, which is worse than it
//! looks: it compiled only because nothing in a *test* build reached the call
//! sites, so the linker never had to find the symbols. The first consumer that
//! actually pulled `nvrtc_kernel::compile_memcpy_peer` into a linked binary
//! would have failed with `undefined reference to nvrtcCreateProgram` — a build
//! error, in a downstream crate, for code that looked fine here.
//!
//! Late binding removes both failure modes: the symbols never reach the linker,
//! and a machine with no CUDA toolkit gets [`NVRTC_ERROR_NOT_FOUND`] back from
//! a call instead of a binary that will not link or start.

use std::ffi::{c_char, c_void};

/// `nvrtcResult` integer return code.
#[allow(non_camel_case_types)]
pub type nvrtcResult = i32;

/// `NVRTC_SUCCESS`
pub const NVRTC_SUCCESS: i32 = 0;

/// `NVRTC_ERROR_OUT_OF_MEMORY`
pub const NVRTC_ERROR_OUT_OF_MEMORY: i32 = 1;

/// `NVRTC_ERROR_PROGRAM_CREATION_FAILURE`
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: i32 = 2;

/// `NVRTC_ERROR_INVALID_INPUT`
pub const NVRTC_ERROR_INVALID_INPUT: i32 = 3;

/// `NVRTC_ERROR_INVALID_PROGRAM`
pub const NVRTC_ERROR_INVALID_PROGRAM: i32 = 4;

/// `NVRTC_ERROR_COMPILATION`
pub const NVRTC_ERROR_COMPILATION: i32 = 6;

/// Not an NVRTC code: VUGVA's sentinel for "libnvrtc is missing, or is too old
/// to export this entry point". Chosen well above the real enum so it can never
/// collide with a value the library itself returns.
pub const NVRTC_ERROR_NOT_FOUND: i32 = 10_000;

/// Opaque handle to a compiled program.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct nvrtcProgram(pub *mut c_void);

impl nvrtcProgram {
    /// A null program handle — the correct initial value for an out-parameter.
    pub const NULL: nvrtcProgram = nvrtcProgram(std::ptr::null_mut());

    /// Is this handle null (never created, or already destroyed)?
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

// SAFETY: an `nvrtcProgram` is an opaque process-local handle. NVRTC permits a
// program to be created on one thread and compiled on another as long as no two
// threads touch the same program concurrently — which is ownership, i.e. `Send`.
unsafe impl Send for nvrtcProgram {}

// ============================================================================
// Lazily-resolved entry points
// ============================================================================

/// Declare NVRTC entry points that are resolved lazily via `dlsym`.
///
/// Mirrors `cuda_api!` in [`super::cuda`]: one `pub unsafe fn` per prototype,
/// address cached in a per-function `OnceLock`, [`NVRTC_ERROR_NOT_FOUND`]
/// returned when the symbol cannot be resolved.
macro_rules! nvrtc_api {
    ($(
        $(#[$attr:meta])*
        fn $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> nvrtcResult;
    )*) => {
        $(
            $(#[$attr])*
            ///
            /// Resolved from `libnvrtc.so.*` on first call. Returns
            /// [`NVRTC_ERROR_NOT_FOUND`] if the library is absent.
            ///
            /// # Safety
            ///
            /// Same contract as the underlying NVRTC entry point: pointers must
            /// be valid and correctly sized, and any output buffer must already
            /// be at least as large as the matching `*Size` query reported.
            #[inline]
            #[allow(non_snake_case)]
            pub unsafe fn $name( $($arg: $ty),* ) -> nvrtcResult {
                type Fp = unsafe extern "C" fn( $($ty),* ) -> nvrtcResult;
                static SLOT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                // 0 is the "unresolved" sentinel: a valid symbol never lives at
                // address 0, so we do not need a second Option discriminant.
                let addr = *SLOT.get_or_init(|| super::nvrtc_sym_addr(stringify!($name)));
                if addr == 0 {
                    return NVRTC_ERROR_NOT_FOUND;
                }
                // SAFETY: `addr` came from `dlsym` on libnvrtc for this exact
                // symbol name, and `Fp` mirrors the documented prototype.
                let f: Fp = std::mem::transmute(addr);
                f( $($arg),* )
            }
        )*
    };
}

nvrtc_api! {
    // ---- Version ----

    /// `nvrtcVersion` — query the compiler's CUDA major/minor version.
    fn nvrtcVersion(major: *mut i32, minor: *mut i32) -> nvrtcResult;

    // ---- Compile ----

    /// `nvrtcCreateProgram` — build a program object from a CUDA source string.
    fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        numHeaders: i32,
        headers: *mut *const c_char,
        includeNames: *mut *const c_char,
    ) -> nvrtcResult;

    /// `nvrtcDestroyProgram` — free a program and null the caller's handle.
    fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;

    /// `nvrtcCompileProgram` — compile with the given command-line options.
    ///
    /// A non-success return does not mean the failure is opaque: call
    /// `nvrtcGetProgramLog` afterwards for the actual compiler diagnostics.
    fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: i32,
        options: *mut *const c_char,
    ) -> nvrtcResult;

    /// `nvrtcGetPTX` — copy the generated PTX into `ptx`.
    fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;

    /// `nvrtcGetPTXSize` — byte size of the PTX, including the NUL terminator.
    fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSize: *mut usize) -> nvrtcResult;

    /// `nvrtcGetCUBIN` — copy the generated cubin into `cubin`.
    ///
    /// Only produced when compiling for a concrete `sm_XX` target rather than a
    /// virtual `compute_XX` one.
    fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;

    /// `nvrtcGetCUBINSize` — byte size of the cubin image.
    fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSize: *mut usize) -> nvrtcResult;

    // ---- Log ----

    /// `nvrtcGetProgramLog` — copy the compiler diagnostics into `log`.
    fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;

    /// `nvrtcGetProgramLogSize` — byte size of the log, including the NUL.
    fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSize: *mut usize) -> nvrtcResult;

    // ---- Specialization ----

    /// `nvrtcAddNameExpression` — register a name to be looked up after
    /// compilation, so a mangled C++ symbol can be recovered.
    fn nvrtcAddNameExpression(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult;

    /// `nvrtcGetLoweredName` — retrieve the mangled name for a previously
    /// registered expression. The returned pointer is owned by the program and
    /// dies with it.
    fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name: *const c_char,
        basename: *mut *const c_char,
    ) -> nvrtcResult;
}

// ============================================================================
// Error strings
// ============================================================================

/// `nvrtcGetErrorString` — human-readable text for an `nvrtcResult`.
///
/// Declared by hand rather than through `nvrtc_api!` because it is the one
/// entry point that returns a string instead of a status code.
///
/// Returns a null pointer when NVRTC is unavailable; prefer [`error_string`],
/// which handles that case for you.
///
/// # Safety
///
/// The returned pointer is owned by NVRTC and is valid for the life of the
/// process. It must not be freed.
#[inline]
#[allow(non_snake_case)]
pub unsafe fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char {
    type Fp = unsafe extern "C" fn(nvrtcResult) -> *const c_char;
    static SLOT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let addr = *SLOT.get_or_init(|| super::nvrtc_sym_addr("nvrtcGetErrorString"));
    if addr == 0 {
        return std::ptr::null();
    }
    // SAFETY: `addr` is the real `nvrtcGetErrorString`, whose prototype `Fp`
    // mirrors exactly.
    let f: Fp = std::mem::transmute(addr);
    f(result)
}

/// Human-readable description of an `nvrtcResult`, as an owned `String`.
///
/// Always returns something usable: falls back to the numeric code when NVRTC
/// is missing or hands back a null or non-UTF-8 string, so error reporting
/// never silently degrades to an empty message.
pub fn error_string(code: nvrtcResult) -> String {
    if code == NVRTC_ERROR_NOT_FOUND {
        return "NVRTC_ERROR_NOT_FOUND (libnvrtc.so is not available)".to_string();
    }
    // SAFETY: the returned pointer is either null (handled) or a static
    // NUL-terminated string owned by NVRTC.
    unsafe {
        let p = nvrtcGetErrorString(code);
        if p.is_null() {
            return format!("nvrtcResult({code})");
        }
        match std::ffi::CStr::from_ptr(p).to_str() {
            Ok(s) if !s.is_empty() => s.to_string(),
            _ => format!("nvrtcResult({code})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of late binding: a call must return an error rather than
    /// failing to link or aborting, whether or not this machine has NVRTC.
    #[test]
    fn version_query_never_crashes() {
        let (mut major, mut minor) = (0i32, 0i32);
        let rc = unsafe { nvrtcVersion(&mut major, &mut minor) };
        if rc == NVRTC_SUCCESS {
            // Present: the reported version must be a plausible CUDA release.
            assert!(major >= 7, "implausible NVRTC major version {major}");
            assert!(minor >= 0);
        } else {
            // Absent: the only acceptable failure is our own sentinel.
            assert_eq!(rc, NVRTC_ERROR_NOT_FOUND, "unexpected nvrtcResult {rc}");
        }
    }

    /// `error_string` must never return an empty message, for any code, on any
    /// machine — that is what makes it safe to use in error paths.
    #[test]
    fn error_string_is_never_empty() {
        for code in [
            NVRTC_SUCCESS,
            NVRTC_ERROR_COMPILATION,
            NVRTC_ERROR_INVALID_PROGRAM,
            NVRTC_ERROR_NOT_FOUND,
            -12345,
        ] {
            assert!(
                !error_string(code).is_empty(),
                "empty description for code {code}"
            );
        }
    }

    #[test]
    fn null_program_reports_null() {
        assert!(nvrtcProgram::NULL.is_null());
        assert!(nvrtcProgram::default().is_null());
    }
}
