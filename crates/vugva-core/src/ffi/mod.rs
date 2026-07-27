//! Raw FFI bindings to NVIDIA's CUDA Driver API and NVRTC runtime compiler.
//!
//! Everything in this module uses raw `dlsym`/`dlopen` — no external crates.
//! All symbols are resolved lazily on first use via
//! [`cuda_module`] / [`nvrtc_module`].

pub mod cuda;
pub mod nvrtc;

use crate::{Result, VugvaError, CUDA_SONAME, NVRTC_SONAME};
use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

// ============================================================================
// Raw dlopen: zero-dep ELF loader
// ============================================================================

extern "C" {
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

/// `RTLD_NOW` — resolve all symbols immediately.
const RTLD_NOW: i32 = 2;

// ============================================================================
// Lazy module loader
// ============================================================================

/// A handle to a single loaded shared library.
/// Stored in a `OnceLock` for thread-safe one-shot init.
pub struct LoadedLib {
    pub(crate) handle: *mut c_void,
}

// SAFETY: CUDA/NVRTC handles are process-local; Send is correct for
// transferring ownership across threads. Sync is not needed because
// library loading is serialized via OnceLock.
unsafe impl Send for LoadedLib {}
unsafe impl Sync for LoadedLib {}

/// Open a shared library by SONAME. Tries several common variants.
fn open_lib(soname: &str, fallbacks: &[&str]) -> Result<*mut c_void> {
    let candidates: Vec<&str> = std::iter::once(soname)
        .chain(fallbacks.iter().copied())
        .collect();
    for name in candidates {
        // Clear dlerror before each attempt so stale errors don't persist.
        // SAFETY: dlerror() is thread-safe and returns a valid pointer or null.
        unsafe {
            dlerror();
        }
        let bytes = match std::ffi::CString::new(name.as_bytes()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // SAFETY: dlopen is called with a valid null-terminated string.
        let h = unsafe { dlopen(bytes.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Ok(h);
        }
    }
    // All candidates failed — return a single consolidated error.
    Err(VugvaError::LibLoad {
        library: "cuda/nvrtc",
        os_error: 0,
    })
}

/// Resolve a named symbol from a previously opened library.
///
/// # Safety
///
/// `handle` must be a valid `dlopen` handle. The returned pointer, if
/// non-null, must be transmuted to the correct function pointer type.
unsafe fn resolve(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
    let bytes = match std::ffi::CString::new(name) {
        Ok(c) => c,
        Err(_) => return None,
    };
    // SAFETY: handle is valid (checked by caller), name is null-terminated.
    let p = dlsym(handle, bytes.as_ptr());
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

// ============================================================================
// Public re-exports
// ============================================================================

/// Singleton handle to `libcuda.so.1`. Resolved on first call.
pub fn cuda_module() -> Result<&'static LoadedLib> {
    static H: OnceLock<Result<LoadedLib>> = OnceLock::new();
    H.get_or_init(|| {
        let handle = open_lib(CUDA_SONAME, &["libcuda.so", "cuda"])?;
        // SAFETY: handle is non-null (checked by open_lib).
        Ok(LoadedLib { handle })
    })
    .as_ref()
    .map_err(|e| match e {
        VugvaError::LibLoad { .. } => VugvaError::LibLoad {
            library: CUDA_SONAME,
            os_error: 0,
        },
        _ => unreachable!(),
    })
}

/// Singleton handle to `libnvrtc.so`. Resolved on first call.
pub fn nvrtc_module() -> Result<&'static LoadedLib> {
    static H: OnceLock<Result<LoadedLib>> = OnceLock::new();
    H.get_or_init(|| {
        let handle = open_lib(
            NVRTC_SONAME,
            &[
                // CUDA 13
                "libnvrtc.so.13",
                // CUDA 12
                "libnvrtc.so.12",
                // CUDA 11
                "libnvrtc.so.11",
                // CUDA 10
                "libnvrtc.so.10",
                // Unversioned fallback
                "libnvrtc.so",
            ],
        )?;
        // SAFETY: handle is non-null (checked by open_lib).
        Ok(LoadedLib { handle })
    })
    .as_ref()
    .map_err(|e| match e {
        VugvaError::LibLoad { .. } => VugvaError::LibLoad {
            library: NVRTC_SONAME,
            os_error: 0,
        },
        _ => unreachable!(),
    })
}

/// Resolve a symbol from the **CUDA driver** and return its address as a
/// `usize`, or `0` when either the driver or the symbol is absent.
///
/// `0` is the agreed "unresolved" sentinel used by the `cuda_api!` wrappers in
/// [`cuda`]: no valid symbol is ever mapped at address zero, so callers do not
/// need a second `Option` discriminant to distinguish "missing" from "found".
/// A machine with no NVIDIA driver therefore gets `CUDA_ERROR_NOT_FOUND` back
/// from every entry point instead of failing to start.
///
/// The driver exports several entry points only under a versioned name — the
/// unsuffixed `cuCtxCreate`, for instance, is an ABI-compatibility alias that
/// is not guaranteed to exist on every install. We look up the caller's name
/// verbatim first, then fall back to the `_v2`/`_v3` spellings so that a
/// binding declared as `cuFoo` still resolves against a driver that only ships
/// `cuFoo_v2`. Names that already carry a version suffix hit on the first try.
pub(crate) fn cuda_sym_addr(name: &str) -> usize {
    let lib = match cuda_module() {
        Ok(l) => l,
        // No driver on this machine: every symbol is "missing", which the
        // wrappers translate into a returnable error rather than a crash.
        Err(_) => return 0,
    };
    // SAFETY: `lib.handle` came from a successful `dlopen` and stays valid for
    // the life of the process — the `OnceLock` never hands it to `dlclose`.
    unsafe {
        if let Some(p) = resolve(lib.handle, name) {
            return p as usize;
        }
        if !name.ends_with("_v2") && !name.ends_with("_v3") {
            for suffix in ["_v2", "_v3"] {
                if let Some(p) = resolve(lib.handle, &format!("{name}{suffix}")) {
                    return p as usize;
                }
            }
        }
    }
    0
}

/// Resolve a symbol from the **NVRTC** runtime compiler and return its address
/// as a `usize`, or `0` when either the library or the symbol is absent.
///
/// Same `0`-as-sentinel contract as [`cuda_sym_addr`]. No `_v2` fallback: NVRTC
/// does not version its entry points that way — it ships a whole new SONAME per
/// CUDA major release instead, which [`nvrtc_module`] already walks.
pub(crate) fn nvrtc_sym_addr(name: &str) -> usize {
    let lib = match nvrtc_module() {
        Ok(l) => l,
        Err(_) => return 0,
    };
    // SAFETY: `lib.handle` came from a successful `dlopen` and stays valid for
    // the life of the process.
    unsafe { resolve(lib.handle, name).map_or(0, |p| p as usize) }
}

/// Resolve a symbol from the **NVRTC** library and transmute to a typed
/// function pointer.
///
/// # Safety
///
/// The caller must ensure `T` matches the actual symbol signature loaded
/// from `libnvrtc.so`.
#[inline]
pub(crate) unsafe fn nvrtc_sym_to_fn<T>(name: &str) -> Result<T> {
    let lib = nvrtc_module()?;
    match resolve(lib.handle, name) {
        // SAFETY: transmute_copy is safe when T matches the symbol's type.
        Some(p) => Ok(std::mem::transmute_copy::<*mut c_void, T>(&p)),
        None => Err(VugvaError::LibLoad {
            library: "libnvrtc.so",
            os_error: 0,
        }),
    }
}
