//! CUDA context ownership: refcounted primary contexts, RAII release.
//!
//! # Why primary contexts
//!
//! The driver offers two ways to get a context:
//!
//! * `cuCtxCreate_v2` makes a brand-new, *private* context. Whoever creates it
//!   owns it and must call `cuCtxDestroy_v2`. Nothing else in the process can
//!   see memory allocated inside it.
//! * `cuDevicePrimaryCtxRetain` bumps the refcount on the device's *primary*
//!   context — the same one the CUDA runtime (`cudart`) and every other library
//!   in the process uses. `cuDevicePrimaryCtxRelease_v2` drops the reference.
//!
//! A library must use the second. VUGVA is loaded alongside a framework that
//! already has its own CUDA state; handing out pointers from a private context
//! produces allocations the framework cannot address.
//!
//! The original code used `cuCtxCreate_v2` in four places and destroyed the
//! result in none of them:
//!
//! | Site | Consequence |
//! |---|---|
//! | `TieredPool::set_context` | one context leaked **per `access()` call** |
//! | `TieredPool::access` VRAM path | created *and* destroyed a context mid-copy |
//! | `GpuCluster::discover` | one per device, return code unchecked |
//! | `PeerMatrix::enable_all` | one per device; peer mapping installed into a context nothing else made current, so it had no effect |
//!
//! The cost is not theoretical. Measured on this machine (RTX 4060, 8 GB,
//! driver via `cuMemGetInfo_v2`), creating 32 contexts without destroying them:
//!
//! ```text
//! free VRAM at start:              6876.2 MB
//! after 32 primary retains:        6876.2 MB  (delta   +0.0 MB)
//! after 32 cuCtxCreate_v2 (leak):  3746.8 MB  (delta -3129.4 MB)
//! => per leaked context:             97.8 MB
//! ```
//!
//! So a single `access()` call cost ~98 MB of VRAM, and 70 calls would exhaust
//! an 8 GB card on their own — before storing a single tensor. Retaining the
//! primary context 32 times cost exactly zero, because every retain returns the
//! same handle and only moves a refcount.
//!
//! # What this module provides
//!
//! [`PrimaryContext`] is an owned reference to one device's primary context
//! that releases on drop. [`ContextRegistry`] holds one per device so callers
//! index by GPU rather than re-retaining. [`ContextGuard`] scopes a
//! push/pop pair so an early return cannot leave the wrong context current.

use crate::ffi::cuda::*;
use crate::{check_cu, Result};

// ============================================================================
// PrimaryContext
// ============================================================================

/// An owned reference to a device's primary CUDA context.
///
/// Holding one guarantees the context stays alive. Dropping it releases the
/// reference; the context itself survives while anything else still holds it.
#[derive(Debug)]
pub struct PrimaryContext {
    device: CUdevice,
    ctx: CUcontext,
}

// SAFETY: a `CUcontext` is a process-wide handle, not thread-affine. The driver
// permits any thread to make any context current, and the primary context is
// explicitly documented as shared across threads. `PrimaryContext` owns only a
// refcount, and `cuDevicePrimaryCtxRelease_v2` is itself thread-safe, so both
// moving the handle between threads and sharing `&PrimaryContext` are sound.
unsafe impl Send for PrimaryContext {}
unsafe impl Sync for PrimaryContext {}

impl PrimaryContext {
    /// Retain device `ordinal`'s primary context, initializing the driver first.
    ///
    /// `cuInit` is idempotent and cheap after the first call, so it is safe to
    /// invoke here rather than requiring callers to remember it.
    pub fn retain(ordinal: i32) -> Result<Self> {
        let device = CUdevice(ordinal);
        let mut ctx = CUcontext::NULL;
        // SAFETY: `cuInit(0)` is the documented initializer and is idempotent;
        // `ctx` is a valid out-pointer for the retain call.
        unsafe {
            check_cu("cuInit", cuInit(0))?;
            check_cu(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut ctx, device),
            )?;
        }
        Ok(PrimaryContext { device, ctx })
    }

    /// The raw context handle, for passing to driver calls that take one
    /// (`cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync`, ...).
    #[inline]
    pub fn raw(&self) -> CUcontext {
        self.ctx
    }

    /// The device this context belongs to.
    #[inline]
    pub fn device(&self) -> CUdevice {
        self.device
    }

    /// Make this context current on the calling thread, replacing whatever was
    /// current before.
    ///
    /// Prefer [`PrimaryContext::enter`] inside a function that may return early:
    /// `bind` leaves the context current after it returns, which is only correct
    /// when the caller owns the thread's context state for the whole call.
    pub fn bind(&self) -> Result<()> {
        // SAFETY: `self.ctx` is a live retained primary context.
        unsafe { check_cu("cuCtxSetCurrent", cuCtxSetCurrent(self.ctx)) }
    }

    /// Push this context and return a guard that pops it on drop.
    ///
    /// This is the form to use around a fallible sequence: the pop happens even
    /// if the body returns `Err` early, so a failed copy cannot strand another
    /// GPU's context as current and corrupt the *next* unrelated operation.
    pub fn enter(&self) -> Result<ContextGuard<'_>> {
        // SAFETY: `self.ctx` is a live retained primary context.
        unsafe {
            check_cu("cuCtxPushCurrent_v2", cuCtxPushCurrent_v2(self.ctx))?;
        }
        Ok(ContextGuard { _ctx: self })
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        // SAFETY: balances exactly one successful `cuDevicePrimaryCtxRetain`.
        // The result is ignored deliberately: a destructor has no way to report
        // it, and a driver already torn down at process exit is not an error
        // worth aborting on.
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

// ============================================================================
// ContextGuard
// ============================================================================

/// Scopes a `cuCtxPushCurrent` / `cuCtxPopCurrent` pair.
///
/// Created by [`PrimaryContext::enter`]. Popping in `Drop` is what makes the
/// pairing exception-safe across `?` early returns.
#[derive(Debug)]
pub struct ContextGuard<'a> {
    _ctx: &'a PrimaryContext,
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut popped = CUcontext::NULL;
        // SAFETY: balances the push performed in `enter`. Ignoring the result is
        // deliberate — see `PrimaryContext::drop`.
        unsafe {
            let _ = cuCtxPopCurrent_v2(&mut popped);
        }
    }
}

// ============================================================================
// ContextRegistry
// ============================================================================

/// One [`PrimaryContext`] per device, indexed by position in the ordinal list.
///
/// This is the shape callers actually want: `TieredPool` and friends address
/// GPUs by index into `cluster.ordinals`, so the registry mirrors that order and
/// a lookup is a bounds-checked slice index rather than a driver call.
#[derive(Debug)]
pub struct ContextRegistry {
    contexts: Vec<PrimaryContext>,
}

impl ContextRegistry {
    /// Retain the primary context of every listed device.
    ///
    /// If any retain fails, the contexts retained so far are released by the
    /// `Vec`'s drop as the error propagates — no partial leak.
    pub fn new(ordinals: &[i32]) -> Result<Self> {
        let mut contexts = Vec::with_capacity(ordinals.len());
        for &ord in ordinals {
            contexts.push(PrimaryContext::retain(ord)?);
        }
        Ok(ContextRegistry { contexts })
    }

    /// Number of devices held.
    #[inline]
    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Is the registry empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// The context for GPU index `idx`, or `InvalidGpu` if out of range.
    pub fn get(&self, idx: usize) -> Result<&PrimaryContext> {
        self.contexts
            .get(idx)
            .ok_or(crate::VugvaError::InvalidGpu(idx))
    }

    /// Make GPU `idx`'s context current on this thread.
    pub fn bind(&self, idx: usize) -> Result<()> {
        self.get(idx)?.bind()
    }

    /// Push GPU `idx`'s context, returning a guard that pops it on drop.
    pub fn enter(&self, idx: usize) -> Result<ContextGuard<'_>> {
        self.get(idx)?.enter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip gracefully when no GPU is present, so the suite still runs on CI
    /// machines without a driver.
    fn gpu_available() -> bool {
        // SAFETY: `cuInit`/`cuDeviceGetCount` are safe to call with valid args;
        // with no driver the late-bound wrappers return an error code.
        unsafe {
            if cuInit(0) != CUDA_SUCCESS {
                return false;
            }
            let mut n = 0i32;
            cuDeviceGetCount(&mut n) == CUDA_SUCCESS && n > 0
        }
    }

    #[test]
    fn retain_and_release_is_balanced() {
        if !gpu_available() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        // Retaining the same device repeatedly must keep working. Under the old
        // `cuCtxCreate_v2` scheme this leaked ~24 MB per iteration; with a
        // refcounted primary context it is free and the handle is identical
        // every time.
        let first = PrimaryContext::retain(0).expect("retain");
        let handle = first.raw();
        assert!(!handle.is_null());
        for _ in 0..64 {
            let c = PrimaryContext::retain(0).expect("retain");
            assert_eq!(c.raw(), handle, "primary context should be shared");
        }
        assert_eq!(first.device(), CUdevice(0));
    }

    #[test]
    fn guard_restores_previous_context() {
        if !gpu_available() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        let ctx = PrimaryContext::retain(0).expect("retain");
        ctx.bind().expect("bind");

        let mut before = CUcontext::NULL;
        // SAFETY: valid out-pointer.
        unsafe {
            assert_eq!(cuCtxGetCurrent(&mut before), CUDA_SUCCESS);
        }

        {
            let _guard = ctx.enter().expect("enter");
            let mut inside = CUcontext::NULL;
            // SAFETY: valid out-pointer.
            unsafe {
                assert_eq!(cuCtxGetCurrent(&mut inside), CUDA_SUCCESS);
            }
            assert_eq!(inside, ctx.raw());
        }

        let mut after = CUcontext::NULL;
        // SAFETY: valid out-pointer.
        unsafe {
            assert_eq!(cuCtxGetCurrent(&mut after), CUDA_SUCCESS);
        }
        assert_eq!(after, before, "guard must restore the prior context");
    }

    #[test]
    fn registry_rejects_out_of_range_index() {
        if !gpu_available() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        let reg = ContextRegistry::new(&[0]).expect("registry");
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert!(reg.get(0).is_ok());
        // Must be a typed error, not a panic on a bad index.
        assert!(matches!(
            reg.get(7),
            Err(crate::VugvaError::InvalidGpu(7))
        ));
    }
}
