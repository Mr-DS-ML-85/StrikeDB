//! CUDA stream management for asynchronous pipelines.
//!
//! Provides a thin safe wrapper around `CUstream` and `CUevent` handles.
//! Each GPU gets a dedicated **prefetch stream** that runs ahead of the
//! compute stream, overlapping DRAM→VRAM DMA transfers with Tensor Core work.

use crate::ffi::cuda::*;
use crate::{check_cu, Result};

// ============================================================================
// Safe stream wrapper
// ============================================================================

/// A managed CUDA stream with automatic cleanup.
#[derive(Debug)]
pub struct CudaStream {
    inner: CUstream,
    /// Priority: higher value = lower priority (CUDA convention).
    /// -1 is the highest priority on most hardware.
    #[allow(dead_code)]
    priority: i32,
}

impl CudaStream {
    /// Create a new stream with default priority.
    pub fn new() -> Result<Self> {
        let mut stream = CUstream::NULL;
        unsafe {
            check_cu(
                "cuStreamCreate",
                cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING),
            )?;
        }
        Ok(CudaStream {
            inner: stream,
            priority: 0,
        })
    }

    /// Create a stream with a specific priority.
    /// `priority` — higher = lower priority. -1 is highest.
    pub fn with_priority(priority: i32) -> Result<Self> {
        let mut stream = CUstream::NULL;
        unsafe {
            check_cu(
                "cuStreamCreateWithPriority",
                cuStreamCreateWithPriority(&mut stream, CU_STREAM_NON_BLOCKING, priority),
            )?;
        }
        Ok(CudaStream {
            inner: stream,
            priority,
        })
    }

    /// Synchronize: block until all work on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cu("cuStreamSynchronize", cuStreamSynchronize(self.inner))?;
        }
        Ok(())
    }

    /// Query whether all work has completed (non-blocking).
    pub fn query_complete(&self) -> Result<bool> {
        unsafe {
            let res = cuStreamQuery(self.inner);
            if res == CUDA_SUCCESS {
                Ok(true)
            } else if res == 900 {
                // CUDA_ERROR_NOT_READY
                Ok(false)
            } else {
                check_cu("cuStreamQuery", res)?;
                unreachable!()
            }
        }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> CUstream {
        self.inner
    }
}

// SAFETY: a `CUstream` is an opaque driver handle, and the CUDA Driver API is
// documented as thread-safe: work may be submitted to one stream from several
// threads, and the driver serialises it internally. The handle is not tied to
// the thread that created it — only to the *context* that was current then,
// which `ContextRegistry` keeps alive for as long as the stream exists.
//
// `Sync` is the meaningful half. Without it nothing that owns a stream —
// `StreamPool`, and therefore `TieredPool` — could be shared or even moved
// across threads, which rules out the background sweep and the prefetch thread
// the paper's pipeline is built on (BUG #11).
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.inner.is_default() {
            unsafe {
                cuStreamDestroy_v2(self.inner);
            }
        }
    }
}

// ============================================================================
// Safe event wrapper
// ============================================================================

/// A managed CUDA event with automatic cleanup.
#[derive(Debug)]
pub struct CudaEvent {
    inner: CUevent,
}

impl CudaEvent {
    /// Create a blocking-sync event (suitable for host-side waits).
    pub fn new_blocking() -> Result<Self> {
        let mut event = CUevent(std::ptr::null_mut());
        unsafe {
            check_cu(
                "cuEventCreate",
                cuEventCreate(&mut event, CU_EVENT_BLOCKING_SYNC),
            )?;
        }
        Ok(CudaEvent { inner: event })
    }

    /// Create a default (non-blocking) event.
    pub fn new_default() -> Result<Self> {
        let mut event = CUevent(std::ptr::null_mut());
        unsafe {
            check_cu("cuEventCreate", cuEventCreate(&mut event, CU_EVENT_DEFAULT))?;
        }
        Ok(CudaEvent { inner: event })
    }

    /// Record this event on the given stream.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cu("cuEventRecord", cuEventRecord(self.inner, stream.as_raw()))?;
        }
        Ok(())
    }

    /// Block the calling thread until this event has been reached.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cu("cuEventSynchronize", cuEventSynchronize(self.inner))?;
        }
        Ok(())
    }

    /// Check if the event has completed (non-blocking).
    pub fn query(&self) -> Result<bool> {
        unsafe {
            let res = cuEventQuery(self.inner);
            if res == CUDA_SUCCESS {
                Ok(true)
            } else if res == 900 {
                // CUDA_ERROR_NOT_READY
                Ok(false)
            } else {
                check_cu("cuEventQuery", res)?;
                unreachable!()
            }
        }
    }

    /// Get elapsed time in milliseconds between two events.
    pub fn elapsed(&self, other: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            check_cu(
                "cuEventElapsedTime",
                cuEventElapsedTime(&mut ms, other.inner, self.inner),
            )?;
        }
        Ok(ms)
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> CUevent {
        self.inner
    }
}

// SAFETY: as for `CudaStream` — an opaque, driver-managed, thread-safe handle.
// Recording an event on one thread and synchronising on it from another is the
// intended cross-thread completion signal.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.inner.0.is_null() {
            unsafe {
                cuEventDestroy_v2(self.inner);
            }
        }
    }
}

// ============================================================================
// Stream pool (one per GPU)
// ============================================================================

/// A set of streams: one compute stream and one prefetch stream per GPU.
pub struct StreamPool {
    /// Compute streams — one per GPU.
    pub compute: Vec<CudaStream>,
    /// Prefetch streams — highest priority, one per GPU.
    pub prefetch: Vec<CudaStream>,
    /// Number of GPUs.
    pub len: usize,
}

impl StreamPool {
    /// Create compute + prefetch stream pairs, one of each per GPU in
    /// `contexts`, each created **inside that GPU's own context**.
    ///
    /// The context matters: a `CUstream` belongs to whichever context was
    /// current when it was created, permanently. The previous signature took a
    /// bare `num_gpus` and created every stream in whatever context happened to
    /// be current, which failed two ways:
    ///
    /// * With no context current — the normal state during `TieredPool::new`,
    ///   since nothing had bound one yet — `cuStreamCreate` returned
    ///   `CUDA_ERROR_INVALID_CONTEXT` (201) and the pool could not be built at
    ///   all. No test caught it because none constructed a pool on a GPU.
    /// * With *some* context current, all 2N streams landed in that one
    ///   context. `compute[3]` would then be a GPU 0 stream, so every "async on
    ///   GPU 3" transfer either failed or silently ran on the wrong device.
    ///
    /// Taking the registry makes the pairing explicit and index-aligned with
    /// `cluster.ordinals`, which is how every caller addresses GPUs.
    pub fn new(contexts: &crate::context::ContextRegistry) -> Result<Self> {
        let num_gpus = contexts.len();
        let mut compute = Vec::with_capacity(num_gpus);
        let mut prefetch = Vec::with_capacity(num_gpus);

        for idx in 0..num_gpus {
            // Guard pops even if a create below fails, so a partial pool does
            // not leave a foreign context current on the caller's thread.
            let _guard = contexts.enter(idx)?;
            compute.push(CudaStream::new()?);
            // Prefetch runs ahead of compute (paper §5.1: transport for layer
            // n+1 overlaps compute for layer n), so it gets the higher
            // priority — but only a priority the device actually supports.
            prefetch.push(CudaStream::with_priority(Self::highest_priority()?)?);
        }

        Ok(StreamPool {
            compute,
            prefetch,
            len: num_gpus,
        })
    }

    /// The most favourable stream priority the current device supports.
    ///
    /// CUDA's convention is inverted — *numerically lower* means higher
    /// priority — and the usable range is device-specific. The old code
    /// hard-coded `-1`, which happens to be valid on most consumer parts but is
    /// out of range on hardware that reports `[0, 0]`, and leaves priority on
    /// the table on hardware whose range is wider than one step.
    fn highest_priority() -> Result<i32> {
        let (mut least, mut greatest) = (0i32, 0i32);
        // SAFETY: two valid out-pointers; a context is current (the caller
        // holds a `ContextGuard`).
        unsafe {
            check_cu(
                "cuCtxGetStreamPriorityRange",
                cuCtxGetStreamPriorityRange(&mut least, &mut greatest),
            )?;
        }
        Ok(greatest)
    }

    /// Synchronize both streams on GPU `idx`.
    pub fn sync(&self, idx: usize) -> Result<()> {
        self.compute[idx].synchronize()?;
        self.prefetch[idx].synchronize()?;
        Ok(())
    }
}
