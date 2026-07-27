//! # VUGVA — Virtual Unified GPU VRAM Architecture
//!
//! Software-defined GPU memory virtualization with CPU-bypass hybrid DRAM/VRAM tiering.
//! Pure-stdlib Rust with raw FFI to `libcuda.so` and `libnvrtc.so`. Zero external crates.
//!
//! ## Architecture (paper)
//!
//! - [`vmt`]          — Virtual Memory Table: maps virtual pointers → GPU chunks
//! - [`allocator`]    — V1 unified allocator: multi-GPU VRAM pool
//! - [`tiered`]       — V2 three-tier: VRAM → DRAM → SSD with full CPU bypass
//! - [`dma`]          — CPU-Bypass DMA engine (GPUDirect + IOMMU P2P)
//! - [`prefetch`]     — Look-Ahead Attention Tracking (predictive prefetch)
//! - [`gpu`]          — GPU device + peer-access discovery via NUMA mapping
//! - [`membrane`]     — LGM tier membrane: frequency-driven T0/T1/T2 placement
//! - [`streams`]      — CUDA stream management for async pipelines
//! - [`nvrtc_kernel`] — NVRTC runtime CUDA kernel compilation
//! - [`ffi::cuda`]    — Raw FFI to libcuda.so (CUDA Driver API)
//! - [`ffi::nvrtc`]   — Raw FFI to libnvrtc.so (runtime compiler)
//!
//! ## Integration
//!
//! ```ignore
//! // 1. Initialize the engine
//! let engine = vugva::VugvaEngine::builder()
//!     .gpus(&[0, 1, 2, 3, 4, 5, 6, 7])
//!     .build()?;
//!
//! // 2. Allocate unified tensor (spans all GPUs)
//! let name = engine.allocate("model.embed.weight", &[8192, 8192], 2)?;
//!
//! // 3. Access (auto-promotes via peer copy if needed)
//! let ptr = engine.access(&name, 0)?; // GPU 0
//! ```

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod allocator;
pub mod context;
pub mod dma;
pub mod ffi;
pub mod gpu;
pub mod membrane;
pub mod nvrtc_kernel;
pub mod prefetch;
pub mod range_alloc;
pub mod spill;
pub mod streams;
pub mod tiered;
pub mod vmt;

/// Engine version constants.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Library name passed to `dlopen` for the CUDA driver.
pub const CUDA_SONAME: &str = "libcuda.so.1";
/// Library name passed to `dlopen` for NVRTC (primary guess).
pub const NVRTC_SONAME: &str = "libnvrtc.so.12";

/// Top-level error type for all VUGVA operations.
#[derive(Debug)]
pub enum VugvaError {
    /// The CUDA Driver API call returned a non-success CUresult.
    CudaError {
        /// The function name that failed.
        fn_name: &'static str,
        /// The CUresult code.
        code: i32,
    },
    /// The NVRTC API returned a non-success nvrtcResult.
    NvrtcError {
        /// The function name that failed.
        fn_name: &'static str,
        /// The nvrtcResult code.
        code: i32,
    },
    /// Failed to `dlopen` the CUDA or NVRTC shared library.
    LibLoad {
        /// Which library failed (`libcuda.so.1` or `libnvrtc.so.*`).
        library: &'static str,
        /// OS error from `dlerror`.
        os_error: i32,
    },
    /// Tried to operate on a GPU ordinal out of range.
    InvalidGpu(usize),
    /// VMT lookup failed: no allocation registered under that name.
    UnknownAllocation(String),
    /// Underlying `std::io::Error`.
    Io(std::io::Error),
    /// Invalid page state transition (e.g. Resident → Allocated).
    InvalidTransition {
        /// Source state.
        from: vmt::PageState,
        /// Target state.
        to: vmt::PageState,
        /// Page name.
        page: String,
    },
    /// DMA command ring is full (all 1024 slots occupied).
    DmaRingFull,
    /// A host DRAM pool could not satisfy a request.
    ///
    /// Distinct from `CudaError { code: CUDA_ERROR_OUT_OF_MEMORY }`, which is
    /// what this used to report. That was actively misleading: it named a
    /// *device* OOM for a failure that has nothing to do with the GPU, so the
    /// obvious remedies — shrink the batch, free something off the card — all
    /// address the wrong resource. Carrying the sizes also separates the two
    /// real causes: `requested > capacity` needs a bigger pool, while a small
    /// request failing with a large `available` is fragmentation.
    DramOom {
        /// Bytes requested, after alignment.
        requested: usize,
        /// Bytes not currently handed out anywhere in the pool.
        available: usize,
        /// Total size of the pool.
        capacity: usize,
    },
}

impl std::fmt::Display for VugvaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VugvaError::CudaError { fn_name, code } => {
                write!(f, "CUDA error in {fn_name}: code {code}")
            }
            VugvaError::NvrtcError { fn_name, code } => {
                write!(f, "NVRTC error in {fn_name}: code {code}")
            }
            VugvaError::LibLoad { library, os_error } => {
                write!(f, "failed to dlopen {library}: OS error {os_error}")
            }
            VugvaError::InvalidGpu(g) => write!(f, "invalid GPU ordinal: {g}"),
            VugvaError::UnknownAllocation(n) => {
                write!(f, "no allocation registered under {n:?}")
            }
            VugvaError::Io(e) => write!(f, "io error: {e}"),
            VugvaError::InvalidTransition { from, to, page } => {
                write!(f, "invalid page transition {from:?} → {to:?} on {page:?}")
            }
            VugvaError::DmaRingFull => write!(f, "DMA command ring is full"),
            VugvaError::DramOom {
                requested,
                available,
                capacity,
            } => write!(
                f,
                "host DRAM pool exhausted: requested {requested} bytes, \
                 {available} free of {capacity} total"
            ),
        }
    }
}

impl std::error::Error for VugvaError {}

impl From<std::io::Error> for VugvaError {
    fn from(e: std::io::Error) -> Self {
        VugvaError::Io(e)
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, VugvaError>;

/// Convert a `CUresult` (i32) into a `Result` with the caller's function name.
#[inline]
pub(crate) fn check_cu(call: &'static str, code: i32) -> Result<()> {
    if code == 0
    /* CUresult::CUDA_SUCCESS */
    {
        Ok(())
    } else {
        Err(VugvaError::CudaError {
            fn_name: call,
            code,
        })
    }
}

/// Convert an `nvrtcResult` (i32) into a `Result`.
#[inline]
pub(crate) fn check_nvrtc(call: &'static str, code: i32) -> Result<()> {
    if code == 0
    /* nvrtcResult::NVRTC_SUCCESS */
    {
        Ok(())
    } else {
        Err(VugvaError::NvrtcError {
            fn_name: call,
            code,
        })
    }
}
