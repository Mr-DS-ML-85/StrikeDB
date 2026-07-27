//! Look-Ahead Attention Tracking engine.
//!
//! Implements Algorithm 3 from the paper (§3.2, §5.2):
//!
//! While Tensor Cores compute layers L through L+K on GPU G_c, the
//! prefetcher spawns background CUDA memcpy operations over isolated
//! streams to copy weight matrices for layers L+K+1 through L+2K
//! into reserved local cache buffers.
//!
//! ```text
//! Compute:    [Layer 1] [Layer 2] [Layer 3] [Layer 4] ...
//! Prefetch:            [Layer 5] [Layer 6] [Layer 7] [Layer 8] ...
//!              ↑ overlap hidden behind compute ↑
//! ```

use crate::streams::CudaEvent;
use crate::vmt::Tier;
use crate::Result;

// ============================================================================
// Layer schedule entry
// ============================================================================

/// Describes where a layer's weights currently reside.
#[derive(Debug, Clone)]
pub struct LayerLocation {
    /// Which GPU ordinal holds the weights.
    pub gpu_ordinal: i32,
    /// Which tier the weights are in.
    pub tier: Tier,
    /// VMT name for the weights.
    pub name: String,
}

/// The global schedule: maps layer index → location info.
pub type LayerSchedule = Vec<LayerLocation>;

// ============================================================================
// Prefetch command
// ============================================================================

/// An in-flight prefetch operation.
struct PrefetchJob {
    /// Name of the weight being prefetched.
    #[allow(dead_code)]
    pub name: String,
    /// Event that fires when the transfer completes.
    pub event: CudaEvent,
    /// Destination GPU ordinal.
    #[allow(dead_code)]
    pub dst_gpu: i32,
}

// ============================================================================
// Look-Ahead Prefetcher
// ============================================================================

/// The Look-Ahead Attention Tracking engine.
///
/// Runs an independent execution path K layers ahead of the main compute
/// thread, hiding PCIe transport latencies behind active Tensor Core cycles.
pub struct LookAheadPrefetcher {
    /// How many layers ahead to prefetch.
    depth: usize,
    /// In-flight prefetch jobs.
    inflight: Vec<PrefetchJob>,
}

impl LookAheadPrefetcher {
    /// Create a prefetcher with the given lookahead depth K.
    pub fn new(depth: usize) -> Self {
        LookAheadPrefetcher {
            depth,
            inflight: Vec::with_capacity(depth * 2),
        }
    }

    // NOTE: `prefetch_ahead` was removed here.
    //
    // It took a layer schedule and dispatched per tier, but only its
    // VRAM->VRAM peer-copy arm did anything: the `Dram` and `Ssd` arms were
    // empty bodies carrying the comment "here we record the intent". Read
    // casually the function looked like the paper's §3.2 prefetcher; it was a
    // shell around one third of it, with zero callers anywhere in the tree.
    //
    // `TieredPool::prefetch` supersedes it and is the real implementation:
    // it issues the DRAM->VRAM copy on the prefetch stream, records an event,
    // and `access` claims it. Measured at 3.2x on the claim (8 x 32 MiB: issue
    // 7.47 ms + claim 12.69 ms against 40.81 ms cold), covered by
    // `paper_prefetch_overlaps_transport_with_compute`.
    //
    // Cold T2 pages are still promoted synchronously — a file read has to
    // complete before any device copy can start, and blocking the prefetch
    // stream on I/O defeats the purpose.

    /// Block until all in-flight prefetches complete.
    pub fn sync_all(&mut self) -> Result<()> {
        for job in self.inflight.drain(..) {
            job.event.synchronize()?;
        }
        Ok(())
    }

    /// Number of in-flight prefetch operations.
    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }
}
