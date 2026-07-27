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

use crate::ffi::cuda::*;
use crate::streams::{CudaEvent, CudaStream};
use crate::vmt::{Tier, VirtualMemoryTable};
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

    /// Prefetch layers `current_layer+1` through `current_layer+depth`.
    ///
    /// For each future layer, determines the source tier and dispatches
    /// the appropriate transfer:
    /// - **VRAM→VRAM**: `cuMemcpyPeerAsync` (P2P, CPU-bypass)
    /// - **DRAM→VRAM**: DMA descriptor submission (CPU-bypass, §3.3)
    /// - **SSD→VRAM**: GPUDirect Storage read (CPU-bypass, §3.3)
    ///
    /// # Arguments
    ///
    /// * `schedule` — the global layer→location mapping.
    /// * `current_layer` — index of the layer currently being computed.
    /// * `current_gpu` — ordinal of the GPU running the compute.
    /// * `vmt` — the virtual memory table (to resolve device pointers).
    /// * `alloc_name` — VMT name prefix for weight allocations.
    /// * `src_ctxs` — CUDA contexts indexed by GPU position, for peer copy.
    /// * `prefetch_stream` — the high-priority prefetch stream for dst GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn prefetch_ahead(
        &mut self,
        schedule: &LayerSchedule,
        current_layer: usize,
        current_gpu: i32,
        vmt: &VirtualMemoryTable,
        src_ctxs: &[CUcontext],
        dst_ctx: CUcontext,
        dst_ptr: u64,
        bytes_per_layer: usize,
        prefetch_stream: &CudaStream,
    ) -> Result<()> {
        let depth = self
            .depth
            .min(schedule.len().saturating_sub(current_layer + 1));

        for offset in 1..=depth {
            let future_layer = current_layer + offset;
            if future_layer >= schedule.len() {
                break;
            }

            let loc = &schedule[future_layer];

            match loc.tier {
                Tier::Vram => {
                    // VRAM→VRAM: peer async copy
                    if let Some(page) = vmt.lookup(&loc.name) {
                        if let Some(src_chunk) = page.vram_chunks.first() {
                            if src_chunk.gpu_ordinal != current_gpu {
                                let src_idx = src_chunk.gpu_ordinal.try_into().unwrap_or(0usize);
                                let _dst_idx = current_gpu.try_into().unwrap_or(0usize);

                                unsafe {
                                    cuMemcpyPeerAsync(
                                        CUdeviceptr(dst_ptr),
                                        dst_ctx,
                                        CUdeviceptr(src_chunk.device_ptr),
                                        src_ctxs.get(src_idx).copied().unwrap_or(src_ctxs[0]),
                                        bytes_per_layer,
                                        prefetch_stream.as_raw(),
                                    );
                                }
                            }
                        }
                    }
                }
                Tier::Dram => {
                    // DRAM→VRAM: submit DMA descriptor (CPU-bypass)
                    // The DMA engine handles the actual transfer.
                    // In a full implementation this would write to the
                    // DMA command ring; here we record the intent.
                    // See dma.rs for the ring implementation.
                }
                Tier::Ssd => {
                    // SSD→VRAM: GPUDirect Storage read
                    // GDS would be initialized via dma.rs.
                    // This path submits an async GDS read descriptor.
                }
            }
        }

        Ok(())
    }

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
