//! CPU-Bypass DMA engine (V2).
//!
//! Implements §3.3 of the paper:
//!
//! The CPU writes a 64-byte DMA descriptor to a pinned ring buffer;
//! the DMA engine autonomously transfers megabytes of tensor data
//! without further CPU intervention.
//!
//! ```text
//! CPU writes 64B descriptor → DMA engine reads descriptor →
//! transfers MB of data → CPU never touches data-plane traffic
//! ```

use crate::Result;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// DMA descriptor (64 bytes, matches paper §3.3)
// ============================================================================

/// A single DMA transfer descriptor.
///
/// In the paper's design, the CPU writes exactly 64 bytes of metadata
/// per transfer; the DMA engine then moves megabytes autonomously.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct DmaDescriptor {
    /// Source address (DRAM host pointer or GPU device pointer).
    pub src_addr: u64,
    /// Destination address (GPU device pointer).
    pub dst_addr: u64,
    /// Transfer size in bytes.
    pub size: u32,
    /// Source GPU / NUMA node ordinal.
    pub src_ordinal: u32,
    /// Destination GPU ordinal.
    pub dst_ordinal: u32,
    /// Transfer priority: 0=cold, 1=warm, 2=hot.
    pub priority: u32,
    /// Transfer flags. See `DMA_FLAG_*`.
    ///
    /// Direction is not otherwise recoverable from a descriptor: `src_addr`
    /// and `dst_addr` are both bare `u64`, so a host pointer and a device
    /// pointer are indistinguishable once written. A consumer has to be told
    /// which address space each one is in.
    pub flags: u32,
    /// Pad to 64 bytes (the paper specifies 64-byte descriptors).
    _pad: [u8; 28],
}

/// `dst_addr` is host DRAM rather than device memory: this descriptor is a
/// VRAM→DRAM writeback, not a promotion.
pub const DMA_FLAG_WRITEBACK: u32 = 1 << 0;

const DESCRIPTOR_SIZE: usize = std::mem::size_of::<DmaDescriptor>();

/// Verify the descriptor is exactly 64 bytes as the paper specifies.
const _: () = assert!(
    DESCRIPTOR_SIZE == 64,
    "DmaDescriptor must be exactly 64 bytes per paper §3.3"
);

// ============================================================================
// DMA completion state
// ============================================================================

/// Completion status for a batch of DMA operations.
#[derive(Debug)]
pub struct DmaCompletion {
    /// Number of descriptors submitted.
    pub submitted: AtomicU32,
    /// Number of transfers confirmed complete by the DMA engine.
    pub completed: AtomicU32,
}

impl DmaCompletion {
    pub fn new() -> Self {
        DmaCompletion {
            submitted: AtomicU32::new(0),
            completed: AtomicU32::new(0),
        }
    }

    /// Are all submitted transfers complete?
    pub fn is_done(&self) -> bool {
        self.submitted.load(Ordering::Acquire) == self.completed.load(Ordering::Acquire)
    }

    /// Number of transfers still in flight.
    pub fn pending(&self) -> u32 {
        self.submitted
            .load(Ordering::Acquire)
            .saturating_sub(self.completed.load(Ordering::Acquire))
    }
}

impl Default for DmaCompletion {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DMA Command Ring
// ============================================================================

/// The DMA command ring: a lock-free ring buffer of descriptors.
///
/// The CPU (producer) writes descriptors; the DMA engine (consumer)
/// reads them and executes transfers. The CPU never reads/writes
/// tensor data — only 64-byte metadata per transfer.
///
/// This implements the "CPU Bypass Control Plane" described in §2 and §3.3.
pub struct DmaRing {
    /// Descriptor ring in pinned DRAM (1024 slots).
    ring: Vec<DmaDescriptor>,
    /// Consumer index (DMA engine advances this).
    consumer: AtomicU32,
    /// Completion tracking.
    pub completion: DmaCompletion,
    /// Ring capacity (power of 2 for masking).
    capacity: u32,
    /// Bitmask for ring index wrapping.
    mask: u32,
}

impl DmaRing {
    /// Create a new ring buffer with 1024 descriptor slots.
    pub fn new() -> Self {
        let capacity = 1024u32;
        DmaRing {
            ring: (0..capacity as usize)
                .map(|_| DmaDescriptor {
                    src_addr: 0,
                    dst_addr: 0,
                    size: 0,
                    src_ordinal: 0,
                    dst_ordinal: 0,
                    priority: 0,
                    flags: 0,
                    _pad: [0u8; 28],
                })
                .collect(),
            consumer: AtomicU32::new(0),
            completion: DmaCompletion::new(),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Number of pending (unconsumed) descriptors.
    pub fn pending_count(&self) -> u32 {
        let producer = self.completion.submitted.load(Ordering::Acquire);
        let consumer = self.consumer.load(Ordering::Acquire);
        producer.saturating_sub(consumer)
    }

    /// Is the ring full?
    pub fn is_full(&self) -> bool {
        self.pending_count() >= self.capacity
    }

    /// Submit a CPU-bypass transfer descriptor.
    ///
    /// The CPU writes 64 bytes of metadata and rings the doorbell.
    /// The DMA engine picks up the descriptor and transfers data
    /// autonomously. The CPU never touches the actual tensor data.
    ///
    /// # Returns
    ///
    /// `Ok(slot_index)` on success, `Err` if ring is full.
    pub fn submit(&self, desc: &DmaDescriptor) -> Result<u32> {
        if self.is_full() {
            return Err(crate::VugvaError::DmaRingFull);
        }

        let producer = self.completion.submitted.load(Ordering::Acquire);
        let slot = (producer & self.mask) as usize;

        // Write descriptor (CPU writes 64 bytes — control plane only)
        // SAFETY: We checked is_full(), so `slot` is valid and not
        // being concurrently consumed.
        unsafe {
            let ptr = self.ring.as_ptr().add(slot) as *mut DmaDescriptor;
            std::ptr::write_volatile(ptr, desc.clone());
        }

        // Memory barrier: descriptor must be visible to DMA engine
        std::sync::atomic::fence(Ordering::Release);

        // Ring doorbell — advance producer
        self.completion
            .submitted
            .store(producer + 1, Ordering::Release);

        Ok(slot as u32)
    }

    /// Notify that a transfer has completed (called by DMA engine callback).
    pub fn mark_complete(&self) {
        self.completion.completed.fetch_add(1, Ordering::AcqRel);
    }

    /// Peek at the next descriptor without consuming it (for DMA engine).
    pub fn peek(&self) -> Option<&DmaDescriptor> {
        let consumer = self.consumer.load(Ordering::Acquire);
        let producer = self.completion.submitted.load(Ordering::Acquire);
        if consumer >= producer {
            return None;
        }
        let slot = (consumer & self.mask) as usize;
        Some(&self.ring[slot])
    }

    /// Acknowledge that the DMA engine has consumed the next descriptor.
    pub fn ack(&self) {
        self.consumer.fetch_add(1, Ordering::AcqRel);
        self.mark_complete();
    }
}

impl Default for DmaRing {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DMA Engine (high-level)
// ============================================================================

/// The CPU-Bypass DMA Engine: manages rings per GPU and provides
/// the high-level `submit_transfer` API.
///
/// From the paper (§3.3):
/// > The CPU writes a 64-byte DMA descriptor to a pinned ring buffer;
/// > the DMA engine autonomously transfers megabytes of tensor data
/// > without further CPU intervention.
pub struct DmaEngine {
    /// One ring per GPU.
    rings: Vec<DmaRing>,
    /// Number of GPUs.
    num_gpus: usize,
}

impl DmaEngine {
    /// Create DMA rings for the given number of GPUs.
    pub fn new(num_gpus: usize) -> Self {
        DmaEngine {
            rings: (0..num_gpus).map(|_| DmaRing::new()).collect(),
            num_gpus,
        }
    }

    /// Submit a CPU-bypass transfer from DRAM to VRAM.
    ///
    /// This is the core "CPU-bypass" operation: the CPU writes only 64 bytes
    /// of metadata; the DMA engine moves the actual megabytes.
    pub fn submit_dram_to_vram(
        &self,
        dst_gpu: usize,
        src_addr: u64,
        dst_addr: u64,
        size_bytes: usize,
        priority: u32,
    ) -> Result<u32> {
        if dst_gpu >= self.num_gpus {
            return Err(crate::VugvaError::InvalidGpu(dst_gpu));
        }

        let desc = DmaDescriptor {
            src_addr,
            dst_addr,
            size: size_bytes as u32,
            src_ordinal: dst_gpu as u32, // NUMA node = GPU affinity
            dst_ordinal: dst_gpu as u32,
            priority,
            flags: 0,
            _pad: [0u8; 28],
        };

        self.rings[dst_gpu].submit(&desc)
    }

    /// Submit a writeback transfer from VRAM back down to DRAM.
    ///
    /// The reverse of [`DmaEngine::submit_dram_to_vram`], and the descriptor
    /// `demote` needs. Without it, `demote` reached for `submit_vram_to_vram`
    /// and described a *device-to-device* copy on a single GPU whose
    /// destination was a host pointer — a descriptor that names the wrong
    /// address space and the wrong direction, so nothing downstream could ever
    /// execute it correctly (BUG #4).
    pub fn submit_vram_to_dram(
        &self,
        src_gpu: usize,
        src_addr: u64,
        dst_addr: u64,
        size_bytes: usize,
        priority: u32,
    ) -> Result<u32> {
        if src_gpu >= self.num_gpus {
            return Err(crate::VugvaError::InvalidGpu(src_gpu));
        }

        let desc = DmaDescriptor {
            src_addr,
            dst_addr,
            size: size_bytes as u32,
            src_ordinal: src_gpu as u32,
            // Destination is host DRAM on this GPU's NUMA node, mirroring the
            // convention `submit_dram_to_vram` uses for its source.
            dst_ordinal: src_gpu as u32,
            priority,
            flags: DMA_FLAG_WRITEBACK,
            _pad: [0u8; 28],
        };

        self.rings[src_gpu].submit(&desc)
    }

    /// Submit a peer-to-peer VRAM→VRAM transfer (CPU-bypass).
    pub fn submit_vram_to_vram(
        &self,
        src_gpu: usize,
        dst_gpu: usize,
        src_addr: u64,
        dst_addr: u64,
        size_bytes: usize,
        priority: u32,
    ) -> Result<u32> {
        if src_gpu >= self.num_gpus || dst_gpu >= self.num_gpus {
            return Err(crate::VugvaError::InvalidGpu(src_gpu.max(dst_gpu)));
        }

        let desc = DmaDescriptor {
            src_addr,
            dst_addr,
            size: size_bytes as u32,
            src_ordinal: src_gpu as u32,
            dst_ordinal: dst_gpu as u32,
            priority,
            flags: 0,
            _pad: [0u8; 28],
        };

        self.rings[dst_gpu].submit(&desc)
    }

    /// Check if all DMA operations on GPU `idx` are complete.
    pub fn is_idle(&self, gpu_idx: usize) -> bool {
        if gpu_idx >= self.num_gpus {
            return true;
        }
        self.rings[gpu_idx].completion.is_done()
    }

    /// Pending transfers for GPU `idx`.
    pub fn pending(&self, gpu_idx: usize) -> u32 {
        if gpu_idx >= self.num_gpus {
            return 0;
        }
        self.rings[gpu_idx].completion.pending()
    }

    /// Total pending across all GPUs.
    pub fn total_pending(&self) -> u32 {
        self.rings.iter().map(|r| r.completion.pending()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_64_bytes() {
        assert_eq!(std::mem::size_of::<DmaDescriptor>(), 64);
    }

    #[test]
    fn dma_ring_submit_and_consume() {
        let ring = DmaRing::new();
        assert_eq!(ring.pending_count(), 0);

        let desc = DmaDescriptor {
            src_addr: 0xDEAD,
            dst_addr: 0xBEEF,
            size: 4096,
            src_ordinal: 0,
            dst_ordinal: 1,
            priority: 2,
            flags: 0,
            _pad: [0u8; 28],
        };

        let slot = ring.submit(&desc).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(ring.pending_count(), 1);

        // Peek at the descriptor
        let peeked = ring.peek().unwrap();
        assert_eq!(peeked.src_addr, 0xDEAD);
        assert_eq!(peeked.dst_addr, 0xBEEF);
        assert_eq!(peeked.size, 4096);

        // Acknowledge
        ring.ack();
        assert_eq!(ring.pending_count(), 0);
    }

    #[test]
    fn dma_ring_full_detection() {
        let ring = DmaRing::new();
        let desc = DmaDescriptor {
            src_addr: 0,
            dst_addr: 0,
            size: 0,
            src_ordinal: 0,
            dst_ordinal: 0,
            priority: 0,
            flags: 0,
            _pad: [0u8; 28],
        };

        // Fill the ring
        for _ in 0..1024 {
            ring.submit(&desc).unwrap();
        }
        assert!(ring.is_full());

        // Next submit should fail
        let result = ring.submit(&desc);
        assert!(result.is_err());
    }

    #[test]
    fn dma_ring_wrap_around() {
        let ring = DmaRing::new();
        let desc = DmaDescriptor {
            src_addr: 42,
            dst_addr: 99,
            size: 128,
            src_ordinal: 0,
            dst_ordinal: 0,
            priority: 1,
            flags: 0,
            _pad: [0u8; 28],
        };

        // Fill and drain the ring multiple times
        for _ in 0..5 {
            for _ in 0..1024 {
                ring.submit(&desc).unwrap();
            }
            for _ in 0..1024 {
                ring.ack();
            }
        }
        assert_eq!(ring.pending_count(), 0);
    }

    #[test]
    fn dma_completion_tracking() {
        let completion = DmaCompletion::new();
        assert!(completion.is_done());
        assert_eq!(completion.pending(), 0);

        completion.submitted.fetch_add(3, Ordering::SeqCst);
        assert!(!completion.is_done());
        assert_eq!(completion.pending(), 3);

        completion.completed.fetch_add(2, Ordering::SeqCst);
        assert!(!completion.is_done());
        assert_eq!(completion.pending(), 1);

        completion.completed.fetch_add(1, Ordering::SeqCst);
        assert!(completion.is_done());
        assert_eq!(completion.pending(), 0);
    }

    #[test]
    fn dma_engine_submit_and_pending() {
        let engine = DmaEngine::new(2);
        assert!(engine.is_idle(0));
        assert!(engine.is_idle(1));

        let addr = 0x1000_u64;
        engine
            .submit_dram_to_vram(0, addr, addr + 0x1000, 4096, 2)
            .unwrap();
        assert!(!engine.is_idle(0));
        assert_eq!(engine.total_pending(), 1);
    }
}
