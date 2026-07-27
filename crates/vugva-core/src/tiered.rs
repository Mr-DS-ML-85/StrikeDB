//! V2 Three-Tier Hybrid Memory: VRAM → DRAM → SSD.
//!
//! Implements §4 and §5.1 of the paper: a unified three-tier hierarchy
//! with full CPU bypass. The CPU touches only 72 bytes of metadata
//! (8-byte page lookup + 64-byte DMA descriptor) per promotion, while
//! the DMA engine transfers megabytes.
//!
//! ## Page migration state machine (Figure 5)
//!
//! ```text
//! UNMAPPED → ALLOCATED → RESIDENT (VRAM)
//!                        ↕ promote/evict
//!                     WARM (DRAM)
//!                        ↕ spill/load
//!                     COLD (SSD)
//! ```

use crate::dma::DmaEngine;
use crate::ffi::cuda::*;
use crate::gpu::GpuCluster;
use crate::streams::StreamPool;
use crate::vmt::{Chunk, DramChunk, PageState, Tier, VirtualMemoryTable};
use crate::{check_cu, Result, VugvaError};

/// Default idle threshold before demotion: 5 seconds.
const DEFAULT_IDLE_NS: u64 = 5_000_000_000;
/// Access count threshold for proactive promotion.
const HOT_ACCESS_THRESHOLD: u64 = 10;

// ============================================================================
// TieredPool
// ============================================================================

/// The V2 three-tier memory pool: VRAM (hot) → DRAM (warm) → SSD (cold).
///
/// Wraps the VMT, DMA engine, and stream pool to provide the unified
/// access interface described in Algorithm 2 of the paper.
pub struct TieredPool {
    /// Virtual memory table.
    pub vmt: VirtualMemoryTable,
    /// DMA engine (CPU-bypass transfers).
    pub dma: DmaEngine,
    /// GPU cluster topology.
    pub cluster: GpuCluster,
    /// Stream pool for async operations.
    pub streams: StreamPool,
    /// One retained primary context per GPU, in `cluster.ordinals` order.
    ///
    /// Held for the pool's whole lifetime. Every allocation and copy this type
    /// performs runs against these contexts, so they must outlive the device
    /// pointers handed out by [`TieredPool::access`] — a context teardown frees
    /// every allocation made inside it, which would turn every live pointer the
    /// framework holds into a dangling one.
    contexts: crate::context::ContextRegistry,
    /// Per-GPU DRAM pools (host pointers, NUMA-local).
    dram_pools: Vec<DramPool>,
    /// Per-GPU cache of released device blocks, indexed like `cluster.ordinals`.
    ///
    /// `cuMemFree_v2` is not a cheap call: it synchronises the whole device, so
    /// returning every evicted page to the driver would put a full barrier in
    /// the middle of the promotion path this module exists to keep
    /// asynchronous. Blocks are recycled here instead and only handed back to
    /// the driver at teardown.
    ///
    /// Reuse is exact-size. Splitting cached blocks would need a real device
    /// suballocator (a `cuMemAlloc` result is an opaque handle that must be
    /// freed as a unit), and the access path allocates a small number of
    /// repeating tensor sizes, so exact match hits almost always.
    vram_free: Vec<Vec<(usize, u64)>>,
    /// T2, the NVMe cold tier. `None` until [`TieredPool::attach_spill`].
    ///
    /// Optional because the path is the caller's to choose — see
    /// [`crate::spill::SpillFile::create`] for why this module refuses to
    /// invent one. Without it the pool is a two-tier VRAM/DRAM cache and a
    /// corpus larger than the DRAM pool fails to allocate, which is the
    /// capacity cliff the third tier exists to remove.
    spill: Option<crate::spill::SpillFile>,
    /// Promotions started by [`TieredPool::prefetch`] and not yet claimed.
    ///
    /// `name → (gpu_idx, device_ptr, size, completion event)`. This is what
    /// makes the paper's §3.2 look-ahead real rather than advisory: the copy is
    /// issued on the prefetch stream and the CPU returns immediately, so the
    /// transfer overlaps whatever the compute stream is doing. `access` then
    /// finds the work already in flight and only has to wait on the event.
    inflight: std::collections::HashMap<String, (usize, u64, usize, crate::streams::CudaEvent)>,
    /// LGM tier membrane: decayed per-page access rates driving placement.
    membrane: crate::membrane::Membrane,
    /// Scoring policy the membrane ranks with.
    policy: crate::membrane::TierPolicy,
    /// VRAM the membrane may fill, per GPU. `None` disables membrane-driven
    /// sweeps and keeps the original idle-timeout behaviour, which is what
    /// callers that have not opted in should get.
    membrane_budget: Option<usize>,
    /// Idle timeout for demotion (nanoseconds).
    idle_threshold_ns: u64,
    /// Hot-access threshold for proactive promotion.
    hot_threshold: u64,
}

/// A NUMA-local DRAM pool for one GPU.
struct DramPool {
    /// NUMA node this pool is allocated on.
    numa_node: usize,
    /// GPU ordinal this pool serves.
    #[allow(dead_code)]
    gpu_ordinal: i32,
    /// Base host pointer.
    base_ptr: usize,
    /// Total capacity in bytes.
    capacity: usize,
    /// High-water mark of the bump region. Everything below this offset has
    /// been handed out at least once; a sub-range of it may be back on
    /// `free_blocks`.
    offset: usize,
    /// Returned ranges available for reuse, kept sorted by offset and
    /// coalesced on insert.
    ///
    /// Without this the pool was bump-only: `offset` never moved backwards, so
    /// a freed allocation was gone for good and a long-running process that
    /// churned tensors hit `CUDA_ERROR_OUT_OF_MEMORY` with the pool almost
    /// entirely idle. That also made `demote`/`evict` pointless — reclaiming
    /// VRAM by pushing a page down a tier is only a win if the tier below can
    /// hand the space back out (BUG #19).
    free_blocks: Vec<(usize, usize)>,
    /// Did `mbind` actually pin this mapping to `numa_node`?
    ///
    /// `false` means the pool works but may be remote, so its bandwidth will
    /// not match the paper's node-local DRAM figures. Recorded rather than
    /// assumed — the previous code discarded the `mbind` result entirely, so a
    /// pool bound to the wrong node was indistinguishable from a correct one.
    numa_bound: bool,
    /// Did `cuMemHostRegister` succeed, i.e. is this pool page-locked and
    /// therefore genuinely DMA-capable?
    ///
    /// `false` means every H2D copy out of it is really synchronous and
    /// CPU-staged no matter which async API is called. Registration can fail
    /// legitimately (e.g. `RLIMIT_MEMLOCK` too low for the requested size), so
    /// this is reported rather than treated as fatal.
    pinned: bool,
}

impl DramPool {
    /// 64-byte alignment: matches the DMA descriptor's expectation and keeps
    /// every chunk on its own cache line.
    fn align(bytes: usize) -> usize {
        (bytes + 63) & !63
    }

    /// Carve out `bytes`, reusing a freed range when one fits.
    ///
    /// Best-fit rather than first-fit: with the size distribution this pool
    /// sees — a handful of distinct tensor shapes allocated repeatedly —
    /// best-fit tends to land on an exact-size hole left by the previous
    /// instance of the same shape, which leaves the larger holes intact for
    /// larger tensors. First-fit would shred a big block to satisfy a small
    /// request.
    fn allocate(&mut self, bytes: usize) -> Result<usize> {
        let aligned = Self::align(bytes);

        // Reuse before extending.
        let mut best: Option<usize> = None;
        for (i, &(_, len)) in self.free_blocks.iter().enumerate() {
            if len >= aligned && best.is_none_or(|b| len < self.free_blocks[b].1) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let (off, len) = self.free_blocks[i];
            if len == aligned {
                self.free_blocks.remove(i);
            } else {
                // Keep the tail; the block stays sorted because its offset
                // only moves up and still precedes the next block.
                self.free_blocks[i] = (off + aligned, len - aligned);
            }
            return Ok(self.base_ptr + off);
        }

        if self.offset + aligned > self.capacity {
            return Err(VugvaError::DramOom {
                requested: aligned,
                available: self.capacity - self.used(),
                capacity: self.capacity,
            });
        }
        let ptr = self.base_ptr + self.offset;
        self.offset += aligned;
        Ok(ptr)
    }

    /// Return a previously allocated range.
    ///
    /// Coalesces with the neighbours on both sides, and if the result reaches
    /// the high-water mark it retracts `offset` instead of parking a block at
    /// the top. Without that retraction a free/alloc cycle of decreasing sizes
    /// would ratchet `offset` up forever even though the pool never held more
    /// than one allocation at a time.
    ///
    /// Ignores pointers that are not inside this pool: a caller mixing up two
    /// GPUs' pools would otherwise corrupt the free list into handing out
    /// overlapping ranges, which is far worse than the leak.
    fn free(&mut self, ptr: usize, bytes: usize) {
        if ptr < self.base_ptr || ptr >= self.base_ptr + self.capacity {
            return;
        }
        let off = ptr - self.base_ptr;
        let len = Self::align(bytes);
        if len == 0 || off + len > self.offset {
            return;
        }

        let pos = self.free_blocks.partition_point(|&(o, _)| o < off);
        // Overlap with an existing free block means a double free; dropping it
        // is the conservative choice (leaks the range, corrupts nothing).
        if pos < self.free_blocks.len() && off + len > self.free_blocks[pos].0 {
            return;
        }
        if pos > 0 {
            let (po, pl) = self.free_blocks[pos - 1];
            if po + pl > off {
                return;
            }
        }
        self.free_blocks.insert(pos, (off, len));

        // Coalesce right, then left.
        if pos + 1 < self.free_blocks.len() {
            let (no, nl) = self.free_blocks[pos + 1];
            if off + len == no {
                self.free_blocks[pos].1 += nl;
                self.free_blocks.remove(pos + 1);
            }
        }
        let mut idx = pos;
        if pos > 0 {
            let (po, pl) = self.free_blocks[pos - 1];
            if po + pl == off {
                self.free_blocks[pos - 1].1 += self.free_blocks[pos].1;
                self.free_blocks.remove(pos);
                idx = pos - 1;
            }
        }

        // Retract the high-water mark if the top block now touches it.
        let (o, l) = self.free_blocks[idx];
        if o + l == self.offset {
            self.offset = o;
            self.free_blocks.remove(idx);
        }
    }

    /// Bytes currently handed out — the high-water mark minus everything on
    /// the free list. Exposed so tests can assert reuse actually happens
    /// rather than inferring it from an allocation not failing.
    fn used(&self) -> usize {
        self.offset - self.free_blocks.iter().map(|&(_, l)| l).sum::<usize>()
    }
}

impl TieredPool {
    /// Create a new tiered pool for the given GPU ordinals.
    ///
    /// * `dram_pool_size_per_gpu` — bytes of DRAM to reserve per GPU.
    pub fn new(gpu_ordinals: &[i32], dram_pool_size_per_gpu: usize) -> Result<Self> {
        let cluster = GpuCluster::discover(gpu_ordinals)?;
        let contexts = crate::context::ContextRegistry::new(gpu_ordinals)?;
        let num_gpus = gpu_ordinals.len();
        let dma = DmaEngine::new(num_gpus);
        let streams = StreamPool::new(&contexts)?;
        let vmt = VirtualMemoryTable::new(num_gpus, cluster.numa.node_count);

        // Allocate NUMA-local DRAM pools, then page-lock them.
        //
        // Order matters. `mmap` + `mbind` places the pages on the right node
        // but leaves them ordinary pageable memory, and the kernel may relocate
        // pageable pages at any time. The driver therefore cannot DMA out of
        // them: `cuMemcpyHtoDAsync` against pageable memory silently degrades
        // into a *synchronous* staged copy through an internal bounce buffer —
        // the CPU-mediated path this whole module exists to avoid (paper §5.1,
        // "72 bytes of CPU involvement"). It also runs at roughly half the
        // bandwidth of a pinned transfer, which would have shown up as the DRAM
        // tier quietly missing its 28–58 GB/s figure with no visible error.
        //
        // `cuMemHostRegister` page-locks the range we already placed, so we
        // keep NUMA control *and* get real DMA. (`cuMemHostAlloc` would pin but
        // gives no say in placement, so it cannot serve a NUMA-aware pool.)
        //
        // Measured on this machine (RTX 4060, PCIe 4.0 x8), 256 MiB promoted
        // DRAM→VRAM in 16 MiB pages, via `paper_dram_tier_promotes_at_dma_speed`
        // with and without `VUGVA_NO_PIN`:
        //
        // ```text
        // pageable (as before):  48.7 ms =  5.5 GB/s
        // pinned   (registered): 21.2 ms = 12.7 GB/s   → 2.3x
        // ```
        //
        // 12.7 GB/s is close to the practical ceiling of this link, so the
        // remaining gap to the paper's 28–58 GB/s is the interconnect, not the
        // code: that range assumes server-class PCIe 5.0 x16 or NVLink.
        let mut dram_pools = Vec::with_capacity(num_gpus);
        for (idx, &ord) in gpu_ordinals.iter().enumerate() {
            let optimal_node = cluster.optimal_dram_node(ord);
            let (base_ptr, numa_bound) =
                allocate_numa_dram(dram_pool_size_per_gpu, optimal_node)?;

            // PORTABLE so the pinning is valid in every context, not just the
            // one current at registration — other GPUs read this pool during
            // peer promotion.
            let pinned = {
                let _guard = contexts.enter(idx)?;
                // SAFETY: the range was just mmap'd with this exact length and
                // is not registered with the driver yet.
                // `VUGVA_NO_PIN` skips registration. It exists so the benefit
                // above can be *measured* rather than asserted from theory:
                // set it, re-run the bandwidth test, and observe the pool fall
                // back to the staged-copy rate. Nothing in normal operation
                // reads it.
                let rc = if std::env::var_os("VUGVA_NO_PIN").is_some() {
                    CUDA_ERROR_INVALID_VALUE
                } else {
                    // SAFETY: the range was just mmap'd with this exact length
                    // and is not registered with the driver yet.
                    unsafe {
                        cuMemHostRegister(
                            base_ptr as *mut std::ffi::c_void,
                            dram_pool_size_per_gpu,
                            CU_MEMHOSTREGISTER_PORTABLE,
                        )
                    }
                };
                rc == CUDA_SUCCESS
            };

            dram_pools.push(DramPool {
                numa_node: optimal_node,
                gpu_ordinal: ord,
                base_ptr,
                capacity: dram_pool_size_per_gpu,
                offset: 0,
                free_blocks: Vec::new(),
                numa_bound,
                pinned,
            });
        }

        Ok(TieredPool {
            vmt,
            dma,
            cluster,
            streams,
            contexts,
            dram_pools,
            vram_free: vec![Vec::new(); num_gpus],
            spill: None,
            inflight: std::collections::HashMap::new(),
            membrane: crate::membrane::Membrane::new(),
            policy: crate::membrane::TierPolicy::default(),
            membrane_budget: None,
            idle_threshold_ns: DEFAULT_IDLE_NS,
            hot_threshold: HOT_ACCESS_THRESHOLD,
        })
    }

    /// Attach the NVMe cold tier, enabling `Tier::Ssd` allocations.
    ///
    /// Separate from [`TieredPool::new`] because the backing path is a policy
    /// decision this module will not make for the caller: the file can be as
    /// large as the whole corpus, and defaulting it to somewhere like `/tmp`
    /// (a tmpfs on most desktop installs, i.e. RAM) would make the cold tier
    /// consume the exact resource it exists to relieve.
    ///
    /// `unlink_on_drop` deletes the file when the pool goes away. Spilled pages
    /// are a cache of entries the VMT still describes, so they carry no meaning
    /// once that VMT is gone.
    pub fn attach_spill<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        capacity: usize,
        unlink_on_drop: bool,
    ) -> Result<()> {
        self.spill = Some(crate::spill::SpillFile::create(path, capacity, unlink_on_drop)?);
        Ok(())
    }

    /// Whether the cold tier is available.
    pub fn has_spill(&self) -> bool {
        self.spill.is_some()
    }

    /// Bytes currently held in the cold tier.
    pub fn spill_used(&self) -> usize {
        self.spill.as_ref().map_or(0, |s| s.used())
    }

    /// Allocate a new tensor in the unified DRAM+VRAM pool.
    ///
    /// From Algorithm 1:
    /// 1. Allocate DRAM region (NUMA-local to each GPU).
    /// 2. Create VRAM aliases (populated on access via DMA).
    /// 3. CPU never touches tensor data.
    pub fn allocate(
        &mut self,
        name: &str,
        shape: &[usize],
        element_size: usize,
        initial_tier: Tier,
    ) -> Result<String> {
        let total_bytes: usize = shape.iter().product::<usize>() * element_size;
        let alloc_name = self.vmt.allocate(name, shape, element_size)?;

        // A cold allocation is backed by the spill file, not DRAM.
        //
        // This is the difference between a three-tier hierarchy and a two-tier
        // cache. Backing every page with DRAM regardless of `initial_tier` — as
        // this did — means the DRAM pool bounds the *whole corpus* rather than
        // the working set, so a tensor larger than the pool fails with
        // `DramOom` and T2 is only ever reachable later via eviction. The
        // capacity cliff simply moves from VRAM down to DRAM.
        //
        // DRAM chunks are attached on first access instead, as staging for the
        // read (see `access`), which is what makes them a cache.
        if initial_tier == Tier::Ssd {
            let spill = self.spill.as_mut().ok_or_else(|| {
                // Naming the remedy matters: the caller asked for a tier that
                // exists but has no backing store, which is a setup mistake and
                // not an out-of-memory condition.
                VugvaError::UnknownAllocation(format!(
                    "{alloc_name}: Tier::Ssd requires a spill file — call \
                     TieredPool::attach_spill first"
                ))
            })?;
            let offset = spill.reserve(total_bytes)?;
            let page = self.vmt.lookup_mut(&alloc_name).unwrap();
            page.tier = Tier::Ssd;
            page.state = PageState::Cold;
            page.ssd_offset = Some(offset);
            return Ok(alloc_name);
        }

        // Allocate DRAM chunks for each GPU's NUMA-local pool
        {
            let num_pools = self.dram_pools.len();
            let page = self.vmt.lookup_mut(&alloc_name).unwrap();
            page.tier = initial_tier;

            // Integer division drops the remainder, so a 100-byte tensor over
            // 3 GPUs used to be backed by 99 bytes and the last element lived
            // past the end of the last chunk. Give the leftover to the final
            // pool instead.
            let base_bytes = total_bytes / num_pools;
            let remainder = total_bytes % num_pools;
            for (i, pool) in self.dram_pools.iter_mut().enumerate() {
                let chunk_bytes = base_bytes + if i + 1 == num_pools { remainder } else { 0 };
                let host_ptr = pool.allocate(chunk_bytes)?;
                page.dram_chunks.push(DramChunk {
                    numa_node: pool.numa_node,
                    host_ptr,
                    size_bytes: chunk_bytes,
                    cuda_registered: false,
                });
            }
        }

        // Set initial state
        let state = match initial_tier {
            Tier::Vram => PageState::Resident,
            Tier::Dram => PageState::Warm,
            Tier::Ssd => PageState::Cold,
        };
        self.vmt.lookup_mut(&alloc_name).unwrap().state = state;

        Ok(alloc_name)
    }

    /// Access a tensor page, promoting to VRAM if needed.
    ///
    /// Implements Algorithm 2 from the paper:
    /// 1. CPU checks page table (8 bytes — control plane).
    /// 2. If in DRAM, CPU submits DMA descriptor (64 bytes).
    /// 3. DMA engine transfers data autonomously (megabytes).
    /// 4. CPU returns VRAM pointer to framework.
    ///
    /// Total CPU data touched: 72 bytes.
    /// Total data transferred: megabytes.
    pub fn access(&mut self, name: &str, gpu_idx: usize) -> Result<u64> {
        let now = current_time_ns();
        // Every access is a sample for the membrane, including one satisfied
        // from VRAM — a page that is hot *because it is resident* still has to
        // register as hot, or it loses its place at the next sweep and the
        // policy oscillates.
        self.membrane.observe(name);

        // A prefetch may already have moved this page. Claiming it turns the
        // access into a wait on an event that is very likely already signalled,
        // instead of starting a transfer now — which is the entire point of
        // running the look-ahead.
        if let Some(ptr) = self.claim_prefetch(name, gpu_idx, now)? {
            return Ok(ptr);
        }

        // Step 1: Check tier and get source info (immutable borrow).
        //
        // `peer_vram_chunk` is the piece that used to be missing. The old code
        // looked up only *this* GPU's chunk, and the "not resident here" branch
        // then called `.unwrap()` on the very `None` that got it into that
        // branch — a guaranteed panic on the first cross-GPU access (BUG #7).
        // Locating a chunk on some other GPU is what that branch actually needs.
        let (tier, maybe_vram_chunk, peer_vram_chunk, maybe_dram_chunk) = {
            let page = self
                .vmt
                .lookup(name)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            let ordinal = self.cluster.ordinals[gpu_idx];
            let vram_chunk = page
                .vram_chunks
                .iter()
                .find(|c| c.gpu_ordinal == ordinal)
                .cloned();
            let peer_chunk = page
                .vram_chunks
                .iter()
                .find(|c| c.gpu_ordinal != ordinal)
                .cloned();
            // This GPU's own shard, not shard 0. `allocate` gives each pool one
            // chunk in pool order, so index `gpu_idx` is the NUMA-local one —
            // the whole point of placing the pools per GPU. Taking `.first()`
            // meant every GPU promoted GPU 0's shard: wrong data on every GPU
            // but the first, and read across the interconnect to boot.
            let dram_chunk = page.dram_chunks.get(gpu_idx).cloned();
            (page.tier, vram_chunk, peer_chunk, dram_chunk)
        };

        match tier {
            Tier::Vram => {
                if let Some(chunk) = maybe_vram_chunk {
                    // Already hot and resident on this GPU — touch and return
                    self.vmt.lookup_mut(name).unwrap().touch(now);
                    return Ok(chunk.device_ptr);
                }
                // Resident in VRAM, but on another GPU. Copy device→device.
                //
                // The old path bounced the whole tensor through a host
                // `vec![0u8; size]`: two full crossings of PCIe plus a CPU-side
                // copy of megabytes, in a module whose entire premise is that
                // the CPU touches 72 bytes per promotion. It also created and
                // destroyed a context in the middle of the copy, which both
                // leaked ~98 MB and — because the destroy tore down the context
                // the source pointer lived in — could invalidate `src` itself.
                //
                // `cuMemcpyPeerAsync` is the correct primitive: it goes over the
                // P2P link when the pair is peer-enabled, and the driver stages
                // it internally when it is not. Either way the CPU submits a
                // descriptor and nothing more.
                let src = peer_vram_chunk.ok_or_else(|| {
                    // Tier says VRAM but no chunk exists anywhere: the VMT and
                    // the tier field disagree. Report it instead of panicking.
                    VugvaError::UnknownAllocation(name.to_string())
                })?;
                let size = src.size_bytes;
                let src_idx = self
                    .cluster
                    .index_of(src.gpu_ordinal)
                    .ok_or(VugvaError::InvalidGpu(src.gpu_ordinal as usize))?;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                let stream = self.streams.compute[gpu_idx].as_raw();
                // SAFETY: both pointers are live device allocations of `size`
                // bytes in the two retained primary contexts named here, and
                // `stream` belongs to the destination GPU.
                unsafe {
                    check_cu(
                        "cuMemcpyPeerAsync",
                        cuMemcpyPeerAsync(
                            CUdeviceptr(dst_ptr),
                            self.contexts.get(gpu_idx)?.raw(),
                            CUdeviceptr(src.device_ptr),
                            self.contexts.get(src_idx)?.raw(),
                            size,
                            stream,
                        ),
                    )?;
                }
                self.streams.compute[gpu_idx].synchronize()?;

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
            Tier::Dram => {
                // In DRAM — promote via H2D copy (CPU-bypass DMA path)
                let dram = maybe_dram_chunk
                    .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
                let size = dram.size_bytes;
                let src_ptr = dram.host_ptr as u64;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                self.dma
                    .submit_dram_to_vram(gpu_idx, src_ptr, dst_ptr, size, 2)?;

                self.set_context(gpu_idx)?;
                unsafe {
                    crate::ffi::cuda::cuMemcpyHtoDAsync_v2(
                        crate::ffi::cuda::CUdeviceptr(dst_ptr),
                        src_ptr as *const std::ffi::c_void,
                        size,
                        self.streams.compute[gpu_idx].as_raw(),
                    );
                    self.streams.compute[gpu_idx].synchronize()?;
                }

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
            Tier::Ssd => {
                // SSD → DRAM → VRAM, genuinely two-step.
                //
                // The previous version was a stub that read `maybe_dram_chunk`
                // and never touched the file, so a page allocated cold — which
                // has no DRAM chunk at all — failed with `UnknownAllocation`,
                // and a page *demoted* to SSD served whatever stale bytes were
                // still in its DRAM chunk. Either way T2 was decorative.
                //
                // The read lands in page-locked pool memory rather than an
                // ordinary buffer, which is what keeps the second hop a real
                // DMA: `cuMemcpyHtoDAsync` out of pageable memory silently
                // degrades to a synchronous staged copy. That is the whole
                // reason to stage through DRAM instead of reading into a
                // scratch `Vec`.
                let (offset, total) = {
                    let page = self
                        .vmt
                        .lookup(name)
                        .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
                    let off = page.ssd_offset.ok_or_else(|| {
                        VugvaError::UnknownAllocation(format!(
                            "{name}: page is on Tier::Ssd but has no spill offset"
                        ))
                    })?;
                    (off, page.size_bytes)
                };

                // Stage through DRAM in bounded chunks.
                //
                // The page can legitimately be larger than the whole warm tier
                // — that is the case T2 exists for — so demanding `total` bytes
                // of staging would reinstate the DRAM ceiling on the very path
                // meant to escape it. Instead take whatever the pool can spare,
                // and stream: read a chunk from the file, push it to the device,
                // reuse the same buffer.
                //
                // Staging is deliberately *not* attached to the page as a
                // `DramChunk`. It is scratch for this promotion, and recording
                // it would make DRAM look occupied by a page whose home is the
                // file, so eviction would later try to write it back.
                let size = total;
                let dst_ptr = self.alloc_vram_on_gpu(gpu_idx, size)?;

                let (stage_ptr, stage_len, borrowed) = match maybe_dram_chunk {
                    // Page already owns DRAM big enough to hold it: use it.
                    Some(c) if c.size_bytes >= total => (c.host_ptr, total, false),
                    _ => {
                        // Halve until the pool can satisfy it, floor 1 MiB —
                        // below that the per-chunk overhead dominates the copy.
                        let mut want = total;
                        let ptr = loop {
                            match self.dram_pools[gpu_idx].allocate(want) {
                                Ok(p) => break p,
                                Err(e) => {
                                    if want <= (1 << 20) {
                                        // Undo the VRAM reservation; leaving it
                                        // would leak a device block per failure.
                                        let _ = self.free_vram_on_gpu(gpu_idx, dst_ptr, size);
                                        return Err(e);
                                    }
                                    want /= 2;
                                }
                            }
                        };
                        (ptr, want, true)
                    }
                };

                let copy_res = (|| -> Result<()> {
                    let spill = self.spill.as_ref().ok_or_else(|| {
                        VugvaError::UnknownAllocation(format!(
                            "{name}: page is on Tier::Ssd but no spill file is attached"
                        ))
                    })?;
                    self.set_context(gpu_idx)?;
                    let mut done = 0usize;
                    while done < total {
                        let this = stage_len.min(total - done);
                        // SAFETY: `stage_ptr` covers `stage_len` pinned bytes,
                        // and `offset + done` is inside the range reserved for
                        // this page's `size_bytes`.
                        unsafe {
                            spill.read_into(offset + done as u64, stage_ptr as *mut u8, this)?;
                            let rc = crate::ffi::cuda::cuMemcpyHtoD_v2(
                                crate::ffi::cuda::CUdeviceptr(dst_ptr + done as u64),
                                stage_ptr as *const std::ffi::c_void,
                                this,
                            );
                            crate::check_cu("cuMemcpyHtoD_v2", rc)?;
                        }
                        done += this;
                    }
                    Ok(())
                })();

                if borrowed {
                    self.dram_pools[gpu_idx].free(stage_ptr, stage_len);
                }
                if let Err(e) = copy_res {
                    let _ = self.free_vram_on_gpu(gpu_idx, dst_ptr, size);
                    return Err(e);
                }

                // No further copy: the streaming loop above already placed all
                // `total` bytes at `dst_ptr`, and the staging buffer has been
                // returned to the pool. Copying again from `stage_ptr` would
                // read freed memory and overwrite the good data with whatever
                // the pool handed to the next allocation.

                self.vmt.lookup_mut(name).unwrap().touch(now);
                let page = self.vmt.lookup_mut(name).unwrap();
                let elem_size = page.element_size;
                page.vram_chunks.push(Chunk {
                    gpu_ordinal: self.cluster.ordinals[gpu_idx],
                    device_ptr: dst_ptr,
                    size_bytes: size,
                    num_elements: size / elem_size,
                });
                page.tier = Tier::Vram;
                page.state = PageState::Resident;
                Ok(dst_ptr)
            }
        }
    }

    /// Demote a page from VRAM back to DRAM, writing its contents back and
    /// releasing the device memory.
    ///
    /// This is the `RESIDENT → WARM` edge of Figure 5, and it is what makes
    /// VRAM a cache rather than a one-way ratchet: nothing else in the pool
    /// returns device memory while an allocation is still live.
    ///
    /// The previous implementation did none of that (BUG #4):
    ///
    /// * It called `submit_vram_to_vram(gpu_idx, gpu_idx, ..)` — a *self*-copy
    ///   on one GPU — and passed `dram.base_ptr`, a **host** pointer, as the
    ///   device destination.
    /// * That destination was the pool's *base*, not the chunk this page
    ///   actually owns, so every demotion aimed at the same address and would
    ///   have overwritten whichever allocation happens to sit at offset 0.
    /// * It submitted a descriptor and never performed a copy, so the data was
    ///   not written back at all — yet the page was marked `Warm`, i.e. "the
    ///   DRAM copy is authoritative". The next promotion would serve stale
    ///   bytes.
    /// * It never freed the VRAM, so a demotion reclaimed nothing.
    pub fn demote(&mut self, name: &str) -> Result<()> {
        let (tier, pinned, vram, dram) = {
            let page = self
                .vmt
                .lookup(name)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            (
                page.tier,
                page.pinned,
                page.vram_chunks.clone(),
                page.dram_chunks.clone(),
            )
        };

        if tier != Tier::Vram || pinned {
            return Ok(());
        }

        for chunk in &vram {
            let gpu_idx = self
                .cluster
                .index_of(chunk.gpu_ordinal)
                .ok_or(VugvaError::InvalidGpu(chunk.gpu_ordinal as usize))?;

            // Each GPU owns the DRAM shard at its own index — the same pairing
            // `allocate` established, one chunk per pool in pool order.
            let backing = dram
                .get(gpu_idx)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            // Writing more than the shard holds would run off the end of the
            // pool into a neighbouring allocation.
            if chunk.size_bytes > backing.size_bytes {
                return Err(VugvaError::DramOom {
                    requested: chunk.size_bytes,
                    available: backing.size_bytes,
                    capacity: self.dram_pools[gpu_idx].capacity,
                });
            }

            self.dma.submit_vram_to_dram(
                gpu_idx,
                chunk.device_ptr,
                backing.host_ptr as u64,
                chunk.size_bytes,
                0, // cold priority
            )?;

            self.set_context(gpu_idx)?;
            let stream = self.streams.compute[gpu_idx].as_raw();
            // SAFETY: `device_ptr` is a live allocation of `size_bytes` in this
            // GPU's context, `host_ptr` is inside its page-locked DRAM pool and
            // was just checked to be at least that large, and `stream` belongs
            // to this GPU.
            unsafe {
                check_cu(
                    "cuMemcpyDtoHAsync_v2",
                    cuMemcpyDtoHAsync_v2(
                        backing.host_ptr as *mut std::ffi::c_void,
                        CUdeviceptr(chunk.device_ptr),
                        chunk.size_bytes,
                        stream,
                    ),
                )?;
            }
            // The copy must land before the device block goes back on the
            // recycle list, or the next promotion could be handed a block that
            // is still being read.
            self.streams.compute[gpu_idx].synchronize()?;
            self.free_vram_on_gpu(gpu_idx, chunk.device_ptr, chunk.size_bytes)?;
        }

        let page_mut = self.vmt.lookup_mut(name).unwrap();
        page_mut.vram_chunks.clear();
        page_mut.tier = Tier::Dram;
        page_mut.state = PageState::Warm;

        Ok(())
    }

    /// Enable LGM membrane-driven placement with `vram_budget` bytes per GPU.
    ///
    /// Opt-in rather than default, because it is a different contract. The
    /// idle-timeout policy promises "touched recently implies resident"; the
    /// membrane ranks pages *against each other* and fills VRAM to a high-water
    /// mark, so it will evict a page that is still being touched when something
    /// hotter needs the space. Callers depending on the old promise should not
    /// have it changed underneath them.
    pub fn enable_membrane(&mut self, vram_budget: usize) {
        self.membrane_budget = Some(vram_budget);
    }

    /// Decayed access rate for `name` — diagnostics and tests.
    pub fn membrane_rate(&self, name: &str) -> f64 {
        self.membrane.rate(name)
    }

    /// One membrane sweep: decay rates, rank every page, make residence match.
    ///
    /// Order matters. Decay first, so the plan sees this window's traffic.
    /// Evict before promoting, so the space a promotion needs is already free —
    /// otherwise each promotion triggers its own eviction and the transfers
    /// serialise behind one another.
    fn membrane_sweep(&mut self) -> Result<()> {
        let budget = match self.membrane_budget {
            Some(b) => b,
            None => return Ok(()),
        };
        self.membrane.decay();

        let cands: Vec<(String, crate::membrane::PageObs)> = self
            .vmt
            .iter()
            .map(|(name, page)| {
                (
                    name.clone(),
                    crate::membrane::PageObs {
                        rate: self.membrane.rate(name),
                        bytes: page.size_bytes,
                        resident: page.tier == Tier::Vram,
                        pinned: page.pinned,
                    },
                )
            })
            .collect();

        let (keep, evict) = self.membrane.plan(&self.policy, &cands, budget);

        for name in &evict {
            let resident = self
                .vmt
                .lookup(name)
                .map(|p| p.tier == Tier::Vram)
                .unwrap_or(false);
            if resident {
                self.demote(name)?;
            }
        }
        // Promotion failure is not fatal: the page stays warm and correct, and
        // the next sweep retries with whatever space eviction has since freed.
        // Treating it as an error would abort the sweep and leave placement
        // half-applied.
        for name in &keep {
            let resident = self
                .vmt
                .lookup(name)
                .map(|p| p.tier == Tier::Vram)
                .unwrap_or(true);
            if !resident {
                let _ = self.access(name, 0);
            }
        }
        Ok(())
    }

    /// Background sweep: demote idle pages, promote hot DRAM pages.
    ///
    /// Delegates to the membrane when one is enabled.
    pub fn background_sweep(&mut self) -> Result<()> {
        if self.membrane_budget.is_some() {
            return self.membrane_sweep();
        }
        let now = current_time_ns();
        let mut to_promote: Vec<String> = Vec::new();
        let mut to_demote: Vec<String> = Vec::new();

        for (name, page) in self.vmt.iter() {
            if page.state == PageState::Resident && page.is_idle(now, self.idle_threshold_ns) {
                to_demote.push(name.clone());
            }
            if page.tier == Tier::Dram && page.is_hot(self.hot_threshold) {
                to_promote.push(name.clone());
            }
        }

        for name in &to_demote {
            self.demote(name)?;
        }
        for name in &to_promote {
            // Promote to VRAM on GPU 0 (first available)
            self.access(name, 0)?;
        }

        Ok(())
    }

    // --- internal helpers ---

    /// Get `bytes` of device memory on GPU `gpu_idx`, recycling a released
    /// block of exactly that size when one is cached.
    fn alloc_vram_on_gpu(&mut self, gpu_idx: usize, bytes: usize) -> Result<u64> {
        let cache = self
            .vram_free
            .get_mut(gpu_idx)
            .ok_or(VugvaError::InvalidGpu(gpu_idx))?;
        if let Some(i) = cache.iter().position(|&(sz, _)| sz == bytes) {
            let (_, ptr) = cache.swap_remove(i);
            return Ok(ptr);
        }

        self.set_context(gpu_idx)?;
        let mut dptr = CUdeviceptr::NULL;
        // SAFETY: `gpu_idx`'s primary context is current (just bound) and
        // `dptr` is a valid out-pointer.
        unsafe {
            check_cu("cuMemAlloc_v2", cuMemAlloc_v2(&mut dptr, bytes))?;
        }
        Ok(dptr.0)
    }

    /// Release a device block back to GPU `gpu_idx`'s recycle cache.
    ///
    /// The block stays allocated from the driver's point of view; see the
    /// `vram_free` field for why it is not returned to the driver immediately.
    fn free_vram_on_gpu(&mut self, gpu_idx: usize, ptr: u64, bytes: usize) -> Result<()> {
        if ptr == 0 {
            return Ok(());
        }
        self.vram_free
            .get_mut(gpu_idx)
            .ok_or(VugvaError::InvalidGpu(gpu_idx))?
            .push((bytes, ptr));
        Ok(())
    }

    /// Start promoting `name` to `gpu_idx` without waiting for it.
    ///
    /// This is Algorithm 3 (Look-Ahead Prefetch) at the pool level. The copy is
    /// issued on the *prefetch* stream and an event is recorded behind it, so
    /// the call returns as soon as the descriptor is queued and the transfer
    /// runs alongside whatever the compute stream is doing. A later [`access`]
    /// finds the work in flight and only has to wait on the event, which is how
    /// `T_total = max(T_compute(n), T_transport(n+1))` from §3.2 is actually
    /// obtained rather than merely described.
    ///
    /// Only the DRAM→VRAM edge is prefetched. A cold page needs a file read
    /// before any device copy can start, and issuing blocking I/O here would
    /// defeat the purpose — cold pages are promoted synchronously by `access`.
    ///
    /// Idempotent and best-effort: prefetching a resident page, a page already
    /// in flight, or a page whose promotion cannot be started is a no-op rather
    /// than an error. A prefetch that fails must never break the access that
    /// follows it.
    ///
    /// [`access`]: TieredPool::access
    pub fn prefetch(&mut self, name: &str, gpu_idx: usize) -> Result<()> {
        if self.inflight.contains_key(name) {
            return Ok(());
        }
        let (tier, total, host_ptr) = {
            let page = match self.vmt.lookup(name) {
                Some(p) => p,
                None => return Ok(()),
            };
            let hp = page.dram_chunks.first().map(|c| c.host_ptr);
            (page.tier, page.size_bytes, hp)
        };
        if tier != Tier::Dram {
            return Ok(());
        }
        let host_ptr = match host_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        let dst = self.alloc_vram_on_gpu(gpu_idx, total)?;
        self.set_context(gpu_idx)?;
        let event = crate::streams::CudaEvent::new_blocking()?;
        // SAFETY: `dst` is a fresh device block of `total` bytes and `host_ptr`
        // names `total` bytes of the pinned pool.
        unsafe {
            let rc = crate::ffi::cuda::cuMemcpyHtoDAsync_v2(
                crate::ffi::cuda::CUdeviceptr(dst),
                host_ptr as *const std::ffi::c_void,
                total,
                self.streams.prefetch[gpu_idx].as_raw(),
            );
            if rc != 0 {
                // Hand the block back rather than stranding it; the caller is
                // no worse off than if prefetch had not been attempted.
                let _ = self.free_vram_on_gpu(gpu_idx, dst, total);
                return check_cu("cuMemcpyHtoDAsync_v2", rc);
            }
        }
        event.record(&self.streams.prefetch[gpu_idx])?;
        self.inflight
            .insert(name.to_string(), (gpu_idx, dst, total, event));
        Ok(())
    }

    /// Number of prefetches issued and not yet claimed by an `access`.
    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }

    /// Claim a prefetch started earlier, if one matches this page and GPU.
    ///
    /// Returns the device pointer once the transfer has landed. A prefetch for
    /// a *different* GPU is discarded rather than used: the pointer would be
    /// invalid in the requesting context, and silently returning it is exactly
    /// the class of bug that produces plausible-looking garbage.
    fn claim_prefetch(&mut self, name: &str, gpu_idx: usize, now: u64) -> Result<Option<u64>> {
        let Some((pf_gpu, ptr, size, event)) = self.inflight.remove(name) else {
            return Ok(None);
        };
        if pf_gpu != gpu_idx {
            let _ = self.free_vram_on_gpu(pf_gpu, ptr, size);
            return Ok(None);
        }
        event.synchronize()?;
        let page = self.vmt.lookup_mut(name).unwrap();
        let elem_size = page.element_size;
        page.vram_chunks.push(Chunk {
            gpu_ordinal: self.cluster.ordinals[gpu_idx],
            device_ptr: ptr,
            size_bytes: size,
            num_elements: size / elem_size,
        });
        page.tier = Tier::Vram;
        page.state = PageState::Resident;
        page.touch(now);
        Ok(Some(ptr))
    }

    /// Fill a page with `data`, writing to whichever tier currently backs it.
    ///
    /// Allocation reserves space; this is what puts bytes in it. Without it a
    /// page could be created, promoted and read, but never *populated* — so a
    /// corpus could be tiered and would come back as zeros, which is exactly
    /// as useless as not tiering it.
    ///
    /// Where the bytes land depends on the tier, and that asymmetry is the
    /// point of the hierarchy:
    ///
    /// * `Ssd` — straight to the spill file at the page's reserved offset. No
    ///   DRAM is touched, so a corpus far larger than the warm tier can be
    ///   loaded; it streams to NVMe and pages back in on access.
    /// * `Dram` / `Vram` — across the page's DRAM chunks, which `allocate`
    ///   split one-per-pool. The split is not uniform: integer division leaves
    ///   a remainder that goes to the final pool, so the write walks the chunks
    ///   and consumes each one's own `size_bytes` rather than assuming an even
    ///   stride.
    ///
    /// A page already resident in VRAM keeps its device copy *stale* — this
    /// writes the host backing only. Callers that overwrite a hot page should
    /// `demote` it first, so the next `access` re-promotes the new bytes.
    pub fn write_page(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let (tier, total, ssd_offset) = {
            let page = self
                .vmt
                .lookup(name)
                .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
            (page.tier, page.size_bytes, page.ssd_offset)
        };
        if data.len() != total {
            return Err(VugvaError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "write_page({name}): {} bytes for a {total}-byte page — a \
                     partial write would leave the tail undefined",
                    data.len()
                ),
            )));
        }

        if tier == Tier::Ssd {
            let offset = ssd_offset.ok_or_else(|| {
                VugvaError::UnknownAllocation(format!(
                    "{name}: page is on Tier::Ssd but has no spill offset"
                ))
            })?;
            let spill = self.spill.as_ref().ok_or_else(|| {
                VugvaError::UnknownAllocation(format!(
                    "{name}: page is on Tier::Ssd but no spill file is attached"
                ))
            })?;
            // SAFETY: `data` is a live slice of `total` bytes, and the range at
            // `offset` was reserved for exactly `page.size_bytes`.
            unsafe { spill.write_at(data.as_ptr(), total, offset) }?;
            return Ok(());
        }

        let chunks: Vec<(usize, usize)> = {
            let page = self.vmt.lookup(name).unwrap();
            page.dram_chunks
                .iter()
                .map(|c| (c.host_ptr, c.size_bytes))
                .collect()
        };
        let mut written = 0usize;
        for (host_ptr, size) in chunks {
            let end = (written + size).min(total);
            if end <= written {
                break;
            }
            // SAFETY: the chunk covers `size` bytes of the pinned pool, and the
            // source range is inside `data` by the bounds above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(written),
                    host_ptr as *mut u8,
                    end - written,
                );
            }
            written = end;
        }
        if written != total {
            return Err(VugvaError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "write_page({name}): DRAM chunks cover {written} of {total} \
                     bytes — the page's backing is short"
                ),
            )));
        }
        Ok(())
    }

    /// Drop an allocation entirely, returning both its VRAM and its DRAM
    /// backing to the pools it came from.
    ///
    /// Until this existed there was no way to release anything: `allocate`
    /// only ever moved the DRAM bump pointer forward and nothing freed device
    /// memory, so a pool's capacity was a hard *lifetime* budget rather than a
    /// working-set budget (BUG #19).
    pub fn deallocate(&mut self, name: &str) -> Result<()> {
        let page = self
            .vmt
            .lookup(name)
            .ok_or_else(|| VugvaError::UnknownAllocation(name.to_string()))?;
        let vram: Vec<Chunk> = page.vram_chunks.clone();
        let dram: Vec<DramChunk> = page.dram_chunks.clone();

        for chunk in vram {
            if let Some(idx) = self.cluster.index_of(chunk.gpu_ordinal) {
                self.free_vram_on_gpu(idx, chunk.device_ptr, chunk.size_bytes)?;
            }
        }
        // A DRAM chunk carries its NUMA node, not its pool index, and two GPUs
        // can share a node — so offer it to every pool and let the range check
        // inside `DramPool::free` pick the owner. Cheap (one pool per GPU) and
        // immune to the index drift a separately stored pool index invites.
        for chunk in dram {
            for pool in self.dram_pools.iter_mut() {
                pool.free(chunk.host_ptr, chunk.size_bytes);
            }
        }

        self.vmt.remove(name);
        Ok(())
    }

    /// Bytes currently handed out by GPU `gpu_idx`'s DRAM pool.
    pub fn dram_used(&self, gpu_idx: usize) -> Result<usize> {
        self.dram_pools
            .get(gpu_idx)
            .map(|p| p.used())
            .ok_or(VugvaError::InvalidGpu(gpu_idx))
    }

    /// Number of device blocks cached for reuse on GPU `gpu_idx`.
    pub fn vram_cached_blocks(&self, gpu_idx: usize) -> Result<usize> {
        self.vram_free
            .get(gpu_idx)
            .map(|c| c.len())
            .ok_or(VugvaError::InvalidGpu(gpu_idx))
    }

    /// Whether GPU `gpu_idx`'s DRAM pool is really pinned to its NUMA node.
    ///
    /// `false` means `mbind` was refused (no NUMA support, a restrictive
    /// cpuset, or a node that does not exist) and the pool may be remote. Its
    /// bandwidth will then fall short of the paper's node-local DRAM tier,
    /// which is worth surfacing rather than silently absorbing into a
    /// disappointing benchmark.
    pub fn dram_numa_bound(&self, gpu_idx: usize) -> Result<bool> {
        self.dram_pools
            .get(gpu_idx)
            .map(|p| p.numa_bound)
            .ok_or(VugvaError::InvalidGpu(gpu_idx))
    }

    /// Whether GPU `gpu_idx`'s DRAM pool is page-locked.
    ///
    /// When `false`, "async" H2D promotions out of this pool are in fact
    /// synchronous CPU-staged copies — the tier still works, but none of the
    /// paper's CPU-bypass or overlap claims hold for it. Usually means
    /// `RLIMIT_MEMLOCK` is smaller than the requested pool.
    pub fn dram_pinned(&self, gpu_idx: usize) -> Result<bool> {
        self.dram_pools
            .get(gpu_idx)
            .map(|p| p.pinned)
            .ok_or(VugvaError::InvalidGpu(gpu_idx))
    }

    /// Make GPU `gpu_idx`'s primary context current on the calling thread.
    ///
    /// This used to call `cuCtxCreate_v2` — a *new private* context — on every
    /// invocation and never destroy it. Two independent failures:
    ///
    /// * Each context cost 97.8 MB of VRAM on this machine (measured; see
    ///   [`crate::context`]). `access()` calls this at least once, so ~70
    ///   accesses exhausted an 8 GB card before a single tensor was stored.
    /// * A private context's allocations are invisible to the CUDA runtime the
    ///   host framework uses, so the device pointers `access()` returned were
    ///   not addressable by the caller that asked for them.
    ///
    /// The registry holds one retained *primary* context per GPU for the pool's
    /// lifetime, so this is now a pointer store with no allocation at all.
    fn set_context(&self, gpu_idx: usize) -> Result<()> {
        self.contexts.bind(gpu_idx)
    }
}

// SAFETY: every field is either plain data (`vmt`, `dma`, the pool metadata,
// the thresholds) or an opaque CUDA handle whose type carries its own
// `Send`/`Sync` justification (`ContextRegistry`, `CudaStream`). The `usize`
// host pointers name page-locked mappings that outlive the pool and are not
// thread-affine.
//
// `Sync` is sound despite `set_context` mutating driver state through `&self`:
// the current context is *per-thread*: `cuCtxSetCurrent` on one thread is
// invisible to every other. Two threads holding `&TieredPool` therefore cannot
// disturb each other's binding. Everything that mutates the pool's own state —
// `access`, `demote`, `deallocate` — takes `&mut self`, so Rust's own rules
// keep those exclusive.
//
// Without these the type could not cross a thread boundary at all, which is
// what blocked the background sweep and the prefetch thread the paper's
// pipeline depends on (BUG #11).
unsafe impl Send for TieredPool {}
unsafe impl Sync for TieredPool {}

impl Drop for TieredPool {
    fn drop(&mut self) {
        // Hand the recycle cache back to the driver first. These blocks belong
        // to primary contexts that outlive this pool (other libraries share
        // them), so nothing else will reclaim them — dropping a `TieredPool`
        // without this would leak device memory for the process's lifetime.
        for (gpu_idx, cache) in self.vram_free.iter().enumerate() {
            if cache.is_empty() {
                continue;
            }
            // A failure to bind means the context is already gone, in which
            // case its allocations went with it and there is nothing to free.
            let Ok(_guard) = self.contexts.enter(gpu_idx) else {
                continue;
            };
            for &(_, ptr) in cache.iter() {
                // SAFETY: each pointer came from `cuMemAlloc_v2` in this
                // context and has not been freed. Errors are unactionable in a
                // destructor.
                unsafe {
                    let _ = cuMemFree_v2(CUdeviceptr(ptr));
                }
            }
        }

        for pool in &self.dram_pools {
            if pool.base_ptr == 0 {
                continue;
            }
            // Unregister before unmapping. Reversing these leaves the driver
            // holding a pinned reference to an address range the kernel has
            // already handed back — the next mapping to land there would be
            // silently DMA-visible to the GPU.
            if pool.pinned {
                // SAFETY: `base_ptr` is exactly the pointer passed to
                // `cuMemHostRegister`. Errors are unactionable in a destructor.
                unsafe {
                    let _ = cuMemHostUnregister(pool.base_ptr as *mut std::ffi::c_void);
                }
            }
            // SAFETY: base_ptr was returned by mmap with pool.capacity bytes.
            unsafe {
                libc_free(pool.base_ptr as *mut u8, pool.capacity);
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get current monotonic time in nanoseconds.
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Allocate DRAM pinned to a specific NUMA node: `mmap` + the `mbind(2)`
/// syscall.
///
/// Returns the base pointer and whether the NUMA binding actually took effect.
/// The caller records that flag rather than assuming success, because the
/// paper's DRAM-tier bandwidth figures (28–58 GB/s, §4) only hold for a
/// node-local mapping; a silently remote pool reads at a fraction of that and
/// nothing downstream would explain why.
///
/// # Why the raw syscall
///
/// This used to declare `fn mbind(..)` in an `extern "C"` block. glibc does not
/// export `mbind` — it is a **libnuma** symbol:
///
/// ```text
/// $ nm -D --defined-only /usr/lib/x86_64-linux-gnu/libc.so.6 | grep -w mbind
/// (no output)
/// ```
///
/// So that declaration could never resolve. It went unnoticed for the same
/// reason as the NVRTC block (see [`crate::ffi::nvrtc`]): nothing in a test
/// build pulled this function into a linked binary, so the linker was never
/// asked to find the symbol. The first test that constructed a `TieredPool`
/// failed with `undefined symbol: mbind`.
///
/// Linking libnuma is not an option — this crate depends on nothing but libc
/// and the CUDA driver. `mbind` is a plain kernel syscall, so we issue it
/// directly and keep the dependency set empty.
fn allocate_numa_dram(size: usize, numa_node: usize) -> Result<(usize, bool)> {
    use std::ffi::c_void;

    // Round up to page size (4KB)
    let page_size = 4096usize;
    let aligned_size = (size + page_size - 1) & !(page_size - 1);

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: isize,
        ) -> *mut c_void;
        /// Generic syscall gate. glibc exports this unconditionally.
        fn syscall(num: std::ffi::c_long, ...) -> std::ffi::c_long;
    }

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    // SAFETY: a null hint with MAP_ANONYMOUS lets the kernel choose the address;
    // fd is -1 as anonymous mappings require.
    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            aligned_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr.is_null() || ptr as isize == -1 {
        return Err(VugvaError::CudaError {
            fn_name: "mmap",
            code: CUDA_ERROR_OUT_OF_MEMORY,
        });
    }

    // __NR_mbind. Only these two architectures are supported; elsewhere we skip
    // binding rather than issue an arbitrary syscall number.
    #[cfg(target_arch = "x86_64")]
    const SYS_MBIND: std::ffi::c_long = 237;
    #[cfg(target_arch = "aarch64")]
    const SYS_MBIND: std::ffi::c_long = 235;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let bound = {
        const MPOL_BIND: i32 = 1;

        // The old code built the mask as `1u64 << numa_node` against a single
        // u64 with `maxnode = 64` (BUG #23). For `numa_node >= 64` that shift
        // is undefined behaviour in Rust — in release builds it wraps, so node
        // 64 set bit 0 and the pool was bound to the *wrong* node with no error
        // anywhere. Large SGI/AMD systems really do have >64 nodes.
        //
        // Sizing the mask to the node and passing the matching bit count makes
        // any node number expressible and lets the kernel reject a genuinely
        // out-of-range one with EINVAL.
        let words = numa_node / 64 + 1;
        let mut nodemask = vec![0u64; words];
        nodemask[numa_node / 64] = 1u64 << (numa_node % 64);
        let maxnode = (words * 64) as u64;

        // SAFETY: `ptr`/`aligned_size` describe the mapping just created, and
        // `nodemask` holds `maxnode` bits as the kernel's ABI requires.
        let rc = unsafe {
            syscall(
                SYS_MBIND,
                ptr,
                aligned_size,
                MPOL_BIND,
                nodemask.as_ptr(),
                maxnode,
                0u32,
            )
        };
        rc == 0
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let bound = false;

    Ok((ptr as usize, bound))
}

/// Free memory allocated with `mmap`.
///
/// # Safety
///
/// `ptr` must be a pointer previously returned by `mmap`, and `size`
/// must be the exact allocation size.
unsafe fn libc_free(ptr: *mut u8, size: usize) {
    extern "C" {
        fn munmap(addr: *mut std::ffi::c_void, length: usize) -> i32;
    }
    if !ptr.is_null() && size > 0 {
        // SAFETY: ptr is a valid mmap'd region of `size` bytes.
        munmap(ptr as *mut std::ffi::c_void, size);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool over a fake address range. `DramPool`'s allocator is pure
    /// arithmetic on `base_ptr` — it never dereferences anything — so the free
    /// list can be tested without a GPU, a NUMA node, or a real mapping.
    fn pool(capacity: usize) -> DramPool {
        DramPool {
            numa_node: 0,
            gpu_ordinal: 0,
            base_ptr: 0x1000_0000,
            capacity,
            offset: 0,
            free_blocks: Vec::new(),
            numa_bound: false,
            pinned: false,
        }
    }

    /// A compile-time check that the pool can cross a thread boundary.
    ///
    /// The paper's pipeline runs prefetch ahead of compute on a separate
    /// thread; without `Send`/`Sync` on `TieredPool` that design does not
    /// typecheck, so this belongs in the test suite rather than in a comment
    /// (BUG #11).
    #[test]
    fn tiered_pool_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TieredPool>();
        assert_send_sync::<crate::streams::StreamPool>();
        assert_send_sync::<crate::context::ContextRegistry>();
    }

    #[test]
    fn dram_allocate_is_aligned_and_sequential() {
        let mut p = pool(4096);
        let a = p.allocate(1).unwrap();
        let b = p.allocate(1).unwrap();
        assert_eq!(a, 0x1000_0000);
        assert_eq!(b - a, 64, "sub-alignment sizes must still consume a line");
        assert_eq!(p.used(), 128);
    }

    #[test]
    fn dram_free_then_allocate_reuses_the_same_range() {
        let mut p = pool(4096);
        let keep = p.allocate(128).unwrap();
        let a = p.allocate(256).unwrap();
        p.free(a, 256);
        let b = p.allocate(256).unwrap();
        assert_eq!(a, b, "the freed range must be handed back out");
        assert_eq!(p.used(), 384);
        assert_ne!(keep, b);
    }

    #[test]
    fn dram_pool_survives_more_churn_than_it_has_capacity_for() {
        // The exact failure the bump-only allocator had: a pool that never
        // holds more than one allocation still ran out after `capacity/size`
        // cycles.
        let mut p = pool(4096);
        for _ in 0..1000 {
            let ptr = p.allocate(1024).unwrap();
            p.free(ptr, 1024);
        }
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn dram_free_coalesces_adjacent_blocks() {
        let mut p = pool(4096);
        let a = p.allocate(256).unwrap();
        let b = p.allocate(256).unwrap();
        let c = p.allocate(256).unwrap();
        let _tail = p.allocate(256).unwrap(); // keeps the top from retracting

        // Free out of order so the middle block has to merge both ways.
        p.free(a, 256);
        p.free(c, 256);
        assert_eq!(p.free_blocks.len(), 2);
        p.free(b, 256);
        assert_eq!(p.free_blocks.len(), 1, "a+b+c must merge into one block");
        assert_eq!(p.free_blocks[0], (0, 768));

        // And that merged block must be usable as a single large allocation,
        // which is the whole point of coalescing.
        let big = p.allocate(768).unwrap();
        assert_eq!(big, a);
    }

    #[test]
    fn dram_free_retracts_the_high_water_mark() {
        let mut p = pool(4096);
        let a = p.allocate(1024).unwrap();
        assert_eq!(p.offset, 1024);
        p.free(a, 1024);
        assert_eq!(p.offset, 0, "a freed top block must not park on the list");
        assert!(p.free_blocks.is_empty());
    }

    #[test]
    fn dram_free_ignores_foreign_and_double_frees() {
        let mut p = pool(4096);
        let a = p.allocate(256).unwrap();
        let _b = p.allocate(256).unwrap();
        let before = p.used();

        p.free(a - 0x1_0000, 256); // below the pool
        p.free(p.base_ptr + p.capacity + 64, 256); // above the pool
        assert_eq!(p.used(), before, "out-of-range frees must be ignored");

        p.free(a, 256);
        let after_one = p.used();
        p.free(a, 256); // double free
        assert_eq!(
            p.used(),
            after_one,
            "a double free must not put the range on the list twice"
        );

        // The decisive check: the range must be handed out exactly once.
        let x = p.allocate(256).unwrap();
        let y = p.allocate(256).unwrap();
        assert_ne!(x, y, "double free must never alias two live allocations");
    }

    #[test]
    fn dram_best_fit_prefers_the_tightest_hole() {
        let mut p = pool(8192);
        let small = p.allocate(128).unwrap();
        let _g1 = p.allocate(64).unwrap();
        let large = p.allocate(1024).unwrap();
        let _g2 = p.allocate(64).unwrap();
        p.free(small, 128);
        p.free(large, 1024);

        let got = p.allocate(128).unwrap();
        assert_eq!(got, small, "must not shred the 1024-byte hole for 128 bytes");
        // The large hole is therefore still intact.
        assert_eq!(p.allocate(1024).unwrap(), large);
    }

    #[test]
    fn dram_allocate_reports_exhaustion_as_a_host_error() {
        let mut p = pool(256);
        assert!(p.allocate(256).is_ok());
        // Must not masquerade as a *device* OOM — the GPU is not involved and
        // the fixes that error suggests all target the wrong resource.
        match p.allocate(64) {
            Err(VugvaError::DramOom {
                requested,
                available,
                capacity,
            }) => {
                assert_eq!(requested, 64);
                assert_eq!(available, 0);
                assert_eq!(capacity, 256);
            }
            other => panic!("expected DramOom, got {other:?}"),
        }
    }

    #[test]
    fn dram_oom_distinguishes_fragmentation_from_exhaustion() {
        let mut p = pool(1024);
        let a = p.allocate(256).unwrap();
        let _b = p.allocate(256).unwrap();
        let c = p.allocate(256).unwrap();
        let _d = p.allocate(256).unwrap();
        p.free(a, 256);
        p.free(c, 256);

        // 512 bytes free, but in two non-adjacent holes.
        match p.allocate(512) {
            Err(VugvaError::DramOom { available, .. }) => assert_eq!(
                available, 512,
                "a fragmentation failure must report the free bytes, so it is \
                 distinguishable from a genuinely full pool"
            ),
            other => panic!("expected DramOom, got {other:?}"),
        }
    }
}
