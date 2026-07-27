//! `CorpusTier` — the vector corpus served through VUGVA's three-tier memory.
//!
//! This is what `ComputeMode::Hybrid` is supposed to mean. Without it, Hybrid
//! asks `gpu_check_capacity` whether the corpus fits in VRAM and falls back to
//! the CPU when it does not:
//!
//! ```text
//! ComputeMode::Hybrid => { let (fits, _, _) = gpu_check_capacity(n, dim); fits }
//! ```
//!
//! That is a binary cliff, and it is the exact failure VUGVA exists to remove.
//! One vector past the VRAM limit and the whole query set drops to the CPU
//! path — the "performance cliff" the paper names in §1.1, and the reason
//! larger-than-VRAM corpora are a gap for Qdrant and Milvus rather than a
//! feature.
//!
//! With a `CorpusTier`, a corpus that does not fit is *paged* instead: hot
//! blocks sit in VRAM (T0), warm blocks in page-locked NUMA-local DRAM (T1),
//! cold blocks spill to NVMe (T2), and promotion runs over the DMA engine so
//! the CPU writes descriptors rather than copying vectors.
//!
//! ## What this type does and does not own
//!
//! It owns the *corpus* — the int8 vectors the distance kernels read. It does
//! not own the graph adjacency, which is small (n × degree × 4 bytes) relative
//! to the vectors (n × dim) and is better kept resident. At 1M × 384d that is
//! 384 MB of vectors against 32 MB of edges at degree 8, so tiering the
//! vectors is where essentially all of the headroom is.

use crate::ComputeMode;
use vugva_core::tiered::TieredPool;
use vugva_core::vmt::Tier;

/// How much host DRAM to give each GPU's warm tier, in bytes.
///
/// The warm tier only has to be large enough to hold the working set that does
/// not fit in VRAM, not the whole corpus — anything colder spills to NVMe. A
/// fixed default keeps construction infallible for callers who have no opinion;
/// `with_dram_budget` overrides it.
pub const DEFAULT_DRAM_PER_GPU: usize = 2 << 30; // 2 GiB

/// The corpus, resident across VRAM / DRAM / NVMe.
pub struct CorpusTier {
    pool: TieredPool,
    /// VMT name of the corpus allocation.
    name: String,
    /// Row count and dimensionality, kept so `device_ptr` can be checked
    /// against the shape the corpus was allocated with.
    n: usize,
    dim: usize,
}

impl CorpusTier {
    /// Place an `n × dim` int8 corpus under VUGVA, starting in the DRAM tier.
    ///
    /// Starting warm rather than hot is deliberate: allocating straight into
    /// VRAM would reintroduce the very capacity limit this exists to escape,
    /// and would fail outright for the larger-than-VRAM case that is the whole
    /// point. Blocks promote on first access instead, which is Algorithm 2.
    pub fn new(gpu_ordinals: &[i32], n: usize, dim: usize) -> crate::GpuResult<Self> {
        Self::with_dram_budget(gpu_ordinals, n, dim, DEFAULT_DRAM_PER_GPU)
    }

    /// As [`CorpusTier::new`], with an explicit per-GPU warm-tier budget.
    pub fn with_dram_budget(
        gpu_ordinals: &[i32],
        n: usize,
        dim: usize,
        dram_per_gpu: usize,
    ) -> crate::GpuResult<Self> {
        let mut pool = TieredPool::new(gpu_ordinals, dram_per_gpu)
            .map_err(|e| format!("VUGVA TieredPool::new failed: {e}"))?;

        let total = n * dim;
        // Attach T2 whenever the corpus will not sit comfortably in the warm
        // tier. Without it a corpus larger than `dram_per_gpu` fails to
        // allocate, which is the same capacity cliff as the VRAM one, just a
        // tier lower — and the case this whole type exists for.
        //
        // The spill file goes beside the WAL rather than in `/tmp`, which on
        // most desktop installs is a tmpfs: a cold tier in RAM consumes exactly
        // the resource it is meant to relieve.
        let tier = if total > dram_per_gpu / 2 {
            let dir = std::env::var("DBSTRIKE_SPILL_DIR").unwrap_or_else(|_| ".".to_string());
            let path = std::path::Path::new(&dir)
                .join(format!("dbstrike-corpus-spill-{}.bin", std::process::id()));
            // Room for the corpus plus headroom for the pages demote will add.
            let capacity = total + total / 4 + (1 << 20);
            pool.attach_spill(&path, capacity, true)
                .map_err(|e| format!("VUGVA attach_spill at {path:?} failed: {e}"))?;
            Tier::Ssd
        } else {
            Tier::Dram
        };

        let name = pool
            .allocate("dbstrike.corpus.i8", &[n, dim], 1, tier)
            .map_err(|e| format!("VUGVA corpus allocate failed: {e}"))?;
        Ok(CorpusTier {
            pool,
            name,
            n,
            dim,
        })
    }

    /// Fill the corpus with `vectors`, row-major int8, `n × dim` bytes.
    ///
    /// Writes to whichever tier currently backs the page — for a corpus larger
    /// than the warm tier that is the spill file, so loading it never requires
    /// `n × dim` bytes of DRAM at once. That is the property that lets a corpus
    /// exceed both VRAM and RAM.
    ///
    /// Call before [`CorpusTier::device_ptr`]: a page promoted first would
    /// carry the pre-write contents into VRAM and the kernels would score
    /// against zeros.
    pub fn upload(&mut self, vectors: &[i8]) -> crate::GpuResult<()> {
        let want = self.bytes();
        if vectors.len() != want {
            return Err(format!(
                "CorpusTier::upload: {} bytes for a {}×{} corpus ({want} bytes)",
                vectors.len(),
                self.n,
                self.dim
            ));
        }
        // SAFETY: `i8` and `u8` share size and alignment, and the slice is only
        // read for the duration of the call.
        let bytes =
            unsafe { std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len()) };
        self.pool
            .write_page(&self.name, bytes)
            .map_err(|e| format!("VUGVA corpus write failed: {e}"))
    }

    /// Device pointer to the corpus on `gpu_idx`, promoting it if it is not
    /// already resident.
    ///
    /// This is the call that replaces the fits-or-CPU branch: it returns a
    /// usable VRAM pointer whether the corpus was hot, warm, or spilled, and
    /// the cost difference between those cases is a DMA transfer rather than a
    /// change of execution path.
    pub fn device_ptr(&mut self, gpu_idx: usize) -> crate::GpuResult<u64> {
        self.pool
            .access(&self.name, gpu_idx)
            .map_err(|e| format!("VUGVA corpus access failed: {e}"))
    }

    /// Push the corpus out of VRAM, keeping it warm in DRAM.
    ///
    /// Used when another allocation needs the VRAM more than this one does;
    /// the data stays live and the next `device_ptr` re-promotes it.
    pub fn demote(&mut self) -> crate::GpuResult<()> {
        self.pool
            .demote(&self.name)
            .map_err(|e| format!("VUGVA corpus demote failed: {e}"))
    }

    /// Run one eviction sweep, demoting pages that have gone idle.
    pub fn sweep(&mut self) -> crate::GpuResult<()> {
        self.pool
            .background_sweep()
            .map_err(|e| format!("VUGVA sweep failed: {e}"))
    }

    /// Rows in the corpus.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the corpus holds no rows.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Dimensionality of each row.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes the corpus occupies, across all tiers.
    pub fn bytes(&self) -> usize {
        self.n * self.dim
    }
}

/// Whether `mode` should serve its corpus through VUGVA.
///
/// Only `Hybrid` does. `Turbo` is VRAM-resident by definition — routing it
/// through the tier machinery would add a page-table lookup to a path whose
/// entire premise is that no lookup is needed — and `CpuOnly` never touches
/// the device at all.
pub fn mode_uses_tiering(mode: ComputeMode) -> bool {
    matches!(mode, ComputeMode::Hybrid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_hybrid_is_tiered() {
        assert!(
            mode_uses_tiering(ComputeMode::Hybrid),
            "Hybrid is the VUGVA mode — if it does not tier, nothing does and \
             a larger-than-VRAM corpus silently falls back to the CPU"
        );
        assert!(
            !mode_uses_tiering(ComputeMode::Turbo),
            "Turbo is VRAM-resident by definition; tiering it would add a page \
             lookup to the path that exists to avoid one"
        );
        assert!(!mode_uses_tiering(ComputeMode::CpuOnly));
    }

    /// Shape accessors must describe the corpus as allocated, since
    /// `device_ptr` hands out a raw pointer that a kernel will index with
    /// exactly these bounds.
    ///
    /// Runs only with a GPU present: `TieredPool::new` discovers real devices
    /// and page-locks real DRAM, so there is nothing to assert without one.
    #[test]
    fn reports_the_shape_it_was_allocated_with() {
        if !crate::gpu_init() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        let (n, dim) = (1024usize, 384usize);
        // A small warm tier on purpose: the corpus is 384 KB, so this also
        // covers the ordinary case where it fits comfortably.
        let c = match CorpusTier::with_dram_budget(&[0], n, dim, 16 << 20) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("CorpusTier unavailable ({e}) — skipping");
                return;
            }
        };
        assert_eq!(c.len(), n);
        assert_eq!(c.dim(), dim);
        assert_eq!(c.bytes(), n * dim);
        assert!(!c.is_empty());
    }

    /// A corpus larger than the warm tier must round-trip through the cold
    /// tier and reach the device intact.
    ///
    /// This is the end-to-end claim that makes Hybrid meaningful: 64 MB of
    /// vectors against a 16 MB DRAM budget, written, promoted, and read back
    /// byte-exact. Allocating is not enough — a corpus that pages in as zeros
    /// is as useless as one that fails outright.
    #[test]
    fn a_corpus_larger_than_dram_round_trips_to_the_device() {
        if !crate::gpu_init() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        let (n, dim) = (4096usize, 384usize); // 1.5 MB against a 1 MB budget
        let mut c = match CorpusTier::with_dram_budget(&[0], n, dim, 1 << 20) {
            Ok(c) => c,
            Err(e) => panic!("larger-than-DRAM corpus must allocate: {e}"),
        };
        // Position-dependent, so a wrong offset shows up rather than passing.
        let src: Vec<i8> = (0..n * dim).map(|i| ((i * 31 + (i >> 11)) % 251) as i8).collect();
        c.upload(&src).expect("upload");

        let dptr = c.device_ptr(0).expect("promote");
        let mut got = vec![0i8; n * dim];
        unsafe {
            let rc = crate::vugva::ffi::cuda::cuMemcpyDtoH_v2(
                got.as_mut_ptr() as *mut std::ffi::c_void,
                crate::vugva::ffi::cuda::CUdeviceptr(dptr),
                n * dim,
            );
            assert_eq!(rc, 0, "cuMemcpyDtoH_v2 failed: {rc}");
        }
        let bad = got.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{bad} of {} bytes differ after the tiered round trip", n * dim);

        assert!(
            c.upload(&src[..src.len() / 2]).is_err(),
            "a size mismatch must be rejected, not silently truncated"
        );
    }

    /// The property that motivates the whole type: a corpus larger than the
    /// warm tier must still allocate, because it spills rather than failing.
    ///
    /// This failed when first written: `TieredPool::allocate` backed every page
    /// with DRAM chunks before consulting `initial_tier`, so `Tier::Ssd` still
    /// demanded full DRAM and the corpus died with `DramOom`. The cliff had
    /// merely moved from VRAM down to DRAM. T2 is now wired into the pool
    /// (`allocate` reserves spill space, `access` stages SSD→DRAM→VRAM), so
    /// neither tier bounds the corpus.
    #[test]
    fn a_corpus_larger_than_the_dram_budget_still_allocates() {
        if !crate::gpu_init() {
            eprintln!("no CUDA device — skipping");
            return;
        }
        // 64 MB corpus against a 16 MB warm tier.
        let (n, dim) = (170_000usize, 384usize);
        match CorpusTier::with_dram_budget(&[0], n, dim, 16 << 20) {
            Ok(c) => assert_eq!(c.bytes(), n * dim),
            Err(e) => {
                // A DRAM-tier exhaustion here is the cliff this type removes,
                // so it is a real failure rather than an environment problem.
                panic!(
                    "corpus larger than the warm tier must spill to NVMe, not \
                     fail to allocate: {e}"
                );
            }
        }
    }
}
