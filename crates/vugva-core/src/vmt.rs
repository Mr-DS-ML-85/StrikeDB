//! Virtual Memory Table (VMT) — the core mapping layer.
//!
//! Implements the paper's VMT (§2, §3.1): a global state machine that maps
//! virtual memory addresses exposed to the framework down to physical,
//! chunked byte offsets distributed across GPU VRAM and system DRAM.
//!
//! ## Page state machine (paper §4.1, Figure 5)
//!
//! ```text
//! UNMAPPED ──cudaMalloc──▶ ALLOCATED ──1st access──▶ RESIDENT (VRAM)
//!                                        │                 │
//!                                   prefetch          evict/promote
//!                                        │                 │
//!                                        ▼                 ▼
//!                                   WARM (DRAM) ◀──▶ COLD (SSD)
//! ```

use crate::Result;

// ============================================================================
// Tier enum
// ============================================================================

/// Memory tier — matches the paper's three-tier hierarchy (Table 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Tier 0: GPU VRAM — hot, active layers (< 100 ns latency, 1008 GB/s).
    Vram,
    /// Tier 1: System DRAM — warm, next N layers (2–8 µs, 28–58 GB/s).
    Dram,
    /// Tier 2: NVMe SSD — cold, spill (50–100 µs, 5–7 GB/s).
    Ssd,
}

// ============================================================================
// Page state machine
// ============================================================================

/// Page state — mirrors Figure 5 of the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// Not yet allocated.
    Unmapped,
    /// `cudaMalloc` called but never accessed.
    Allocated,
    /// Resident in GPU VRAM (hot).
    Resident,
    /// Demoted to system DRAM (warm).
    Warm,
    /// Spilled to NVMe SSD (cold).
    Cold,
}

// ============================================================================
// Virtual page
// ============================================================================

/// A single virtual page in the VMT.
///
/// Each page tracks its current tier, state, and the physical chunks
/// distributed across GPUs and DRAM nodes.
#[derive(Debug)]
pub struct Page {
    /// Logical name (e.g. `"model.layer.3.weight"`).
    pub name: String,
    /// Current memory tier.
    pub tier: Tier,
    /// State in the migration state machine.
    pub state: PageState,
    /// Shape of the tensor this page backs (element dims).
    pub shape: Vec<usize>,
    /// Element size in bytes.
    pub element_size: usize,
    /// Total size in bytes.
    pub size_bytes: usize,

    /// Per-GPU VRAM chunk offsets: `gpu_ordinal → CUdeviceptr`.
    pub vram_chunks: Vec<Chunk>,
    /// Per-NUMA DRAM chunk pointers: `numa_node → *mut u8`.
    pub dram_chunks: Vec<DramChunk>,
    /// SSD offset (if spilled).
    pub ssd_offset: Option<u64>,

    /// Number of accesses since last demotion.
    pub access_count: u64,
    /// Timestamp of last access (monotonic nanoseconds).
    pub last_access_ns: u64,
    /// If true, cannot be evicted (pinned by framework).
    pub pinned: bool,
}

/// A contiguous chunk of a page residing in a GPU's VRAM.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// GPU ordinal that owns this chunk.
    pub gpu_ordinal: i32,
    /// Device pointer offset into that GPU's VRAM.
    pub device_ptr: u64,
    /// Byte length of this chunk.
    pub size_bytes: usize,
    /// Number of elements in this chunk.
    pub num_elements: usize,
}

/// A contiguous chunk of a page residing in system DRAM.
#[derive(Debug, Clone)]
pub struct DramChunk {
    /// NUMA node where this DRAM region is allocated.
    pub numa_node: usize,
    /// Host pointer (userspace virtual address).
    pub host_ptr: usize,
    /// Byte length of this chunk.
    pub size_bytes: usize,
    /// Whether this region is registered with CUDA for direct DMA.
    pub cuda_registered: bool,
}

impl Page {
    /// Create a new unmapped page.
    pub fn new(
        name: String,
        shape: Vec<usize>,
        element_size: usize,
        num_gpus: usize,
        num_numa_nodes: usize,
    ) -> Self {
        let size_bytes: usize = shape.iter().product::<usize>() * element_size;
        Page {
            name,
            tier: Tier::Vram,
            state: PageState::Unmapped,
            shape,
            element_size,
            size_bytes,
            vram_chunks: Vec::with_capacity(num_gpus),
            dram_chunks: Vec::with_capacity(num_numa_nodes),
            ssd_offset: None,
            access_count: 0,
            last_access_ns: 0,
            pinned: false,
        }
    }

    /// Transition to a new state (validates the state machine).
    pub fn transition(&mut self, new_state: PageState) -> Result<()> {
        use PageState::*;
        let valid = matches!(
            (self.state, new_state),
            (Unmapped, Allocated)
                | (Allocated, Resident)
                | (Resident, Warm)
                | (Warm, Resident)
                | (Warm, Cold)
                | (Cold, Warm)
                | (Cold, Resident)
                | (Resident, Cold)
        );
        if valid {
            self.state = new_state;
            Ok(())
        } else {
            Err(crate::VugvaError::InvalidTransition {
                from: self.state,
                to: new_state,
                page: self.name.clone(),
            })
        }
    }

    /// Update access metadata (count + timestamp).
    pub fn touch(&mut self, now_ns: u64) {
        self.access_count += 1;
        self.last_access_ns = now_ns;
    }

    /// Check if the page should be demoted based on age.
    /// `idle_threshold_ns` — nanoseconds since last access before demotion.
    pub fn is_idle(&self, now_ns: u64, idle_threshold_ns: u64) -> bool {
        !self.pinned && now_ns.saturating_sub(self.last_access_ns) > idle_threshold_ns
    }

    /// Check if the page is "hot" enough for proactive promotion.
    pub fn is_hot(&self, access_threshold: u64) -> bool {
        self.access_count >= access_threshold
    }
}

// ============================================================================
// Virtual Memory Table
// ============================================================================

/// The VMT: maps string names → virtual pages.
///
/// This is the "global state machine" described in §2 of the paper.
/// The framework (PyTorch/vLLM) never sees physical GPU chunks — it only
/// interacts with string-named allocations through this table.
pub struct VirtualMemoryTable {
    pages: std::collections::HashMap<String, Page>,
    /// Number of GPUs in the cluster.
    num_gpus: usize,
    /// Number of NUMA nodes.
    num_numa_nodes: usize,
    /// Monotonic counter for allocation IDs.
    next_id: u64,
}

impl VirtualMemoryTable {
    /// Create an empty VMT for a cluster with the given topology.
    pub fn new(num_gpus: usize, num_numa_nodes: usize) -> Self {
        VirtualMemoryTable {
            pages: std::collections::HashMap::new(),
            num_gpus,
            num_numa_nodes,
            next_id: 0,
        }
    }

    /// Register a new virtual allocation. Returns the generated name.
    pub fn allocate(&mut self, name: &str, shape: &[usize], element_size: usize) -> Result<String> {
        let alloc_name = if name.is_empty() {
            self.next_id += 1;
            format!("vugva_alloc_{}", self.next_id)
        } else {
            name.to_string()
        };

        let mut page = Page::new(
            alloc_name.clone(),
            shape.to_vec(),
            element_size,
            self.num_gpus,
            self.num_numa_nodes,
        );
        page.state = PageState::Allocated;
        page.tier = Tier::Vram;
        self.pages.insert(alloc_name.clone(), page);
        Ok(alloc_name)
    }

    /// Look up a page by name.
    pub fn lookup(&self, name: &str) -> Option<&Page> {
        self.pages.get(name)
    }

    /// Mutable lookup.
    pub fn lookup_mut(&mut self, name: &str) -> Option<&mut Page> {
        self.pages.get_mut(name)
    }

    /// Remove and return a page.
    pub fn remove(&mut self, name: &str) -> Option<Page> {
        self.pages.remove(name)
    }

    /// Number of registered pages.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// True if no pages are registered.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Iterate all pages (for background manager sweeps).
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Page)> {
        self.pages.iter()
    }

    /// Mutable iterate all pages.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut Page)> {
        self.pages.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_state_transitions() {
        let mut page = Page::new("test".into(), vec![1024], 2, 1, 1);

        assert_eq!(page.state, PageState::Unmapped);

        page.transition(PageState::Allocated).unwrap();
        assert_eq!(page.state, PageState::Allocated);

        page.transition(PageState::Resident).unwrap();
        assert_eq!(page.state, PageState::Resident);

        page.transition(PageState::Warm).unwrap();
        assert_eq!(page.state, PageState::Warm);

        // Warm → Resident (promotion)
        page.transition(PageState::Resident).unwrap();
        assert_eq!(page.state, PageState::Resident);
    }

    #[test]
    fn invalid_transition_rejected() {
        let mut page = Page::new("test".into(), vec![1024], 2, 1, 1);
        page.transition(PageState::Allocated).unwrap();

        // Allocated → Warm is invalid
        let result = page.transition(PageState::Warm);
        assert!(result.is_err());
    }

    #[test]
    fn page_size_calculation() {
        let page = Page::new("test".into(), vec![128, 256], 2, 1, 1);
        assert_eq!(page.size_bytes, 128 * 256 * 2);
    }

    #[test]
    fn page_touch_updates_count_and_time() {
        let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
        assert_eq!(page.access_count, 0);
        assert_eq!(page.last_access_ns, 0);

        page.touch(1000);
        assert_eq!(page.access_count, 1);
        assert_eq!(page.last_access_ns, 1000);

        page.touch(2000);
        assert_eq!(page.access_count, 2);
    }

    #[test]
    fn page_idle_detection() {
        let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
        page.touch(1000);

        // Not idle: only 500ns elapsed, threshold is 1000ns
        assert!(!page.is_idle(1500, 1000));

        // Idle: 2000ns elapsed
        assert!(page.is_idle(3000, 1000));
    }

    #[test]
    fn page_pinned_cannot_idle() {
        let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
        page.pinned = true;
        page.touch(0);

        assert!(!page.is_idle(10_000_000, 1000));
    }

    #[test]
    fn page_hot_detection() {
        let mut page = Page::new("test".into(), vec![100], 2, 1, 1);
        assert!(!page.is_hot(10));

        for i in 0..10 {
            page.touch(i);
        }
        assert!(page.is_hot(10));
    }

    #[test]
    fn vmt_allocate_and_lookup() {
        let mut vmt = VirtualMemoryTable::new(2, 1);
        let name = vmt.allocate("layer.0.weight", &[4096, 4096], 2).unwrap();

        assert_eq!(vmt.len(), 1);
        let page = vmt.lookup(&name).unwrap();
        assert_eq!(page.name, "layer.0.weight");
        assert_eq!(page.state, PageState::Allocated);
        assert_eq!(page.tier, Tier::Vram);
        assert_eq!(page.size_bytes, 4096 * 4096 * 2);
    }

    #[test]
    fn vmt_empty_name_generates_id() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        let name = vmt.allocate("", &[100], 4).unwrap();
        assert!(name.starts_with("vugva_alloc_"));
    }

    #[test]
    fn vmt_remove() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        let name = vmt.allocate("test", &[100], 4).unwrap();
        assert_eq!(vmt.len(), 1);

        let removed = vmt.remove(&name).unwrap();
        assert_eq!(removed.name, "test");
        assert!(vmt.is_empty());
    }

    #[test]
    fn vmt_iter() {
        let mut vmt = VirtualMemoryTable::new(1, 1);
        vmt.allocate("a", &[10], 1).unwrap();
        vmt.allocate("b", &[20], 1).unwrap();
        vmt.allocate("c", &[30], 1).unwrap();

        let names: Vec<_> = vmt.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"c".to_string()));
    }
}
