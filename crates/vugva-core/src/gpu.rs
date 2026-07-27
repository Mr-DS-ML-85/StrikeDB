//! GPU device discovery, P2P peer access, and NUMA topology mapping.
//!
//! Implements the **GPU NUMA Router** from the paper (§2):
//! a hardware-aware traffic controller that calculates physical distance
//! between computing Tensor Cores and requested tensor segments across
//! the PCIe switch topology.

use crate::context::PrimaryContext;
use crate::ffi::cuda::*;
use crate::{check_cu, Result, VugvaError};
use std::collections::HashMap;

// ============================================================================
// NUMA distance matrix
// ============================================================================

/// NUMA distance between two nodes. Local = 10, cross-socket = 20–30.
/// Values from `numactl --hardware` or sysfs.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// `distances[src_node][dst_node]` = NUMA distance.
    pub distances: Vec<Vec<u32>>,
    /// Number of NUMA nodes detected.
    pub node_count: usize,
}

impl NumaTopology {
    /// Parse `numactl --hardware` output into a distance matrix.
    pub fn from_numactl() -> Result<Self> {
        let output = std::process::Command::new("numactl")
            .arg("--hardware")
            .output()
            .map_err(VugvaError::Io)?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut distances: Vec<Vec<u32>> = Vec::new();
        let mut node_count = 0;

        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("node") && trimmed.contains("distances:") {
                // Parse: "node 0 distances: 10 21 21 31 31 41 41 51"
                if let Some(colon_pos) = trimmed.find(':') {
                    let nums: Vec<u32> = trimmed[colon_pos + 1..]
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if !nums.is_empty() {
                        node_count = node_count.max(nums.len());
                        distances.push(nums);
                    }
                }
            }
        }

        if distances.is_empty() {
            // Fallback: assume single-node (distance 10 everywhere)
            distances = vec![vec![10]];
            node_count = 1;
        }

        Ok(NumaTopology {
            distances,
            node_count,
        })
    }

    /// Create a trivial single-node topology (all distances = 10).
    pub fn single_node() -> Self {
        NumaTopology {
            distances: vec![vec![10]],
            node_count: 1,
        }
    }

    /// Distance between two NUMA nodes.
    pub fn distance(&self, src: usize, dst: usize) -> u32 {
        self.distances
            .get(src)
            .and_then(|row| row.get(dst).copied())
            .unwrap_or(30)
    }

    /// Bandwidth efficiency factor based on NUMA distance.
    /// From the paper (§4.2): 0.95 for dist ≤ 12, 0.80 for ≤ 20, else 0.65.
    pub fn dma_bandwidth_factor(&self, src_node: usize, dst_node: usize) -> f64 {
        let d = self.distance(src_node, dst_node);
        if d <= 12 {
            0.95
        } else if d <= 20 {
            0.80
        } else {
            0.65
        }
    }
}

// ============================================================================
// Per-GPU information
// ============================================================================

/// Information about a single detected GPU.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// CUDA device ordinal.
    pub device_id: i32,
    /// Human-readable name (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram: usize,
    /// Free VRAM in bytes (at discovery time).
    pub free_vram: usize,
    /// Compute capability (major, minor).
    pub compute_capability: (i32, i32),
    /// Number of SMs (streaming multiprocessors).
    pub sm_count: i32,
    /// NUMA node this GPU is closest to.
    pub numa_node: usize,
    /// PCI bus ID.
    pub pci_bus_id: i32,
    /// PCI device ID.
    pub pci_device_id: i32,
    /// Supports concurrent kernels.
    pub concurrent_kernels: bool,
}

// ============================================================================
// Peer access matrix
// ============================================================================

/// Tracks which GPUs can directly access each other's memory via P2P.
#[derive(Debug)]
pub struct PeerMatrix {
    /// `can_access[src][dst]` = true if src can DMA into dst's VRAM.
    can_access: Vec<Vec<bool>>,
    /// `enabled[src][dst]` = true if peer access has been activated.
    enabled: Vec<Vec<bool>>,
    num_gpus: usize,
    /// Primary contexts retained by [`PeerMatrix::enable_all`], one per device
    /// in `gpu_ordinals` order. Held for the life of the matrix — the peer
    /// mappings installed against these contexts die with the last reference —
    /// and released by [`PrimaryContext`]'s own `Drop`. Empty until
    /// `enable_all` runs.
    primary_ctxs: Vec<PrimaryContext>,
}

impl PeerMatrix {
    /// Build the peer access matrix for the given device ordinals.
    pub fn discover(gpu_ordinals: &[i32]) -> Result<Self> {
        let n = gpu_ordinals.len();
        let mut can_access = vec![vec![false; n]; n];
        let enabled = vec![vec![false; n]; n];

        for (i, &src_ord) in gpu_ordinals.iter().enumerate() {
            let src_dev = CUdevice(src_ord);
            for (j, &dst_ord) in gpu_ordinals.iter().enumerate() {
                if i == j {
                    can_access[i][j] = true; // self-access is always allowed
                    continue;
                }
                let dst_dev = CUdevice(dst_ord);
                let mut can: i32 = 0;
                unsafe {
                    check_cu(
                        "cuDeviceCanAccessPeer",
                        cuDeviceCanAccessPeer(&mut can, src_dev, dst_dev),
                    )?;
                }
                can_access[i][j] = can != 0;
            }
        }

        Ok(PeerMatrix {
            can_access,
            enabled,
            num_gpus: n,
            primary_ctxs: Vec::new(),
        })
    }

    /// Enable P2P access for every pair the hardware reports as capable.
    ///
    /// `cuCtxEnablePeerAccess` is directional and operates on the *currently
    /// current* context: it grants that context the right to address the peer
    /// **context** passed as its argument. So for each pair we make the source
    /// GPU's context current and hand it the destination's context.
    ///
    /// Contexts are obtained with `cuDevicePrimaryCtxRetain`, not
    /// `cuCtxCreate_v2`. The primary context is reference-counted and shared
    /// with everything else in the process, so retaining it twice is free and
    /// the mapping we install stays visible to other users of the same device.
    /// The previous implementation created a fresh context per source GPU and
    /// never destroyed it. Measured on this machine that is 97.8 MB of VRAM
    /// leaked per device per call (see [`crate::context`]), and the peer mapping
    /// was installed into a private context that no other code path ever made
    /// current, so it had no effect on real transfers either.
    ///
    /// Retained contexts are kept in `self.primary_ctxs` for the life of the
    /// matrix: releasing them here would drop the refcount to zero and tear
    /// down the very mappings we just installed.
    pub fn enable_all(&mut self, gpu_ordinals: &[i32]) -> Result<()> {
        // Retain a primary context per device, once, before touching any pair.
        // `PrimaryContext` owns its own release, so a failure partway through
        // this loop drops the ones already retained instead of leaking them —
        // which the hand-rolled retain/release pair here used to do.
        if self.primary_ctxs.is_empty() {
            let mut ctxs = Vec::with_capacity(gpu_ordinals.len());
            for &ord in gpu_ordinals {
                ctxs.push(PrimaryContext::retain(ord)?);
            }
            self.primary_ctxs = ctxs;
        }

        for i in 0..gpu_ordinals.len() {
            // The guard pops on drop, so the `?` below cannot leave GPU `i`'s
            // context current on the caller's thread.
            let _guard = self.primary_ctxs[i].enter()?;

            for j in 0..gpu_ordinals.len() {
                if i == j || !self.can_access[i][j] || self.enabled[i][j] {
                    continue;
                }
                let dst_ctx = self.primary_ctxs[j].raw();
                // SAFETY: `dst_ctx` is a live retained primary context, and
                // GPU `i`'s context is current, which is what this call needs.
                let res = unsafe { cuCtxEnablePeerAccess(dst_ctx, 0) };
                // ALREADY_ENABLED means another user of this primary context
                // installed the same mapping first — success, not failure.
                // (This was previously compared against the literal 724, which
                // is not a CUresult at all; the real code is 704, so an
                // already-enabled pair was recorded as *not* enabled.)
                //
                // Any other failure is left as `enabled[i][j] == false` rather
                // than propagated: TOO_MANY_PEERS on a large box is a real
                // hardware limit, and callers must consult `is_enabled` before
                // attempting a peer copy regardless.
                if res == CUDA_SUCCESS || res == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                    self.enabled[i][j] = true;
                }
            }
        }
        Ok(())
    }

    /// Can GPU `src` directly access GPU `dst`'s memory?
    pub fn can_access(&self, src: usize, dst: usize) -> bool {
        self.can_access
            .get(src)
            .and_then(|row| row.get(dst).copied())
            .unwrap_or(false)
    }

    /// Is P2P access currently active for (src, dst)?
    pub fn is_enabled(&self, src: usize, dst: usize) -> bool {
        self.enabled
            .get(src)
            .and_then(|row| row.get(dst).copied())
            .unwrap_or(false)
    }
}

// ============================================================================
// GPU cluster (high-level)
// ============================================================================

/// The complete GPU cluster: discovered devices, NUMA topology, and P2P matrix.
pub struct GpuCluster {
    /// Per-GPU information, indexed by position in `ordinals`.
    pub infos: Vec<GpuInfo>,
    /// The CUDA device ordinals used.
    pub ordinals: Vec<i32>,
    /// P2P peer access matrix.
    pub peer_matrix: PeerMatrix,
    /// NUMA distance matrix.
    pub numa: NumaTopology,
    /// Ordinal → position index in `infos`.
    ordinal_map: HashMap<i32, usize>,
}

impl GpuCluster {
    /// Discover all available GPUs and build the cluster.
    pub fn discover(gpu_ordinals: &[i32]) -> Result<Self> {
        unsafe {
            check_cu("cuInit", cuInit(0))?;
        }

        let mut infos = Vec::with_capacity(gpu_ordinals.len());
        let numa = NumaTopology::from_numactl().unwrap_or_else(|_| NumaTopology::single_node());

        for &ord in gpu_ordinals {
            let dev = CUdevice(ord);
            let mut info = GpuInfo {
                device_id: ord,
                name: String::new(),
                total_vram: 0,
                free_vram: 0,
                compute_capability: (0, 0),
                sm_count: 0,
                numa_node: 0,
                pci_bus_id: 0,
                pci_device_id: 0,
                concurrent_kernels: false,
            };

            // Device name
            let mut name_buf = [0u8; 256];
            unsafe {
                cuDeviceGetName(name_buf.as_mut_ptr() as *mut i8, 256, dev);
            }
            info.name = std::ffi::CStr::from_bytes_until_nul(&name_buf)
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();

            // Total VRAM
            let mut total: usize = 0;
            unsafe {
                cuDeviceTotalMem_v2(&mut total, dev);
            }
            info.total_vram = total;

            // Free VRAM. `cuMemGetInfo_v2` reports for whatever context is
            // current, so one has to be pushed first.
            //
            // This used to be `cuCtxCreate_v2` / `cuCtxDestroy_v2` with both
            // return codes discarded (BUG #17). Two things were wrong. The
            // private context cost ~98 MB of VRAM while it lived, so the
            // "free VRAM" figure it produced was already short by roughly the
            // size of the probe. And if the create failed — GPU in exclusive
            // mode, or another process holding it — `ctx` stayed null,
            // `cuMemGetInfo_v2` failed against no context, its error was
            // dropped too, and `free_vram` silently stayed 0. A perfectly
            // healthy GPU then looked like it had no memory, and the allocator
            // skipped it.
            //
            // Retaining the primary context costs nothing (it is refcounted and
            // already alive), the guard pops it even if a later `?` in this
            // loop returns early, and both errors now propagate.
            {
                let pctx = crate::context::PrimaryContext::retain(ord)?;
                let _guard = pctx.enter()?;
                let mut free: usize = 0;
                let mut tot: usize = 0;
                // SAFETY: both are valid out-pointers and a context is current.
                unsafe {
                    check_cu("cuMemGetInfo_v2", cuMemGetInfo_v2(&mut free, &mut tot))?;
                }
                info.free_vram = free;
            }

            // Compute capability
            let (mut major, mut minor) = (0i32, 0i32);
            unsafe {
                cuDeviceComputeCapability(&mut major, &mut minor, dev);
            }
            info.compute_capability = (major, minor);

            // SM count
            let mut sm = 0i32;
            unsafe {
                cuDeviceGetAttribute(&mut sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
            }
            info.sm_count = sm;

            // Concurrent kernels
            let mut ck = 0i32;
            unsafe {
                cuDeviceGetAttribute(&mut ck, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
            }
            info.concurrent_kernels = ck != 0;

            // PCI bus ID
            let mut bus = 0i32;
            unsafe {
                cuDeviceGetAttribute(&mut bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
            }
            info.pci_bus_id = bus;

            // PCI device ID
            let mut pdev = 0i32;
            unsafe {
                cuDeviceGetAttribute(
                    &mut pdev,
                    crate::ffi::cuda::CUDEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                    dev,
                );
            }
            info.pci_device_id = pdev;

            // NUMA node — try sysfs
            info.numa_node = read_numa_node(ord).unwrap_or(0);

            infos.push(info);
        }

        let peer_matrix = PeerMatrix::discover(gpu_ordinals)?;

        let ordinal_map: HashMap<i32, usize> = gpu_ordinals
            .iter()
            .enumerate()
            .map(|(i, &o)| (o, i))
            .collect();

        Ok(GpuCluster {
            infos,
            ordinals: gpu_ordinals.to_vec(),
            peer_matrix,
            numa,
            ordinal_map,
        })
    }

    /// Enable peer access for all GPU pairs (must be called once).
    pub fn enable_peer_access(&mut self) -> Result<()> {
        self.peer_matrix.enable_all(&self.ordinals)
    }

    /// Index of a GPU ordinal in the cluster.
    pub fn index_of(&self, ordinal: i32) -> Option<usize> {
        self.ordinal_map.get(&ordinal).copied()
    }

    /// NUMA bandwidth factor for a GPU → DRAM-node transfer.
    pub fn dma_factor(&self, gpu_ordinal: i32, dram_numa_node: usize) -> f64 {
        if let Some(idx) = self.index_of(gpu_ordinal) {
            let gpu_numa = self.infos[idx].numa_node;
            self.numa.dma_bandwidth_factor(dram_numa_node, gpu_numa)
        } else {
            0.65
        }
    }

    /// Select the optimal DRAM NUMA node for a given GPU.
    pub fn optimal_dram_node(&self, gpu_ordinal: i32) -> usize {
        if let Some(idx) = self.index_of(gpu_ordinal) {
            let gpu_numa = self.infos[idx].numa_node;
            let mut best = 0usize;
            let mut best_dist = u32::MAX;
            for node in 0..self.numa.node_count {
                let d = self.numa.distance(gpu_numa, node);
                if d < best_dist {
                    best_dist = d;
                    best = node;
                }
            }
            best
        } else {
            0
        }
    }
}

/// Read GPU NUMA node from sysfs.
fn read_numa_node(gpu_ordinal: i32) -> Option<usize> {
    // Try several sysfs paths
    let paths = [
        format!(
            "/sys/bus/pci/devices/0000:{:02x}:00.0/numa_node",
            gpu_ordinal
        ),
        format!("/sys/class/drm/card{gpu_ordinal}/device/numa_node"),
    ];
    for path in &paths {
        if let Ok(s) = std::fs::read_to_string(path) {
            if let Ok(n) = s.trim().parse::<isize>() {
                // -1 means "no NUMA node" — default to 0
                return Some(if n < 0 { 0 } else { n as usize });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_single_node() {
        let topo = NumaTopology::single_node();
        assert_eq!(topo.node_count, 1);
        assert_eq!(topo.distance(0, 0), 10);
        assert_eq!(topo.dma_bandwidth_factor(0, 0), 0.95);
    }

    #[test]
    fn numa_bandwidth_factors() {
        let topo = NumaTopology {
            distances: vec![vec![10, 18], vec![18, 10]],
            node_count: 2,
        };
        // Same node: 0.95
        assert_eq!(topo.dma_bandwidth_factor(0, 0), 0.95);
        assert_eq!(topo.dma_bandwidth_factor(1, 1), 0.95);
        // Cross-socket (18, ≤20): 0.80
        assert_eq!(topo.dma_bandwidth_factor(0, 1), 0.80);
        assert_eq!(topo.dma_bandwidth_factor(1, 0), 0.80);
    }

    #[test]
    fn numa_far_node() {
        let topo = NumaTopology {
            distances: vec![vec![10, 31], vec![31, 10]],
            node_count: 2,
        };
        assert_eq!(topo.dma_bandwidth_factor(0, 1), 0.65);
    }

    #[test]
    fn numa_out_of_bounds_distance() {
        let topo = NumaTopology::single_node();
        // Out-of-bounds returns 30 (far)
        assert_eq!(topo.distance(0, 99), 30);
    }

    #[test]
    fn peer_matrix_self_access() {
        let matrix = PeerMatrix {
            can_access: vec![vec![true, false], vec![false, true]],
            enabled: vec![vec![false, false], vec![false, false]],
            num_gpus: 2,
            primary_ctxs: Vec::new(),
        };
        assert!(matrix.can_access(0, 0));
        assert!(matrix.can_access(1, 1));
        assert!(!matrix.can_access(0, 1));
    }

    #[test]
    fn gpu_cluster_optimal_dram_node() {
        let cluster = GpuCluster {
            infos: vec![GpuInfo {
                device_id: 0,
                name: "test".into(),
                total_vram: 0,
                free_vram: 0,
                compute_capability: (0, 0),
                sm_count: 0,
                numa_node: 0,
                pci_bus_id: 0,
                pci_device_id: 0,
                concurrent_kernels: false,
            }],
            ordinals: vec![0],
            peer_matrix: PeerMatrix {
                can_access: vec![vec![true]],
                enabled: vec![vec![false]],
                num_gpus: 1,
                primary_ctxs: Vec::new(),
            },
            numa: NumaTopology {
                distances: vec![vec![10, 18], vec![18, 10]],
                node_count: 2,
            },
            ordinal_map: [(0, 0)].into(),
        };
        // GPU 0 is on NUMA node 0, closest DRAM node is 0
        assert_eq!(cluster.optimal_dram_node(0), 0);
        assert_eq!(cluster.dma_factor(0, 0), 0.95);
        // Distance 18 (≤20) → 0.80
        assert_eq!(cluster.dma_factor(0, 1), 0.80);
    }
}
