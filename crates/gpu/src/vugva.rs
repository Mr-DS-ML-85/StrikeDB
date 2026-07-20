//! VUGVA — Virtual Unified GPU VRAM Architecture
//!
//! Paper: "A Software-Defined Virtual Unified GPU VRAM Architecture
//! for Non-NVLink Multi-GPU Clusters" (Irfan, June 2026)
//!
//! Architecture (§2, §3):
//! ┌─────────────────────────────────────────────┐
//! │     VUGVA Memory Routing Engine (Rust)       │
//! │  ┌──────────────┐  ┌─────────────────────┐  │
//! │  │ Virtual Memory│  │ Look-Ahead Tracker  │  │
//! │  │ Table (VMT)   │  │ (predict next I/O)  │  │
//! │  └──────┬───────┘  └──────────┬──────────┘  │
//! │         │   NUMA Router       │             │
//! └─────────┼─────────────────────┼─────────────┘
//!     ┌─────┼──────┐         ┌────┼────────┐
//!     ▼     ▼      ▼         ▼    ▼        ▼
//!  ┌─────┐┌─────┐┌─────┐  ┌─────┐┌─────┐┌─────┐
//!  │GPU 0││GPU 1││ GPU N│  │RAM 0││RAM 1││NVMe │
//!  │VRAM ││VRAM ││ VRAM │  │     ││     ││     │
//!  └─────┘└─────┘└─────┘  └─────┘└─────┘└─────┘
//!       ▲       ▲              ▲
//!       └───Async Prefetch────┘
//!           (CUDA streams)
//!
//! Three foundational components (§3):
//! 1. CUDA Runtime Interceptor → we use Rust-level allocation hooks
//! 2. Virtual Memory Table → maps virtual addresses to physical chunks
//! 3. GPU NUMA Router → routes data based on compute↔memory distance

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Where a chunk currently lives in the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChunkLocation {
    /// Not loaded yet.
    Cold,
    /// In CPU RAM (warm tier).
    Ram,
    /// In GPU VRAM (hot tier).
    Vram,
}

/// A chunk of vector data managed by VUGVA.
struct VugvaChunk {
    /// Source of truth: vector data in CPU RAM.
    ram_data: Vec<i8>,
    /// GPU device pointer (0 = not in VRAM).
    gpu_ptr: u64,
    /// Current location in the memory hierarchy.
    location: ChunkLocation,
    /// LRU timestamp.
    last_access: usize,
    /// Currently being prefetched to GPU?
    prefetching: bool,
}

/// Look-Ahead Attention Tracker (§3.2 of VUGVA paper).
/// Predicts upcoming data access based on deterministic graph traversal patterns.
/// During HNSW search, the graph walk follows a predictable trajectory:
///   node → neighbors → neighbors_of_neighbors → ...
/// The tracker records the traversal path and prefetches the NEXT chunks
/// before the GPU finishes processing the current one.
struct LookAheadTracker {
    /// Recent access history (circular buffer).
    history: VecDeque<usize>,
    /// Maximum history depth.
    max_depth: usize,
    /// Predictive prefetch queue: chunks likely needed next.
    prefetch_queue: VecDeque<usize>,
}

impl LookAheadTracker {
    fn new(max_depth: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_depth),
            max_depth,
            prefetch_queue: VecDeque::new(),
        }
    }

    /// Record a chunk access and predict next chunks to prefetch.
    fn record_access(&mut self, chunk_idx: usize, chunk_size: usize, n_vectors: usize) {
        self.history.push_back(chunk_idx);
        if self.history.len() > self.max_depth {
            self.history.pop_front();
        }
        // Predictive heuristic: adjacent chunks likely needed next.
        // During HNSW graph traversal, consecutive node accesses
        // often span nearby vector indices (spatial locality).
        let next = (chunk_idx + 1) * chunk_size;
        let next_chunk = next / chunk_size;
        if next_chunk < (n_vectors + chunk_size - 1) / chunk_size {
            self.prefetch_queue.push_back(next_chunk);
        }
        // Also predict the reverse direction (bidirectional graph traversal).
        if chunk_idx > 0 {
            self.prefetch_queue.push_back(chunk_idx - 1);
        }
    }

    /// Get next chunk to prefetch (drain one at a time).
    fn next_prefetch(&mut self) -> Option<usize> {
        // Deduplicate against recent history
        while let Some(next) = self.prefetch_queue.pop_front() {
            if !self.history.contains(&next) {
                return Some(next);
            }
        }
        None
    }
}

/// GPU NUMA Router (§3.3 of VUGVA paper).
/// Calculates the physical distance between a compute unit and a memory chunk.
/// In single-GPU, distance is 0 (local) or 1 (host RAM).
/// In multi-GPU, distance is the hop count through the PCIe/PEX switch.
#[allow(dead_code)]
struct NumaRouter {
    num_gpus: usize,
    gpu_vram: Vec<usize>,
}

#[allow(dead_code)]
impl NumaRouter {
    fn new(num_gpus: usize, gpu_vram: Vec<usize>) -> Self {
        Self { num_gpus, gpu_vram }
    }

    /// Compute "distance" between GPU `compute_gpu` and a chunk on `target_gpu`.
    /// 0 = same GPU (fastest), 1 = different GPU (slower, PCIe P2P).
    fn distance(&self, compute_gpu: usize, target_gpu: usize) -> u32 {
        if compute_gpu == target_gpu { 0 } else { 1 }
    }

    /// Select the best GPU to place a new chunk on (for multi-GPU).
    /// Picks the GPU with the most free VRAM.
    fn best_gpu_for_chunk(&self, current_usage: &[usize]) -> usize {
        let mut best = 0;
        let mut best_free = 0;
        for (i, usage) in current_usage.iter().enumerate() {
            if i < self.num_gpus {
                let free = self.gpu_vram[i].saturating_sub(*usage);
                if free > best_free {
                    best_free = free;
                    best = i;
                }
            }
        }
        best
    }
}

#[allow(dead_code)]
pub struct VugvaVmt {
    chunks: Vec<Mutex<VugvaChunk>>,
    chunk_size: usize,
    dim: usize,
    n: usize,
    vram_budget: usize,
    vram_slots: usize,
    access_counter: AtomicUsize,
    lookahead: Mutex<LookAheadTracker>,
    numa: NumaRouter,
    prefetch_active: AtomicBool,
}

/// GPU chunk pointer returned after prefetch.
pub struct GpuChunkPtr {
    pub device_ptr: u64,
    pub chunk_offset: usize,
    pub chunk_len: usize,
}

impl VugvaVmt {
    pub fn new(vectors: &[i8], dim: usize, chunk_size: usize, vram_budget_bytes: usize) -> Self {
        let n = vectors.len() / dim;
        let chunks_count = (n + chunk_size - 1) / chunk_size;
        let vec_bytes = chunk_size * dim;
        let vram_slots = vram_budget_bytes / vec_bytes;
        eprintln!("[VUGVA] VMT init: {} vecs, {} chunks, {} VRAM slots ({:.0} MB)",
            n, chunks_count, vram_slots, vram_budget_bytes as f64 / 1024.0 / 1024.0);

        let mut chunks = Vec::with_capacity(chunks_count);
        for i in 0..chunks_count {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(n);
            chunks.push(Mutex::new(VugvaChunk {
                ram_data: vectors[start * dim..end * dim].to_vec(),
                gpu_ptr: 0,
                location: ChunkLocation::Ram,
                last_access: 0,
                prefetching: false,
            }));
        }

        Self {
            chunks, chunk_size, dim, n,
            vram_budget: vram_budget_bytes, vram_slots,
            access_counter: AtomicUsize::new(0),
            lookahead: Mutex::new(LookAheadTracker::new(64)),
            numa: NumaRouter::new(1, vec![vram_budget_bytes]),
            prefetch_active: AtomicBool::new(false),
        }
    }

    pub fn num_chunks(&self) -> usize { self.chunks.len() }
    pub fn chunk_of(&self, vec_idx: usize) -> usize { vec_idx / self.chunk_size }

    /// Prefetch chunk to GPU. This is the VUGVA §3.2 async prefetch.
    /// If chunk is already in VRAM, returns immediately (cache hit).
    /// If VRAM is full, evicts the LRU chunk first.
    pub fn prefetch(&self, chunk_idx: usize) -> Option<GpuChunkPtr> {
        if chunk_idx >= self.chunks.len() { return None; }
        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);

        // Update look-ahead tracker
        {
            let mut la = self.lookahead.lock().unwrap();
            la.record_access(chunk_idx, self.chunk_size, self.n);
        }

        // Cache hit: already in VRAM
        {
            let mut chunk = self.chunks[chunk_idx].lock().unwrap();
            if chunk.location == ChunkLocation::Vram && chunk.gpu_ptr != 0 {
                chunk.last_access = access;
                let start = chunk_idx * self.chunk_size;
                let len = chunk.ram_data.len() / self.dim;
                return Some(GpuChunkPtr { device_ptr: chunk.gpu_ptr, chunk_offset: start, chunk_len: len });
            }
        }

        // Cache miss: prefetch from RAM to GPU
        let mut chunk = self.chunks[chunk_idx].lock().unwrap();
        if chunk.location == ChunkLocation::Vram && chunk.gpu_ptr != 0 {
            chunk.last_access = access;
            let start = chunk_idx * self.chunk_size;
            let len = chunk.ram_data.len() / self.dim;
            return Some(GpuChunkPtr { device_ptr: chunk.gpu_ptr, chunk_offset: start, chunk_len: len });
        }

        // Evict LRU chunk if VRAM full
        self.evict_if_needed(chunk_idx);

        // Allocate + copy RAM → GPU
        let bytes = chunk.ram_data.len();
        let gpu_ptr = unsafe {
            let mut ptr = 0u64;
            if crate::cuMemAlloc_v2(&mut ptr, bytes) != 0 { return None; }
            crate::cuMemcpyHtoD_v2(ptr, chunk.ram_data.as_ptr() as *const std::ffi::c_void, bytes);
            ptr
        };
        chunk.gpu_ptr = gpu_ptr;
        chunk.location = ChunkLocation::Vram;
        chunk.last_access = access;
        chunk.prefetching = false;

        let start = chunk_idx * self.chunk_size;
        let len = chunk.ram_data.len() / self.dim;
        Some(GpuChunkPtr { device_ptr: gpu_ptr, chunk_offset: start, chunk_len: len })
    }

    /// Prefetch batch: deduplicate + sort by chunk index for coalesced access.
    pub fn prefetch_batch(&self, vec_indices: &[usize]) -> Vec<GpuChunkPtr> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for &idx in vec_indices {
            let ci = self.chunk_of(idx);
            if seen.insert(ci) {
                if let Some(ptr) = self.prefetch(ci) { result.push(ptr); }
            }
        }
        result.sort_by_key(|c| c.chunk_offset);
        result
    }

    /// Trigger look-ahead prefetch: predict next chunks and prefetch them
    /// in the background (called between search iterations).
    pub fn prefetch_lookahead(&self) {
        let next = {
            let mut la = self.lookahead.lock().unwrap();
            la.next_prefetch()
        };
        if let Some(ci) = next {
            if ci < self.chunks.len() {
                let mut chunk = self.chunks[ci].lock().unwrap();
                if chunk.location != ChunkLocation::Vram && !chunk.prefetching {
                    chunk.prefetching = true;
                    // Background prefetch (in production: use CUDA stream)
                    let bytes = chunk.ram_data.len();
                    let gpu_ptr = unsafe {
                        let mut ptr = 0u64;
                        if crate::cuMemAlloc_v2(&mut ptr, bytes) == 0 {
                            crate::cuMemcpyHtoD_v2(ptr, chunk.ram_data.as_ptr() as *const std::ffi::c_void, bytes);
                            ptr
                        } else { 0 }
                    };
                    if gpu_ptr != 0 {
                        chunk.gpu_ptr = gpu_ptr;
                        chunk.location = ChunkLocation::Vram;
                        chunk.prefetching = false;
                    }
                }
            }
        }
    }

    /// Evict least-recently-used chunk from VRAM to make room.
    fn evict_if_needed(&self, _needed_for: usize) {
        // Count VRAM chunks
        let mut vram_count = 0;
        for chunk in &self.chunks {
            if chunk.lock().unwrap().location == ChunkLocation::Vram { vram_count += 1; }
        }
        if vram_count < self.vram_slots { return; }

        // Find LRU chunk in VRAM
        let mut lru_idx = 0;
        let mut lru_time = usize::MAX;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let c = chunk.lock().unwrap();
            if c.location == ChunkLocation::Vram && c.last_access < lru_time {
                lru_time = c.last_access;
                lru_idx = i;
            }
        }

        // Evict it
        let mut chunk = self.chunks[lru_idx].lock().unwrap();
        if chunk.location == ChunkLocation::Vram && chunk.gpu_ptr != 0 {
            unsafe { crate::cuMemFree_v2(chunk.gpu_ptr); }
            chunk.gpu_ptr = 0;
            chunk.location = ChunkLocation::Ram;
        }
    }

    /// Read a vector from RAM (source of truth). Copies to avoid lock issues.
    pub fn read_vector_copy(&self, vec_idx: usize) -> Vec<i8> {
        let ci = self.chunk_of(vec_idx);
        let local = vec_idx - ci * self.chunk_size;
        let chunk = self.chunks[ci].lock().unwrap();
        let start = local * self.dim;
        chunk.ram_data[start..start + self.dim].to_vec()
    }

    pub fn cleanup(&self) {
        for chunk in &self.chunks {
            let mut c = chunk.lock().unwrap();
            if c.gpu_ptr != 0 {
                unsafe { crate::cuMemFree_v2(c.gpu_ptr); }
                c.gpu_ptr = 0;
                c.location = ChunkLocation::Ram;
            }
        }
    }
}

impl Drop for VugvaVmt {
    fn drop(&mut self) { self.cleanup(); }
}

pub fn vugva_stats(vmt: &VugvaVmt) -> Vec<(&'static str, String)> {
    let mut info = Vec::new();
    let mut in_vram = 0usize;
    let mut in_ram = 0usize;
    let mut cold = 0usize;
    for i in 0..vmt.num_chunks() {
        let c = vmt.chunks[i].lock().unwrap();
        match c.location {
            ChunkLocation::Vram => in_vram += 1,
            ChunkLocation::Ram => in_ram += 1,
            ChunkLocation::Cold => cold += 1,
        }
    }
    info.push(("vugva_chunks", vmt.num_chunks().to_string()));
    info.push(("vugva_in_vram", in_vram.to_string()));
    info.push(("vugva_in_ram", in_ram.to_string()));
    info.push(("vugva_cold", cold.to_string()));
    info.push(("vugva_vram_slots", vmt.vram_slots.to_string()));
    info.push(("vugva_hot_pct", format!("{:.0}%", if vmt.num_chunks() > 0 { in_vram as f64 / vmt.num_chunks() as f64 * 100.0 } else { 0.0 })));
    info
}
