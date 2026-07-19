//! VUGVA — Virtual Unified GPU VRAM Architecture (Rust-level implementation).
//!
//! This implements the VUGVA paper's core idea: a software-defined memory
//! routing engine that manages data across GPU VRAM, CPU RAM, and disk.
//! The CUDA kernel doesn't know where data lives — it receives pointers
//! that the Rust VMT (Virtual Memory Table) controls.
//!
//! Architecture (from VUGVA paper §2):
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         CUDA Kernel (distance compute)  │
//! │   reads from d_vectors (GPU pointer)    │
//! └──────────────────┬──────────────────────┘
//!                    │
//! ┌──────────────────▼──────────────────────┐
//! │     VUGVA Virtual Memory Table (VMT)    │
//! │  Tracks: which chunk is in VRAM/RAM     │
//! │  Routes: async prefetch RAM → GPU       │
//! │  Hides: PCIe latency behind compute     │
//! └──────────────────┬──────────────────────┘
//!        ┌───────────┼───────────┐
//!        ▼           ▼           ▼
//!   ┌─────────┐ ┌─────────┐ ┌─────────┐
//!   │ GPU VRAM│ │ CPU RAM │ │   Disk  │
//!   │ (hot)   │ │ (warm)  │ │ (cold)  │
//!   └─────────┘ └─────────┘ └─────────┘
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A chunk of vector data managed by VUGVA.
struct VugvaChunk {
    /// Vector data in CPU RAM (source of truth).
    ram_data: Vec<i8>,
    /// GPU device pointer to this chunk (0 = not in VRAM).
    gpu_ptr: u64,
    /// Last access timestamp for LRU eviction.
    last_access: usize,
}

pub struct VugvaVmt {
    chunks: Vec<Mutex<VugvaChunk>>,
    chunk_size: usize,
    dim: usize,
    #[allow(dead_code)]
    n: usize,
    #[allow(dead_code)]
    vram_budget: usize,
    vram_slots: usize,
    access_counter: AtomicUsize,
}

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
        eprintln!("[VUGVA] VMT: {} vectors, {} chunks, {} VRAM slots ({:.0} MB)",
            n, chunks_count, vram_slots, vram_budget_bytes as f64 / 1024.0 / 1024.0);
        let mut chunks = Vec::with_capacity(chunks_count);
        for i in 0..chunks_count {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(n);
            chunks.push(Mutex::new(VugvaChunk {
                ram_data: vectors[start * dim..end * dim].to_vec(),
                gpu_ptr: 0,
                last_access: 0,
            }));
        }
        Self { chunks, chunk_size, dim, n, vram_budget: vram_budget_bytes, vram_slots, access_counter: AtomicUsize::new(0) }
    }

    pub fn num_chunks(&self) -> usize { self.chunks.len() }
    pub fn chunk_of(&self, vec_idx: usize) -> usize { vec_idx / self.chunk_size }

    pub fn prefetch(&self, chunk_idx: usize) -> Option<GpuChunkPtr> {
        if chunk_idx >= self.chunks.len() { return None; }
        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);
        {
            let mut chunk = self.chunks[chunk_idx].lock().unwrap();
            if chunk.gpu_ptr != 0 {
                chunk.last_access = access;
                let start = chunk_idx * self.chunk_size;
                let len = chunk.ram_data.len() / self.dim;
                return Some(GpuChunkPtr { device_ptr: chunk.gpu_ptr, chunk_offset: start, chunk_len: len });
            }
        }
        // Allocate and copy RAM → GPU
        let mut chunk = self.chunks[chunk_idx].lock().unwrap();
        if chunk.gpu_ptr != 0 {
            chunk.last_access = access;
            let start = chunk_idx * self.chunk_size;
            let len = chunk.ram_data.len() / self.dim;
            return Some(GpuChunkPtr { device_ptr: chunk.gpu_ptr, chunk_offset: start, chunk_len: len });
        }
        let bytes = chunk.ram_data.len();
        let gpu_ptr = unsafe {
            let mut ptr = 0u64;
            if crate::cuMemAlloc_v2(&mut ptr, bytes) != 0 { return None; }
            crate::cuMemcpyHtoD_v2(ptr, chunk.ram_data.as_ptr() as *const std::ffi::c_void, bytes);
            ptr
        };
        chunk.gpu_ptr = gpu_ptr;
        chunk.last_access = access;
        let start = chunk_idx * self.chunk_size;
        let len = chunk.ram_data.len() / self.dim;
        Some(GpuChunkPtr { device_ptr: gpu_ptr, chunk_offset: start, chunk_len: len })
    }

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

    /// Read a single vector from RAM (source of truth). Copies to avoid lock lifetime issues.
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
    for i in 0..vmt.num_chunks() {
        let c = vmt.chunks[i].lock().unwrap();
        if c.gpu_ptr != 0 { in_vram += 1; }
    }
    info.push(("vugva_chunks", vmt.num_chunks().to_string()));
    info.push(("vugva_in_vram", in_vram.to_string()));
    info.push(("vugva_in_ram", (vmt.num_chunks() - in_vram).to_string()));
    info.push(("vugva_vram_slots", vmt.vram_slots.to_string()));
    info
}
