/// VUGVA — Virtual Unified GPU VRAM Architecture
///
/// Software-defined memory virtualization that tiers data across:
/// - Tier 1: GPU VRAM (~1µs access, limited capacity)
/// - Tier 2: System RAM (~100µs access, large capacity)
/// - Tier 3: NVMe SSD (~10ms access, largest capacity)
///
/// Core concepts:
/// - **Chunk**: Fixed-size block of data (default 256KB) that moves between tiers
/// - **VMT (Virtual Memory Table)**: Maps chunk IDs → physical location + status
/// - **LRU Eviction**: When VRAM is full, least-recently-used chunks are evicted to RAM
/// - **Lookahead Prefetch**: Predicts upcoming chunks from graph traversal history
/// - **NUMA Router**: Routes data to optimal memory tier based on access pattern

pub mod chunk;
pub mod vmt;
pub mod eviction;
pub mod prefetch;

pub use chunk::{Chunk, ChunkId, ChunkState, Tier};
pub use vmt::VugvaVmt;
pub use eviction::LruEvictor;
pub use prefetch::LookaheadTracker;

/// Configuration for the VUGVA memory manager.
#[derive(Clone)]
pub struct VugvaConfig {
    pub chunk_size: usize,        // bytes per chunk (default 256KB)
    pub vram_capacity: usize,     // max bytes in GPU VRAM tier
    pub ram_capacity: usize,      // max bytes in system RAM tier
    pub nvme_capacity: usize,     // max bytes in NVMe tier (0 = unlimited)
    pub prefetch_window: usize,   // how many chunks ahead to prefetch
    pub lru_high_watermark: f64,  // evict when VRAM usage exceeds this (0.0-1.0)
    pub lru_low_watermark: f64,   // target after eviction (0.0-1.0)
}

impl Default for VugvaConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024,        // 256KB
            vram_capacity: 6 * 1024 * 1024 * 1024, // 6GB (leave 2GB for model)
            ram_capacity: 32 * 1024 * 1024 * 1024,  // 32GB
            nvme_capacity: 0,              // unlimited
            prefetch_window: 16,
            lru_high_watermark: 0.85,
            lru_low_watermark: 0.70,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = VugvaConfig::default();
        assert_eq!(cfg.chunk_size, 256 * 1024);
        assert_eq!(cfg.vram_capacity, 6 * 1024 * 1024 * 1024);
        assert_eq!(cfg.prefetch_window, 16);
    }
}
