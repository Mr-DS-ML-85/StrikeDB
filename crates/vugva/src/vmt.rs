/// VMT — Virtual Memory Table
///
/// Maps chunk IDs to physical locations across tiers.
/// This is the core data structure of VUGVA.

use std::collections::HashMap;
use crate::chunk::{Chunk, ChunkId, ChunkState, Tier};
use crate::VugvaConfig;

/// Metadata entry for a chunk (without the actual data).
#[derive(Debug, Clone)]
pub struct ChunkMeta {
    pub id: ChunkId,
    pub tier: Tier,
    pub state: ChunkState,
    pub size: usize,
    pub last_access: u64,
    pub access_count: u64,
}

/// The Virtual Memory Table tracks all chunks and their locations.
pub struct VugvaVmt {
    config: VugvaConfig,
    /// Chunk metadata indexed by ChunkId
    chunks: HashMap<ChunkId, ChunkMeta>,
    /// Actual chunk data (only for chunks in RAM tier; VRAM/NVMe managed externally)
    ram_store: HashMap<ChunkId, Vec<u8>>,
    /// VRAM usage in bytes
    vram_used: usize,
    /// RAM usage in bytes
    ram_used: usize,
    /// Monotonic clock for access timestamps
    clock: u64,
}

impl VugvaVmt {
    pub fn new(config: VugvaConfig) -> Self {
        Self {
            config,
            chunks: HashMap::new(),
            ram_store: HashMap::new(),
            vram_used: 0,
            ram_used: 0,
            clock: 0,
        }
    }

    /// Register a new chunk into the VMT at a specific tier.
    pub fn insert(&mut self, chunk: Chunk) -> ChunkId {
        let id = chunk.id;
        let size = chunk.size;
        let tier = chunk.tier;

        self.chunks.insert(id, ChunkMeta {
            id,
            tier,
            state: chunk.state,
            size,
            last_access: self.clock,
            access_count: 1,
        });

        if tier == Tier::Vram {
            self.vram_used += size;
        } else if tier == Tier::Ram {
            self.ram_store.insert(id, chunk.data);
            self.ram_used += size;
        }

        id
    }

    /// Lookup a chunk by ID. Returns None if not found.
    pub fn get(&mut self, id: ChunkId) -> Option<Tier> {
        self.clock += 1;
        if let Some(meta) = self.chunks.get_mut(&id) {
            meta.last_access = self.clock;
            meta.access_count += 1;
            meta.state = ChunkState::Hot;
            Some(meta.tier)
        } else {
            None
        }
    }

    /// Promote a chunk to VRAM (from RAM or NVMe).
    pub fn promote_to_vram(&mut self, id: ChunkId) -> Option<Vec<u8>> {
        self.clock += 1;
        let meta = self.chunks.get_mut(&id)?;
        if meta.tier == Tier::Vram {
            meta.last_access = self.clock;
            return None; // already in VRAM
        }

        let old_tier = meta.tier;
        let size = meta.size;

        // Remove from old tier
        if old_tier == Tier::Ram {
            let data = self.ram_store.remove(&id).unwrap_or_default();
            self.ram_used -= size;
            meta.tier = Tier::Vram;
            meta.state = ChunkState::Hot;
            meta.last_access = self.clock;
            self.vram_used += size;
            return Some(data);
        }

        // NVMe → VRAM (data would come from disk I/O in real impl)
        meta.tier = Tier::Vram;
        meta.state = ChunkState::Hot;
        meta.last_access = self.clock;
        self.vram_used += size;
        Some(vec![0u8; size]) // placeholder
    }

    /// Evict a chunk from VRAM to RAM (LRU candidate).
    pub fn evict_to_ram(&mut self, id: ChunkId, data: Vec<u8>) -> bool {
        let meta = match self.chunks.get_mut(&id) {
            Some(m) => m,
            None => return false,
        };
        if meta.tier != Tier::Vram { return false; }

        let size = meta.size;
        meta.tier = Tier::Ram;
        meta.state = ChunkState::Cold;
        self.vram_used -= size;
        self.ram_store.insert(id, data);
        self.ram_used += size;
        true
    }

    /// Find LRU candidate for eviction (oldest access in VRAM).
    pub fn lru_candidate(&self) -> Option<ChunkId> {
        self.chunks.values()
            .filter(|c| c.tier == Tier::Vram)
            .min_by_key(|c| c.last_access)
            .map(|c| c.id)
    }

    /// Get VRAM usage ratio (0.0 to 1.0).
    pub fn vram_usage_ratio(&self) -> f64 {
        if self.config.vram_capacity == 0 { return 0.0; }
        self.vram_used as f64 / self.config.vram_capacity as f64
    }

    /// Get RAM usage in bytes.
    pub fn ram_usage(&self) -> usize {
        self.ram_used
    }

    /// Get VRAM usage in bytes.
    pub fn vram_usage(&self) -> usize {
        self.vram_used
    }

    /// Total number of tracked chunks.
    pub fn chunks(&self) -> &std::collections::HashMap<ChunkId, ChunkMeta> {
        &self.chunks
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Check if VRAM needs eviction.
    pub fn needs_eviction(&self) -> bool {
        self.vram_usage_ratio() > self.config.lru_high_watermark
    }

    /// Remove a chunk entirely from the VMT.
    pub fn remove(&mut self, id: ChunkId) -> Option<Vec<u8>> {
        let meta = self.chunks.remove(&id)?;
        if meta.tier == Tier::Vram {
            self.vram_used -= meta.size;
        } else if meta.tier == Tier::Ram {
            self.ram_used -= meta.size;
        }
        self.ram_store.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmt_insert_and_lookup() {
        let cfg = VugvaConfig::default();
        let mut vmt = VugvaVmt::new(cfg);
        let chunk = Chunk::new(1, Tier::Vram, vec![1, 2, 3, 4]);
        vmt.insert(chunk);
        assert_eq!(vmt.get(1), Some(Tier::Vram));
        assert_eq!(vmt.vram_usage(), 4);
    }

    #[test]
    fn test_vmt_eviction() {
        let cfg = VugvaConfig { vram_capacity: 8, ..Default::default() };
        let mut vmt = VugvaVmt::new(cfg);
        vmt.insert(Chunk::new(1, Tier::Vram, vec![0u8; 4]));
        vmt.insert(Chunk::new(2, Tier::Vram, vec![0u8; 4]));
        assert!(vmt.needs_eviction());
        let lru = vmt.lru_candidate().unwrap();
        let data = vmt.promote_to_vram(lru).unwrap_or_default();
        vmt.evict_to_ram(lru, data);
        assert_eq!(vmt.vram_usage(), 4);
    }

    #[test]
    fn test_vmt_promote() {
        let cfg = VugvaConfig::default();
        let mut vmt = VugvaVmt::new(cfg);
        vmt.insert(Chunk::new(1, Tier::Ram, vec![1, 2, 3]));
        assert_eq!(vmt.ram_usage(), 3);
        vmt.promote_to_vram(1);
        assert_eq!(vmt.vram_usage(), 3);
        assert_eq!(vmt.ram_usage(), 0);
    }
}
