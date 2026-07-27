/// LRU Evictor — manages eviction from VRAM to lower tiers.

use crate::vmt::VugvaVmt;

/// Eviction statistics for monitoring.
#[derive(Debug, Default)]
pub struct EvictionStats {
    pub total_evictions: u64,
    pub bytes_evicted: u64,
}

/// LRU-based eviction policy.
pub struct LruEvictor {
    stats: EvictionStats,
}

impl LruEvictor {
    pub fn new() -> Self {
        Self { stats: EvictionStats::default() }
    }

    /// Evict chunks until VRAM usage drops below the low watermark.
    pub fn evict_to_target(&mut self, vmt: &mut VugvaVmt) -> usize {
        let mut evicted = 0;
        while vmt.needs_eviction() {
            if let Some(lru_id) = vmt.lru_candidate() {
                let placeholder = vec![0u8; 4096];
                if vmt.evict_to_ram(lru_id, placeholder) {
                    self.stats.total_evictions += 1;
                    self.stats.bytes_evicted += 4096;
                    evicted += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        evicted
    }

    pub fn stats(&self) -> &EvictionStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Tier};
    use crate::VugvaConfig;

    #[test]
    fn test_eviction_reduces_vram() {
        let cfg = VugvaConfig { vram_capacity: 16, ..Default::default() };
        let mut vmt = VugvaVmt::new(cfg);
        vmt.insert(Chunk::new(1, Tier::Vram, vec![0u8; 8]));
        vmt.insert(Chunk::new(2, Tier::Vram, vec![0u8; 8]));
        vmt.insert(Chunk::new(3, Tier::Vram, vec![0u8; 8]));

        let mut evictor = LruEvictor::new();
        let evicted = evictor.evict_to_target(&mut vmt);
        assert!(evicted > 0);
        assert!(vmt.vram_usage() <= 16);
    }
}
