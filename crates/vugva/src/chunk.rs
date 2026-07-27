/// Chunk: Fixed-size data block that moves between memory tiers.

/// Memory tier where a chunk resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Vram,
    Ram,
    Nvme,
}

/// Chunk identifier (unique within the VMT).
pub type ChunkId = u64;

/// State of a chunk in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    /// Chunk is actively being used (in VRAM, high priority)
    Hot,
    /// Chunk was recently accessed (in VRAM or RAM)
    Warm,
    /// Chunk hasn't been accessed recently (evicted to RAM or NVMe)
    Cold,
    /// Chunk is being transferred between tiers
    Transferring,
}

/// A chunk of data that lives in one of the memory tiers.
pub struct Chunk {
    pub id: ChunkId,
    pub tier: Tier,
    pub state: ChunkState,
    pub data: Vec<u8>,
    pub size: usize,
    pub last_access: u64,       // monotonic timestamp
    pub access_count: u64,      // total accesses (for frequency-based eviction)
    pub dirty: bool,            // has uncommitted writes
}

impl Chunk {
    pub fn new(id: ChunkId, tier: Tier, data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            id,
            tier,
            state: ChunkState::Hot,
            data,
            size,
            last_access: 0,
            access_count: 0,
            dirty: false,
        }
    }

    /// Create an empty chunk placeholder (for metadata-only tracking).
    pub fn placeholder(id: ChunkId, tier: Tier, size: usize) -> Self {
        Self {
            id,
            tier,
            state: ChunkState::Cold,
            data: Vec::new(),
            size,
            last_access: 0,
            access_count: 0,
            dirty: false,
        }
    }

    /// Mark chunk as accessed (updates timestamp and count).
    pub fn touch(&mut self, timestamp: u64) {
        self.last_access = timestamp;
        self.access_count += 1;
        self.state = ChunkState::Hot;
    }

    /// Demote chunk to a lower tier.
    pub fn demote(&mut self, new_tier: Tier) {
        self.tier = new_tier;
        self.state = ChunkState::Cold;
    }

    /// Check if chunk is in VRAM.
    pub fn in_vram(&self) -> bool {
        self.tier == Tier::Vram
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let c = Chunk::new(1, Tier::Vram, vec![0u8; 1024]);
        assert_eq!(c.id, 1);
        assert_eq!(c.tier, Tier::Vram);
        assert_eq!(c.size, 1024);
        assert!(c.in_vram());
    }

    #[test]
    fn test_chunk_touch() {
        let mut c = Chunk::new(1, Tier::Vram, vec![0u8; 1024]);
        c.touch(100);
        assert_eq!(c.last_access, 100);
        assert_eq!(c.access_count, 1);
        c.touch(200);
        assert_eq!(c.last_access, 200);
        assert_eq!(c.access_count, 2);
    }

    #[test]
    fn test_chunk_demote() {
        let mut c = Chunk::new(1, Tier::Vram, vec![0u8; 1024]);
        c.demote(Tier::Ram);
        assert_eq!(c.tier, Tier::Ram);
        assert!(!c.in_vram());
    }
}
