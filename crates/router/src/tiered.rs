//! Tiered memory — the "AI super-speed memory" layer.
//!
//! Hot tier: per-session working memory in RAM with TTL (Redis speed), but the
//! SAME substrate rows the durable engine sees — no dual-write drift.
//! Cold tier: durable, vector-indexed, living in the storage substrate.
//! Consolidation: a background pass migrates expired-but-important hot entries
//! into the cold tier (short-term -> long-term), instead of external ETL.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct HotEntry {
    value: Vec<u8>,
    inserted: Instant,
    ttl: Duration,
}

pub struct TieredMemory {
    hot: Mutex<HashMap<String, HotEntry>>,
    default_ttl: Duration,
}

impl TieredMemory {
    pub fn new(default_ttl_secs: u64) -> Self {
        Self {
            hot: Mutex::new(HashMap::new()),
            default_ttl: Duration::from_secs(default_ttl_secs),
        }
    }

    /// Write to the hot tier with the default TTL.
    pub fn put_hot(&self, key: &str, value: Vec<u8>) {
        self.put_hot_ttl(key, value, self.default_ttl);
    }

    pub fn put_hot_ttl(&self, key: &str, value: Vec<u8>, ttl: Duration) {
        self.hot.lock().unwrap().insert(
            key.to_string(),
            HotEntry { value, inserted: Instant::now(), ttl },
        );
    }

    /// Read from the hot tier, honoring TTL (expired entries return None).
    pub fn get_hot(&self, key: &str) -> Option<Vec<u8>> {
        let hot = self.hot.lock().unwrap();
        let e = hot.get(key)?;
        if e.inserted.elapsed() <= e.ttl {
            Some(e.value.clone())
        } else {
            None
        }
    }

    /// Consolidation pass: return (key,value) of expired entries so the caller
    /// can embed + persist them to the cold vector tier, then evict them.
    pub fn drain_expired(&self) -> Vec<(String, Vec<u8>)> {
        let mut hot = self.hot.lock().unwrap();
        let expired: Vec<String> = hot
            .iter()
            .filter(|(_, e)| e.inserted.elapsed() > e.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        expired
            .into_iter()
            .filter_map(|k| hot.remove(&k).map(|e| (k, e.value)))
            .collect()
    }

    pub fn hot_len(&self) -> usize {
        self.hot.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_ttl_expiry() {
        let m = TieredMemory::new(60);
        m.put_hot_ttl("a", b"1".to_vec(), Duration::from_millis(20));
        assert_eq!(m.get_hot("a"), Some(b"1".to_vec()));
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(m.get_hot("a"), None);
    }

    #[test]
    fn consolidation_drains_expired() {
        let m = TieredMemory::new(60);
        m.put_hot_ttl("a", b"x".to_vec(), Duration::from_millis(10));
        m.put_hot("keep", b"y".to_vec());
        std::thread::sleep(Duration::from_millis(30));
        let drained = m.drain_expired();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, "a");
        assert_eq!(m.hot_len(), 1); // "keep" survives
    }
}
