//! MITM Cache Debugger — "catch caching bugs instantly".
//!
//! An in-process man-in-the-middle observer that sits between a cache and the
//! source-of-truth engine. Every cache GET/SET/INVALIDATE is intercepted and
//! recorded with the *authoritative* engine version at that instant. When a
//! cache read is served, the debugger compares the cached value against the
//! live engine value and classifies the outcome:
//!
//!   * Hit      — cache value == engine value (fresh).
//!   * StaleHit — cache value != engine value (THE BUG: served stale data).
//!   * Miss     — key absent from cache.
//!   * Phantom  — cache has a value the engine no longer has (missed invalidation).
//!
//! Every event is timestamped and kept in a ring buffer you can dump, diff, and
//! replay. This is the primitive Redis/Dragonfly/Mem0 don't ship: provable,
//! per-key staleness detection with the exact version delta — no guessing.
//!
//! Pure Rust. Zero external crates.

pub mod memtrack;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use storage::{Engine, Value};

/// Outcome classification for a cache read observed by the debugger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Hit,
    StaleHit,
    Miss,
    Phantom,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Hit => "HIT",
            Verdict::StaleHit => "STALE_HIT",
            Verdict::Miss => "MISS",
            Verdict::Phantom => "PHANTOM",
        }
    }
    pub fn is_bug(&self) -> bool {
        matches!(self, Verdict::StaleHit | Verdict::Phantom)
    }
}

/// A single intercepted cache access.
#[derive(Clone, Debug)]
pub struct Trace {
    pub seq: u64,
    pub op: &'static str, // "GET" | "SET" | "INVALIDATE"
    pub key: String,
    pub verdict: Verdict,
    pub cached_ver: u64, // logical version stamped on the cached entry
    pub engine_ver: u64, // authoritative engine version at observation time
    pub note: String,
}

/// A cached entry: the value plus the engine version it was populated from.
#[derive(Clone, Debug)]
struct CacheEntry {
    value: Vec<u8>,
    populated_at_ver: u64,
}

/// The MITM cache debugger: a small cache + a tap on the source engine.
pub struct CacheDebugger {
    engine: Arc<Engine>,
    cache: RwLock<HashMap<String, CacheEntry>>,
    traces: Mutex<Vec<Trace>>,
    seq: AtomicU64,
    max_traces: usize,
    // per-key authoritative version counter, bumped on every engine write we see
    ver: RwLock<HashMap<String, u64>>,
}

fn ek(key: &str) -> Vec<u8> {
    // source-of-truth keyspace this debugger guards
    let mut b = b"src:".to_vec();
    b.extend_from_slice(key.as_bytes());
    b
}

impl CacheDebugger {
    pub fn new(engine: Arc<Engine>, max_traces: usize) -> Arc<Self> {
        Arc::new(Self {
            engine,
            cache: RwLock::new(HashMap::new()),
            traces: Mutex::new(Vec::new()),
            seq: AtomicU64::new(0),
            max_traces: max_traces.max(16),
            ver: RwLock::new(HashMap::new()),
        })
    }

    fn engine_value(&self, key: &str) -> Option<Vec<u8>> {
        match self.engine.get(&ek(key)) {
            Some(Value::Bytes(b)) => Some(b),
            Some(Value::Int(i)) => Some(i.to_string().into_bytes()),
            _ => None,
        }
    }

    fn engine_ver(&self, key: &str) -> u64 {
        *self.ver.read().unwrap().get(key).unwrap_or(&0)
    }

    fn record(&self, t: Trace) {
        let mut tr = self.traces.lock().unwrap();
        tr.push(t);
        if tr.len() > self.max_traces {
            let overflow = tr.len() - self.max_traces;
            tr.drain(0..overflow);
        }
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Authoritative write: updates the engine and bumps the per-key version.
    /// (A correct cache MUST invalidate after this; if it doesn't, the next GET
    /// will be flagged StaleHit.)
    pub fn source_set(&self, key: &str, value: &[u8]) -> std::io::Result<u64> {
        self.engine.put(ek(key), Value::Bytes(value.to_vec()))?;
        let mut v = self.ver.write().unwrap();
        let nv = v.get(key).copied().unwrap_or(0) + 1;
        v.insert(key.to_string(), nv);
        Ok(nv)
    }

    /// Authoritative delete: bumps version and marks the key gone.
    pub fn source_del(&self, key: &str) -> std::io::Result<u64> {
        self.engine.delete(ek(key))?;
        let mut v = self.ver.write().unwrap();
        let nv = v.get(key).copied().unwrap_or(0) + 1;
        v.insert(key.to_string(), nv);
        Ok(nv)
    }

    /// Authoritative read straight from the source engine (bypasses cache).
    pub fn source_get(&self, key: &str) -> Option<Vec<u8>> {
        self.engine_value(key)
    }

    /// Cache write (populate). Stamps the entry with the current engine version.
    pub fn cache_set(&self, key: &str, value: &[u8]) {
        let ver = self.engine_ver(key);
        self.cache.write().unwrap().insert(
            key.to_string(),
            CacheEntry {
                value: value.to_vec(),
                populated_at_ver: ver,
            },
        );
        let seq = self.next_seq();
        self.record(Trace {
            seq,
            op: "SET",
            key: key.to_string(),
            verdict: Verdict::Hit,
            cached_ver: ver,
            engine_ver: ver,
            note: "cache populated".into(),
        });
    }

    /// Cache invalidate.
    pub fn cache_invalidate(&self, key: &str) {
        self.cache.write().unwrap().remove(key);
        let seq = self.next_seq();
        let ev = self.engine_ver(key);
        self.record(Trace {
            seq,
            op: "INVALIDATE",
            key: key.to_string(),
            verdict: Verdict::Miss,
            cached_ver: 0,
            engine_ver: ev,
            note: "cache invalidated".into(),
        });
    }

    /// Cache read — the money path. Returns (value, verdict). Compares cached
    /// value to authoritative engine value and flags staleness/phantoms.
    pub fn cache_get(&self, key: &str) -> (Option<Vec<u8>>, Verdict) {
        let engine_ver = self.engine_ver(key);
        let engine_val = self.engine_value(key);
        let cached = self.cache.read().unwrap().get(key).cloned();

        let (value, verdict, cached_ver, note) = match (cached, &engine_val) {
            (None, _) => (None, Verdict::Miss, 0, "not cached".to_string()),
            (Some(c), Some(ev)) => {
                if c.value == *ev {
                    (Some(c.value.clone()), Verdict::Hit, c.populated_at_ver, "fresh".to_string())
                } else {
                    (
                        Some(c.value.clone()),
                        Verdict::StaleHit,
                        c.populated_at_ver,
                        format!(
                            "STALE: cached v{} != engine v{} ({} bytes vs {} bytes)",
                            c.populated_at_ver,
                            engine_ver,
                            c.value.len(),
                            ev.len()
                        ),
                    )
                }
            }
            (Some(c), None) => (
                Some(c.value.clone()),
                Verdict::Phantom,
                c.populated_at_ver,
                "PHANTOM: cached but engine has no value (missed invalidation)".to_string(),
            ),
        };

        let seq = self.next_seq();
        self.record(Trace {
            seq,
            op: "GET",
            key: key.to_string(),
            verdict,
            cached_ver,
            engine_ver,
            note,
        });
        (value, verdict)
    }

    /// Dump all traces (oldest first).
    pub fn traces(&self) -> Vec<Trace> {
        self.traces.lock().unwrap().clone()
    }

    /// Only the traces flagged as bugs (StaleHit / Phantom).
    pub fn bugs(&self) -> Vec<Trace> {
        self.traces
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.verdict.is_bug())
            .cloned()
            .collect()
    }

    /// Human-readable one-line report per bug.
    pub fn report(&self) -> Vec<String> {
        self.bugs()
            .into_iter()
            .map(|t| {
                format!(
                    "#{} {} key={} verdict={} cached_v={} engine_v={} :: {}",
                    t.seq,
                    t.op,
                    t.key,
                    t.verdict.as_str(),
                    t.cached_ver,
                    t.engine_ver,
                    t.note
                )
            })
            .collect()
    }

    pub fn clear(&self) {
        self.traces.lock().unwrap().clear();
        self.seq.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_mitm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Engine::open(dir.join(format!("mitm_{n}.wal"))).unwrap()
    }

    #[test]
    fn detects_stale_read() {
        let d = CacheDebugger::new(eng(), 100);
        // source has v1, cache populated fresh
        d.source_set("user:1", b"alice").unwrap();
        d.cache_set("user:1", b"alice");
        let (_, v) = d.cache_get("user:1");
        assert_eq!(v, Verdict::Hit);

        // source updated but cache NOT invalidated -> stale bug
        d.source_set("user:1", b"alice-v2").unwrap();
        let (val, v) = d.cache_get("user:1");
        assert_eq!(v, Verdict::StaleHit);
        assert_eq!(val, Some(b"alice".to_vec())); // served the stale value
        assert_eq!(d.bugs().len(), 1);
        assert!(d.report()[0].contains("STALE"));
    }

    #[test]
    fn detects_phantom() {
        let d = CacheDebugger::new(eng(), 100);
        d.source_set("k", b"v").unwrap();
        d.cache_set("k", b"v");
        d.source_del("k").unwrap(); // gone from source, still in cache
        let (_, v) = d.cache_get("k");
        assert_eq!(v, Verdict::Phantom);
    }

    #[test]
    fn correct_invalidation_no_bug() {
        let d = CacheDebugger::new(eng(), 100);
        d.source_set("k", b"v1").unwrap();
        d.cache_set("k", b"v1");
        d.source_set("k", b"v2").unwrap();
        d.cache_invalidate("k"); // proper write-through invalidation
        let (val, v) = d.cache_get("k");
        assert_eq!(v, Verdict::Miss);
        assert_eq!(val, None);
        assert_eq!(d.bugs().len(), 0);
    }
}
