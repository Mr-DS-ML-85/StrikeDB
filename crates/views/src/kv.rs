//! KV view — Redis-style key/value over the substrate.
//! Keyspace convention: "kv:" + user key.
//!
//! The DB is a *byte* store, not a text store. Keys are forwarded as raw
//! RESP argument bytes (SET/GET/DEL/KEYS use the `_b` variants) so
//! non-UTF-8 keys (e.g. `\xff\xfe...`) round-trip exactly — no
//! `from_utf8_lossy` corruption. The str-based wrappers remain for
//! callers that already hold text keys (unit tests, internal helpers).

use std::sync::{Arc, Mutex};
use storage::{Engine, Value, TxnError};

pub struct Kv {
    engine: Arc<Engine>,
    /// Serializes read-modify-write transactions (INCR/INCRBY). The engine's
    /// OCC validates conflicts BEFORE the write is enqueued to the group
    /// commit flusher, so N racing txns that read the same snapshot all pass
    /// validation and then blind-overwrite each other — 8 clients × 250
    /// INCRBYs collapsed to a final value of 250. Serializing RMW at this
    /// level restores exact counter semantics (Redis INCR is serialized too);
    /// plain SET/GET never take this lock.
    txn_lock: Mutex<()>,
}

fn k(key: &[u8]) -> Vec<u8> {
    let mut b = b"kv:".to_vec();
    b.extend_from_slice(key);
    b
}

impl Kv {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine, txn_lock: Mutex::new(()) }
    }

    // ── Byte-key API (raw RESP arg bytes) ────────────────────────────
    pub fn set_b(&self, key: &[u8], val: &[u8]) -> std::io::Result<()> {
        self.engine.put(k(key), Value::Bytes(val.to_vec())).map(|_| ())
    }

    pub fn get_b(&self, key: &[u8]) -> Option<Vec<u8>> {
        match self.engine.get(&k(key)) {
            Some(Value::Bytes(b)) => Some(b),
            Some(Value::Int(i)) => Some(i.to_string().into_bytes()),
            _ => None,
        }
    }

    pub fn del_b(&self, key: &[u8]) -> std::io::Result<bool> {
        let existed = self.engine.get(&k(key)).is_some();
        self.engine.delete(k(key))?;
        Ok(existed)
    }

    /// INCRBY: key is a raw byte key (no UTF-8 assumption), but the
    /// *value* is still parsed as text (integers are text). Returns the new value.
    pub fn incr_by_lossy(&self, key: &[u8], by: i64) -> Result<i64, String> {
        // Hold the RMW lock across read→write so concurrent counter ops
        // cannot validate against the same snapshot and overwrite each other
        // in the group-commit queue.
        let _serial = self.txn_lock.lock().unwrap();
        let kkey = k(key);
        loop {
            let mut txn = self.engine.begin();
            let cur = match txn.get(&kkey) {
                Some(Value::Int(i)) => i,
                Some(Value::Bytes(b)) => String::from_utf8_lossy(&b)
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| "value is not an integer".to_string())?,
                None => 0,
                _ => return Err("wrong type".to_string()),
            };
            let next = cur + by;
            txn.put(kkey.clone(), Value::Int(next));
            match txn.commit() {
                Ok(_) => return Ok(next),
                Err(TxnError::Conflict) => continue, // retry on contention
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    // ── Text-key convenience wrappers ─────────────────────────────────
    pub fn set(&self, key: &str, val: &[u8]) -> std::io::Result<()> {
        self.set_b(key.as_bytes(), val)
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.get_b(key.as_bytes())
    }

    pub fn del(&self, key: &str) -> std::io::Result<bool> {
        self.del_b(key.as_bytes())
    }

    pub fn incr_by(&self, key: &str, by: i64) -> Result<i64, String> {
        self.incr_by_lossy(key.as_bytes(), by)
    }

    // ── Batch / scan ───────────────────────────────────────────────────
    /// Coalesced multi-SET: land N (key, value) pairs under ONE commit ts,
    /// ONE WAL append, ONE fsync. Keys arrive pre-prefixed with `kv:`.
    pub fn set_batch(&self, kvs: Vec<(Vec<u8>, Vec<u8>)>) -> std::io::Result<()> {
        if kvs.is_empty() {
            return Ok(());
        }
        let entries: Vec<(Vec<u8>, Value)> = kvs
            .into_iter()
            .map(|(key, val)| (key, Value::Bytes(val)))
            .collect();
        self.engine.put_batch(entries).map(|_| ())
    }

    /// List keys matching a simple prefix (KEYS prefix*). Returns raw key
    /// bytes (minus the `kv:` namespace) so non-UTF-8 keys survive.
    pub fn keys_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        let mut full = b"kv:".to_vec();
        full.extend_from_slice(prefix);
        self.engine
            .scan_prefix(&full, self.engine.snapshot())
            .into_iter()
            .filter_map(|(key, _)| key.strip_prefix(b"kv:").map(|s| s.to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_kv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("kv_{}.wal", rand_suffix()));
        let _ = std::fs::remove_file(&p);
        Engine::open(p).unwrap()
    }
    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    #[test]
    fn set_get_del() {
        let kv = Kv::new(eng());
        kv.set("a", b"1").unwrap();
        assert_eq!(kv.get("a"), Some(b"1".to_vec()));
        assert!(kv.del("a").unwrap());
        assert_eq!(kv.get("a"), None);
    }

    #[test]
    fn non_utf8_key_roundtrip() {
        let kv = Kv::new(eng());
        let key = b"\xff\xfe\x00\x01\x80\x81binary-key".to_vec();
        let val = b"\x00\xff\xfe\xfd".to_vec();
        kv.set_b(&key, &val).unwrap();
        assert_eq!(kv.get_b(&key), Some(val));
    }

    #[test]
    fn incr() {
        let kv = Kv::new(eng());
        assert_eq!(kv.incr_by("c", 1).unwrap(), 1);
        assert_eq!(kv.incr_by("c", 5).unwrap(), 6);
        assert_eq!(kv.incr_by("c", -2).unwrap(), 4);
    }
}
