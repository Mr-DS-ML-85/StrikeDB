//! KV view — Redis-style key/value over the substrate.
//! Keyspace convention: "kv:" + user key.
//!
//! The DB is a *byte* store, not a text store. Keys are forwarded as raw
//! RESP argument bytes (SET/GET/DEL/KEYS use the `_b` variants) so
//! non-UTF-8 keys (e.g. `\xff\xfe...`) round-trip exactly — no
//! `from_utf8_lossy` corruption. The str-based wrappers remain for
//! callers that already hold text keys (unit tests, internal helpers).

use std::sync::Arc;
use storage::{Engine, Value};

pub struct Kv {
    engine: Arc<Engine>,
}

fn k(key: &[u8]) -> Vec<u8> {
    let mut b = b"kv:".to_vec();
    b.extend_from_slice(key);
    b
}

impl Kv {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
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
        Ok(self.del_many(&[key])? == 1)
    }

    /// DEL/UNLINK of many keys in ONE commit (one fsync), writing tombstones
    /// only for keys that exist — deleting a missing key used to append a
    /// tombstone to the WAL anyway. Returns how many existed.
    pub fn del_many(&self, keys: &[&[u8]]) -> std::io::Result<usize> {
        let snap = self.engine.snapshot();
        let mut dels: Vec<(Vec<u8>, Value)> = Vec::new();
        for key in keys {
            let kk = k(key);
            if self.engine.get_at(&kk, snap).is_some() && !dels.iter().any(|(d, _)| *d == kk) {
                dels.push((kk, Value::Tombstone));
            }
        }
        let n = dels.len();
        if n > 0 {
            self.engine.put_batch(dels)?;
        }
        Ok(n)
    }

    /// Atomic read-modify-write of one key. `f` sees the current value and
    /// returns `(write, result)`: `write` = `None` (leave as is),
    /// `Some(Some(v))` (set) or `Some(None)` (delete). Retries on conflict;
    /// the engine validates the read at its commit point, so this is a true
    /// compare-and-set (SETNX, SET NX/XX/GET, GETSET, GETDEL, APPEND).
    pub fn update<R>(
        &self,
        key: &[u8],
        mut f: impl FnMut(Option<Vec<u8>>) -> (Option<Option<Vec<u8>>>, R),
    ) -> std::io::Result<R> {
        let kk = k(key);
        loop {
            let mut txn = self.engine.begin();
            let cur = match txn.get(&kk) {
                Some(Value::Bytes(b)) => Some(b),
                Some(Value::Int(i)) => Some(i.to_string().into_bytes()),
                _ => None,
            };
            let (write, r) = f(cur);
            match write {
                None => return Ok(r),
                Some(Some(v)) => txn.put(kk.clone(), Value::Bytes(v)),
                Some(None) => txn.delete(kk.clone()),
            }
            match txn.commit() {
                Ok(_) => return Ok(r),
                Err(storage::TxnError::Conflict) => continue,
                Err(storage::TxnError::Io(e)) => return Err(e),
            }
        }
    }

    /// INCRBY: key is a raw byte key (no UTF-8 assumption), but the
    /// *value* is still parsed as text (integers are text). Returns the new value.
    pub fn incr_by_lossy(&self, key: &[u8], by: i64) -> Result<i64, String> {
        // Resolved atomically on the engine's commit path: no global lock
        // (the old `txn_lock` serialized every INCR on every key across a
        // full fsync — ~3k ops/s) and no conflict retries; concurrent INCRs
        // batch into one group-commit fsync like SETs do.
        self.engine.incr_by(k(key), by).map_err(|e| e.to_string())?
    }

    /// Pipelined INCRBYs sharing one group-commit fsync (see
    /// `Engine::incr_many`). Results are per-op, in order.
    pub fn incr_many(&self, ops: Vec<(Vec<u8>, i64)>) -> Result<Vec<Result<i64, String>>, String> {
        self.engine
            .incr_many(ops.into_iter().map(|(key, by)| (k(&key), by)).collect())
            .map_err(|e| e.to_string())
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
    /// Redis glob KEYS: scan only the literal prefix of the pattern, then
    /// filter with the full glob (`*2`, `x?`, `[ab]*` used to match nothing:
    /// the old code just trimmed trailing `*` and did a prefix scan).
    pub fn keys_glob(&self, pattern: &[u8], matches: impl Fn(&[u8]) -> bool) -> Vec<Vec<u8>> {
        let lit = pattern.iter().position(|b| matches!(b, b'*' | b'?' | b'[' | b'\\')).unwrap_or(pattern.len());
        if lit == pattern.len() {
            // No glob metacharacters: keep DB-Strike's documented legacy
            // behaviour (`KEYS user:` = prefix scan) for existing clients.
            return self.keys_prefix(pattern);
        }
        self.keys_prefix(&pattern[..lit]).into_iter().filter(|k| matches(k)).collect()
    }

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
