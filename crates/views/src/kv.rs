//! KV view — Redis-style key/value over the substrate.
//! Keyspace convention: "kv:" + user key.

use std::sync::Arc;
use storage::{Engine, Value};

pub struct Kv {
    engine: Arc<Engine>,
}

fn k(key: &str) -> Vec<u8> {
    let mut b = b"kv:".to_vec();
    b.extend_from_slice(key.as_bytes());
    b
}

impl Kv {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    pub fn set(&self, key: &str, val: &[u8]) -> std::io::Result<()> {
        self.engine.put(k(key), Value::Bytes(val.to_vec())).map(|_| ())
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        match self.engine.get(&k(key)) {
            Some(Value::Bytes(b)) => Some(b),
            Some(Value::Int(i)) => Some(i.to_string().into_bytes()),
            _ => None,
        }
    }

    pub fn del(&self, key: &str) -> std::io::Result<bool> {
        let existed = self.engine.get(&k(key)).is_some();
        self.engine.delete(k(key))?;
        Ok(existed)
    }

    /// Atomic increment-by (INCRBY). Returns the new value.
    pub fn incr_by(&self, key: &str, by: i64) -> Result<i64, String> {
        loop {
            let mut txn = self.engine.begin();
            let cur = match txn.get(&k(key)) {
                Some(Value::Int(i)) => i,
                Some(Value::Bytes(b)) => String::from_utf8_lossy(&b)
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| "value is not an integer".to_string())?,
                None => 0,
                _ => return Err("wrong type".to_string()),
            };
            let next = cur + by;
            txn.put(k(key), Value::Int(next));
            match txn.commit() {
                Ok(_) => return Ok(next),
                Err(storage::TxnError::Conflict) => continue, // retry on contention
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// List keys matching a simple prefix (KEYS prefix*).
    pub fn keys_prefix(&self, prefix: &str) -> Vec<String> {
        let full = k(prefix);
        self.engine
            .scan_prefix(&full, self.engine.snapshot())
            .into_iter()
            .filter_map(|(key, _)| {
                let s = String::from_utf8_lossy(&key);
                s.strip_prefix("kv:").map(|x| x.to_string())
            })
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
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
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
    fn incr() {
        let kv = Kv::new(eng());
        assert_eq!(kv.incr_by("c", 1).unwrap(), 1);
        assert_eq!(kv.incr_by("c", 5).unwrap(), 6);
        assert_eq!(kv.incr_by("c", -2).unwrap(), 4);
    }
}
