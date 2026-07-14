//! Table view — relational rows with primary keys and simple secondary lookups.
//! Keyspace: "tbl:" + table + ":" + pk  ->  Value::Row.

use std::collections::BTreeMap;
use std::sync::Arc;
use storage::{Engine, Value};

pub struct Tables {
    engine: Arc<Engine>,
}

fn row_key(table: &str, pk: &str) -> Vec<u8> {
    let mut b = b"tbl:".to_vec();
    b.extend_from_slice(table.as_bytes());
    b.push(b':');
    b.extend_from_slice(pk.as_bytes());
    b
}
fn table_prefix(table: &str) -> Vec<u8> {
    let mut b = b"tbl:".to_vec();
    b.extend_from_slice(table.as_bytes());
    b.push(b':');
    b
}

pub type Row = BTreeMap<String, Vec<u8>>;

impl Tables {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    /// Insert or replace a row by primary key.
    pub fn upsert(&self, table: &str, pk: &str, row: Row) -> std::io::Result<()> {
        self.engine.put(row_key(table, pk), Value::Row(row)).map(|_| ())
    }

    pub fn get(&self, table: &str, pk: &str) -> Option<Row> {
        match self.engine.get(&row_key(table, pk)) {
            Some(Value::Row(r)) => Some(r),
            _ => None,
        }
    }

    pub fn delete(&self, table: &str, pk: &str) -> std::io::Result<()> {
        self.engine.delete(row_key(table, pk)).map(|_| ())
    }

    /// Full scan of a table (pk, row).
    pub fn scan(&self, table: &str) -> Vec<(String, Row)> {
        let prefix = table_prefix(table);
        self.engine
            .scan_prefix(&prefix, self.engine.snapshot())
            .into_iter()
            .filter_map(|(key, v)| {
                let pk = String::from_utf8_lossy(&key[prefix.len()..]).to_string();
                if let Value::Row(r) = v {
                    Some((pk, r))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Filtered scan: rows where column == value (linear predicate scan).
    pub fn filter_eq(&self, table: &str, col: &str, val: &[u8]) -> Vec<(String, Row)> {
        self.scan(table)
            .into_iter()
            .filter(|(_, r)| r.get(col).map(|v| v.as_slice() == val).unwrap_or(false))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_tbl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Engine::open(dir.join(format!("tbl_{n}.wal"))).unwrap()
    }

    fn row(pairs: &[(&str, &str)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.as_bytes().to_vec())).collect()
    }

    #[test]
    fn upsert_get_scan_filter() {
        let t = Tables::new(eng());
        t.upsert("users", "1", row(&[("name", "ada"), ("tier", "pro")])).unwrap();
        t.upsert("users", "2", row(&[("name", "bob"), ("tier", "free")])).unwrap();
        t.upsert("users", "3", row(&[("name", "cid"), ("tier", "pro")])).unwrap();

        assert_eq!(t.get("users", "1").unwrap().get("name").unwrap(), b"ada");
        assert_eq!(t.scan("users").len(), 3);
        let pros = t.filter_eq("users", "tier", b"pro");
        assert_eq!(pros.len(), 2);
    }
}
