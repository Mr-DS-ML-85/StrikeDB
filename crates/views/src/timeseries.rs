//! Time-series view — append points under a series, scan by time range.
//! Keyspace convention: "ts:" + series + ":" + be(ts)[8] + ":" + be(seq)[8].
//! The trailing sequence makes every point UNIQUE even when timestamps collide
//! (out-of-order / duplicate-ts ingestion), and the be(ts) prefix keeps the
//! series lexically ordered by time so range scans stay O(log n + k).

use std::sync::Arc;
use std::sync::Mutex;
use storage::{Engine, Value};

pub struct TimeSeries {
    engine: Arc<Engine>,
    /// Per-series monotonic sequence counter, persisted durably in the engine
    /// under "tsmeta:<series>" so uniqueness survives a restart.
    seq: Mutex<std::collections::HashMap<String, u64>>,
}

fn prefix(series: &str) -> Vec<u8> {
    let mut b = b"ts:".to_vec();
    b.extend_from_slice(series.as_bytes());
    b.push(b':');
    b
}

fn point_key(series: &str, ts: u64, seq: u64) -> Vec<u8> {
    let mut b = prefix(series);
    b.extend_from_slice(&ts.to_be_bytes());
    b.push(b':');
    b.extend_from_slice(&seq.to_be_bytes());
    b
}

fn meta_key(series: &str) -> Vec<u8> {
    let mut b = b"tsmeta:".to_vec();
    b.extend_from_slice(series.as_bytes());
    b
}

impl TimeSeries {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            seq: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Append a value at time `ts` (nanos/millis — caller's unit).
    /// Duplicate timestamps are preserved (unique per-series sequence).
    pub fn append(&self, series: &str, ts: u64, val: i64) -> std::io::Result<()> {
        // Reserve the next sequence for this series, durably.
        let next = {
            let mut guard = self.seq.lock().unwrap();
            let cur = guard.get(series).copied().unwrap_or_else(|| {
                // First sight of this series in this process: seed from durable meta.
                match self.engine.get(&meta_key(series)) {
                    Some(Value::Int(n)) => n as u64,
                    _ => 0,
                }
            });
            let next = cur + 1;
            guard.insert(series.to_string(), next);
            next
        };
        // Persist the meta counter AND the point in one atomic batch.
        self.engine
            .put_batch(vec![
                (meta_key(series), Value::Int(next as i64)),
                (point_key(series, ts, next), Value::Int(val)),
            ])
            .map(|_| ())
    }

    /// Range query [from, to] inclusive, returning (ts, value) in time order.
    /// Inverted ranges (from > to) return an empty vec.
    pub fn range(&self, series: &str, from: u64, to: u64) -> Vec<(u64, i64)> {
        if from > to {
            return Vec::new();
        }
        let mut start = prefix(series);
        start.extend_from_slice(&from.to_be_bytes());
        let mut end = prefix(series);
        end.extend_from_slice(&to.to_be_bytes());
        // make `to` inclusive by nudging the end bound up by one
        for byte in end.iter_mut().rev() {
            if *byte == 0xFF {
                *byte = 0;
            } else {
                *byte += 1;
                break;
            }
        }
        self.engine
            .scan(&start, &end, self.engine.snapshot())
            .into_iter()
            .filter_map(|(key, v)| {
                // Layout: "ts:" + series + ":" + be(ts)[8] + ":" + be(seq)[8]
                // From the end: seq = [len-8..len], ":" = [len-9], ts = [len-17..len-9].
                if key.len() < 17 {
                    return None;
                }
                let tail = &key[key.len() - 17..key.len() - 9];
                let ts = u64::from_be_bytes(tail.try_into().ok()?);
                if let Value::Int(i) = v {
                    Some((ts, i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Simple aggregate: average over a range.
    pub fn avg(&self, series: &str, from: u64, to: u64) -> Option<f64> {
        let pts = self.range(series, from, to);
        if pts.is_empty() {
            return None;
        }
        let sum: i64 = pts.iter().map(|(_, v)| *v).sum();
        Some(sum as f64 / pts.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_ts_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let p = dir.join(format!("ts_{n}.wal"));
        Engine::open(p).unwrap()
    }

    #[test]
    fn append_range_avg() {
        let ts = TimeSeries::new(eng());
        ts.append("cpu", 100, 10).unwrap();
        ts.append("cpu", 200, 20).unwrap();
        ts.append("cpu", 300, 30).unwrap();
        let r = ts.range("cpu", 100, 200);
        assert_eq!(r, vec![(100, 10), (200, 20)]);
        assert_eq!(ts.avg("cpu", 100, 300), Some(20.0));
    }

    #[test]
    fn duplicate_timestamps_preserved() {
        let ts = TimeSeries::new(eng());
        ts.append("ev", 100, 11).unwrap();
        ts.append("ev", 100, 12).unwrap(); // duplicate ts
        ts.append("ev", 100, 13).unwrap();
        ts.append("ev", 50, 5).unwrap(); // out of order
        let r = ts.range("ev", 0, 1000);
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], (50, 5)); // time ordering preserved
        assert_eq!(r[1..], vec![(100, 11), (100, 12), (100, 13)]);
    }

    #[test]
    fn inverted_range_is_empty() {
        let ts = TimeSeries::new(eng());
        ts.append("x", 700, 7).unwrap();
        assert!(ts.range("x", 700, 100).is_empty());
    }
}
