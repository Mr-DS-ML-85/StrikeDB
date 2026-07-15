//! Time-series view — append points under a series, scan by time range.
//!
//! Keyspace convention: `ts:{<series>}:be(ts)[8]:be(seq)[8]`.
//! The `{<series>}` is a Redis-style **hash tag** — the storage layer routes
//! ALL keys with the same tag into the same shard, so a whole series lives
//! in one BTreeMap. TSRANGE then becomes a single-shard range scan
//! (`Engine::scan_pinned`) instead of touching all 32 shards + merging +
//! sorting — the p50 drops dramatically because we skip 31 useless rwlock
//! acquires per query.
//!
//! `latest(series, n)` uses `Engine::scan_pinned_reverse` for the "give me
//! the last N samples for this dashboard" pattern — O(log n + N) from the
//! pinned shard, no full-range materialization.

use std::sync::Arc;
use std::sync::Mutex;
use storage::{Engine, Value};

pub struct TimeSeries {
    engine: Arc<Engine>,
    /// Per-series monotonic sequence counter, persisted durably in the engine
    /// under "tsmeta:{<series>}" so uniqueness survives a restart.
    seq: Mutex<std::collections::HashMap<String, u64>>,
}

fn prefix(series: &str) -> Vec<u8> {
    // `ts:{<series>}:` — the {} is the hash-tag; shard_of hashes only what
    // is inside the braces, so every point of `series` lands in one shard.
    let mut b = b"ts:{".to_vec();
    b.extend_from_slice(series.as_bytes());
    b.extend_from_slice(b"}:");
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
    // Meta also uses the same hash tag so read + write use the same shard.
    let mut b = b"tsmeta:{".to_vec();
    b.extend_from_slice(series.as_bytes());
    b.push(b'}');
    b
}

impl TimeSeries {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            seq: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Append an integer value at time `ts`. Convenience wrapper around
    /// `append_f` — dashboards should prefer the float form.
    pub fn append(&self, series: &str, ts: u64, val: i64) -> std::io::Result<()> {
        self.append_f(series, ts, val as f64)
    }

    /// Append a **float** value at time `ts` (nanos/millis — caller's unit).
    /// Duplicate timestamps are preserved via a unique per-series sequence.
    /// Metric workloads (cpu%, temperature, latency in ms) always want floats.
    pub fn append_f(&self, series: &str, ts: u64, val: f64) -> std::io::Result<()> {
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
                (point_key(series, ts, next), Value::Float(val)),
            ])
            .map(|_| ())
    }

    /// Range query [from, to] inclusive, returning (ts, value) in time order.
    /// Inverted ranges (from > to) return an empty vec.
    /// **Single-shard** thanks to hash-tag routing on the series.
    pub fn range(&self, series: &str, from: u64, to: u64) -> Vec<(u64, f64)> {
        if from > to {
            return Vec::new();
        }
        let series_prefix = prefix(series);
        let mut start = series_prefix.clone();
        start.extend_from_slice(&from.to_be_bytes());
        let mut end = series_prefix.clone();
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
        // `hint_key = series_prefix` guarantees we read exactly the shard
        // where this series lives — skipping 31 rwlock acquires per query.
        self.engine
            .scan_pinned(&series_prefix, &start, &end, self.engine.snapshot())
            .into_iter()
            .filter_map(|(key, v)| decode_point(&key, v))
            .collect()
    }

    /// Dashboard primitive: **last N points** for this series, newest N
    /// selected in O(log n + N) via a reverse scan on the pinned shard.
    /// Real-time UIs never want the full history — they want the tail.
    pub fn latest(&self, series: &str, limit: usize) -> Vec<(u64, f64)> {
        if limit == 0 {
            return Vec::new();
        }
        let series_prefix = prefix(series);
        let start = series_prefix.clone();
        // upper bound = prefix + 0xFF*17 so the range covers every point
        let mut end = series_prefix.clone();
        end.extend(std::iter::repeat(0xFFu8).take(17));
        self.engine
            .scan_pinned_reverse(&series_prefix, &start, &end,
                                 self.engine.snapshot(), limit)
            .into_iter()
            .filter_map(|(key, v)| decode_point(&key, v))
            .collect()
    }

    /// Simple aggregate: average over a range.
    pub fn avg(&self, series: &str, from: u64, to: u64) -> Option<f64> {
        let pts = self.range(series, from, to);
        if pts.is_empty() {
            return None;
        }
        let sum: f64 = pts.iter().map(|(_, v)| *v).sum();
        Some(sum / pts.len() as f64)
    }
}

/// Parse the trailing `be(ts)[8]:be(seq)[8]` off a point key.
/// Accepts both Int (legacy) and Float (post-fix) values so old WALs still read.
fn decode_point(key: &[u8], v: Value) -> Option<(u64, f64)> {
    if key.len() < 17 {
        return None;
    }
    let tail = &key[key.len() - 17..key.len() - 9];
    let ts = u64::from_be_bytes(tail.try_into().ok()?);
    match v {
        Value::Float(f) => Some((ts, f)),
        Value::Int(i) => Some((ts, i as f64)),
        _ => None,
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
        assert_eq!(r, vec![(100, 10.0), (200, 20.0)]);
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
        assert_eq!(r[0], (50, 5.0)); // time ordering preserved
        assert_eq!(r[1..], vec![(100, 11.0), (100, 12.0), (100, 13.0)]);
    }

    #[test]
    fn inverted_range_is_empty() {
        let ts = TimeSeries::new(eng());
        ts.append("x", 700, 7).unwrap();
        assert!(ts.range("x", 700, 100).is_empty());
    }

    #[test]
    fn latest_returns_last_n_in_order() {
        let ts = TimeSeries::new(eng());
        for i in 0..100u64 {
            ts.append("s", i * 10, i as i64).unwrap();
        }
        let last5 = ts.latest("s", 5);
        assert_eq!(last5.len(), 5);
        // Must be the LAST 5 (95, 96, 97, 98, 99) in ascending order.
        assert_eq!(last5[0], (950, 95.0));
        assert_eq!(last5[4], (990, 99.0));
    }

    #[test]
    fn latest_smaller_than_series_ok() {
        let ts = TimeSeries::new(eng());
        ts.append("s", 5, 5).unwrap();
        assert_eq!(ts.latest("s", 10), vec![(5, 5.0)]);
        assert!(ts.latest("s", 0).is_empty());
    }

    #[test]
    fn append_f_and_range_returns_floats() {
        let ts = TimeSeries::new(eng());
        ts.append_f("cpu", 100, 42.5).unwrap();
        ts.append_f("cpu", 200, 66.25).unwrap();
        let r = ts.range("cpu", 0, 1000);
        assert_eq!(r, vec![(100, 42.5), (200, 66.25)]);
        assert_eq!(ts.avg("cpu", 0, 1000), Some(54.375));
    }
}
