//! Unified MVCC storage engine — the single substrate.
//!
//! Every key maps to a version chain (Vec<Version>) ordered by commit timestamp.
//! Readers take a snapshot (a timestamp) and see the newest version <= snapshot.
//! Writers buffer in a Txn, then commit atomically: the WAL record is flushed
//! first (durability), then the in-memory version chains are updated.
//!
//! This is the "one storage substrate" from the architecture — tables, KV,
//! vectors, time-series and the CDC log are all just key conventions over this.

use crate::value::Value;
use crate::wal::Wal;
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};

pub type Key = Vec<u8>;

#[derive(Clone, Debug)]
pub struct Version {
    pub ts: u64,
    pub value: Value, // Value::Tombstone means deleted at ts
}

/// A key's version history plus an O(1) cache of the newest visible value.
///
/// Reads (GET/INCR/snapshot) almost always want the *latest* committed value.
/// Scanning the whole `versions` vec on every read is O(N) and N grows with
/// every write to the key — counters in particular become pathologically slow.
/// We keep `latest`/`latest_ts` so `get_at` is O(1) in the hot case and only
/// falls back to a reverse scan when the snapshot is older than the newest
/// version (rare: historical MVCC reads).
#[derive(Default)]
struct Chain {
    latest: Option<Value>,
    latest_ts: u64,
    versions: Vec<Version>,
}

impl Chain {
    /// Append a committed version and refresh the O(1) cache.
    fn push(&mut self, ts: u64, value: Value) {
        self.latest_ts = ts;
        match &value {
            Value::Tombstone => self.latest = None,
            v => self.latest = Some(v.clone()),
        }
        self.versions.push(Version { ts, value });
    }

    /// Newest version visible at `snapshot`, honoring tombstones. O(1) when the
    /// newest version is already <= snapshot (the common case).
    fn visible(&self, snapshot: u64) -> Option<&Value> {
        if let Some(v) = &self.latest {
            if self.latest_ts <= snapshot {
                return Some(v);
            }
        }
        for ver in self.versions.iter().rev() {
            if ver.ts <= snapshot {
                return match &ver.value {
                    Value::Tombstone => None,
                    v => Some(v),
                };
            }
        }
        None
    }
}

/// A committed mutation, as written to the WAL and broadcast to subscribers.
#[derive(Clone, Debug)]
pub struct Mutation {
    pub key: Key,
    pub value: Value,
    pub ts: u64,
}

impl Mutation {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.ts.to_le_bytes());
        out.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.key);
        out.extend_from_slice(&self.value.encode());
        out
    }
    fn decode(buf: &[u8]) -> Option<Mutation> {
        let ts = u64::from_le_bytes(buf.get(0..8)?.try_into().ok()?);
        let kl = u32::from_le_bytes(buf.get(8..12)?.try_into().ok()?) as usize;
        let key = buf.get(12..12 + kl)?.to_vec();
        let value = Value::decode(buf.get(12 + kl..)?)?;
        Some(Mutation { key, value, ts })
    }
}

/// Callback invoked for every committed mutation (used by the reactive layer).
pub type Subscriber = Arc<dyn Fn(&Mutation) + Send + Sync>;

/// A pending write that has reserved a commit timestamp and is queued for the
/// background flush thread to durably persist. All writes that arrive during a
/// single fsync are batched into ONE WAL flush — true GROUP COMMIT (the same
/// trick Postgres/MySQL/Kafka use to scale durable writes across many cores).
struct PendingWrite {
    muts: Vec<Mutation>,
    state: Mutex<WriteState>,
    cond: Condvar,
}

struct WriteState {
    done: bool,
    err: Option<String>,
}

/// Number of independent shard maps. Reads/writes on different shards run in
/// parallel — this is the "sharded RwLock" pattern that DashMap and every
/// modern KV store uses to escape the single-writer bottleneck of one big
/// RwLock. 32 balances lock-contention reduction vs cache-line waste; a good
/// sweet spot up to ~64 cores.
pub const SHARD_COUNT: usize = 32;

#[inline]
fn shard_of(key: &[u8]) -> usize {
    // FNV-1a — stable, deterministic, no allocation.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h as usize) & (SHARD_COUNT - 1)
}

pub struct Engine {
    /// Sharded data map. `SHARD_COUNT` must be a power of two for the mask
    /// above. Each shard holds its own BTreeMap under its own RwLock.
    shards: Vec<RwLock<BTreeMap<Key, Chain>>>,
    wal: Mutex<Wal>,
    clock: AtomicU64,
    subscribers: RwLock<Vec<Subscriber>>,
    /// Queue of writes awaiting the background flusher.
    write_queue: Mutex<Vec<Arc<PendingWrite>>>,
    /// Wakes the flusher when new writes arrive (and on shutdown).
    queue_cv: Condvar,
    /// Set by `Drop` to stop the flusher thread.
    shutdown: AtomicBool,
    flusher: Mutex<Option<JoinHandle<()>>>,
}

impl Engine {
    /// Open an engine backed by a WAL at `path`, replaying it on startup.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Arc<Self>> {
        // SHARD_COUNT must be a power of two for the shard mask.
        assert!(SHARD_COUNT.is_power_of_two(), "SHARD_COUNT must be power of two");

        let mut wal = Wal::open(path)?;
        let records = wal.replay()?;

        // Build per-shard maps directly to avoid a merge step.
        let mut shard_data: Vec<BTreeMap<Key, Chain>> =
            (0..SHARD_COUNT).map(|_| BTreeMap::new()).collect();
        let mut max_ts = 0u64;
        for rec in records {
            if let Some(m) = Mutation::decode(&rec) {
                max_ts = max_ts.max(m.ts);
                let s = shard_of(&m.key);
                shard_data[s].entry(m.key).or_default().push(m.ts, m.value);
            }
        }
        let shards: Vec<RwLock<BTreeMap<Key, Chain>>> =
            shard_data.into_iter().map(RwLock::new).collect();

        let engine = Arc::new(Self {
            shards,
            wal: Mutex::new(wal),
            clock: AtomicU64::new(max_ts),
            subscribers: RwLock::new(Vec::new()),
            write_queue: Mutex::new(Vec::new()),
            queue_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            flusher: Mutex::new(None),
        });

        // Start the background group-commit flusher. It wakes on new writes or
        // shutdown, drains the whole queue in one WAL append + single fsync.
        {
            let mut guard = engine.flusher.lock().unwrap();
            *guard = Some(Self::spawn_flusher(Arc::clone(&engine)));
        }
        Ok(engine)
    }

    /// Monotonic timestamp source (also serves as the logical commit clock).
    pub fn now(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Take a read snapshot at the current logical time.
    pub fn snapshot(&self) -> u64 {
        self.clock.load(Ordering::SeqCst)
    }

    /// Register a commit subscriber (reactive sync / CDC).
    pub fn subscribe(&self, cb: Subscriber) {
        self.subscribers.write().unwrap().push(cb);
    }

    /// Point read as of `snapshot`: newest non-future version, honoring
    /// tombstones. O(1) via the latest-value cache in the common case.
    pub fn get_at(&self, key: &[u8], snapshot: u64) -> Option<Value> {
        let data = self.shards[shard_of(key)].read().unwrap();
        let chain = data.get(key)?;
        chain.visible(snapshot).cloned()
    }

    /// Convenience read at the latest snapshot.
    pub fn get(&self, key: &[u8]) -> Option<Value> {
        self.get_at(key, self.snapshot())
    }

    /// Range scan [start, end) as of snapshot, returning live key/value pairs.
    /// Sharded: touches every shard's map (cheap because BTreeMap range is
    /// O(log n + m)), then merges results in-order.
    pub fn scan(&self, start: &[u8], end: &[u8], snapshot: u64) -> Vec<(Key, Value)> {
        let mut out: Vec<(Key, Value)> = Vec::new();
        for shard in &self.shards {
            let data = shard.read().unwrap();
            for (k, chain) in data.range(start.to_vec()..end.to_vec()) {
                if let Some(v) = chain.visible(snapshot) {
                    out.push((k.clone(), v.clone()));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Scan all keys sharing a prefix, as of snapshot.
    pub fn scan_prefix(&self, prefix: &[u8], snapshot: u64) -> Vec<(Key, Value)> {
        let mut end = prefix.to_vec();
        // compute the least key greater than all keys with this prefix
        while let Some(last) = end.last().copied() {
            if last == 0xFF {
                end.pop();
            } else {
                *end.last_mut().unwrap() = last + 1;
                break;
            }
        }
        if end.is_empty() {
            // prefix was all 0xFF — scan to the very end of every shard.
            let mut out: Vec<(Key, Value)> = Vec::new();
            for shard in &self.shards {
                let data = shard.read().unwrap();
                for (k, chain) in data.range(prefix.to_vec()..) {
                    if let Some(v) = chain.visible(snapshot) {
                        out.push((k.clone(), v.clone()));
                    }
                }
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            return out;
        }
        self.scan(prefix, &end, snapshot)
    }

    /// Begin a transaction snapshotted at the current logical time.
    pub fn begin(self: &Arc<Self>) -> Txn {
        Txn {
            engine: Arc::clone(self),
            snapshot: self.snapshot(),
            writes: BTreeMap::new(),
            reads: Vec::new(),
        }
    }

    /// Internal: enqueue a batch of mutations for the background group-commit
    /// flusher, then block until they are durably persisted + made visible.
    ///
    /// GROUP COMMIT (the real fix for weak multicore write scaling):
    ///   * The caller reserves a commit timestamp and pushes its mutations onto
    ///     `write_queue`, then wakes the flusher thread.
    ///   * The flusher thread waits for either new work or shutdown, then drains
    ///     the ENTIRE queue in one WAL append + a SINGLE fsync. A burst of N
    ///     concurrent writers therefore costs one fsync, not N — exactly how
    ///     Postgres/MySQL/Kafka turn a 1-core fsync ceiling into N-core
    ///     throughput.
    ///   * After the durable flush, the (short) data write-lock applies the
    ///     version chains, then wakes all waiters. Readers are unaffected because
    ///     versions are appended under a fresh timestamp (snapshot isolation).
    fn commit_batch(&self, writes: BTreeMap<Key, Value>) -> io::Result<u64> {
        let ts = self.now();
        let muts: Vec<Mutation> = writes
            .into_iter()
            .map(|(key, value)| Mutation { key, value, ts })
            .collect();

        let pending = Arc::new(PendingWrite {
            muts,
            state: Mutex::new(WriteState { done: false, err: None }),
            cond: Condvar::new(),
        });
        {
            let mut q = self.write_queue.lock().unwrap();
            q.push(Arc::clone(&pending));
        }
        self.queue_cv.notify_all();

        // Wait for this write's group to be flushed.
        let mut state = pending.state.lock().unwrap();
        while !state.done {
            state = pending
                .cond
                .wait(state)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "commit wait poisoned"))?;
        }
        match &state.err {
            Some(e) => Err(io::Error::new(io::ErrorKind::Other, e.clone())),
            None => Ok(ts),
        }
    }

/// Background group-commit flusher: drains the write queue, appends every
/// pending mutation in one WAL write, fsyncs ONCE, applies version chains, and
/// wakes all waiters. Stops when `shutdown` is set.
fn spawn_flusher(engine: Arc<Engine>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        // Collect the current batch of pending writes.
        let batch: Vec<Arc<PendingWrite>> = {
            let mut q = engine.write_queue.lock().unwrap();
            // Wait until there is work, or it's time to shut down.
            while q.is_empty() && !engine.shutdown.load(Ordering::SeqCst) {
                let _g = engine
                    .queue_cv
                    .wait(q)
                    .expect("group-commit flusher condvar poisoned");
                q = _g;
            }
            if engine.shutdown.load(Ordering::SeqCst) && q.is_empty() {
                return;
            }
            q.drain(..).collect()
        };

        // Durable step: one append + one fsync for the whole batch.
        let flush_result: io::Result<()> = (|| {
            let mut wal = engine.wal.lock().unwrap();
            for pw in &batch {
                for m in &pw.muts {
                    wal.append(&m.encode())?;
                }
            }
            // single fsync amortised across the whole group
            wal.sync()
        })();

        // Visibility + broadcast happen ONLY if the durable flush succeeded.
        // Otherwise the mutations were never made durable, and subscribers must
        // not observe a "commit" that never happened.
        //
        // Sharded apply: group mutations by shard and take each shard's write
        // lock only when we have work for it. Writers touching disjoint shards
        // don't serialize on a global data lock any more.
        if flush_result.is_ok() {
            let mut per_shard: Vec<Vec<(Key, u64, Value)>> =
                (0..SHARD_COUNT).map(|_| Vec::new()).collect();
            for pw in &batch {
                for m in &pw.muts {
                    let s = shard_of(&m.key);
                    per_shard[s].push((m.key.clone(), m.ts, m.value.clone()));
                }
            }
            for (i, items) in per_shard.into_iter().enumerate() {
                if items.is_empty() {
                    continue;
                }
                let mut data = engine.shards[i].write().unwrap();
                for (k, ts, v) in items {
                data.entry(k).or_default().push(ts, v);
                }
                }
            let subs = engine.subscribers.read().unwrap();
            for pw in &batch {
                for m in &pw.muts {
                    for s in subs.iter() {
                        s(m);
                    }
                }
            }
        }
        // Wake every waiter with the outcome.
        for pw in batch {
            let mut st = pw.state.lock().unwrap();
            st.done = true;
            if let Err(e) = &flush_result {
                st.err = Some(e.to_string());
            }
            pw.cond.notify_all();
        }
    })
}

    /// One-shot durable write outside an explicit transaction.
    pub fn put(&self, key: Key, value: Value) -> io::Result<u64> {
        let mut b = BTreeMap::new();
        b.insert(key, value);
        self.commit_batch(b)
    }

    /// One-shot durable delete (writes a tombstone).
    pub fn delete(&self, key: Key) -> io::Result<u64> {
        let mut b = BTreeMap::new();
        b.insert(key, Value::Tombstone);
        self.commit_batch(b)
    }

    /// Atomic batch write: all `kvs` are made visible under ONE commit timestamp
    /// (and durably WAL-flushed together). Used by views that need to persist
    /// two keys atomically (e.g. a time-series point + its sequence counter).
    pub fn put_batch(&self, kvs: Vec<(Key, Value)>) -> io::Result<u64> {
        let mut b = BTreeMap::new();
        for (k, v) in kvs {
            b.insert(k, v);
        }
        self.commit_batch(b)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.queue_cv.notify_all();
        if let Some(h) = self.flusher.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

/// A transaction with snapshot-isolation + optimistic write-conflict detection.
pub struct Txn {
    engine: Arc<Engine>,
    snapshot: u64,
    writes: BTreeMap<Key, Value>,
    reads: Vec<Key>,
}

impl Txn {
    /// Read within the txn: sees own writes, else the snapshot view.
    pub fn get(&mut self, key: &[u8]) -> Option<Value> {
        if let Some(v) = self.writes.get(key) {
            return if matches!(v, Value::Tombstone) { None } else { Some(v.clone()) };
        }
        self.reads.push(key.to_vec());
        self.engine.get_at(key, self.snapshot)
    }

    /// Buffer a write.
    pub fn put(&mut self, key: Key, value: Value) {
        self.writes.insert(key, value);
    }

    /// Buffer a delete.
    pub fn delete(&mut self, key: Key) {
        self.writes.insert(key, Value::Tombstone);
    }

    /// Commit: abort if any key we read was modified after our snapshot
    /// (optimistic concurrency control), otherwise apply atomically.
    pub fn commit(self) -> Result<u64, TxnError> {
        // Sharded conflict check: take each shard's read lock only if we
        // actually read a key from it.
        for k in &self.reads {
            let data = self.engine.shards[shard_of(k)].read().unwrap();
            if let Some(chain) = data.get(k) {
                // A version committed after our snapshot means a conflict.
                if chain.latest_ts > self.snapshot {
                    return Err(TxnError::Conflict);
                }
            }
        }
        self.engine
            .commit_batch(self.writes)
            .map_err(TxnError::Io)
    }
}

#[derive(Debug)]
pub enum TxnError {
    Conflict,
    Io(io::Error),
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TxnError::Conflict => write!(f, "transaction conflict"),
            TxnError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}
impl std::error::Error for TxnError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("dbstrike_eng_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn put_get_delete() {
        let e = Engine::open(tmp("a.wal")).unwrap();
        e.put(b"k".to_vec(), Value::Bytes(b"v".to_vec())).unwrap();
        assert_eq!(e.get(b"k"), Some(Value::Bytes(b"v".to_vec())));
        e.delete(b"k".to_vec()).unwrap();
        assert_eq!(e.get(b"k"), None);
    }

    #[test]
    fn snapshot_isolation() {
        let e = Engine::open(tmp("b.wal")).unwrap();
        e.put(b"k".to_vec(), Value::Int(1)).unwrap();
        let snap = e.snapshot();
        e.put(b"k".to_vec(), Value::Int(2)).unwrap();
        // old snapshot still sees 1
        assert_eq!(e.get_at(b"k", snap), Some(Value::Int(1)));
        // latest sees 2
        assert_eq!(e.get(b"k"), Some(Value::Int(2)));
    }

    #[test]
    fn recovery_replays_wal() {
        let path = tmp("c.wal");
        {
            let e = Engine::open(&path).unwrap();
            e.put(b"x".to_vec(), Value::Int(7)).unwrap();
            e.put(b"y".to_vec(), Value::Bytes(b"z".to_vec())).unwrap();
        }
        let e = Engine::open(&path).unwrap();
        assert_eq!(e.get(b"x"), Some(Value::Int(7)));
        assert_eq!(e.get(b"y"), Some(Value::Bytes(b"z".to_vec())));
    }

    #[test]
    fn txn_conflict_detected() {
        let e = Engine::open(tmp("d.wal")).unwrap();
        e.put(b"k".to_vec(), Value::Int(0)).unwrap();
        let mut t1 = e.begin();
        let _ = t1.get(b"k"); // read at snapshot
        // concurrent write bumps the version past t1's snapshot
        e.put(b"k".to_vec(), Value::Int(99)).unwrap();
        t1.put(b"k".to_vec(), Value::Int(1));
        assert!(matches!(t1.commit(), Err(TxnError::Conflict)));
    }

    #[test]
    fn prefix_scan() {
        let e = Engine::open(tmp("e.wal")).unwrap();
        e.put(b"user:1".to_vec(), Value::Int(1)).unwrap();
        e.put(b"user:2".to_vec(), Value::Int(2)).unwrap();
        e.put(b"post:1".to_vec(), Value::Int(3)).unwrap();
        let got = e.scan_prefix(b"user:", e.snapshot());
        assert_eq!(got.len(), 2);
    }
}
