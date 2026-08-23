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
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
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
/// Cap on how many historical versions we keep per key in the in-memory
/// arena. Once exceeded, the OLDEST version is dropped. Very old snapshots
/// (older than the N most recent commits to a hot key) lose visibility —
/// this is the standard MVCC/GC bound that InnoDB, Postgres, and CockroachDB
/// all impose. For agent-memory / RAG workloads that read the latest state,
/// this is a pure memory win and never observed.
///
/// 8 is enough to cover typical read-modify-write transactions, snapshot
/// isolation across N concurrent txns, and short-lived analytical scans,
/// while capping per-key memory at 8 · sizeof(Version).
pub const MAX_VERSIONS_PER_KEY: usize = 8;

#[derive(Default)]
struct Chain {
    latest: Option<Value>,
    latest_ts: u64,
    versions: Vec<Version>,
}

impl Chain {
    /// Append a committed version and refresh the O(1) cache. Prunes the
    /// oldest version when the chain exceeds `MAX_VERSIONS_PER_KEY` — bounds
    /// memory growth on counters and hot keys.
    fn push(&mut self, ts: u64, value: Value) {
        // Versions can arrive OUT OF TIMESTAMP ORDER. `commit_batch` reserves
        // `ts` from the clock and only then takes the write-queue lock, so two
        // writers can reserve (10, 11) and enqueue (11, 10). The flusher then
        // applies 11 before 10, and blindly assigning `latest_ts = ts` would
        // let the OLDER version win — a lost update. Measured at ~3 in 2000
        // rounds of 32-way same-key contention, so rare but real. WAL replay
        // has the same exposure: records are replayed in file order, which is
        // queue order, not timestamp order.
        //
        // Nothing is ever lost on disk in that scenario — both records are
        // durable — so the crash-recovery tests cannot see it. It is a
        // *visibility* bug, not a durability one.
        if ts >= self.latest_ts {
            self.latest_ts = ts;
            match &value {
                Value::Tombstone => self.latest = None,
                v => self.latest = Some(v.clone()),
            }
        }
        // Keep `versions` sorted by ts, because `visible()` relies on a reverse
        // scan finding the newest version <= snapshot. In the overwhelmingly
        // common in-order case `partition_point` returns `len()` and this is
        // exactly the old `push` — no added cost on the fast path.
        let pos = self.versions.partition_point(|v| v.ts <= ts);
        self.versions.insert(pos, Version { ts, value });
        if self.versions.len() > MAX_VERSIONS_PER_KEY {
            // O(N) shift once per push, but N is tiny (8) so it's cheap.
            // Sorted order makes this the genuinely oldest version, which the
            // previous insertion-ordered vec did not guarantee.
            self.versions.remove(0);
        }
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

// ── WAL record framing ────────────────────────────────────────────────
// Tagged so a whole commit (a `put_batch`, e.g. a bulk load) is ONE WAL
// frame. A frame is atomic: `Wal::replay` either gets the whole frame
// (CRC valid → all mutations apply) or stops at a torn/corrupt frame and
// drops it entirely — never a partial batch. Without this, a crash during
// the append of a 400k-mutation load would replay the first N records and
// leave a half-loaded namespace.
const WAL_TAG_SINGLE: u8 = 0x01;
const WAL_TAG_BATCH: u8 = 0x02;

fn encode_single_record(m: &Mutation) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + m.encode().len());
    out.push(WAL_TAG_SINGLE);
    out.extend_from_slice(&m.encode());
    out
}

fn encode_batch_record(muts: &[Mutation]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(WAL_TAG_BATCH);
    out.extend_from_slice(&(muts.len() as u64).to_le_bytes());
    for m in muts {
        let enc = m.encode();
        out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
        out.extend_from_slice(&enc);
    }
    out
}

/// Decode one WAL frame into zero or more mutations. Accepts the tagged
/// single/batch frames written by the current flusher, plus legacy untagged
/// records (which decode as a single mutation) so an older WAL still opens.
fn decode_records(rec: &[u8]) -> Vec<Mutation> {
    match rec.first() {
        Some(&WAL_TAG_BATCH) => {
            let mut out = Vec::new();
            if rec.len() < 9 {
                return out;
            }
            let count = u64::from_le_bytes(rec[1..9].try_into().unwrap()) as usize;
            let mut p = 9usize;
            for _ in 0..count {
                if p + 4 > rec.len() {
                    break;
                }
                let len = u32::from_le_bytes(rec[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                if p + len > rec.len() {
                    break;
                }
                if let Some(m) = Mutation::decode(&rec[p..p + len]) {
                    out.push(m);
                }
                p += len;
            }
            out
        }
        Some(&WAL_TAG_SINGLE) => Mutation::decode(&rec[1..]).into_iter().collect(),
        _ => Mutation::decode(rec).into_iter().collect(),
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
    /// Full-flush op (`FLUSHALL`): when `Some(backup_path)`, the flusher
    /// fsyncs the live WAL, atomically renames it (plus its `.snap`) to the
    /// backup path, reopens a fresh empty WAL and clears every shard map.
    /// Routing the wipe through this queue serializes it against in-flight
    /// commits on the single flusher thread — it can never interleave with
    /// a batch that was drained before it, and later commits land on the
    /// fresh WAL. `None` for ordinary commits (the overwhelmingly common
    /// case), so the hot path is untouched.
    flush_all: Option<String>,
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

/// Derive the checkpoint-snapshot path from a WAL path (`foo.wal` → `foo.wal.snap`).
fn snap_path_for(wal: &Path) -> PathBuf {
    let mut s = wal.as_os_str().to_owned();
    s.push(".snap");
    PathBuf::from(s)
}

/// Snapshot file format:
///   [u64 record_count]
///   For each record: [u32 payload_len][payload bytes]        (payload = Mutation::encode())
///   [u32 crc32(entire body)]                                 (trailing checksum)
///
/// Writer path: write to `<path>.tmp`, fsync, atomic-rename into place.
fn write_snapshot(
    snap_path: &Path,
    muts: &[Mutation],
) -> io::Result<()> {
    let tmp_path: PathBuf = {
        let mut s = snap_path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    let mut body = Vec::with_capacity(muts.len() * 32);
    body.extend_from_slice(&(muts.len() as u64).to_le_bytes());
    for m in muts {
        let payload = m.encode();
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
    }
    let checksum = crate::crc::crc32(&body);
    body.extend_from_slice(&checksum.to_le_bytes());

    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)?;
    f.write_all(&body)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp_path, snap_path)?;

    // Durability sandwich: the rename is a directory-entry change. If the
    // power dies after the WAL is truncated but before this rename reaches
    // disk, both the snapshot and the log would be gone. fsync the parent
    // directory so the rename is as durable as the file it points at.
    if let Some(parent) = snap_path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn load_snapshot(snap_path: &Path) -> io::Result<Vec<Mutation>> {
    let mut f = File::open(snap_path)?;
    let mut body = Vec::new();
    f.read_to_end(&mut body)?;
    if body.len() < 12 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "snapshot too short"));
    }
    // Trailing 4-byte crc32.
    let (payload, crc_bytes) = body.split_at(body.len() - 4);
    let want_crc = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    if crate::crc::crc32(payload) != want_crc {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "snapshot crc mismatch"));
    }
    let n = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
    let mut p = 8usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if p + 4 > payload.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "snapshot truncated"));
        }
        let len = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if p + len > payload.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "snapshot record overrun"));
        }
        if let Some(m) = Mutation::decode(&payload[p..p + len]) {
            out.push(m);
        }
        p += len;
    }
    Ok(out)
}

/// FNV-1a of a byte slice.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Shard selection for a key.
///
/// Supports **Redis-style hash tags** (`{tag}`): if the key contains a
/// balanced `{...}` region, only the bytes INSIDE the braces are hashed.
/// This lets callers force a group of related keys onto the same shard
/// — critical for time-series and any prefix-scan workload where the
/// alternative is fanning out across all 32 shards.
///
/// Examples:
///   `ts:{cpu}:00000001:00000042`  → hash("cpu")     (co-located with all cpu points)
///   `ts:{mem}:00000002:00000009`  → hash("mem")
///   `kv:user:1`                   → hash("kv:user:1")  (no tag, unchanged)
#[inline]
pub fn shard_of(key: &[u8]) -> usize {
    let hash_input = if let Some(start) = key.iter().position(|&b| b == b'{') {
        // Look for a matching '}' after '{'; if none, fall back to whole key.
        let rest = &key[start + 1..];
        if let Some(end) = rest.iter().position(|&b| b == b'}') {
            if end > 0 {
                &rest[..end]
            } else {
                key
            }
        } else {
            key
        }
    } else {
        key
    };
    (fnv1a(hash_input) as usize) & (SHARD_COUNT - 1)
}

/// Everything the background flusher thread touches, split out of `Engine`
/// so the thread can hold an `Arc<FlushCore>` instead of an `Arc<Engine>`.
///
/// WHY THIS SPLIT EXISTS. The flusher used to be handed a strong
/// `Arc<Engine>`. That is a reference cycle in disguise: the engine's
/// refcount could never fall to zero while the thread lived, and the thread
/// only exited once `Drop for Engine` set `shutdown`. Each waited on the
/// other, so `Drop` was unreachable dead code — the flusher thread leaked
/// once per engine, graceful shutdown never ran, and any Drop-based cleanup
/// was silently skipped.
///
/// `Weak<Engine>` does not fix it either. If the flusher happens to hold a
/// temporary upgraded strong ref at the moment the last external `Arc` is
/// dropped, the refcount reaches zero *on the flusher thread*, so
/// `Engine::drop` runs there and tries to join itself. The condvar it must
/// wait on also lives inside the object being dropped.
///
/// Splitting the state is the fix that has neither problem: `Engine` owns
/// the only handle to the thread and holds a strong `Arc<FlushCore>`; the
/// thread holds a second strong `Arc<FlushCore>` and *no* reference to
/// `Engine`. Dropping the last `Engine` therefore always runs `Drop`, on the
/// dropping thread, which signals `shutdown` and joins. The `FlushCore`
/// itself is freed after the join, when the thread's `Arc` goes away.
struct FlushCore {
    /// Sharded data map. `SHARD_COUNT` must be a power of two for the mask
    /// above. Each shard holds its own BTreeMap under its own RwLock.
    shards: Vec<RwLock<BTreeMap<Key, Chain>>>,
    wal: Mutex<Wal>,
    subscribers: RwLock<Vec<Subscriber>>,
    /// Queue of writes awaiting the background flusher.
    write_queue: Mutex<Vec<Arc<PendingWrite>>>,
    /// Wakes the flusher when new writes arrive (and on shutdown).
    queue_cv: Condvar,
    /// Set by `Drop for Engine` to stop the flusher thread.
    shutdown: AtomicBool,
}

pub struct Engine {
    /// State shared with the flusher thread. See `FlushCore`.
    core: Arc<FlushCore>,
    /// Path of the primary WAL file — needed to derive the checkpoint file
    /// name (`<wal>.snap`) and for atomic rename during `checkpoint()`.
    wal_path: PathBuf,
    clock: AtomicU64,
    /// Owned solely by `Engine`, so `Drop` is the only joiner.
    flusher: Mutex<Option<JoinHandle<()>>>,
    /// Durability mode. If false (opt-in via `DBSTRIKE_SYNC=0`), commit_batch
    /// applies writes directly to shards and skips the WAL entirely — Redis's
    /// default behavior. Trade-off is honest: a crash loses recent writes.
    /// Ideal for sessions / presence / cache / tests. Default is TRUE
    /// (fsync every batch — durable).
    sync_writes: bool,
}

impl Engine {
    /// Open an engine backed by a WAL at `path`. Recovery order:
    ///   1. If a checkpoint snapshot exists at `<path>.snap`, load every
    ///      Mutation from it and apply into shards. This is one dense file
    ///      with one Mutation per key (the compacted world at snapshot time).
    ///   2. Then replay the WAL on top — any commits made AFTER the last
    ///      successful checkpoint. Torn tail in the WAL is detected + dropped
    ///      by `Wal::replay`.
    ///   3. `clock` resumes from `max(snap_ts, wal_max_ts)` so new commits
    ///      never reuse a historical timestamp.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Arc<Self>> {
        // SHARD_COUNT must be a power of two for the shard mask.
        assert!(SHARD_COUNT.is_power_of_two(), "SHARD_COUNT must be power of two");

        let wal_path = path.as_ref().to_path_buf();
        let snap_path = snap_path_for(&wal_path);

        // Build per-shard maps.
        let mut shard_data: Vec<BTreeMap<Key, Chain>> =
            (0..SHARD_COUNT).map(|_| BTreeMap::new()).collect();
        let mut max_ts = 0u64;

        // Step 1 — load snapshot if present.
        if snap_path.exists() {
            match load_snapshot(&snap_path) {
                Ok(muts) => {
                    for m in muts {
                        max_ts = max_ts.max(m.ts);
                        let s = shard_of(&m.key);
                        shard_data[s].entry(m.key).or_default().push(m.ts, m.value);
                    }
                }
                Err(e) => {
                    // A corrupt snapshot must NOT drop the WAL. Log-and-continue:
                    // WAL replay will still reconstruct the full state from
                    // wherever the WAL starts (may be a full history).
                    eprintln!(
                        "warning: snapshot at {} unreadable ({}); falling back to WAL only",
                        snap_path.display(), e
                    );
                }
            }
        }

        // Step 2 — WAL on top.
        let mut wal = Wal::open(&wal_path)?;
        let records = wal.replay()?;
        for rec in records {
            for m in decode_records(&rec) {
                max_ts = max_ts.max(m.ts);
                let s = shard_of(&m.key);
                shard_data[s].entry(m.key).or_default().push(m.ts, m.value);
            }
        }

        let shards: Vec<RwLock<BTreeMap<Key, Chain>>> =
            shard_data.into_iter().map(RwLock::new).collect();

        // Opt-in non-durable mode: DBSTRIKE_SYNC=0 skips the WAL entirely.
        // Matches Redis's default (no AOF fsync per write). Great for sessions,
        // presence, caches, test rigs. Default is TRUE (fsync every batch).
        let sync_writes = std::env::var("DBSTRIKE_SYNC")
            .map(|v| v != "0")
            .unwrap_or(true);

        let core = Arc::new(FlushCore {
            shards,
            wal: Mutex::new(wal),
            subscribers: RwLock::new(Vec::new()),
            write_queue: Mutex::new(Vec::new()),
            queue_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });

        // Start the background group-commit flusher. It wakes on new writes or
        // shutdown, drains the whole queue in one WAL append + single fsync.
        // It gets an `Arc<FlushCore>` — deliberately NOT an `Arc<Engine>`, so
        // the engine's refcount is unaffected and `Drop` stays reachable.
        let handle = Self::spawn_flusher(Arc::clone(&core), wal_path.clone());

        Ok(Arc::new(Self {
            core,
            wal_path,
            clock: AtomicU64::new(max_ts),
            flusher: Mutex::new(Some(handle)),
            sync_writes,
        }))
    }

    /// Non-durable, throwaway engine for in-process graph builds (the
    /// parallel-segment path mutates only the in-memory HNSW, so no WAL
    /// durability is needed). Uses a unique temp WAL with `DBSTRIKE_SYNC=0`
    /// semantics. Cheap: a single empty file, no fsyncs.
    pub fn open_for_build() -> Arc<Self> {
        let dir = std::env::temp_dir().join(format!("dbstrike_build_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("build.wal");
        let _ = std::fs::remove_file(&p);
        match Engine::open(&p) {
            Ok(e) => {
                // Unlink the WAL and its directory NOW, while the engine holds
                // the fd. The mapping stays valid — the inode lives until the
                // last descriptor closes — so the build engine keeps working,
                // it just leaves nothing behind however the process dies.
                //
                // `Drop for Engine` now does run (the flusher holds an
                // `Arc<FlushCore>`, not an `Arc<Engine>`), but eager unlink is
                // still the right call here: Drop does not run on abort,
                // SIGKILL, or OOM-kill, which is exactly how these leaked.
                //
                // Safe for build engines specifically: they are throwaway and
                // nothing calls `checkpoint()` on them, so `wal_path` is never
                // re-derived into `<wal>.snap`. Do NOT copy this to
                // `Engine::open` — a real engine must keep its WAL on disk.
                let _ = std::fs::remove_file(&p);
                let _ = std::fs::remove_dir(&dir);
                e
            }
            Err(_) => {
                let _ = std::fs::remove_dir_all(&dir);
                Engine::open(std::env::temp_dir().join("dbstrike_fallback_build.wal")).unwrap()
            }
        }
    }

    /// Monotonic timestamp source (also serves as the logical commit clock).
    pub fn now(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Take a read snapshot at the current logical time.
    pub fn snapshot(&self) -> u64 {
        self.clock.load(Ordering::SeqCst)
    }

    /// Total live-key count across every shard (excludes tombstoned entries).
    /// Used by the RESP `DBSIZE` command. O(K) — visits every key once — so
    /// don't call it in a tight loop on multi-million-key stores; that's the
    /// same trade-off Redis makes (SCAN preferred over DBSIZE at scale).
    pub fn dbsize(&self) -> usize {
        let snap = self.snapshot();
        let mut n = 0usize;
        for shard in &self.core.shards {
            let data = shard.read().unwrap();
            for (_, chain) in data.iter() {
                if chain.visible(snap).is_some() {
                    n += 1;
                }
            }
        }
        n
    }

    /// Register a commit subscriber (reactive sync / CDC).
    pub fn subscribe(&self, cb: Subscriber) {
        self.core.subscribers.write().unwrap().push(cb);
    }

    /// Point read as of `snapshot`: newest non-future version, honoring
    /// tombstones. O(1) via the latest-value cache in the common case.
    pub fn get_at(&self, key: &[u8], snapshot: u64) -> Option<Value> {
        let data = self.core.shards[shard_of(key)].read().unwrap();
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
        for shard in &self.core.shards {
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

    /// Range scan restricted to a SINGLE shard determined by `hint_key`.
    /// Caller guarantees that every key within [start, end) uses the same
    /// hash tag as `hint_key` (i.e. `shard_of(hint_key) == shard_of(key)` for
    /// every matching key). Skips 31 rwlock acquires + 31 useless BTreeMap
    /// ranges — the single biggest win for time-series and other
    /// tag-partitioned workloads.
    pub fn scan_pinned(
        &self,
        hint_key: &[u8],
        start: &[u8],
        end: &[u8],
        snapshot: u64,
    ) -> Vec<(Key, Value)> {
        let s = shard_of(hint_key);
        let data = self.core.shards[s].read().unwrap();
        let mut out: Vec<(Key, Value)> = Vec::new();
        for (k, chain) in data.range(start.to_vec()..end.to_vec()) {
            if let Some(v) = chain.visible(snapshot) {
                out.push((k.clone(), v.clone()));
            }
        }
        out
    }

    /// Reverse scan restricted to a single shard, early-exiting after `limit`
    /// live values. Used by TSRANGE.LATEST — dashboards want "last N points",
    /// not "the whole history sorted then tailed", and this delivers that
    /// in `O(log n + limit)` from just the pinned shard's BTreeMap.
    pub fn scan_pinned_reverse(
        &self,
        hint_key: &[u8],
        start: &[u8],
        end: &[u8],
        snapshot: u64,
        limit: usize,
    ) -> Vec<(Key, Value)> {
        let s = shard_of(hint_key);
        let data = self.core.shards[s].read().unwrap();
        let mut out: Vec<(Key, Value)> = Vec::with_capacity(limit);
        for (k, chain) in data.range(start.to_vec()..end.to_vec()).rev() {
            if let Some(v) = chain.visible(snapshot) {
                out.push((k.clone(), v.clone()));
                if out.len() >= limit {
                    break;
                }
            }
        }
        // return in ascending order to match `range()` convention
        out.reverse();
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
            for shard in &self.core.shards {
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

        // ── Non-durable fast path (DBSTRIKE_SYNC=0) ──
        // Apply directly to the sharded maps, notify subscribers, done.
        // No WAL append, no flusher round-trip. This is exactly Redis's
        // default (no AOF fsync per write) — trade off crash-durability for
        // Redis-class SET throughput. The write is still visible to every
        // reader on this process; only durability is dropped.
        if !self.sync_writes {
            let mut per_shard: Vec<Vec<(Key, u64, Value)>> =
                (0..SHARD_COUNT).map(|_| Vec::new()).collect();
            // Collect mutations for subscriber broadcast (reference form).
            let mut broadcast: Vec<Mutation> = Vec::with_capacity(writes.len());
            for (key, value) in writes {
                let s = shard_of(&key);
                broadcast.push(Mutation { key: key.clone(), value: value.clone(), ts });
                per_shard[s].push((key, ts, value));
            }
            for (i, items) in per_shard.into_iter().enumerate() {
                if items.is_empty() {
                    continue;
                }
                let mut data = self.core.shards[i].write().unwrap();
                for (k, t, v) in items {
                    data.entry(k).or_default().push(t, v);
                }
            }
            let subs = self.core.subscribers.read().unwrap();
            if !subs.is_empty() {
                for m in &broadcast {
                    for s in subs.iter() {
                        s(m);
                    }
                }
            }
            return Ok(ts);
        }

        // ── Durable path (default): route through the group-commit flusher ──
        let muts: Vec<Mutation> = writes
            .into_iter()
            .map(|(key, value)| Mutation { key, value, ts })
            .collect();

        let pending = Arc::new(PendingWrite {
            muts,
            flush_all: None,
            state: Mutex::new(WriteState { done: false, err: None }),
            cond: Condvar::new(),
        });
        {
            let mut q = self.core.write_queue.lock().unwrap();
            q.push(Arc::clone(&pending));
        }
        self.core.queue_cv.notify_all();

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
fn spawn_flusher(core: Arc<FlushCore>, wal_path: PathBuf) -> JoinHandle<()> {
    thread::spawn(move || loop {
        // Collect the current batch of pending writes.
        let batch: Vec<Arc<PendingWrite>> = {
            let mut q = core.write_queue.lock().unwrap();
            // Wait until there is work, or it's time to shut down.
            while q.is_empty() && !core.shutdown.load(Ordering::SeqCst) {
                let _g = core
                    .queue_cv
                    .wait(q)
                    .expect("group-commit flusher condvar poisoned");
                q = _g;
            }
            if core.shutdown.load(Ordering::SeqCst) && q.is_empty() {
                return;
            }
            q.drain(..).collect()
        };

        // Durable step: one append + one fsync for the whole group.
        // Each PendingWrite is ONE commit and is written as ONE atomic frame
        // (single or batch-tagged) — so a crash mid-append can never replay a
        // partial batch. The single `sync()` below makes the whole group durable.
        let flush_result: io::Result<()> = (|| {
            let mut wal = core.wal.lock().unwrap();
            for pw in &batch {
                if pw.flush_all.is_some() {
                    continue; // the flush op carries no mutations
                }
                let frame = if pw.muts.len() == 1 {
                    encode_single_record(&pw.muts[0])
                } else {
                    encode_batch_record(&pw.muts)
                };
                wal.append_unsynced(&frame)?;
            }
            // single fsync amortised across the whole group
            wal.sync()
        })();

        // Full-flush op: runs AFTER every earlier commit in this drained
        // group is already durable. Backs up + removes WAL/snapshot, reopens
        // a fresh WAL and clears all shard maps (see `perform_flush_all`).
        // When a wipe ran, the shard-apply phase below must NOT run — those
        // same-batch commits were logically "before the FLUSHALL" and their
        // keys belong to the backed-up world.
        let mut flush_all_err: Option<String> = None;
        let wipe_ran = batch.iter().any(|pw| pw.flush_all.is_some());
        if wipe_ran {
            if let Some(bak) = batch.iter().find_map(|pw| pw.flush_all.clone()) {
                match Self::perform_flush_all(&core, &wal_path, &bak) {
                    Ok(()) => eprintln!(
                        "[FLUSH] full wipe OK · WAL+snap backed up at {}",
                        bak
                    ),
                    Err(e) => flush_all_err = Some(e.to_string()),
                }
            }
        }

        // Visibility + broadcast happen ONLY if the durable flush succeeded.
        // Otherwise the mutations were never made durable, and subscribers must
        // not observe a "commit" that never happened. After a full wipe the
        // apply is skipped too: those same-batch keys belong to the pre-flush
        // world and must not resurrect into the wiped shard maps.
        //
        // Sharded apply: group mutations by shard and take each shard's write
        // lock only when we have work for it. Writers touching disjoint shards
        // don't serialize on a global data lock any more.
        if flush_result.is_ok() && !wipe_ran {
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
                let mut data = core.shards[i].write().unwrap();
                for (k, ts, v) in items {
                    data.entry(k).or_default().push(ts, v);
                }
            }
            let subs = core.subscribers.read().unwrap();
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
            } else if let Some(e) = &flush_all_err {
                st.err = Some(e.clone());
            }
            pw.cond.notify_all();
        }
    })
}

/// Execute one full-flush op on behalf of the flusher thread. Caller ordering
/// guarantees: every mutation committed before this op is already fsynced to
/// the live WAL and applied to the shard maps.
///
/// Steps, crash-safe at every boundary:
///   1. fsync the live WAL so the backup is a complete durable world.
///   2. Rename WAL → `<bak>` (atomic; same filesystem). The open fd stays
///      valid but we replace the `Wal` object right after.
///   3. Rename `<wal>.snap` → `<bak>.snap` if present.
///   4. Open a fresh WAL at the original path. On failure, roll the backups
///      back so the engine keeps its pre-flush world.
///   5. Clear every shard map — frees all key/chain memory for real.
fn perform_flush_all(core: &FlushCore, wal_path: &Path, bak: &str) -> io::Result<()> {
    let mut wal = core.wal.lock().unwrap();
    wal.sync()?;
    let snap = snap_path_for(wal_path);
    let snap_bak = format!("{}.snap", bak);
    std::fs::rename(wal_path, bak)?;
    if snap.exists() {
        if let Err(e) = std::fs::rename(&snap, &snap_bak) {
            // Restore the live WAL so we don't strand the engine file-less.
            let _ = std::fs::rename(bak, wal_path);
            return Err(e);
        }
    }
    match Wal::open(wal_path) {
        Ok(fresh) => {
            *wal = fresh;
        }
        Err(e) => {
            let _ = std::fs::rename(bak, wal_path);
            if snap.exists() {
                let _ = std::fs::rename(&snap_bak, &snap);
            }
            return Err(e);
        }
    }
    drop(wal);
    for sh in &core.shards {
        sh.write().unwrap().clear();
    }
    Ok(())
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

    /// **Checkpoint.** Snapshot the current visible state to `<wal>.snap`,
    /// then truncate the WAL. Bounds unbounded WAL growth — the file that
    /// used to be O(commits-since-forever) becomes O(commits-since-last-ckpt).
    ///
    /// Guarantees (crash-safe by construction):
    ///   1. Take the WAL mutex so no new commits can flush during the snapshot
    ///      capture (in-flight writers waiting on `write_queue` still block).
    ///   2. For each shard: collect (key, latest_ts, latest_value) — one
    ///      Mutation per live key, tombstones included so DELETEs replicate.
    ///   3. Write to `<snap>.tmp`, `fsync`, atomic-rename → `<snap>`.
    ///      Rename is atomic on POSIX; if we crash before this step the old
    ///      snapshot is still valid and WAL isn't truncated.
    ///   4. Only NOW truncate the WAL. Next open sees the fresh snapshot +
    ///      an empty (or nearly-empty) WAL.
    ///
    /// Returns (records_snapshotted, snap_file_bytes).
    pub fn checkpoint(&self) -> io::Result<(u64, u64)> {
        // Hold the WAL mutex for the whole checkpoint. New commits queue on
        // `write_queue` and are drained by the flusher — but the flusher can't
        // touch the WAL file while we hold this lock, so no writes race the
        // snapshot boundary.
        let mut wal_g = self.core.wal.lock().unwrap();

        // Collect one mutation per live key from the CURRENT visible state.
        let snapshot_ts = self.clock.load(Ordering::SeqCst);
        let mut muts: Vec<Mutation> = Vec::new();
        for shard in &self.core.shards {
            let data = shard.read().unwrap();
            for (key, chain) in data.iter() {
                // Take the latest (post-prune) view; skip pure tombstones with
                // no live successor — they were never durable state anyway.
                if let Some(last) = chain.versions.last() {
                    muts.push(Mutation {
                        key: key.clone(),
                        value: last.value.clone(),
                        ts: last.ts,
                    });
                }
            }
        }
        let n = muts.len() as u64;

        // Durable snapshot write (tmp + fsync + atomic rename).
        let snap_path = snap_path_for(&self.wal_path);
        write_snapshot(&snap_path, &muts)?;

        // WAL truncate — safe now that the snapshot is durable on disk.
        wal_g.truncate()?;

        // Nudge the clock so the next commit's ts > any snapshotted ts.
        // (Snapshot uses per-record ts, but the trailing marker also matters
        // if we ever want to skip WAL records with ts <= snapshot_ts.)
        self.clock.fetch_max(snapshot_ts, Ordering::SeqCst);

        let bytes = std::fs::metadata(&snap_path).map(|m| m.len()).unwrap_or(0);
        Ok((n, bytes))
    }

    /// Current WAL size in bytes — callers use this to decide whether a
    /// checkpoint (snapshot + truncate) is worth its cost.
    pub fn wal_bytes(&self) -> u64 {
        self.core.wal.lock().unwrap().len()
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

    /// **Full flush (`FLUSHALL`).** Backs up the live WAL — and the checkpoint
    /// snapshot if one exists — with an instant zero-copy rename to
    /// `<wal>.bak-<millis>` (snapshot twin: `<...>.snap`), deletes them from
    /// the active path, reopens a fresh empty WAL and wipes every shard map.
    ///
    /// Why rename instead of tombstone-per-key: a durability engine keeps its
    /// promises by making the destructive step atomic at the filesystem level.
    /// A 7 GB WAL becomes its own backup in one syscall; a crash mid-flush
    /// leaves either the old world or the new one on disk, never a mixture.
    /// Recovery is untouched: `open` still loads `<wal>.snap` + replays the
    /// WAL, both of which simply no longer exist post-flush.
    ///
    /// Serialized through the group-commit flusher: commits drained before
    /// this op are durable AND applied before the wipe; anything enqueued
    /// after lands on the fresh WAL. Returns the WAL backup path so callers
    /// can log/report where the pre-flush world lives.
    pub fn flushall_with_backup(&self) -> io::Result<String> {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let backup = format!("{}.bak-{}", self.wal_path.display(), millis);
        let pending = Arc::new(PendingWrite {
            muts: Vec::new(),
            flush_all: Some(backup.clone()),
            state: Mutex::new(WriteState { done: false, err: None }),
            cond: Condvar::new(),
        });
        {
            let mut q = self.core.write_queue.lock().unwrap();
            q.push(Arc::clone(&pending));
        }
        self.core.queue_cv.notify_all();
        let mut state = pending.state.lock().unwrap();
        while !state.done {
            state = pending
                .cond
                .wait(state)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "flush wait poisoned"))?;
        }
        match &state.err {
            Some(e) => Err(io::Error::new(io::ErrorKind::Other, e.clone())),
            None => Ok(backup),
        }
    }
}

impl Drop for Engine {
    /// Graceful shutdown: stop the flusher and join it.
    ///
    /// This is reachable only because the flusher thread holds an
    /// `Arc<FlushCore>` rather than an `Arc<Engine>` — see `FlushCore` for why
    /// the previous arrangement made this function dead code.
    ///
    /// Ordering matters. `shutdown` is set BEFORE `notify_all`, and the flusher
    /// re-checks it while holding the `write_queue` lock, so there is no lost
    /// wakeup: either the flusher is already inside `wait` and the notify
    /// reaches it, or it has not yet re-acquired the lock and will observe
    /// `shutdown` on its next check.
    ///
    /// The flusher only returns once the queue is empty, so any writes still
    /// queued at drop time are flushed and fsynced before the join completes.
    fn drop(&mut self) {
        self.core.shutdown.store(true, Ordering::SeqCst);
        self.core.queue_cv.notify_all();
        // `lock()` can only be poisoned by a panic while holding this mutex;
        // nothing but `Drop` and the constructor touch it, so recover rather
        // than double-panic during unwind.
        let handle = match self.flusher.lock() {
            Ok(mut g) => g.take(),
            Err(p) => p.into_inner().take(),
        };
        if let Some(h) = handle {
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
            let data = self.engine.core.shards[shard_of(k)].read().unwrap();
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

    /// Regression: dropping an `Engine` must actually run `Drop` and reap the
    /// flusher thread.
    ///
    /// Before the `FlushCore` split, `spawn_flusher` was handed a strong
    /// `Arc<Engine>`. The refcount could never reach zero while the thread
    /// lived, and the thread only exited once `Drop` set `shutdown` — so
    /// `Drop` never ran and every engine leaked its flusher thread.
    ///
    /// The probe is a canary `Arc` captured by a subscriber closure.
    /// Subscribers live inside `FlushCore`, and `FlushCore` is only freed once
    /// BOTH the engine and the flusher thread have released their handles. So
    /// `strong_count == 1` after the drop proves two things at once: `Drop`
    /// ran, and the thread it joined had genuinely exited.
    ///
    /// Written to fail rather than hang under the old code: the join deadlock
    /// would never be reached, because `drop(e)` was simply a no-op there and
    /// the assertion below would see `strong_count == 2`.
    #[test]
    fn drop_reaps_flusher_thread() {
        let canary = Arc::new(());
        {
            let e = Engine::open(tmp("drop_reaps.wal")).unwrap();
            let held = Arc::clone(&canary);
            e.subscribe(Arc::new(move |_m: &Mutation| {
                // Keep `held` alive for as long as the subscriber list is.
                let _ = &held;
            }));
            // Exercise the durable path so the flusher is definitely parked in
            // `wait` on the condvar when we drop, not still starting up.
            e.put(b"k".to_vec(), Value::Int(1)).unwrap();
            assert_eq!(Arc::strong_count(&canary), 2, "subscriber should hold the canary");

            let e = Arc::try_unwrap(e).unwrap_or_else(|_| panic!("engine Arc unexpectedly shared"));
            drop(e);
        }
        assert_eq!(
            Arc::strong_count(&canary),
            1,
            "FlushCore outlived the Engine — Drop did not run or the flusher thread leaked"
        );
    }

    /// A queued write must still be flushed and fsynced before `Drop` returns.
    /// The flusher's shutdown check is `shutdown && queue.is_empty()`, so it
    /// drains before exiting; this pins that ordering down.
    #[test]
    fn drop_flushes_pending_writes() {
        let path = tmp("drop_flushes.wal");
        {
            let e = Engine::open(&path).unwrap();
            for i in 0..64u64 {
                e.put(format!("k{i}").into_bytes(), Value::Int(i as i64)).unwrap();
            }
        }
        // Reopen from the WAL alone and confirm every write survived.
        let e2 = Engine::open(&path).unwrap();
        for i in 0..64u64 {
            assert_eq!(
                e2.get(format!("k{i}").as_bytes()),
                Some(Value::Int(i as i64)),
                "write {i} lost across drop + reopen"
            );
        }
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
    fn checkpoint_truncates_wal_and_restores_state() {
        let path = tmp("ckpt.wal");
        // Phase 1: write, checkpoint, write more, drop engine.
        let (wal_before, snap_after) = {
            let e = Engine::open(&path).unwrap();
            for i in 0..500u64 {
                e.put(format!("k{i}").into_bytes(), Value::Int(i as i64)).unwrap();
            }
            let wal_size = std::fs::metadata(&path).unwrap().len();
            let (n, snap_bytes) = e.checkpoint().unwrap();
            assert_eq!(n, 500, "checkpoint captured every key");
            assert!(snap_bytes > 0, "snapshot file not empty");
            // Post-checkpoint the WAL is truncated to zero, then we write more.
            let wal_after_ckpt = std::fs::metadata(&path).unwrap().len();
            assert_eq!(wal_after_ckpt, 0, "WAL truncated after checkpoint");
            for i in 500..600u64 {
                e.put(format!("k{i}").into_bytes(), Value::Int(i as i64)).unwrap();
            }
            (wal_size, snap_bytes)
        };
        // Sanity: WAL grew back to hold just the 100 post-checkpoint records.
        let wal_now = std::fs::metadata(&path).unwrap().len();
        assert!(
            wal_now < wal_before / 3,
            "post-checkpoint WAL should be much smaller (was {wal_now}, before {wal_before})"
        );
        assert!(snap_after > 0);

        // Phase 2: reopen, verify both snapshot keys AND post-ckpt WAL keys.
        let e = Engine::open(&path).unwrap();
        // A key from the snapshot era.
        assert_eq!(e.get(b"k7"), Some(Value::Int(7)), "snapshot key survived");
        // A key from AFTER the checkpoint (WAL replay on top of snapshot).
        assert_eq!(e.get(b"k550"), Some(Value::Int(550)), "post-ckpt key survived");
        // A key that shouldn't exist.
        assert_eq!(e.get(b"k999"), None);
    }

    #[test]
    fn version_pruning_bounds_memory() {
        let e = Engine::open(tmp("prune.wal")).unwrap();
        // Overwrite the same key many times; chain should never exceed MAX.
        for i in 0..50i64 {
            e.put(b"hot".to_vec(), Value::Int(i)).unwrap();
        }
        // Latest read still correct.
        assert_eq!(e.get(b"hot"), Some(Value::Int(49)));
        // Peek the chain length via the internal shard read.
        let shard_ix = shard_of(b"hot");
        let d = e.core.shards[shard_ix].read().unwrap();
        let chain = d.get(b"hot".as_slice()).unwrap();
        assert!(
            chain.versions.len() <= MAX_VERSIONS_PER_KEY,
            "chain should be pruned; got {} versions", chain.versions.len()
        );
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

    /// FLUSHALL must (1) wipe live state immediately, (2) leave a complete
    /// restorable backup of the pre-flush world (WAL + checkpoint snapshot
    /// twin), (3) NOT resurrect wiped keys on reopen, while (4) post-flush
    /// writes survive restarts normally. Restoring both backup files and
    /// reopening brings every pre-flush key back — proving the rename-based
    /// backup is a real durable world, not a truncated stub.
    #[test]
    fn flushall_backs_up_wal_and_wipes_state() {
        let p = tmp("flushall.wal");
        let bak: String;
        {
            let e = Engine::open(&p).unwrap();
            e.put(b"k1".to_vec(), Value::Int(1)).unwrap();
            e.put(b"k2".to_vec(), Value::Int(2)).unwrap();
            // Checkpoint so k1/k2 live in <wal>.snap; the flush must back up
            // that snapshot too, or a later open would resurrect them.
            e.checkpoint().unwrap();
            e.put(b"k3".to_vec(), Value::Int(3)).unwrap();
            assert_eq!(e.dbsize(), 3);

            bak = e.flushall_with_backup().unwrap();

            // Live state is empty right now.
            assert_eq!(e.dbsize(), 0);
            assert!(e.get(b"k1").is_none());
            // Backups exist.
            assert!(std::path::Path::new(&bak).exists(), "WAL backup missing");
            assert!(
                std::path::Path::new(&format!("{}.snap", bak)).exists(),
                "snapshot backup twin missing"
            );
            // Live WAL is freshly reopened (exists, zero bytes).
            assert_eq!(std::fs::metadata(&p).unwrap().len(), 0);

            // Post-flush writes land in the fresh world.
            e.put(b"fresh".to_vec(), Value::Int(9)).unwrap();
        }
        {
            // Reopen: no resurrection of pre-flush keys, fresh key survives.
            let e = Engine::open(&p).unwrap();
            assert_eq!(e.dbsize(), 1, "reopen must NOT resurrect pre-flush keys");
            assert!(e.get(b"fresh").is_some());
            assert!(e.get(b"k3").is_none());
        }
        {
            // The backup is restorable: put both files back and open.
            std::fs::remove_file(&p).unwrap();
            std::fs::rename(&bak, &p).unwrap();
            let live_snap = snap_path_for(&p);
            let _ = std::fs::remove_file(&live_snap);
            std::fs::rename(format!("{}.snap", bak), &live_snap).unwrap();

            let e = Engine::open(&p).unwrap();
            assert_eq!(e.dbsize(), 3, "restored backup must replay all pre-flush keys");
            assert_eq!(e.get(b"k1"), Some(Value::Int(1)));
            assert_eq!(e.get(b"k3"), Some(Value::Int(3)));
            assert!(e.get(b"fresh").is_none());
        }
    }

    /// A FLUSHALL racing ordinary commits through the group-commit queue must
    /// serialize cleanly: commits drained before the wipe are visible before
    /// it runs, commits enqueued after land on the fresh WAL and stay.
    #[test]
    fn flushall_serializes_with_concurrent_commits() {
        let p = tmp("flushall_race.wal");
        let e = Engine::open(&p).unwrap();
        e.put(b"pre".to_vec(), Value::Int(1)).unwrap();

        let e2 = Arc::clone(&e);
        let writer = std::thread::spawn(move || {
            for i in 0..200i64 {
                e2.put(format!("post:{i}").into_bytes(), Value::Int(i)).unwrap();
            }
        });
        // Interleave the wipe with the writer thread's commits.
        std::thread::sleep(std::time::Duration::from_millis(1));
        e.flushall_with_backup().unwrap();
        writer.join().unwrap();

        // Whatever survived must be exactly the fresh-world set: `post:*`
        // keys committed after the wipe. `pre` may or may not have been wiped
        // depending on interleaving, but NO key may come back once gone —
        // checked implicitly by reopening: the WAL must replay consistently.
        let count = e.scan_prefix(b"post:", e.snapshot()).len();
        assert!(
            count > 0 && count <= 200,
            "post-wipe commits should partially or fully survive, got {count}"
        );
        drop(e);
        let e = Engine::open(&p).unwrap();
        let reopened = e.scan_prefix(b"post:", e.snapshot()).len();
        assert_eq!(reopened, count, "WAL replay must match in-memory survivor set");
    }
}
