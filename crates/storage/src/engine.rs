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
    /// ts of the newest version (== `versions.last().ts`), cached because it
    /// is read on every OCC validation.
    latest_ts: u64,
    /// Versions sorted by ts; the last one is the current value. (A separate
    /// `latest: Option<Value>` cache used to hold a SECOND copy of every
    /// current value — the single biggest per-key memory cost.)
    versions: Vec<Version>,
    /// ts of the newest version ever pruned (0 = none). A read at a snapshot
    /// below the oldest retained version but at/after this cannot be answered
    /// and must say so instead of reporting "key absent".
    pruned_ts: u64,
    /// ts of the first version this process knows for the key (0 = unknown,
    /// e.g. restored from a checkpoint). A snapshot before it predates the
    /// key's existence.
    first_ts: u64,
}

impl Chain {
    /// Current value (`None` if the newest version is a tombstone).
    fn latest(&self) -> Option<&Value> {
        self.versions.last().map(|v| &v.value).filter(|v| !matches!(v, Value::Tombstone))
    }

    /// Insert a committed version, keeping `versions` sorted by ts (records
    /// can arrive out of order on WAL replay). Prunes the oldest version past
    /// `MAX_VERSIONS_PER_KEY`, except one an in-progress checkpoint at `pin`
    /// still needs (the chain may grow briefly instead).
    fn push(&mut self, ts: u64, value: Value, pin: u64) {
        if self.first_ts == 0 && self.pruned_ts == 0 && self.versions.is_empty() {
            self.first_ts = ts;
        }
        if self.versions.capacity() == 0 {
            // Most keys only ever hold one version: allocate exactly one slot
            // instead of Vec's default first growth to four.
            self.versions.reserve_exact(1);
        }
        let pos = self.versions.partition_point(|v| v.ts <= ts);
        self.versions.insert(pos, Version { ts, value });
        self.latest_ts = self.latest_ts.max(ts);
        while self.versions.len() > MAX_VERSIONS_PER_KEY {
            let needed_by_pin = pin != 0
                && self.versions[0].ts <= pin
                && self.versions[1].ts > pin;
            if needed_by_pin {
                break;
            }
            let gone = self.versions.remove(0);
            self.pruned_ts = self.pruned_ts.max(gone.ts);
        }
    }

    /// Newest version visible at `snapshot`, honoring tombstones. O(1) when the
    /// newest version is already <= snapshot (the common case).
    fn visible(&self, snapshot: u64) -> Option<&Value> {
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
    /// Wire/disk encoding (replication stream, snapshot records).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode()
    }

    pub fn from_bytes(b: &[u8]) -> Option<Mutation> {
        Mutation::decode(b)
    }

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
    /// Commit timestamp, reserved while holding the queue lock so queue
    /// order == timestamp order (see `visible_ts`).
    ts: u64,
    muts: Vec<Mutation>,
    /// Optimistic-concurrency read set + the snapshot it was read at. The
    /// flusher validates it at the single serialization point, so
    /// validate-then-commit is atomic. Empty for blind writes.
    reads: Vec<(Key, u64)>,
    #[allow(dead_code)]
    snapshot: u64,
    /// Server-side `INCRBY key by`, resolved by the flusher against the latest
    /// state in queue order: N concurrent increments (even of one key) commit
    /// in ONE fsync with no lock and no conflict retries.
    incr: Option<(Key, i64)>,
    /// An operation executed BY the commit thread against the latest state
    /// (see `Engine::submit_op`). Taken (run once) during resolve.
    op: Mutex<Option<OpFn>>,
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
    /// The read set was invalidated by a newer commit.
    conflict: bool,
    /// The op itself failed (e.g. INCR of a non-integer) — nothing written.
    op_err: Option<String>,
    /// Result of an `incr` op.
    value: Option<i64>,
}

impl WriteState {
    fn new() -> Mutex<Self> {
        Mutex::new(WriteState { done: false, err: None, conflict: false, op_err: None, value: None })
    }
}

/// Outcome of resolving one pending write against current state.
enum Resolved {
    /// Commit these mutations; `None` = the commit's own queued `muts`
    /// (borrowed, not cloned — a bulk-load batch can be hundreds of MB).
    Apply(Option<Vec<Mutation>>, Option<i64>),
    Conflict,
    OpErr(String),
}

/// Parse an integer with Redis `string2ll` strictness: optional leading
/// '-', digits only, no leading zeros (except "0"), no '+', no whitespace.
pub fn parse_strict_i64(b: &[u8]) -> Option<i64> {
    let digits = b.strip_prefix(b"-").unwrap_or(b);
    if digits.is_empty() || digits.len() > 20 || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if digits[0] == b'0' && (digits.len() > 1 || b.len() != digits.len()) {
        return None; // "01", "-0"
    }
    std::str::from_utf8(b).ok()?.parse::<i64>().ok()
}

/// Interpret a stored value as an i64 counter (Redis INCR semantics).
fn as_counter(v: Option<&Value>) -> Result<i64, String> {
    match v {
        None | Some(Value::Tombstone) => Ok(0),
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Bytes(b)) => parse_strict_i64(b).ok_or_else(|| "value is not an integer or out of range".to_string()),
        Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value".to_string()),
    }
}

/// Resolve `pw` against the shard maps plus `overlay` (keys written earlier
/// in the same flush batch, which are newer than any snapshot still pending).
fn resolve(core: &FlushCore, pw: &PendingWrite, overlay: &BTreeMap<Key, Value>) -> Resolved {
    if let Some(op) = pw.op.lock().unwrap().take() {
        let mut st = OpStore { core, overlay, writes: BTreeMap::new() };
        op(&mut st);
        let muts = st.writes.into_iter().map(|(key, value)| Mutation { key, value, ts: pw.ts }).collect();
        return Resolved::Apply(Some(muts), None);
    }
    for (k, snap) in &pw.reads {
        if overlay.contains_key(k) {
            return Resolved::Conflict;
        }
        let data = core.shards[shard_of(k)].read().unwrap();
        if data.get(k).is_some_and(|c| c.latest_ts > *snap) {
            return Resolved::Conflict;
        }
    }
    if let Some((key, by)) = &pw.incr {
        let cur = match overlay.get(key) {
            Some(v) => as_counter(Some(v)),
            None => {
                let data = core.shards[shard_of(key)].read().unwrap();
                as_counter(data.get(key).and_then(|c| c.latest()))
            }
        };
        let next = match cur.and_then(|c| {
            c.checked_add(*by).ok_or_else(|| "increment or decrement would overflow".to_string())
        }) {
            Ok(n) => n,
            Err(e) => return Resolved::OpErr(e),
        };
        return Resolved::Apply(Some(vec![Mutation { key: key.clone(), value: Value::Int(next), ts: pw.ts }]), Some(next));
    }
    Resolved::Apply(None, None)
}

impl Resolved {
    fn muts<'a>(&'a self, pw: &'a PendingWrite) -> Option<&'a [Mutation]> {
        match self {
            Resolved::Apply(Some(m), _) => Some(m),
            Resolved::Apply(None, _) => Some(&pw.muts),
            _ => None,
        }
    }
}

/// Number of independent shard maps. Reads/writes on different shards run in
/// parallel — this is the "sharded RwLock" pattern that DashMap and every
/// modern KV store uses to escape the single-writer bottleneck of one big
/// RwLock. 32 balances lock-contention reduction vs cache-line waste; a good
/// sweet spot up to ~64 cores.
pub const SHARD_COUNT: usize = 32;

/// Sentinel op-error string for an OCC conflict (never a user-visible message).
const CONFLICT: &str = "\0conflict";

/// Apply committed mutations, taking each shard's write lock once.
fn apply_to_shards(core: &FlushCore, muts: &[Mutation]) {
    apply_refs_to_shards(core, &muts.iter().collect::<Vec<_>>());
}

fn apply_refs_to_shards(core: &FlushCore, muts: &[&Mutation]) {
    let mut per_shard: Vec<Vec<&Mutation>> = (0..SHARD_COUNT).map(|_| Vec::new()).collect();
    for m in muts {
        per_shard[shard_of(&m.key)].push(m);
    }
    for (i, items) in per_shard.into_iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        let mut data = core.shards[i].write().unwrap();
        // Keep whatever version the OLDEST open snapshot (a checkpoint, a
        // replica sync, a transaction) still needs.
        let pin = core.active.lock().unwrap().keys().next().copied().unwrap_or(0);
        let mut delta = 0i64;
        let mut dead: Vec<(Key, u64)> = Vec::new();
        for m in items {
            let chain = data.entry(m.key.clone()).or_default();
            let before = chain.latest().is_some();
            chain.push(m.ts, m.value.clone(), pin);
            if m.key.starts_with(b"kv:") {
                delta += chain.latest().is_some() as i64 - before as i64;
            }
            if chain.latest().is_none() {
                dead.push((m.key.clone(), chain.latest_ts));
            }
        }
        drop(data);
        if !dead.is_empty() {
            core.graveyard.lock().unwrap().extend(dead);
        }
        if delta != 0 {
            core.kv_live.fetch_add(delta, Ordering::Relaxed);
        }
    }
}

/// WAL segment rotated out by an in-progress checkpoint (`foo.wal.ckpt`).
/// Recovery replays it between the snapshot and the live WAL.
fn pending_path_for(wal: &Path) -> PathBuf {
    let mut s = wal.as_os_str().to_owned();
    s.push(".ckpt");
    PathBuf::from(s)
}

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
    /// Highest timestamp T such that EVERY commit with ts <= T has been
    /// applied. `snapshot()` returns this — not the raw clock — so a snapshot
    /// never later gains an older version (the clock runs ahead of writes still
    /// queued for the flusher, which made "repeatable" reads change).
    visible_ts: AtomicU64,
    /// Serializes ts-allocation + apply on the non-durable path, giving it the
    /// same ordering guarantees the single flusher thread gives durable mode.
    apply_lock: Mutex<()>,
    /// Live `kv:` key count (see `Engine::kv_count`).
    kv_live: std::sync::atomic::AtomicI64,
    /// Snapshots held by open transactions (snapshot → count). Tombstone GC
    /// must not remove a chain an open snapshot could still read through.
    active: Mutex<BTreeMap<u64, usize>>,
    /// Keys whose newest version is a tombstone, with that tombstone's ts,
    /// awaiting physical removal.
    graveyard: Mutex<Vec<(Key, u64)>>,
    /// Newest tombstone ts whose chain was physically removed: a point read
    /// of a missing key at a snapshot below this cannot be answered exactly.
    gc_horizon: AtomicU64,
    /// Callbacks run right after a FLUSHALL wipe, on the commit thread, in
    /// order with the mutation stream (replication needs the exact point).
    flush_subscribers: RwLock<Vec<Arc<dyn Fn() + Send + Sync>>>,
    /// Highest commit ts written to the WAL so far (the rotation boundary of
    /// a non-blocking checkpoint).
    last_appended: AtomicU64,
}

/// Physically remove chains whose newest version is a tombstone that every
/// open (and every future) snapshot already sees. Without this, a deleted key
/// stayed in its shard map forever: memory grew with churn, and range scans
/// (e.g. ZPOPMIN, which deletes from the front) walked ever more dead entries.
fn sweep_tombstones(core: &FlushCore) {
    let horizon = {
        let active = core.active.lock().unwrap();
        let vis = core.visible_ts.load(Ordering::SeqCst);
        active.keys().next().map_or(vis, |&m| m.min(vis))
    };
    let ready: Vec<(Key, u64)> = {
        let mut g = core.graveyard.lock().unwrap();
        if g.is_empty() {
            return;
        }
        let (ready, keep): (Vec<_>, Vec<_>) = g.drain(..).partition(|(_, ts)| *ts <= horizon);
        *g = keep;
        ready
    };
    let mut max_removed = 0;
    for (k, ts) in ready {
        let mut data = core.shards[shard_of(&k)].write().unwrap();
        let dead = data.get(&k).is_some_and(|c| c.latest().is_none() && c.latest_ts == ts);
        if dead {
            data.remove(&k);
            max_removed = max_removed.max(ts);
        }
    }
    core.gc_horizon.fetch_max(max_removed, Ordering::SeqCst);
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
    /// One checkpoint at a time.
    ckpt_lock: Mutex<()>,
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
                        let chain = shard_data[s].entry(m.key).or_default();
                        // The checkpoint compacted away this key's history:
                        // reads before this version can't be answered.
                        chain.pruned_ts = chain.pruned_ts.max(m.ts.saturating_sub(1).max(1));
                        chain.push(m.ts, m.value, 0);
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
        // A checkpoint that crashed after rotating left `<wal>.ckpt`: those
        // commits are not in any snapshot yet, so replay them first. (If the
        // crash came after the new snapshot was written, re-applying these
        // older versions is harmless: chains order versions by ts.)
        let pending = pending_path_for(&wal_path);
        if pending.exists() {
            let mut seg = Wal::open(&pending)?;
            seg.replay_with(|rec| {
                for m in decode_records(&rec) {
                    max_ts = max_ts.max(m.ts);
                    let s = shard_of(&m.key);
                    shard_data[s].entry(m.key).or_default().push(m.ts, m.value, 0);
                }
            })?;
        }
        // Streamed: frames are applied as they are read instead of first
        // materializing the entire log (a 246 MB WAL used to need ~2× that
        // in RAM just to boot).
        let mut wal = Wal::open(&wal_path)?;
        wal.replay_with(|rec| {
            for m in decode_records(&rec) {
                max_ts = max_ts.max(m.ts);
                let s = shard_of(&m.key);
                shard_data[s].entry(m.key).or_default().push(m.ts, m.value, 0);
            }
        })?;

        // Nothing can read history yet: drop chains that end in a tombstone.
        let mut gc_horizon = 0u64;
        for m in shard_data.iter_mut() {
            m.retain(|_, c| {
                let keep = c.latest().is_some();
                if !keep {
                    gc_horizon = gc_horizon.max(c.latest_ts);
                }
                keep
            });
        }
        let kv_live: i64 = shard_data
            .iter()
            .map(|m| m.range(b"kv:".to_vec()..b"kv;".to_vec()).filter(|(_, c)| c.latest().is_some()).count() as i64)
            .sum();
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
            visible_ts: AtomicU64::new(max_ts),
            apply_lock: Mutex::new(()),
            kv_live: std::sync::atomic::AtomicI64::new(kv_live),
            active: Mutex::new(BTreeMap::new()),
            graveyard: Mutex::new(Vec::new()),
            gc_horizon: AtomicU64::new(gc_horizon),
            flush_subscribers: RwLock::new(Vec::new()),
            last_appended: AtomicU64::new(max_ts),
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
            ckpt_lock: Mutex::new(()),
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
    /// Durable-state file paths `(wal, snapshot)` — used by the snapshot
    /// API to copy/restore the checkpointed world.
    pub fn paths(&self) -> (PathBuf, PathBuf) {
        (self.wal_path.clone(), snap_path_for(&self.wal_path))
    }

    pub fn now(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Take a read snapshot at the current logical time.
    pub fn snapshot(&self) -> u64 {
        self.core.visible_ts.load(Ordering::SeqCst)
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

    /// Register a callback run right after every FLUSHALL wipe, in stream
    /// order with commit subscribers.
    pub fn subscribe_flush(&self, cb: Arc<dyn Fn() + Send + Sync>) {
        self.core.flush_subscribers.write().unwrap().push(cb);
    }

    /// Visit every live key as of the current snapshot, shard by shard (each
    /// shard's read lock is held only while copying it). The snapshot is
    /// pinned for the duration, and its ts is returned BEFORE visiting via
    /// `on_start`, so a caller that subscribed to commits beforehand can
    /// forward exactly the commits newer than the dump.
    pub fn dump(&self, on_start: impl FnOnce(u64), mut f: impl FnMut(Mutation)) {
        let snap = {
            let mut a = self.core.active.lock().unwrap();
            let s = self.snapshot();
            *a.entry(s).or_insert(0) += 1;
            s
        };
        on_start(snap);
        for sh in &self.core.shards {
            let page: Vec<Mutation> = {
                let data = sh.read().unwrap();
                data.iter()
                    .filter_map(|(k, c)| {
                        let v = c.visible(snap)?;
                        let ts = c.versions.iter().rev().find(|x| x.ts <= snap).map_or(0, |x| x.ts);
                        Some(Mutation { key: k.clone(), value: v.clone(), ts })
                    })
                    .collect()
            };
            for m in page {
                f(m);
            }
        }
        let mut a = self.core.active.lock().unwrap();
        if let Some(c) = a.get_mut(&snap) {
            *c -= 1;
            if *c == 0 {
                a.remove(&snap);
            }
        }
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

    /// Like `get_at`, but `Err(())` when `snapshot` predates the retained
    /// version history of `key` (MAX_VERSIONS_PER_KEY pruning), where the old
    /// answer — `None`, "did not exist" — was simply wrong.
    pub fn get_at_checked(&self, key: &[u8], snapshot: u64) -> Result<Option<Value>, ()> {
        let data = self.core.shards[shard_of(key)].read().unwrap();
        match data.get(key) {
            // The key may have existed and been deleted + collected.
            None if snapshot < self.core.gc_horizon.load(Ordering::SeqCst) => Err(()),
            None => Ok(None),
            Some(chain) => {
                if chain.first_ts > 0 && snapshot < chain.first_ts {
                    return Ok(None); // before the key ever existed
                }
                let oldest = chain.versions.first().map_or(u64::MAX, |v| v.ts);
                if chain.pruned_ts > 0 && snapshot < oldest {
                    return Err(());
                }
                Ok(chain.visible(snapshot).cloned())
            }
        }
    }

    /// True when commits wait for an fsync (the default; `DBSTRIKE_SYNC=0`
    /// turns it off). Network layers use it to decide what can run inline.
    pub fn is_durable(&self) -> bool {
        self.sync_writes
    }

    /// Commit ts of the newest version of `key` (0 if never written).
    pub fn key_version(&self, key: &[u8]) -> u64 {
        let data = self.core.shards[shard_of(key)].read().unwrap();
        data.get(key).map_or(0, |c| c.latest_ts)
    }

    /// Number of live user KV keys (`kv:` prefix), maintained incrementally
    /// on apply — O(1). `DBSIZE` used to walk every shard under read locks and
    /// also counted internal keys (RAG generation counters, CRDT state, ...).
    pub fn kv_count(&self) -> u64 {
        self.core.kv_live.load(Ordering::Relaxed).max(0) as u64
    }

    /// Up to `limit` live pairs with `start <= key < end` in key order,
    /// touching at most `limit` entries per shard: the building block for a
    /// cursor SCAN that costs O(limit) per call instead of O(keyspace).
    pub fn scan_from(&self, start: &[u8], end: &[u8], limit: usize, snapshot: u64) -> Vec<(Key, Value)> {
        if start >= end || limit == 0 {
            return Vec::new();
        }
        let mut out: Vec<(Key, Value)> = Vec::new();
        for shard in &self.core.shards {
            let data = shard.read().unwrap();
            let mut n = 0;
            for (k, chain) in data.range(start.to_vec()..end.to_vec()) {
                if let Some(v) = chain.visible(snapshot) {
                    out.push((k.clone(), v.clone()));
                    n += 1;
                    if n >= limit {
                        break;
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out.truncate(limit);
        out
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

    /// Forward single-shard scan stopping after `limit` live values.
    pub fn scan_pinned_limit(&self, hint_key: &[u8], start: &[u8], end: &[u8], snapshot: u64, limit: usize) -> Vec<(Key, Value)> {
        if start >= end {
            return Vec::new();
        }
        let data = self.core.shards[shard_of(hint_key)].read().unwrap();
        let mut out = Vec::new();
        for (k, chain) in data.range(start.to_vec()..end.to_vec()) {
            if out.len() >= limit {
                break;
            }
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
        if start >= end {
            return Vec::new();
        }
        let s = shard_of(hint_key);
        let data = self.core.shards[s].read().unwrap();
        let mut out: Vec<(Key, Value)> = Vec::with_capacity(limit.min(1024));
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
        // Read the snapshot and register it under the same lock the GC
        // horizon is computed under, so no tombstone sweep can slip between.
        let snapshot = {
            let mut a = self.core.active.lock().unwrap();
            let s = self.snapshot();
            *a.entry(s).or_insert(0) += 1;
            s
        };
        Txn {
            engine: Arc::clone(self),
            snapshot,
            writes: BTreeMap::new(),
            reads: Vec::new(),
            watches: Vec::new(),
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
        let muts: Vec<(Key, Value)> = writes.into_iter().collect();
        let (ts, st) = self.submit(muts, Vec::new(), 0, None, None)?;
        match st {
            Ok(_) => Ok(ts),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        }
    }

    /// Atomic `INCRBY`: resolved on the commit path against the latest state,
    /// so concurrent increments never lose updates and share one fsync. Errors
    /// (non-integer value, overflow) are `Err(msg)`; nothing is written.
    pub fn incr_by(&self, key: Key, by: i64) -> io::Result<Result<i64, String>> {
        let (_, st) = self.submit(Vec::new(), Vec::new(), 0, Some((key, by)), None)?;
        Ok(st.map(|v| v.unwrap_or(0)))
    }

    /// Run `op` ON THE COMMIT PATH: in queue (= timestamp) order, against
    /// the latest state including earlier commits of the same batch, then
    /// commit its writes in the shared group fsync. Operations are therefore
    /// serializable with no conflicts or retries — N clients hammering one
    /// key cost one fsync per batch, not one retry storm (the Redis single-
    /// writer model, with group commit). `op` must be short: it runs on the
    /// single commit thread. Returns once its writes are durable and visible.
    pub fn submit_op(&self, op: OpFn) -> io::Result<()> {
        self.submit(Vec::new(), Vec::new(), 0, None, Some(op)).map(|_| ())
    }

    /// Several commit-path operations at once (a pipelined run of commands).
    /// Each is its own commit in queue order, but they are enqueued together
    /// and share the group fsync instead of each waiting a full fsync before
    /// the next is even submitted.
    pub fn submit_ops(&self, ops: Vec<OpFn>) -> io::Result<()> {
        if !self.sync_writes {
            for op in ops {
                self.submit_op(op)?;
            }
            return Ok(());
        }
        let pending: Vec<Arc<PendingWrite>> = {
            let mut q = self.core.write_queue.lock().unwrap();
            ops.into_iter()
                .map(|op| {
                    let pw = Arc::new(PendingWrite {
                        ts: self.now(),
                        muts: Vec::new(),
                        reads: Vec::new(),
                        snapshot: 0,
                        incr: None,
                        op: Mutex::new(Some(op)),
                        flush_all: None,
                        state: WriteState::new(),
                        cond: Condvar::new(),
                    });
                    q.push(Arc::clone(&pw));
                    pw
                })
                .collect()
        };
        self.core.queue_cv.notify_all();
        for pw in pending {
            let mut st = pw.state.lock().unwrap();
            while !st.done {
                st = pw.cond.wait(st).map_err(|_| io::Error::new(io::ErrorKind::Other, "commit wait poisoned"))?;
            }
            if let Some(e) = &st.err {
                return Err(io::Error::new(io::ErrorKind::Other, e.clone()));
            }
        }
        Ok(())
    }

    /// Many `INCRBY`s at once — a pipelined run of INCR commands. Every op is
    /// its own commit (own ts, own result, applied in order), but they are
    /// enqueued together and share one group-commit fsync instead of each
    /// waiting out a full fsync before the next is even submitted.
    pub fn incr_many(&self, ops: Vec<(Key, i64)>) -> io::Result<Vec<Result<i64, String>>> {
        if !self.sync_writes {
            return ops.into_iter().map(|(k, by)| self.incr_by(k, by)).collect();
        }
        let pending: Vec<Arc<PendingWrite>> = {
            let mut q = self.core.write_queue.lock().unwrap();
            ops.into_iter()
                .map(|op| {
                    let pw = Arc::new(PendingWrite {
                        ts: self.now(),
                        muts: Vec::new(),
                        reads: Vec::new(),
                        snapshot: 0,
                        incr: Some(op),
                        op: Mutex::new(None),
                        flush_all: None,
                        state: WriteState::new(),
                        cond: Condvar::new(),
                    });
                    q.push(Arc::clone(&pw));
                    pw
                })
                .collect()
        };
        self.core.queue_cv.notify_all();
        let mut out = Vec::with_capacity(pending.len());
        for pw in pending {
            let mut st = pw.state.lock().unwrap();
            while !st.done {
                st = pw.cond.wait(st).map_err(|_| io::Error::new(io::ErrorKind::Other, "commit wait poisoned"))?;
            }
            if let Some(e) = &st.err {
                return Err(io::Error::new(io::ErrorKind::Other, e.clone()));
            }
            out.push(match &st.op_err {
                Some(e) => Err(e.clone()),
                None => Ok(st.value.unwrap_or(0)),
            });
        }
        Ok(out)
    }

    /// Enqueue (durable) or apply (non-durable) one commit and wait for it.
    /// Returns its ts and the op outcome: `Ok(incr value)` on success,
    /// `Err("\0conflict")` on an OCC conflict, `Err(msg)` on an op error. WAL
    /// failures are the outer `io::Error`.
    fn submit(
        &self,
        writes: Vec<(Key, Value)>,
        reads: Vec<(Key, u64)>,
        snapshot: u64,
        incr: Option<(Key, i64)>,
        mut op: Option<OpFn>,
    ) -> io::Result<(u64, Result<Option<i64>, String>)> {
        // ── Non-durable fast path (DBSTRIKE_SYNC=0) ──
        // Apply directly to the shard maps. ts allocation, validation and
        // apply happen under `apply_lock` so commits become visible in ts
        // order (the same guarantee the flusher gives durable mode).
        if !self.sync_writes {
            let guard = self.core.apply_lock.lock().unwrap();
            let ts = self.now();
            let pw = PendingWrite {
                ts,
                muts: writes.into_iter().map(|(key, value)| Mutation { key, value, ts }).collect(),
                reads,
                snapshot,
                incr,
                op: Mutex::new(op.take()),
                flush_all: None,
                state: WriteState::new(),
                cond: Condvar::new(),
            };
            let (muts, val) = match resolve(&self.core, &pw, &BTreeMap::new()) {
                Resolved::Apply(m, v) => (m.unwrap_or(pw.muts), v),
                Resolved::Conflict => {
                    self.core.visible_ts.fetch_max(ts, Ordering::SeqCst);
                    return Ok((ts, Err(CONFLICT.into())));
                }
                Resolved::OpErr(e) => {
                    self.core.visible_ts.fetch_max(ts, Ordering::SeqCst);
                    return Ok((ts, Err(e)));
                }
            };
            apply_to_shards(&self.core, &muts);
            self.core.visible_ts.fetch_max(ts, Ordering::SeqCst);
            if ts % 64 == 0 {
                sweep_tombstones(&self.core);
            }
            drop(guard);
            let subs = self.core.subscribers.read().unwrap();
            if !subs.is_empty() {
                for m in &muts {
                    for s in subs.iter() {
                        s(m);
                    }
                }
            }
            return Ok((ts, Ok(val)));
        }

        // ── Durable path (default): route through the group-commit flusher ──
        let pending = {
            let mut q = self.core.write_queue.lock().unwrap();
            // ts reserved UNDER the queue lock: queue order == ts order, which
            // is what lets the flusher publish `visible_ts` batch by batch.
            let ts = self.now();
            let pw = Arc::new(PendingWrite {
                ts,
                muts: writes.into_iter().map(|(key, value)| Mutation { key, value, ts }).collect(),
                reads,
                snapshot,
                incr,
                op: Mutex::new(op.take()),
                flush_all: None,
                state: WriteState::new(),
                cond: Condvar::new(),
            });
            q.push(Arc::clone(&pw));
            pw
        };
        self.core.queue_cv.notify_all();

        let mut state = pending.state.lock().unwrap();
        while !state.done {
            state = pending
                .cond
                .wait(state)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "commit wait poisoned"))?;
        }
        if let Some(e) = &state.err {
            return Err(io::Error::new(io::ErrorKind::Other, e.clone()));
        }
        if state.conflict {
            return Ok((pending.ts, Err(CONFLICT.into())));
        }
        if let Some(e) = &state.op_err {
            return Ok((pending.ts, Err(e.clone())));
        }
        Ok((pending.ts, Ok(state.value)))
    }

/// Background group-commit flusher: drains the write queue, validates and
/// resolves every pending commit IN QUEUE (= timestamp) ORDER, appends the
/// survivors in one WAL write, fsyncs ONCE, applies version chains, publishes
/// `visible_ts`, and wakes all waiters. Stops when `shutdown` is set.
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
        sweep_tombstones(&core);

        // Resolve: OCC validation + INCR evaluation, in order, against the
        // shard maps plus an overlay of this batch's earlier writes. This is
        // the serialization point, so a conflict check can no longer race a
        // concurrent commit (the old check ran on the caller thread, then
        // enqueued — two txns could both validate and both commit).
        // The overlay is only consulted by commits with a read set or an
        // INCR; plain blind-write batches (the SET hot path) skip building it.
        let need_overlay = batch.iter().any(|pw| !pw.reads.is_empty() || pw.incr.is_some() || pw.op.lock().unwrap().is_some());
        let mut overlay: BTreeMap<Key, Value> = BTreeMap::new();
        let mut outcomes: Vec<Option<Resolved>> = Vec::with_capacity(batch.len());
        for pw in &batch {
            if pw.flush_all.is_some() {
                outcomes.push(None);
                continue;
            }
            let r = resolve(&core, pw, &overlay);
            if need_overlay {
                if let Some(muts) = r.muts(pw) {
                    for m in muts {
                        overlay.insert(m.key.clone(), m.value.clone());
                    }
                }
            }
            outcomes.push(Some(r));
        }

        // Durable step: one contiguous append + one fsync for the whole
        // group. Each commit is ONE atomic frame (single or batch-tagged), so
        // a crash mid-append never replays a partial commit.
        let flush_result: io::Result<()> = (|| {
            let frames: Vec<Vec<u8>> = batch
                .iter()
                .zip(&outcomes)
                .filter_map(|(pw, o)| o.as_ref()?.muts(pw))
                .filter(|m| !m.is_empty())
                .map(|m| if m.len() == 1 { encode_single_record(&m[0]) } else { encode_batch_record(m) })
                .collect();
            if frames.is_empty() {
                return Ok(());
            }
            let max_ts = batch.iter().map(|pw| pw.ts).max().unwrap_or(0);
            let mut wal = core.wal.lock().unwrap();
            wal.append_frames(&frames)?;
            core.last_appended.fetch_max(max_ts, Ordering::SeqCst);
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
                    Ok(()) => {
                        eprintln!("[FLUSH] full wipe OK · WAL+snap backed up at {}", bak);
                        for f in core.flush_subscribers.read().unwrap().iter() {
                            f();
                        }
                    }
                    Err(e) => flush_all_err = Some(e.to_string()),
                }
            }
        }

        // Visibility + broadcast happen ONLY if the durable flush succeeded.
        if flush_result.is_ok() && !wipe_ran {
            let all: Vec<&Mutation> = batch
                .iter()
                .zip(&outcomes)
                .filter_map(|(pw, o)| o.as_ref()?.muts(pw))
                .flatten()
                .collect();
            apply_refs_to_shards(&core, &all);
            let subs = core.subscribers.read().unwrap();
            for m in &all {
                for s in subs.iter() {
                    s(m);
                }
            }
        }
        // Publish: every ts in this batch is now applied (or skipped), and
        // all earlier ts were in earlier batches. Must precede the wakeups so
        // a client reads its own write.
        if let Some(max) = batch.iter().map(|pw| pw.ts).max() {
            core.visible_ts.fetch_max(max, Ordering::SeqCst);
        }
        // Wake every waiter with the outcome.
        for (pw, out) in batch.iter().zip(outcomes) {
            let mut st = pw.state.lock().unwrap();
            st.done = true;
            if let Err(e) = &flush_result {
                st.err = Some(e.to_string());
            } else if let Some(e) = &flush_all_err {
                st.err = Some(e.clone());
            } else {
                match out {
                    Some(Resolved::Apply(_, v)) => st.value = v,
                    Some(Resolved::Conflict) => st.conflict = true,
                    Some(Resolved::OpErr(e)) => st.op_err = Some(e),
                    None => {}
                }
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
    // Make the renames + new file durable before acking the wipe.
    crate::wal::sync_parent_dir(wal_path);
    for sh in &core.shards {
        sh.write().unwrap().clear();
    }
    core.kv_live.store(0, Ordering::Relaxed);
    core.graveyard.lock().unwrap().clear();
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
        // NON-BLOCKING. The old checkpoint held the WAL mutex while copying
        // the entire dataset and writing the snapshot, stalling every commit
        // for the duration. Now writers are blocked only for the instant it
        // takes to rotate the log:
        //   1. Under the WAL lock: rename the live WAL to `<wal>.ckpt` and
        //      open a fresh one. S = highest ts in the rotated segment; every
        //      later commit has a larger ts and lands in the new WAL.
        //   2. Pin S: register it as an open snapshot, so tombstone GC and
        //      version pruning both keep what S needs.
        //   3. Stream the state visible at S into `<snap>.tmp`, one shard at
        //      a time, fsync, atomically rename over `<snap>`.
        //   4. Delete `<wal>.ckpt` — the snapshot now covers it.
        // Crash at any point: recovery = snapshot + `.ckpt` (if present) +
        // live WAL, which is complete either way.
        let _one = self.ckpt_lock.lock().unwrap();
        let pending = pending_path_for(&self.wal_path);
        let snap_ts = {
            let mut wal = self.core.wal.lock().unwrap();
            wal.sync()?;
            if pending.exists() {
                // An earlier checkpoint died after rotating. Its segment is
                // still uncovered: fold the live WAL into it (keeping the
                // order snapshot → .ckpt → live) instead of overwriting it.
                let mut seg = OpenOptions::new().append(true).open(&pending)?;
                let mut live = File::open(&self.wal_path)?;
                io::copy(&mut live, &mut seg)?;
                seg.sync_all()?;
                wal.truncate()?;
            } else {
                std::fs::rename(&self.wal_path, &pending)?;
                *wal = Wal::open(&self.wal_path)?;
                crate::wal::sync_parent_dir(&self.wal_path);
            }
            self.core.last_appended.load(Ordering::SeqCst)
        };
        // Pin S (register under the GC lock, like `begin`).
        {
            let mut a = self.core.active.lock().unwrap();
            *a.entry(snap_ts).or_insert(0) += 1;
        }
        let result = (|| -> io::Result<(u64, u64)> {
            // Everything up to S was appended; wait until it is also applied.
            while self.core.visible_ts.load(Ordering::SeqCst) < snap_ts {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let snap_path = snap_path_for(&self.wal_path);
            let n = self.write_snapshot_at(&snap_path, snap_ts)?;
            std::fs::remove_file(&pending)?;
            crate::wal::sync_parent_dir(&self.wal_path);
            let bytes = std::fs::metadata(&snap_path).map(|m| m.len()).unwrap_or(0);
            Ok((n, bytes))
        })();
        {
            let mut a = self.core.active.lock().unwrap();
            if let Some(c) = a.get_mut(&snap_ts) {
                *c -= 1;
                if *c == 0 {
                    a.remove(&snap_ts);
                }
            }
        }
        result
    }

    /// Stream every live key visible at `snap` into the snapshot file
    /// (format: `[count u64][len u32, mutation]*[crc32 u32]`), holding each
    /// shard's read lock only while copying that shard. The state at a
    /// pinned snapshot is immutable, so the counting pass and the writing
    /// pass agree.
    fn write_snapshot_at(&self, snap_path: &Path, snap: u64) -> io::Result<u64> {
        let tmp_path: PathBuf = {
            let mut s = snap_path.as_os_str().to_owned();
            s.push(".tmp");
            PathBuf::from(s)
        };
        let count: u64 = self
            .core
            .shards
            .iter()
            .map(|sh| sh.read().unwrap().values().filter(|c| c.visible(snap).is_some()).count() as u64)
            .sum();
        let f = OpenOptions::new().create(true).truncate(true).write(true).open(&tmp_path)?;
        let mut w = io::BufWriter::with_capacity(1 << 20, f);
        let mut crc = crate::crc::Crc32Hasher::default();
        let mut emit = |w: &mut io::BufWriter<File>, b: &[u8]| -> io::Result<()> {
            crc.update(b);
            w.write_all(b)
        };
        emit(&mut w, &count.to_le_bytes())?;
        let mut written = 0u64;
        for sh in &self.core.shards {
            let page: Vec<Vec<u8>> = {
                let data = sh.read().unwrap();
                data.iter()
                    .filter_map(|(k, c)| {
                        let v = c.visible(snap)?;
                        let ts = c.versions.iter().rev().find(|x| x.ts <= snap).map_or(0, |x| x.ts);
                        Some(Mutation { key: k.clone(), value: v.clone(), ts }.encode())
                    })
                    .collect()
            };
            for rec in page {
                emit(&mut w, &(rec.len() as u32).to_le_bytes())?;
                emit(&mut w, &rec)?;
                written += 1;
            }
        }
        if written != count {
            return Err(io::Error::new(io::ErrorKind::Other, "snapshot changed while pinned (bug)"));
        }
        let sum = crc.finish();
        w.write_all(&sum.to_le_bytes())?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp_path, snap_path)?;
        crate::wal::sync_parent_dir(snap_path);
        Ok(written)
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
        let pending = {
            let mut q = self.core.write_queue.lock().unwrap();
            let pw = Arc::new(PendingWrite {
                ts: self.now(),
                muts: Vec::new(),
                reads: Vec::new(),
                snapshot: 0,
                incr: None,
                op: Mutex::new(None),
                flush_all: Some(backup.clone()),
                state: WriteState::new(),
                cond: Condvar::new(),
            });
            q.push(Arc::clone(&pw));
            pw
        };
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

/// Read/write access to the keyspace for code that must run unchanged both
/// inside a `Txn` (snapshot reads + optimistic commit) and inside an
/// operation executed on the commit path (`Engine::submit_op`).
pub trait Store {
    fn get(&mut self, key: &[u8]) -> Option<Value>;
    fn put(&mut self, key: Key, value: Value);
    fn delete(&mut self, key: Key);
    /// See `Txn::scan_range`.
    fn scan_range(&mut self, hint: &[u8], start: &[u8], end: &[u8], limit: usize, reverse: bool) -> Vec<(Key, Value)>;
    /// Commit ts of the newest version of `key` visible here (`u64::MAX` if
    /// written in the current batch/transaction).
    fn version(&mut self, key: &[u8]) -> u64;
    fn has_writes(&self) -> bool;
    /// Drop every buffered write (e.g. the command failed).
    fn discard(&mut self);
}

/// An operation run on the commit path.
pub type OpFn = Box<dyn FnOnce(&mut dyn Store) + Send>;

/// `Store` for operations on the commit path: reads see the applied state
/// plus every earlier commit of the current batch (`overlay`).
pub struct OpStore<'a> {
    core: &'a FlushCore,
    overlay: &'a BTreeMap<Key, Value>,
    writes: BTreeMap<Key, Value>,
}

fn live(v: &Value) -> Option<Value> {
    (!matches!(v, Value::Tombstone)).then(|| v.clone())
}

impl Store for OpStore<'_> {
    fn get(&mut self, key: &[u8]) -> Option<Value> {
        if let Some(v) = self.writes.get(key) {
            return live(v);
        }
        if let Some(v) = self.overlay.get(key) {
            return live(v);
        }
        let data = self.core.shards[shard_of(key)].read().unwrap();
        data.get(key).and_then(|c| c.latest().cloned())
    }

    fn put(&mut self, key: Key, value: Value) {
        self.writes.insert(key, value);
    }

    fn delete(&mut self, key: Key) {
        self.writes.insert(key, Value::Tombstone);
    }

    fn scan_range(&mut self, hint: &[u8], start: &[u8], end: &[u8], limit: usize, reverse: bool) -> Vec<(Key, Value)> {
        if start >= end || limit == 0 {
            return Vec::new();
        }
        // Lazy three-way merge in key order — own writes > batch overlay >
        // applied state — stopping after `limit` live entries. (Over-fetching
        // `limit + overlay` and cloning it made a batch of N pops from one
        // sorted set O(N²) in clones.)
        let range = start.to_vec()..end.to_vec();
        let data = self.core.shards[shard_of(hint)].read().unwrap();
        type It<'x> = Box<dyn Iterator<Item = (&'x Key, Option<&'x Value>)> + 'x>;
        let mk = |rev: bool| -> [It<'_>; 3] {
            let w = self.writes.range(range.clone()).map(|(k, v)| (k, Some(v)));
            let o = self.overlay.range(range.clone()).map(|(k, v)| (k, Some(v)));
            let d = data.range(range.clone()).map(|(k, c)| (k, c.latest()));
            if rev {
                [Box::new(w.rev()), Box::new(o.rev()), Box::new(d.rev())]
            } else {
                [Box::new(w), Box::new(o), Box::new(d)]
            }
        };
        let mut its: Vec<std::iter::Peekable<It<'_>>> = mk(reverse).into_iter().map(|i| i.peekable()).collect();
        let mut out = Vec::new();
        while out.len() < limit {
            // Next key across the three sources (min, or max when reverse).
            let mut next: Option<&Key> = None;
            for it in its.iter_mut() {
                if let Some((k, _)) = it.peek() {
                    let better = match next {
                        None => true,
                        Some(n) => if reverse { *k > n } else { *k < n },
                    };
                    if better {
                        next = Some(k);
                    }
                }
            }
            let Some(key) = next.cloned() else { break };
            // Highest-priority source holding this key wins; advance all.
            let mut chosen: Option<Option<&Value>> = None;
            for it in its.iter_mut() {
                if it.peek().is_some_and(|(k, _)| **k == key) {
                    let (_, v) = it.next().unwrap();
                    if chosen.is_none() {
                        chosen = Some(v);
                    }
                }
            }
            if let Some(Some(v)) = chosen {
                if !matches!(v, Value::Tombstone) {
                    out.push((key, v.clone()));
                }
            }
        }
        out
    }

    fn version(&mut self, key: &[u8]) -> u64 {
        if self.writes.contains_key(key) || self.overlay.contains_key(key) {
            return u64::MAX;
        }
        let data = self.core.shards[shard_of(key)].read().unwrap();
        data.get(key).map_or(0, |c| c.latest_ts)
    }

    fn has_writes(&self) -> bool {
        !self.writes.is_empty()
    }

    fn discard(&mut self) {
        self.writes.clear();
    }
}

impl Drop for Txn {
    fn drop(&mut self) {
        let mut a = self.engine.core.active.lock().unwrap();
        if let Some(c) = a.get_mut(&self.snapshot) {
            *c -= 1;
            if *c == 0 {
                a.remove(&self.snapshot);
            }
        }
    }
}

impl Store for Txn {
    fn get(&mut self, key: &[u8]) -> Option<Value> {
        Txn::get(self, key)
    }
    fn put(&mut self, key: Key, value: Value) {
        Txn::put(self, key, value)
    }
    fn delete(&mut self, key: Key) {
        Txn::delete(self, key)
    }
    fn scan_range(&mut self, hint: &[u8], start: &[u8], end: &[u8], limit: usize, reverse: bool) -> Vec<(Key, Value)> {
        Txn::scan_range(self, hint, start, end, limit, reverse)
    }
    fn version(&mut self, key: &[u8]) -> u64 {
        if self.writes.contains_key(key) {
            return u64::MAX;
        }
        self.engine.key_version(key)
    }
    fn has_writes(&self) -> bool {
        Txn::has_writes(self)
    }
    fn discard(&mut self) {
        self.writes.clear();
    }
}

/// A transaction with snapshot-isolation + optimistic write-conflict detection.
pub struct Txn {
    engine: Arc<Engine>,
    snapshot: u64,
    writes: BTreeMap<Key, Value>,
    reads: Vec<Key>,
    /// Extra (key, snapshot) validations — Redis WATCH: the commit fails if
    /// `key` changed after the given snapshot.
    watches: Vec<(Key, u64)>,
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

    /// The snapshot this transaction reads at.
    pub fn snapshot(&self) -> u64 {
        self.snapshot
    }

    /// True if the transaction has buffered any write.
    pub fn has_writes(&self) -> bool {
        !self.writes.is_empty()
    }

    /// Fail the commit if `key` changed after `snapshot` (Redis WATCH).
    pub fn watch(&mut self, key: Key, snapshot: u64) {
        self.watches.push((key, snapshot));
    }

    /// Range read `[start, end)` at the txn snapshot merged with this txn's
    /// own buffered writes, in key order (`reverse` = descending), at most
    /// `limit` entries. Every key in the range must share `hint`'s shard
    /// (hash tag), so only one shard is touched. Ranges are not added to the
    /// read set: callers protect them through a header key they `get`.
    pub fn scan_range(&mut self, hint: &[u8], start: &[u8], end: &[u8], limit: usize, reverse: bool) -> Vec<(Key, Value)> {
        if start >= end || limit == 0 {
            return Vec::new();
        }
        // Over-fetch by the number of local writes in range, which can at
        // most hide that many committed entries.
        let local: Vec<(&Key, &Value)> = self.writes.range(start.to_vec()..end.to_vec()).collect();
        let fetch = limit.saturating_add(local.len());
        let base = if reverse {
            self.engine.scan_pinned_reverse(hint, start, end, self.snapshot, fetch)
        } else {
            self.engine.scan_pinned_limit(hint, start, end, self.snapshot, fetch)
        };
        let mut merged: BTreeMap<Key, Value> = base.into_iter().collect();
        for (k, v) in local {
            if matches!(v, Value::Tombstone) {
                merged.remove(k);
            } else {
                merged.insert(k.clone(), v.clone());
            }
        }
        if reverse {
            merged.into_iter().rev().take(limit).collect()
        } else {
            merged.into_iter().take(limit).collect()
        }
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
        // The read set travels with the commit and is validated by the
        // flusher at the serialization point, so no other commit can slip in
        // between validation and apply.
        let mut this = self;
        let writes: Vec<(Key, Value)> = std::mem::take(&mut this.writes).into_iter().collect();
        let snap = this.snapshot;
        let mut reads: Vec<(Key, u64)> = std::mem::take(&mut this.reads).into_iter().map(|k| (k, snap)).collect();
        reads.extend(std::mem::take(&mut this.watches));
        match this.engine.submit(writes, reads, 0, None, None) {
            Ok((ts, Ok(_))) => Ok(ts),
            Ok((_, Err(e))) if e == CONFLICT => Err(TxnError::Conflict),
            Ok((_, Err(e))) => Err(TxnError::Io(io::Error::new(io::ErrorKind::Other, e))),
            Err(e) => Err(TxnError::Io(e)),
        }
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

    fn tmp_keep(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("dbstrike_eng_{}", std::process::id())).join(name)
    }

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

    /// Read-modify-write through `Txn` from many threads with retry on
    /// conflict must never lose an update. Before validation moved into the
    /// flusher, two txns could both validate and both commit.
    #[test]
    fn concurrent_txns_never_lose_updates() {
        let e = Engine::open(tmp("occ.wal")).unwrap();
        e.put(b"c".to_vec(), Value::Int(0)).unwrap();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let e = Arc::clone(&e);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        loop {
                            let mut t = e.begin();
                            let cur = match t.get(b"c") { Some(Value::Int(i)) => i, _ => 0 };
                            t.put(b"c".to_vec(), Value::Int(cur + 1));
                            match t.commit() {
                                Ok(_) => break,
                                Err(TxnError::Conflict) => continue,
                                Err(e) => panic!("{e}"),
                            }
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(matches!(e.get(b"c"), Some(Value::Int(400))), "{:?}", e.get(b"c"));
    }

    fn chains(e: &Engine) -> usize {
        e.core.shards.iter().map(|s| s.read().unwrap().len()).sum()
    }

    /// Deleted keys are physically reclaimed, but never while an open
    /// snapshot could still read their old value.
    #[test]
    fn tombstones_are_collected_but_not_under_open_snapshots() {
        let e = Engine::open(tmp("gc.wal")).unwrap();
        for i in 0..100u32 {
            e.put(format!("k{i}").into_bytes(), Value::Int(i as i64)).unwrap();
        }
        let reader = e.begin(); // holds a snapshot that still sees every key
        for i in 0..100u32 {
            e.delete(format!("k{i}").into_bytes()).unwrap();
        }
        e.put(b"tick".to_vec(), Value::Int(0)).unwrap(); // flusher sweeps
        let mut r = reader;
        assert!(matches!(r.get(b"k7"), Some(Value::Int(7))), "open snapshot must still see the value");
        assert!(chains(&e) >= 100);
        drop(r);
        e.put(b"tick".to_vec(), Value::Int(1)).unwrap();
        e.put(b"tick".to_vec(), Value::Int(2)).unwrap();
        assert_eq!(chains(&e), 1, "only `tick` should remain");
        assert!(e.get(b"k7").is_none());
        // History of a collected key is reported unknown, not "absent".
        assert!(e.get_at_checked(b"k7", 1).is_err());
    }

    #[test]
    fn strict_integer_parsing() {
        for ok in [&b"0"[..], b"-1", b"42", b"9223372036854775807", b"-9223372036854775808"] {
            assert!(parse_strict_i64(ok).is_some(), "{:?}", ok);
        }
        for bad in [&b""[..], b"-", b"02", b"-0", b"+5", b" 5", b"5 ", b"1.0", b"9223372036854775808"] {
            assert!(parse_strict_i64(bad).is_none(), "{:?}", bad);
        }
    }

    #[test]
    fn incr_by_is_atomic_and_checked() {
        let e = Engine::open(tmp("incr.wal")).unwrap();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let e = Arc::clone(&e);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        e.incr_by(b"n".to_vec(), 1).unwrap().unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(e.incr_by(b"n".to_vec(), 0).unwrap().unwrap(), 1600);
        e.put(b"max".to_vec(), Value::Int(i64::MAX)).unwrap();
        assert!(e.incr_by(b"max".to_vec(), 1).unwrap().is_err(), "overflow must error, not wrap");
        assert!(matches!(e.get(b"max"), Some(Value::Int(i64::MAX))));
        e.put(b"s".to_vec(), Value::Bytes(b"abc".to_vec())).unwrap();
        assert!(e.incr_by(b"s".to_vec(), 1).unwrap().is_err());
        e.put(b"t".to_vec(), Value::Bytes(b"41".to_vec())).unwrap();
        assert_eq!(e.incr_by(b"t".to_vec(), 1).unwrap().unwrap(), 42);
        drop(e);
        let e = Engine::open(tmp_keep("incr.wal")).unwrap();
        assert!(matches!(e.get(b"n"), Some(Value::Int(1600))), "incr must be durable");
    }

    /// A snapshot must be stable: once taken, later commits are invisible.
    #[test]
    fn snapshot_is_repeatable_under_concurrent_writes() {
        let e = Engine::open(tmp("snap_rr.wal")).unwrap();
        e.put(b"k".to_vec(), Value::Int(0)).unwrap();
        let w = {
            let e = Arc::clone(&e);
            std::thread::spawn(move || {
                for i in 1..300 {
                    e.put(b"k".to_vec(), Value::Int(i)).unwrap();
                }
            })
        };
        for _ in 0..300 {
            let snap = e.snapshot();
            let a = e.get_at(b"k", snap);
            std::thread::yield_now();
            let b = e.get_at(b"k", snap);
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
        w.join().unwrap();
    }

    /// GETAT semantics around pruning: before the key existed → None; inside
    /// pruned history → Err (unanswerable); retained history → the value.
    #[test]
    fn get_at_checked_distinguishes_absent_from_pruned() {
        let e = Engine::open(tmp("getat.wal")).unwrap();
        e.put(b"other".to_vec(), Value::Int(0)).unwrap();
        let before = e.snapshot();
        let mut stamps = Vec::new();
        for i in 0..(MAX_VERSIONS_PER_KEY as i64 + 4) {
            stamps.push(e.put(b"k".to_vec(), Value::Int(i)).unwrap());
        }
        assert!(matches!(e.get_at_checked(b"k", before), Ok(None)));
        assert!(e.get_at_checked(b"k", stamps[1]).is_err());
        let last = *stamps.last().unwrap();
        assert!(matches!(e.get_at_checked(b"k", last), Ok(Some(Value::Int(_)))));
    }

    /// Writes that land while a checkpoint is running are neither blocked
    /// nor lost, and recovery = snapshot + live WAL reproduces everything.
    #[test]
    fn checkpoint_does_not_lose_concurrent_writes() {
        let p = tmp("ckpt_conc.wal");
        let _ = std::fs::remove_file(snap_path_for(&p));
        {
            let e = Engine::open(&p).unwrap();
            for i in 0..5000u32 {
                e.put(format!("pre{i}").into_bytes(), Value::Int(i as i64)).unwrap();
            }
            let w = {
                let e = Arc::clone(&e);
                std::thread::spawn(move || {
                    for i in 0..2000u32 {
                        e.put(format!("during{i}").into_bytes(), Value::Int(i as i64)).unwrap();
                        e.put(b"hot".to_vec(), Value::Int(i as i64)).unwrap();
                    }
                })
            };
            for _ in 0..3 {
                e.checkpoint().unwrap();
            }
            w.join().unwrap();
            assert!(!pending_path_for(&p).exists());
        }
        let e = Engine::open(tmp_keep("ckpt_conc.wal")).unwrap();
        for i in (0..5000u32).step_by(97) {
            assert!(matches!(e.get(format!("pre{i}").as_bytes()), Some(Value::Int(v)) if v == i as i64));
        }
        for i in (0..2000u32).step_by(37) {
            assert!(e.get(format!("during{i}").as_bytes()).is_some(), "during{i} lost");
        }
        assert!(matches!(e.get(b"hot"), Some(Value::Int(1999))));
    }

    /// A checkpoint that died after rotating leaves `<wal>.ckpt`; recovery
    /// must replay it, and the next checkpoint must fold into it.
    #[test]
    fn crashed_checkpoint_rotation_recovers() {
        let p = tmp("ckpt_crash.wal");
        let _ = std::fs::remove_file(snap_path_for(&p));
        let _ = std::fs::remove_file(pending_path_for(&p));
        {
            let e = Engine::open(&p).unwrap();
            e.put(b"a".to_vec(), Value::Int(1)).unwrap();
            e.put(b"b".to_vec(), Value::Int(2)).unwrap();
        }
        // Simulate "rotated, then crashed before the snapshot".
        std::fs::rename(&p, pending_path_for(&p)).unwrap();
        {
            let e = Engine::open(tmp_keep("ckpt_crash.wal")).unwrap();
            assert!(matches!(e.get(b"a"), Some(Value::Int(1))), "pending segment replayed");
            e.put(b"c".to_vec(), Value::Int(3)).unwrap();
            e.checkpoint().unwrap(); // folds live WAL into the pending segment
            assert!(!pending_path_for(&p).exists());
            e.delete(b"b".to_vec()).unwrap();
        }
        let e = Engine::open(tmp_keep("ckpt_crash.wal")).unwrap();
        assert!(matches!(e.get(b"a"), Some(Value::Int(1))));
        assert!(e.get(b"b").is_none());
        assert!(matches!(e.get(b"c"), Some(Value::Int(3))));
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
