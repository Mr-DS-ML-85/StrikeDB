//! Redis keyspace executor: strings, key expiry (TTL), hashes, lists, sets and
//! sorted sets, all stored on the MVCC engine and executed inside an engine
//! transaction.
//!
//! Layout (every key below lives in the one engine keyspace):
//!   * `kv:<key>`            string value (`Bytes`/`Int`), or a collection
//!                            header (`Value::Meta`: type, length, list
//!                            head/tail, version)
//!   * `c:{<hex key>}:h<f>`  hash field          -> Bytes
//!   * `c:{<hex key>}:s<m>`  set member          -> Int(position)
//!   * `c:{<hex key>}:t<pos>` set member by position -> Bytes(member)
//!   * `c:{<hex key>}:l<seq>` list element (seq = order-preserving i64) -> Bytes
//!   * `c:{<hex key>}:m<m>`  sorted-set member   -> Float(score)
//!   * `c:{<hex key>}:z<score><m>` sorted-set score index -> Int(1)
//!   * `exp:<key>`           absolute expiry, unix ms -> Int
//!
//! The `{hex}` hash tag pins a collection's elements to one engine shard, so
//! range reads touch a single shard. Lists keep their elements at contiguous
//! sequence numbers `head..tail`, making push/pop/LINDEX/LSET O(1) and LRANGE
//! O(m).
//!
//! Every command runs against a `Txn`; `Keyspace::run` commits it with
//! automatic retry on an optimistic-concurrency conflict, and MULTI/EXEC runs
//! a whole queue inside ONE transaction — so EXEC is atomic, isolated and
//! durable, not merely "not interleaved". Each structural change rewrites the
//! header (bumping its version), and every operation reads the header, so a
//! concurrent change to the same collection always surfaces as a conflict.

use protocol::Resp;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use storage::{Engine, Store, Value};

pub const WRONGTYPE: &str = "WRONGTYPE Operation against a key holding the wrong kind of value";

fn err(msg: &str) -> Resp {
    super::err(msg)
}

fn wrong_args(name: &str) -> Resp {
    err(&format!("wrong number of arguments for '{}' command", name.to_lowercase()))
}

fn ok() -> Resp {
    Resp::Simple("OK".into())
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── Key encodings ──────────────────────────────────────────────────────────

const T_HASH: u8 = 1;
const T_LIST: u8 = 2;
const T_SET: u8 = 3;
const T_ZSET: u8 = 4;

const SUB_HASH: u8 = b'h';
const SUB_SET: u8 = b's';
/// Set members by dense position 0..len (swap-remove on delete), so
/// SPOP/SRANDMEMBER pick uniformly in O(1) instead of reading the whole set.
const SUB_SETPOS: u8 = b't';
const SUB_LIST: u8 = b'l';
const SUB_ZMEM: u8 = b'm';
const SUB_ZSCORE: u8 = b'z';

fn kv(key: &[u8]) -> Vec<u8> {
    let mut k = b"kv:".to_vec();
    k.extend_from_slice(key);
    k
}

fn expk(key: &[u8]) -> Vec<u8> {
    let mut k = b"exp:".to_vec();
    k.extend_from_slice(key);
    k
}

/// `c:{<hex key>}:<sub>` — the common prefix of one element family.
fn ebase(key: &[u8], sub: u8) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut k = Vec::with_capacity(6 + key.len() * 2);
    k.extend_from_slice(b"c:{");
    for b in key {
        k.push(HEX[(b >> 4) as usize]);
        k.push(HEX[(b & 15) as usize]);
    }
    k.extend_from_slice(b"}:");
    k.push(sub);
    k
}

fn ekey(key: &[u8], sub: u8, rest: &[u8]) -> Vec<u8> {
    let mut k = ebase(key, sub);
    k.extend_from_slice(rest);
    k
}

/// `[start, end)` covering every key of one element family.
fn erange(key: &[u8], sub: u8) -> (Vec<u8>, Vec<u8>) {
    (ebase(key, sub), ebase(key, sub + 1))
}

fn seq_bytes(i: i64) -> [u8; 8] {
    ((i as u64) ^ (1 << 63)).to_be_bytes()
}

#[cfg(test)]
fn seq_from(b: &[u8]) -> i64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    (u64::from_be_bytes(a) ^ (1 << 63)) as i64
}

/// Order-preserving encoding of an f64 score.
fn score_bytes(f: f64) -> [u8; 8] {
    let f = if f == 0.0 { 0.0 } else { f }; // -0.0 == 0.0
    let b = f.to_bits();
    let o = if b >> 63 == 1 { !b } else { b ^ (1 << 63) };
    o.to_be_bytes()
}

fn score_from(b: &[u8]) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    let o = u64::from_be_bytes(a);
    let b = if o >> 63 == 1 { o ^ (1 << 63) } else { !o };
    f64::from_bits(b)
}

/// Redis-style double formatting (`1`, `1.5`, `inf`, `-inf`).
pub fn fmt_f64(f: f64) -> String {
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f.fract() == 0.0 && f.abs() < 1e17 {
        return format!("{}", f as i64);
    }
    format!("{f}")
}

fn parse_f64(a: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(a).ok()?.trim();
    let v = match s.to_ascii_lowercase().as_str() {
        "inf" | "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => s.parse::<f64>().ok()?,
    };
    (!v.is_nan()).then_some(v)
}

/// Redis `string2ll` rules: optional '-', no '+', no leading zeros, no
/// spaces. ("02" or "+5" are not integers to Redis, and INCR must agree.)
fn parse_i64(a: &[u8]) -> Option<i64> {
    storage::engine::parse_strict_i64(a)
}

fn not_int() -> Resp {
    err("value is not an integer or out of range")
}

fn not_float() -> Resp {
    err("value is not a valid float")
}

// ── Collection header ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Meta {
    ty: u8,
    len: u64,
    head: i64,
    tail: i64,
    ver: u64,
}

impl Meta {
    fn new(ty: u8) -> Self {
        Meta { ty, len: 0, head: 0, tail: 0, ver: 0 }
    }

    fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(33);
        v.push(self.ty);
        v.extend_from_slice(&self.len.to_le_bytes());
        v.extend_from_slice(&self.head.to_le_bytes());
        v.extend_from_slice(&self.tail.to_le_bytes());
        v.extend_from_slice(&self.ver.to_le_bytes());
        v
    }

    fn decode(b: &[u8]) -> Option<Self> {
        if b.len() != 33 {
            return None;
        }
        let u = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Some(Meta { ty: b[0], len: u(1), head: u(9) as i64, tail: u(17) as i64, ver: u(25) })
    }

    fn type_name(&self) -> &'static str {
        match self.ty {
            T_HASH => "hash",
            T_LIST => "list",
            T_SET => "set",
            T_ZSET => "zset",
            _ => "unknown",
        }
    }
}

fn type_subs(ty: u8) -> &'static [u8] {
    match ty {
        T_HASH => &[SUB_HASH],
        T_LIST => &[SUB_LIST],
        T_SET => &[SUB_SET, SUB_SETPOS],
        T_ZSET => &[SUB_ZMEM, SUB_ZSCORE],
        _ => &[],
    }
}

// ── Expiry index ───────────────────────────────────────────────────────────

/// In-memory mirror of the durable `exp:` keys: answers "is this key
/// expired?" on the hot GET path without an engine read, and drives the
/// active-expiry reaper in deadline order.
#[derive(Default)]
pub struct Expiry {
    any: AtomicBool,
    inner: RwLock<ExpInner>,
}

#[derive(Default)]
struct ExpInner {
    at: HashMap<Vec<u8>, u64>,
    order: BTreeSet<(u64, Vec<u8>)>,
}

impl Expiry {
    /// Fast path: false until the first TTL is ever set.
    pub fn any(&self) -> bool {
        self.any.load(Ordering::Relaxed)
    }

    pub fn get(&self, key: &[u8]) -> Option<u64> {
        if !self.any() {
            return None;
        }
        self.inner.read().unwrap().at.get(key).copied()
    }

    pub fn is_expired(&self, key: &[u8], now: u64) -> bool {
        self.get(key).is_some_and(|at| at <= now)
    }

    fn set(&self, key: &[u8], at: Option<u64>) {
        let mut g = self.inner.write().unwrap();
        if let Some(old) = g.at.remove(key) {
            g.order.remove(&(old, key.to_vec()));
        }
        if let Some(at) = at {
            g.at.insert(key.to_vec(), at);
            g.order.insert((at, key.to_vec()));
            self.any.store(true, Ordering::Relaxed);
        }
    }

    pub fn apply(&self, changes: &[(Vec<u8>, Option<u64>)]) {
        for (k, at) in changes {
            self.set(k, *at);
        }
    }

    /// Up to `limit` keys whose deadline has passed.
    fn due(&self, now: u64, limit: usize) -> Vec<Vec<u8>> {
        let g = self.inner.read().unwrap();
        g.order.iter().take_while(|(at, _)| *at <= now).take(limit).map(|(_, k)| k.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().at.len()
    }

    pub fn clear(&self) {
        let mut g = self.inner.write().unwrap();
        g.at.clear();
        g.order.clear();
    }
}

// ── Keyspace ───────────────────────────────────────────────────────────────

pub struct Keyspace {
    engine: Arc<Engine>,
    pub expiry: Expiry,
    /// True once any collection exists; until then SET/DEL keep their
    /// zero-read fast paths (no "was this key a hash?" check needed).
    colls: AtomicBool,
}

static RNG: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);

/// xorshift; quality is irrelevant here (SPOP/SRANDMEMBER picks).
fn rand() -> u64 {
    let mut x = RNG.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    RNG.store(x, Ordering::Relaxed);
    x
}

/// What one executor run produced, carried back from the commit thread.
struct Outcome {
    reply: Resp,
    exp: Vec<(Vec<u8>, Option<u64>)>,
    created: bool,
}

/// Outcome of a MULTI/EXEC.
pub enum ExecResult {
    /// A WATCHed key changed: reply nil, nothing executed.
    Aborted,
    Done(Vec<Resp>),
}

impl Keyspace {
    pub fn open(engine: Arc<Engine>) -> Self {
        let ks = Keyspace {
            engine: Arc::clone(&engine),
            expiry: Expiry::default(),
            colls: AtomicBool::new(false),
        };
        ks.reload();
        ks
    }

    /// Rebuild the RAM mirrors (expiry index, collections flag) from the
    /// substrate: startup, FLUSHALL, and replicas after a full sync.
    pub fn reload(&self) {
        self.expiry.clear();
        self.expiry.any.store(false, Ordering::Relaxed);
        let snap = self.engine.snapshot();
        for (k, v) in self.engine.scan_prefix(b"exp:", snap) {
            if let Value::Int(at) = v {
                self.expiry.set(&k[4..], Some(at as u64));
            }
        }
        let has = !self.engine.scan_from(b"c:{", b"c:|", 1, snap).is_empty();
        self.colls.store(has, Ordering::Relaxed);
    }

    pub fn has_collections(&self) -> bool {
        self.colls.load(Ordering::Relaxed)
    }

    /// Note keys written by an external path (replication) so the RAM
    /// mirrors stay in step.
    pub fn observe(&self, key: &[u8], value: &Value) {
        if let Some(user) = key.strip_prefix(b"exp:") {
            match value {
                Value::Int(at) => self.expiry.set(user, Some(*at as u64)),
                _ => self.expiry.set(user, None),
            }
        } else if key.starts_with(b"c:{") {
            self.colls.store(true, Ordering::Relaxed);
        }
    }

    /// Run one command.
    ///
    /// Reads run on a consistent snapshot without touching the commit path.
    /// Writes run AS an operation on the commit path (`Engine::submit_op`):
    /// serialized against every other write with no conflicts or retries,
    /// and sharing the batch's single fsync — so 50 clients pushing to one
    /// list don't degrade into an optimistic-retry storm.
    pub fn run(&self, name: &str, args: &[Vec<u8>]) -> Resp {
        if !is_keyspace_write(name) {
            let mut tx = self.engine.begin();
            let mut cx = Ctx { tx: &mut tx, now: now_ms(), exp: Vec::new(), created: false, readonly: true };
            let r = exec(&mut cx, name, args);
            if !tx.has_writes() {
                return r;
            }
            // A command classified read-only tried to write: run it properly.
        }
        let (name, args) = (name.to_string(), args.to_vec());
        self.on_commit_path(move |st, now| {
            let mut cx = Ctx { tx: st, now, exp: Vec::new(), created: false, readonly: false };
            let reply = exec(&mut cx, &name, &args);
            if matches!(reply, Resp::Error(_)) {
                cx.tx.discard(); // a failed command writes nothing
                cx.exp.clear();
            }
            Outcome { reply, exp: cx.exp, created: cx.created }
        })
    }

    /// Run a pipelined sequence of WRITE commands: each executes on the
    /// commit path in order (exactly as `run` would), but all share one
    /// submission and therefore one group fsync.
    pub fn run_writes(&self, cmds: &[(String, Vec<Vec<u8>>)]) -> Vec<Resp> {
        let slots: Vec<Arc<std::sync::Mutex<Option<Outcome>>>> =
            cmds.iter().map(|_| Arc::new(std::sync::Mutex::new(None))).collect();
        let ops: Vec<storage::OpFn> = cmds
            .iter()
            .cloned()
            .zip(slots.iter().cloned())
            .map(|((name, args), slot)| {
                Box::new(move |st: &mut dyn Store| {
                    let mut cx = Ctx { tx: st, now: now_ms(), exp: Vec::new(), created: false, readonly: false };
                    let reply = exec(&mut cx, &name, &args);
                    if matches!(reply, Resp::Error(_)) {
                        cx.tx.discard();
                        cx.exp.clear();
                    }
                    *slot.lock().unwrap() = Some(Outcome { reply, exp: cx.exp, created: cx.created });
                }) as storage::OpFn
            })
            .collect();
        if let Err(e) = self.engine.submit_ops(ops) {
            let e = err(&e.to_string());
            return cmds.iter().map(|_| e.clone()).collect();
        }
        slots
            .into_iter()
            .map(|s| match s.lock().unwrap().take() {
                Some(o) => {
                    self.after_commit(&o.exp, o.created);
                    o.reply
                }
                None => err("operation did not run"),
            })
            .collect()
    }

    /// Execute `f` on the commit path and apply its side effects afterwards.
    fn on_commit_path(&self, f: impl FnOnce(&mut dyn Store, u64) -> Outcome + Send + 'static) -> Resp {
        let slot: Arc<std::sync::Mutex<Option<Outcome>>> = Arc::new(std::sync::Mutex::new(None));
        let out = Arc::clone(&slot);
        let res = self.engine.submit_op(Box::new(move |st: &mut dyn Store| {
            *out.lock().unwrap() = Some(f(st, now_ms()));
        }));
        if let Err(e) = res {
            return err(&e.to_string());
        }
        let Some(o) = slot.lock().unwrap().take() else { return err("operation did not run") };
        self.after_commit(&o.exp, o.created);
        o.reply
    }

    /// Execute a MULTI queue atomically. `watches` are (key, snapshot taken
    /// at WATCH time): if any changed since, the transaction is aborted.
    pub fn exec_multi(&self, cmds: &[Vec<Vec<u8>>], watches: &[(Vec<u8>, u64)]) -> ExecResult {
        // The whole queue is ONE commit-path operation: nothing interleaves,
        // its writes commit atomically in one frame, and the WATCH check runs
        // at the exact serialization point (no check-then-commit window).
        let n = cmds.len();
        let (cmds, watches) = (cmds.to_vec(), watches.to_vec());
        let reply = self.on_commit_path(move |st, now| {
            let broken = watches.iter().any(|(k, snap)| {
                st.version(&kv(k)) > *snap
                    || st.version(&expk(k)) > *snap
                    // A key whose TTL passed after WATCH counts as modified.
                    || matches!(st.get(&expk(k)), Some(Value::Int(at)) if (at as u64) <= now)
            });
            if broken {
                return Outcome { reply: Resp::Nil, exp: Vec::new(), created: false };
            }
            let mut cx = Ctx { tx: st, now, exp: Vec::new(), created: false, readonly: false };
            let mut out = Vec::with_capacity(cmds.len());
            for c in &cmds {
                let name = String::from_utf8_lossy(&c[0]).to_ascii_uppercase();
                out.push(exec(&mut cx, &name, &c[1..]));
            }
            Outcome { reply: Resp::Array(out), exp: cx.exp, created: cx.created }
        });
        match reply {
            Resp::Nil => ExecResult::Aborted,
            Resp::Array(v) => ExecResult::Done(v),
            e => ExecResult::Done((0..n).map(|_| e.clone()).collect()),
        }
    }

    fn after_commit(&self, exp: &[(Vec<u8>, Option<u64>)], created: bool) {
        if created {
            self.colls.store(true, Ordering::Relaxed);
        }
        self.expiry.apply(exp);
    }

    /// Active expiry: delete up to `limit` keys whose TTL has passed.
    /// Returns how many were examined.
    pub fn reap(&self, limit: usize) -> usize {
        let now = now_ms();
        let due = self.expiry.due(now, limit);
        for key in &due {
            self.purge(key);
            // If the durable state had no (or a later) deadline, the index
            // entry was stale: drop it so it is not retried forever.
            if self.expiry.get(key).is_some_and(|at| at <= now) {
                self.expiry.set(key, None);
            }
        }
        due.len()
    }

    /// Delete `key` if (per the durable state) its TTL has passed. Runs on the
    /// commit path: a read-only snapshot load deliberately never deletes.
    fn purge(&self, key: &[u8]) {
        let key = key.to_vec();
        self.on_commit_path(move |st, now| {
            let mut cx = Ctx { tx: st, now, exp: Vec::new(), created: false, readonly: false };
            let _ = cx.load(&key);
            Outcome { reply: Resp::Nil, exp: cx.exp, created: false }
        });
    }

    /// Lazily delete `key` now if it has expired (used before fast paths
    /// that bypass the executor, e.g. INCR).
    pub fn reap_if_expired(&self, key: &[u8]) {
        let now = now_ms();
        if self.expiry.is_expired(key, now) {
            self.purge(key);
            if self.expiry.get(key).is_some_and(|at| at <= now) {
                self.expiry.set(key, None);
            }
        }
    }
}

/// Commands this executor implements. MULTI may only queue these.
pub fn is_keyspace_cmd(name: &str) -> bool {
    matches!(
        name,
        "GET" | "SET" | "SETNX" | "SETEX" | "PSETEX" | "GETSET" | "GETDEL" | "GETEX" | "APPEND" | "STRLEN"
            | "GETRANGE" | "SETRANGE" | "INCR" | "DECR" | "INCRBY" | "DECRBY" | "INCRBYFLOAT" | "MGET" | "MSET"
            | "MSETNX" | "DEL" | "UNLINK" | "EXISTS" | "TYPE" | "RENAME" | "RENAMENX" | "EXPIRE" | "PEXPIRE"
            | "EXPIREAT" | "PEXPIREAT" | "TTL" | "PTTL" | "EXPIRETIME" | "PEXPIRETIME" | "PERSIST"
            | "HSET" | "HSETNX" | "HMSET" | "HGET" | "HMGET" | "HDEL" | "HEXISTS" | "HLEN" | "HGETALL"
            | "HKEYS" | "HVALS" | "HINCRBY" | "HINCRBYFLOAT" | "HSTRLEN"
            | "LPUSH" | "RPUSH" | "LPUSHX" | "RPUSHX" | "LPOP" | "RPOP" | "LLEN" | "LINDEX" | "LSET"
            | "LRANGE" | "LTRIM" | "LREM" | "LINSERT" | "LPOS" | "RPOPLPUSH" | "LMOVE"
            | "SADD" | "SREM" | "SISMEMBER" | "SMISMEMBER" | "SCARD" | "SMEMBERS" | "SPOP" | "SRANDMEMBER"
            | "SINTER" | "SUNION" | "SDIFF" | "SINTERSTORE" | "SUNIONSTORE" | "SDIFFSTORE" | "SMOVE"
            | "ZADD" | "ZREM" | "ZSCORE" | "ZMSCORE" | "ZINCRBY" | "ZCARD" | "ZCOUNT" | "ZRANK"
            | "ZREVRANK" | "ZRANGE" | "ZREVRANGE" | "ZRANGEBYSCORE" | "ZREVRANGEBYSCORE"
            | "ZREMRANGEBYSCORE" | "ZREMRANGEBYRANK" | "ZPOPMIN" | "ZPOPMAX" | "ZRANGEBYLEX"
            | "ZREVRANGEBYLEX" | "ZLEXCOUNT" | "ZREMRANGEBYLEX"
            | "PING" | "ECHO" | "TIME"
    )
}

/// Commands that never write (used to skip read-only replica checks etc.).
pub fn is_keyspace_write(name: &str) -> bool {
    is_keyspace_cmd(name)
        && !matches!(
            name,
            "GET" | "STRLEN" | "GETRANGE" | "MGET" | "EXISTS" | "TYPE" | "TTL" | "PTTL" | "EXPIRETIME"
                | "PEXPIRETIME" | "HGET" | "HMGET" | "HEXISTS" | "HLEN" | "HGETALL" | "HKEYS" | "HVALS"
                | "HSTRLEN" | "LLEN" | "LINDEX" | "LRANGE" | "LPOS" | "SISMEMBER" | "SMISMEMBER" | "SCARD"
                | "SMEMBERS" | "SRANDMEMBER" | "SINTER" | "SUNION" | "SDIFF" | "ZSCORE" | "ZMSCORE" | "ZCARD"
                | "ZCOUNT" | "ZRANK" | "ZREVRANK" | "ZRANGE" | "ZREVRANGE" | "ZRANGEBYSCORE"
                | "ZREVRANGEBYSCORE" | "ZRANGEBYLEX" | "ZREVRANGEBYLEX" | "ZLEXCOUNT" | "PING" | "ECHO" | "TIME"
        )
}

// ── Execution context ──────────────────────────────────────────────────────

struct Ctx<'a> {
    tx: &'a mut dyn Store,
    now: u64,
    /// Snapshot read: an expired key reads as absent but is left for the
    /// reaper instead of being deleted here.
    readonly: bool,
    /// Expiry index changes to apply after a successful commit.
    exp: Vec<(Vec<u8>, Option<u64>)>,
    /// A collection was created (flip `Keyspace::colls` after commit).
    created: bool,
}

enum Ent {
    None,
    Str(Vec<u8>),
    Coll(Meta),
}

struct Entry {
    ent: Ent,
    ttl: Option<u64>,
}

impl Entry {
    fn exists(&self) -> bool {
        !matches!(self.ent, Ent::None)
    }
}

type R<T> = Result<T, Resp>;

impl Ctx<'_> {
    /// Read a key's current entry, deleting it first if its TTL has passed.
    fn load(&mut self, key: &[u8]) -> R<Entry> {
        let ttl = match self.tx.get(&expk(key)) {
            Some(Value::Int(at)) => Some(at as u64),
            _ => None,
        };
        let raw = self.tx.get(&kv(key));
        let ent = match raw {
            None => Ent::None,
            Some(Value::Bytes(b)) => Ent::Str(b),
            Some(Value::Int(i)) => Ent::Str(i.to_string().into_bytes()),
            Some(Value::Float(f)) => Ent::Str(fmt_f64(f).into_bytes()),
            Some(Value::Meta(m)) => Ent::Coll(Meta::decode(&m).ok_or_else(|| err("corrupt collection header"))?),
            Some(_) => return Err(err(WRONGTYPE)),
        };
        let e = Entry { ent, ttl };
        if let Some(at) = ttl {
            if at <= self.now {
                if !self.readonly {
                    self.delete_all(key, &e);
                }
                return Ok(Entry { ent: Ent::None, ttl: None });
            }
        }
        Ok(e)
    }

    fn set_ttl(&mut self, key: &[u8], had: Option<u64>, at: Option<u64>) {
        match at {
            Some(at) => {
                self.tx.put(expk(key), Value::Int(at as i64));
                self.exp.push((key.to_vec(), Some(at)));
            }
            None if had.is_some() => {
                self.tx.delete(expk(key));
                self.exp.push((key.to_vec(), None));
            }
            None => {}
        }
    }

    /// Remove a key entirely: value/header, every element, and its TTL.
    fn delete_all(&mut self, key: &[u8], e: &Entry) {
        if let Ent::Coll(m) = &e.ent {
            self.delete_elements(key, m.ty);
        }
        if e.exists() {
            self.tx.delete(kv(key));
        }
        self.set_ttl(key, e.ttl, None);
    }

    /// The header of `key` if it is a collection of type `ty`; `None` if the
    /// key is absent; WRONGTYPE otherwise.
    fn coll(&mut self, key: &[u8], ty: u8) -> R<Option<(Meta, Option<u64>)>> {
        let e = self.load(key)?;
        match e.ent {
            Ent::None => Ok(None),
            Ent::Coll(m) if m.ty == ty => Ok(Some((m, e.ttl))),
            _ => Err(err(WRONGTYPE)),
        }
    }

    fn coll_or_new(&mut self, key: &[u8], ty: u8) -> R<(Meta, Option<u64>)> {
        Ok(self.coll(key, ty)?.unwrap_or((Meta::new(ty), None)))
    }

    /// Persist a header; an emptied collection disappears (with its TTL).
    fn put_meta(&mut self, key: &[u8], mut m: Meta, ttl: Option<u64>) {
        if m.len == 0 {
            self.tx.delete(kv(key));
            self.set_ttl(key, ttl, None);
            return;
        }
        if m.ver == 0 {
            self.created = true;
        }
        m.ver += 1;
        self.tx.put(kv(key), Value::Meta(m.encode()));
    }

    /// Overwrite `key` with a string, replacing any collection.
    fn put_str(&mut self, key: &[u8], val: Vec<u8>, old: &Entry, ttl: Option<u64>) {
        if let Ent::Coll(m) = &old.ent {
            self.delete_elements(key, m.ty);
        }
        self.tx.put(kv(key), Value::Bytes(val));
        self.set_ttl(key, old.ttl, ttl);
    }

    fn delete_elements(&mut self, key: &[u8], ty: u8) {
        for &sub in type_subs(ty) {
            let (s, end) = erange(key, sub);
            for (k, _) in self.tx.scan_range(&s, &s, &end, usize::MAX, false) {
                self.tx.delete(k);
            }
        }
    }

    /// Visit `[start, end)` in pages; `f` returns false to stop.
    fn each(&mut self, hint: &[u8], start: Vec<u8>, end: Vec<u8>, reverse: bool, mut f: impl FnMut(&[u8], &Value) -> bool) {
        const PAGE: usize = 256;
        let (mut s, mut e) = (start, end);
        loop {
            let page = self.tx.scan_range(hint, &s, &e, PAGE, reverse);
            let n = page.len();
            let Some(last) = page.last().map(|(k, _)| k.clone()) else { return };
            for (k, v) in &page {
                if !f(k, v) {
                    return;
                }
            }
            if n < PAGE {
                return;
            }
            if reverse {
                e = last;
            } else {
                s = last;
                s.push(0);
            }
        }
    }

    fn elements(&mut self, key: &[u8], sub: u8) -> Vec<(Vec<u8>, Value)> {
        let (s, e) = erange(key, sub);
        let plen = s.len();
        self.tx.scan_range(&s.clone(), &s, &e, usize::MAX, false).into_iter().map(|(k, v)| (k[plen..].to_vec(), v)).collect()
    }
}

fn bulk(b: Vec<u8>) -> Resp {
    Resp::Bulk(b)
}

fn int(n: i64) -> Resp {
    Resp::Int(n)
}

fn arr(v: Vec<Resp>) -> Resp {
    Resp::Array(v)
}

fn val_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::Bytes(b) => b.clone(),
        Value::Int(i) => i.to_string().into_bytes(),
        Value::Float(f) => fmt_f64(*f).into_bytes(),
        _ => Vec::new(),
    }
}

/// Normalize a Redis [start, stop] index pair against `len`. `None` = empty.
fn norm_range(start: i64, stop: i64, len: i64) -> Option<(i64, i64)> {
    let s = if start < 0 { (len + start).max(0) } else { start };
    let e = if stop < 0 { len + stop } else { stop.min(len - 1) };
    (s <= e && s < len).then_some((s, e))
}

/// Run one command. Never commits; the caller decides.
fn exec(cx: &mut Ctx, name: &str, a: &[Vec<u8>]) -> Resp {
    match exec_inner(cx, name, a) {
        Ok(r) | Err(r) => r,
    }
}

fn exec_inner(cx: &mut Ctx, name: &str, a: &[Vec<u8>]) -> R<Resp> {
    let n = a.len();
    let need = |ok: bool| if ok { Ok(()) } else { Err(wrong_args(name)) };
    match name {
        "PING" => Ok(match a.first() {
            Some(m) => bulk(m.clone()),
            None => Resp::Simple("PONG".into()),
        }),
        "ECHO" => {
            need(n == 1)?;
            Ok(bulk(a[0].clone()))
        }
        "TIME" => {
            let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            Ok(arr(vec![bulk(d.as_secs().to_string().into_bytes()), bulk(d.subsec_micros().to_string().into_bytes())]))
        }

        // ── strings ────────────────────────────────────────────────────
        "GET" => {
            need(n == 1)?;
            Ok(match cx.load(&a[0])?.ent {
                Ent::None => Resp::Nil,
                Ent::Str(v) => bulk(v),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            })
        }
        "MGET" => {
            need(n >= 1)?;
            let mut out = Vec::new();
            for k in a {
                out.push(match cx.load(k)?.ent {
                    Ent::Str(v) => bulk(v),
                    _ => Resp::Nil,
                });
            }
            Ok(arr(out))
        }
        "SET" => {
            need(n >= 2)?;
            let (mut nx, mut xx, mut get, mut keep) = (false, false, false, false);
            let mut ttl: Option<u64> = None;
            let mut i = 2;
            while i < n {
                let o = String::from_utf8_lossy(&a[i]).to_ascii_uppercase();
                match o.as_str() {
                    "NX" => nx = true,
                    "XX" => xx = true,
                    "GET" => get = true,
                    "KEEPTTL" => keep = true,
                    "EX" | "PX" | "EXAT" | "PXAT" => {
                        i += 1;
                        let v = a.get(i).and_then(|v| parse_i64(v)).ok_or_else(|| err("syntax error"))?;
                        if v <= 0 {
                            return Err(err("invalid expire time in 'set' command"));
                        }
                        ttl = Some(abs_deadline(&o, v, cx.now).ok_or_else(|| err("invalid expire time in 'set' command"))?);
                    }
                    _ => return Err(err("syntax error")),
                }
                i += 1;
            }
            if (nx && xx) || (keep && ttl.is_some()) {
                return Err(err("syntax error"));
            }
            let old = cx.load(&a[0])?;
            let prev = match &old.ent {
                Ent::Str(v) => Some(v.clone()),
                Ent::None => None,
                Ent::Coll(_) if get => return Err(err(WRONGTYPE)),
                Ent::Coll(_) => None,
            };
            let proceed = !(nx && old.exists()) && !(xx && !old.exists());
            if proceed {
                let t = if keep { old.ttl } else { ttl };
                cx.put_str(&a[0], a[1].clone(), &old, t);
            }
            Ok(if get {
                prev.map_or(Resp::Nil, bulk)
            } else if proceed {
                ok()
            } else {
                Resp::Nil
            })
        }
        "SETNX" => {
            need(n == 2)?;
            let old = cx.load(&a[0])?;
            if old.exists() {
                return Ok(int(0));
            }
            cx.put_str(&a[0], a[1].clone(), &old, None);
            Ok(int(1))
        }
        "SETEX" | "PSETEX" => {
            need(n == 3)?;
            let v = parse_i64(&a[1]).ok_or_else(not_int)?;
            if v <= 0 {
                return Err(err(&format!("invalid expire time in '{}' command", name.to_lowercase())));
            }
            let at = abs_deadline(if name == "SETEX" { "EX" } else { "PX" }, v, cx.now)
                .ok_or_else(|| err(&format!("invalid expire time in '{}' command", name.to_lowercase())))?;
            let old = cx.load(&a[0])?;
            cx.put_str(&a[0], a[2].clone(), &old, Some(at));
            Ok(ok())
        }
        "GETSET" => {
            need(n == 2)?;
            let old = cx.load(&a[0])?;
            let prev = match &old.ent {
                Ent::Str(v) => Some(v.clone()),
                Ent::None => None,
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            cx.put_str(&a[0], a[1].clone(), &old, None);
            Ok(prev.map_or(Resp::Nil, bulk))
        }
        "GETDEL" => {
            need(n == 1)?;
            let old = cx.load(&a[0])?;
            match &old.ent {
                Ent::None => Ok(Resp::Nil),
                Ent::Coll(_) => Err(err(WRONGTYPE)),
                Ent::Str(v) => {
                    let v = v.clone();
                    cx.delete_all(&a[0], &old);
                    Ok(bulk(v))
                }
            }
        }
        "GETEX" => {
            need(n >= 1)?;
            let old = cx.load(&a[0])?;
            let v = match &old.ent {
                Ent::None => return Ok(Resp::Nil),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
                Ent::Str(v) => v.clone(),
            };
            if n > 1 {
                let o = String::from_utf8_lossy(&a[1]).to_ascii_uppercase();
                if o == "PERSIST" && n == 2 {
                    cx.set_ttl(&a[0], old.ttl, None);
                } else if n == 3 && matches!(o.as_str(), "EX" | "PX" | "EXAT" | "PXAT") {
                    let t = parse_i64(&a[2]).ok_or_else(not_int)?;
                    let at = Some(t)
                        .filter(|&t| t > 0)
                        .and_then(|t| abs_deadline(&o, t, cx.now))
                        .ok_or_else(|| err("invalid expire time in 'getex' command"))?;
                    cx.set_ttl(&a[0], old.ttl, Some(at));
                } else {
                    return Err(err("syntax error"));
                }
            }
            Ok(bulk(v))
        }
        "APPEND" => {
            need(n == 2)?;
            let old = cx.load(&a[0])?;
            let mut v = match &old.ent {
                Ent::Str(v) => v.clone(),
                Ent::None => Vec::new(),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            v.extend_from_slice(&a[1]);
            let len = v.len();
            let ttl = old.ttl;
            cx.put_str(&a[0], v, &old, ttl);
            Ok(int(len as i64))
        }
        "STRLEN" => {
            need(n == 1)?;
            Ok(match cx.load(&a[0])?.ent {
                Ent::Str(v) => int(v.len() as i64),
                Ent::None => int(0),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            })
        }
        "GETRANGE" => {
            need(n == 3)?;
            let (s, e) = (parse_i64(&a[1]).ok_or_else(not_int)?, parse_i64(&a[2]).ok_or_else(not_int)?);
            let v = match cx.load(&a[0])?.ent {
                Ent::Str(v) => v,
                Ent::None => Vec::new(),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            // GETRANGE clamps BOTH ends to the string (a negative end past
            // the start becomes 0), unlike LRANGE.
            let len = v.len() as i64;
            if s < 0 && e < 0 && s > e {
                return Ok(bulk(Vec::new()));
            }
            let fix = |x: i64| if x < 0 { (len + x).max(0) } else { x };
            let (s, e) = (fix(s), fix(e).min(len - 1));
            Ok(bulk(if len == 0 || s > e { Vec::new() } else { v[s as usize..=e as usize].to_vec() }))
        }
        "SETRANGE" => {
            need(n == 3)?;
            let off = parse_i64(&a[1]).filter(|&o| (0..512 * 1024 * 1024).contains(&o)).ok_or_else(|| err("offset is out of range"))? as usize;
            let old = cx.load(&a[0])?;
            let mut v = match &old.ent {
                Ent::Str(v) => v.clone(),
                Ent::None => Vec::new(),
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            if a[2].is_empty() {
                return Ok(int(v.len() as i64));
            }
            if v.len() < off + a[2].len() {
                v.resize(off + a[2].len(), 0);
            }
            v[off..off + a[2].len()].copy_from_slice(&a[2]);
            let len = v.len();
            let ttl = old.ttl;
            cx.put_str(&a[0], v, &old, ttl);
            Ok(int(len as i64))
        }
        "INCR" | "DECR" | "INCRBY" | "DECRBY" => {
            let by = match name {
                "INCR" => {
                    need(n == 1)?;
                    1
                }
                "DECR" => {
                    need(n == 1)?;
                    -1
                }
                "INCRBY" => {
                    need(n == 2)?;
                    parse_i64(&a[1]).ok_or_else(not_int)?
                }
                _ => {
                    need(n == 2)?;
                    parse_i64(&a[1]).and_then(|v| v.checked_neg()).ok_or_else(not_int)?
                }
            };
            let old = cx.load(&a[0])?;
            let cur = match &old.ent {
                Ent::None => 0,
                Ent::Str(v) => parse_i64(v).ok_or_else(not_int)?,
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            let next = cur.checked_add(by).ok_or_else(|| err("increment or decrement would overflow"))?;
            cx.tx.put(kv(&a[0]), Value::Int(next));
            Ok(int(next))
        }
        "INCRBYFLOAT" => {
            need(n == 2)?;
            let by = parse_f64(&a[1]).ok_or_else(not_float)?;
            let old = cx.load(&a[0])?;
            let cur = match &old.ent {
                Ent::None => 0.0,
                Ent::Str(v) => parse_f64(v).ok_or_else(not_float)?,
                Ent::Coll(_) => return Err(err(WRONGTYPE)),
            };
            let next = cur + by;
            if !next.is_finite() {
                return Err(err("increment would produce NaN or Infinity"));
            }
            let s = fmt_f64(next).into_bytes();
            let ttl = old.ttl;
            cx.put_str(&a[0], s.clone(), &old, ttl);
            Ok(bulk(s))
        }
        "MSET" | "MSETNX" => {
            need(n >= 2 && n % 2 == 0)?;
            let mut olds = Vec::new();
            for p in a.chunks(2) {
                olds.push(cx.load(&p[0])?);
            }
            if name == "MSETNX" && olds.iter().any(|o| o.exists()) {
                return Ok(int(0));
            }
            for (p, old) in a.chunks(2).zip(olds) {
                cx.put_str(&p[0], p[1].clone(), &old, None);
            }
            Ok(if name == "MSET" { ok() } else { int(1) })
        }

        // ── generic ────────────────────────────────────────────────────
        "DEL" | "UNLINK" => {
            need(n >= 1)?;
            let mut c = 0;
            let mut seen = std::collections::HashSet::new();
            for k in a {
                if !seen.insert(k.clone()) {
                    continue;
                }
                let e = cx.load(k)?;
                if e.exists() {
                    cx.delete_all(k, &e);
                    c += 1;
                }
            }
            Ok(int(c))
        }
        "EXISTS" => {
            need(n >= 1)?;
            let mut c = 0;
            for k in a {
                if cx.load(k)?.exists() {
                    c += 1;
                }
            }
            Ok(int(c))
        }
        "TYPE" => {
            need(n == 1)?;
            Ok(Resp::Simple(
                match cx.load(&a[0])?.ent {
                    Ent::None => "none",
                    Ent::Str(_) => "string",
                    Ent::Coll(m) => m.type_name(),
                }
                .into(),
            ))
        }
        "RENAME" | "RENAMENX" => {
            need(n == 2)?;
            let src = cx.load(&a[0])?;
            if !src.exists() {
                return Err(err("no such key"));
            }
            if a[0] == a[1] {
                return Ok(if name == "RENAME" { ok() } else { int(0) });
            }
            let dst = cx.load(&a[1])?;
            if name == "RENAMENX" && dst.exists() {
                return Ok(int(0));
            }
            cx.delete_all(&a[1], &dst);
            match &src.ent {
                Ent::Str(v) => {
                    cx.tx.put(kv(&a[1]), Value::Bytes(v.clone()));
                }
                Ent::Coll(m) => {
                    for &sub in type_subs(m.ty) {
                        for (rest, v) in cx.elements(&a[0], sub) {
                            cx.tx.put(ekey(&a[1], sub, &rest), v);
                        }
                    }
                    cx.tx.put(kv(&a[1]), Value::Meta(m.encode()));
                }
                Ent::None => {}
            }
            let ttl = src.ttl;
            cx.delete_all(&a[0], &src);
            cx.set_ttl(&a[1], None, ttl);
            Ok(if name == "RENAME" { ok() } else { int(1) })
        }
        "EXPIRE" | "PEXPIRE" | "EXPIREAT" | "PEXPIREAT" => {
            need(n >= 2)?;
            let v = parse_i64(&a[1]).ok_or_else(not_int)?;
            let unit = match name {
                "EXPIRE" => "EX",
                "PEXPIRE" => "PX",
                "EXPIREAT" => "EXAT",
                _ => "PXAT",
            };
            let (mut nx, mut xx, mut gt, mut lt) = (false, false, false, false);
            for o in &a[2..] {
                match String::from_utf8_lossy(o).to_ascii_uppercase().as_str() {
                    "NX" => nx = true,
                    "XX" => xx = true,
                    "GT" => gt = true,
                    "LT" => lt = true,
                    _ => return Err(err(&format!("Unsupported option {}", String::from_utf8_lossy(o)))),
                }
            }
            if (nx && (xx || gt || lt)) || (gt && lt) {
                return Err(err("NX and XX, GT or LT options at the same time are not compatible"));
            }
            let e = cx.load(&a[0])?;
            if !e.exists() {
                return Ok(int(0));
            }
            // A deadline at or before now deletes the key (Redis).
            let at = abs_deadline_signed(unit, v, cx.now).ok_or_else(|| err(&format!("invalid expire time in '{}' command", name.to_lowercase())))?;
            let cur = e.ttl;
            let allowed = match cur {
                None => !xx && !gt, // no TTL counts as infinite
                Some(c) => !nx && !(gt && at <= c) && !(lt && at >= c),
            };
            if !allowed {
                return Ok(int(0));
            }
            if at <= cx.now {
                cx.delete_all(&a[0], &e);
            } else {
                cx.set_ttl(&a[0], cur, Some(at));
            }
            Ok(int(1))
        }
        "TTL" | "PTTL" | "EXPIRETIME" | "PEXPIRETIME" => {
            need(n == 1)?;
            let e = cx.load(&a[0])?;
            if !e.exists() {
                return Ok(int(-2));
            }
            Ok(int(match e.ttl {
                None => -1,
                Some(at) => match name {
                    "TTL" => ((at.saturating_sub(cx.now)) + 500) as i64 / 1000,
                    "PTTL" => at.saturating_sub(cx.now) as i64,
                    "EXPIRETIME" => (at / 1000) as i64,
                    _ => at as i64,
                },
            }))
        }
        "PERSIST" => {
            need(n == 1)?;
            let e = cx.load(&a[0])?;
            if !e.exists() || e.ttl.is_none() {
                return Ok(int(0));
            }
            cx.set_ttl(&a[0], e.ttl, None);
            Ok(int(1))
        }

        // ── hashes ─────────────────────────────────────────────────────
        "HSET" | "HMSET" => {
            need(n >= 3 && n % 2 == 1)?;
            let (mut m, ttl) = cx.coll_or_new(&a[0], T_HASH)?;
            let mut added = 0;
            for p in a[1..].chunks(2) {
                let k = ekey(&a[0], SUB_HASH, &p[0]);
                if cx.tx.get(&k).is_none() {
                    added += 1;
                    m.len += 1;
                }
                cx.tx.put(k, Value::Bytes(p[1].clone()));
            }
            cx.put_meta(&a[0], m, ttl);
            Ok(if name == "HSET" { int(added) } else { ok() })
        }
        "HSETNX" => {
            need(n == 3)?;
            let (mut m, ttl) = cx.coll_or_new(&a[0], T_HASH)?;
            let k = ekey(&a[0], SUB_HASH, &a[1]);
            if cx.tx.get(&k).is_some() {
                return Ok(int(0));
            }
            cx.tx.put(k, Value::Bytes(a[2].clone()));
            m.len += 1;
            cx.put_meta(&a[0], m, ttl);
            Ok(int(1))
        }
        "HGET" => {
            need(n == 2)?;
            if cx.coll(&a[0], T_HASH)?.is_none() {
                return Ok(Resp::Nil);
            }
            Ok(cx.tx.get(&ekey(&a[0], SUB_HASH, &a[1])).map_or(Resp::Nil, |v| bulk(val_bytes(&v))))
        }
        "HMGET" => {
            need(n >= 2)?;
            let present = cx.coll(&a[0], T_HASH)?.is_some();
            Ok(arr(a[1..]
                .iter()
                .map(|f| {
                    if !present {
                        return Resp::Nil;
                    }
                    cx.tx.get(&ekey(&a[0], SUB_HASH, f)).map_or(Resp::Nil, |v| bulk(val_bytes(&v)))
                })
                .collect()))
        }
        "HDEL" => {
            need(n >= 2)?;
            let Some((mut m, ttl)) = cx.coll(&a[0], T_HASH)? else { return Ok(int(0)) };
            let mut c = 0;
            for f in &a[1..] {
                let k = ekey(&a[0], SUB_HASH, f);
                if cx.tx.get(&k).is_some() {
                    cx.tx.delete(k);
                    m.len -= 1;
                    c += 1;
                }
            }
            if c > 0 {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(int(c))
        }
        "HEXISTS" => {
            need(n == 2)?;
            if cx.coll(&a[0], T_HASH)?.is_none() {
                return Ok(int(0));
            }
            Ok(int(cx.tx.get(&ekey(&a[0], SUB_HASH, &a[1])).is_some() as i64))
        }
        "HLEN" => {
            need(n == 1)?;
            Ok(int(cx.coll(&a[0], T_HASH)?.map_or(0, |(m, _)| m.len as i64)))
        }
        "HSTRLEN" => {
            need(n == 2)?;
            if cx.coll(&a[0], T_HASH)?.is_none() {
                return Ok(int(0));
            }
            Ok(int(cx.tx.get(&ekey(&a[0], SUB_HASH, &a[1])).map_or(0, |v| val_bytes(&v).len() as i64)))
        }
        "HGETALL" | "HKEYS" | "HVALS" => {
            need(n == 1)?;
            if cx.coll(&a[0], T_HASH)?.is_none() {
                return Ok(arr(vec![]));
            }
            let mut out = Vec::new();
            for (f, v) in cx.elements(&a[0], SUB_HASH) {
                if name != "HVALS" {
                    out.push(bulk(f));
                }
                if name != "HKEYS" {
                    out.push(bulk(val_bytes(&v)));
                }
            }
            Ok(arr(out))
        }
        "HINCRBY" | "HINCRBYFLOAT" => {
            need(n == 3)?;
            let (mut m, ttl) = cx.coll_or_new(&a[0], T_HASH)?;
            let k = ekey(&a[0], SUB_HASH, &a[1]);
            let cur = cx.tx.get(&k);
            if cur.is_none() {
                m.len += 1;
            }
            let cur_b = cur.as_ref().map(val_bytes);
            let out = if name == "HINCRBY" {
                let by = parse_i64(&a[2]).ok_or_else(not_int)?;
                let c = match &cur_b {
                    None => 0,
                    Some(b) => parse_i64(b).ok_or_else(|| err("hash value is not an integer"))?,
                };
                let nv = c.checked_add(by).ok_or_else(|| err("increment or decrement would overflow"))?;
                cx.tx.put(k, Value::Bytes(nv.to_string().into_bytes()));
                int(nv)
            } else {
                let by = parse_f64(&a[2]).ok_or_else(not_float)?;
                let c = match &cur_b {
                    None => 0.0,
                    Some(b) => parse_f64(b).ok_or_else(|| err("hash value is not a float"))?,
                };
                let nv = c + by;
                if !nv.is_finite() {
                    return Err(err("increment would produce NaN or Infinity"));
                }
                let s = fmt_f64(nv).into_bytes();
                cx.tx.put(k, Value::Bytes(s.clone()));
                bulk(s)
            };
            cx.put_meta(&a[0], m, ttl);
            Ok(out)
        }

        // ── lists ──────────────────────────────────────────────────────
        "LPUSH" | "RPUSH" | "LPUSHX" | "RPUSHX" => {
            need(n >= 2)?;
            let existing = cx.coll(&a[0], T_LIST)?;
            if existing.is_none() && name.ends_with('X') {
                return Ok(int(0));
            }
            let (mut m, ttl) = existing.unwrap_or((Meta::new(T_LIST), None));
            let left = name.starts_with('L');
            for v in &a[1..] {
                let seq = if left {
                    m.head -= 1;
                    m.head
                } else {
                    m.tail += 1;
                    m.tail - 1
                };
                cx.tx.put(ekey(&a[0], SUB_LIST, &seq_bytes(seq)), Value::Bytes(v.clone()));
                m.len += 1;
            }
            let len = m.len;
            cx.put_meta(&a[0], m, ttl);
            Ok(int(len as i64))
        }
        "LPOP" | "RPOP" => {
            need(n == 1 || n == 2)?;
            let count = match a.get(1) {
                Some(c) => Some(parse_i64(c).filter(|&c| c >= 0).ok_or_else(|| err("value is out of range, must be positive"))? as u64),
                None => None,
            };
            let Some((mut m, ttl)) = cx.coll(&a[0], T_LIST)? else {
                return Ok(if count.is_some() { Resp::Nil } else { Resp::Nil });
            };
            let take = count.unwrap_or(1).min(m.len);
            let mut out = Vec::new();
            for _ in 0..take {
                let seq = if name == "LPOP" {
                    m.head += 1;
                    m.head - 1
                } else {
                    m.tail -= 1;
                    m.tail
                };
                let k = ekey(&a[0], SUB_LIST, &seq_bytes(seq));
                let v = cx.tx.get(&k).map(|v| val_bytes(&v)).unwrap_or_default();
                cx.tx.delete(k);
                out.push(bulk(v));
                m.len -= 1;
            }
            cx.put_meta(&a[0], m, ttl);
            Ok(match count {
                None => out.pop().unwrap_or(Resp::Nil),
                Some(_) => arr(out),
            })
        }
        "LLEN" => {
            need(n == 1)?;
            Ok(int(cx.coll(&a[0], T_LIST)?.map_or(0, |(m, _)| m.len as i64)))
        }
        "LINDEX" => {
            need(n == 2)?;
            let i = parse_i64(&a[1]).ok_or_else(not_int)?;
            let Some((m, _)) = cx.coll(&a[0], T_LIST)? else { return Ok(Resp::Nil) };
            let len = m.len as i64;
            let i = if i < 0 { len + i } else { i };
            if i < 0 || i >= len {
                return Ok(Resp::Nil);
            }
            Ok(cx.tx.get(&ekey(&a[0], SUB_LIST, &seq_bytes(m.head + i))).map_or(Resp::Nil, |v| bulk(val_bytes(&v))))
        }
        "LSET" => {
            need(n == 3)?;
            let i = parse_i64(&a[1]).ok_or_else(not_int)?;
            let Some((m, ttl)) = cx.coll(&a[0], T_LIST)? else { return Err(err("no such key")) };
            let len = m.len as i64;
            let i = if i < 0 { len + i } else { i };
            if i < 0 || i >= len {
                return Err(err("index out of range"));
            }
            cx.tx.put(ekey(&a[0], SUB_LIST, &seq_bytes(m.head + i)), Value::Bytes(a[2].clone()));
            cx.put_meta(&a[0], m, ttl);
            Ok(ok())
        }
        "LRANGE" => {
            need(n == 3)?;
            let (s, e) = (parse_i64(&a[1]).ok_or_else(not_int)?, parse_i64(&a[2]).ok_or_else(not_int)?);
            let Some((m, _)) = cx.coll(&a[0], T_LIST)? else { return Ok(arr(vec![])) };
            let Some((s, e)) = norm_range(s, e, m.len as i64) else { return Ok(arr(vec![])) };
            let start = ekey(&a[0], SUB_LIST, &seq_bytes(m.head + s));
            let end = ekey(&a[0], SUB_LIST, &seq_bytes(m.head + e + 1));
            let mut out = Vec::new();
            cx.each(&start.clone(), start, end, false, |_, v| {
                out.push(bulk(val_bytes(v)));
                true
            });
            Ok(arr(out))
        }
        "LTRIM" => {
            need(n == 3)?;
            let (s, e) = (parse_i64(&a[1]).ok_or_else(not_int)?, parse_i64(&a[2]).ok_or_else(not_int)?);
            let Some((mut m, ttl)) = cx.coll(&a[0], T_LIST)? else { return Ok(ok()) };
            let len = m.len as i64;
            let keep = norm_range(s, e, len);
            let (ks, ke) = keep.unwrap_or((len, len - 1));
            for i in (0..ks).chain((ke + 1).max(0)..len) {
                cx.tx.delete(ekey(&a[0], SUB_LIST, &seq_bytes(m.head + i)));
            }
            let (nh, nt) = (m.head + ks, m.head + ke + 1);
            m.head = nh;
            m.tail = nt.max(nh);
            m.len = (m.tail - m.head) as u64;
            cx.put_meta(&a[0], m, ttl);
            Ok(ok())
        }
        "LREM" | "LINSERT" => {
            let Some((m, ttl)) = (if name == "LREM" {
                need(n == 3)?;
                cx.coll(&a[0], T_LIST)?
            } else {
                need(n == 4)?;
                cx.coll(&a[0], T_LIST)?
            }) else {
                return Ok(int(if name == "LREM" { 0 } else { 0 }));
            };
            let items: Vec<Vec<u8>> = cx.elements(&a[0], SUB_LIST).into_iter().map(|(_, v)| val_bytes(&v)).collect();
            let (new, reply) = if name == "LREM" {
                let count = parse_i64(&a[1]).ok_or_else(not_int)?;
                let target = &a[2];
                let mut removed = 0i64;
                let limit = if count == 0 { i64::MAX } else { count.abs() };
                let mut keep = vec![true; items.len()];
                let idxs: Vec<usize> = if count < 0 { (0..items.len()).rev().collect() } else { (0..items.len()).collect() };
                for i in idxs {
                    if removed < limit && &items[i] == target {
                        keep[i] = false;
                        removed += 1;
                    }
                }
                (items.iter().zip(keep).filter(|(_, k)| *k).map(|(v, _)| v.clone()).collect::<Vec<_>>(), int(removed))
            } else {
                let before = match String::from_utf8_lossy(&a[1]).to_ascii_uppercase().as_str() {
                    "BEFORE" => true,
                    "AFTER" => false,
                    _ => return Err(err("syntax error")),
                };
                match items.iter().position(|v| v == &a[2]) {
                    None => return Ok(int(-1)),
                    Some(p) => {
                        let mut v = items.clone();
                        v.insert(if before { p } else { p + 1 }, a[3].clone());
                        let len = v.len() as i64;
                        (v, int(len))
                    }
                }
            };
            // Rewrite with contiguous sequence numbers.
            for i in 0..m.len as i64 {
                cx.tx.delete(ekey(&a[0], SUB_LIST, &seq_bytes(m.head + i)));
            }
            let mut nm = m;
            nm.head = 0;
            nm.tail = new.len() as i64;
            nm.len = new.len() as u64;
            for (i, v) in new.into_iter().enumerate() {
                cx.tx.put(ekey(&a[0], SUB_LIST, &seq_bytes(i as i64)), Value::Bytes(v));
            }
            cx.put_meta(&a[0], nm, ttl);
            Ok(reply)
        }
        "LPOS" => {
            need(n >= 2)?;
            let (mut rank, mut count, mut maxlen) = (1i64, None::<i64>, 0i64);
            let mut i = 2;
            while i < n {
                let opt = String::from_utf8_lossy(&a[i]).to_ascii_uppercase();
                let v = a.get(i + 1).ok_or_else(|| err("syntax error"))?;
                let v = parse_i64(v).ok_or_else(not_int)?;
                match opt.as_str() {
                    "RANK" if v == 0 => return Err(err("RANK can't be zero: use 1 to start from the first match, 2 from the second ... or use negative to start from the end of the list")),
                    "RANK" if v == i64::MIN => return Err(err("value is out of range, value must between -9223372036854775807 and 9223372036854775807")),
                    "RANK" => rank = v,
                    "COUNT" if v < 0 => return Err(err("COUNT can't be negative")),
                    "COUNT" => count = Some(v),
                    "MAXLEN" if v < 0 => return Err(err("MAXLEN can't be negative")),
                    "MAXLEN" => maxlen = v,
                    _ => return Err(err("syntax error")),
                }
                i += 2;
            }
            let Some((m, _)) = cx.coll(&a[0], T_LIST)? else {
                return Ok(if count.is_some() { arr(vec![]) } else { Resp::Nil });
            };
            let len = m.len as i64;
            let mut elems = cx.elements(&a[0], SUB_LIST);
            if rank < 0 {
                elems.reverse();
            }
            let want = match count {
                Some(0) => usize::MAX,
                Some(c) => c as usize,
                None => 1,
            };
            let (mut skip, mut found) = (rank.unsigned_abs() - 1, Vec::new());
            for (idx, (_, v)) in elems.iter().enumerate() {
                if maxlen != 0 && idx as i64 >= maxlen {
                    break;
                }
                if val_bytes(v) == a[1] {
                    if skip > 0 {
                        skip -= 1;
                        continue;
                    }
                    found.push(if rank < 0 { len - 1 - idx as i64 } else { idx as i64 });
                    if found.len() >= want {
                        break;
                    }
                }
            }
            Ok(match count {
                Some(_) => arr(found.into_iter().map(int).collect()),
                None => found.first().map_or(Resp::Nil, |&p| int(p)),
            })
        }
        "RPOPLPUSH" | "LMOVE" => {
            let (from_left, to_left) = if name == "RPOPLPUSH" {
                need(n == 2)?;
                (false, true)
            } else {
                need(n == 4)?;
                let side = |x: &[u8]| match String::from_utf8_lossy(x).to_ascii_uppercase().as_str() {
                    "LEFT" => Ok(true),
                    "RIGHT" => Ok(false),
                    _ => Err(err("syntax error")),
                };
                (side(&a[2])?, side(&a[3])?)
            };
            // Redis order: a missing source is a nil reply (whatever the
            // destination holds); otherwise the destination must be a list
            // (or absent) BEFORE anything is popped.
            if cx.coll(&a[0], T_LIST)?.is_none() {
                return Ok(Resp::Nil);
            }
            if a[0] != a[1] {
                cx.coll(&a[1], T_LIST)?;
            }
            let popped = exec_inner(cx, if from_left { "LPOP" } else { "RPOP" }, &a[..1])?;
            let Resp::Bulk(v) = popped else { return Ok(Resp::Nil) };
            exec_inner(cx, if to_left { "LPUSH" } else { "RPUSH" }, &[a[1].clone(), v.clone()])?;
            Ok(bulk(v))
        }

        // ── sets ───────────────────────────────────────────────────────
        "SADD" => {
            need(n >= 2)?;
            let (mut m, ttl) = cx.coll_or_new(&a[0], T_SET)?;
            let mut added = 0;
            for v in &a[1..] {
                if set_add(cx, &a[0], &mut m, v) {
                    added += 1;
                }
            }
            if added > 0 {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(int(added))
        }
        "SREM" => {
            need(n >= 2)?;
            let Some((mut m, ttl)) = cx.coll(&a[0], T_SET)? else { return Ok(int(0)) };
            let mut c = 0;
            for v in &a[1..] {
                if set_rem(cx, &a[0], &mut m, v) {
                    c += 1;
                }
            }
            if c > 0 {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(int(c))
        }
        "SISMEMBER" => {
            need(n == 2)?;
            if cx.coll(&a[0], T_SET)?.is_none() {
                return Ok(int(0));
            }
            Ok(int(cx.tx.get(&ekey(&a[0], SUB_SET, &a[1])).is_some() as i64))
        }
        "SMISMEMBER" => {
            need(n >= 2)?;
            let present = cx.coll(&a[0], T_SET)?.is_some();
            Ok(arr(a[1..].iter().map(|v| int((present && cx.tx.get(&ekey(&a[0], SUB_SET, v)).is_some()) as i64)).collect()))
        }
        "SCARD" => {
            need(n == 1)?;
            Ok(int(cx.coll(&a[0], T_SET)?.map_or(0, |(m, _)| m.len as i64)))
        }
        "SMEMBERS" => {
            need(n == 1)?;
            Ok(arr(set_members(cx, &a[0])?.into_iter().map(bulk).collect()))
        }
        "SPOP" | "SRANDMEMBER" => {
            need(n == 1 || n == 2)?;
            let count = match a.get(1) {
                Some(c) => Some(parse_i64(c).ok_or_else(not_int)?),
                None => None,
            };
            if name == "SPOP" && count.is_some_and(|c| c < 0) {
                return Err(err("value is out of range, must be positive"));
            }
            // Redis rejects i64::MIN (it cannot be negated); a huge negative
            // count would otherwise build an unbounded reply and abort.
            if count.is_some_and(|c| c == i64::MIN || c < -(MAX_RANDOM_PICKS as i64)) {
                return Err(err("value is out of range, value must between -9223372036854775807 and 9223372036854775807"));
            }
            let Some((mut m, ttl)) = cx.coll(&a[0], T_SET)? else {
                return Ok(if count.is_some() { arr(vec![]) } else { Resp::Nil });
            };
            let want = count.unwrap_or(1);
            let mut picked: Vec<Vec<u8>> = Vec::new();
            if name == "SRANDMEMBER" && want < 0 {
                // Negative count: independent picks, may repeat.
                for _ in 0..want.unsigned_abs() {
                    let p = rand() % m.len;
                    picked.push(set_at(cx, &a[0], p));
                }
            } else if name == "SRANDMEMBER" {
                let want = (want as u64).min(m.len);
                if want * 2 >= m.len {
                    let mut all = set_members(cx, &a[0])?;
                    while all.len() as u64 > want {
                        let i = (rand() % all.len() as u64) as usize;
                        all.swap_remove(i);
                    }
                    picked = all;
                } else {
                    let mut seen = std::collections::HashSet::new();
                    while (picked.len() as u64) < want {
                        let p = rand() % m.len;
                        if seen.insert(p) {
                            picked.push(set_at(cx, &a[0], p));
                        }
                    }
                }
            } else {
                for _ in 0..(want as u64).min(m.len) {
                    let p = rand() % m.len;
                    let mbr = set_at(cx, &a[0], p);
                    set_rem(cx, &a[0], &mut m, &mbr);
                    picked.push(mbr);
                }
                if !picked.is_empty() {
                    cx.put_meta(&a[0], m, ttl);
                }
            }
            Ok(match count {
                None => picked.pop().map_or(Resp::Nil, bulk),
                Some(_) => arr(picked.into_iter().map(bulk).collect()),
            })
        }
        "SINTER" | "SUNION" | "SDIFF" | "SINTERSTORE" | "SUNIONSTORE" | "SDIFFSTORE" => {
            let store = name.ends_with("STORE");
            need(n >= if store { 2 } else { 1 })?;
            let (dst, srcs) = if store { (Some(&a[0]), &a[1..]) } else { (None, a) };
            let mut sets: Vec<BTreeSet<Vec<u8>>> = Vec::new();
            for k in srcs {
                sets.push(set_members(cx, k)?.into_iter().collect());
            }
            let mut res: BTreeSet<Vec<u8>> = sets.first().cloned().unwrap_or_default();
            for s in sets.iter().skip(1) {
                res = match &name[..5] {
                    "SINTE" => res.intersection(s).cloned().collect(),
                    "SUNIO" => res.union(s).cloned().collect(),
                    _ => res.difference(s).cloned().collect(),
                };
            }
            match dst {
                None => Ok(arr(res.into_iter().map(bulk).collect())),
                Some(d) => {
                    let old = cx.load(d)?;
                    cx.delete_all(d, &old);
                    let mut m = Meta::new(T_SET);
                    for v in &res {
                        set_add(cx, d, &mut m, v);
                    }
                    cx.put_meta(d, m, None);
                    Ok(int(res.len() as i64))
                }
            }
        }
        "SMOVE" => {
            need(n == 3)?;
            let Some((mut sm, sttl)) = cx.coll(&a[0], T_SET)? else { return Ok(int(0)) };
            let (mut dm, dttl) = cx.coll_or_new(&a[1], T_SET)?;
            if cx.tx.get(&ekey(&a[0], SUB_SET, &a[2])).is_none() {
                return Ok(int(0));
            }
            if a[0] == a[1] {
                return Ok(int(1));
            }
            set_rem(cx, &a[0], &mut sm, &a[2]);
            cx.put_meta(&a[0], sm, sttl);
            if set_add(cx, &a[1], &mut dm, &a[2]) {
                cx.put_meta(&a[1], dm, dttl);
            }
            Ok(int(1))
        }

        // ── sorted sets ────────────────────────────────────────────────
        "ZADD" => {
            need(n >= 3)?;
            let (mut nx, mut xx, mut gt, mut lt, mut ch, mut incr) = (false, false, false, false, false, false);
            let mut i = 1;
            while i < n {
                match String::from_utf8_lossy(&a[i]).to_ascii_uppercase().as_str() {
                    "NX" => nx = true,
                    "XX" => xx = true,
                    "GT" => gt = true,
                    "LT" => lt = true,
                    "CH" => ch = true,
                    "INCR" => incr = true,
                    _ => break,
                }
                i += 1;
            }
            let pairs = &a[i..];
            if pairs.is_empty() || pairs.len() % 2 != 0 {
                return Err(err("syntax error"));
            }
            if nx && xx {
                return Err(err("XX and NX options at the same time are not compatible"));
            }
            if (gt && lt) || (nx && (gt || lt)) {
                return Err(err("GT, LT, and/or NX options at the same time are not compatible"));
            }
            if incr && pairs.len() != 2 {
                return Err(err("INCR option supports a single increment-element pair"));
            }
            let mut parsed = Vec::new();
            for p in pairs.chunks(2) {
                parsed.push((parse_f64(&p[0]).ok_or_else(not_float)?, p[1].clone()));
            }
            let (mut m, ttl) = cx.coll_or_new(&a[0], T_ZSET)?;
            let (mut added, mut changed) = (0i64, 0i64);
            let mut incr_out = Resp::Nil;
            for (score, member) in parsed {
                let mk = ekey(&a[0], SUB_ZMEM, &member);
                let cur = match cx.tx.get(&mk) {
                    Some(Value::Float(f)) => Some(f),
                    _ => None,
                };
                if (nx && cur.is_some()) || (xx && cur.is_none()) {
                    continue;
                }
                let new = if incr { cur.unwrap_or(0.0) + score } else { score };
                if new.is_nan() {
                    return Err(err("resulting score is not a number (NaN)"));
                }
                if let Some(c) = cur {
                    if (gt && new <= c) || (lt && new >= c) {
                        continue;
                    }
                    if new == c {
                        if incr {
                            incr_out = bulk(fmt_f64(new).into_bytes());
                        }
                        continue;
                    }
                    let mut old_sk = ebase(&a[0], SUB_ZSCORE);
                    old_sk.extend_from_slice(&score_bytes(c));
                    old_sk.extend_from_slice(&member);
                    cx.tx.delete(old_sk);
                    changed += 1;
                } else {
                    added += 1;
                    m.len += 1;
                }
                cx.tx.put(mk, Value::Float(new));
                let mut sk = ebase(&a[0], SUB_ZSCORE);
                sk.extend_from_slice(&score_bytes(new));
                sk.extend_from_slice(&member);
                cx.tx.put(sk, Value::Int(1));
                incr_out = bulk(fmt_f64(new).into_bytes());
            }
            if added + changed > 0 {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(if incr { incr_out } else { int(if ch { added + changed } else { added }) })
        }
        "ZINCRBY" => {
            need(n == 3)?;
            exec_inner(cx, "ZADD", &[a[0].clone(), b"INCR".to_vec(), a[1].clone(), a[2].clone()])
        }
        "ZREM" => {
            need(n >= 2)?;
            let Some((mut m, ttl)) = cx.coll(&a[0], T_ZSET)? else { return Ok(int(0)) };
            let mut c = 0;
            for member in &a[1..] {
                if zremove(cx, &a[0], member) {
                    m.len -= 1;
                    c += 1;
                }
            }
            if c > 0 {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(int(c))
        }
        "ZSCORE" => {
            need(n == 2)?;
            if cx.coll(&a[0], T_ZSET)?.is_none() {
                return Ok(Resp::Nil);
            }
            Ok(zscore(cx, &a[0], &a[1]).map_or(Resp::Nil, |f| bulk(fmt_f64(f).into_bytes())))
        }
        "ZMSCORE" => {
            need(n >= 2)?;
            let present = cx.coll(&a[0], T_ZSET)?.is_some();
            Ok(arr(a[1..]
                .iter()
                .map(|mbr| if present { zscore(cx, &a[0], mbr).map_or(Resp::Nil, |f| bulk(fmt_f64(f).into_bytes())) } else { Resp::Nil })
                .collect()))
        }
        "ZCARD" => {
            need(n == 1)?;
            Ok(int(cx.coll(&a[0], T_ZSET)?.map_or(0, |(m, _)| m.len as i64)))
        }
        "ZCOUNT" => {
            need(n == 3)?;
            let (lo, hi) = (score_bound(&a[1])?, score_bound(&a[2])?);
            if cx.coll(&a[0], T_ZSET)?.is_none() {
                return Ok(int(0));
            }
            Ok(int(zby_score(cx, &a[0], lo, hi, false, 0, usize::MAX).len() as i64))
        }
        "ZRANK" | "ZREVRANK" => {
            need(n == 2 || n == 3)?;
            let with = a.get(2).is_some_and(|o| o.eq_ignore_ascii_case(b"WITHSCORE"));
            let Some((m, _)) = cx.coll(&a[0], T_ZSET)? else { return Ok(Resp::Nil) };
            let Some(score) = zscore(cx, &a[0], &a[1]) else { return Ok(Resp::Nil) };
            let (s, _) = erange(&a[0], SUB_ZSCORE);
            let mut target = s.clone();
            target.extend_from_slice(&score_bytes(score));
            target.extend_from_slice(&a[1]);
            // rank = number of entries strictly before the member's score key
            let mut rank = 0i64;
            cx.each(&s.clone(), s, target, false, |_, _| {
                rank += 1;
                true
            });
            let r = if name == "ZRANK" { rank } else { m.len as i64 - 1 - rank };
            Ok(if with { arr(vec![int(r), bulk(fmt_f64(score).into_bytes())]) } else { int(r) })
        }
        "ZRANGE" | "ZREVRANGE" | "ZRANGEBYSCORE" | "ZREVRANGEBYSCORE" | "ZRANGEBYLEX" | "ZREVRANGEBYLEX" => {
            zrange(cx, name, a)
        }
        "ZLEXCOUNT" => {
            need(n == 3)?;
            let (lo, hi) = (lex_bound(&a[1])?, lex_bound(&a[2])?);
            if cx.coll(&a[0], T_ZSET)?.is_none() {
                return Ok(int(0));
            }
            Ok(int(zby_lex(cx, &a[0], &lo, &hi, false, 0, usize::MAX).len() as i64))
        }
        "ZREMRANGEBYSCORE" | "ZREMRANGEBYRANK" | "ZREMRANGEBYLEX" => {
            need(n == 3)?;
            let lex = if name == "ZREMRANGEBYLEX" { Some((lex_bound(&a[1])?, lex_bound(&a[2])?)) } else { None };
            let Some((mut m, ttl)) = cx.coll(&a[0], T_ZSET)? else { return Ok(int(0)) };
            let victims: Vec<(f64, Vec<u8>)> = if let Some((lo, hi)) = lex {
                zby_lex(cx, &a[0], &lo, &hi, false, 0, usize::MAX)
            } else if name == "ZREMRANGEBYSCORE" {
                zby_score(cx, &a[0], score_bound(&a[1])?, score_bound(&a[2])?, false, 0, usize::MAX)
            } else {
                let (s, e) = (parse_i64(&a[1]).ok_or_else(not_int)?, parse_i64(&a[2]).ok_or_else(not_int)?);
                match norm_range(s, e, m.len as i64) {
                    None => Vec::new(),
                    Some((s, e)) => zby_rank(cx, &a[0], s as usize, (e - s + 1) as usize, false),
                }
            };
            for (_, mbr) in &victims {
                zremove(cx, &a[0], mbr);
                m.len -= 1;
            }
            if !victims.is_empty() {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(int(victims.len() as i64))
        }
        "ZPOPMIN" | "ZPOPMAX" => {
            need(n == 1 || n == 2)?;
            let count = match a.get(1) {
                Some(c) => parse_i64(c).filter(|&c| c >= 0).ok_or_else(|| err("value is out of range, must be positive"))? as usize,
                None => 1,
            };
            let Some((mut m, ttl)) = cx.coll(&a[0], T_ZSET)? else { return Ok(arr(vec![])) };
            let got = zby_rank(cx, &a[0], 0, count, name == "ZPOPMAX");
            let mut out = Vec::new();
            for (s, mbr) in got {
                zremove(cx, &a[0], &mbr);
                m.len -= 1;
                out.push(bulk(mbr));
                out.push(bulk(fmt_f64(s).into_bytes()));
            }
            if !out.is_empty() {
                cx.put_meta(&a[0], m, ttl);
            }
            Ok(arr(out))
        }
        other => Err(err(&format!("unknown command '{other}'"))),
    }
}

/// The smallest byte string greater than every string starting with `b`
/// (increment as a big-endian number); `None` if `b` is all 0xff.
fn next_prefix(b: &[u8]) -> Option<Vec<u8>> {
    let mut v = b.to_vec();
    while let Some(last) = v.pop() {
        if last != 0xff {
            v.push(last + 1);
            return Some(v);
        }
    }
    None
}

/// EX/PX/EXAT/PXAT value → absolute unix-ms deadline (positive values).
fn abs_deadline(unit: &str, v: i64, now: u64) -> Option<u64> {
    if v <= 0 {
        return None;
    }
    abs_deadline_signed(unit, v, now)
}

/// Like `abs_deadline` but allows non-positive values (EXPIRE with a past
/// deadline deletes the key). Clamps at 0.
fn abs_deadline_signed(unit: &str, v: i64, now: u64) -> Option<u64> {
    // Overflow in either direction is an error, as in Redis (`EXPIRE k
    // -9223372036854775808` must not silently delete the key).
    let ms = match unit {
        "EX" | "EXAT" => v.checked_mul(1000)?,
        "PX" | "PXAT" => v,
        _ => return None,
    };
    let ms = if matches!(unit, "EX" | "PX") { ms.checked_add(i64::try_from(now).ok()?)? } else { ms };
    Some(ms.max(0) as u64)
}

/// Most picks one `SRANDMEMBER key -count` may return.
const MAX_RANDOM_PICKS: u64 = 1 << 24;

fn set_add(cx: &mut Ctx, key: &[u8], m: &mut Meta, member: &[u8]) -> bool {
    let mk = ekey(key, SUB_SET, member);
    if cx.tx.get(&mk).is_some() {
        return false;
    }
    let pos = m.len as i64;
    cx.tx.put(mk, Value::Int(pos));
    cx.tx.put(ekey(key, SUB_SETPOS, &seq_bytes(pos)), Value::Bytes(member.to_vec()));
    m.len += 1;
    true
}

fn set_rem(cx: &mut Ctx, key: &[u8], m: &mut Meta, member: &[u8]) -> bool {
    let mk = ekey(key, SUB_SET, member);
    let Some(Value::Int(pos)) = cx.tx.get(&mk) else { return false };
    let last = m.len as i64 - 1;
    if pos != last {
        // Move the last member into the hole.
        if let Some(Value::Bytes(lm)) = cx.tx.get(&ekey(key, SUB_SETPOS, &seq_bytes(last))) {
            cx.tx.put(ekey(key, SUB_SETPOS, &seq_bytes(pos)), Value::Bytes(lm.clone()));
            cx.tx.put(ekey(key, SUB_SET, &lm), Value::Int(pos));
        }
    }
    cx.tx.delete(ekey(key, SUB_SETPOS, &seq_bytes(last)));
    cx.tx.delete(mk);
    m.len -= 1;
    true
}

fn set_at(cx: &mut Ctx, key: &[u8], pos: u64) -> Vec<u8> {
    match cx.tx.get(&ekey(key, SUB_SETPOS, &seq_bytes(pos as i64))) {
        Some(Value::Bytes(b)) => b,
        _ => Vec::new(),
    }
}

fn set_members(cx: &mut Ctx, key: &[u8]) -> R<Vec<Vec<u8>>> {
    if cx.coll(key, T_SET)?.is_none() {
        return Ok(Vec::new());
    }
    Ok(cx.elements(key, SUB_SET).into_iter().map(|(m, _)| m).collect())
}

fn zscore(cx: &mut Ctx, key: &[u8], member: &[u8]) -> Option<f64> {
    match cx.tx.get(&ekey(key, SUB_ZMEM, member)) {
        Some(Value::Float(f)) => Some(f),
        _ => None,
    }
}

/// Remove a member from both zset indexes; true if it existed.
fn zremove(cx: &mut Ctx, key: &[u8], member: &[u8]) -> bool {
    let Some(score) = zscore(cx, key, member) else { return false };
    cx.tx.delete(ekey(key, SUB_ZMEM, member));
    let mut sk = ebase(key, SUB_ZSCORE);
    sk.extend_from_slice(&score_bytes(score));
    sk.extend_from_slice(member);
    cx.tx.delete(sk);
    true
}

#[derive(Clone, Copy)]
struct Bound {
    v: f64,
    excl: bool,
}

fn score_bound(a: &[u8]) -> R<Bound> {
    let (excl, rest) = match a.first() {
        Some(b'(') => (true, &a[1..]),
        _ => (false, a),
    };
    let v = parse_f64(rest).ok_or_else(|| err("min or max is not a float"))?;
    Ok(Bound { v, excl })
}

fn in_bounds(s: f64, lo: Bound, hi: Bound) -> (bool, bool) {
    let above_lo = if lo.excl { s > lo.v } else { s >= lo.v };
    let below_hi = if hi.excl { s < hi.v } else { s <= hi.v };
    (above_lo, below_hi)
}

/// Members with lo <= score <= hi, ascending (or descending), after skipping
/// `offset`, at most `limit`.
fn zby_score(cx: &mut Ctx, key: &[u8], lo: Bound, hi: Bound, rev: bool, offset: usize, limit: usize) -> Vec<(f64, Vec<u8>)> {
    let (s, e) = erange(key, SUB_ZSCORE);
    let plen = s.len();
    let mut out = Vec::new();
    if limit == 0 {
        return out;
    }
    let mut skipped = 0;
    let (start, end) = if rev {
        // Upper bound: just past every member whose score is exactly hi.
        let end = match next_prefix(&score_bytes(hi.v)) {
            Some(nx) => {
                let mut k = s.clone();
                k.extend_from_slice(&nx);
                k
            }
            None => e.clone(),
        };
        (s.clone(), end)
    } else {
        let mut lo_key = s.clone();
        lo_key.extend_from_slice(&score_bytes(lo.v));
        (lo_key, e.clone())
    };
    let hint = s.clone();
    cx.each(&hint, start, end, rev, |k, _| {
        let sc = score_from(&k[plen..plen + 8]);
        let (al, bh) = in_bounds(sc, lo, hi);
        if rev && !al {
            return false;
        }
        if !rev && !bh {
            return false;
        }
        if al && bh {
            if skipped < offset {
                skipped += 1;
            } else {
                out.push((sc, k[plen + 8..].to_vec()));
                if out.len() >= limit {
                    return false;
                }
            }
        }
        true
    });
    out
}

/// Members by rank: skip `offset`, take `count`, ascending or descending.
fn zby_rank(cx: &mut Ctx, key: &[u8], offset: usize, count: usize, rev: bool) -> Vec<(f64, Vec<u8>)> {
    let (s, e) = erange(key, SUB_ZSCORE);
    let plen = s.len();
    let mut out = Vec::new();
    let mut i = 0;
    if count == 0 {
        return out;
    }
    cx.each(&s.clone(), s, e, rev, |k, _| {
        if i >= offset {
            out.push((score_from(&k[plen..plen + 8]), k[plen + 8..].to_vec()));
        }
        i += 1;
        out.len() < count
    });
    out
}

/// One end of a lex range: `-` / `+` (unbounded) or `[x` / `(x`.
enum LexBound {
    Min,
    Max,
    Incl(Vec<u8>),
    Excl(Vec<u8>),
}

fn lex_bound(a: &[u8]) -> R<LexBound> {
    match a.first() {
        Some(b'-') if a.len() == 1 => Ok(LexBound::Min),
        Some(b'+') if a.len() == 1 => Ok(LexBound::Max),
        Some(b'[') => Ok(LexBound::Incl(a[1..].to_vec())),
        Some(b'(') => Ok(LexBound::Excl(a[1..].to_vec())),
        _ => Err(err("min or max not valid string range item")),
    }
}

fn lex_above(m: &[u8], lo: &LexBound) -> bool {
    match lo {
        LexBound::Min => true,
        LexBound::Max => false,
        LexBound::Incl(x) => m >= x.as_slice(),
        LexBound::Excl(x) => m > x.as_slice(),
    }
}

fn lex_below(m: &[u8], hi: &LexBound) -> bool {
    match hi {
        LexBound::Min => false,
        LexBound::Max => true,
        LexBound::Incl(x) => m <= x.as_slice(),
        LexBound::Excl(x) => m < x.as_slice(),
    }
}

/// Members within a lex range (Redis requires equal scores for lex ranges
/// to be meaningful; members are then in byte order within the index).
fn zby_lex(cx: &mut Ctx, key: &[u8], lo: &LexBound, hi: &LexBound, rev: bool, offset: usize, count: usize) -> Vec<(f64, Vec<u8>)> {
    let (s, e) = erange(key, SUB_ZSCORE);
    let plen = s.len();
    let mut out = Vec::new();
    let mut skipped = 0;
    if count == 0 {
        return out;
    }
    cx.each(&s.clone(), s, e, rev, |k, _| {
        let m = &k[plen + 8..];
        if lex_above(m, lo) && lex_below(m, hi) {
            if skipped < offset {
                skipped += 1;
            } else {
                out.push((score_from(&k[plen..plen + 8]), m.to_vec()));
            }
        }
        out.len() < count
    });
    out
}

fn zrange(cx: &mut Ctx, name: &str, a: &[Vec<u8>]) -> R<Resp> {
    if a.len() < 3 {
        return Err(wrong_args(name));
    }
    let (mut byscore, mut rev, mut with) = (name.contains("BYSCORE"), name.starts_with("ZREV"), false);
    let mut bylex = name.contains("BYLEX");
    let mut limit: Option<(i64, i64)> = None;
    let mut i = 3;
    while i < a.len() {
        match String::from_utf8_lossy(&a[i]).to_ascii_uppercase().as_str() {
            "WITHSCORES" => with = true,
            "BYSCORE" if name == "ZRANGE" => byscore = true,
            "REV" if name == "ZRANGE" => rev = true,
            "BYLEX" if name == "ZRANGE" => bylex = true,
            "LIMIT" => {
                let o = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(|| err("syntax error"))?;
                let c = a.get(i + 2).and_then(|v| parse_i64(v)).ok_or_else(|| err("syntax error"))?;
                limit = Some((o, c));
                i += 2;
            }
            _ => return Err(err("syntax error")),
        }
        i += 1;
    }
    if byscore && bylex {
        return Err(err("syntax error"));
    }
    if bylex && with {
        return Err(err("syntax error, WITHSCORES not supported in combination with BYLEX"));
    }
    let (off, cnt) = match limit {
        Some((o, c)) => (o.max(0) as usize, if c < 0 { usize::MAX } else { c as usize }),
        None => (0, usize::MAX),
    };
    let lex = if bylex {
        // REV takes max first, like BYSCORE.
        let (b1, b2) = (lex_bound(&a[1])?, lex_bound(&a[2])?);
        Some(if rev { (b2, b1) } else { (b1, b2) })
    } else {
        None
    };
    let Some((m, _)) = cx.coll(&a[0], T_ZSET)? else { return Ok(arr(vec![])) };
    let items = if limit.is_some_and(|(o, _)| o < 0) && (bylex || byscore) {
        Vec::new() // Redis: a negative LIMIT offset matches nothing
    } else if let Some((lo, hi)) = lex {
        zby_lex(cx, &a[0], &lo, &hi, rev, off, cnt)
    } else if byscore {
        // ZREVRANGEBYSCORE / ZRANGE .. BYSCORE REV take max first.
        let (b1, b2) = (score_bound(&a[1])?, score_bound(&a[2])?);
        let (lo, hi) = if rev { (b2, b1) } else { (b1, b2) };
        zby_score(cx, &a[0], lo, hi, rev, off, cnt)
    } else {
        if limit.is_some() {
            return Err(err("syntax error, LIMIT is only supported in combination with either BYSCORE or BYLEX"));
        }
        let (s, e) = (parse_i64(&a[1]).ok_or_else(not_int)?, parse_i64(&a[2]).ok_or_else(not_int)?);
        match norm_range(s, e, m.len as i64) {
            None => Vec::new(),
            Some((s, e)) => zby_rank(cx, &a[0], s as usize, (e - s + 1) as usize, rev),
        }
    };
    let mut out = Vec::new();
    for (sc, mbr) in items {
        out.push(bulk(mbr));
        if with {
            out.push(bulk(fmt_f64(sc).into_bytes()));
        }
    }
    Ok(arr(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orderable_encodings() {
        let xs = [f64::NEG_INFINITY, -5.5, -1.0, -0.0, 0.0, 0.25, 1.0, 1e300, f64::INFINITY];
        for w in xs.windows(2) {
            assert!(score_bytes(w[0]) <= score_bytes(w[1]), "{} {}", w[0], w[1]);
            assert_eq!(score_from(&score_bytes(w[1])), if w[1] == 0.0 { 0.0 } else { w[1] });
        }
        for w in [i64::MIN, -3, -1, 0, 1, 7, i64::MAX].windows(2) {
            assert!(seq_bytes(w[0]) < seq_bytes(w[1]));
            assert_eq!(seq_from(&seq_bytes(w[1])), w[1]);
        }
        assert_eq!(fmt_f64(1.0), "1");
        assert_eq!(fmt_f64(1.5), "1.5");
        assert_eq!(fmt_f64(f64::INFINITY), "inf");
    }

    #[test]
    fn meta_roundtrip() {
        let m = Meta { ty: T_LIST, len: 3, head: -2, tail: 1, ver: 9 };
        let d = Meta::decode(&m.encode()).unwrap();
        assert_eq!((d.ty, d.len, d.head, d.tail, d.ver), (T_LIST, 3, -2, 1, 9));
    }
}
