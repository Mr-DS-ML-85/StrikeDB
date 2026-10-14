//! Git-style version control for the keyspace: commits, branches, tags,
//! diffs, three-way merge, revert, reset and time-travel reads (`VC.*`).
//!
//! Modelled on Dolt / Irmin semantics, at Redis granularity: the unit of
//! change and of merge conflicts is a USER KEY (a string, hash, list, set,
//! sorted set or stream, together with its TTL). Underneath, everything is
//! tracked at the level of the engine keys a user key is stored as
//! (`kv:<k>`, `exp:<k>`, `c:{hex k}:...`), so every data type is versioned
//! the same way and a checkout restores collections exactly.
//!
//! There is one working copy: the live keyspace, like a git working tree.
//! `VC.CHECKOUT` rewrites it to another branch (only the keys that differ),
//! and `VC.GET`/`VC.KEYS`/`VC.DIFF` read any commit without a checkout.
//!
//! Storage (all under the `{r}` hash tag, so in one shard and range-
//! scannable on the commit path):
//!   * `vcs:{r}:head`            current branch name
//!   * `vcs:{r}:seq`             next commit sequence number
//!   * `vcs:{r}:c:<seq>`         commit record (id, parents, time, author, msg)
//!   * `vcs:{r}:id:<id>`         commit id → seq
//!   * `vcs:{r}:b:<name>`        branch → seq;  `vcs:{r}:t:<name>` tag → seq
//!   * `vcs:{r}:h:<key*><seq>`   history: the key's value as of that commit
//!   * `vcs:{r}:s:<seq><key*>`   changeset: keys a commit changed
//!     (`key*` = key with 0x00 escaped as 00 FF, terminated by 00 00, so
//!     the encoding is order- and prefix-preserving)
//!
//! Reading key K at commit C = the newest history entry of K made by an
//! ancestor of C. A diff between two commits only looks at the keys in the
//! changesets of commits reachable from one but not the other, so diffs,
//! checkouts and merges cost O(changes), not O(database).
//!
//! What changed since HEAD is tracked from the engine's commit hook (a
//! dirty-key set), so `VC.COMMIT` does not scan the database. After a
//! restart the set is unknown, and the first status/commit scans once.
//!
//! Every `VC.*` command runs as one operation on the engine's commit path:
//! it is serialized against all writes and its effects commit atomically.

use crate::keyspace;
use crate::{err, Db};
use protocol::Resp;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use storage::{Key, Mutation, Store, Value};

const P: &[u8] = b"vcs:{r}:";

fn k(parts: &[&[u8]]) -> Vec<u8> {
    let mut v = P.to_vec();
    for p in parts {
        v.extend_from_slice(p);
    }
    v
}

fn esc(key: &[u8], terminate: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(key.len() + 2);
    for &b in key {
        v.push(b);
        if b == 0 {
            v.push(0xFF);
        }
    }
    if terminate {
        v.extend_from_slice(&[0, 0]);
    }
    v
}

/// Split `<key*><rest>` → (key, rest).
fn unesc(b: &[u8]) -> Option<(Vec<u8>, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == 0 {
            match b.get(i + 1)? {
                0xFF => out.push(0),
                0 => return Some((out, &b[i + 2..])),
                _ => return None,
            }
            i += 2;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    None
}

fn next_prefix(b: &[u8]) -> Vec<u8> {
    let mut v = b.to_vec();
    while let Some(last) = v.pop() {
        if last != 0xff {
            v.push(last + 1);
            return v;
        }
    }
    vec![0xff; b.len() + 1]
}

fn hist_key(key: &[u8], seq: u64) -> Vec<u8> {
    let mut v = k(&[b"h:"]);
    v.extend_from_slice(&esc(key, true));
    v.extend_from_slice(&seq.to_be_bytes());
    v
}

fn cs_key(seq: u64, key: &[u8]) -> Vec<u8> {
    let mut v = k(&[b"s:", &seq.to_be_bytes()]);
    v.extend_from_slice(&esc(key, true));
    v
}

/// History values: tag 0 = absent (deleted), 1 = present + encoded Value.
fn enc_val(v: &Option<Value>) -> Value {
    match v {
        None => Value::Bytes(vec![0]),
        Some(v) => {
            let mut b = vec![1];
            b.extend_from_slice(&v.encode());
            Value::Bytes(b)
        }
    }
}

fn dec_val(v: &Value) -> Option<Value> {
    match v {
        Value::Bytes(b) if b.first() == Some(&1) => Value::decode(&b[1..]),
        _ => None,
    }
}

/// Engine keys that belong to the versioned keyspace, and their user key.
pub fn user_key(engine_key: &[u8]) -> Option<Vec<u8>> {
    if let Some(u) = engine_key.strip_prefix(b"kv:") {
        return Some(u.to_vec());
    }
    if let Some(u) = engine_key.strip_prefix(b"exp:") {
        return Some(u.to_vec());
    }
    let rest = engine_key.strip_prefix(b"c:{")?;
    let end = rest.iter().position(|&b| b == b'}')?;
    let hex = &rest[..end];
    if hex.len() % 2 != 0 {
        return None;
    }
    let nib = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    hex.chunks(2).map(|p| Some(nib(p[0])? << 4 | nib(p[1])?)).collect()
}

/// The engine keys / prefix a user key may occupy.
fn engine_keys_of(user: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut kv = b"kv:".to_vec();
    kv.extend_from_slice(user);
    let mut exp = b"exp:".to_vec();
    exp.extend_from_slice(user);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut c = b"c:{".to_vec();
    for b in user {
        c.push(HEX[(b >> 4) as usize]);
        c.push(HEX[(b & 15) as usize]);
    }
    c.extend_from_slice(b"}:");
    (kv, exp, c)
}

fn hash64(parts: &[&[u8]]) -> u64 {
    // FNV-1a over the parts, then a splitmix finalizer.
    let mut h: u64 = 0xcbf29ce484222325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

// ── Commit records ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct Commit {
    id: String,
    parents: Vec<u64>,
    time: u64,
    author: Vec<u8>,
    msg: Vec<u8>,
}

impl Commit {
    fn enc(&self) -> Value {
        let mut b = Vec::new();
        b.extend_from_slice(&(self.parents.len() as u32).to_le_bytes());
        for p in &self.parents {
            b.extend_from_slice(&p.to_le_bytes());
        }
        b.extend_from_slice(&self.time.to_le_bytes());
        for f in [self.id.as_bytes(), &self.author, &self.msg] {
            b.extend_from_slice(&(f.len() as u32).to_le_bytes());
            b.extend_from_slice(f);
        }
        Value::Bytes(b)
    }

    fn dec(v: &Value) -> Option<Commit> {
        let Value::Bytes(b) = v else { return None };
        let u32_at = |i: usize| b.get(i..i + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize);
        let u64_at = |i: usize| b.get(i..i + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap()));
        let np = u32_at(0)?;
        let mut i = 4;
        let mut parents = Vec::with_capacity(np);
        for _ in 0..np {
            parents.push(u64_at(i)?);
            i += 8;
        }
        let time = u64_at(i)?;
        i += 8;
        let mut fields = Vec::new();
        for _ in 0..3 {
            let l = u32_at(i)?;
            i += 4;
            fields.push(b.get(i..i + l)?.to_vec());
            i += l;
        }
        let msg = fields.pop()?;
        let author = fields.pop()?;
        let id = String::from_utf8(fields.pop()?).ok()?;
        Some(Commit { id, parents, time, author, msg })
    }

    fn resp(&self, st: &mut dyn Store) -> Resp {
        let parents: Vec<Resp> = self
            .parents
            .iter()
            .filter_map(|p| load_commit(st, *p))
            .map(|c| Resp::Bulk(c.id.into_bytes()))
            .collect();
        Resp::Array(vec![
            bulk("id"),
            Resp::Bulk(self.id.clone().into_bytes()),
            bulk("parents"),
            Resp::Array(parents),
            bulk("time"),
            Resp::Int(self.time as i64),
            bulk("author"),
            Resp::Bulk(self.author.clone()),
            bulk("message"),
            Resp::Bulk(self.msg.clone()),
        ])
    }
}

fn bulk(s: &str) -> Resp {
    Resp::Bulk(s.as_bytes().to_vec())
}

fn int_of(v: Option<Value>) -> Option<u64> {
    match v {
        Some(Value::Int(i)) => Some(i as u64),
        _ => None,
    }
}

fn load_commit(st: &mut dyn Store, seq: u64) -> Option<Commit> {
    Commit::dec(&st.get(&k(&[b"c:", &seq.to_be_bytes()]))?)
}

fn head_branch(st: &mut dyn Store) -> Option<Vec<u8>> {
    match st.get(&k(&[b"head"])) {
        Some(Value::Bytes(b)) => Some(b),
        _ => None,
    }
}

fn branch_seq(st: &mut dyn Store, name: &[u8]) -> Option<u64> {
    int_of(st.get(&k(&[b"b:", name])))
}

fn head_seq(st: &mut dyn Store) -> R<u64> {
    let b = head_branch(st).ok_or_else(not_init)?;
    branch_seq(st, &b).ok_or_else(|| err("HEAD branch is missing"))
}

type R<T> = Result<T, Resp>;

fn not_init() -> Resp {
    err("version control is not initialized (run VC.INIT)")
}

/// Resolve `HEAD`, a branch, a tag, a commit id (or unique prefix of at
/// least 4 hex digits), each optionally followed by `~N` / `^` (first
/// parent).
fn resolve(st: &mut dyn Store, r: &[u8]) -> R<u64> {
    let s = String::from_utf8_lossy(r).to_string();
    let (base, mut up) = match s.find(['~', '^']) {
        Some(p) => {
            let suffix = &s[p..];
            let mut n = 0usize;
            let mut chars = suffix.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '^' => n += 1,
                    '~' => {
                        let digits: String = std::iter::from_fn(|| chars.next_if(|d| d.is_ascii_digit())).collect();
                        n += if digits.is_empty() { 1 } else { digits.parse::<usize>().map_err(|_| bad_ref(&s))? };
                    }
                    _ => return Err(bad_ref(&s)),
                }
            }
            (s[..p].to_string(), n)
        }
        None => (s.clone(), 0),
    };
    let mut seq = if base.eq_ignore_ascii_case("HEAD") {
        head_seq(st)?
    } else if let Some(b) = branch_seq(st, base.as_bytes()) {
        b
    } else if let Some(t) = int_of(st.get(&k(&[b"t:", base.as_bytes()]))) {
        t
    } else if let Some(c) = int_of(st.get(&k(&[b"id:", base.as_bytes()]))) {
        c
    } else if base.len() >= 4 && base.bytes().all(|c| c.is_ascii_hexdigit()) {
        let start = k(&[b"id:", base.as_bytes()]);
        let end = next_prefix(&start);
        let hits = st.scan_range(&start, &start, &end, 2, false);
        match hits.as_slice() {
            [(_, v)] => int_of(Some(v.clone())).ok_or_else(|| bad_ref(&s))?,
            [] => return Err(bad_ref(&s)),
            _ => return Err(err(&format!("ambiguous commit prefix '{base}'"))),
        }
    } else {
        return Err(bad_ref(&s));
    };
    while up > 0 {
        let c = load_commit(st, seq).ok_or_else(|| bad_ref(&s))?;
        seq = *c.parents.first().ok_or_else(|| err(&format!("'{s}' goes past the root commit")))?;
        up -= 1;
    }
    Ok(seq)
}

fn bad_ref(s: &str) -> Resp {
    err(&format!("unknown revision '{s}'"))
}

fn ancestors(st: &mut dyn Store, seq: u64) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut todo = vec![seq];
    while let Some(s) = todo.pop() {
        if !seen.insert(s) {
            continue;
        }
        if let Some(c) = load_commit(st, s) {
            todo.extend(c.parents);
        }
    }
    seen
}

/// Value of engine key `key` as of the commit whose ancestor set is `anc`.
fn value_at(st: &mut dyn Store, key: &[u8], anc: &HashSet<u64>) -> Option<Value> {
    let mut start = k(&[b"h:"]);
    start.extend_from_slice(&esc(key, true));
    let plen = start.len();
    let mut end = start.clone();
    end.extend_from_slice(&[0xff; 9]);
    let mut hi = end;
    loop {
        let page = st.scan_range(&start, &start, &hi, 64, true);
        let n = page.len();
        for (hk, v) in &page {
            let seq = u64::from_be_bytes(hk[plen..plen + 8].try_into().ok()?);
            if anc.contains(&seq) {
                return dec_val(v);
            }
        }
        if n < 64 {
            return None;
        }
        hi = page.last()?.0.clone();
    }
}

/// Every engine key a user key occupies as of a commit, with its value.
fn user_state_at(st: &mut dyn Store, user: &[u8], anc: &HashSet<u64>) -> Vec<(Vec<u8>, Value)> {
    let (kv, exp, c) = engine_keys_of(user);
    let mut out = Vec::new();
    for ek in [&kv, &exp] {
        if let Some(v) = value_at(st, ek, anc) {
            out.push((ek.clone(), v));
        }
    }
    // Every element key that ever existed under the collection prefix.
    let mut start = k(&[b"h:"]);
    start.extend_from_slice(&esc(&c, false));
    let end = next_prefix(&start);
    let plen = k(&[b"h:"]).len();
    let mut best: BTreeMap<Vec<u8>, (u64, Value)> = BTreeMap::new();
    let mut s = start.clone();
    loop {
        let page = st.scan_range(&start, &s, &end, 512, false);
        let n = page.len();
        for (hk, v) in &page {
            let Some((ek, rest)) = unesc(&hk[plen..]) else { continue };
            let Ok(seq_b) = <[u8; 8]>::try_from(&rest[..8.min(rest.len())]) else { continue };
            let seq = u64::from_be_bytes(seq_b);
            if anc.contains(&seq) && best.get(&ek).is_none_or(|(s0, _)| seq > *s0) {
                best.insert(ek, (seq, v.clone()));
            }
        }
        if n < 512 {
            break;
        }
        s = page.last().unwrap().0.clone();
        s.push(0);
    }
    for (ek, (_, v)) in best {
        if let Some(v) = dec_val(&v) {
            out.push((ek, v));
        }
    }
    out
}

/// Engine keys changed by any commit in `seqs` (from their changesets).
fn changed_keys(st: &mut dyn Store, seqs: impl IntoIterator<Item = u64>) -> BTreeSet<Vec<u8>> {
    let mut out = BTreeSet::new();
    for seq in seqs {
        let start = k(&[b"s:", &seq.to_be_bytes()]);
        let end = next_prefix(&start);
        let plen = start.len();
        let mut s = start.clone();
        loop {
            let page = st.scan_range(&start, &s, &end, 1024, false);
            let n = page.len();
            for (ck, _) in &page {
                if let Some((ek, _)) = unesc(&ck[plen..]) {
                    out.insert(ek);
                }
            }
            if n < 1024 {
                break;
            }
            s = page.last().unwrap().0.clone();
            s.push(0);
        }
    }
    out
}

/// Every engine key with history reachable from `anc` (the full tree of a
/// commit). Used only by the post-restart full scan and `VC.KEYS`.
fn all_history_keys(st: &mut dyn Store, anc: &HashSet<u64>) -> BTreeSet<Vec<u8>> {
    changed_keys(st, anc.iter().copied())
}

fn group_by_user(keys: impl IntoIterator<Item = Vec<u8>>) -> BTreeSet<Vec<u8>> {
    keys.into_iter().filter_map(|k| user_key(&k)).collect()
}

/// Live engine keys of a user key.
fn live_state(st: &mut dyn Store, user: &[u8]) -> Vec<(Vec<u8>, Value)> {
    let (kv, exp, c) = engine_keys_of(user);
    let mut out = Vec::new();
    for ek in [&kv, &exp] {
        if let Some(v) = st.get(ek) {
            out.push((ek.clone(), v));
        }
    }
    let end = next_prefix(&c);
    out.extend(st.scan_range(&c, &c, &end, usize::MAX, false));
    out
}

/// Replace the live state of `user` with `state`.
fn write_user(st: &mut dyn Store, user: &[u8], state: &[(Vec<u8>, Value)]) {
    for (ek, _) in live_state(st, user) {
        st.delete(ek);
    }
    for (ek, v) in state {
        st.put(ek.clone(), v.clone());
    }
}

fn change_kind(before: &[(Vec<u8>, Value)], after: &[(Vec<u8>, Value)]) -> Option<&'static str> {
    match (before.is_empty(), after.is_empty()) {
        _ if before == after => None,
        (true, _) => Some("added"),
        (_, true) => Some("deleted"),
        _ => Some("modified"),
    }
}

fn kind_of(state: &[(Vec<u8>, Value)]) -> &'static str {
    for (ek, v) in state {
        if ek.starts_with(b"kv:") {
            return match v {
                Value::Meta(m) => match m.first() {
                    Some(1) => "hash",
                    Some(2) => "list",
                    Some(3) => "set",
                    Some(4) => "zset",
                    Some(5) => "stream",
                    _ => "collection",
                },
                _ => "string",
            };
        }
    }
    "none"
}

// ── The component ──────────────────────────────────────────────────────────

#[derive(Default)]
pub struct Vcs {
    /// A repository exists (dirty tracking on).
    enabled: AtomicBool,
    /// The dirty set is incomplete (after a restart): scan once.
    full_scan: AtomicBool,
    /// Engine keys changed since HEAD, stamped with the epoch at which the
    /// change was observed (see `on_commit` / `take_dirty`).
    dirty: Mutex<HashMap<Key, u64>>,
    epoch: AtomicU64,
}

impl Vcs {
    pub fn open(engine: &storage::Engine) -> Arc<Self> {
        let v = Arc::new(Vcs::default());
        if engine.get(&k(&[b"head"])).is_some() {
            v.enabled.store(true, Ordering::Relaxed);
            v.full_scan.store(true, Ordering::Relaxed);
        }
        v
    }

    /// Engine commit hook (commit thread): remember changed user keys.
    pub fn on_commit(&self, m: &Mutation) {
        if self.enabled.load(Ordering::Relaxed) && user_key(&m.key).is_some() {
            let e = self.epoch.load(Ordering::SeqCst);
            self.dirty.lock().unwrap().insert(m.key.clone(), e);
        }
    }

    /// FLUSHALL wipes the repository too.
    pub fn on_flush(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.full_scan.store(false, Ordering::Relaxed);
        self.dirty.lock().unwrap().clear();
    }

    /// Engine keys that may differ from HEAD. Runs inside the commit-path
    /// op: keys committed earlier in the same flush batch are not yet
    /// notified, so the batch's pending keys are included too.
    fn candidates(&self, st: &mut dyn Store, scanned: &[Vec<u8>]) -> BTreeSet<Vec<u8>> {
        let mut c: BTreeSet<Vec<u8>> = self.dirty.lock().unwrap().keys().cloned().collect();
        c.extend(st.batch_keys().into_iter().filter(|k| user_key(k).is_some()));
        c.extend(scanned.iter().cloned());
        c
    }

    /// Forget dirty entries observed before epoch `e` (they are now part of
    /// HEAD). Later notifications carry a larger epoch and are kept.
    fn settle(&self, e: u64) {
        self.dirty.lock().unwrap().retain(|_, at| *at > e);
    }

    /// Mark keys as possibly differing from HEAD (soft reset, status after
    /// a restart). Called inside an op, after its epoch bump: the op's own
    /// `settle` keeps them, the next commit's clears them.
    fn mark(&self, keys: impl IntoIterator<Item = Vec<u8>>) {
        let e = self.epoch.load(Ordering::SeqCst);
        let mut d = self.dirty.lock().unwrap();
        for k in keys {
            d.insert(k, e);
        }
    }
}

/// Working-copy changes vs HEAD, at engine-key level: key → live value.
fn working_changes(st: &mut dyn Store, vcs: &Vcs, scanned: &[Vec<u8>], anc: &HashSet<u64>, include_head_tree: bool) -> Vec<(Vec<u8>, Option<Value>)> {
    let mut cands = vcs.candidates(st, scanned);
    if include_head_tree {
        cands.extend(all_history_keys(st, anc));
    }
    let mut out = Vec::new();
    for ek in cands {
        let live = st.get(&ek);
        let head = value_at(st, &ek, anc);
        if live != head {
            out.push((ek, live));
        }
    }
    out
}

/// Write a commit with the given engine-key changes on top of `parents`,
/// advancing `branch`. Returns the commit.
fn write_commit(st: &mut dyn Store, branch: &[u8], parents: Vec<u64>, changes: &[(Vec<u8>, Option<Value>)], author: &[u8], msg: &[u8], now: u64) -> Commit {
    let seq = int_of(st.get(&k(&[b"seq"]))).unwrap_or(1);
    let mut digest = Vec::new();
    for (ek, v) in changes {
        digest.extend_from_slice(&hash64(&[ek, &enc_val(v).encode()]).to_le_bytes());
    }
    let pbytes: Vec<u8> = parents.iter().flat_map(|p| p.to_le_bytes()).collect();
    let id = format!("{:016x}", hash64(&[&pbytes, &digest, msg, author, &now.to_le_bytes(), &seq.to_le_bytes()]));
    let c = Commit { id: id.clone(), parents, time: now, author: author.to_vec(), msg: msg.to_vec() };
    st.put(k(&[b"c:", &seq.to_be_bytes()]), c.enc());
    st.put(k(&[b"id:", id.as_bytes()]), Value::Int(seq as i64));
    for (ek, v) in changes {
        st.put(hist_key(ek, seq), enc_val(v));
        st.put(cs_key(seq, ek), Value::Int(0));
    }
    st.put(k(&[b"b:", branch]), Value::Int(seq as i64));
    st.put(k(&[b"seq"]), Value::Int(seq as i64 + 1));
    c
}

fn arg_str(a: &[u8]) -> String {
    String::from_utf8_lossy(a).to_ascii_uppercase()
}

fn valid_name(n: &[u8]) -> bool {
    !n.is_empty()
        && n.len() <= 200
        && !n.eq_ignore_ascii_case(b"HEAD")
        && n.iter().all(|&c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'/'))
        && !n.contains(&b'~')
}

/// What a command needs scanned before it can run on the commit path.
fn needs_scan(name: &str) -> bool {
    matches!(name, "VC.INIT" | "VC.COMMIT" | "VC.STATUS" | "VC.DIFF" | "VC.CHECKOUT" | "VC.MERGE" | "VC.RESET" | "VC.REVERT")
}

pub fn is_vcs_cmd(name: &str) -> bool {
    name.starts_with("VC.")
}

/// Entry point for every `VC.*` command.
pub fn command(db: &Db, name: &str, args: &[Vec<u8>]) -> Resp {
    let vcs = Arc::clone(&db.vcs);
    // Live versioned keys, for VC.INIT and the first command after a
    // restart (the dirty set is incomplete then). Scanned on a snapshot
    // BEFORE the op: anything written after the snapshot is in the dirty
    // set or the op's pending batch.
    let full = name == "VC.INIT" || (needs_scan(name) && vcs.full_scan.load(Ordering::Relaxed));
    if name == "VC.INIT" {
        vcs.enabled.store(true, Ordering::SeqCst);
    }
    let scanned: Vec<Vec<u8>> = if full {
        let snap = db.engine.snapshot();
        let mut v = Vec::new();
        for p in [&b"kv:"[..], b"exp:", b"c:{"] {
            v.extend(db.engine.scan_prefix(p, snap).into_iter().map(|(k, _)| k).filter(|k| user_key(k).is_some()));
        }
        v
    } else {
        Vec::new()
    };
    let (name_s, args_v) = (name.to_string(), args.to_vec());
    let slot: Arc<Mutex<Option<(Resp, Option<u64>, bool)>>> = Arc::new(Mutex::new(None));
    let out = Arc::clone(&slot);
    let v2 = Arc::clone(&vcs);
    let res = db.engine.submit_op(Box::new(move |st: &mut dyn Store| {
        let epoch = v2.epoch.fetch_add(1, Ordering::SeqCst);
        let mut cx = Cx { st, vcs: &v2, scanned: &scanned, full, epoch, settle: None, reload: false, now: keyspace::now_ms() };
        let sp = cx.st.savepoint();
        let r = match cx.run(&name_s, &args_v) {
            Ok(r) => r,
            Err(e) => {
                cx.st.rollback_to(sp);
                cx.settle = None;
                cx.reload = false;
                e
            }
        };
        *out.lock().unwrap() = Some((r, cx.settle, cx.reload));
    }));
    if let Err(e) = res {
        if name == "VC.INIT" {
            vcs.enabled.store(false, Ordering::SeqCst);
        }
        return err(&e.to_string());
    }
    let Some((reply, settle, reload)) = slot.lock().unwrap().take() else { return err("operation did not run") };
    if name == "VC.INIT" && matches!(reply, Resp::Error(_)) && db.engine.get(&k(&[b"head"])).is_none() {
        vcs.enabled.store(false, Ordering::SeqCst);
    }
    if let Some(e) = settle {
        vcs.settle(e);
        if full {
            vcs.full_scan.store(false, Ordering::Relaxed);
        }
    }
    if reload {
        // The working copy was rewritten: refresh the RAM mirrors.
        db.ks.reload();
    }
    reply
}

struct Cx<'a> {
    st: &'a mut dyn Store,
    vcs: &'a Vcs,
    scanned: &'a [Vec<u8>],
    full: bool,
    epoch: u64,
    /// On success: drop dirty entries observed up to this epoch.
    settle: Option<u64>,
    /// On success: the live keyspace was rewritten.
    reload: bool,
    now: u64,
}

impl Cx<'_> {
    fn head(&mut self) -> R<(Vec<u8>, u64, HashSet<u64>)> {
        let b = head_branch(self.st).ok_or_else(not_init)?;
        let seq = branch_seq(self.st, &b).ok_or_else(|| err("HEAD branch is missing"))?;
        let anc = ancestors(self.st, seq);
        Ok((b, seq, anc))
    }

    fn anc_of(&mut self, r: &[u8]) -> R<HashSet<u64>> {
        let seq = resolve(self.st, r)?;
        Ok(ancestors(self.st, seq))
    }

    fn changes(&mut self, anc: &HashSet<u64>) -> Vec<(Vec<u8>, Option<Value>)> {
        working_changes(self.st, self.vcs, self.scanned, anc, self.full)
    }

    fn require_clean(&mut self, anc: &HashSet<u64>) -> R<()> {
        let ch = self.changes(anc);
        if ch.is_empty() {
            Ok(())
        } else {
            let users = group_by_user(ch.into_iter().map(|(k, _)| k));
            Err(err(&format!(
                "you have uncommitted changes to {} key(s); VC.COMMIT them or use FORCE",
                users.len()
            )))
        }
    }

    /// Rewrite the live state of engine keys `keys` to their values at `anc`.
    fn materialize(&mut self, keys: impl IntoIterator<Item = Vec<u8>>, anc: &HashSet<u64>) {
        for ek in keys {
            match value_at(self.st, &ek, anc) {
                Some(v) => self.st.put(ek, v),
                None => self.st.delete(ek),
            }
        }
        self.reload = true;
    }

    fn diff_users(&mut self, from: &HashSet<u64>, to: Option<&HashSet<u64>>, keys: BTreeSet<Vec<u8>>) -> Vec<Resp> {
        let mut out = Vec::new();
        for u in group_by_user(keys) {
            let a = user_state_at(self.st, &u, from);
            let b = match to {
                Some(t) => user_state_at(self.st, &u, t),
                None => live_state(self.st, &u),
            };
            if let Some(kind) = change_kind(&a, &b) {
                let ty = if b.is_empty() { kind_of(&a) } else { kind_of(&b) };
                out.push(Resp::Array(vec![Resp::Bulk(u), bulk(kind), bulk(ty)]));
            }
        }
        out
    }

    fn run(&mut self, name: &str, a: &[Vec<u8>]) -> R<Resp> {
        let n = a.len();
        let wrong = || err(&format!("wrong number of arguments for '{}' command", name.to_lowercase()));
        if name != "VC.INIT" && head_branch(self.st).is_none() {
            return Err(not_init());
        }
        match name {
            "VC.INIT" => {
                if head_branch(self.st).is_some() {
                    return Err(err("version control is already initialized"));
                }
                let branch = a.first().cloned().unwrap_or_else(|| b"main".to_vec());
                if !valid_name(&branch) {
                    return Err(err("invalid branch name"));
                }
                // Root commit: the whole current keyspace.
                let changes = working_changes(self.st, self.vcs, self.scanned, &HashSet::new(), false);
                self.st.put(k(&[b"head"]), Value::Bytes(branch.clone()));
                let c = write_commit(self.st, &branch, Vec::new(), &changes, b"", b"Initial commit", self.now);
                self.settle = Some(self.epoch);
                Ok(Resp::Bulk(c.id.into_bytes()))
            }
            "VC.COMMIT" => {
                if n == 0 {
                    return Err(wrong());
                }
                let (mut author, mut allow_empty, mut i) = (b"".to_vec(), false, 1);
                while i < n {
                    match arg_str(&a[i]).as_str() {
                        "AUTHOR" => {
                            author = a.get(i + 1).ok_or_else(|| err("syntax error"))?.clone();
                            i += 2;
                        }
                        "ALLOWEMPTY" => {
                            allow_empty = true;
                            i += 1;
                        }
                        _ => return Err(err("syntax error")),
                    }
                }
                let (b, seq, anc) = self.head()?;
                let changes = self.changes(&anc);
                if changes.is_empty() && !allow_empty {
                    return Err(err("nothing to commit, working tree clean"));
                }
                let c = write_commit(self.st, &b, vec![seq], &changes, &author, &a[0], self.now);
                self.settle = Some(self.epoch);
                Ok(Resp::Bulk(c.id.into_bytes()))
            }
            "VC.STATUS" => {
                let (b, seq, anc) = self.head()?;
                let keys: BTreeSet<Vec<u8>> = self.changes(&anc).into_iter().map(|(k, _)| k).collect();
                if self.full {
                    // The scan found every changed key: keep them in the
                    // dirty set and leave full-scan mode.
                    self.vcs.mark(keys.iter().cloned());
                    self.settle = Some(self.epoch);
                }
                let diff = self.diff_users(&anc, None, keys);
                let c = load_commit(self.st, seq).ok_or_else(|| err("HEAD commit is missing"))?;
                Ok(Resp::Array(vec![bulk("branch"), Resp::Bulk(b), bulk("head"), Resp::Bulk(c.id.into_bytes()), bulk("changes"), Resp::Array(diff)]))
            }
            "VC.LOG" => {
                let (mut r, mut count, mut i) = (None, usize::MAX, 0);
                while i < n {
                    if a[i].eq_ignore_ascii_case(b"COUNT") {
                        let c = a.get(i + 1).and_then(|v| std::str::from_utf8(v).ok()?.parse::<usize>().ok()).ok_or_else(|| err("value is not an integer or out of range"))?;
                        count = c;
                        i += 2;
                    } else if r.is_none() {
                        r = Some(a[i].clone());
                        i += 1;
                    } else {
                        return Err(err("syntax error"));
                    }
                }
                let seq = resolve(self.st, r.as_deref().unwrap_or(b"HEAD"))?;
                let mut anc: Vec<u64> = ancestors(self.st, seq).into_iter().collect();
                anc.sort_unstable_by(|x, y| y.cmp(x));
                let mut out = Vec::new();
                for s in anc.into_iter().take(count) {
                    if let Some(c) = load_commit(self.st, s) {
                        out.push(c.resp(self.st));
                    }
                }
                Ok(Resp::Array(out))
            }
            "VC.BRANCH" | "VC.TAG" => {
                let (pre, what): (&[u8], &str) = if name == "VC.BRANCH" { (b"b:", "branch") } else { (b"t:", "tag") };
                if n == 0 {
                    let head = head_branch(self.st);
                    let start = k(&[pre]);
                    let end = next_prefix(&start);
                    let mut out = Vec::new();
                    for (key, v) in self.st.scan_range(&start, &start, &end, usize::MAX, false) {
                        let nm = key[start.len()..].to_vec();
                        let id = int_of(Some(v)).and_then(|s| load_commit(self.st, s)).map(|c| c.id).unwrap_or_default();
                        let mut row = vec![Resp::Bulk(nm.clone()), Resp::Bulk(id.into_bytes())];
                        if name == "VC.BRANCH" {
                            row.push(Resp::Int((head.as_deref() == Some(&nm[..])) as i64));
                        }
                        out.push(Resp::Array(row));
                    }
                    return Ok(Resp::Array(out));
                }
                if a[0].eq_ignore_ascii_case(b"DELETE") {
                    let nm = a.get(1).ok_or_else(wrong)?;
                    if self.st.get(&k(&[pre, nm])).is_none() {
                        return Err(err(&format!("{what} '{}' not found", String::from_utf8_lossy(nm))));
                    }
                    if name == "VC.BRANCH" && head_branch(self.st).as_deref() == Some(&nm[..]) {
                        return Err(err("cannot delete the checked-out branch"));
                    }
                    self.st.delete(k(&[pre, nm]));
                    return Ok(Resp::Int(1));
                }
                if n > 2 {
                    return Err(wrong());
                }
                let nm = &a[0];
                if !valid_name(nm) {
                    return Err(err(&format!("invalid {what} name")));
                }
                if self.st.get(&k(&[pre, nm])).is_some() {
                    return Err(err(&format!("{what} '{}' already exists", String::from_utf8_lossy(nm))));
                }
                let seq = resolve(self.st, a.get(1).map(|v| v.as_slice()).unwrap_or(b"HEAD"))?;
                self.st.put(k(&[pre, nm]), Value::Int(seq as i64));
                Ok(Resp::Simple("OK".into()))
            }
            "VC.CHECKOUT" => {
                if n == 0 || n > 2 {
                    return Err(wrong());
                }
                let force = a.get(1).is_some_and(|f| f.eq_ignore_ascii_case(b"FORCE"));
                if n == 2 && !force {
                    return Err(err("syntax error"));
                }
                let target = &a[0];
                let tseq = branch_seq(self.st, target).ok_or_else(|| err(&format!("branch '{}' not found (VC.CHECKOUT takes a branch; use VC.BRANCH <name> <rev> to start one at a commit)", String::from_utf8_lossy(target))))?;
                let (_, _, anc) = self.head()?;
                let mut keys = BTreeSet::new();
                if force {
                    keys.extend(self.changes(&anc).into_iter().map(|(k, _)| k));
                } else {
                    self.require_clean(&anc)?;
                }
                let tanc = ancestors(self.st, tseq);
                keys.extend(changed_keys(self.st, anc.symmetric_difference(&tanc).copied()));
                let changed = keys.len();
                self.materialize(keys, &tanc);
                self.st.put(k(&[b"head"]), Value::Bytes(target.clone()));
                self.settle = Some(self.epoch);
                Ok(Resp::Array(vec![bulk("branch"), Resp::Bulk(target.clone()), bulk("keys_rewritten"), Resp::Int(changed as i64)]))
            }
            "VC.DIFF" => {
                if n > 2 {
                    return Err(wrong());
                }
                let (_, _, head_anc) = self.head()?;
                match n {
                    0 | 1 => {
                        let from = if n == 1 { self.anc_of(&a[0])? } else { head_anc.clone() };
                        let mut keys: BTreeSet<Vec<u8>> = self.changes(&head_anc).into_iter().map(|(k, _)| k).collect();
                        keys.extend(changed_keys(self.st, from.symmetric_difference(&head_anc).copied()));
                        Ok(Resp::Array(self.diff_users(&from, None, keys)))
                    }
                    _ => {
                        let fa = self.anc_of(&a[0])?;
                        let ta = self.anc_of(&a[1])?;
                        let keys = changed_keys(self.st, fa.symmetric_difference(&ta).copied());
                        Ok(Resp::Array(self.diff_users(&fa, Some(&ta), keys)))
                    }
                }
            }
            "VC.GET" => {
                if n != 2 {
                    return Err(wrong());
                }
                let anc = self.anc_of(&a[0])?;
                let state = user_state_at(self.st, &a[1], &anc);
                Ok(match state.iter().find(|(ek, _)| ek.starts_with(b"kv:")).map(|(_, v)| v) {
                    None => Resp::Nil,
                    Some(Value::Bytes(b)) => Resp::Bulk(b.clone()),
                    Some(Value::Int(i)) => Resp::Bulk(i.to_string().into_bytes()),
                    Some(Value::Float(f)) => Resp::Bulk(keyspace::fmt_f64(*f).into_bytes()),
                    Some(_) => return Err(err(&format!("WRONGTYPE key holds a {} at that revision; VC.GET reads strings (use VC.RESTORE to bring it back)", kind_of(&state)))),
                })
            }
            "VC.KEYS" => {
                if n == 0 || n > 2 {
                    return Err(wrong());
                }
                let anc = self.anc_of(&a[0])?;
                let pat = a.get(1).cloned().unwrap_or_else(|| b"*".to_vec());
                let users = group_by_user(all_history_keys(self.st, &anc));
                let mut out = Vec::new();
                for u in users {
                    if crate::acl::glob_match(&pat, &u) && !user_state_at(self.st, &u, &anc).is_empty() {
                        out.push(Resp::Bulk(u));
                    }
                }
                Ok(Resp::Array(out))
            }
            "VC.HISTORY" => {
                if n == 0 {
                    return Err(wrong());
                }
                let mut count = usize::MAX;
                if n == 3 && a[1].eq_ignore_ascii_case(b"COUNT") {
                    count = std::str::from_utf8(&a[2]).ok().and_then(|v| v.parse().ok()).ok_or_else(|| err("value is not an integer or out of range"))?;
                } else if n != 1 {
                    return Err(err("syntax error"));
                }
                let (_, _, anc) = self.head()?;
                let (kv, exp, c) = engine_keys_of(&a[0]);
                let mut seqs = BTreeSet::new();
                let plen = k(&[b"h:"]).len();
                for (pref, exact) in [(kv, true), (exp, true), (c, false)] {
                    let mut start = k(&[b"h:"]);
                    start.extend_from_slice(&esc(&pref, exact));
                    let end = if exact { let mut e = start.clone(); e.extend_from_slice(&[0xff; 9]); e } else { next_prefix(&start) };
                    for (hk, _) in self.st.scan_range(&start, &start, &end, usize::MAX, false) {
                        if let Some((_, rest)) = unesc(&hk[plen..]) {
                            if let Ok(b) = <[u8; 8]>::try_from(&rest[..8.min(rest.len())]) {
                                let s = u64::from_be_bytes(b);
                                if anc.contains(&s) {
                                    seqs.insert(s);
                                }
                            }
                        }
                    }
                }
                let mut out = Vec::new();
                for s in seqs.into_iter().rev().take(count) {
                    if let Some(cm) = load_commit(self.st, s) {
                        out.push(cm.resp(self.st));
                    }
                }
                Ok(Resp::Array(out))
            }
            "VC.RESTORE" => {
                if n < 2 {
                    return Err(wrong());
                }
                let anc = self.anc_of(&a[0])?;
                let mut restored = 0;
                for u in &a[1..] {
                    let state = user_state_at(self.st, u, &anc);
                    if live_state(self.st, u) != state {
                        write_user(self.st, u, &state);
                        restored += 1;
                    }
                }
                if restored > 0 {
                    self.reload = true;
                }
                Ok(Resp::Int(restored))
            }
            "VC.RESET" => {
                if n == 0 || n > 2 {
                    return Err(wrong());
                }
                let hard = match a.get(1) {
                    None => false,
                    Some(m) if m.eq_ignore_ascii_case(b"HARD") => true,
                    Some(m) if m.eq_ignore_ascii_case(b"SOFT") => false,
                    _ => return Err(err("syntax error")),
                };
                let (b, _, anc) = self.head()?;
                let tseq = resolve(self.st, &a[0])?;
                let tanc = ancestors(self.st, tseq);
                let moved = changed_keys(self.st, anc.symmetric_difference(&tanc).copied());
                if hard {
                    let mut keys = moved;
                    keys.extend(self.changes(&anc).into_iter().map(|(k, _)| k));
                    self.materialize(keys, &tanc);
                    self.settle = Some(self.epoch);
                } else {
                    // Working copy unchanged: what moved is now "uncommitted".
                    self.vcs.mark(moved);
                }
                self.st.put(k(&[b"b:", &b]), Value::Int(tseq as i64));
                let id = load_commit(self.st, tseq).map(|c| c.id).unwrap_or_default();
                Ok(Resp::Bulk(id.into_bytes()))
            }
            "VC.MERGE" => self.merge(a),
            "VC.REVERT" => {
                if n == 0 {
                    return Err(wrong());
                }
                let (b, seq, anc) = self.head()?;
                self.require_clean(&anc)?;
                let cseq = resolve(self.st, &a[0])?;
                let c = load_commit(self.st, cseq).ok_or_else(|| bad_ref("commit"))?;
                let parent = *c.parents.first().ok_or_else(|| err("cannot revert the root commit"))?;
                if !anc.contains(&cseq) {
                    return Err(err("that commit is not in the current branch's history"));
                }
                let canc = ancestors(self.st, cseq);
                let panc = ancestors(self.st, parent);
                let users = group_by_user(changed_keys(self.st, [cseq]));
                let mut writes = Vec::new();
                let mut conflicts = Vec::new();
                for u in users {
                    let at_c = user_state_at(self.st, &u, &canc);
                    let at_p = user_state_at(self.st, &u, &panc);
                    let cur = user_state_at(self.st, &u, &anc);
                    if cur == at_c {
                        writes.push((u, at_p));
                    } else if cur != at_p {
                        conflicts.push(Resp::Bulk(u));
                    }
                }
                if !conflicts.is_empty() {
                    return Err(err(&format!("revert conflicts on {} key(s) changed again since that commit", conflicts.len())));
                }
                let changes = self.apply_users(&writes);
                if changes.is_empty() {
                    return Err(err("nothing to revert"));
                }
                let msg = format!("Revert \"{}\" ({})", String::from_utf8_lossy(&c.msg), c.id);
                let nc = write_commit(self.st, &b, vec![seq], &changes, b"", msg.as_bytes(), self.now);
                self.settle = Some(self.epoch);
                Ok(Resp::Bulk(nc.id.into_bytes()))
            }
            _ => Err(err(&format!("unknown command '{}'", name.to_lowercase()))),
        }
    }

    /// Write user-key states to the working copy; returns the engine-key
    /// changes (for a commit).
    fn apply_users(&mut self, writes: &[(Vec<u8>, Vec<(Vec<u8>, Value)>)]) -> Vec<(Vec<u8>, Option<Value>)> {
        let mut changes = Vec::new();
        for (u, state) in writes {
            let before = live_state(self.st, u);
            if before == *state {
                continue;
            }
            for (ek, _) in &before {
                if !state.iter().any(|(k2, _)| k2 == ek) {
                    changes.push((ek.clone(), None));
                }
            }
            for (ek, v) in state {
                changes.push((ek.clone(), Some(v.clone())));
            }
            write_user(self.st, u, state);
        }
        if !changes.is_empty() {
            self.reload = true;
        }
        changes
    }

    fn merge(&mut self, a: &[Vec<u8>]) -> R<Resp> {
        if a.is_empty() {
            return Err(err("wrong number of arguments for 'vc.merge' command"));
        }
        let (mut strategy, mut dry, mut msg, mut i) = (None, false, None, 1);
        while i < a.len() {
            match arg_str(&a[i]).as_str() {
                "STRATEGY" => {
                    let s = a.get(i + 1).map(|v| arg_str(v)).ok_or_else(|| err("syntax error"))?;
                    if s != "OURS" && s != "THEIRS" {
                        return Err(err("STRATEGY must be OURS or THEIRS"));
                    }
                    strategy = Some(s);
                    i += 2;
                }
                "DRYRUN" => {
                    dry = true;
                    i += 1;
                }
                "MESSAGE" => {
                    msg = Some(a.get(i + 1).ok_or_else(|| err("syntax error"))?.clone());
                    i += 2;
                }
                _ => return Err(err("syntax error")),
            }
        }
        let (b, ours, oanc) = self.head()?;
        self.require_clean(&oanc)?;
        let theirs = resolve(self.st, &a[0])?;
        let tanc = ancestors(self.st, theirs);
        if oanc.contains(&theirs) {
            return Ok(Resp::Array(vec![bulk("result"), bulk("up-to-date")]));
        }
        let report = |result: &str, taken: usize, conflicts: Vec<Resp>, id: Option<String>| {
            Resp::Array(vec![
                bulk("result"),
                bulk(result),
                bulk("keys_merged"),
                Resp::Int(taken as i64),
                bulk("conflicts"),
                Resp::Array(conflicts),
                bulk("commit"),
                id.map_or(Resp::Nil, |s| Resp::Bulk(s.into_bytes())),
            ])
        };
        if tanc.contains(&ours) {
            // Fast-forward: theirs already contains everything we have.
            let keys = changed_keys(self.st, tanc.difference(&oanc).copied());
            let users = group_by_user(keys.iter().cloned()).len();
            if dry {
                return Ok(report("fast-forward", users, vec![], None));
            }
            self.materialize(keys, &tanc);
            self.st.put(k(&[b"b:", &b]), Value::Int(theirs as i64));
            self.settle = Some(self.epoch);
            let id = load_commit(self.st, theirs).map(|c| c.id);
            return Ok(report("fast-forward", users, vec![], id));
        }
        // Merge base: the newest common ancestor.
        let base = *oanc.intersection(&tanc).max().ok_or_else(|| err("no common ancestor"))?;
        let banc = ancestors(self.st, base);
        let ours_users = group_by_user(changed_keys(self.st, oanc.difference(&banc).copied()));
        let theirs_users = group_by_user(changed_keys(self.st, tanc.difference(&banc).copied()));
        let mut writes = Vec::new();
        let mut conflicts = Vec::new();
        for u in &theirs_users {
            let t = user_state_at(self.st, u, &tanc);
            let bs = user_state_at(self.st, u, &banc);
            if t == bs {
                continue; // changed and changed back on their side
            }
            if !ours_users.contains(u) {
                writes.push((u.clone(), t));
                continue;
            }
            let o = user_state_at(self.st, u, &oanc);
            if o == t {
                continue; // same change on both sides
            }
            if o == bs {
                writes.push((u.clone(), t));
                continue;
            }
            match strategy.as_deref() {
                Some("THEIRS") => writes.push((u.clone(), t)),
                Some(_) => {} // keep ours (recorded below)
                None => conflicts.push(Resp::Array(vec![Resp::Bulk(u.clone()), bulk(kind_of(&bs)), bulk(kind_of(&o)), bulk(kind_of(&t))])),
            }
        }
        if !conflicts.is_empty() {
            // Nothing is changed: resolve by re-running with a STRATEGY, or
            // by fixing the keys on one side and committing first.
            return Ok(report("conflict", writes.len(), conflicts, None));
        }
        if dry {
            return Ok(report("merge", writes.len(), vec![], None));
        }
        self.apply_users(&writes);
        // Record, in the merge commit, the final state of every key their
        // side changed, over every engine key either side ever wrote for it.
        // A read at the merge commit takes the newest history entry among
        // its ancestors; without this it could fall through to whichever
        // side happened to commit last instead of the merged result.
        let mut changes = Vec::new();
        for u in &theirs_users {
            let fin = live_state(self.st, u);
            let mut eks: BTreeSet<Vec<u8>> = fin.iter().map(|(e, _)| e.clone()).collect();
            for anc in [&oanc, &tanc, &banc] {
                eks.extend(user_state_at(self.st, u, anc).into_iter().map(|(e, _)| e));
            }
            for ek in eks {
                let v = fin.iter().find(|(e, _)| *e == ek).map(|(_, v)| v.clone());
                changes.push((ek, v));
            }
        }
        let theirs_id = load_commit(self.st, theirs).map(|c| c.id).unwrap_or_default();
        let msg = msg.unwrap_or_else(|| format!("Merge {} ({}) into {}", String::from_utf8_lossy(&a[0]), theirs_id, String::from_utf8_lossy(&b)).into_bytes());
        let c = write_commit(self.st, &b, vec![ours, theirs], &changes, b"", &msg, self.now);
        self.settle = Some(self.epoch);
        Ok(report("merge", writes.len(), vec![], Some(c.id)))
    }
}
