//! Agent Memory Engine — "the #1 agent memory + RAG core".
//!
//! One engine, four memory structures, all on the unified MVCC+WAL substrate:
//!
//!   * Working memory  (STM)  — hot per-agent context, TTL-cleared. Key: `mem:wm:<agent>:<k>`
//!   * Long-term memory (LTM)  — durable semantic store.        Key: `mem:ltm:<id>`
//!   * Episodic memory        — append-only event log.           Key: `mem:ep:<agent>:<seq>`
//!   * Keyword index          — BM25-ish inverted index over LTM text. Key: `mem:kw:<tok>:<id>`
//!
//! LTM vectors are also pushed into the shared HNSW so semantic recall is a
//! single ANN hop. Every LTM entry carries provenance/lineage (who wrote it,
//! when, from what trigger) — the non-malleable authority hardening that 2026
//! agent-memory research (MemLineage, poisoning defenses) demands and that
//! Mem0/Zep/Letta do not ship as a primitive.
//!
//! Pure Rust. Zero external crates. Durable by construction (WAL-backed).

use std::collections::HashMap;
use std::sync::Arc;

use storage::{Engine, Value};
use views::VectorIndex;

// ── key conventions ────────────────────────────────────────────────────────
fn wm_key(agent: &str, k: &str) -> Vec<u8> {
    let mut b = b"mem:wm:".to_vec();
    b.extend_from_slice(agent.as_bytes());
    b.push(b':');
    b.extend_from_slice(k.as_bytes());
    b
}
fn wm_exp_key(agent: &str, k: &str) -> Vec<u8> {
    let mut b = b"mem:wmx:".to_vec();
    b.extend_from_slice(agent.as_bytes());
    b.push(b':');
    b.extend_from_slice(k.as_bytes());
    b
}
fn ltm_key(id: u64) -> Vec<u8> {
    let mut b = b"mem:ltm:".to_vec();
    b.extend_from_slice(&id.to_be_bytes());
    b
}
fn meta_key(id: u64) -> Vec<u8> {
    let mut b = b"mem:meta:".to_vec();
    b.extend_from_slice(&id.to_be_bytes());
    b
}
fn ep_key(agent: &str, seq: u64) -> Vec<u8> {
    let mut b = b"mem:ep:".to_vec();
    b.extend_from_slice(agent.as_bytes());
    b.push(b':');
    b.extend_from_slice(&seq.to_be_bytes());
    b
}
fn kw_key(token: &str, id: u64) -> Vec<u8> {
    let mut b = b"mem:kw:".to_vec();
    b.extend_from_slice(token.as_bytes());
    b.push(b':');
    b.extend_from_slice(&id.to_be_bytes());
    b
}
/// Forward graph edge: from -> to, tagged with a relation type.
/// Value is a f32 weight encoded as bytes. Layout: `mem:edge:<from>:<rel>:<to>`.
fn edge_key(from: u64, rel: &str, to: u64) -> Vec<u8> {
    let mut b = b"mem:edge:".to_vec();
    b.extend_from_slice(&from.to_be_bytes());
    b.push(b':');
    b.extend_from_slice(rel.as_bytes());
    b.push(b':');
    b.extend_from_slice(&to.to_be_bytes());
    b
}
/// Reverse edge index for backward traversal: `mem:redge:<to>:<rel>:<from>`.
fn redge_key(to: u64, rel: &str, from: u64) -> Vec<u8> {
    let mut b = b"mem:redge:".to_vec();
    b.extend_from_slice(&to.to_be_bytes());
    b.push(b':');
    b.extend_from_slice(rel.as_bytes());
    b.push(b':');
    b.extend_from_slice(&from.to_be_bytes());
    b
}
/// Procedural memory: named workflow / rule / pattern for an agent.
/// Layout: `mem:proc:<agent>:<name>`.
fn proc_key(agent: &str, name: &str) -> Vec<u8> {
    let mut b = b"mem:proc:".to_vec();
    b.extend_from_slice(agent.as_bytes());
    b.push(b':');
    b.extend_from_slice(name.as_bytes());
    b
}

// ── value-codec (self-describing, no serde) ─────────────────────────────────
// LTM meta is stored as Value::Bytes with a small fixed-format blob.
// v1: source_len, source, created_ts, salience, lineage
// v2: v1 + valid_from (u64) + valid_to (u64; 0 = open-ended)  ← bi-temporal
const META_VERSION_V1: u8 = 1;
/// v2 added bi-temporal valid_from/valid_to. v3 adds per-agent ownership
/// (`owner`) so LTM recall can be scoped — an agent must not recall another
/// agent's/user's memories. v1/v2 records decode with `owner = ""`, which is
/// treated as LEGACY/PUBLIC: visible to every scope until re-written.
const META_VERSION: u8 = 3;

fn encode_meta(m: &Meta) -> Vec<u8> {
    let mut out = vec![META_VERSION];
    let src = m.source.as_bytes();
    out.extend_from_slice(&(src.len() as u32).to_le_bytes());
    out.extend_from_slice(src);
    out.extend_from_slice(&m.created_ts.to_le_bytes());
    out.extend_from_slice(&m.salience.to_le_bytes());
    // v2 adds a fixed-width valid_from / valid_to *before* lineage so the tail
    // lineage decode stays trivial.
    out.extend_from_slice(&m.valid_from.to_le_bytes());
    out.extend_from_slice(&m.valid_to.to_le_bytes());
    // v3 adds the owning agent scope (length-prefixed), still before lineage.
    let own = m.owner.as_bytes();
    out.extend_from_slice(&(own.len() as u32).to_le_bytes());
    out.extend_from_slice(own);
    out.extend_from_slice(m.lineage.as_bytes());
    out
}
fn decode_meta(buf: &[u8]) -> Option<Meta> {
    let ver = *buf.first()?;
    if ver != META_VERSION && ver != META_VERSION_V1 && ver != 2 {
        return None;
    }
    let mut p = 1usize;
    let sl = u32::from_le_bytes(buf[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    let source = String::from_utf8(buf[p..p + sl].to_vec()).ok()?;
    p += sl;
    let created_ts = u64::from_le_bytes(buf[p..p + 8].try_into().ok()?);
    p += 8;
    let salience = f32::from_le_bytes(buf[p..p + 4].try_into().ok()?);
    p += 4;
    let (valid_from, valid_to) = if ver >= 2 {
        let vf = u64::from_le_bytes(buf[p..p + 8].try_into().ok()?);
        p += 8;
        let vt = u64::from_le_bytes(buf[p..p + 8].try_into().ok()?);
        p += 8;
        (vf, vt)
    } else {
        // v1 records: fact was true from creation, still valid.
        (created_ts, 0)
    };
    // v3: owner sits between the temporal fields and the lineage tail.
    let owner = if ver >= 3 {
        let ol = u32::from_le_bytes(buf[p..p + 4].try_into().ok()?) as usize;
        p += 4;
        let o = String::from_utf8(buf[p..p + ol].to_vec()).ok()?;
        p += ol;
        o
    } else {
        String::new()
    };
    let lineage = String::from_utf8(buf[p..].to_vec()).ok()?;
    Some(Meta { source, created_ts, salience, lineage, valid_from, valid_to, owner })
}
/// Provenance + importance + bi-temporal validity for one LTM entry.
///
/// The bi-temporal fields (`valid_from`, `valid_to`) match Zep Graphiti's
/// primitive: `valid_from` = when the fact became true in the world,
/// `valid_to` = when it stopped (0 = still valid). This lets recall answer
/// "what did the agent know as of time T" instead of "what does the agent
/// know now" — the difference between a stateful and a naive agent.
#[derive(Clone, Debug)]
pub struct Meta {
    pub source: String, // "agent:planner" | "user" | "tool:search" ...
    pub created_ts: u64,
    pub salience: f32, // 0..1 importance, drives consolidation + eviction
    pub lineage: String, // free-form derivation chain (utf8)
    pub valid_from: u64, // world-time the fact became true
    pub valid_to: u64,   // world-time it stopped (0 = still valid / open interval)
    /// Owning agent scope. Recall is filtered to `owner == scope` — an agent
    /// can never recall another agent's memories. Empty = LEGACY record from
    /// before scoping existed; those stay visible to every scope so existing
    /// corpora keep working (documented migration semantics, not a hole for
    /// anything written after this field shipped).
    pub owner: String,
}

/// A long-term memory record.
#[derive(Clone, Debug)]
pub struct LtmRecord {
    pub id: u64,
    pub text: String,
    pub vector: Vec<f32>,
    pub meta: Meta,
}

/// A working-memory entry (opaque bytes + TTL in logical-ms).
#[derive(Clone, Debug)]
pub struct WmEntry {
    pub value: Vec<u8>,
    pub expires_at: u64, // logical clock; 0 = no expiry
}

/// An episodic event.
#[derive(Clone, Debug)]
pub struct Episode {
    pub seq: u64,
    pub agent: String,
    pub kind: String,
    pub payload: Vec<u8>,
}

/// Recall result mixing semantic + keyword hits, ranked by a blended score.
#[derive(Clone, Debug)]
pub struct RecallHit {
    pub id: u64,
    pub text: String,
    pub score: f32,
    pub kind: &'static str, // "semantic" | "keyword"
    pub meta: Meta,
}

/// Minimum `1 - cosine_dist/2` for an ANN candidate to count as a semantic
/// hit (0.5 ⇔ cosine 0).
const MIN_SEMANTIC_SIM: f32 = 0.5;

/// The agent memory engine.
pub struct Memory {
    engine: Arc<Engine>,
    vectors: VectorIndex,
    next_id: std::sync::atomic::AtomicU64,
    /// Live count of LTM entries. Maintained by `ltm_store`/`ltm_forget` so BM25
    /// IDF (which needs N) is O(1) — was previously an O(N) prefix scan per
    /// recall, catastrophic at scale.
    ltm_count: std::sync::atomic::AtomicU64,
    /// Monotonic per-engine counter guaranteeing collision-free episodic seqs
    /// (EP-1): two `episode()` calls in the same millisecond can no longer
    /// generate identical keys and silently overwrite each other.
    ep_seq: std::sync::atomic::AtomicU64,
    /// In-memory record index: id → (salience, owner, validity window), plus
    /// one normalized-vector arena per owner. Recall scores candidates with
    /// zero substrate reads AND applies the owner scope + bi-temporal window
    /// BEFORE the top-k cut (see [`RecIndex`]). Full Meta (source, lineage)
    /// is fetched only for the final top-k that gets returned.
    idx: std::sync::RwLock<RecIndex>,
}

/// Per-record facts recall needs while scoring, kept in RAM.
#[derive(Clone, Copy, Debug)]
struct RecInfo {
    sal: f32,
    owner: u32,
    valid_from: u64,
    valid_to: u64,
}

impl RecInfo {
    fn valid_at(&self, at: u64) -> bool {
        (self.valid_from == 0 || self.valid_from <= at) && (self.valid_to == 0 || self.valid_to > at)
    }
}

/// Normalized vectors of ONE owner, flat, for exact per-scope search.
#[derive(Default)]
struct ScopeArena {
    ids: Vec<u64>,
    pos: HashMap<u64, usize>,
    vecs: Vec<f32>,
    dim: usize,
}

impl ScopeArena {
    fn push(&mut self, id: u64, v: &[f32]) {
        if v.is_empty() || (self.dim != 0 && v.len() != self.dim) || self.pos.contains_key(&id) {
            return;
        }
        self.dim = v.len();
        let mut n = v.to_vec();
        normalize(&mut n);
        self.pos.insert(id, self.ids.len());
        self.ids.push(id);
        self.vecs.extend_from_slice(&n);
    }
    fn remove(&mut self, id: u64) {
        let Some(i) = self.pos.remove(&id) else { return };
        let last = self.ids.len() - 1;
        if i != last {
            let moved = self.ids[last];
            self.ids[i] = moved;
            self.pos.insert(moved, i);
            let d = self.dim;
            let (head, tail) = self.vecs.split_at_mut(last * d);
            head[i * d..(i + 1) * d].copy_from_slice(&tail[..d]);
        }
        self.ids.pop();
        self.vecs.truncate(last * self.dim);
    }
}

/// Why this exists — the scoped-recall starvation bug: owner and validity
/// filtering used to run AFTER a global top-(3·k) cut over every agent's
/// memories, so a user whose memories sat behind a few thousand closer
/// memories of OTHER users (or behind their own superseded facts) got zero
/// hits. Filtering now happens during candidate generation; scopes of up to
/// `exact_max` vectors are searched exactly (perfect recall, and faster than
/// a graph walk at agent-memory sizes), bigger ones via a widened ANN pass.
#[derive(Default)]
struct RecIndex {
    owners: Vec<String>,
    owner_ids: HashMap<String, u32>,
    recs: HashMap<u64, RecInfo>,
    arenas: HashMap<u32, ScopeArena>,
}

impl RecIndex {
    fn intern(&mut self, owner: &str) -> u32 {
        if let Some(&i) = self.owner_ids.get(owner) {
            return i;
        }
        let i = self.owners.len() as u32;
        self.owners.push(owner.to_string());
        self.owner_ids.insert(owner.to_string(), i);
        i
    }
    fn insert(&mut self, id: u64, meta: &Meta, vector: Option<&[f32]>) {
        let owner = self.intern(&meta.owner);
        self.recs.insert(
            id,
            RecInfo { sal: meta.salience, owner, valid_from: meta.valid_from, valid_to: meta.valid_to },
        );
        if let Some(v) = vector {
            self.arenas.entry(owner).or_default().push(id, v);
        }
    }
    fn remove(&mut self, id: u64) {
        if let Some(info) = self.recs.remove(&id) {
            if let Some(a) = self.arenas.get_mut(&info.owner) {
                a.remove(id);
            }
        }
    }
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// 8-lane dot product; with target-cpu=native this lowers to packed FMA.
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Scope size up to which recall scores every vector exactly.
fn exact_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DBSTRIKE_MEM_EXACT_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000)
    })
}

impl Memory {
    /// Live count of LTM entries (resumed from the WAL on `open`).
    pub fn ltm_count(&self) -> u64 {
        self.ltm_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn open(engine: Arc<Engine>) -> Self {
        // LTM embeddings live in a RESERVED namespace (`vec:__ltm__:`), never
        // the user-facing default index. Sharing the default namespace meant
        // two separate in-RAM graphs (Router's and Memory's) over the same
        // `vec:` keys with independent dims — after a restart the rebuild
        // merged them, the first-inserted dim won, and mismatched-dim vectors
        // were silently dropped (a user VADD could vanish while a memory doc
        // answered in its place). The reserved prefix gives memory its own
        // graph, its own dim, and keeps VLISTNS clean.
        let vectors = VectorIndex::open_ns(Arc::clone(&engine), "__ltm__".to_string());
        let (max, count, idx) = Self::scan_ltm(&engine, &vectors);
        Self {
            engine,
            vectors,
            next_id: std::sync::atomic::AtomicU64::new(max + 1),
            ltm_count: std::sync::atomic::AtomicU64::new(count),
            ep_seq: std::sync::atomic::AtomicU64::new(0),
            idx: std::sync::RwLock::new(idx),
        }
    }

    /// Resume id counter + live count + salience cache from persisted LTM.
    fn scan_ltm(engine: &Engine, vectors: &VectorIndex) -> (u64, u64, RecIndex) {
        let mut max = 0u64;
        let mut count = 0u64;
        let mut idx = RecIndex::default();
        let snap = engine.snapshot();
        for (k, _) in engine.scan_prefix(b"mem:ltm:", snap) {
            if k.len() >= 8 {
                let id = u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap());
                if id > max {
                    max = id;
                }
                count += 1;
                // Repopulate salience so recall stays hot after restart.
                if let Some(Value::Bytes(mb)) = engine.get(&meta_key(id)) {
                    if let Some(m) = decode_meta(&mb) {
                        let v = vectors.get_vector(id);
                        idx.insert(id, &m, v.as_deref());
                    }
                }
            }
        }
        (max, count, idx)
    }

    /// Rebuild every RAM mirror (ids, count, salience, semantic graph) from
    /// the substrate — a replica after a full sync or new replicated writes.
    pub fn reload(&self) {
        self.vectors.reload();
        let (max, count, idx) = Self::scan_ltm(&self.engine, &self.vectors);
        self.next_id.store(max + 1, std::sync::atomic::Ordering::SeqCst);
        self.ltm_count.store(count, std::sync::atomic::Ordering::SeqCst);
        *self.idx.write().unwrap() = idx;
    }

    fn alloc_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Reset every in-RAM structure derived from durable LTM state: the id
    /// allocator, live count, episodic seq counter, salience mirror and the
    /// semantic HNSW graph. Called by FLUSHALL right after the engine wipe —
    /// the `mem:*` payloads are already gone at that point, so these mirrors
    /// must not keep reporting a ghost corpus (MEM.COUNT stuck at N, ids
    /// continuing past wiped records).
    pub fn reset_volatile(&self) {
        self.next_id.store(1, std::sync::atomic::Ordering::SeqCst);
        self.ep_seq.store(0, std::sync::atomic::Ordering::SeqCst);
        self.ltm_count.store(0, std::sync::atomic::Ordering::SeqCst);
        *self.idx.write().unwrap() = RecIndex::default();
        self.vectors.reset_ram();
    }

    // ── WORKING MEMORY (STM) ───────────────────────────────────────────────
    pub fn wm_set(
        &self,
        agent: &str,
        key: &str,
        value: &[u8],
        ttl_ms: u64,
        now: u64,
    ) -> std::io::Result<()> {
        let exp = if ttl_ms == 0 { 0 } else { now + ttl_ms };
        // WM-2: write value + expiry as a single atomic batch so a crash
        // between the two puts can never leave a permanent (orphaned) entry
        // with no expiry key.
        self.engine.put_batch(vec![
            (wm_key(agent, key), Value::Bytes(value.to_vec())),
            (wm_exp_key(agent, key), Value::Int(exp as i64)),
        ])?;
        Ok(())
    }

    pub fn wm_get(&self, agent: &str, key: &str, now: u64) -> Option<Vec<u8>> {
        if let Some(Value::Int(exp)) = self.engine.get(&wm_exp_key(agent, key)) {
            if exp != 0 && now >= exp as u64 {
                let _ = self.engine.delete(wm_key(agent, key));
                let _ = self.engine.delete(wm_exp_key(agent, key));
                return None;
            }
        }
        match self.engine.get(&wm_key(agent, key)) {
            Some(Value::Bytes(b)) => Some(b),
            // WM-3: a value deleted externally (DEL / compaction) leaves an
            // orphaned expiry key behind — clean it up to avoid a slow leak.
            _ => {
                let _ = self.engine.delete(wm_exp_key(agent, key));
                None
            }
        }
    }

    // WM-1: proactive eviction before TTL expiry.
    pub fn wm_delete(&self, agent: &str, key: &str) -> std::io::Result<()> {
        let _ = self.engine.delete(wm_key(agent, key));
        let _ = self.engine.delete(wm_exp_key(agent, key));
        Ok(())
    }

    // ── LONG-TERM MEMORY (semantic) ────────────────────────────────────────
    pub fn ltm_store(
        &self,
        text: &str,
        vector: Vec<f32>,
        source: &str,
        salience: f32,
        lineage: &str,
        owner: &str,
    ) -> std::io::Result<u64> {
        self.ltm_store_temporal(text, vector, source, salience, lineage, 0, 0, owner)
    }

    /// Bi-temporal store: `valid_from` = when the fact became true (0 = now);
    /// `valid_to` = when it stopped (0 = open-ended / still valid).
    pub fn ltm_store_temporal(
        &self,
        text: &str,
        vector: Vec<f32>,
        source: &str,
        salience: f32,
        lineage: &str,
        valid_from: u64,
        valid_to: u64,
        owner: &str,
    ) -> std::io::Result<u64> {
        // Validate the embedding BEFORE any write. The text/meta records are
        // separate puts, so a vector rejected later would strand a half-
        // written memory; and a NaN or wrong-dim vector stored here used to
        // be returned as a hit for every later query.
        self.check_embedding(&vector)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let id = self.alloc_id();
        let ts = self.engine.now();
        let vf = if valid_from == 0 { ts } else { valid_from };
        self.engine
            .put(ltm_key(id), Value::Bytes(text.as_bytes().to_vec()))?;
        let meta = Meta {
            source: source.to_string(),
            created_ts: ts,
            salience,
            lineage: lineage.to_string(),
            valid_from: vf,
            valid_to,
            owner: owner.to_string(),
        };
        self.engine.put(meta_key(id), Value::Bytes(encode_meta(&meta)))?;
        self.vectors.insert(id, vector.clone())?;
        // keyword index
        for tok in tokenize(text) {
            self.engine.put(kw_key(&tok, id), Value::Int(1))?;
        }
        self.ltm_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // index the record so recall scoring + filtering is zero-substrate-read
        self.idx.write().unwrap().insert(id, &meta, Some(&vector));
        Ok(id)
    }

    /// Validate an embedding (ingest or query) against the LTM index: finite
    /// components and the index's native dimensionality.
    pub fn check_embedding(&self, v: &[f32]) -> Result<(), String> {
        if v.iter().any(|x| !x.is_finite()) {
            return Err("embedding contains NaN or infinity".into());
        }
        self.vectors.check_dim(v.len())
    }

    /// Invalidate a fact "as of" a world-time. Used to supersede without
    /// deleting — the fact remains visible for `as_of` queries earlier than
    /// `invalid_at`, but recall at "now" hides it.
    pub fn ltm_invalidate(&self, id: u64, invalid_at: u64) -> std::io::Result<()> {
        if let Some(rec) = self.ltm_get(id) {
            let mut m = rec.meta;
            m.valid_to = invalid_at;
            self.engine
                .put(meta_key(id), Value::Bytes(encode_meta(&m)))?;
            self.set_valid_to(id, invalid_at);
        }
        Ok(())
    }

    /// Owner-scoped invalidate: the wire-facing form of [`Self::ltm_invalidate`].
    ///
    /// Succeeds only when `requester` matches the fact's `Meta.owner`. Facts
    /// stored without an agent scope carry owner `"default"`, so a bare
    /// `MEM.INVALIDATE` keeps working on them while remaining unable to touch
    /// any `[AGENT name]`-scoped fact. Unknown ids are an error, not a silent
    /// OK — a typo'd id must not look like a successful supersede.
    ///
    /// The unscoped [`Self::ltm_invalidate`] stays for trusted in-process
    /// callers (the bench) and must not be exposed to the wire.
    pub fn ltm_invalidate_scoped(
        &self,
        id: u64,
        invalid_at: u64,
        requester: &str,
    ) -> std::io::Result<()> {
        let rec = self.ltm_get(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("ERR no such fact {id}"),
            )
        })?;
        if rec.meta.owner != requester {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "ERR fact {id} owned by '{}' — requester '{}' denied",
                    rec.meta.owner, requester
                ),
            ));
        }
        let mut m = rec.meta;
        m.valid_to = invalid_at;
        self.engine
            .put(meta_key(id), Value::Bytes(encode_meta(&m)))?;
        self.set_valid_to(id, invalid_at);
        Ok(())
    }

    fn set_valid_to(&self, id: u64, at: u64) {
        if let Some(r) = self.idx.write().unwrap().recs.get_mut(&id) {
            r.valid_to = at;
        }
    }

    /// Owner-scoped forget: like [`Self::ltm_forget`] but denied unless
    /// `requester` owns the fact — the wire form for `MEM.FORGET AGENT n id`.
    pub fn ltm_forget_scoped(&self, id: u64, requester: &str) -> std::io::Result<()> {
        let rec = self.ltm_get_light(id).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("ERR no such fact {id}"))
        })?;
        if rec.1.owner != requester {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("ERR fact {id} owned by '{}' — requester '{}' denied", rec.1.owner, requester),
            ));
        }
        self.ltm_forget(id)
    }

    pub fn ltm_get(&self, id: u64) -> Option<LtmRecord> {
        let text = match self.engine.get(&ltm_key(id)) {
            Some(Value::Bytes(b)) => String::from_utf8_lossy(&b).to_string(),
            _ => return None,
        };
        let meta = match self.engine.get(&meta_key(id)) {
            Some(Value::Bytes(b)) => decode_meta(&b)?,
            _ => return None,
        };
        let vector = self.vectors.get_vector(id).unwrap_or_default();
        Some(LtmRecord { id, text, vector, meta })
    }

    pub fn ltm_forget(&self, id: u64) -> std::io::Result<()> {
        let existed = self.ltm_get(id);
        if let Some(rec) = &existed {
            for tok in tokenize(&rec.text) {
                let _ = self.engine.delete(kw_key(&tok, id));
            }
        }
        self.vectors.forget(id);
        self.engine.delete(ltm_key(id))?;
        self.engine.delete(meta_key(id))?;
        if existed.is_some() {
            // Only decrement when we actually removed a live record. Guard the
            // saturating floor so a stray forget can't wrap the counter.
            let _ = self.ltm_count.fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |n| if n == 0 { None } else { Some(n - 1) },
            );
            self.idx.write().unwrap().remove(id);
        }
        Ok(())
    }

    // ── EPISODIC ───────────────────────────────────────────────────────────
    /// Append an episode. Returns a globally-unique, monotonic `seq`.
    ///
    /// EP-1 fix: `seq` is `now() << 22 | counter`, so two calls in the same
    /// millisecond get distinct counters instead of clobbering the same key.
    /// The high bits keep wall-clock ordering; the low bits guarantee
    /// uniqueness without a per-agent metadata round-trip.
    pub fn episode(&self, agent: &str, kind: &str, payload: &[u8]) -> std::io::Result<u64> {
        let ts = self.engine.now();
        let c = self.ep_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let seq = (ts << 22) | (c & 0x3F_FFFF);
        // EP-2 fix: store kind inline in the value (kind_len || kind || payload)
        // so `episodes()` is a single prefix scan with zero per-episode reads.
        let mut val = Vec::with_capacity(2 + kind.len() + payload.len());
        let klen = kind.len() as u16;
        val.extend_from_slice(&klen.to_be_bytes());
        val.extend_from_slice(kind.as_bytes());
        val.extend_from_slice(payload);
        self.engine.put(ep_key(agent, seq), Value::Bytes(val))?;
        Ok(seq)
    }

    pub fn episodes(&self, agent: &str, limit: usize) -> Vec<Episode> {
        let prefix = {
            let mut b = b"mem:ep:".to_vec();
            b.extend_from_slice(agent.as_bytes());
            b.push(b':');
            b
        };
        let mut out = Vec::new();
        for (k, v) in self.engine.scan_prefix(&prefix, self.engine.snapshot()) {
            // EP-3 fix: require exactly 8 bytes of seq AFTER the prefix.
            if k.len() < prefix.len() + 8 {
                continue;
            }
            let seq = u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap());
            let payload = match v {
                Value::Bytes(b) => b,
                _ => continue,
            };
            // EP-2: split inline kind from payload (kind_len || kind || payload)
            let (kind, body) = if payload.len() >= 2 {
                let klen = u16::from_be_bytes([payload[0], payload[1]]) as usize;
                if payload.len() >= 2 + klen {
                    (
                        String::from_utf8_lossy(&payload[2..2 + klen]).to_string(),
                        payload[2 + klen..].to_vec(),
                    )
                } else {
                    (String::new(), payload[2..].to_vec())
                }
            } else {
                (String::new(), payload)
            };
            out.push(Episode {
                seq,
                agent: agent.to_string(),
                kind,
                payload: body,
            });
        }
        out.sort_by_key(|e| e.seq);
        out.truncate(limit);
        out
    }

    /// EP-4: evict a single episode.
    pub fn episode_forget(&self, agent: &str, seq: u64) -> std::io::Result<()> {
        let _ = self.engine.delete(ep_key(agent, seq));
        Ok(())
    }

    /// EP-4: bulk-clear all episodes for an agent.
    pub fn episodes_clear(&self, agent: &str) -> std::io::Result<()> {
        let prefix = {
            let mut b = b"mem:ep:".to_vec();
            b.extend_from_slice(agent.as_bytes());
            b.push(b':');
            b
        };
        for (k, _) in self.engine.scan_prefix(&prefix, self.engine.snapshot()) {
            let _ = self.engine.delete(k);
        }
        Ok(())
    }

    // ── RECALL (blended semantic + keyword, OWNER-SCOPED) ──────────────────
    //
    // Fast path: score ALL candidates using only the in-memory salience cache
    // (zero substrate reads during scoring). Fetch text + meta from the
    // substrate ONLY for the final top-k that gets returned. At N=2k memories
    // with common query tokens this went from ~2.3 ms → ~200 µs.
    //
    // SECURITY: `scope` is the requesting agent identity. Hits whose owner
    // differs are dropped BEFORE the top-k truncation (with a 3× overfetch so
    // a scoped recall still fills k). Legacy records (`owner == ""`, written
    // before scoping existed) remain visible to every scope — documented
    // migration semantics.
    /// World-time "now" for validity evaluation.
    ///
    /// TIME-DOMAIN CONTRACT (decided 2026-08-24 after the memgent breakage):
    /// valid_from/valid_to are WORLD times — callers use unix milliseconds;
    /// benchmarks may use small synthetic values. The engine's logical clock
    /// ticks from ~0, so evaluating windows against it alone judges every
    /// wall-clock-dated fact "not yet valid" and silently hides it. Taking
    /// max(logical, wall-clock-ms) makes both dialects behave: synthetic
    /// windows (≤ logical range) resolve against the clock, real dates
    /// resolve against the wall. AS_OF queries stay caller-supplied verbatim
    /// — the caller's world, the caller's time.
    fn effective_now(&self) -> u64 {
        let logical = self.engine.snapshot();
        let wall_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        logical.max(wall_ms)
    }

    pub fn recall_scoped(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Vec<RecallHit> {
        // NOT engine.snapshot() alone — see effective_now(). snapshot() also
        // has the virtue of not advancing the logical clock, which now()
        // would; wall-clock reading is side-effect-free too.
        self.recall_at(scope, query, query_vec, k, self.effective_now())
    }

    /// The single recall pipeline. `at` is the world-time the validity
    /// window is evaluated against: current logical time for "now" recall,
    /// caller-supplied for time-travel. Applying the window INSIDE the
    /// materialization loop is what keeps AS_OF correct — filtering after a
    /// now-scoped top-k would resurrect nothing that the present has already
    /// superseded.
    fn recall_at(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
        at: u64,
    ) -> Vec<RecallHit> {
        let fetch = k * 3;
        // One read lock for the whole scoring phase: salience, owner and the
        // bi-temporal window of every candidate come from RAM, and records
        // the requester cannot see are dropped BEFORE the top-k cut.
        let idx = self.idx.read().unwrap();
        let scope_id = idx.owner_ids.get(scope).copied();
        let legacy_id = idx.owner_ids.get("").copied();
        let visible = |id: u64| -> Option<f32> {
            let r = idx.recs.get(&id)?;
            let owned = Some(r.owner) == scope_id || Some(r.owner) == legacy_id;
            (owned && r.valid_at(at)).then_some(r.sal)
        };

        // 1) semantic: exact over the requester's own arena(s) when small
        //    enough, else a widened ANN pass filtered by `visible`.
        let mut fused: HashMap<u64, (f32, &'static str)> = HashMap::new();
        let arenas: Vec<&ScopeArena> = [scope_id, legacy_id]
            .into_iter()
            .flatten()
            .filter_map(|o| idx.arenas.get(&o))
            .collect();
        let scoped_n: usize = arenas.iter().map(|a| a.ids.len()).sum();
        // Relevance floor: sim <= 0.5 means cosine <= 0 — orthogonal or
        // opposite (and every match of a zero query vector). Those are not
        // evidence of relevance; admitting them made a query that matched
        // nothing still "find" every document by rank alone.
        let mut admit = |id: u64, sim: f32, fused: &mut HashMap<u64, (f32, &'static str)>| {
            if !(sim > MIN_SEMANTIC_SIM) {
                return;
            }
            if let Some(sal) = visible(id) {
                fused.insert(id, (sim * (0.5 + 0.5 * sal), "semantic"));
            }
        };
        if scoped_n > 0 && scoped_n <= exact_max() {
            let mut q = query_vec.to_vec();
            normalize(&mut q);
            // min-heap of the best `fetch` VISIBLE candidates (validity is
            // checked before a candidate can take a slot)
            #[derive(PartialEq)]
            struct C(f32, u64);
            impl Eq for C {}
            impl PartialOrd for C {
                fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(o))
                }
            }
            impl Ord for C {
                fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                    o.0.total_cmp(&self.0)
                }
            }
            let mut best: std::collections::BinaryHeap<C> = std::collections::BinaryHeap::with_capacity(fetch + 1);
            for a in &arenas {
                if a.dim != q.len() {
                    continue;
                }
                for (i, id) in a.ids.iter().enumerate() {
                    let cos = dot8(&q, &a.vecs[i * a.dim..(i + 1) * a.dim]);
                    if cos <= 0.0 || (best.len() == fetch && best.peek().is_some_and(|w| cos <= w.0)) {
                        continue;
                    }
                    if visible(*id).is_none() {
                        continue;
                    }
                    best.push(C(cos, *id));
                    if best.len() > fetch {
                        best.pop();
                    }
                }
            }
            for C(cos, id) in best {
                admit(id, (1.0 + cos) / 2.0, &mut fused);
            }
        } else if scoped_n > 0 {
            let total = idx.recs.len().max(1);
            let widen = (total / scoped_n).clamp(1, 64);
            let want = (fetch * widen).clamp(fetch, 4096);
            for (id, dist) in self.vectors.search_ef(query_vec, want, want.max(128)) {
                admit(id, 1.0 - dist / 2.0, &mut fused);
            }
        }

        // 2) keyword BM25 postings — accumulate scores WITHOUT substrate reads
        let qtokens = tokenize(query);
        let total_docs = self.ltm_count.load(std::sync::atomic::Ordering::SeqCst) as f32;
        let snap = self.engine.snapshot();
        for tok in &qtokens {
            let mut prefix = b"mem:kw:".to_vec();
            prefix.extend_from_slice(tok.as_bytes());
            prefix.push(b':');
            let matches = self.engine.scan_prefix(&prefix, snap);
            let df = matches.len() as u32;
            if df == 0 {
                continue;
            }
            let idf = ((total_docs - df as f32 + 0.5) / (df as f32 + 0.5))
                .ln()
                .max(0.0)
                + 1.0;
            let contrib_per_doc = (1.0f32 * 2.0 / (1.0 + 1.0)) * idf;
            for (kk, _) in matches {
                if kk.len() >= 8 {
                    let id = u64::from_be_bytes(kk[kk.len() - 8..].try_into().unwrap());
                    let Some(sal) = visible(id) else { continue };
                    let add = contrib_per_doc * 0.8 * (0.5 + 0.5 * sal);
                    fused
                        .entry(id)
                        .and_modify(|e| e.0 += add)
                        .or_insert((add, "keyword"));
                }
            }
        }
        drop(idx);

        // 3) Streaming top-k via a small max-heap (O(N log k), no full sort).
        // Capacity is the overfetch width (3×k): owner filtering happens at
        // materialization, so extra candidates keep scoped recall filled.
        use std::collections::BinaryHeap;
        #[derive(Copy, Clone, PartialEq)]
        struct Scored { score: f32, id: u64, kind: &'static str }
        impl Eq for Scored {}
        impl PartialOrd for Scored {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                // reversed so the smallest score is at the top of the max-heap
                o.score.partial_cmp(&self.score)
            }
        }
        impl Ord for Scored {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(fetch + 1);
        for (id, (score, kind)) in fused {
            if heap.len() < fetch {
                heap.push(Scored { score, id, kind });
            } else if let Some(worst) = heap.peek() {
                if score > worst.score {
                    heap.pop();
                    heap.push(Scored { score, id, kind });
                }
            }
        }
        let mut top: Vec<Scored> = heap.into_vec();
        top.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4) Materialize text + meta ONLY for the winners, enforcing the
        //    owner scope AND the bi-temporal validity window at `at`.
        //    Mirrors the documented supersede contract: a fact whose interval
        //    excludes `at` is invisible there, whatever the current time is.
        //    Legacy (owner == "") stays visible to everyone.
        let mut out: Vec<RecallHit> = Vec::with_capacity(k);
        for s in top {
            if out.len() == k {
                break;
            }
            if let Some((text, meta)) = self.ltm_get_light(s.id) {
                if !meta.owner.is_empty() && meta.owner != scope {
                    continue;
                }
                let vf_ok = meta.valid_from == 0 || meta.valid_from <= at;
                let vt_ok = meta.valid_to == 0 || meta.valid_to > at;
                if !(vf_ok && vt_ok) {
                    continue;
                }
                out.push(RecallHit { id: s.id, text, score: s.score, kind: s.kind, meta });
            }
        }
        out
    }

    /// Unscoped convenience wrapper — scope "default". Used by tests and
    /// callers that legitimately operate on the default agent pool.
    pub fn recall(&self, query: &str, query_vec: &[f32], k: usize) -> Vec<RecallHit> {
        self.recall_scoped("default", query, query_vec, k)
    }

    /// Consolidation hook: bump an entry's salience (importance).
    pub fn touch_salience(&self, id: u64, delta: f32) -> std::io::Result<()> {
        if let Some(mut rec) = self.ltm_get(id) {
            rec.meta.salience = (rec.meta.salience + delta).clamp(0.0, 1.0);
            let new_sal = rec.meta.salience;
            self.engine
                .put(meta_key(id), Value::Bytes(encode_meta(&rec.meta)))?;
            if let Some(r) = self.idx.write().unwrap().recs.get_mut(&id) {
                r.sal = new_sal;
            }
        }
        Ok(())
    }

    /// Lightweight LTM fetch — text + meta, NO vector. Used by recall's
    /// final top-k materialization path so we don't pull the whole vector
    /// (4·dim bytes per candidate) just to render a hit.
    pub fn ltm_get_light(&self, id: u64) -> Option<(String, Meta)> {
        let text = match self.engine.get(&ltm_key(id)) {
            Some(Value::Bytes(b)) => String::from_utf8_lossy(&b).to_string(),
            _ => return None,
        };
        let meta = match self.engine.get(&meta_key(id)) {
            Some(Value::Bytes(b)) => decode_meta(&b)?,
            _ => return None,
        };
        Some((text, meta))
    }

    /// Live count of LTM entries (O(1); replaces the previous O(N) scan).
    pub fn ltm_len(&self) -> usize {
        self.ltm_count.load(std::sync::atomic::Ordering::SeqCst) as usize
    }

    // ── GRAPH MEMORY (typed edges + multi-hop traversal) ───────────────────
    //
    // Mem0 GraphRAG / Zep Graphiti's core primitive: directional edges between
    // memories with a relation type and weight. Enables multi-hop reasoning
    // like "what did the user say about the deploy that failed?" without a
    // separate graph DB.

    /// Create/replace a directional edge from -> to with a relation type.
    /// Weight defaults to 1.0 if callers don't care.
    pub fn link(
        &self,
        from: u64,
        to: u64,
        rel: &str,
        weight: f32,
    ) -> std::io::Result<()> {
        let w = weight.to_le_bytes().to_vec();
        self.engine.put(edge_key(from, rel, to), Value::Bytes(w.clone()))?;
        self.engine.put(redge_key(to, rel, from), Value::Bytes(w))?;
        Ok(())
    }

    /// Remove a specific edge.
    pub fn unlink(&self, from: u64, to: u64, rel: &str) -> std::io::Result<()> {
        self.engine.delete(edge_key(from, rel, to))?;
        self.engine.delete(redge_key(to, rel, from))?;
        Ok(())
    }

    /// Outgoing neighbors of `from`. `rel` empty = all relations.
    /// Returns (neighbor_id, relation, weight).
    pub fn neighbors(&self, from: u64, rel: &str) -> Vec<(u64, String, f32)> {
        let mut prefix = b"mem:edge:".to_vec();
        prefix.extend_from_slice(&from.to_be_bytes());
        prefix.push(b':');
        if !rel.is_empty() {
            prefix.extend_from_slice(rel.as_bytes());
            prefix.push(b':');
        }
        self.engine
            .scan_prefix(&prefix, self.engine.snapshot())
            .into_iter()
            .filter_map(|(k, v)| {
                // key = mem:edge:<from-8>:<rel>:<to-8>
                if k.len() < 8 {
                    return None;
                }
                let to = u64::from_be_bytes(k[k.len() - 8..].try_into().ok()?);
                // extract relation between the two `:` separators
                let rel_start = 9 /*"mem:edge:"*/ + 8 /*from*/ + 1 /*":"*/;
                let rel_end = k.len() - 1 - 8; // strip ":<to-8>"
                if rel_end <= rel_start {
                    return None;
                }
                let rel_bytes = &k[rel_start..rel_end];
                let rel = String::from_utf8_lossy(rel_bytes).to_string();
                let w = match v {
                    Value::Bytes(b) if b.len() == 4 => {
                        f32::from_le_bytes(b.try_into().unwrap_or([0; 4]))
                    }
                    _ => 1.0,
                };
                Some((to, rel, w))
            })
            .collect()
    }

    /// Incoming neighbors of `to`. `rel` empty = all relations.
    pub fn incoming(&self, to: u64, rel: &str) -> Vec<(u64, String, f32)> {
        let mut prefix = b"mem:redge:".to_vec();
        prefix.extend_from_slice(&to.to_be_bytes());
        prefix.push(b':');
        if !rel.is_empty() {
            prefix.extend_from_slice(rel.as_bytes());
            prefix.push(b':');
        }
        self.engine
            .scan_prefix(&prefix, self.engine.snapshot())
            .into_iter()
            .filter_map(|(k, v)| {
                if k.len() < 8 {
                    return None;
                }
                let from = u64::from_be_bytes(k[k.len() - 8..].try_into().ok()?);
                let rel_start = 10 /*"mem:redge:"*/ + 8 + 1;
                let rel_end = k.len() - 1 - 8;
                if rel_end <= rel_start {
                    return None;
                }
                let rel_bytes = &k[rel_start..rel_end];
                let rel = String::from_utf8_lossy(rel_bytes).to_string();
                let w = match v {
                    Value::Bytes(b) if b.len() == 4 => {
                        f32::from_le_bytes(b.try_into().unwrap_or([0; 4]))
                    }
                    _ => 1.0,
                };
                Some((from, rel, w))
            })
            .collect()
    }

    /// Breadth-first traversal from `start`, following outgoing edges up to
    /// `max_depth` hops. Returns unique visited ids in visit order. Cheap
    /// substitute for a full graph DB; enough for RAG-style multi-hop recall.
    pub fn traverse(&self, start: u64, max_depth: usize, rel: &str) -> Vec<u64> {
        use std::collections::{HashSet, VecDeque};
        let mut seen: HashSet<u64> = HashSet::new();
        let mut order: Vec<u64> = Vec::new();
        let mut q: VecDeque<(u64, usize)> = VecDeque::new();
        q.push_back((start, 0));
        seen.insert(start);
        order.push(start);
        while let Some((cur, depth)) = q.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for (n, _, _) in self.neighbors(cur, rel) {
                if seen.insert(n) {
                    order.push(n);
                    q.push_back((n, depth + 1));
                }
            }
        }
        order
    }

    // ── BI-TEMPORAL RECALL ─────────────────────────────────────────────────

    /// Recall restricted to facts whose validity interval contains `as_of`.
    /// The Graphiti primitive: "what did we know at time T" instead of "now".
    pub fn recall_as_of(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
        as_of: u64,
    ) -> Vec<RecallHit> {
        // The window is applied at `as_of` INSIDE the pipeline — never as a
        // post-filter over a now-scoped top-k (that order would make every
        // past-superseded fact unreachable, breaking time travel).
        self.recall_at(scope, query, query_vec, k, as_of)
    }

    // ── PROCEDURAL MEMORY (Mem0's third pillar) ────────────────────────────
    //
    // Learned workflows / rules / patterns per agent. Not vector-indexed
    // (retrieval is by name, not similarity) — this is the "how to do things"
    // memory that plans and coding conventions live in.

    pub fn proc_store(&self, agent: &str, name: &str, body: &[u8]) -> std::io::Result<()> {
        self.engine
            .put(proc_key(agent, name), Value::Bytes(body.to_vec()))?;
        Ok(())
    }
    pub fn proc_get(&self, agent: &str, name: &str) -> Option<Vec<u8>> {
        match self.engine.get(&proc_key(agent, name)) {
            Some(Value::Bytes(b)) => Some(b),
            _ => None,
        }
    }
    pub fn proc_list(&self, agent: &str) -> Vec<String> {
        let mut prefix = b"mem:proc:".to_vec();
        prefix.extend_from_slice(agent.as_bytes());
        prefix.push(b':');
        self.engine
            .scan_prefix(&prefix, self.engine.snapshot())
            .into_iter()
            .filter_map(|(k, _)| {
                let s = String::from_utf8_lossy(&k).to_string();
                s.strip_prefix(&format!("mem:proc:{agent}:"))
                    .map(|x| x.to_string())
            })
            .collect()
    }
}

/// Tiny deterministic tokenizer: lowercase, split on non-alnum, drop stopwords.
pub fn tokenize(text: &str) -> Vec<String> {
    let stop: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is", "are", "was",
        "were", "it", "this", "that", "with", "as", "at", "by", "be", "from", "we", "you", "they",
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 3 && !stop.contains(s))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MEM.INVALIDATE must be owner-scoped: an agent may only invalidate
    /// facts it owns. Regression test for the cross-scope write hole
    /// (bare `MEM.INVALIDATE id at` let any agent kill any fact).
    #[test]
    fn ltm_invalidate_is_owner_scoped() {
        let m = Memory::open(eng());
        let v = vec![0.1f32; 8];
        let alice_fact = m
            .ltm_store_temporal("alice owns this fact", v.clone(), "t", 0.5, "x", 0, 0, "alice")
            .unwrap();
        let bob_fact = m
            .ltm_store_temporal("bob owns this fact", v.clone(), "t", 0.5, "x", 0, 0, "bob")
            .unwrap();
        let bare_fact = m
            .ltm_store_temporal("unscoped fact", v.clone(), "t", 0.5, "x", 0, 0, "default")
            .unwrap();

        // Cross-scope: bob must NOT be able to invalidate alice's fact.
        assert!(m.ltm_invalidate_scoped(alice_fact, 9999, "bob").is_err());
        // Bare ("default") requester must NOT touch a scoped agent's fact.
        assert!(m.ltm_invalidate_scoped(alice_fact, 9999, "default").is_err());
        // The denial must have left the fact intact and still valid at now.
        let rec = m.ltm_get(alice_fact).unwrap();
        assert_eq!(rec.meta.valid_to, 0, "denied invalidate mutated valid_to");

        // Owner may invalidate their own fact.
        m.ltm_invalidate_scoped(alice_fact, 5000, "alice").unwrap();
        assert_eq!(m.ltm_get(alice_fact).unwrap().meta.valid_to, 5000);

        // Unscoped facts stay manageable by the default scope only.
        assert!(m.ltm_invalidate_scoped(bare_fact, 1, "bob").is_err());
        m.ltm_invalidate_scoped(bare_fact, 100, "default").unwrap();
        assert_eq!(m.ltm_get(bare_fact).unwrap().meta.valid_to, 100);

        // Unknown ids are reported, not silently OK'd.
        assert!(m.ltm_invalidate_scoped(u64::MAX, 7, "alice").is_err());

        let _ = bob_fact;
    }

    /// Recall at "now" must apply the fact's validity window (documented
    /// supersede contract: "recall at 'now' hides it"), while AS_OF keeps
    /// time-travel intact.
    #[test]
    fn recall_applies_bitemporal_window_at_now() {
        let m = Memory::open(eng());
        let v = vec![0.2f32; 8];
        let q = "merge scheduled";
        let has = |hits: &[RecallHit], target| hits.iter().any(|h| h.id == target);

        let id = m
            .ltm_store_temporal("the merge is scheduled", v.clone(), "t", 0.9, "x", 0, 0, "alice")
            .unwrap();
        assert_eq!(
            m.recall_scoped("alice", q, &v, 5).len(),
            1,
            "fact must be visible before invalidation"
        );
        // Superseded at world-time 1 — strictly before every later now().
        m.ltm_invalidate_scoped(id, 1, "alice").unwrap();
        assert!(
            !has(&m.recall_scoped("alice", q, &v, 5), id),
            "invalidated fact leaked into now-recall"
        );

        // A FUTURE-dated supersede must NOT hide the fact early: valid_to is
        // world-time — expressed here in wall-clock ms per the time-domain
        // contract, since the logical clock sits below wall time.
        let id2 = m
            .ltm_store_temporal("the second merge is scheduled", v.clone(), "t", 0.9, "x", 0, 0, "alice")
            .unwrap();
        let wall_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let vt_future = wall_ms + 100_000;
        m.ltm_invalidate_scoped(id2, vt_future, "alice").unwrap();
        assert!(
            has(&m.recall_scoped("alice", q, &v, 50), id2),
            "future-dated supersede applied before its world-time"
        );

        // Time-travel leg: a fact superseded shortly after creation is gone
        // from the present but still reachable at its creation moment.
        let id3 = m
            .ltm_store_temporal("the third merge is scheduled", v.clone(), "t", 0.9, "x", 0, 0, "alice")
            .unwrap();
        let vf3 = m.ltm_get(id3).unwrap().meta.valid_from;
        let now3 = m.engine.snapshot();
        assert!(now3 > vf3, "store must advance the logical clock past valid_from");
        m.ltm_invalidate_scoped(id3, vf3 + 1, "alice").unwrap();
        assert!(
            !has(&m.recall_scoped("alice", q, &v, 50), id3),
            "superseded fact leaked into now-recall"
        );
        assert!(
            has(&m.recall_as_of("alice", q, &v, 50, vf3), id3),
            "AS_OF(valid_from) lost the fact"
        );
    }

    /// Regression for the memgent/LoCoMo ingest breakage: callers pass
    /// WALL-CLOCK unix-ms windows (1.677e12 ≈ Mar 2023), while the engine's
    /// logical clock ticks from ~0. "Now"-recall must evaluate validity
    /// against world time — max(logical, wall-clock-ms) — or every
    /// historically-dated fact is judged "not yet valid" and vanishes.
    #[test]
    fn recall_honours_unix_ms_windows_from_callers() {
        let m = Memory::open(eng());
        let v = vec![0.3f32; 8];
        let mar2023_ms: u64 = 1_677_628_800_000; // [2023-03-01] in unix-ms
        let id = m
            .ltm_store_temporal(
            "jon works at google",
                v.clone(),
                "conv",
                0.9,
                "x",
                mar2023_ms,
                0,
                "h1",
            )
            .unwrap();
        assert_eq!(m.ltm_get(id).unwrap().meta.valid_from, mar2023_ms);
        let hits = m.recall_scoped("h1", "jon google", &v, 5);
        assert!(
            hits.iter().any(|h| h.id == id),
            "unix-ms dated fact invisible at now — time-domain mismatch"
        );

        // Superseding it at any moment AFTER its window started hides it.
        m.ltm_invalidate_scoped(id, mar2023_ms + 1000, "h1").unwrap();
        assert!(!m.recall_scoped("h1", "jon google", &v, 5).iter().any(|h| h.id == id));
        // ...and AS_OF before the supersede resurrects it.
        assert!(m
            .recall_as_of("h1", "jon google", &v, 5, mar2023_ms)
            .iter()
            .any(|h| h.id == id));
    }

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_mem_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Engine::open(dir.join(format!("m_{n}.wal"))).unwrap()
    }

    #[test]
    fn ltm_recall_blended() {
        let m = Memory::open(eng());
        let v = |s: &str| {
            let mut v = vec![0f32; 8];
            for (i, c) in s.chars().enumerate() {
                v[i % 8] += (c as u32 % 7) as f32;
            }
            v
        };
        m.ltm_store(
            "the agent planned a deployment to production",
            v("deploy production plan"),
            "agent:planner",
            0.9,
            "trigger:goal",
            "default",
        )
        .unwrap();
        m.ltm_store(
            "user prefers rust over python for speed",
            v("rust speed preference"),
            "user",
            0.7,
            "trigger:chat",
            "default",
        )
        .unwrap();
        m.ltm_store(
            "cache returned a stale value during load test",
            v("cache stale bug"),
            "tool:monitor",
            0.5,
            "trigger:alert",
            "default",
        )
        .unwrap();

        let hits = m.recall("production deployment plan", &v("production deployment plan"), 3);
        assert!(!hits.is_empty());
        assert!(hits[0].text.contains("deployment"));
    }

    /// SECURITY: per-agent LTM recall scoping. Agent A must never recall
    /// agent B's memories, and the unscoped default pool must not leak either
    /// way. Legacy records (owner == "", from pre-scoping corpora) stay
    /// visible to every scope so existing data keeps working.
    #[test]
    fn ltm_recall_is_agent_scoped() {
        let m = Memory::open(eng());
        let v = |s: &str| -> Vec<f32> {
            // deterministic pseudo-embedding from tokens
            let mut o = vec![0f32; 8];
            for (i, t) in s.split_whitespace().enumerate() {
                let h = t.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
                o[i % 8] += (h % 1000) as f32 / 1000.0;
            }
            o
        };

        // alice's secret + bob's secret + one legacy record.
        m.ltm_store("alice aws key is AKIA-ALICE-123",
                    v("alice aws key secret credential"), "agent:alice", 0.9, "t", "alice").unwrap();
        m.ltm_store("bob gpg passphrase is hunter2-bob",
                    v("bob gpg passphrase secret credential"), "agent:bob", 0.9, "t", "bob").unwrap();
        m.ltm_store("legacy public onboarding doc",
                    v("legacy public onboarding doc"), "doc", 0.5, "t", "").unwrap();

        // Alice asks about credentials: sees hers + legacy, NEVER bob's.
        let a = m.recall_scoped("alice", "secret credential key",
                                &v("alice aws key secret credential"), 10);
        assert!(a.iter().any(|h| h.text.contains("AKIA-ALICE")),
                "alice must see her own memory");
        assert!(a.iter().any(|h| h.text.contains("onboarding")),
                "legacy records visible to all scopes");
        assert!(!a.iter().any(|h| h.text.contains("hunter2")),
                "SECURITY: alice recalled bob's memory!");

        // Bob symmetric.
        let b = m.recall_scoped("bob", "secret credential passphrase",
                                &v("bob gpg passphrase secret credential"), 10);
        assert!(b.iter().any(|h| h.text.contains("hunter2")));
        assert!(!b.iter().any(|h| h.text.contains("AKIA-ALICE")),
                "SECURITY: bob recalled alice's memory!");

        // Default pool sees only legacy — neither agent's writes leak into it.
        let d = m.recall_scoped("default", "secret credential",
                                &v("secret credential key passphrase"), 10);
        assert!(!d.iter().any(|h| h.text.contains("AKIA-ALICE")));
        assert!(!d.iter().any(|h| h.text.contains("hunter2")));

        // Meta round-trips the owner through encode/decode (v3).
        let rec = m.ltm_get(a.iter().find(|h| h.text.contains("AKIA")).unwrap().id).unwrap();
        assert_eq!(rec.meta.owner, "alice");
    }

    /// FLUSHALL must reset Memory's RAM mirrors, not just the substrate:
    /// MEM.COUNT must drop to 0 and the id allocator must restart at 1 —
    /// otherwise the engine is empty but the API reports a ghost corpus.
    #[test]
    fn reset_volatile_after_flush_clears_mirrors() {
        let e = eng();
        let m = Memory::open(Arc::clone(&e));
        let v = |s: &str| vec![s.len() as f32, 1.0, 0.5];
        m.ltm_store("fact one about rust", v("rust"), "doc", 0.9, "t", "default").unwrap();
        m.ltm_store("fact two about llvm", v("llvm"), "doc", 0.8, "t", "default").unwrap();
        assert_eq!(m.ltm_count(), 2);
        assert!(m.recall("rust", &v("rust"), 5).len() >= 1);

        // Exactly what the server's FLUSHALL does: engine wipe + mirror reset.
        e.flushall_with_backup().unwrap();
        m.reset_volatile();

        assert_eq!(m.ltm_count(), 0, "MEM.COUNT must be 0 after flush");
        assert!(m.recall("rust", &v("rust"), 5).is_empty(), "no ghost hits");
        // Id allocator restarts where a fresh engine starts: first id = 1.
        let id = m.ltm_store("fresh after flush", v("fresh"), "doc", 0.5, "t", "default").unwrap();
        assert_eq!(id, 1, "ids must restart at 1 after flush, got {id}");
        assert_eq!(m.ltm_count(), 1);
    }

    /// Legacy v2 meta blobs (no owner) still decode after the version bump —
    /// an old WAL must open without losing its memories.
    #[test]
    fn v2_meta_decodes_with_empty_owner() {
        // hand-encode a v2 blob: [ver=2][src][ts][sal][vf][vt][lineage]
        let mut blob = vec![2u8];
        let src = b"tool:cli";
        blob.extend_from_slice(&(src.len() as u32).to_le_bytes());
        blob.extend_from_slice(src);
        blob.extend_from_slice(&42u64.to_le_bytes());
        blob.extend_from_slice(&0.6f32.to_le_bytes());
        blob.extend_from_slice(&7u64.to_le_bytes());
        blob.extend_from_slice(&0u64.to_le_bytes());
        blob.extend_from_slice(b"op:rotate");
        let meta = decode_meta(&blob).expect("v2 blob must decode");
        assert_eq!(meta.source, "tool:cli");
        assert_eq!(meta.lineage, "op:rotate");
        assert_eq!(meta.valid_from, 7);
        assert_eq!(meta.owner, "");
    }

    /// Scoped recall must not be starved by OTHER scopes. Owner filtering
    /// used to run after a global top-(3·k) cut, so once a few thousand
    /// memories from other users sat closer to the query, a user with only
    /// three memories got zero hits back — every per-user benchmark
    /// (LoCoMo, LongMemEval, multi-tenant agents) silently lost recall.
    #[test]
    fn scoped_recall_is_not_starved_by_other_scopes() {
        let m = Memory::open(eng());
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..3000 {
            let mut v = q.clone();
            v[1] = 0.01 * (i % 7) as f32; // all very close to the query
            m.ltm_store(&format!("crowd memory {i}"), v, "t", 0.9, "x", "crowd").unwrap();
        }
        let mut alice = Vec::new();
        for i in 0..3 {
            // cos ≈ 0.7 with the query: relevant, but far behind the crowd
            let v = vec![1.0f32, 1.0, 0.1 * i as f32, 0.0, 0.0, 0.0, 0.0, 0.0];
            alice.push(m.ltm_store(&format!("alice private note {i}"), v, "t", 0.5, "x", "alice").unwrap());
        }
        let hits = m.recall_scoped("alice", "unrelated words", &q, 3);
        let got: Vec<u64> = hits.iter().map(|h| h.id).collect();
        for id in &alice {
            assert!(got.contains(id), "alice's memory {id} starved out; got {got:?}");
        }
        // superseded facts must not starve current ones either
        for i in 0..500 {
            let id = m.ltm_store(&format!("old fact {i}"), q.clone(), "t", 0.9, "x", "bob").unwrap();
            m.ltm_invalidate_scoped(id, 1, "bob").unwrap();
        }
        let cur = m.ltm_store("current fact", vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "t", 0.5, "x", "bob").unwrap();
        let hits = m.recall_scoped("bob", "zzz", &q, 1);
        assert_eq!(hits.first().map(|h| h.id), Some(cur), "current fact starved by superseded ones");
    }

    #[test]
    fn wm_ttl_expires() {
        let m = Memory::open(eng());
        m.wm_set("agent1", "ctx", b"hot", 100, 0).unwrap();
        assert_eq!(m.wm_get("agent1", "ctx", 50), Some(b"hot".to_vec()));
        assert_eq!(m.wm_get("agent1", "ctx", 200), None);
    }

    #[test]
    fn lineage_survives_roundtrip() {
        let m = Memory::open(eng());
        let id = m
            .ltm_store("secret key rotated", vec![1.0; 4], "tool:cli", 0.6, "op:rotate", "default")
            .unwrap();
        let rec = m.ltm_get(id).unwrap();
        assert_eq!(rec.meta.source, "tool:cli");
        assert_eq!(rec.meta.lineage, "op:rotate");
        assert_eq!(rec.meta.salience, 0.6);
        assert_eq!(rec.meta.owner, "default");
    }
}
