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
    /// In-memory salience mirror: id → salience. Recall's scoring loop needs
    /// per-doc salience for every candidate that matched any keyword — that
    /// was fetching `Meta` from the substrate per candidate (3 engine.get()
    /// calls each). Caching salience makes scoring zero-substrate-read; the
    /// full Meta (source, lineage) is only fetched for the FINAL top-k that
    /// gets returned. Drops recall latency ~10× at N=2k+ memories.
    salience_cache: std::sync::RwLock<HashMap<u64, f32>>,
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
        // resume id counter + live count + salience cache from persisted LTM
        let mut max = 0u64;
        let mut count = 0u64;
        let mut sal_cache: HashMap<u64, f32> = HashMap::new();
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
                        sal_cache.insert(id, m.salience);
                    }
                }
            }
        }
        Self {
            engine,
            vectors,
            next_id: std::sync::atomic::AtomicU64::new(max + 1),
            ltm_count: std::sync::atomic::AtomicU64::new(count),
            ep_seq: std::sync::atomic::AtomicU64::new(0),
            salience_cache: std::sync::RwLock::new(sal_cache),
        }
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
        self.salience_cache.write().unwrap().clear();
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
        let id = self.alloc_id();
        let ts = self.engine.now();
        let vf = if valid_from == 0 { ts } else { valid_from };
        self.engine
            .put(ltm_key(id), Value::Bytes(text.as_bytes().to_vec()))?;
        self.engine.put(
            meta_key(id),
            Value::Bytes(encode_meta(&Meta {
                source: source.to_string(),
                created_ts: ts,
                salience,
                lineage: lineage.to_string(),
                valid_from: vf,
                valid_to,
                owner: owner.to_string(),
            })),
        )?;
        self.vectors.insert(id, vector.clone())?;
        // keyword index
        for tok in tokenize(text) {
            self.engine.put(kw_key(&tok, id), Value::Int(1))?;
        }
        self.ltm_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // populate salience cache so recall scoring is zero-substrate-read
        self.salience_cache.write().unwrap().insert(id, salience);
        Ok(id)
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
        }
        Ok(())
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
            self.salience_cache.write().unwrap().remove(&id);
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
    pub fn recall_scoped(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Vec<RecallHit> {
        let fetch = k * 3;
        // Fast salience lookup (RwLock<read>, held briefly).
        let sal_cache = self.salience_cache.read().unwrap();
        let sal_of = |id: u64| -> f32 {
            *sal_cache.get(&id).unwrap_or(&0.5)
        };

        // 1) semantic ANN
        let mut fused: HashMap<u64, (f32, &'static str)> = HashMap::new();
        for (id, dist) in self.vectors.search(query_vec, fetch) {
            let sim = 1.0 - dist / 2.0;
            let score = sim * (0.5 + 0.5 * sal_of(id));
            fused.insert(id, (score, "semantic"));
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
                    let add = contrib_per_doc * 0.8 * (0.5 + 0.5 * sal_of(id));
                    fused
                        .entry(id)
                        .and_modify(|e| e.0 += add)
                        .or_insert((add, "keyword"));
                }
            }
        }
        drop(sal_cache);

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
        //    owner scope. Legacy (owner == "") stays visible to everyone.
        let mut out: Vec<RecallHit> = Vec::with_capacity(k);
        for s in top {
            if out.len() == k {
                break;
            }
            if let Some((text, meta)) = self.ltm_get_light(s.id) {
                if !meta.owner.is_empty() && meta.owner != scope {
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
            self.salience_cache.write().unwrap().insert(id, new_sal);
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
        // Over-fetch, then filter by the fact's validity window.
        self.recall_scoped(scope, query, query_vec, k * 4)
            .into_iter()
            .filter(|h| {
                let vf_ok = h.meta.valid_from == 0 || h.meta.valid_from <= as_of;
                let vt_ok = h.meta.valid_to == 0 || h.meta.valid_to > as_of;
                vf_ok && vt_ok
            })
            .take(k)
            .collect()
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
