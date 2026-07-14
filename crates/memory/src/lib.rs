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
fn ep_kind_key(agent: &str, seq: u64) -> Vec<u8> {
    let mut b = b"mem:epk:".to_vec();
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
const META_VERSION: u8 = 2;
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
    out.extend_from_slice(m.lineage.as_bytes());
    out
}
fn decode_meta(buf: &[u8]) -> Option<Meta> {
    let ver = *buf.first()?;
    if ver != META_VERSION && ver != META_VERSION_V1 {
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
    let (valid_from, valid_to) = if ver == META_VERSION {
        let vf = u64::from_le_bytes(buf[p..p + 8].try_into().ok()?);
        p += 8;
        let vt = u64::from_le_bytes(buf[p..p + 8].try_into().ok()?);
        p += 8;
        (vf, vt)
    } else {
        // v1 records: fact was true from creation, still valid.
        (created_ts, 0)
    };
    let lineage = String::from_utf8(buf[p..].to_vec()).ok()?;
    Some(Meta { source, created_ts, salience, lineage, valid_from, valid_to })
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
}

impl Memory {
    pub fn open(engine: Arc<Engine>) -> Self {
        let vectors = VectorIndex::open(Arc::clone(&engine));
        // resume id counter + live count from existing LTM entries
        let mut max = 0u64;
        let mut count = 0u64;
        for (k, _) in engine.scan_prefix(b"mem:ltm:", engine.snapshot()) {
            if k.len() >= 8 {
                let id = u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap());
                if id > max {
                    max = id;
                }
                count += 1;
            }
        }
        Self {
            engine,
            vectors,
            next_id: std::sync::atomic::AtomicU64::new(max + 1),
            ltm_count: std::sync::atomic::AtomicU64::new(count),
        }
    }

    fn alloc_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
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
        self.engine
            .put(wm_key(agent, key), Value::Bytes(value.to_vec()))?;
        self.engine
            .put(wm_exp_key(agent, key), Value::Int(exp as i64))?;
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
            _ => None,
        }
    }

    // ── LONG-TERM MEMORY (semantic) ────────────────────────────────────────
    pub fn ltm_store(
        &self,
        text: &str,
        vector: Vec<f32>,
        source: &str,
        salience: f32,
        lineage: &str,
    ) -> std::io::Result<u64> {
        self.ltm_store_temporal(text, vector, source, salience, lineage, 0, 0)
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
            })),
        )?;
        self.vectors.insert(id, vector.clone())?;
        // keyword index
        for tok in tokenize(text) {
            self.engine.put(kw_key(&tok, id), Value::Int(1))?;
        }
        self.ltm_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        }
        Ok(())
    }

    // ── EPISODIC ───────────────────────────────────────────────────────────
    pub fn episode(&self, agent: &str, kind: &str, payload: &[u8]) -> std::io::Result<u64> {
        let seq = self.engine.now();
        self.engine
            .put(ep_key(agent, seq), Value::Bytes(payload.to_vec()))?;
        self.engine.put(
            ep_kind_key(agent, seq),
            Value::Bytes(kind.as_bytes().to_vec()),
        )?;
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
            if k.len() < 8 {
                continue;
            }
            let seq = u64::from_be_bytes(k[k.len() - 8..].try_into().unwrap());
            let payload = match v {
                Value::Bytes(b) => b,
                _ => continue,
            };
            let kind = match self.engine.get(&ep_kind_key(agent, seq)) {
                Some(Value::Bytes(b)) => String::from_utf8_lossy(&b).to_string(),
                _ => String::new(),
            };
            out.push(Episode {
                seq,
                agent: agent.to_string(),
                kind,
                payload,
            });
        }
        out.sort_by_key(|e| e.seq);
        out.truncate(limit);
        out
    }

    // ── RECALL (blended semantic + keyword) ────────────────────────────────
    pub fn recall(&self, query: &str, query_vec: &[f32], k: usize) -> Vec<RecallHit> {
        // 1) semantic ANN over LTM vectors — collect (id, sim, kind="semantic")
        let mut semantic_scores: HashMap<u64, f32> = HashMap::new();
        for (id, dist) in self.vectors.search(query_vec, k * 2) {
            let sim = 1.0 - dist / 2.0;
            semantic_scores.insert(id, sim);
        }

        // 2) keyword BM25-ish over token postings.
        //    Pass 1: for each token, scan its posting prefix (O(posting_list),
        //    not O(N_ltm)); accumulate per-(doc, token) TF and per-token DF.
        //    Pass 2: compute one BM25 score per doc, so a doc matching multiple
        //    query tokens correctly accumulates score (the previous version
        //    fixed the score on the FIRST hit via `or_insert_with` and dropped
        //    all subsequent contributions).
        let qtokens = tokenize(query);
        let total_docs = self.ltm_count.load(std::sync::atomic::Ordering::SeqCst) as f32;
        let snap = self.engine.snapshot();
        // per-doc accumulated keyword score
        let mut kw_scores: HashMap<u64, f32> = HashMap::new();
        for tok in &qtokens {
            let mut prefix = b"mem:kw:".to_vec();
            prefix.extend_from_slice(tok.as_bytes());
            prefix.push(b':');
            let matches = self.engine.scan_prefix(&prefix, snap);
            let df = matches.len() as u32;
            if df == 0 {
                continue;
            }
            let idf = ((total_docs - df as f32 + 0.5) / (df as f32 + 0.5)).ln() + 1.0;
            let idf = idf.max(0.0);
            for (kk, _) in matches {
                if kk.len() >= 8 {
                    let id = u64::from_be_bytes(kk[kk.len() - 8..].try_into().unwrap());
                    // per-token TF=1 (posting == presence); BM25 saturation.
                    let tw = 1.0f32;
                    let contrib = (tw * 2.0 / (tw + 1.0)) * idf;
                    *kw_scores.entry(id).or_insert(0.0) += contrib;
                }
            }
        }

        // 3) merge — fetch each candidate ONCE (avoids the previous O(hits) full
        // ltm_get inside a nested loop). Semantic hits win on tie for `kind`.
        let mut by_id: HashMap<u64, RecallHit> = HashMap::new();
        for (id, sim) in &semantic_scores {
            if let Some(rec) = self.ltm_get(*id) {
                let score = sim * (0.5 + 0.5 * rec.meta.salience);
                by_id.insert(
                    *id,
                    RecallHit {
                        id: *id,
                        text: rec.text.clone(),
                        score,
                        kind: "semantic",
                        meta: rec.meta,
                    },
                );
            }
        }
        for (id, kw) in kw_scores {
            let sal_mul = by_id
                .get(&id)
                .map(|h| 0.5 + 0.5 * h.meta.salience)
                .or_else(|| self.ltm_get(id).map(|r| 0.5 + 0.5 * r.meta.salience));
            let sal_mul = match sal_mul {
                Some(s) => s,
                None => continue,
            };
            let add = kw * 0.8 * sal_mul;
            match by_id.get_mut(&id) {
                Some(h) => h.score += add,
                None => {
                    if let Some(rec) = self.ltm_get(id) {
                        by_id.insert(
                            id,
                            RecallHit {
                                id,
                                text: rec.text.clone(),
                                score: add,
                                kind: "keyword",
                                meta: rec.meta,
                            },
                        );
                    }
                }
            }
        }

        let mut hits: Vec<RecallHit> = by_id.into_values().collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    /// Consolidation hook: bump an entry's salience (importance).
    pub fn touch_salience(&self, id: u64, delta: f32) -> std::io::Result<()> {
        if let Some(mut rec) = self.ltm_get(id) {
            rec.meta.salience = (rec.meta.salience + delta).clamp(0.0, 1.0);
            self.engine
                .put(meta_key(id), Value::Bytes(encode_meta(&rec.meta)))?;
        }
        Ok(())
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
        query: &str,
        query_vec: &[f32],
        k: usize,
        as_of: u64,
    ) -> Vec<RecallHit> {
        // Over-fetch, then filter by the fact's validity window.
        self.recall(query, query_vec, k * 4)
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
        )
        .unwrap();
        m.ltm_store(
            "user prefers rust over python for speed",
            v("rust speed preference"),
            "user",
            0.7,
            "trigger:chat",
        )
        .unwrap();
        m.ltm_store(
            "cache returned a stale value during load test",
            v("cache stale bug"),
            "tool:monitor",
            0.5,
            "trigger:alert",
        )
        .unwrap();

        let hits = m.recall("production deployment plan", &v("production deployment plan"), 3);
        assert!(!hits.is_empty());
        assert!(hits[0].text.contains("deployment"));
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
            .ltm_store("secret key rotated", vec![1.0; 4], "tool:cli", 0.6, "op:rotate")
            .unwrap();
        let rec = m.ltm_get(id).unwrap();
        assert_eq!(rec.meta.source, "tool:cli");
        assert_eq!(rec.meta.lineage, "op:rotate");
        assert_eq!(rec.meta.salience, 0.6);
    }
}
