//! RAG Pipeline — "the fastest AI RAG core".
//!
//! Hybrid retrieval over the same substrate the agent memory uses:
//!
//!   1. Chunk + ingest documents (text + embedding) into LTM.
//!   2. Retrieve with BOTH dense (HNSW ANN) and sparse (BM25-ish keyword) signals.
//!   3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF) — the
//!      technique that consistently beats either signal alone.
//!   4. Optional lightweight rerank by lexical overlap + salience.
//!   5. Cache query results, guarded by the MITM cache-debugger so a stale RAG
//!      answer (the classic "why did my agent cite old docs?" bug) is caught
//!      the instant the underlying corpus changes.
//!
//! Pure Rust. Zero external crates. Runs entirely in-process on DB-Strike.

use std::sync::Arc;

use memory::{tokenize, Memory};
use mitm::CacheDebugger;
use storage::Engine;

/// One retrieved chunk with its fused relevance score.
#[derive(Clone, Debug)]
pub struct Retrieved {
    pub id: u64,
    pub text: String,
    pub score: f32,
    pub source: String,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
}

/// The RAG engine. Wraps agent memory (for storage + hybrid recall) and a
/// MITM-guarded query cache.
pub struct Rag {
    memory: Arc<Memory>,
    cache: Arc<CacheDebugger>,
    rrf_k: f32,
}

impl Rag {
    pub fn open(engine: Arc<Engine>) -> Self {
        let memory = Arc::new(Memory::open(Arc::clone(&engine)));
        let cache = CacheDebugger::new(engine, 4096);
        Self { memory, cache, rrf_k: 60.0 }
    }

    pub fn memory(&self) -> &Arc<Memory> {
        &self.memory
    }
    pub fn cache(&self) -> &Arc<CacheDebugger> {
        &self.cache
    }

    /// Ingest a document chunk: store text + embedding + provenance in LTM.
    pub fn ingest(
        &self,
        text: &str,
        embedding: Vec<f32>,
        source: &str,
        owner: &str,
    ) -> std::io::Result<u64> {
        let id = self
            .memory
            .ltm_store(text, embedding, source, 0.5, "rag:ingest", owner)?;
        // any ingest invalidates the whole query cache generation
        self.bump_corpus();
        Ok(id)
    }

    /// A monotonically-increasing corpus generation. Every ingest bumps it; the
    /// query cache is keyed by generation so a changed corpus can't serve stale
    /// answers (and the MITM debugger records the source write).
    fn bump_corpus(&self) {
        let cur = self.corpus_gen();
        let _ = self
            .cache
            .source_set("rag:corpus_gen", (cur + 1).to_string().as_bytes());
    }

    /// Invalidate every cached RAG query result. Called by FLUSHALL after the
    /// substrate wipe: cached id lists would otherwise point at deleted
    /// memories (and a gen-0 collision could even serve them as "cached").
    pub fn invalidate_query_cache(&self) {
        self.bump_corpus();
    }
    fn corpus_gen(&self) -> u64 {
        // read straight from the source engine (authoritative, bypasses cache)
        self.cache
            .source_get("rag:corpus_gen")
            .and_then(|b| String::from_utf8_lossy(&b).parse::<u64>().ok())
            .unwrap_or(0)
    }

    /// Hybrid retrieve with Reciprocal Rank Fusion over dense + sparse lists.
    pub fn retrieve_scoped(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Vec<Retrieved> {
        let pool = (k * 4).max(20);

        // Single blended pass — the previous version called `recall` twice with
        // identical args, doubling every ANN + keyword scan. We now derive both
        // dense and sparse orderings from one pass. Scoped to the requesting
        // agent so cross-agent recall is impossible.
        let blended = self.memory.recall_scoped(scope, query, query_vec, pool);

        // dense list (ANN) — semantic ranking (already in blended order by score)
        let dense: Vec<(u64, f32)> = blended
            .iter()
            .filter(|h| h.kind == "semantic")
            .map(|h| (h.id, h.score))
            .collect();

        // sparse ordering: rank documents by keyword score only
        let mut sparse: Vec<(u64, f32)> = blended
            .iter()
            .filter(|h| h.kind == "keyword")
            .map(|h| (h.id, h.score))
            .collect();
        sparse.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // RRF fusion: score(d) = sum 1/(k + rank_in_list)
        use std::collections::HashMap;
        let mut fused: HashMap<u64, (f32, Option<usize>, Option<usize>)> = HashMap::new();
        for (rank, (id, _)) in dense.iter().enumerate() {
            let e = fused.entry(*id).or_insert((0.0, None, None));
            e.0 += 1.0 / (self.rrf_k + rank as f32 + 1.0);
            e.1 = Some(rank);
        }
        for (rank, (id, _)) in sparse.iter().enumerate() {
            let e = fused.entry(*id).or_insert((0.0, None, None));
            e.0 += 1.0 / (self.rrf_k + rank as f32 + 1.0);
            e.2 = Some(rank);
        }

        // materialize + lightweight rerank (lexical overlap boost)
        let qtok = tokenize(query);
        let mut out: Vec<Retrieved> = Vec::new();
        for (id, (mut score, drank, srank)) in fused {
            if let Some(rec) = self.memory.ltm_get(id) {
                let doc_tok = tokenize(&rec.text);
                let overlap = qtok.iter().filter(|t| doc_tok.contains(t)).count() as f32;
                score += 0.01 * overlap; // small lexical tie-breaker
                out.push(Retrieved {
                    id,
                    text: rec.text,
                    score,
                    source: rec.meta.source,
                    dense_rank: drank,
                    sparse_rank: srank,
                });
            }
        }
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    /// Cached retrieve: serves from the MITM-guarded cache when the corpus
    /// generation matches; otherwise recomputes and repopulates. Returns
    /// (results, served_from_cache).
    pub fn retrieve_cached(
        &self,
        scope: &str,
        query: &str,
        query_vec: &[f32],
        k: usize,
    ) -> (Vec<Retrieved>, bool) {
        let gen = self.corpus_gen();
        // Scope is part of the cache key: agent A must never be served
        // agent B's cached retrieval.
        let ckey = format!("rag:q:{gen}:{scope}:{k}:{query}");
        let (cached, verdict) = self.cache.cache_get(&ckey);
        if let Some(bytes) = cached {
            if !verdict.is_bug() {
                if let Some(ids) = decode_ids(&bytes) {
                    let mut out = Vec::new();
                    for (id, score) in ids {
                        if let Some(rec) = self.memory.ltm_get(id) {
                            out.push(Retrieved {
                                id,
                                text: rec.text,
                                score,
                                source: rec.meta.source,
                                dense_rank: None,
                                sparse_rank: None,
                            });
                        }
                    }
                    return (out, true);
                }
            }
        }
        let results = self.retrieve_scoped(scope, query, query_vec, k);
        // also register the cache-source value so staleness is detectable
        let payload = encode_ids(&results);
        let _ = self.cache.source_set(&ckey, &payload);
        self.cache.cache_set(&ckey, &payload);
        (results, false)
    }

    /// Assemble a prompt-ready context block from the top-k chunks.
    pub fn context_block(&self, scope: &str, query: &str, query_vec: &[f32], k: usize) -> String {
        let hits = self.retrieve_scoped(scope, query, query_vec, k);
        let mut s = String::new();
        for (i, h) in hits.iter().enumerate() {
            s.push_str(&format!("[{}] ({}) {}\n", i + 1, h.source, h.text));
        }
        s
    }
}

fn encode_ids(items: &[Retrieved]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for it in items {
        out.extend_from_slice(&it.id.to_le_bytes());
        out.extend_from_slice(&it.score.to_le_bytes());
    }
    out
}
fn decode_ids(buf: &[u8]) -> Option<Vec<(u64, f32)>> {
    if buf.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let mut p = 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = u64::from_le_bytes(buf.get(p..p + 8)?.try_into().ok()?);
        p += 8;
        let score = f32::from_le_bytes(buf.get(p..p + 4)?.try_into().ok()?);
        p += 4;
        out.push((id, score));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_rag_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Engine::open(dir.join(format!("rag_{n}.wal"))).unwrap()
    }

    // deterministic pseudo-embedding
    fn emb(s: &str) -> Vec<f32> {
        let mut v = vec![0f32; 16];
        for (i, c) in s.chars().enumerate() {
            v[i % 16] += (c as u32 % 11) as f32;
        }
        v
    }

    #[test]
    fn hybrid_retrieve_finds_relevant() {
        let r = Rag::open(eng());
        r.ingest("Rust ownership prevents data races at compile time", emb("rust ownership data races"), "doc:rust", "default").unwrap();
        r.ingest("Python uses a global interpreter lock called the GIL", emb("python gil interpreter lock"), "doc:python", "default").unwrap();
        r.ingest("HNSW is a graph index for approximate nearest neighbor search", emb("hnsw graph index nearest neighbor"), "doc:ann", "default").unwrap();

        let q = "nearest neighbor graph index";
        let hits = r.retrieve_scoped("default", q, &emb(q), 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source, "doc:ann");
    }

    #[test]
    fn cache_serves_then_invalidates_on_ingest() {
        let r = Rag::open(eng());
        r.ingest("the sky is blue during the day", emb("sky blue day"), "doc:1", "default").unwrap();
        let q = "sky color";
        let (_, cached1) = r.retrieve_cached("default", q, &emb(q), 2);
        assert!(!cached1); // first call computes
        let (_, cached2) = r.retrieve_cached("default", q, &emb(q), 2);
        assert!(cached2); // second call served from cache

        // new ingest bumps corpus generation -> cache key changes -> recompute
        r.ingest("the ocean is also blue", emb("ocean blue water"), "doc:2", "default").unwrap();
        let (_, cached3) = r.retrieve_cached("default", q, &emb(q), 2);
        assert!(!cached3);
    }
}
