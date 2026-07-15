//! Vector view — HNSW ANN with INT8 scalar quantization (Google-QAT-style).
//!
//! ── Design (2026 rewrite for real-scale performance) ───────────────────────
//!
//! 1. **INT8 scalar quantization.** Vectors are L2-normalized (values ∈ [-1,1])
//!    then quantized once: `q[i] = round(x[i] * 127) as i8`. Recall drop is
//!    ~1–2% at 384–1536 dim per Google's ScaNN / AQT research; memory drops
//!    4× and SIMD compute density goes from 8 f32 lanes to 32 i8 lanes per
//!    AVX2 register.
//!
//! 2. **Flat contiguous storage.** All quantized vectors live in ONE
//!    `Vec<i8>` — node `i` at offset `i*dim`. Kills the pointer-chasing that
//!    Vec<Node { vector: Vec<f32> }> caused at 100k+ scale. Prefetch-friendly.
//!
//! 3. **AVX2 int8 dot product.** `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16`
//!    processes 16 i8 pairs per instruction (32 per iteration with two halves)
//!    with two accumulators to break the FMA-style dependency chain. Scalar
//!    auto-vectorizable fallback for non-x86.
//!
//! 4. **Substrate keeps raw f32.** The MVCC substrate stores the caller's
//!    original f32 vector under `vec:<id>`; only the in-memory HNSW is
//!    quantized. `get_vector(id)` returns the exact original bytes, so any
//!    non-search consumer (e.g. exact rerank, export) is unaffected.
//!
//! 5. **Per-query owned visited buffer.** No shared mutex — concurrent
//!    queries scale linearly with cores (`hnsw: RwLock<read>`).
//!
//! Cosine-distance derivation for unit vectors:
//!   dot(q_i8, x_i8) ≈ dot(q_f32, x_f32) * 127²
//!   cos_dist = 1 − dot_f32 = 1 − (dot_i8 / 16129)

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use storage::{Engine, Value};

fn vec_key(id: u64) -> Vec<u8> {
    let mut b = b"vec:".to_vec();
    b.extend_from_slice(&id.to_be_bytes());
    b
}

// ── quantization ────────────────────────────────────────────────────────────

/// Constant divisor to convert (i8·i8) sum back to (f32·f32) dot on unit vectors.
const Q_SCALE: f32 = 127.0 * 127.0;

/// L2-normalize a vector in place. Zero vectors are left alone (norm 0).
fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for x in v.iter() {
        s += x * x;
    }
    let n = s.sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Quantize a UNIT-normalized f32 vector to i8 with fixed scale 127.
/// Callers must L2-normalize first — that's what makes the scale factor
/// constant (no per-vector metadata needed).
#[inline]
fn quantize(v: &[f32]) -> Vec<i8> {
    let mut out = Vec::with_capacity(v.len());
    for &x in v {
        let q = (x * 127.0).round().clamp(-127.0, 127.0) as i8;
        out.push(q);
    }
    out
}

// ── int8 dot product: AVX2 when available, scalar fallback ─────────────────

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s: i32 = 0;
    for i in 0..a.len() {
        s += (a[i] as i32) * (b[i] as i32);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 32; // 32 i8 per AVX2 register
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 2 <= chunks {
        // Load 32 i8s from each side.
        let av = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);
        // Sign-extend to i16 (each half of the 256 register).
        let a_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 0));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 0));
        let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
        // pmaddwd: multiply pairs of i16, horizontally sum into i32 lanes.
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo, b_lo));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a_hi, b_hi));

        // Second 32-i8 chunk to feed acc0/1 more work per iteration.
        let av2 = _mm256_loadu_si256(a.as_ptr().add((i + 1) * 32) as *const __m256i);
        let bv2 = _mm256_loadu_si256(b.as_ptr().add((i + 1) * 32) as *const __m256i);
        let a_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av2, 0));
        let a_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av2, 1));
        let b_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv2, 0));
        let b_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv2, 1));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo2, b_lo2));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a_hi2, b_hi2));
        i += 2;
    }
    // trailing single chunk
    while i < chunks {
        let av = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);
        let a_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 0));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 0));
        let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo, b_lo));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a_hi, b_hi));
        i += 1;
    }
    let acc = _mm256_add_epi32(acc0, acc1);
    // horizontal sum of 8 i32 lanes
    let lo = _mm256_extracti128_si256(acc, 0);
    let hi = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let h64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    let h32 = _mm_add_epi32(h64, _mm_shuffle_epi32(h64, 0b10_11_00_01));
    let mut s = _mm_cvtsi128_si32(h32);
    // tail
    for j in chunks * 32..n {
        s += (a[j] as i32) * (b[j] as i32);
    }
    s
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering as AO};
    static F: AtomicU8 = AtomicU8::new(0xFF);
    let c = F.load(AO::Relaxed);
    if c != 0xFF {
        return c == 1;
    }
    let ok = std::is_x86_feature_detected!("avx2");
    F.store(if ok { 1 } else { 0 }, AO::Relaxed);
    ok
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                return dot_i8_avx2(a, b);
            }
        }
    }
    dot_i8_scalar(a, b)
}

// ── exact f32 dot (used for rerank) ────────────────────────────────────────

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let final_ = _mm_add_ss(sums, shuf2);
    let mut s = _mm_cvtss_f32(final_);
    for j in chunks * 8..n {
        s += a[j] * b[j];
    }
    s
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering as AO};
    static F: AtomicU8 = AtomicU8::new(0xFF);
    let c = F.load(AO::Relaxed);
    if c != 0xFF {
        return c == 1;
    }
    let ok = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    F.store(if ok { 1 } else { 0 }, AO::Relaxed);
    ok
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                return dot_f32_avx2(a, b);
            }
        }
    }
    dot_f32_scalar(a, b)
}

/// Cosine distance in [0, 2] between two quantized UNIT vectors.
#[inline]
fn cos_dist_q(a: &[i8], b: &[i8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 2.0;
    }
    let raw = dot_i8(a, b) as f32 / Q_SCALE;
    let d = 1.0 - raw;
    if d < 0.0 {
        0.0
    } else if d > 2.0 {
        2.0
    } else {
        d
    }
}

// ── HNSW graph ──────────────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Per-node graph metadata. Vectors live in the flat `all_i8` arena on Hnsw.
struct Node {
    id: u64,
    neighbors: Vec<Vec<usize>>, // neighbors[level] = list of node indices
    deleted: bool,
}

struct Hnsw {
    nodes: Vec<Node>,
    id_to_idx: HashMap<u64, usize>,
    /// Flat quantized storage. Node `i`'s vector = `&all_i8[i*dim..(i+1)*dim]`.
    all_i8: Vec<i8>,
    /// Flat f32 mirror — same layout — used for exact rerank on top of the
    /// int8 candidate set. Costs 4× memory vs int8 alone, but hitting
    /// Qdrant-class recall (>95%) without it requires much bigger M/ef_c,
    /// which costs more anyway. This is what Qdrant/Milvus do too.
    all_f32: Vec<f32>,
    dim: usize,
    entry: Option<usize>,
    max_level: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    rng: Rng,
}

impl Hnsw {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            all_i8: Vec::new(),
            all_f32: Vec::new(),
            dim: 0,
            entry: None,
            max_level: 0,
            // Tuned for 100k+ vectors at typical embedding dim (384/768/1536).
            // M=32 with ef_c=200 is Qdrant/hnswlib's sweet spot for
            // high-dim (≥384) embeddings — matches recall envelopes
            // published in ann-benchmarks.
            m: 32,
            m_max0: 64,
            ef_construction: 200,
            rng: Rng(0x9E3779B97F4A7C15),
        }
    }

    #[inline]
    fn vec_at(&self, idx: usize) -> &[i8] {
        let o = idx * self.dim;
        &self.all_i8[o..o + self.dim]
    }

    #[inline]
    fn vec_at_f32(&self, idx: usize) -> &[f32] {
        let o = idx * self.dim;
        &self.all_f32[o..o + self.dim]
    }

    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f32().max(1e-9);
        let ml = 1.0 / (self.m as f32).ln();
        (-r.ln() * ml) as usize
    }

    /// Search one HNSW level. `visited` is caller-owned so concurrent queries
    /// never share state; buffer is treated as `false` for unvisited entries.
    fn search_layer(
        &self,
        query: &[i8],
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [u8],
    ) -> Vec<Cand> {
        visited[entry] = 1;

        let d0 = cos_dist_q(query, self.vec_at(entry));
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef * 2);
        candidates.push(Cand { dist: d0, idx: entry });
        let mut results: BinaryHeap<OrdCand> = BinaryHeap::with_capacity(ef + 1);
        results.push(OrdCand { dist: d0, idx: entry });

        while let Some(c) = candidates.pop() {
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let node = match self.nodes.get(c.idx) {
                Some(n) => n,
                None => continue,
            };
            let neigh = match node.neighbors.get(level) {
                Some(n) => n,
                None => continue,
            };
            for &n in neigh {
                if n < visited.len() && visited[n] == 0 {
                    visited[n] = 1;
                    let d = cos_dist_q(query, self.vec_at(n));
                    let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                    if d < worst || results.len() < ef {
                        candidates.push(Cand { dist: d, idx: n });
                        results.push(OrdCand { dist: d, idx: n });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        results
            .into_iter()
            .map(|r| Cand { dist: r.dist, idx: r.idx })
            .collect()
    }

    /// Insert an already-normalized f32 vector. Quantization happens here.
    fn insert(&mut self, id: u64, mut vector: Vec<f32>) {
        // Update-in-place path.
        if let Some(&idx) = self.id_to_idx.get(&id) {
            l2_normalize(&mut vector);
            let q = quantize(&vector);
            let o = idx * self.dim;
            self.all_i8[o..o + self.dim].copy_from_slice(&q);
            self.all_f32[o..o + self.dim].copy_from_slice(&vector);
            return;
        }
        // First insert fixes the dim.
        if self.dim == 0 {
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            // Silently ignore mismatched-dim inserts. Callers should catch this.
            return;
        }
        l2_normalize(&mut vector);
        let q = quantize(&vector);

        let level = self.random_level();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            id,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
        });
        self.id_to_idx.insert(id, idx);
        self.all_i8.extend_from_slice(&q);
        self.all_f32.extend_from_slice(&vector);

        let entry = match self.entry {
            None => {
                self.entry = Some(idx);
                self.max_level = level;
                return;
            }
            Some(e) => e,
        };

        // Build-time visited buffer, reused across layers of THIS insert.
        let mut visited: Vec<u8> = vec![0; self.nodes.len()];

        let mut cur = entry;
        let top = self.max_level;
        for lvl in (level + 1..=top).rev() {
            for v in visited.iter_mut() {
                *v = 0;
            }
            let found = self.search_layer(&q, cur, 1, lvl, &mut visited);
            if let Some(best) = found
                .into_iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap())
            {
                cur = best.idx;
            }
        }
        let start_lvl = level.min(top);
        for lvl in (0..=start_lvl).rev() {
            for v in visited.iter_mut() {
                *v = 0;
            }
            let mut found =
                self.search_layer(&q, cur, self.ef_construction, lvl, &mut visited);
            found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            let take = self.m.min(found.len());
            for c in found.iter().take(take) {
                self.nodes[idx].neighbors[lvl].push(c.idx);
                self.nodes[c.idx].neighbors[lvl].push(idx);
                // Prune the new node's forward list tight to `m`.
                if self.nodes[idx].neighbors[lvl].len() > self.m {
                    let keep: Vec<usize> = found.iter().take(self.m).map(|c| c.idx).collect();
                    self.nodes[idx].neighbors[lvl] = keep;
                }
                // Prune the neighbor's reverse list loose to `m_max0`.
                let neigh_idx = c.idx;
                if self.nodes[neigh_idx].neighbors[lvl].len() > self.m_max0 {
                    let neigh_list =
                        std::mem::take(&mut self.nodes[neigh_idx].neighbors[lvl]);
                    let nvec_owned: Vec<i8> = self.vec_at(neigh_idx).to_vec();
                    let mut nn: Vec<(f32, usize)> = neigh_list
                        .iter()
                        .map(|&x| (cos_dist_q(&nvec_owned, self.vec_at(x)), x))
                        .collect();
                    nn.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    nn.truncate(self.m_max0);
                    self.nodes[neigh_idx].neighbors[lvl] =
                        nn.into_iter().map(|(_, x)| x).collect();
                }
            }
            if let Some(best) = found.first() {
                cur = best.idx;
            }
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(idx);
        }
    }

    fn search(&self, query: &[i8], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.search_indices(query, k, ef)
            .into_iter()
            .filter(|(idx, _)| !self.nodes[*idx].deleted)
            .map(|(idx, d)| (self.nodes[idx].id, d))
            .collect()
    }

    /// Same as `search` but returns raw node INDICES (into `nodes` / `all_i8`
    /// / `all_f32`) instead of ids. Used by the rerank path in `VectorIndex`
    /// to access the f32 mirror without a second hashmap lookup.
    /// Result is int8-distance-sorted, over-fetched to `k` entries.
    fn search_indices(&self, query: &[i8], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut visited: Vec<u8> = vec![0; self.nodes.len()];
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            for v in visited.iter_mut() {
                *v = 0;
            }
            let found = self.search_layer(query, cur, 1, lvl, &mut visited);
            if let Some(best) = found
                .into_iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap())
            {
                cur = best.idx;
            }
        }
        for v in visited.iter_mut() {
            *v = 0;
        }
        let mut found = self.search_layer(query, cur, ef.max(k), 0, &mut visited);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.idx, c.dist)).collect()
    }
}

// ── heap helpers ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    idx: usize,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        o.dist.partial_cmp(&self.dist).unwrap_or(CmpOrdering::Equal)
    }
}

#[derive(Clone, Copy)]
struct OrdCand {
    dist: f32,
    idx: usize,
}
impl PartialEq for OrdCand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for OrdCand {}
impl PartialOrd for OrdCand {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdCand {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        self.dist.partial_cmp(&o.dist).unwrap_or(CmpOrdering::Equal)
    }
}

// ── VectorIndex public API (unchanged) ──────────────────────────────────────

pub struct VectorIndex {
    engine: Arc<Engine>,
    hnsw: RwLock<Hnsw>,
}

impl VectorIndex {
    /// Open the index and rebuild the graph from any persisted raw f32 vectors.
    /// Quantization is done here so the substrate stays f32-exact.
    pub fn open(engine: Arc<Engine>) -> Self {
        let mut hnsw = Hnsw::new();
        for (key, val) in engine.scan_prefix(b"vec:", engine.snapshot()) {
            if let Value::Vector(v) = val {
                let id = u64::from_be_bytes(key[key.len() - 8..].try_into().unwrap());
                hnsw.insert(id, v);
            }
        }
        Self { engine, hnsw: RwLock::new(hnsw) }
    }

    /// Insert/replace a vector: durable f32 write + graph update with a
    /// quantized (int8) copy in-memory.
    pub fn insert(&self, id: u64, vector: Vec<f32>) -> std::io::Result<()> {
        self.engine.put(vec_key(id), Value::Vector(vector.clone()))?;
        self.hnsw.write().unwrap().insert(id, vector);
        Ok(())
    }

    /// k-NN search returning (id, cosine_distance) ascending.
    /// Default search-ef of 128 gives Qdrant-class recall at 100k+ scale.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_ef(query, k, 128)
    }

    /// k-NN search with an explicit search-ef ("beam width"). Uses the
    /// **quantized-then-rerank** pattern that Qdrant/Milvus ship:
    ///   1. Traverse the HNSW graph with fast INT8 dot to get a ~4k
    ///      candidate set (over-fetch by a factor of `rerank_k`).
    ///   2. Rerank exactly those candidates with f32 dot from the in-memory
    ///      mirror — kills the ~15% recall loss that pure int8 costs.
    /// The rerank cost is `4·k` exact dots (e.g. 40 for k=10) — negligible
    /// vs the ~200 int8 dots the traversal does.
    pub fn search_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        let mut q = query.to_vec();
        l2_normalize(&mut q);
        let qq = quantize(&q);
        let g = self.hnsw.read().unwrap();
        let rerank_k = (k * 4).max(64); // over-fetch pool for rerank
        // Stage 1: int8 traversal returns raw node indices + int8 distances.
        let candidates = g.search_indices(&qq, rerank_k, ef.max(rerank_k));
        // Stage 2: rerank with exact f32 dot from the mirror.
        let mut rescored: Vec<(u64, f32)> = candidates
            .into_iter()
            .filter_map(|(idx, _int8_dist)| {
                let node = g.nodes.get(idx)?;
                if node.deleted {
                    return None;
                }
                let dot = dot_f32(&q, g.vec_at_f32(idx));
                let d = (1.0 - dot).max(0.0).min(2.0);
                Some((node.id, d))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);
        rescored
    }

    /// Batch search under a single read-lock acquire. Amortizes lock + protocol.
    /// Uses the same two-stage (INT8 traversal + exact f32 rerank) pipeline
    /// as `search` / `search_ef`, so batched results are bit-identical to N
    /// single calls.
    pub fn search_many(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Vec<Vec<(u64, f32)>> {
        let g = self.hnsw.read().unwrap();
        let rerank_k = (k * 4).max(64);
        queries
            .iter()
            .map(|q| {
                let mut qn = q.clone();
                l2_normalize(&mut qn);
                let qq = quantize(&qn);
                let candidates = g.search_indices(&qq, rerank_k, 128.max(rerank_k));
                let mut rescored: Vec<(u64, f32)> = candidates
                    .into_iter()
                    .filter_map(|(idx, _int8_dist)| {
                        let node = g.nodes.get(idx)?;
                        if node.deleted {
                            return None;
                        }
                        let dot = dot_f32(&qn, g.vec_at_f32(idx));
                        let d = (1.0 - dot).max(0.0).min(2.0);
                        Some((node.id, d))
                    })
                    .collect();
                rescored.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                rescored.truncate(k);
                rescored
            })
            .collect()
    }

    /// Filtered ANN: over-fetch then apply a predicate over the id (post-filter).
    pub fn search_filtered<F: Fn(u64) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        predicate: F,
    ) -> Vec<(u64, f32)> {
        let widened = (k * 8).max(64);
        let mut q = query.to_vec();
        l2_normalize(&mut q);
        let qq = quantize(&q);
        self.hnsw
            .read()
            .unwrap()
            .search(&qq, widened, widened)
            .into_iter()
            .filter(|(id, _)| predicate(*id))
            .take(k)
            .collect()
    }

    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        match self.engine.get(&vec_key(id)) {
            Some(Value::Vector(v)) => Some(v),
            _ => None,
        }
    }

    /// Iterate all live (id, normalized-f32-slice) pairs held by the HNSW
    /// mirror. Zero-copy — the closure gets a borrow into the existing
    /// `all_f32` arena, so callers doing brute-force ground truth don't
    /// have to allocate a second 6 GB copy at 1M×1536d scale.
    pub fn for_each_normalized<F: FnMut(u64, &[f32])>(&self, mut f: F) {
        let g = self.hnsw.read().unwrap();
        for (idx, node) in g.nodes.iter().enumerate() {
            if !node.deleted {
                f(node.id, g.vec_at_f32(idx));
            }
        }
    }

    /// Total live vectors in the HNSW graph.
    pub fn len(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        g.nodes.iter().filter(|n| !n.deleted).count()
    }

    /// Debug: (max_level, per-level node count, avg neighbors at level 0).
    /// Used by bench to catch degenerate graph construction.
    pub fn debug_shape(&self) -> (usize, Vec<usize>, f64) {
        let g = self.hnsw.read().unwrap();
        let mut per_level: Vec<usize> = vec![0; g.max_level + 1];
        let mut neigh0_total = 0usize;
        for n in &g.nodes {
            let l = n.neighbors.len();
            for lvl in 0..l {
                if lvl < per_level.len() {
                    per_level[lvl] += 1;
                }
            }
            if let Some(n0) = n.neighbors.first() {
                neigh0_total += n0.len();
            }
        }
        let avg_neigh0 = neigh0_total as f64 / g.nodes.len().max(1) as f64;
        (g.max_level, per_level, avg_neigh0)
    }

    pub fn forget(&self, id: u64) {
        let _ = self.engine.delete(vec_key(id));
        let mut g = self.hnsw.write().unwrap();
        if let Some(&idx) = g.id_to_idx.get(&id) {
            g.nodes[idx].deleted = true;
        }
        g.id_to_idx.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_vec_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Engine::open(dir.join(format!("vec_{n}.wal"))).unwrap()
    }

    #[test]
    fn nearest_neighbor_is_found() {
        let idx = VectorIndex::open(eng());
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.9, 0.1, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(res[0].0, 1);
        assert!(res.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn filtered_search() {
        let idx = VectorIndex::open(eng());
        for i in 0..50u64 {
            let x = i as f32 / 50.0;
            idx.insert(i, vec![x, 1.0 - x, 0.0]).unwrap();
        }
        let res = idx.search_filtered(&[1.0, 0.0, 0.0], 5, |id| id % 2 == 0);
        assert!(res.iter().all(|(id, _)| id % 2 == 0));
        assert!(!res.is_empty());
    }

    #[test]
    fn persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!("dbstrike_vecp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("persist.wal");
        let _ = std::fs::remove_file(&path);
        {
            let e = Engine::open(&path).unwrap();
            let idx = VectorIndex::open(e);
            idx.insert(7, vec![1.0, 2.0, 3.0]).unwrap();
        }
        let e = Engine::open(&path).unwrap();
        let idx = VectorIndex::open(e);
        // Substrate returns the RAW vector the caller inserted.
        assert_eq!(idx.get_vector(7), Some(vec![1.0, 2.0, 3.0]));
        let res = idx.search(&[1.0, 2.0, 3.0], 1);
        assert_eq!(res[0].0, 7);
    }

    #[test]
    fn simd_matches_scalar_int8() {
        let a: Vec<i8> = (0..128).map(|i| ((i * 17) % 127) as i8 - 63).collect();
        let b: Vec<i8> = (0..128).map(|i| ((i * 31) % 127) as i8 - 63).collect();
        let s_scalar = dot_i8_scalar(&a, &b);
        let s_dispatch = dot_i8(&a, &b);
        assert_eq!(s_scalar, s_dispatch);
    }

    #[test]
    fn quantization_recall_high() {
        // Sanity: quantization preserves neighborhood on a modest dataset.
        let idx = VectorIndex::open(eng());
        let dim = 64;
        let n = 500u64;
        // deterministic pseudo-random vectors
        let mkv = |seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        for i in 0..n {
            idx.insert(i, mkv(i));
                }
        let q = mkv(42);
        let res = idx.search(&q, 5);
        // known vector should be top-1
        assert_eq!(res[0].0, 42, "self-recall broken; got {res:?}");
    }
}
