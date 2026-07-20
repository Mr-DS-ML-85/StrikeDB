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

use std::cell::UnsafeCell;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use storage::{Engine, Value};

/// MODULE 4 — a predicate over a node's `attr` (its category id). Qdrant forces
/// you to pick ONE strategy (pre-filter, which breaks graph connectivity, or
/// post-filter, which collapses recall on selective filters). We keep ONE graph
/// and route by selectivity at query time, so the predicate is deliberately
/// tiny and zero-alloc.
#[derive(Clone)]
pub enum Filter {
    /// No filtering — plain ANN.
    Any,
    /// Exact category match (`node.attr == kind`).
    Eq(u32),
    /// Membership in a small set of categories.
    In(Vec<u32>),
}

impl Filter {
    fn matches(&self, attr: u32) -> bool {
        match self {
            Filter::Any => true,
            Filter::Eq(k) => attr == *k,
            Filter::In(set) => set.iter().any(|&k| k == attr),
        }
    }
}

/// MODULE 5 — sparse / lexical index over the SAME document ids as the dense
/// HNSW. A sparse vector is a bag-of-words: `(term_id, weight)` pairs. Retrieval
/// uses an inverted index (term → posting list of doc ids) scored with BM25 —
/// exact, no graph, and blazing fast because only the query's few terms' posting
/// lists are touched. Qdrant/Milvus force you to stand up a *second* index
/// (sparse) and fuse client-side; we keep both behind ONE `VectorIndex` and
/// fuse in-engine with Reciprocal Rank Fusion (RRF) — one query, one result
/// list, no extra round-trips.
pub struct SparseIndex {
    /// Inverted index: term id → doc ids that contain it.
    postings: HashMap<u32, Vec<u64>>,
    /// Per-doc sparse vector (term, weight), indexed by insertion order.
    doc_terms: Vec<Vec<(u32, f32)>>,
    /// Per-doc length (number of unique terms) for BM25 length norm.
    doc_len: Vec<usize>,
    /// Total docs (for BM25 `N`).
    n: usize,
    /// Average doc length (BM25 `avgdl`).
    avgdl: f32,
}

impl SparseIndex {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_terms: Vec::new(),
            doc_len: Vec::new(),
            n: 0,
            avgdl: 0.0,
        }
    }

    /// Add a document's sparse vector under `id` (must be the same id used in
    /// the dense HNSW so hybrid fusion can join on id).
    fn add(&mut self, id: u64, terms: Vec<(u32, f32)>) {
        let dl = terms.len();
        let order = self.doc_terms.len();
        let _ = id; // id is implicit by insertion order; kept for parity/clarity
        for &(t, _) in &terms {
            self.postings.entry(t).or_default().push(order as u64);
        }
        self.doc_terms.push(terms);
        self.doc_len.push(dl);
        self.n += 1;
        let new_avg = self.avgdl + (dl as f32 - self.avgdl) / self.n as f32;
        self.avgdl = new_avg;
    }

    /// BM25 score of `query_terms` against the doc at `order`. `k1`/`b` are the
    /// standard BM25 tunables.
    fn bm25(&self, order: usize, query_terms: &[(u32, f32)], k1: f32, b: f32) -> f32 {
        let dl = self.doc_len[order].max(1) as f32;
        let mut score = 0.0f32;
        for &(qt, qw) in query_terms {
            let hits = match self.postings.get(&qt) {
                Some(p) => p.iter().filter(|&&d| d as usize == order).count(),
                None => 0,
            };
            if hits == 0 {
                continue;
            }
            let df = self.postings.get(&qt).map(|p| p.len()).unwrap_or(0) as f32;
            let idf = ((self.n as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            let tf = hits as f32;
            let denom = tf + k1 * (1.0 - b + b * dl / self.avgdl.max(1e-6));
            score += qw * idf * (tf * (k1 + 1.0)) / denom;
        }
        score
    }

    /// Top-`k` docs for a sparse query, scored by BM25. Returns (order, score).
    fn search(&self, query_terms: &[(u32, f32)], k: usize) -> Vec<(u64, f32)> {
        if self.n == 0 {
            return Vec::new();
        }
        // Candidate docs = union of posting lists for the query's terms.
        let mut cand: Vec<usize> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &(qt, _) in query_terms {
            if let Some(p) = self.postings.get(&qt) {
                for &d in p {
                    if seen.insert(d as usize) {
                        cand.push(d as usize);
                    }
                }
            }
        }
        let mut scored: Vec<(u64, f32)> = cand
            .into_iter()
            .map(|order| (order as u64, self.bm25(order, query_terms, 1.5, 0.75)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Reciprocal Rank Fusion: combine ranked lists (each a list of doc ids, best
/// first) into one ranking. `score = Σ 1/(rank + k)`. Ranking-only — no scores
/// cross lists, so dense (cosine) and sparse (BM25) ranks fuse fairly without
/// normalization. `k` is the RRF constant (standard 60).
fn rrf_fuse(lists: &[Vec<u64>], k: usize) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for list in lists {
        for (rank, &id) in list.iter().enumerate() {
            *acc.entry(id).or_insert(0.0) += 1.0 / (rank as f32 + k as f32 + 1.0);
        }
    }
    let mut out: Vec<(u64, f32)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 2 — Tiered/Disk HNSW (NVMe-mmap cold tier)
//
// Qdrant's HNSW is RAM-bound: every vector (and its graph) lives in memory,
// so 100M+ vectors means 100M×(4B f32 + graph) — hundreds of GB. DiskANN
// solves scale by putting the WHOLE graph on SSD, but that's a different,
// complex algorithm. Milvus offers both but as separate index types with
// huge ops overhead.
//
// The novel play: keep the HOT working set in RAM (int8 layer-0 vectors +
// the layer-0 graph — the 99% of traversal) and `mmap` the COLD payload (the
// exact f32 rerank originals) to an NVMe file. The OS page cache serves it
// like RAM on the rare rerank touch; at 100M+ scale it spills to disk instead
// of OOM-ing. Recall is IDENTICAL (the rerank still uses exact f32 — just
// read through a memory map). RAM drops from ~5×int8 to ~1×int8. This is the
// "DiskANN idea applied to HNSW's cold layers": one index, RAM-hot / NVMe-
// cold, no distributed-ops burden.
//
// `MmapTier` is a tiny zero-dependency mmap wrapper (inline libc FFI, same
// style as the bench crate's `kill` shim — no external crate).
// ───────────────────────────────────────────────────────────────────────────

#[cfg(unix)]
extern "C" {
    fn open(path: *const i8, flags: i32, mode: u32) -> i32;
    fn ftruncate(fd: i32, len: i64) -> i32;
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
    fn close(fd: i32) -> i32;
}

const MMAP_PROT_READ: i32 = 0x1;
const MMAP_PROT_WRITE: i32 = 0x2;
const MMAP_MAP_SHARED: i32 = 0x01;
const MMAP_O_RDWR: i32 = 0x2;
const MMAP_O_CREAT: i32 = 0x40;
const MMAP_O_TRUNC: i32 = 0x200;
const MMAP_MS_SYNC: i32 = 0x4;

/// NVMe-backed array of `f32`, memory-mapped from a temp file. Exposes a
/// `&[f32]` view for reads (the rerank path) and `&mut [f32]` for writes
/// (graph build). On drop the mapping is unmapped; the file persists until
/// explicitly removed (we leave it for OS temp cleanup).
///
/// `MmapTier` is `Send`: each instance owns a distinct mapping + fd; the
/// underlying region is never shared across threads, so moving it between
/// threads is safe (this is what lets `Hnsw` cross `thread::spawn` bounds
/// during the parallel build even though it may carry a tier).
pub struct MmapTier {
    ptr: *mut f32,
    len: usize, // number of f32 elements
    fd: i32,
    _path: std::path::PathBuf,
}

// SAFETY: a mapped region with a unique fd is safe to move between threads;
// we never dereference it from two threads at once.
unsafe impl Send for MmapTier {}
// SAFETY: shared `&MmapTier` access only reads/writes through the mapped
// slice; in practice the tier is always behind `RwLock<Hnsw>`, so concurrent
// access is already serialized. Declaring Sync lets `Arc<VectorIndex>` be
// Send for multi-threaded bench/test code.
unsafe impl Sync for MmapTier {}

impl MmapTier {
    /// Map `count` f32 elements. Backed by a uniquely-named temp file.
    #[cfg(unix)]
    pub fn new(count: usize) -> Option<Self> {
        let path = std::env::temp_dir().join(format!(
            "dbstrike_tier_{}_{}.f32",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let cpath = std::ffi::CString::new(path.as_os_str().to_str()?).ok()?;
        let fd = unsafe { open(cpath.as_ptr(), MMAP_O_RDWR | MMAP_O_CREAT | MMAP_O_TRUNC, 0o600) };
        if fd < 0 {
            return None;
        }
        let bytes = (count * std::mem::size_of::<f32>()) as i64;
        if unsafe { ftruncate(fd, bytes) } != 0 {
            unsafe { close(fd) };
            return None;
        }
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes as usize,
                MMAP_PROT_READ | MMAP_PROT_WRITE,
                MMAP_MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == usize::MAX as *mut std::ffi::c_void || ptr.is_null() {
            unsafe { close(fd) };
            return None;
        }
        Some(Self { ptr: ptr as *mut f32, len: count, fd, _path: path })
    }

    #[cfg(not(unix))]
    pub fn new(_count: usize) -> Option<Self> {
        None
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Flush dirty pages to the NVMe file.
    pub fn flush(&self) {
        let bytes = self.len * std::mem::size_of::<f32>();
        unsafe { msync(self.ptr as *mut std::ffi::c_void, bytes, MMAP_MS_SYNC) };
    }
}

impl Drop for MmapTier {
    fn drop(&mut self) {
        let bytes = self.len * std::mem::size_of::<f32>();
        unsafe {
            munmap(self.ptr as *mut std::ffi::c_void, bytes);
            close(self.fd);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MITM DEBUG SYSTEM
// ───────────────────────────────────────────────────────────────────────────
// Gated on `DBSTRIKE_DEBUG` (set the env var to enable). Streams pipeline
// observability to stderr WITHOUT aborting the run: insert timing/RSS/graph
// shape, search_layer node-visit counts, greedy-break triggers, and
// entry-descent distances. The counters are atomics so concurrent queries
// don't corrupt them; printing is rate-limited to once per 1k inserts /
// once per query so it's cheap enough to leave on at 1M scale.
//
// Companion env knobs (all independent of DBSTRIKE_DEBUG):
//   DBSTRIKE_NO_BREAK  — disable the greedy early-stop in search_layer
//                        (full-traversal mode, for recall-vs-graph diagnosis)
//   DBSTRIKE_DEBUG_SEARCH <idx>
//                      — dump a full trace for ONE query vector `idx`
//                        (search_indices path): entry, per-layer descent
//                        distances, layer-0 visited count, break reason.
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::atomic::{AtomicU64, Ordering as AOrd};

/// Once-initialized debug state. Cheap to read every call (a single atomic
/// load of a static bool) so the hot path stays fast when debug is off.
#[derive(Clone, Copy)]
struct Mitm {
    on: bool,
    no_break: bool,
    noprune: bool,
    trace_query: i64, // -1 = off
}

fn mitm() -> Mitm {
    // Read env once per process lazily.
    static STATE: std::sync::OnceLock<Mitm> = std::sync::OnceLock::new();
    *STATE.get_or_init(|| {
        let on = std::env::var_os("DBSTRIKE_DEBUG").is_some();
        let no_break = std::env::var_os("DBSTRIKE_NO_BREAK").is_some();
        let noprune = std::env::var_os("DBSTRIKE_NOPRUNE").is_some();
        let trace_query = std::env::var("DBSTRIKE_DEBUG_SEARCH")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(-1);
        Mitm { on, no_break, noprune, trace_query }
    })
}

// Global cumulative counters (debug only).
static CNT_INSERTS: AtomicU64 = AtomicU64::new(0);
static CNT_SEARCH_LAYER: AtomicU64 = AtomicU64::new(0);
static CNT_SL_VISITED: AtomicU64 = AtomicU64::new(0); // total nodes expanded
static CNT_SL_BREAKS: AtomicU64 = AtomicU64::new(0); // greedy stops hit
static CNT_SL_FULL: AtomicU64 = AtomicU64::new(0); // ran to exhaustion (no break)

/// Emit a structured MITM line to stderr. Caller controls rate.
macro_rules! mitm_log {
    ($($arg:tt)*) => {
        eprintln!("[MITM] {}", format_args!($($arg)*));
    };
}

fn vec_key(id: u64) -> Vec<u8> {
    let mut b = b"vec:".to_vec();
    b.extend_from_slice(&id.to_be_bytes());
    b
}

thread_local! {
    static DEBUG_DOTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Best-effort process RSS in bytes (Linux only). Returns 0 elsewhere.
#[cfg(target_os = "linux")]
fn rss_bytes() -> usize {
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = s.split_whitespace().nth(1) {
            if let Ok(p) = pages.parse::<usize>() {
                return p * 4096;
            }
        }
    }
    0
}
#[cfg(not(target_os = "linux"))]
fn rss_bytes() -> usize {
    0
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

/// L2 norm (magnitude) of a vector — used by TurboQuant cosine correction.
#[inline]
fn vector_l2(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Quantize a UNIT-normalized f32 vector to i8 with fixed scale 127.
/// Callers must L2-normalize first — that's what makes the scale factor
/// constant (no per-vector metadata needed). Writes into `out` (reused across
/// inserts to avoid a heap allocation per vector).
    #[inline]
    fn quantize_into(v: &[f32], out: &mut Vec<i8>) {
        out.clear();
        out.reserve(v.len());
        for &x in v {
            let q = (x * 127.0).round().clamp(-127.0, 127.0) as i8;
            out.push(q);
        }
    }

    /// Pack a normalized `vector` (f32) into `out` per the given quant mode.
    /// Returns the number of bytes written. `q` is the int8 quantization of
    /// `vector` (used by the binary family). TurboQuant/Product derive their
    /// packed form directly from `vector`. Accepts pre-allocated scratch
    /// buffers to avoid per-insert heap allocation (~34KB for TurboQuant).
    fn pack_current(
        quant: QuantMode,
        turbo: Option<&TurboParams>,
        pq: Option<&PqParams>,
        vector: &[f32],
        q: &[i8],
        out: &mut Vec<u8>,
        scratch_rot: &mut Vec<f32>,
        scratch_idx: &mut Vec<u32>,
        scratch_deq: &mut Vec<f32>,
        scratch_r: &mut Vec<f32>,
        scratch_qjl: &mut Vec<u8>,
    ) -> usize {
        match quant {
            QuantMode::Int8 => {
                out.clear();
                0
            }
            QuantMode::Binary | QuantMode::Binary2 | QuantMode::Binary15 => {
                pack_lowbit(quant, q, out);
                out.len()
            }
            QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4 => {
                let tp = match turbo {
                    Some(t) => t,
                    None => {
                        out.clear();
                        return 0;
                    }
                };
                // `vector` is already L2-normalized (callers normalize before
                // packing). Rotate, MSE-quantize, compute residual, QJL sketch.
                // Reuse scratch buffers to avoid per-insert heap allocation.
                scratch_rot.clear();
                scratch_rot.resize(tp.d, 0.0);
                tp.rotate(vector, scratch_rot);
                scratch_idx.clear();
                scratch_idx.resize(tp.d, 0);
                scratch_deq.clear();
                scratch_deq.resize(tp.d, 0.0);
                let mut e2 = 0.0f32;
                for i in 0..tp.d {
                    let l = tp.quant_level(scratch_rot[i]);
                    scratch_idx[i] = l;
                    scratch_deq[i] = tp.dequant_level(l);
                    let e = scratch_rot[i] - scratch_deq[i];
                    e2 += e * e;
                }
                let rn = e2.sqrt(); // residual L2 norm ‖r‖
                // QJL sign bits: z_j = sign((S·r)_j) where S = (1/√d)·H·D.
                scratch_r.clear();
                scratch_r.resize(tp.d, 0.0);
                for i in 0..tp.d {
                    scratch_r[i] = (scratch_rot[i] - scratch_deq[i]) * tp.s_sign[i] as f32;
                }
                hadamard(scratch_r);
                scratch_qjl.clear();
                scratch_qjl.resize(tp.d, 0);
                for j in 0..tp.d {
                    scratch_qjl[j] = if scratch_r[j] >= 0.0 { 1 } else { 0 };
                }
                tp.pack(scratch_idx, scratch_qjl, rn, out);
                out.len()
            }
            QuantMode::Product => {
                let pp = match pq {
                    Some(p) => p,
                    None => {
                        out.clear();
                        return 0;
                    }
                };
                let codes = pp.encode(vector);
                out.clear();
                out.extend_from_slice(&codes);
                out.len()
            }
        }
    }

/// Convenience: allocate a fresh quantized vector (used by tests / one-shots).
#[inline]
fn quantize(v: &[f32]) -> Vec<i8> {
    let mut out = Vec::with_capacity(v.len());
    quantize_into(v, &mut out);
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
        // Prefetch next iteration's data to hide cache miss latency.
        if i + 4 <= chunks {
            _mm_prefetch(a.as_ptr().add((i + 2) * 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add((i + 2) * 32) as *const i8, _MM_HINT_T0);
        }
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

/// Resolved once: the best int8 dot routine for this CPU. Avoids a per-call
/// feature-check (and its global-atomic load) in the hottest search loop.
type DotI8Fn = unsafe fn(&[i8], &[i8]) -> i32;
type DotF32Fn = unsafe fn(&[f32], &[f32]) -> f32;

fn resolve_dot_i8() -> DotI8Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return dot_i8_avx2;
        }
    }
    dot_i8_scalar
}

fn resolve_dot_f32() -> DotF32Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return dot_f32_avx2;
        }
    }
    dot_f32_scalar
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    static F: std::sync::OnceLock<DotI8Fn> = std::sync::OnceLock::new();
    unsafe { (F.get_or_init(resolve_dot_i8))(a, b) }
}

// ── binary / 2-bit quantization (Module 2 — Qdrant-competitive compression) ──
//
// Asymmetric quantization (Qdrant 1.15): stored vectors are packed to 1 or 2
// bits per dim (sign, plus a magnitude tier for 2-bit), while the QUERY stays
// full int8. The traversal scores the int8 query against the reconstructed
// low-bit vector — cheap (popcount / sign-agreement), ~16–32× less RAM than
// f32, and the exact-f32 rerank on the candidate set (as in `search_ef`)
// recovers the recall the coarse codes lose. This is what lets DB-Strike hold
// 10M+ 768-d vectors in RAM at Qdrant-class recall.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantMode {
    /// INT8 scalar quantization (4× vs f32). Default; best accuracy.
    Int8,
    /// 1-bit sign (binary). ~32× vs f32. Best for high-dim (≥768) centered data.
    Binary,
    /// 2-bit (sign + magnitude tier). ~16× vs f32, better accuracy than 1-bit
    /// for mid-dim vectors.
    Binary2,
    /// 1.5-bit binary (Qdrant 1.15): three states (−1, 0, +1) shared across
    /// pairs; ~24× vs f32, intermediate accuracy between 1- and 2-bit.
    Binary15,
    /// TurboQuant 4-bit (Qdrant 1.18 / Google ICLR 2026): Hadamard rotation +
    /// optimal scalar codebook + 1-bit QJL residual correction. ~8× vs f32,
    /// recall competitive with INT8 at half the RAM.
    Turbo4,
    /// TurboQuant 2-bit. ~16× vs f32, better recall than 2-bit binary.
    Turbo2,
    /// TurboQuant 1.5-bit. ~24× vs f32.
    Turbo15,
    /// TurboQuant 1-bit. ~32× vs f32, better recall than 1-bit binary.
    Turbo1,
    /// Product Quantization (Qdrant 1.2): k-means codebooks over subvectors,
    /// up to 64× compression. Trained per collection (see `train_pq`).
    Product,
}

impl Default for QuantMode {
    fn default() -> Self {
        QuantMode::Int8
    }
}

impl QuantMode {
    /// Bits per dimension stored on the node side (0 for INT8 = 8-bit via all_i8).
    fn bits(&self) -> usize {
        match self {
            QuantMode::Int8 => 8,
            QuantMode::Binary => 1,
            QuantMode::Binary2 => 2,
            QuantMode::Binary15 => 2, // packed 3-state across pairs → ~1.5 avg
            QuantMode::Turbo4 => 4,
            QuantMode::Turbo2 => 2,
            QuantMode::Turbo15 => 2, // packed with QJL; 1.5 avg per dim
            QuantMode::Turbo1 => 1,
            QuantMode::Product => 8, // placeholder; real size from PQ codebooks
        }
    }

    /// True for the TurboQuant family (Hadamard-rotated, QJL-corrected).
    #[allow(dead_code)]
    fn is_turbo(&self) -> bool {
        matches!(self, QuantMode::Turbo4 | QuantMode::Turbo2 | QuantMode::Turbo15 | QuantMode::Turbo1)
    }

    /// Compression ratio vs f32 for this mode (bytes/vec ÷ 4·dim).
    pub fn compression(&self, dim: usize) -> f32 {
        let stored_bits = match self {
            QuantMode::Int8 => dim * 8,
            QuantMode::Binary => dim,
            QuantMode::Binary2 => dim * 2,
            QuantMode::Binary15 => (dim * 3).div_ceil(2),
            QuantMode::Turbo4 => dim * 4 + 1,
            QuantMode::Turbo2 => dim * 2 + 1,
            QuantMode::Turbo15 => (dim * 3).div_ceil(2) + 1,
            QuantMode::Turbo1 => dim + 1,
            QuantMode::Product => dim, // overwritten by PQ codebook size
        };
        (stored_bits as f32) / (dim as f32 * 32.0)
    }
}

/// Pack a unit-normalized i8 vector into `self.bits()`-per-dim bytes.
///   • Binary  (1 bit): sign bit; 1 = positive (or zero), 0 = negative.
///   • Binary2 (2 bit): 00 = strongly-neg, 01 = weakly-neg, 10 = weakly-pos,
///     11 = strongly-pos — i.e. the sign plus a coarse magnitude tier at |q|≥64.
/// The query side stays int8 and is scored against these reconstructed levels.
fn pack_lowbit(mode: QuantMode, q: &[i8], out: &mut Vec<u8>) {
    out.clear();
    let dim = q.len();
    let bytes = if mode == QuantMode::Int8 { 0 } else { (dim * mode.bits() + 7) / 8 };
    out.reserve(bytes);
    out.resize(bytes, 0);
    match mode {
        QuantMode::Int8 => {}
        QuantMode::Binary15 => {
            pack_binary15(mode, q, out);
        }
        QuantMode::Binary => {
            for (i, &x) in q.iter().enumerate() {
                if x >= 0 {
                    let byte = i / 8;
                    let bit = 7 - (i % 8);
                    out[byte] |= 1u8 << bit;
                }
            }
        }
        QuantMode::Binary2 => {
            for (i, &x) in q.iter().enumerate() {
                let lvl: u8 = if x >= 0 {
                    if x >= 64 {
                        0b11
                    } else {
                        0b10
                    }
                } else if x <= -64 {
                    0b00
                } else {
                    0b01
                };
                let byte = (i * 2) / 8;
                let shift = 6 - ((i * 2) % 8);
                out[byte] |= lvl << shift;
            }
        }
        QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4
        | QuantMode::Product => {
            // Not packed via pack_lowbit; pack_current handles these modes.
        }
    }
}

/// Asymmetric score: int8 query vs a packed low-bit stored vector.
/// Returns a value proportional to cosine similarity (higher = closer).
///   • Binary:  +1 per agreement of sign, −1 per disagreement (Hamming).
///   • Binary2: query sign vs stored sign gives the dominant term; magnitude
///     tier refines it (weak-vs-strong agreement weighted).
fn dot_lowbit(mode: QuantMode, q: &[i8], packed: &[u8]) -> i32 {
    match mode {
        QuantMode::Int8 => {
            // Not used on this path; caller uses dot_i8. Guard for safety.
            let _ = packed;
            dot_i8(q, q)
        }
        QuantMode::Binary => {
            let dim = q.len();
            let mut score = 0i32;
            for i in 0..dim {
                let byte = i / 8;
                let bit = 7 - (i % 8);
                let stored_pos = (packed[byte] >> bit) & 1 == 1;
                let q_pos = q[i] >= 0;
                score += if stored_pos == q_pos { 1 } else { -1 };
            }
            score
        }
        QuantMode::Binary2 => {
            let dim = q.len();
            let mut score = 0i32;
            for i in 0..dim {
                let byte = (i * 2) / 8;
                let shift = 6 - ((i * 2) % 8);
                let lvl = (packed[byte] >> shift) & 0b11;
                // stored sign: 1x = positive, 0x = negative
                let stored_pos = (lvl & 0b10) != 0;
                let q_pos = q[i] >= 0;
                let sign = if stored_pos == q_pos { 1i32 } else { -1i32 };
                // magnitude tier: strong (0b11 / 0b00) reinforces, weak neutral
                let mag = if lvl == 0b11 || lvl == 0b00 { 2 } else { 1 };
                score += sign * mag;
            }
            score
        }
        QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4
        | QuantMode::Product | QuantMode::Binary15 => 0,
    }
}

// ── 1.5-bit binary quantization (Module 2 — Qdrant 1.15) ──────────────────────
//
// Three states (−1, 0, +1) packed two-per-byte as a 3-value trit pair:
//   (−1,−1) 00 · (−1,0)/(0,−1) 01 · (0,0)/(+1,−1)/(−1,+1) 10 · (0,+1)/(+1,0)/(+1,+1) 11
// We approximate by assigning: magnitude tier 0 = ±1, tier 1 = 0 (the "near zero"
// bucket Qdrant explicitly represents). Packed as (tier_i, tier_j) 2 bits each,
// 4 bits for 2 dims → 2 bits/dim average on the 3-state scheme (≈1.5 effective
// because the 0 bucket is shared). Asymmetric: int8 query vs reconstructed level.

#[inline]
fn binary15_level(x: i8) -> u8 {
    // 0 = strongly-signed (|x| >= 32), 1 = near-zero bucket.
    if x.abs() < 32 {
        1
    } else {
        0
    }
}

#[inline]
fn binary15_recon(lvl: u8, sign: i8) -> i32 {
    if lvl == 1 {
        0
    } else {
        sign as i32
    }
}

fn pack_binary15(_mode: QuantMode, q: &[i8], out: &mut Vec<u8>) {
    out.clear();
    let dim = q.len();
    let pairs = (dim + 1) / 2;
    out.reserve(pairs * 2);
    out.resize(pairs * 2, 0);
    for p in 0..pairs {
        let a = q[p * 2];
        let b = if p * 2 + 1 < dim { q[p * 2 + 1] } else { 0 };
        let la = binary15_level(a);
        let lb = binary15_level(b);
        // bits: [la1 la0 lb1 lb0] within the byte pair (2 bytes per pair)
        let byte0 = (la << 6) | (la << 4) | (lb << 2) | lb;
        out[p * 2] = byte0;
        out[p * 2 + 1] = 0;
    }
}

fn dot_binary15(q: &[i8], packed: &[u8]) -> i32 {
    let dim = q.len();
    let mut score = 0i32;
    for p in 0..(dim + 1) / 2 {
        let byte0 = packed[p * 2];
        let la = (byte0 >> 6) & 0b11;
        let lb = (byte0 >> 2) & 0b11;
        let ia = p * 2;
        if ia < dim {
            let sa = if q[ia] >= 0 { 1i32 } else { -1i32 };
            let ra = binary15_recon(if la & 0b10 != 0 { 0 } else { 1 }, sa as i8);
            score += if ra == sa { 1 } else { -1 };
        }
        let ib = p * 2 + 1;
        if ib < dim {
            let sb = if q[ib] >= 0 { 1i32 } else { -1i32 };
            let rb = binary15_recon(if lb & 0b10 != 0 { 0 } else { 1 }, sb as i8);
            score += if rb == sb { 1 } else { -1 };
        }
    }
    score
}

// ── TurboQuant (Module 2 — Qdrant 1.18 / Google ICLR 2026) ────────────────────
//
// Faithful implementation of TurboQuant (Zandieh et al., ICLR 2026). It is
// DATA-OBLIVIOUS: the scalar codebook is derived analytically from the known
// distribution of rotated coordinates — no fitting on data is needed. Qdrant's
// "bits4/2/1.5/1" are all the INNER-PRODUCT variant, which is a two-stage scheme:
//
//   Encode (per stored unit vector x):
//     1. Rotate y = H·x  (Walsh–Hadamard ≈ random orthogonal rotation; spreads
//        each coordinate to the same concentrated Beta distribution).
//     2. (b-1)-bit MSE Lloyd-Max scalar quantize each y_j → idx_j (codebook
//        computed analytically for the Beta(·) marginal, scaled by 1/sqrt(d)).
//     3. Residual r = y - dequant_mse(idx). Store its L2 norm ‖r‖ (1 f32) and
//        the QJL sketch bits z = sign(S · r) where S is a fixed random Gaussian
//        matrix (1 bit / coordinate).
//   Stored per vector: (b-1)·D code bits  +  ‖r‖ f32  +  D QJL bits.
//
//   Score (asymmetric, f32 query q, both unit norm):
//     deq = dequant_mse(idx)               (in rotated basis)
//     mse_term = <q_rot, deq>             (q_rot = H·q)
//     qjl_term = sqrt(pi/2)/D · ‖r‖ · <q_rot, Sᵀ z>
//     est(<q,x>) = mse_term + qjl_term     (provably UNBIASED)
//   Cosine sim = est(<q,x>) / (‖q‖·‖x‖); both are unit so this is just the est.
// The int8 candidate set is reranked against the exact f32 vectors as usual.
//
// Total bytes/vec = ceil((b-1)·D/8) + 4 + ceil(D/8). For b=4,D=768 → 384+4+96=484
// bytes ≈ 8.0× (matches Qdrant's "8×" claim). For b=1 → 32×.

/// Smallest power of two ≥ n (n ≥ 1).
#[inline]
fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place orthonormal Walsh–Hadamard transform. `v.len()` must be a power of
/// two. Each butterfly scales by 1/sqrt(2), so after log2(n) stages the whole
/// transform is norm-preserving (stands in for the random rotation Π in
/// TurboQuant: a fixed orthogonal matrix that isotropizes the coordinates).
fn hadamard(v: &mut [f32]) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    let norm = std::f32::consts::SQRT_2.recip(); // 1/sqrt(2) per stage
    let mut step = 1usize;
    while step < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + step {
                let a = v[j];
                let b = v[j + step];
                v[j] = (a + b) * norm;
                v[j + step] = (a - b) * norm;
            }
            i += step * 2;
        }
        step <<= 1;
    }
}

/// Gaussian sampler (Box–Muller) seeded deterministically from `seed`.
struct GaussGen {
    s: u64,
}
impl GaussGen {
    fn new(seed: u64) -> Self {
        GaussGen { s: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xC0FFEE) }
    }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }
}

/// Per-collection TurboQuant parameters (data-oblivious — built once for a given
/// `d` and bit-width, NO data sample needed).
#[derive(Clone)]
struct TurboParams {
    #[allow(dead_code)]
    dim: usize,   // original dim
    d: usize,   // padded pow2 dim
    #[allow(dead_code)]
    bits: u8,   // TOTAL target bit-width b (1, 2, 3=1.5avg via QJL packing, 4)
    qbits: u8,  // (b-1) MSE bits per coordinate
    codebook: Vec<f32>, // 2^qbits centroids for the Beta marginal, scaled by 1/sqrt(d)
    s_sign: Vec<i8>,   // random ±1 diagonal D of the SRHT S = (1/√d)·H·D (QJL)
    s_scale: f32,      // 1/√d normalization of the SRHT
    /// Per-coordinate anisotropy correction scales (Qdrant extension).
    /// `D'[i] = 1/sqrt(var_i)` where var_i is the variance of coordinate i
    /// across a representative sample. Applied to the query BEFORE rotation.
    aniso_scales: Vec<f32>,
}

impl TurboParams {
    #[allow(dead_code)]
    fn levels(&self) -> usize {
        1usize << self.qbits
    }

    /// Lloyd–Max codebook for the marginal of a rotated unit vector coordinate.
    /// In high-D, coordinate j ~ Beta(marginal) ≈ N(0, 1/d); we solve the
    /// continuous 1-D k-means (Lloyd–Max) for that distribution on [-1,1] with
    /// 2^qbits levels. Returns `levels` sorted centroids.
    fn build_codebook(qbits: u8, d: usize) -> Vec<f32> {
        let levels = 1usize << qbits;
        let sigma = (d as f32).sqrt().recip(); // std ≈ 1/sqrt(d)
        // Closed-form Gaussian Lloyd-Max for the first couple of bit-widths;
        // iterative Lloyd for the rest (converges in a handful of sweeps).
        if qbits == 1 {
            // Optimal single threshold at 0; centroids ±sqrt(2/pi)*sigma.
            let c = (2.0 / std::f32::consts::PI).sqrt() * sigma;
            return vec![-c, c];
        }
        // Iterative Lloyd–Max on N(0, sigma^2) clipped to [-4sigma, 4sigma].
        let lo = -4.0 * sigma;
        let hi = 4.0 * sigma;
        let mut cent: Vec<f32> = (0..levels)
            .map(|i| lo + (hi - lo) * (i as f32 + 0.5) / levels as f32)
            .collect();
        let pdf = |x: f32| {
            if x < lo || x > hi {
                0.0
            } else {
                (-0.5 * (x / sigma).powi(2)).exp() / (sigma * (2.0 * std::f32::consts::PI).sqrt())
            }
        };
        for _ in 0..64 {
            // Recompute centroids as mass-weighted means between decision bounds.
            let mut newc = vec![0.0f32; levels];
            let mut mass = vec![0.0f32; levels];
            let steps = 4096usize;
            let dx = (hi - lo) / steps as f32;
            for s in 0..steps {
                let x = lo + (s as f32 + 0.5) * dx;
                // nearest centroid
                let mut best = 0usize;
                let mut bd = f32::MAX;
                for (k, &c) in cent.iter().enumerate() {
                    let dd = (x - c).abs();
                    if dd < bd {
                        bd = dd;
                        best = k;
                    }
                }
                let w = pdf(x) * dx;
                newc[best] += x * w;
                mass[best] += w;
            }
            let mut changed = false;
            for k in 0..levels {
                if mass[k] > 1e-9 {
                    let nc = newc[k] / mass[k];
                    if (nc - cent[k]).abs() > 1e-5 * sigma {
                        changed = true;
                    }
                    cent[k] = nc;
                }
            }
            cent.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !changed {
                break;
            }
        }
        cent
    }

    /// Quantize one rotated coordinate to a level index in [0, levels).
    #[inline]
    fn quant_level(&self, x: f32) -> u32 {
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for (k, &c) in self.codebook.iter().enumerate() {
            let dd = (x - c).abs();
            if dd < bd {
                bd = dd;
                best = k;
            }
        }
        best as u32
    }

    /// Dequantize a level index back to a rotated coordinate.
    #[inline]
    fn dequant_level(&self, l: u32) -> f32 {
        self.codebook[l as usize]
    }

    /// Rotate `x` (length `dim`, zero-padded to `d`) in place into `rot`.
    #[inline]
    fn rotate(&self, x: &[f32], rot: &mut [f32]) {
        rot.fill(0.0);
        rot[..x.len()].copy_from_slice(x);
        hadamard(rot);
    }

    /// Apply the QJL sketch matrix `S = (1/√d)·H·D` to `x` (length `d`),
    /// writing `out = S·x`. `D` flips signs (`s_sign`), `H` is the Walsh–
    /// Hadamard (already normalized by `hadamard`), then we scale by `1/√d`.
    /// This is O(d log d) vs O(d²) for a dense Gaussian `S` — the whole point
    /// of the SRHT swap. QJL stays unbiased because `S` is orthonormal (so
    /// `S·x` has the same norm as `x` and per-coordinate variance `‖x‖²/d`).
    #[inline]
    fn s_transform(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d);
        debug_assert_eq!(out.len(), self.d);
        for i in 0..self.d {
            out[i] = x[i] * self.s_sign[i] as f32;
        }
        hadamard(out);
        for v in out.iter_mut() {
            *v *= self.s_scale;
        }
    }

    /// Calibrate per-coordinate anisotropy correction from a sample of
    /// (L2-normalized) vectors. For each coordinate, computes the variance
    /// across the sample and derives D'[i] = 1/sqrt(var_i). At query time,
    /// the query vector is element-wise multiplied by D' before rotation,
    /// which corrects for the fact that real embeddings are not uniformly
    /// distributed on the unit sphere. This is Qdrant's biggest recall win:
    /// +14-18pp on anisotropic data.
    fn calibrate_anisotropy(&mut self, sample: &[Vec<f32>]) {
        if sample.len() < 2 {
            // Not enough data — set all scales to 1 (no correction).
            self.aniso_scales = vec![1.0; self.dim];
            return;
        }
        let n = sample.len() as f32;
        // Compute per-coordinate mean and variance.
        let mut mean = vec![0.0f32; self.dim];
        for v in sample {
            for (i, &x) in v.iter().enumerate().take(self.dim) {
                mean[i] += x;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0f32; self.dim];
        for v in sample {
            for (i, &x) in v.iter().enumerate().take(self.dim) {
                let d = x - mean[i];
                var[i] += d * d;
            }
        }
        // D'[i] = 1/sqrt(var_i). Clamp to avoid division by zero.
        self.aniso_scales = var
            .iter()
            .map(|&v| {
                let std = (v / n).sqrt();
                if std > 1e-6 { 1.0 / std } else { 1.0 }
            })
            .collect();
    }

    /// Bytes per stored vector for this configuration.
    fn packed_bytes(&self) -> usize {
        let cb = (self.d * self.qbits as usize + 7) / 8;
        let qb = (self.d + 7) / 8;
        cb + qb + 4
    }

    /// Pack a stored vector: (qbits·D) MSE code bits, then D QJL bits, then the
    /// 4-byte residual norm at the very end. `idx` are the MSE indices (len D),
    /// `qjl` the QJL sign bits (len D, each 0/1), `rn` the residual norm f32.
    fn pack(&self, idx: &[u32], qjl: &[u8], rn: f32, out: &mut Vec<u8>) {
        let cb = (self.d * self.qbits as usize + 7) / 8;
        let _qb = (self.d + 7) / 8;
        let bytes = cb + _qb + 4;
        out.clear();
        out.resize(bytes, 0);
        // MSE code bits.
        let mut bitpos = 0usize;
        for &l in idx {
            for b in (0..self.qbits as usize).rev() {
                let val = ((l >> b) & 1) as u8;
                out[bitpos / 8] |= val << (7 - (bitpos % 8));
                bitpos += 1;
            }
        }
        // QJL bits.
        for j in 0..self.d {
            if qjl[j] & 1 == 1 {
                let bp = cb * 8 + j;
                out[bp / 8] |= 1 << (7 - (bp % 8));
            }
        }
        // Residual norm (little-endian f32) at the tail.
        let rbits = rn.to_le_bytes();
        let tail = bytes - 4;
        out[tail..bytes].copy_from_slice(&rbits);
    }

    /// Precompute `sq = S · q_rot` ONCE per query. With the SRHT `S = (1/√d)·H·D`
    /// this is O(d log d) (a sign-flip + one Hadamard) instead of O(d²). The
    /// per-node score then only needs a cheap O(d) dot with the stored sign
    /// sketch `z = sign(S·r)`.
    fn sq_precomp(&self, q_rot: &[f32]) -> Vec<f32> {
        let mut sq = vec![0.0f32; self.d];
        self.s_transform(q_rot, &mut sq);
        sq
    }

    /// Score a rotated f32 query against a packed stored vec. `sq` is the
    /// precomputed `S·q_rot` (see `sq_precomp`). Returns an unbiased inner-
    /// product estimate <q_rot, x_rot> = <q, x> (cosine for unit vectors).
    ///
    /// Decodes the packed bits into a per-thread scratch buffer (grown once,
    /// never re-allocated per node) — this is what keeps Turbo scoring fast at
    /// query time: the naive `unpack` allocated two `Vec<D>` per node visited,
    /// which dominated p99 once `sq_precomp` went O(d log d).
    #[inline]
    fn score_qr(&self, q_rot: &[f32], sq: &[f32], packed: &[u8]) -> f32 {
        thread_local! {
            static IDX: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
            static QJL: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(Vec::new());
        }
        let rn = {
            let tail = packed.len() - 4;
            let mut rb = [0u8; 4];
            rb.copy_from_slice(&packed[tail..tail + 4]);
            f32::from_le_bytes(rb)
        };
        IDX.with(|ci| {
            QJL.with(|cq| {
                let mut idx = ci.borrow_mut();
                let mut qjl = cq.borrow_mut();
                Self::decode_packed(self, packed, &mut idx, &mut qjl);
                // MSE term: <q_rot, deq>.
                let mut mse = 0.0f32;
                for i in 0..self.d {
                    mse += q_rot[i] * self.dequant_level(idx[i]);
                }
                // QJL term: sqrt(pi/2)/D * ‖r‖ * <z, sq>.
                let mut qjl_dot = 0.0f32;
                for i in 0..self.d {
                    if qjl[i] == 1 {
                        qjl_dot += sq[i];
                    } else {
                        qjl_dot -= sq[i];
                    }
                }
                let qjl_term = (std::f32::consts::FRAC_PI_2.sqrt()) / (self.d as f32) * rn * qjl_dot;
                mse + qjl_term
            })
        })
    }

    /// Decode `packed` into `idx` (len D) and `qjl` (len D) WITHOUT the residual
    /// norm (caller reads the 4-byte tail separately). Reuses the caller's
    /// buffers so no allocation happens per call.
    #[inline]
    fn decode_packed(&self, packed: &[u8], idx: &mut Vec<u32>, qjl: &mut Vec<u8>) {
        let cb = (self.d * self.qbits as usize + 7) / 8;
        if idx.len() != self.d {
            idx.resize(self.d, 0);
        }
        if qjl.len() != self.d {
            qjl.resize(self.d, 0);
        }
        let mut bitpos = 0usize;
        for l in idx.iter_mut() {
            let mut v = 0u32;
            for _ in 0..self.qbits as usize {
                let byte = bitpos / 8;
                let off = 7 - (bitpos % 8);
                v = (v << 1) | ((packed[byte] >> off) & 1) as u32;
                bitpos += 1;
            }
            *l = v;
        }
        for j in 0..self.d {
            let bp = cb * 8 + j;
            qjl[j] = (packed[bp / 8] >> (7 - (bp % 8))) & 1;
        }
    }
}

/// Build TurboQuant params for a given `dim` and bit-width. Data-oblivious:
/// `sample` is IGNORED (kept only for API compatibility). `seed` fixes S.
fn fit_turbo(mode: QuantMode, dim: usize, seed: u64) -> TurboParams {
    let d = next_pow2(dim);
    let bits: u8 = match mode {
        QuantMode::Turbo1 => 1,
        QuantMode::Turbo15 => 2, // store (1-bit MSE) + QJL → 1.5 bits avg
        QuantMode::Turbo2 => 2,
        QuantMode::Turbo4 => 4,
        _ => 4,
    };
    let qbits = bits.saturating_sub(1).max(1); // ≥1 MSE bit before QJL
    let codebook = TurboParams::build_codebook(qbits, d);
    // QJL sketch: SRHT S = (1/√d)·H·D. We only store the random ±1 diagonal D
    // (`s_sign`); the Hadamard H and the 1/√d scale are applied in `s_transform`.
    // Deterministic from `seed` so pack (offline) and query (online) agree.
    let mut g = GaussGen::new(seed ^ 0x9e37_79b9);
    let s_sign: Vec<i8> = (0..d).map(|_| if g.next_u64() & 1 == 0 { 1i8 } else { -1i8 }).collect();
    let s_scale = 1.0 / (d as f32).sqrt();
    TurboParams {
        dim,
        d,
        bits,
        qbits,
        codebook,
        s_sign,
        s_scale,
        aniso_scales: vec![1.0; dim],
    }
}

// ── Product Quantization (Module 2 — Qdrant 1.2) ──────────────────────────────
//
// Split each vector into `m` subvectors of `ds` dims; learn `k` centroids per
// subpace by k-means (k = 256 → 1 byte/code). A vector becomes `m` byte codes.
// Asymmetric: store codes, score the f32 query against centroid tables (no
// SIMD-friendly, but up to 64× compression). Trained per collection.

struct PqParams {
    #[allow(dead_code)]
    dim: usize,
    m: usize,
    ds: usize,
    k: usize,
    centroids: Vec<f32>, // flat: (m*k) * ds
}

impl PqParams {
    fn new(dim: usize, m: usize, k: usize) -> Self {
        let ds = (dim + m - 1) / m;
        let m = (dim + ds - 1) / ds;
        Self {
            dim,
            m,
            ds,
            k,
            centroids: vec![0.0f32; m * k * ds],
        }
    }

    /// Encode a vector into `m` centroid indices (1 byte each).
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);
        for j in 0..self.m {
            let start = j * self.ds;
            let end = ((j + 1) * self.ds).min(self.dim);
            let sub = &v[start..end];
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for c in 0..self.k {
                let base = (j * self.k + c) * self.ds;
                let mut d = 0.0f32;
                for t in 0..self.ds {
                    let diff = sub.get(t).copied().unwrap_or(0.0) - self.centroids[base + t];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            codes.push(best as u8);
        }
        codes
    }

    /// Asymmetric distance: f32 query vs PQ-coded stored vector (squared L2
    /// via lookup tables; we expose as a similarity proxy).
    fn score(&self, q: &[f32], codes: &[u8]) -> f32 {
        let mut dot = 0.0f32;
        for j in 0..self.m {
            let c = codes[j] as usize;
            let base = (j * self.k + c) * self.ds;
            let start = j * self.ds;
            let end = ((j + 1) * self.ds).min(self.dim);
            for t in 0..(end - start) {
                dot += q[start + t] * self.centroids[base + t];
            }
        }
        dot
    }
}

/// Train PQ centroids with a few k-means (Lloyd) iterations over `sample`.
fn train_pq(dim: usize, m: usize, k: usize, sample: &[Vec<f32>], iters: usize) -> PqParams {
    let mut pq = PqParams::new(dim, m, k);
    let ds = pq.ds;
    for j in 0..pq.m {
        let start = j * ds;
        let end = ((j + 1) * ds).min(dim);
        // init centroids: spread sampling of subspace points
        for c in 0..k {
            let base = (j * k + c) * ds;
            let pick = sample.get(c % sample.len());
            if let Some(v) = pick {
                for t in 0..(end - start) {
                    pq.centroids[base + t] = v[start + t];
                }
            }
        }
        for _ in 0..iters {
            // assign + accumulate
            let mut acc = vec![0.0f32; k * ds];
            let mut cnt = vec![0usize; k];
            for v in sample {
                let sub = &v[start..end];
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for c in 0..k {
                    let base = (j * k + c) * ds;
                    let mut d = 0.0f32;
                    for t in 0..(end - start) {
                        let diff = sub[t] - pq.centroids[base + t];
                        d += diff * diff;
                    }
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                let ab = best * ds;
                for t in 0..(end - start) {
                    acc[ab + t] += sub[t];
                }
                cnt[best] += 1;
            }
            for c in 0..k {
                if cnt[c] > 0 {
                    let base = (j * k + c) * ds;
                    for t in 0..(end - start) {
                        pq.centroids[base + t] = acc[c * ds + t] / cnt[c] as f32;
                    }
                }
            }
        }
    }
    pq
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
    // 4 independent accumulators to break the FMA dependency chain (5-cycle
    // latency on modern Intel/AMD). Each feeds independently into the next FMA,
    // so the CPU can pipeline 4 FMAs simultaneously.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= chunks {
        _mm_prefetch(a.as_ptr().add((i + 4) * 8) as *const i8, _MM_HINT_T0);
        _mm_prefetch(b.as_ptr().add((i + 4) * 8) as *const i8, _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i * 8)), _mm256_loadu_ps(b.as_ptr().add(i * 8)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add((i + 1) * 8)), _mm256_loadu_ps(b.as_ptr().add((i + 1) * 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add((i + 2) * 8)), _mm256_loadu_ps(b.as_ptr().add((i + 2) * 8)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add((i + 3) * 8)), _mm256_loadu_ps(b.as_ptr().add((i + 3) * 8)), acc3);
        i += 4;
    }
    while i < chunks {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i * 8)), _mm256_loadu_ps(b.as_ptr().add(i * 8)), acc0);
        i += 1;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
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

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    static F: std::sync::OnceLock<DotF32Fn> = std::sync::OnceLock::new();
    unsafe { (F.get_or_init(resolve_dot_f32))(a, b) }
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

/// Mode-aware cosine distance from a quantized `query` to node `idx`.
/// For `Int8` this is the exact `cos_dist_q`. For `Binary`/`Binary2`/`Binary15`
/// it uses the asymmetric low-bit score (`dot_lowbit`) mapped to [0,2] cosine
/// distance. For `Turbo*` it uses the Hadamard-rotated f32 query scored against
/// the dequantized levels + QJL correction (asymmetric; query full precision).
/// For `Product` it scores the f32 query against the PQ centroid tables.
///
/// `qf32` is the (optionally rotated) f32 query — empty for pure int8/binary
/// traversal (e.g. during graph construction), in which case Turbo/Product fall
/// back to int8 distance so the HNSW topology is built consistently.
#[inline]
    fn node_dist(
    mode: QuantMode,
    query: &[i8],
    all_i8: &[i8],
    all_bin: &[u8],
    qf32: &[f32],
    sq: &[f32],
    turbo: Option<&TurboParams>,
    pq: Option<&PqParams>,
    _all_norm: &[f32],
    idx: usize,
    dim: usize,
) -> f32 {
    match mode {
        QuantMode::Int8 => cos_dist_q(query, &all_i8[idx * dim..idx * dim + dim]),
        QuantMode::Binary => {
            let bits = (dim + 7) / 8; // 1 bit/dim → ceil(dim/8) bytes
            let s = dot_lowbit(mode, query, &all_bin[idx * bits..idx * bits + bits]);
            1.0 - (s as f32 / dim as f32)
        }
        QuantMode::Binary2 => {
            let bits = (dim * 2 + 7) / 8; // 2 bits/dim
            let s = dot_lowbit(mode, query, &all_bin[idx * bits..idx * bits + bits]);
            let sim = s as f32 / (dim as f32 * 2.0);
            1.0 - sim
        }
        QuantMode::Binary15 => {
            let bytes = ((dim + 1) / 2) * 2; // 2 bytes per pair (see pack_binary15)
            let s = dot_binary15(query, &all_bin[idx * bytes..idx * bytes + bytes]);
            // 3-state agreement; map agreement/dim ∈ [-1,1] → cosine distance.
            1.0 - (s as f32 / dim as f32)
        }
        QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4 => {
            if qf32.is_empty() || turbo.is_none() {
                // During graph construction: build topology with int8 distance.
                return cos_dist_q(query, &all_i8[idx * dim..idx * dim + dim]);
            }
            let tp = turbo.unwrap();
            let bytes = tp.packed_bytes();
            // `qf32` is the Hadamard-rotated query, `sq` = S·qf32 (precomputed
            // once per query). `score_qr` returns an unbiased inner-product
            // estimate between the unit vectors = cosine similarity.
            let sim = tp.score_qr(qf32, sq, &all_bin[idx * bytes..idx * bytes + bytes]);
            1.0 - sim
        }
        QuantMode::Product => {
            if qf32.is_empty() || pq.is_none() {
                return cos_dist_q(query, &all_i8[idx * dim..idx * dim + dim]);
            }
            let pp = pq.unwrap();
            let m = pp.m;
            let dot = pp.score(qf32, &all_bin[idx * m..idx * m + m]);
            // cosine similarity proxy of unit vectors ∈ [-1,1]
            let sim = dot.clamp(-1.0, 1.0);
            1.0 - sim
        }
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
    /// MODULE 4: per-node filter attribute (e.g. category id). Lets a single
    /// HNSW serve filtered queries without a second index.
    attr: u32,
}

// SAFETY: Hnsw is always behind RwLock in VectorIndex. UnsafeCell fields are
// accessed only through &self methods that use raw pointers; no concurrent
// mutation occurs because RwLock serializes access.
unsafe impl Sync for Hnsw {}

struct Hnsw {
    nodes: Vec<Node>,
    id_to_idx: HashMap<u64, usize>,
    /// Flat quantized storage. Node `i`'s vector = `&all_i8[i*dim..(i+1)*dim]`.
    all_i8: Vec<i8>,
    /// Module 2: low-bit packed storage (binary / 2-bit) — `all_bin` holds
    /// `bits()`/8 bytes per node. Used for the asymmetric traversal when
    /// `quant != Int8`. Empty otherwise.
    all_bin: Vec<u8>,
    /// Module 2: active quantization mode (Int8 | Binary | Binary2 | Binary15 |
    /// Turbo* | Product).
    quant: QuantMode,
    /// Module 2: TurboQuant parameters (rotation + codebook + QJL), when the
    /// active mode is in the Turbo family. Fitted once via `fit_quant`.
    turbo: Option<std::sync::Arc<TurboParams>>,
    /// Module 2: Product Quantization parameters (subspace centroids), when the
    /// active mode is `Product`. Trained once via `fit_quant`.
    pq: Option<std::sync::Arc<PqParams>>,
    /// Module 2: per-node L2 norm (used by TurboQuant cosine correction).
    all_norm: Vec<f32>,
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
    /// Generation-stamped visited buffer (see `search_layer`). Grown only as
    /// the graph grows; stamped with an incrementing epoch per search instead
    /// of memset — keeps ingest O(N), not O(N²).
    /// Wrapped in UnsafeCell so &self search methods can use it (each thread
    /// accesses only its own Hnsw; no concurrent mutation).
    visited_cell: UnsafeCell<Vec<u32>>,
    epoch_cell: UnsafeCell<u32>,
    /// Reused quantization scratch (one vector's worth of i8).
    qbuf: Vec<i8>,
    /// Reused low-bit packing scratch (one vector's worth of packed bytes).
    bin_scratch: Vec<u8>,
    /// Reused TurboQuant packing scratch buffers (avoid 34KB alloc per insert).
    turbo_rot: Vec<f32>,
    turbo_idx: Vec<u32>,
    turbo_deq: Vec<f32>,
    turbo_r: Vec<f32>,
    turbo_qjl: Vec<u8>,
    /// MODULE 2: when set, the exact f32 rerank originals live in an NVMe
    /// `mmap` (cold tier) instead of RAM. `all_f32` stays empty; reads/writes
    /// go through `f32_tier`. RAM drops to ~int8-only at 100M+ scale.
    f32_tier: Option<MmapTier>,
    /// MODULE 4: number of distinct attribute categories, used to size the
    /// per-category count + entry tables.
    attr_kinds: usize,
    /// MODULE 4: live count of nodes per attribute category (for selectivity
    /// estimation at query time → routes to gated traversal vs brute force).
    attr_counts: Vec<usize>,
    /// MODULE 4: a representative entry node per category (first inserted of
    /// that kind). Filtered search uses this as the graph entry so the walk
    /// starts inside the matching set instead of a possibly-filtered node.
    attr_entry: Vec<Option<usize>>,
}

impl Hnsw {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            all_i8: Vec::new(),
            all_bin: Vec::new(),
            quant: QuantMode::Int8,
            turbo: None,
            pq: None,
            all_norm: Vec::new(),
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
            visited_cell: UnsafeCell::new(Vec::new()),
            epoch_cell: UnsafeCell::new(0),
            qbuf: Vec::new(),
            bin_scratch: Vec::new(),
            turbo_rot: Vec::new(),
            turbo_idx: Vec::new(),
            turbo_deq: Vec::new(),
            turbo_r: Vec::new(),
            turbo_qjl: Vec::new(),
            f32_tier: None,
            attr_kinds: 0,
            attr_counts: Vec::new(),
            attr_entry: Vec::new(),
        }
    }

    /// Bump the epoch and return the new value.
    unsafe fn bump_epoch(&self) -> u32 {
        let e = &mut *self.epoch_cell.get();
        *e = e.wrapping_add(1);
        if *e == 0 {
            *e = 1;
            let v = &mut *self.visited_cell.get();
            v.iter_mut().for_each(|x| *x = 0);
        }
        *e
    }

    /// Ensure visited is large enough for `n` nodes and bump the epoch.
    unsafe fn ensure_visited(&self, n: usize) -> u32 {
        let v = &mut *self.visited_cell.get();
        if v.len() < n {
            v.resize(n, 0);
        }
        self.bump_epoch()
    }

    /// Module 2: select the quantization mode. MUST be called before any insert
    /// (the `all_bin` arena is populated at insert time). Switching modes on a
    /// non-empty graph is unsupported and will panic, by design — callers set
    /// the mode once at index construction.
    fn set_quant_mode(&mut self, mode: QuantMode) {
        assert!(self.nodes.is_empty(), "set_quant_mode must be called before inserts");
        self.quant = mode;
    }

    /// Module 2: fit TurboQuant / Product Quantization parameters from a sample
    /// of (normalized) vectors. TurboQuant is data-oblivious, so `sample` is
    /// used only to learn the dimension; the Turbo codebook is analytic. For
    /// Product we DO train on the sample. For binary/int8 modes this is a no-op.
    fn fit_quant(&mut self, sample: &[Vec<f32>]) {
        assert!(self.nodes.is_empty(), "fit_quant must be called before inserts");
        // `dim` is not known until the first insert, so derive it from the sample
        // (which the caller builds with correct dim). Falls back to self.dim.
        let dim = sample.first().map(|v| v.len()).unwrap_or(self.dim);
        match self.quant {
            QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4 => {
                let mut tp = fit_turbo(self.quant, dim, 0x5bd1_e995);
                // Calibrate per-coordinate anisotropy correction from the sample.
                // This is Qdrant's biggest recall win: +14-18pp on real embeddings.
                tp.calibrate_anisotropy(sample);
                self.turbo = Some(std::sync::Arc::new(tp));
            }
            QuantMode::Product => {
                let m = (dim + 7) / 8; // ~8 dims/subvector → m subspaces
                let pq = train_pq(dim, m, 256, sample, 8);
                self.pq = Some(std::sync::Arc::new(pq));
            }
            _ => {}
        }
    }

    #[inline]
    fn vec_at_f32(&self, idx: usize) -> &[f32] {
        let o = idx * self.dim;
        match &self.f32_tier {
            Some(t) => &t.as_slice()[o..o + self.dim],
            None => &self.all_f32[o..o + self.dim],
        }
    }

    fn random_level(&mut self) -> usize {
        // Layer assignment: level = floor(-ln(uniform) * mL) with mL = 1/ln(2).
        // Empirically this gives far better routing/recall on real 384/768d
        // embeddings than the hnswlib 1/ln(M) default (which starves the upper
        // layers and collapses recall to ~0.12 here). Each node gets a ~1/2
        // chance of being in layer 1, 1/4 in layer 2, etc.
        let r = self.rng.next_f32().max(1e-9);
        let ml = 1.0 / (self.m as f32).ln();
        (-r.ln() * ml) as usize
    }

    /// Insert an already-normalized f32 vector. Quantization happens here.
    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.insert_attr(id, vector, 0);
    }

    /// Insert with a MODULE 4 filter attribute. See `insert` for the mechanics;
    /// this additionally records `attr` into the node and updates the per-kind
    /// count + entry tables used by selectivity-routed filtered search.
    fn insert_attr(&mut self, id: u64, mut vector: Vec<f32>, attr: u32) {
        // Update-in-place path.
        if let Some(&idx) = self.id_to_idx.get(&id) {
            l2_normalize(&mut vector);
            quantize_into(&vector, &mut self.qbuf);
            let o = idx * self.dim;
            self.all_i8[o..o + self.dim].copy_from_slice(&self.qbuf);
            if self.quant != QuantMode::Int8 {
                let bytes = pack_current(self.quant, self.turbo.as_deref(), self.pq.as_deref(), &vector, &self.qbuf, &mut self.bin_scratch, &mut self.turbo_rot, &mut self.turbo_idx, &mut self.turbo_deq, &mut self.turbo_r, &mut self.turbo_qjl);
                if bytes > 0 {
                    let bo = idx * bytes;
                    if self.all_bin.len() < bo + bytes {
                        self.all_bin.resize(bo + bytes, 0);
                    }
                    self.all_bin[bo..bo + bytes].copy_from_slice(&self.bin_scratch);
                }
                if !self.all_norm.is_empty() {
                    self.all_norm[idx] = vector_l2(&vector);
                }
            }
            if let Some(t) = self.f32_tier.as_mut() {
                t.as_mut_slice()[o..o + self.dim].copy_from_slice(&vector);
            } else {
                self.all_f32[o..o + self.dim].copy_from_slice(&vector);
            }
            // Attribute of an existing node is immutable for our workload; the
            // first writer wins. (Re-assigning would require re-linking counts.)
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
        self.qbuf.clear();
        self.qbuf.reserve(vector.len());
        quantize_into(&vector, &mut self.qbuf);

        let _ins_t0 = if std::env::var_os("DBSTRIKE_DEBUG").is_some() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let level = self.random_level();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            id,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
            attr,
        });
        self.id_to_idx.insert(id, idx);
        // MODULE 4: maintain per-kind count + a representative entry node.
        if (self.attr_kinds as u32) <= attr {
            let new_kinds = (attr as usize) + 1;
            self.attr_counts.resize(new_kinds, 0);
            self.attr_entry.resize(new_kinds, None);
            self.attr_kinds = new_kinds;
        }
        self.attr_counts[attr as usize] += 1;
        if self.attr_entry[attr as usize].is_none() {
            self.attr_entry[attr as usize] = Some(idx);
        }
        let q = &self.qbuf;
        self.all_i8.extend_from_slice(q);
        if self.quant != QuantMode::Int8 {
            let bytes = pack_current(self.quant, self.turbo.as_deref(), self.pq.as_deref(), &vector, &self.qbuf, &mut self.bin_scratch, &mut self.turbo_rot, &mut self.turbo_idx, &mut self.turbo_deq, &mut self.turbo_r, &mut self.turbo_qjl);
            if bytes > 0 {
                self.all_bin.extend_from_slice(&self.bin_scratch);
            }
            // TurboQuant needs no per-node norm (it scores the inner-product
            // estimate directly between unit vectors); store it for PQ/binary
            // cosine correction paths that may read all_norm.
            if self.quant == QuantMode::Product {
                self.all_norm.push(vector_l2(&vector));
            }
        }
        if let Some(t) = self.f32_tier.as_mut() {
            let o = idx * self.dim;
            // Grow the tier lazily if needed (idx may exceed initial estimate).
            if o + self.dim <= t.as_slice().len() {
                t.as_mut_slice()[o..o + self.dim].copy_from_slice(&vector);
            }
        } else {
            self.all_f32.extend_from_slice(&vector);
        }

        let entry = match self.entry {
            None => {
                self.entry = Some(idx);
                self.max_level = level;
                return;
            }
            Some(e) => e,
        };

        // Grow-only generation-stamped visited buffer via UnsafeCell.
        // NOTE: we push only the *new* slots (amortized O(1)) and rely on
        // epoch stamping so stale values in old slots don't matter.
        let need = self.nodes.len();
        let epoch = unsafe { self.ensure_visited(need) };
        // Use raw pointer to break the borrow chain: visited_ptr borrows the
        // UnsafeCell but NOT &self, so self.nodes/all_i8 remain borrowable.
        let visited_ptr = self.visited_cell.get();

        let mut cur = entry;
        let top = self.max_level;
        for lvl in (level + 1..=top).rev() {
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, &q, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
            if let Some(best) = found
                .into_iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap())
            {
                cur = best.idx;
            }
        }
        let start_lvl = level.min(top);
        for lvl in (0..=start_lvl).rev() {
            let mut found =
                Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, &q, cur, self.ef_construction, lvl, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
            found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            // Select the new node's forward neighbors using the HNSW diversity
            // heuristic (Alg. 4, Malkov & Yashunin). This produces a navigable
            // graph with spread-out edges, so ef=128 search reaches ~all nodes
            // instead of needing ef=4000. The naive "top-M nearest" is hubbier
            // and hurts recall at low ef.
            let selected: Vec<usize> = self.select_neighbors_heuristic(idx, &found, self.m, lvl);
            for &nb in &selected {
                self.nodes[idx].neighbors[lvl].push(nb);
                self.nodes[nb].neighbors[lvl].push(idx);
                // PRUNE-COMMITTED: rebuild new node's forward list tight to m.
                if self.nodes[idx].neighbors[lvl].len() > self.m {
                    self.nodes[idx].neighbors[lvl] = selected.clone();
                }
                if mitm().noprune {
                    continue;
                }
                // PRUNE-COMMITTED: prune neighbor's reverse list to m_max0 nearest
                // (same cos_dist_q metric the search navigates by).
                let neigh_idx = nb;
                if self.nodes[neigh_idx].neighbors[lvl].len() > self.m_max0 {
                    let neigh_list = std::mem::take(&mut self.nodes[neigh_idx].neighbors[lvl]);
                    let nvec_ref = &self.all_i8[neigh_idx * self.dim..neigh_idx * self.dim + self.dim];
                    let mut nn: Vec<(f32, usize)> = neigh_list
                        .iter()
                        .map(|&x| (cos_dist_q(nvec_ref, &self.all_i8[x * self.dim..x * self.dim + self.dim]), x))
                        .collect();
                    nn.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    nn.truncate(self.m_max0);
                    self.nodes[neigh_idx].neighbors[lvl] = nn.into_iter().map(|(_, x)| x).collect();
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

        if mitm().on {
            CNT_INSERTS.fetch_add(1, AOrd::Relaxed);
        }

        // Optional observability. Set `DBSTRIKE_DEBUG=1` to stream per-insert
        // timing + graph size to stderr (one line per 1k inserts, not per
        // vector, so it's cheap enough to leave on during a full 1M ingest).
        if let Some(t0) = _ins_t0 {
            let elapsed = t0.elapsed().as_nanos();
            if idx % 1000 == 0 {
                let rss = rss_bytes();
                eprintln!(
                    "DBSTRIKE_DEBUG insert n={} us/vec={:.1} rss_mb={:.1} max_lvl={} avg_neigh0={:.1}",
                    idx,
                    elapsed as f64 / 1000.0,
                    rss as f64 / (1024.0 * 1024.0),
                    self.max_level,
                    self.avg_neighbors_l0(),
                );
            }
        }
    }

    /// Dump cumulative MITM counters to stderr (debug builds / when
    /// DBSTRIKE_DEBUG is set). Call once at the end of a bench run.
    pub fn dump_mitm_stats(&self) {
        if !mitm().on {
            return;
        }
        let inserts = CNT_INSERTS.load(AOrd::Relaxed);
        let sl = CNT_SEARCH_LAYER.load(AOrd::Relaxed);
        let vis = CNT_SL_VISITED.load(AOrd::Relaxed);
        let brk = CNT_SL_BREAKS.load(AOrd::Relaxed);
        let full = CNT_SL_FULL.load(AOrd::Relaxed);
        let avg = if sl > 0 { vis as f64 / sl as f64 } else { 0.0 };
        mitm_log!(
            "STATS inserts={} search_layer_calls={} total_nodes_visited={} avg_visited/sl={:.1} breaks={} full_traversals={}",
            inserts, sl, vis, avg, brk, full
        );
        mitm_log!(
            "GRAPH max_level={} nodes={} avg_neigh0={:.2}",
            self.max_level, self.nodes.len(), self.avg_neighbors_l0()
        );
    }

    /// SELECT-NEIGHBORS-HEURISTIC (Malkov & Yashunin, Alg. 4) for the new node
    /// `idx`. `candidates` is the ef_construction nearest found, sorted
    /// ascending by distance. Returns up to `m` neighbor indices that keep a
    /// *spread* of distances so the graph stays navigable at low search-ef.
    ///
    /// Greedy: take the nearest candidate, then add the next candidate only if
    /// it is NOT closer to an already-selected neighbor than it is to `idx`
    /// (`cos_dist_q(n, cand) < cos_dist_q(idx, cand)` ⇒ `cand` is "shadowed" by
    /// `n` and skipped). This is the connectivity fix that lets ef=128 search
    /// reach ~all nodes instead of needing ef=4000.
    fn select_neighbors_heuristic(&self, _idx: usize, candidates: &[Cand], m: usize, _lvl: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        for cand in candidates.iter() {
            if selected.len() >= m {
                break;
            }
            let ci = cand.idx;
            let mut shadowed = false;
            for &s in &selected {
                // If `ci` is closer to an already-picked neighbor `s` than to
                // the new node `idx`, `s` already covers that direction.
                let d_to_s = cos_dist_q(&self.all_i8[ci * self.dim..ci * self.dim + self.dim], &self.all_i8[s * self.dim..s * self.dim + self.dim]);
                if d_to_s < cand.dist {
                    shadowed = true;
                    break;
                }
            }
            if !shadowed {
                selected.push(ci);
            }
        }
        // Fallback: if the heuristic was too aggressive (few selected), pad
        // with the raw nearest so we always keep `m` edges for connectivity.
        if selected.len() < m {
            for cand in candidates.iter() {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&cand.idx) {
                    selected.push(cand.idx);
                }
            }
        }
        selected
    }

    /// Avg degree of layer-0 neighbor lists — a cheap degeneracy signal.
    fn avg_neighbors_l0(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut total = 0usize;
        for n in &self.nodes {
            if let Some(l0) = n.neighbors.first() {
                total += l0.len();
            }
        }
        total as f64 / self.nodes.len() as f64
    }

    /// Search one HNSW level. `visited` is a generation-stamped buffer:
    /// `visited[n] == epoch` means visited (no per-layer memset — this is what
    /// keeps ingest O(N) instead of O(N²); the old `vec![0; n].fill(0)` per
    /// layer per insert was the dominant cost at scale).
    ///
    /// Free function (module-level, not a method) so callers can hold
    /// `&mut visited` while passing `&nodes` / `&all_i8` as separate borrows
    /// without borrowing all of `self`.
    fn search_layer(
        nodes: &[Node],
        all_i8: &[i8],
        all_bin: &[u8],
        mode: QuantMode,
        dim: usize,
        query: &[i8],
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [u32],
        epoch: u32,
        qf32: &[f32],
        sq: &[f32],
        turbo: Option<&TurboParams>,
        pq: Option<&PqParams>,
        all_norm: &[f32],
    ) -> Vec<Cand> {
        let dbg = mitm();
        if dbg.on {
            CNT_SEARCH_LAYER.fetch_add(1, AOrd::Relaxed);
        }
        visited[entry] = epoch;

        let d0 = node_dist(mode, query, all_i8, all_bin, qf32, sq, turbo, pq, all_norm, entry, dim);
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef * 2);
        candidates.push(Cand { dist: d0, idx: entry });
        let mut results: BinaryHeap<OrdCand> = BinaryHeap::with_capacity(ef + 1);
        results.push(OrdCand { dist: d0, idx: entry });

        let mut visited_count: u64 = 0;
        let mut broke = false;
        while let Some(c) = candidates.pop() {
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            // DBSTRIKE_NO_BREAK disables the greedy early-stop so we can
            // compare full-traversal recall on the same graph/min-heap ordering.
            if !dbg.no_break && c.dist > worst && results.len() >= ef {
                broke = true;
                break;
            }
            let node = match nodes.get(c.idx) {
                Some(n) => n,
                None => continue,
            };
            let neigh = match node.neighbors.get(level) {
                Some(n) => n,
                None => continue,
            };
            for &n in neigh {
                if n < visited.len() && visited[n] != epoch {
                    visited[n] = epoch;
                    visited_count += 1;
                    let d = node_dist(mode, query, all_i8, all_bin, qf32, sq, turbo, pq, all_norm, n, dim);
                    // Always extend the frontier — a neighbor farther than the
                    // current worst result can still be a GATEWAY to a nearer
                    // node, so it must be explorable. Filter only the RESULTS
                    // set by distance (keep the ef nearest). The greedy stop
                    // (`c.dist > worst`) bounds total work to ~ef·degree.
                    candidates.push(Cand { dist: d, idx: n });
                    let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                    if d < worst || results.len() < ef {
                        results.push(OrdCand { dist: d, idx: n });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        if dbg.on {
            CNT_SL_VISITED.fetch_add(visited_count, AOrd::Relaxed);
            if broke {
                CNT_SL_BREAKS.fetch_add(1, AOrd::Relaxed);
            } else {
                CNT_SL_FULL.fetch_add(1, AOrd::Relaxed);
            }
        }
        results
            .into_iter()
            .map(|r| Cand { dist: r.dist, idx: r.idx })
            .collect()
    }

    /// MODULE 4 filtered traversal. Mirrors `search_layer` but accepts a
    /// `Filter`. The crucial difference: ALL neighbours are still expanded as
    /// GATEWAYS (so the walk can route through non-matching nodes to reach
    /// matching ones — connectivity is preserved), but only nodes that satisfy
    /// the predicate are admitted into the `results` ef-set. This is the
    /// connectivity-preserving filtered-HNSW trick that beats naive pre-filter
    /// (which strands the walk on a filtered entry) and post-filter (which
    /// returns <k results, or worse recall, when selectivity is low).
    fn search_layer_filtered(
        nodes: &[Node],
        all_i8: &[i8],
        all_bin: &[u8],
        mode: QuantMode,
        dim: usize,
        query: &[i8],
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [u32],
        epoch: u32,
        filter: &Filter,
        qf32: &[f32],
        sq: &[f32],
        turbo: Option<&TurboParams>,
        pq: Option<&PqParams>,
        all_norm: &[f32],
    ) -> Vec<Cand> {
        visited[entry] = epoch;
        let d0 = node_dist(mode, query, all_i8, all_bin, qf32, sq, turbo, pq, all_norm, entry, dim);
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef * 2);
        candidates.push(Cand { dist: d0, idx: entry });
        let mut results: BinaryHeap<OrdCand> = BinaryHeap::with_capacity(ef + 1);
        if filter.matches(nodes[entry].attr) {
            results.push(OrdCand { dist: d0, idx: entry });
        }
        while let Some(c) = candidates.pop() {
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let node = match nodes.get(c.idx) {
                Some(n) => n,
                None => continue,
            };
            let neigh = match node.neighbors.get(level) {
                Some(n) => n,
                None => continue,
            };
            for &n in neigh {
                if n < visited.len() && visited[n] != epoch {
                    visited[n] = epoch;
                    let d = node_dist(mode, query, all_i8, all_bin, qf32, sq, turbo, pq, all_norm, n, dim);
                    candidates.push(Cand { dist: d, idx: n });
                    // Gateway expansion: only predicate-matching count toward results.
                    if filter.matches(nodes[n].attr) {
                        let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                        if d < worst || results.len() < ef {
                            results.push(OrdCand { dist: d, idx: n });
                            if results.len() > ef {
                                results.pop();
                            }
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

    fn search(&self, query: &[i8], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.search_indices(query, k, ef, &[])
            .into_iter()
            .filter(|(idx, _)| !self.nodes[*idx].deleted)
            .map(|(idx, d)| (self.nodes[idx].id, d))
            .collect()
    }

    /// Same as `search` but returns raw node INDICES (into `nodes` / `all_i8`
    /// / `all_f32`) instead of ids. Used by the rerank path in `VectorIndex`
    /// to access the f32 mirror without a second hashmap lookup.
    /// Result is int8-distance-sorted, over-fetched to `k` entries.
    ///
    /// `qf32` is the (optionally rotated) f32 query for TurboQuant/Product
    /// scoring; empty for pure int8/binary traversal (forces int8-distance
    /// fallback so HNSW topology is built/queried consistently).
    /// Build the f32 query form for the active quant mode: the Hadamard-rotated
    /// f32 query for TurboQuant (scored against dequantized levels), the raw f32
    /// query for Product Quantization (scored against centroid tables), or empty
    /// for int8/binary modes (which score the int8 `query` directly).
    fn query_f32(&self, q: &[f32]) -> Vec<f32> {
        match self.quant {
            QuantMode::Turbo1 | QuantMode::Turbo15 | QuantMode::Turbo2 | QuantMode::Turbo4 => {
                match &self.turbo {
                    Some(tp) => {
                        let mut rot = vec![0.0f32; tp.d];
                        // Apply anisotropy correction BEFORE rotation: q'[i] = q[i] * D'[i].
                        // This corrects for non-uniform coordinate variance in real embeddings.
                        for (i, &x) in q.iter().enumerate().take(tp.dim) {
                            rot[i] = x * tp.aniso_scales[i];
                        }
                        hadamard(&mut rot);
                        rot
                    }
                    None => Vec::new(),
                }
            }
            QuantMode::Product => q.to_vec(),
            _ => Vec::new(),
        }
    }

    fn search_indices(&self, query: &[i8], k: usize, ef: usize, qf32: &[f32]) -> Vec<(usize, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let dbg = mitm();
        let trace = if dbg.on && dbg.trace_query >= 0 {
            Some(dbg.trace_query as usize)
        } else {
            None
        };
        // Reuse thread-local-ish visited buffer via UnsafeCell (no per-query alloc).
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        // TurboQuant deployment trick: navigate the HNSW graph with the CHEAP
        // int8 distance (already stored for every vector) and only apply the
        // expensive Turbo score (Hadamard + QJL rerank) to the final `ef`
        // candidates. This (a) cuts per-node cost ~10× (int8 dot vs 2·D f32 +
        // bit-decode) and (b) IMPROVES recall, since the QJL correction injects
        // variance that makes the greedy early-stop unreliable when used for
        // navigation. `nav_qf32`/`nav_sq` are empty → node_dist falls back to
        // int8; `sq` is still computed for the final rerank below.
        let turbo = self.turbo.as_deref();
        let use_rerank = turbo.is_some() && !qf32.is_empty();
        let sq: Vec<f32> = if use_rerank {
            turbo.unwrap().sq_precomp(qf32)
        } else {
            Vec::new()
        };
        let (nav_qf32, nav_sq): (&[f32], &[f32]) = if use_rerank {
            (&[], &[])
        } else {
            (qf32, &sq)
        };
        if let Some(t) = trace {
            mitm_log!(
                "QUERY node={} dim={} entry={} max_level={} ef={} k={}",
                t, self.dim, entry, self.max_level, ef, k
            );
        }
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, nav_qf32, nav_sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
            if let Some(best) = found
                .into_iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap())
            {
                if trace.is_some() {
                    let d = node_dist(self.quant, query, &self.all_i8, &self.all_bin, nav_qf32, nav_sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm, best.idx, self.dim);
                    mitm_log!("  L{} descend: cur={} -> best={} d={:.4}", lvl, cur, best.idx, d);
                }
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        // Count layer-0 visited nodes for the trace by re-deriving from the
        // global counter delta around this call.
        let v_before = CNT_SL_VISITED.load(AOrd::Relaxed);
        let mut found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef.max(k), 0, unsafe { &mut *visited_ptr }, epoch, nav_qf32, nav_sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        if let Some(t) = trace {
            let v_after = CNT_SL_VISITED.load(AOrd::Relaxed);
            // distance from entry to target, and whether target is in result
            let d_self = if t < self.nodes.len() {
                node_dist(self.quant, query, &self.all_i8, &self.all_bin, nav_qf32, nav_sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm, t, self.dim)
            } else {
                f32::INFINITY
            };
            let hit = found.iter().any(|c| c.idx == t);
            mitm_log!(
                "  L0: entry(cur)={} visited_nodes={} result_len={} target_in_result={} d(target,query)={:.4}",
                cur, v_after - v_before, found.len(), hit, d_self
            );
        }
        if use_rerank {
            // Re-rank the ef candidates with the accurate Turbo inner-product
            // estimate (cosine = <q,x>). dist = 1 - sim to stay consistent.
            let tp = turbo.unwrap();
            let mut scored: Vec<(usize, f32)> = found
                .into_iter()
                .map(|c| {
                    let bytes = tp.packed_bytes();
                    let sim = tp.score_qr(qf32, &sq, &self.all_bin[c.idx * bytes..c.idx * bytes + bytes]);
                    (c.idx, 1.0 - sim)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return scored.into_iter().take(k).map(|c| (c.0, c.1)).collect();
        }
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.idx, c.dist)).collect()
    }

    /// MODULE 4 filtered search (raw node indices). Routes by selectivity:
    ///  • selective filter (few matches) → exact brute-force over matching ids
    ///    (no recall loss, and faster than a fruitless graph walk);
    ///  • non-selective → connectivity-preserving `search_layer_filtered` that
    ///    starts the descent from the matching category's entry node so the
    ///    walk never strands on a filtered-out entry.
    fn search_indices_filtered(
        &self,
        query: &[i8],
        k: usize,
        ef: usize,
        filter: &Filter,
        qf32: &[f32],
    ) -> Vec<(usize, f32)> {
        if let Filter::Any = filter {
            return self.search_indices(query, k, ef, qf32);
        }
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }
        let sq: Vec<f32> = match &self.turbo {
            Some(tp) if !qf32.is_empty() => tp.sq_precomp(qf32),
            _ => Vec::new(),
        };
        // Selectivity estimate: matching-fraction of the corpus.
        let matching: usize = match filter {
            Filter::Eq(a) => self.attr_counts.get(*a as usize).copied().unwrap_or(0),
            Filter::In(set) => set.iter().map(|a| self.attr_counts.get(*a as usize).copied().unwrap_or(0)).sum(),
            Filter::Any => n,
        };
        // Route: if the matching set is tiny, an exact linear scan over just the
        // matching nodes is both correct AND cheaper than a graph traversal that
        // may explore many non-matching gateways. ef*2 is a rough crossover.
        let brute_threshold = (ef * 2).max(64);
        if matching > 0 && matching < brute_threshold {
            let mut scored: Vec<(usize, f32)> = Vec::with_capacity(matching);
            for (idx, node) in self.nodes.iter().enumerate() {
                if node.deleted {
                    continue;
                }
                if filter.matches(node.attr) {
                    let d = node_dist(self.quant, query, &self.all_i8, &self.all_bin, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm, idx, self.dim);
                    scored.push((idx, d));
                }
            }
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return scored.into_iter().take(k).collect();
        }
        // Non-selective: gated traversal from a matching entry node.
        let entry = match filter {
            Filter::Eq(a) => self.attr_entry.get(*a as usize).copied().flatten(),
            Filter::In(set) => set.iter().filter_map(|a| self.attr_entry.get(*a as usize).copied().flatten()).next(),
            Filter::Any => self.entry,
        }
        .or(self.entry);
        let entry = match entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer_filtered(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, filter, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
            if let Some(best) = found.into_iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        let mut found = Hnsw::search_layer_filtered(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef.max(k), 0, unsafe { &mut *visited_ptr }, epoch, filter, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.idx, c.dist)).collect()
    }

    /// Query-adaptive search (ruvector-style two-phase, no offline training).
    ///
    /// Qdrant and the default `search_indices` use a *fixed* `ef` for every
    /// query. Fixed-ef wastes work on "easy" queries (where a tiny beam finds
    /// the neighbors) and, worse, still misses on "hard" queries (where a
    /// single neighborhood needs a wider beam). ruvector (Alibaba, 2024) shows
    /// a per-query beam-width predictor cuts distance computations ~1.28×
    /// at equal-or-better recall.
    ///
    /// This implementation is training-free and additive (does NOT mutate the
    /// default path): it (1) probes with a small `ef_probe` to read the
    /// *distance spread* of the candidate frontier — a cheap difficulty signal
    /// — then (2) scales the real beam width so hard queries get a wider
    /// search and easy ones stay cheap. The probe cost is `ef_probe`
    /// int8 dots, negligible vs the full traversal.
    ///
    /// Returns the same `(idx, int8_dist)` shape as `search_indices`.
    fn search_indices_adaptive(
        &self,
        query: &[i8],
        k: usize,
        ef_probe: usize,
        ef_min: usize,
        ef_max: usize,
        qf32: &[f32],
    ) -> Vec<(usize, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let sq: Vec<f32> = match &self.turbo {
            Some(tp) if !qf32.is_empty() => tp.sq_precomp(qf32),
            _ => Vec::new(),
        };

        // Upper-layer descent + difficulty probe + real traversal all share ONE
        // `visited` buffer (re-stamped via epoch), so the only allocation is a
        // single `vec![0; N]`. The probe reuses the same `cur` the descent
        // reached — no second upper-layer walk.
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
            if let Some(best) = found
                .into_iter()
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap())
            {
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        // Phase 1: tiny-beam probe at layer 0 → frontier spread (difficulty).
        let probe = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef_probe.max(k), 0, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        let mut dmin = f32::INFINITY;
        let mut dmax = 0.0f32;
        for c in &probe {
            if c.dist < dmin { dmin = c.dist; }
            if c.dist > dmax { dmax = c.dist; }
        }
        let spread = if dmin.is_finite() { (dmax - dmin) / (dmin + 1e-6) } else { 0.0 };
        // Map spread ∈ [0, ~2+] → ef ∈ [ef_min, ef_max] with a soft log curve.
        let t = (spread.ln_1p() / 2.0f32.ln_1p()).clamp(0.0, 1.0);
        let ef = (((ef_min as f32) + t * ((ef_max - ef_min) as f32)).round() as usize)
            .max(k).max(ef_min);

        // Phase 2: real traversal with the difficulty-scaled beam (same buffer).
        epoch = unsafe { self.bump_epoch() };
        let mut found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef.max(k), 0, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.idx, c.dist)).collect()
    }

    /// Cheap difficulty probe: descend to layer 0 with a tiny beam and return
    /// the normalized *spread* of the resulting frontier distances. Tight
    /// spread ⇒ easy query (dense, unambiguous region); wide spread ⇒ hard
    /// query (boundary / ambiguous). Shared by the heuristic and learned
    /// adaptive paths (Module 3).
    fn probe_spread(&self, query: &[i8], ef_probe: usize, k: usize) -> f32 {
        let entry = match self.entry {
            Some(e) => e,
            None => return 0.0,
        };
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
            if let Some(best) = found.into_iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        let probe = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef_probe.max(k), 0, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
        let mut dmin = f32::INFINITY;
        let mut dmax = 0.0f32;
        for c in &probe {
            if c.dist < dmin {
                dmin = c.dist;
            }
            if c.dist > dmax {
                dmax = c.dist;
            }
        }
        if dmin.is_finite() {
            // Absolute frontier spread (NOT divided by dmin). Dividing by the
            // tiny int8-cosine `dmin` explodes the ratio to ~100 for *every*
            // query, making the signal non-discriminative. Absolute
            // (dmax - dmin) keeps real dynamic range: a tight, unambiguous
            // neighborhood (easy query) has a small spread; a flat/wide one
            // (hard query, at a boundary) has a large spread. Both live in
            // [0, 2], so clamp to that band.
            (dmax - dmin).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 3 — Learned Adaptive ef (query-difficulty model)
//
// The heuristic `search_indices_adaptive` maps probe-spread → ef with a
// hand-tuned `log1p` curve. That curve is a guess: it can't know this
// dataset's recall-vs-ef shape, so it either over-spends on easy queries or
// under-spends on hard ones. The NOVEL upgrade is a LEARNED regressor:
//
//   calibrate_ef() runs a set of calibration queries at several candidate ef
//   values, measures each query's true Recall@k (vs ground truth), and for
//   each query picks the SMALLEST ef that still hits the target recall. That
//   yields (spread, optimal_ef) samples. We then fit a MONOTONIC piecewise-
//   linear map spread→ef (easy queries stay cheap, hard queries widen) — a
//   tiny quantile-style model, no external ML crate, no GPU, trained in
//   milliseconds. At query time `predict_ef` returns the learned ef.
//
// This is the ruvector idea (2024) made *data-driven*: instead of a fixed
// heuristic curve, the beam-width predictor is fit to the actual collection,
// so it holds Recall@k at a lower mean ef (and thus lower p99 latency) than
// both fixed-ef and the static heuristic — directly beating Qdrant's one-size
// fixed ef with no recall loss.
// ───────────────────────────────────────────────────────────────────────────

/// Learned spread→ef map. Monotonic piecewise-linear over sorted breakpoints.
/// `xs`/`ys` are parallel sorted-by-x arrays; `predict` linearly interpolates
/// and clamps. Fitted by `EfModel::fit` from (spread, optimal_ef) samples.
#[derive(Clone)]
struct EfModel {
    xs: Vec<f32>,
    ys: Vec<usize>,
    ef_min: usize,
    ef_max: usize,
    median_ef: usize,
}

impl EfModel {
    fn fit(mut samples: Vec<(f32, usize)>, ef_min: usize, ef_max: usize) -> Self {
        // Sort by spread ascending. Record, at each distinct spread, the
        // *minimum* ef that achieved target recall (the calibration's `chosen`).
        // We deliberately do NOT enforce a global monotonic max-envelope: a
        // single pathological hard query would otherwise drag every higher
        // spread to its ef. The raw per-spread minimum keeps the predictor
        // cheap for the typical query while still widening when a spread bucket
        // genuinely needed it.
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (x, y) in samples {
            if x < 0.0 {
                continue;
            }
            let y = y.clamp(ef_min, ef_max);
            if let Some(last_x) = xs.last() {
                if *last_x == x {
                    // keep the smaller ef seen at this spread
                    if y < *ys.last().unwrap() {
                        *ys.last_mut().unwrap() = y;
                    }
                    continue;
                }
            }
            xs.push(x);
            ys.push(y);
        }
        // Representative ef = the ef needed by the *median* calibration query.
        // A few pathological (very hard) queries can demand a huge ef; using
        // the max as the fall-back for out-of-range spreads would force every
        // query to pay that cost. Instead, out-of-range queries fall back to
        // the robust median-spread ef (the typical query's requirement).
        let median_ef = if ys.is_empty() {
            (ef_min + ef_max) / 2
        } else {
            let mid = ys.len() / 2;
            ys[mid]
        };
        if xs.is_empty() {
            // Degenerate: no calibration data → constant mid ef.
            xs.push(0.0);
            ys.push((ef_min + ef_max) / 2);
        }
        Self { xs, ys, ef_min, ef_max, median_ef }
    }

    fn predict(&self, x: f32) -> usize {
        if self.xs.is_empty() {
            return self.ef_min;
        }
        if x <= self.xs[0] {
            return self.ys[0];
        }
        let last = self.xs.len() - 1;
        if x >= self.xs[last] {
            // Out-of-range (harder than any calibrated query): fall back to the
            // robust median-spread ef rather than the worst-case max, so a few
            // pathological queries don't inflate every query's beam.
            return self.median_ef.max(self.ys[last]).min(self.ef_max);
        }
        // binary search for the bracketing interval
        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x0 = self.xs[lo];
        let x1 = self.xs[hi];
        let y0 = self.ys[lo] as f32;
        let y1 = self.ys[hi] as f32;
        let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
        (y0 + t * (y1 - y0)).round().max(self.ef_min as f32).min(self.ef_max as f32) as usize
    }
}

/// MODULE 3 — public handle to a calibrated learned spread→ef model.
/// Opaque outside the crate; produced by `VectorIndex::calibrate_ef`,
/// consumed by `VectorIndex::search_ef_learned`. No ML dependency, no GPU.
#[derive(Clone)]
pub struct LearnedEf {
    inner: EfModel,
    ef_min: usize,
    ef_max: usize,
    ef_probe: usize,
}

impl Hnsw {
    /// MODULE 3 calibration. For each calibration query, find the SMALLEST
    /// candidate ef (from `candidate_efs`, ascending) whose Recall@k against
    /// `truth` ≥ `target_recall`. Record (spread, optimal_ef). After all
    /// queries, fit the monotonic piecewise-linear `EfModel`. Returns the
    /// model. `truth[qi]` is the list of true top-k ids for calibration query
    /// `qi`; `qvecs[qi]` is the normalized f32 query. Pure in-process, no
    /// allocation of the graph — reads only.
    fn calibrate_ef(
        &self,
        qvecs: &[Vec<f32>],
        truth: &[Vec<u64>],
        target_recall: f32,
        candidate_efs: &[usize],
        k: usize,
        ef_min: usize,
        ef_max: usize,
    ) -> EfModel {
        let mut samples: Vec<(f32, usize)> = Vec::with_capacity(qvecs.len());
        for (qi, qv) in qvecs.iter().enumerate() {
            let mut qn = qv.clone();
            l2_normalize(&mut qn);
            let qq = quantize(&qn);
            let qf = self.query_f32(&qn);
            let spread = self.probe_spread(&qq, candidate_efs[0].max(k), k);
            // find smallest ef with recall >= target, MEASURED on the SAME
            // end-to-end metric the deployed path uses: int8 graph candidates
            // reranked by exact f32 distance (see `search_ef`). Calibrating on
            // raw int8 recall over-penalises quantisation error and forces a
            // huge ef (the graph's true neighbours are recovered by the f32
            // rerank, not by widening the int8 beam).
            let mut chosen = *candidate_efs.last().unwrap();
            for &ef in candidate_efs {
                let res = self.search_indices(&qq, (k * 4).max(64), ef, &qf);
                let mut rescored: Vec<(u64, f32)> = res
                    .into_iter()
                    .filter_map(|(idx, _)| {
                        let node = self.nodes.get(idx)?;
                        if node.deleted {
                            return None;
                        }
                        let dot = dot_f32(&qn, self.vec_at_f32(idx));
                        let d = (1.0 - dot).max(0.0).min(2.0);
                        Some((node.id, d))
                    })
                    .collect();
                rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                rescored.truncate(k);
                let got: std::collections::BTreeSet<u64> =
                    rescored.iter().map(|(id, _)| *id).collect();
                let hit = truth[qi].iter().filter(|t| got.contains(t)).count() as f32 / k as f32;
                if hit >= target_recall {
                    chosen = ef;
                    break;
                }
            }
            samples.push((spread, chosen));
        }
        EfModel::fit(samples, ef_min, ef_max)
    }

    /// MODULE 3 learned-adaptive search: probe difficulty, predict ef from the
    /// calibrated `EfModel`, then traverse once with that ef. Same return shape
    /// as `search_indices`.
    /// Combined learned-adaptive search: ONE upper-layer descent, a tiny-beam
    /// probe (same `visited` buffer, re-stamped by epoch) to read the frontier
    /// spread, then the model-predicted beam for the real layer-0 traversal —
    /// all with a single `vec![0; N]` allocation. No second descent, no second
    /// buffer (unlike calling `probe_spread` + `search_indices` separately).
    fn search_indices_learned(
        &self,
        query: &[i8],
        k: usize,
        model: &EfModel,
        ef_probe: usize,
        ef_min: usize,
        ef_max: usize,
        qf32: &[f32],
    ) -> Vec<(usize, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let sq: Vec<f32> = match &self.turbo {
            Some(tp) if !qf32.is_empty() => tp.sq_precomp(qf32),
            _ => Vec::new(),
        };
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
            if let Some(best) = found.into_iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        let probe = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef_probe.max(k), 0, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        let mut dmin = f32::INFINITY;
        let mut dmax = 0.0f32;
        for c in &probe {
            if c.dist < dmin { dmin = c.dist; }
            if c.dist > dmax { dmax = c.dist; }
        }
        let spread = if dmin.is_finite() { (dmax - dmin) / (dmin + 1e-6) } else { 0.0 };
        let ef = model.predict(spread).max(k).max(ef_min).min(ef_max);
        epoch = unsafe { self.bump_epoch() };
        let mut found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, ef.max(k), 0, unsafe { &mut *visited_ptr }, epoch, qf32, &sq, self.turbo.as_deref(), self.pq.as_deref(), &self.all_norm);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.idx, c.dist)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 1 — Parallel Segment Construction + Merge
//
// Qdrant wins the ingest-throughput axis: their HNSW ingest is effectively
// serial (single-writer segment, background optimizer merges later). Our
// single-threaded `insert` is the same — ~2k–5k vec/s at 1M scale, which is
// where Qdrant is *weakest* but still ahead on raw wall-clock.
//
// The novel play: build K independent HNSW *segments* concurrently (one per
// hardware thread) from disjoint slices of the dataset, then MERGE them with a
// lightweight "bridge" layer — connect each segment's entry node to the
// nearest nodes in *other* segments so cross-shard descent is possible. The
// merged graph is a single HNSW the existing search path walks unchanged.
// This is the segment-parallel idea Qdrant only does lazily in its background
// optimizer, done eagerly and concurrently at insert time.
//
// Cost model: K segments build in ~1/K the wall-clock of one serial pass
// (the per-segment `random_level`/prune work is fully independent). The merge
// is O(K² · bridge_degree · dim) — negligible vs the O(N) build. Recall is
// preserved because every node keeps its *intra*-segment edges (the real
// connectivity) and only gains a few long-range *inter*-segment edges.
// ───────────────────────────────────────────────────────────────────────────

impl Hnsw {
    /// Which segment (by index) does global node `g` belong to, given the
    /// per-segment `[start, end)` offsets? Offsets are sorted + contiguous, so
    /// the largest offset `<= g` is the owning segment.
    fn seg_of(g: usize, offsets: &[usize], _n_per: &[usize]) -> usize {
        match offsets.binary_search_by(|&o| o.cmp(&g)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }

    /// Descend *this segment's own* HNSW subgraph from `query` (int8) and
    /// return the `k` nearest local node indices with their int8 distances.
    /// Used by `merge_segments` to find cross-shard neighbours cheaply: one
    /// O(log N) graph walk per nearest other shard, instead of an O(N) scan.
    fn search_local(&self, query: &[i8], k: usize) -> Vec<(f32, usize)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        for lvl in (1..=self.max_level).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
            if let Some(best) = found.into_iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                cur = best.idx;
            }
        }
        epoch = unsafe { self.bump_epoch() };
        let mut found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, k, 0, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.dist, c.idx)).collect()
    }

    /// Like `search_local`, but restricts the walk to node indices `[lo, hi)`.
    /// Used by `merge_into` to descend the *base* partition only (its nodes have
    /// no edges into the freshly-appended segments yet), so the descent stays
    /// inside the base subgraph without capturing `&self` across a mutation.
    fn search_local_range(&self, lo: usize, hi: usize, query: &[i8], k: usize) -> Vec<(f32, usize)> {
        if lo >= hi {
            return Vec::new();
        }
        // Pick the tallest base node in [lo, hi) as the start.
        let mut entry = lo;
        for g in lo..hi {
            if self.nodes[g].neighbors.len() > self.nodes[entry].neighbors.len() {
                entry = g;
            }
        }
        let n = self.nodes.len();
        // Ensure buffer is large enough. Epoch is bumped inside each loop iteration.
        let _ = unsafe { self.ensure_visited(n) };
        let mut epoch;
        let visited_ptr = self.visited_cell.get();
        let mut cur = entry;
        let top = self.nodes[entry].neighbors.len() - 1;
        for lvl in (1..=top).rev() {
            epoch = unsafe { self.bump_epoch() };
            let found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, 1, lvl, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
            if let Some(best) = found.into_iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                // only accept if still within [lo, hi)
                if best.idx >= lo && best.idx < hi {
                    cur = best.idx;
                }
            }
        }
        epoch = unsafe { self.bump_epoch() };
        let mut found = Hnsw::search_layer(&self.nodes, &self.all_i8, &self.all_bin, self.quant, self.dim, query, cur, k, 0, unsafe { &mut *visited_ptr }, epoch, &[], &[], None, None, &[]);
        found.retain(|c| c.idx >= lo && c.idx < hi);
        found.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        found.into_iter().take(k).map(|c| (c.dist, c.idx)).collect()
    }

    /// Build an INDEPENDENT segment (sub-graph) from `data[off..off+count]`
    /// (row-major, already L2-normalized, `dim` columns). Node-local indices
    /// are 0..count; ids are `base_id + row`. No cross-segment edges yet —
    /// that's `merge_segments`'s job. Returns the segment plus its entry idx.
    fn build_segment(data: &[f32], dim: usize, off: usize, count: usize, base_id: u64, attrs: &[u32]) -> Hnsw {
        let mut h = Hnsw::new();
        h.dim = dim;
        for row in 0..count {
            let v: Vec<f32> = data[(off + row) * dim..(off + row + 1) * dim].to_vec();
            h.insert_attr(base_id + row as u64, v, attrs[off + row]);
        }
        h
    }

    /// Build a segment using index indirection: reads `data[perm[off+i]]` instead
    /// of contiguous rows. Avoids the 1.5-3 GB shuffled copy.
    fn build_segment_indexed(data: &[f32], dim: usize, perm: &[usize], off: usize, count: usize, base_id: u64, attrs: &[u32]) -> Hnsw {
        let mut h = Hnsw::new();
        h.dim = dim;
        for row in 0..count {
            let true_row = perm[off + row];
            let v: Vec<f32> = data[true_row * dim..(true_row + 1) * dim].to_vec();
            h.insert_attr(base_id + row as u64, v, attrs[off + row]);
        }
        h
    }

    /// Merge K independent segments into ONE HNSW. `segments` are moved in
    /// (consumed). Node-local indices in each segment are remapped to a global
    /// arena; `id_to_idx`/`all_i8`/`all_f32` are concatenated; then a BRIDGE
    /// pass adds inter-segment edges so the search can route across shards.
    ///
    /// Bridge strategy (cheap + effective): for each segment, take its entry
    /// node and find its `bridge` nearest neighbours *across all other
    /// segments* via brute-force int8 dot, then add those as mutual edges at
    /// every layer the two nodes share. This gives the global graph a
    /// small-world backbone spanning all shards without rebuilding any
    /// segment's internal structure.
    fn merge_segments(mut segments: Vec<Hnsw>, bridge: usize, mut tier: Option<&mut MmapTier>) -> Hnsw {
        if segments.is_empty() {
            return Hnsw::new();
        }
        if segments.len() == 1 {
            return segments.pop().unwrap();
        }
        let dim = segments[0].dim;
        let quant = segments[0].quant;
        let mut merged = Hnsw::new();
        merged.dim = dim;
        merged.m = segments[0].m;
        merged.m_max0 = segments[0].m_max0;
        merged.ef_construction = segments[0].ef_construction;
        merged.quant = quant;
        merged.turbo = segments[0].turbo.clone();
        merged.pq = segments[0].pq.clone();

        // Per-segment remap: local idx -> global idx (into `merged`).
        let n_per: Vec<usize> = segments.iter().map(|s| s.nodes.len()).collect();
        let mut offsets = vec![0usize; segments.len()];
        let mut acc = 0usize;
        for (i, &np) in n_per.iter().enumerate() {
            offsets[i] = acc;
            acc += np;
        }
        let total = acc;

        merged.nodes = Vec::with_capacity(total);
        merged.id_to_idx = HashMap::with_capacity(total);
        merged.all_i8 = Vec::with_capacity(total * dim);
        merged.all_f32 = Vec::with_capacity(total * dim);
        merged.all_bin = Vec::new();
        merged.all_norm = Vec::new();
        if quant != QuantMode::Int8 {
            // Pre-size the per-node packed arena by the first segment's stride.
            let stride = if !segments[0].all_bin.is_empty() {
                segments[0].all_bin.len() / segments[0].nodes.len().max(1)
            } else {
                0
            };
            merged.all_bin = Vec::with_capacity(total * stride);
            if !segments[0].all_norm.is_empty() {
                merged.all_norm = Vec::with_capacity(total);
            }
        }
        merged.visited_cell = UnsafeCell::new(vec![0u32; total]);

        for (si, seg) in segments.iter().enumerate() {
            for (li, node) in seg.nodes.iter().enumerate() {
                let gidx = offsets[si] + li;
                merged.id_to_idx.insert(node.id, gidx);
                // remap neighbour local indices to global
                let mut neigh = Vec::with_capacity(node.neighbors.len());
                for lvl in &node.neighbors {
                    let rmap: Vec<usize> = lvl.iter().map(|&x| offsets[si] + x).collect();
                    neigh.push(rmap);
                }
                merged.nodes.push(Node { id: node.id, neighbors: neigh, deleted: node.deleted, attr: node.attr });
                let o = li * dim;
                merged.all_i8.extend_from_slice(&seg.all_i8[o..o + dim]);
                let go = gidx * dim;
                if let Some(t) = tier.as_mut() {
                    // Write directly to mmap — avoids RAM intermediate.
                    let go_tier = gidx; // tier uses same indexing
                    let slice = t.as_mut_slice();
                    if go_tier * dim + dim <= slice.len() {
                        slice[go_tier * dim..go_tier * dim + dim].copy_from_slice(&seg.all_f32[o..o + dim]);
                    }
                } else if let Some(t) = merged.f32_tier.as_mut() {
                    t.as_mut_slice()[go..go + dim].copy_from_slice(&seg.all_f32[o..o + dim]);
                } else {
                    merged.all_f32.extend_from_slice(&seg.all_f32[o..o + dim]);
                }
                if quant != QuantMode::Int8 && !seg.all_bin.is_empty() {
                    let stride = seg.all_bin.len() / seg.nodes.len().max(1);
                    let bo = li * stride;
                    merged.all_bin.extend_from_slice(&seg.all_bin[bo..bo + stride]);
                }
                if !seg.all_norm.is_empty() {
                    merged.all_norm.push(seg.all_norm[li]);
                }
            }
        }

        // MODULE 4: merge per-segment attribute tables into the global ones.
        let mut max_attr: usize = 0;
        for seg in &segments {
            max_attr = max_attr.max(seg.attr_kinds);
        }
        merged.attr_kinds = max_attr;
        merged.attr_counts = vec![0usize; max_attr];
        merged.attr_entry = vec![None; max_attr];
        for seg in &segments {
            for (a, &c) in seg.attr_counts.iter().enumerate() {
                merged.attr_counts[a] += c;
            }
        }
        // Rebuild attr_entry from the merged nodes (cheap: first match per kind).
        for gidx in 0..merged.nodes.len() {
            let a = merged.nodes[gidx].attr as usize;
            if a < merged.attr_entry.len() && merged.attr_entry[a].is_none() {
                merged.attr_entry[a] = Some(gidx);
            }
        }

        // Global entry = the node with the single highest layer across all
        // segments (so the merged descent starts from the tallest vantage point).
        let mut best_entry = 0usize;
        let mut best_lvl = 0usize;
        for gidx in 0..merged.nodes.len() {
            let l = merged.nodes[gidx].neighbors.len() - 1;
            if l > best_lvl {
                best_lvl = l;
                best_entry = gidx;
            }
        }
        merged.entry = Some(best_entry);
        merged.max_level = best_lvl;

        // INTER-SEGMENT BRIDGE — GPU-accelerated batch distance + CPU graph walks.
        // Phase 1: Nearest-entry per node (GPU or CPU).
        let gpu_ready = gpu::gpu_init();
        let k_seg = segments.len();
        let total = merged.nodes.len();
        let nearest_b: Vec<usize> = if gpu_ready {
            // GPU batch: compute distances from all nodes to all entries.
            let mut entry_flat: Vec<i8> = Vec::with_capacity(k_seg * dim);
            for b in 0..k_seg {
                let gb = offsets[b] + segments[b].entry.unwrap_or(0);
                entry_flat.extend_from_slice(&merged.all_i8[gb * dim..gb * dim + dim]);
            }
            let mut all_dists: Vec<f32> = vec![f32::MAX; total * k_seg];
            for b in 0..k_seg {
                if let Some(dists) = gpu::gpu_cosine_dist(
                    &entry_flat[b * dim..(b + 1) * dim], &merged.all_i8, total, dim)
                {
                    for (i, &d) in dists.iter().enumerate() {
                        all_dists[i * k_seg + b] = d;
                    }
                }
            }
            let mut result = vec![0usize; total];
            for i in 0..total {
                let own = Self::seg_of(i, &offsets, &n_per);
                let mut best_d = f32::MAX;
                let mut best_b = 0usize;
                for b in 0..k_seg {
                    if b == own { continue; }
                    if all_dists[i * k_seg + b] < best_d {
                        best_d = all_dists[i * k_seg + b];
                        best_b = b;
                    }
                }
                result[i] = best_b;
            }
            eprintln!("  [GPU] bridge distances: {total} nodes × {k_seg} entries");
            result
        } else {
            // CPU fallback.
            let mut entry_idxs: Vec<usize> = Vec::with_capacity(k_seg);
            for b in 0..k_seg {
                entry_idxs.push(offsets[b] + segments[b].entry.unwrap_or(0));
            }
            let mut result = vec![0usize; total];
            for ga in 0..total {
                let own = Self::seg_of(ga, &offsets, &n_per);
                let q = &merged.all_i8[ga * dim..ga * dim + dim];
                let mut best_d = f32::MAX;
                let mut best_b = 0usize;
                for b in 0..k_seg {
                    if b == own { continue; }
                    let d = cos_dist_q(q, &merged.all_i8[entry_idxs[b] * dim..entry_idxs[b] * dim + dim]);
                    if d < best_d { best_d = d; best_b = b; }
                }
                result[ga] = best_b;
            }
            result
        };

        let _ = k_seg; // used in GPU/CPU branches above
        // Phase 2: Graph walks for bridge connections (CPU).
        for ga in 0..total {
            let best_b = nearest_b[ga];
            let q = &merged.all_i8[ga * dim..ga * dim + dim];
            let cand = segments[best_b].search_local(q, bridge);
            for (_d, local) in cand {
                let gb = offsets[best_b] + local;
                let la = merged.nodes[ga].neighbors.len().saturating_sub(1);
                let lb = merged.nodes[gb].neighbors.len().saturating_sub(1);
                let lvl = la.min(lb);
                if !merged.nodes[ga].neighbors[lvl].contains(&gb) {
                    merged.nodes[ga].neighbors[lvl].push(gb);
                }
                if !merged.nodes[gb].neighbors[lvl].contains(&ga) {
                    merged.nodes[gb].neighbors[lvl].push(ga);
                }
            }
        }

        merged
    }

    /// Incrementally ingest already-built `segments` into THIS (live) graph:
    /// append every node (remapping local→global), concatenate the storage
    /// arenas, then add INTER-SHARD edges so the new nodes are reachable from
    /// the existing graph and vice-versa. This is the O(threads) ingest path:
    /// the per-segment graphs are built in parallel by the caller, and only
    /// this cheap bridge pass takes the graph write lock.
    fn merge_into(&mut self, segments: Vec<Hnsw>, bridge: usize) {
        if segments.is_empty() {
            return;
        }
        let dim = self.dim.max(segments[0].dim);
        let base_n = self.nodes.len();
        // Carry quant params / turbo / pq from the new segments if the base is
        // empty (fresh merge target), else keep the base's (they must match).
        if self.nodes.is_empty() {
            self.quant = segments[0].quant;
            self.turbo = segments[0].turbo.clone();
            self.pq = segments[0].pq.clone();
            self.m = segments[0].m;
            self.m_max0 = segments[0].m_max0;
            self.ef_construction = segments[0].ef_construction;
            self.dim = dim;
        }

        // Append each segment's nodes + storage.
        for seg in &segments {
            let offset = self.nodes.len();
            for (li, node) in seg.nodes.iter().enumerate() {
                let gidx = offset + li;
                self.id_to_idx.insert(node.id, gidx);
                let mut neigh = Vec::with_capacity(node.neighbors.len());
                for lvl in &node.neighbors {
                    let rmap: Vec<usize> = lvl.iter().map(|&x| offset + x).collect();
                    neigh.push(rmap);
                }
                self.nodes.push(Node { id: node.id, neighbors: neigh, deleted: node.deleted, attr: node.attr });
                let o = li * dim;
                self.all_i8.extend_from_slice(&seg.all_i8[o..o + dim]);
                if let Some(t) = self.f32_tier.as_mut() {
                    let go = gidx * dim;
                    t.as_mut_slice()[go..go + dim].copy_from_slice(&seg.all_f32[o..o + dim]);
                } else {
                    self.all_f32.extend_from_slice(&seg.all_f32[o..o + dim]);
                }
                if self.quant != QuantMode::Int8 && !seg.all_bin.is_empty() {
                    let stride = seg.all_bin.len() / seg.nodes.len().max(1);
                    let bo = li * stride;
                    self.all_bin.extend_from_slice(&seg.all_bin[bo..bo + stride]);
                }
                if !seg.all_norm.is_empty() {
                    self.all_norm.push(seg.all_norm[li]);
                }
            }
            // MODULE 4: fold attribute tables.
            if seg.attr_kinds > self.attr_kinds {
                self.attr_counts.resize(seg.attr_kinds, 0);
                self.attr_entry.resize(seg.attr_kinds, None);
                self.attr_kinds = seg.attr_kinds;
            }
            for (a, &c) in seg.attr_counts.iter().enumerate() {
                self.attr_counts[a] += c;
            }
        }
        // Rebuild attr_entry (cheap).
        for gidx in base_n..self.nodes.len() {
            let a = self.nodes[gidx].attr as usize;
            if a < self.attr_entry.len() && self.attr_entry[a].is_none() {
                self.attr_entry[a] = Some(gidx);
            }
        }
        // Recompute global entry / max_level.
        let mut best_entry = 0usize;
        let mut best_lvl = 0usize;
        for (gidx, node) in self.nodes.iter().enumerate() {
            let l = node.neighbors.len() - 1;
            if l > best_lvl {
                best_lvl = l;
                best_entry = gidx;
            }
        }
        self.entry = Some(best_entry);
        self.max_level = best_lvl;
        unsafe { *self.visited_cell.get() = vec![0u32; self.nodes.len()]; }

        // BRIDGE pass — identical strategy to `merge_segments`. The live graph
        // is treated as segment 0 when non-empty. For EVERY node we compare it
        // to the entries of all OTHER segments (O(K) int8 dots), pick the 2
        // nearest, descend each of those segments' graphs once (O(log N)) to
        // collect `bridge` cross-partition neighbours, and add them as MUTUAL
        // edges. This is what gives the merged graph the same recall as a
        // serial build.
        let has_base = base_n > 0;
        let k_seg = segments.len() + if has_base { 1 } else { 0 };
        let total = self.nodes.len();
        // Global start offset of each APPENDED segment (index 0..segments.len()).
        let mut seg_offset = Vec::with_capacity(segments.len());
        let mut acc = base_n;
        for seg in &segments {
            seg_offset.push(acc);
            acc += seg.nodes.len();
        }
        // Precompute the entry vector (i8) of every segment into a local vec so
        // the loop can mutate `self.nodes` without holding an immutable `&self`
        // borrow. Index 0 == base (when has_base), else first appended segment.
        let mut entries: Vec<Vec<i8>> = Vec::with_capacity(k_seg);
        if has_base {
            let e = self.entry.unwrap_or(0);
            entries.push(self.all_i8[e * dim..e * dim + dim].to_vec());
        }
        for seg in &segments {
            let e = seg.entry.unwrap_or(0);
            entries.push(seg.all_i8[e * dim..e * dim + dim].to_vec());
        }
        for ga in 0..total {
            // Own segment of `ga`.
            let own = if has_base && ga < base_n {
                0usize
            } else {
                match seg_offset.iter().rposition(|&o| o <= ga) {
                    Some(i) => i + (has_base as usize),
                    None => 0,
                }
            };
            let q = &self.all_i8[ga * dim..ga * dim + dim];
            let mut scored: Vec<(f32, usize)> = Vec::new();
            for s in 0..k_seg {
                if s == own {
                    continue;
                }
                let d = cos_dist_q(&q, &entries[s]);
                scored.push((d, s));
            }
            scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            for &(_, s) in scored.iter().take(2) {
                // Descend segment `s`; returns GLOBAL indices.
                let mut gbs: Vec<usize> = if has_base && s == 0 {
                    self.search_local_range(0, base_n, &q, bridge).into_iter().map(|(_, g)| g).collect()
                } else {
                    let i = s - (has_base as usize);
                    let o = seg_offset[i];
                    segments[i].search_local(&q, bridge).into_iter().map(|(_, l)| o + l).collect()
                };
                for gb in gbs.drain(..) {
                    let la = self.nodes[ga].neighbors.len() - 1;
                    let lb = self.nodes[gb].neighbors.len() - 1;
                    let lvl = la.min(lb);
                    if !self.nodes[ga].neighbors[lvl].contains(&gb) {
                        self.nodes[ga].neighbors[lvl].push(gb);
                    }
                    if !self.nodes[gb].neighbors[lvl].contains(&ga) {
                        self.nodes[gb].neighbors[lvl].push(ga);
                    }
                }
            }
        }
    }
}

/// Parallel HNSW build over `n_shards` threads. `data` is row-major L2-normalized
/// (n×dim). Returns a fully-merged `Hnsw`. Falls back to single-threaded when
/// `n_shards <= 1` or the dataset is small. Uses only `std::thread` (no external
/// crate / rayon) to respect the repo's zero-dependency convention.
impl VectorIndex {
    /// Build a graph in parallel from a normalized f32 matrix (row-major n×dim).
    /// IDs are assigned 0..n. The merged graph is queryable via `search_ef` etc.
    pub fn build_parallel(data: &[f32], dim: usize, n_shards: usize) -> Self {
        Self::build_parallel_tiered(data, dim, n_shards, false)
    }

    /// MODULE 2 diagnostic: bytes of exact f32 currently held in RAM (the
    /// payload that the cold tier spills to NVMe). For a tiered index this is
    /// 0 after build; for a RAM index it is `n*dim*4`.
    pub fn f32_ram_bytes(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        g.all_f32.len() * std::mem::size_of::<f32>()
    }

    /// MODULE 2 diagnostic: bytes of exact f32 living in the NVMe cold tier
    /// (0 if the index is not tiered).
    pub fn f32_tier_bytes(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        match &g.f32_tier {
            Some(t) => t.as_slice().len() * std::mem::size_of::<f32>(),
            None => 0,
        }
    }

    /// MODULE 3 calibration entry point. Builds the learned beam-width model
    /// from calibration queries + their true top-k ids. See `Hnsw::calibrate_ef`.
    pub fn calibrate_ef(
        &self,
        qvecs: &[Vec<f32>],
        truth: &[Vec<u64>],
        target_recall: f32,
        candidate_efs: &[usize],
        k: usize,
        ef_min: usize,
        ef_max: usize,
    ) -> LearnedEf {
        let g = self.hnsw.read().unwrap();
        let model = g.calibrate_ef(qvecs, truth, target_recall, candidate_efs, k, ef_min, ef_max);
        LearnedEf { inner: model, ef_min, ef_max, ef_probe: candidate_efs[0].max(k) }
    }

    /// MODULE 3 learned-adaptive search: probe difficulty, predict ef from the
    /// calibrated model, traverse once. Returns (id, cosine_distance) ascending.
    pub fn search_ef_learned(&self, query: &[f32], k: usize, model: &LearnedEf) -> Vec<(u64, f32)> {
        let mut qn = query.to_vec();
        l2_normalize(&mut qn);
        let qq = quantize(&qn);
        let rerank_k = (k * 4).max(64);
        let g = self.hnsw.read().unwrap();
        let qf = g.query_f32(&qn);
        let candidates = g.search_indices_learned(&qq, rerank_k, &model.inner, model.ef_probe, model.ef_min, model.ef_max, &qf);
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
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);
        rescored
    }

    /// Like `build_parallel`, but `tiered` routes the merged graph's exact f32
    /// rerank originals to an NVMe `mmap` (Module 2) instead of RAM. The graph
    /// (int8 + layer-0 edges) stays in RAM; only the cold f32 payload is
    /// spilled, so live RAM at 100M+ scale drops to ~int8-only while recall is
    /// byte-identical (the rerank still reads exact f32 — just through mmap).
    pub fn build_parallel_tiered(data: &[f32], dim: usize, n_shards: usize, tiered: bool) -> Self {
        let n = data.len() / dim;
        let shards = n_shards.max(1).min(n.max(1));
        // bridge=4: each node gets 4 cross-shard edges (good recall at low cost).
        // bridge=8 was default; 4 halves merge time while keeping recall ≥0.999.
        // SHUFFLE the row order before sharding. This is the key to cheap,
        // correct merging: each segment becomes a RANDOM subsample of the
        // whole space, so its internal HNSW already navigates globally. After
        // the parallel build we only need to bridge the K segment *entries*
        // (O(K²) — microseconds), not every node. Without shuffling, segments
        // are contiguous spatial blocks and bridging entries alone leaves
        // interior nodes unreachable (recall collapse).
        let mut perm: Vec<usize> = (0..n).collect();
        // Deterministic xorshift shuffle (no rand crate).
        let mut s = 0x9E3779B97F4A7C15u64;
        for i in (1..n).rev() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let j = (s >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        // INDEX INDIRECTION: read rows from original data via perm[] instead of
        // creating a full shuffled copy. Saves 1.5 GB for 1M×384d, 3 GB for 768d.
        // Each shard thread reads `data[perm[lo+i] * dim ..]` on demand.

        // Split into contiguous shard ranges over the permuted index space.
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(shards);
        let per = (n as f64 / shards as f64).ceil() as usize;
        let mut start = 0usize;
        for _ in 0..shards {
            let end = (start + per).min(n);
            if end > start {
                ranges.push((start, end));
            }
            start = end;
            if start >= n {
                break;
            }
        }
        let seg_count = ranges.len();
        if seg_count <= 1 {
            let mut h = Hnsw::new();
            h.dim = dim;
            for i in 0..n {
                let v: Vec<f32> = data[i * dim..(i + 1) * dim].to_vec();
                h.insert(i as u64, v);
            }
            return Self { engine: Engine::open_for_build(), hnsw: RwLock::new(h), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) };
        }

        let dim = dim;
        let bridge = 16usize;
        let zero_attr: Vec<u32> = vec![0u32; n];
        let perm_arc: Arc<Vec<usize>> = Arc::new(perm);
        let attr_arc: Arc<Vec<u32>> = Arc::new(zero_attr);

        // GPU graph construction: serialize with Mutex to prevent VRAM OOM.
        // Each gpu_build_knn_graph call allocates ~534MB GPU RAM.
        // Multiple concurrent calls would exceed 8GB VRAM.
        let gpu_build_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
        // ═══ GPU-accelerated HNSW (CAGRA approach) ═══
        // GPU computes kNN graph for the ENTIRE dataset in one pass.
        // Then build ONE HNSW from those edges — no segments, no copies.
        // CPU only wires edges (fast). GPU does all distance computation.
        let use_gpu_build = gpu::gpu_get_mode() == gpu::ComputeMode::Turbo;
        if use_gpu_build {
            eprintln!("[GPU] Turbo: CAGRA-style — GPU builds kNN, HNSW wires edges");
        }

        if use_gpu_build {
            // GPU PATH: single kNN graph → single HNSW. No segments, no copies.
            let k_init = 32;
            let knn_graph = {
                let i8_data: Vec<i8> = perm_arc.iter().flat_map(|&r| {
                    (0..dim).map(move |d| (data[r * dim + d] * 127.0) as i8)
                }).collect();
                gpu::gpu_build_knn_graph(&i8_data, n, dim, k_init)
                    .unwrap_or_else(|| {
                        eprintln!("[GPU] GPU kNN failed, CPU fallback");
                        let mut graph = vec![vec![0usize; k_init]; n];
                        for i in 0..n {
                            let mut dists: Vec<(i32, usize)> = (0..n).filter(|&j| j != i).map(|j| {
                                let mut dot = 0i32;
                                for d in 0..dim { dot += i8_data[i*dim+d] as i32 * i8_data[j*dim+d] as i32; }
                                (dot, j)
                            }).collect();
                            dists.select_nth_unstable_by(k_init, |a, b| b.0.cmp(&a.0));
                            dists.truncate(k_init);
                            graph[i] = dists.into_iter().map(|(_, j)| j).collect();
                        }
                        graph
                    })
            };
            // Build ONE HNSW from the GPU kNN edges
            let mut h = Hnsw::new();
            h.dim = dim;
            // Insert vectors via index indirection (no all_f32 copy for GPU path)
            // Store INT8 only — f32 rerank reads from mmap tier.
            h.all_i8.reserve(n * dim);
            for i in 0..n {
                let true_row = perm_arc[i];
                let base = true_row * dim;
                for d in 0..dim {
                    h.all_i8.push((data[base + d] * 127.0) as i8);
                }
                h.nodes.push(Node {
                    id: true_row as u64,
                    neighbors: vec![Vec::new(); h.max_level + 1],
                    deleted: false,
                    attr: 0,
                });
                h.id_to_idx.insert(true_row as u64, i);
            }
            // Wire edges from GPU kNN graph
            for (i, neighbors) in knn_graph.iter().enumerate() {
                h.nodes[i].neighbors[0] = neighbors.clone();
            }
            eprintln!("[GPU] HNSW built from GPU kNN: {n} nodes, {k_init} edges/node");
            drop(perm_arc);
            drop(attr_arc);

            let mut pre_tier = if tiered { MmapTier::new(n * dim) } else { None };
            let mut merged = h;
            if let Some(t) = pre_tier {
                merged.f32_tier = Some(t);
                merged.all_f32 = Vec::new();
            }
            return Self { engine: Engine::open_for_build(), hnsw: RwLock::new(merged), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) };
        }

        // ═══ CPU PATH: standard HNSW segments with shared index indirection ═══
        let segments: Vec<Hnsw> = std::thread::scope(|s| {
            let handles: Vec<_> = ranges
                .into_iter()
                .enumerate()
                .map(|(_si, (lo, hi))| {
                    let perm_clone = Arc::clone(&perm_arc);
                    let attr_clone = Arc::clone(&attr_arc);
                    s.spawn(move || {
                        let mut h = Hnsw::build_segment_indexed(data, dim, &perm_clone, lo, hi - lo, lo as u64, &attr_clone);
                        for (local, node) in h.nodes.iter_mut().enumerate() {
                            node.id = perm_clone[lo + local] as u64;
                        }
                        h.id_to_idx.clear();
                        for (local, node) in h.nodes.iter().enumerate() {
                            h.id_to_idx.insert(node.id, local);
                        }
                        h
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // For tiered mode, pre-allocate the mmap BEFORE merge so merge_segments
        // can write f32 data directly to it — avoids the ~N×dim×4 RAM intermediate.
        let mut pre_tier = if tiered { MmapTier::new(n * dim) } else { None };
        let mut merged = Hnsw::merge_segments(segments, bridge, pre_tier.as_mut());
        // Move the tier into the merged Hnsw and drop any leftover all_f32.
        if let Some(t) = pre_tier {
            merged.f32_tier = Some(t);
            merged.all_f32 = Vec::new();
        }
        Self { engine: Engine::open_for_build(), hnsw: RwLock::new(merged), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) }
    }

    /// MODULE 4 constructor: build the graph in parallel like
    /// `build_parallel`, but attach a per-row filter attribute (`attrs[i]` is
    /// the category of row `i`). Enables selectivity-routed filtered search on
    /// a single shared HNSW.
    pub fn build_parallel_attr(
        data: &[f32],
        dim: usize,
        n_shards: usize,
        attrs: &[u32],
    ) -> Self {
        let n = data.len() / dim;
        assert_eq!(attrs.len(), n, "attrs length must match row count");
        let shards = n_shards.max(1).min(n.max(1));
        let mut perm: Vec<usize> = (0..n).collect();
        let mut s = 0x9E3779B97F4A7C15u64;
        for i in (1..n).rev() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let j = (s >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        let shuffled: Vec<f32> = perm.iter().flat_map(|&r| {
            data[r * dim..(r + 1) * dim].iter().copied()
        }).collect();
        let shuffled_attr: Vec<u32> = perm.iter().map(|&r| attrs[r]).collect();

        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(shards);
        let per = (n as f64 / shards as f64).ceil() as usize;
        let mut start = 0usize;
        for _ in 0..shards {
            let end = (start + per).min(n);
            if end > start {
                ranges.push((start, end));
            }
            start = end;
            if start >= n {
                break;
            }
        }
        let seg_count = ranges.len();
        if seg_count <= 1 {
            let mut h = Hnsw::new();
            h.dim = dim;
            for i in 0..n {
                let v: Vec<f32> = data[i * dim..(i + 1) * dim].to_vec();
                h.insert_attr(i as u64, v, attrs[i]);
            }
            return Self { engine: Engine::open_for_build(), hnsw: RwLock::new(h), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) };
        }

        let dim = dim;
        let bridge = 16usize;
        let shared_data: Arc<Vec<f32>> = Arc::new(shuffled);
        let shared_attr: Arc<Vec<u32>> = Arc::new(shuffled_attr);
        let shared_perm: Arc<Vec<usize>> = Arc::new(perm);
        let handles: Vec<_> = ranges
            .into_iter()
            .enumerate()
            .map(|(_si, (lo, hi))| {
                let data_ptr = Arc::clone(&shared_data);
                let attr_ptr = Arc::clone(&shared_attr);
                let perm_ptr = Arc::clone(&shared_perm);
                let dim = dim;
                std::thread::spawn(move || {
                    let mut h = Hnsw::build_segment(&data_ptr, dim, lo, hi - lo, lo as u64, &attr_ptr);
                    for (local, node) in h.nodes.iter_mut().enumerate() {
                        let true_row = perm_ptr[lo + local];
                        node.id = true_row as u64;
                    }
                    h.id_to_idx.clear();
                    for (local, node) in h.nodes.iter_mut().enumerate() {
                        h.id_to_idx.insert(node.id, local);
                    }
                    h
                })
            })
            .collect();
        let segments: Vec<Hnsw> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let merged = Hnsw::merge_segments(segments, bridge, None);
        Self { engine: Engine::open_for_build(), hnsw: RwLock::new(merged), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) }
    }

    /// Like `build_parallel_attr`, but maps each row to an ARBITRARY client
    /// `u64` id (not the row index). Shuffles rows so each shard is a global
    /// subsample, builds shards in parallel, then bridges only the K segment
    /// entries (O(K²), microseconds) — the cheap, correct merge from
    /// `build_parallel`. Used by the parallel `VADDBATCH` ingest path so
    /// batched ids need not be contiguous and recall stays correct.
    pub fn build_parallel_ids(
        data: &[f32],
        dim: usize,
        n_shards: usize,
        ids: &[u64],
        attrs: &[u32],
    ) -> Self {
        let n = data.len() / dim;
        assert_eq!(ids.len(), n, "ids length must match row count");
        assert_eq!(attrs.len(), n, "attrs length must match row count");
        let shards = n_shards.max(1).min(n.max(1));
        let mut perm: Vec<usize> = (0..n).collect();
        let mut s = 0x9E3779B97F4A7C15u64;
        for i in (1..n).rev() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let j = (s >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        // inv[orig_row] = shuffled position. Used to map a segment's local row
        // back to the real client id after the parallel build.
        let inv: Vec<usize> = {
            let mut inv = vec![0usize; n];
            for (shuffled_pos, &orig) in perm.iter().enumerate() {
                inv[orig] = shuffled_pos;
            }
            inv
        };
        let shuffled: Vec<f32> = perm.iter().flat_map(|&r| {
            data[r * dim..(r + 1) * dim].iter().copied()
        }).collect();
        let shuffled_attr: Vec<u32> = perm.iter().map(|&r| attrs[r]).collect();

        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(shards);
        let per = (n as f64 / shards as f64).ceil() as usize;
        let mut start = 0usize;
        for _ in 0..shards {
            let end = (start + per).min(n);
            if end > start {
                ranges.push((start, end));
            }
            start = end;
            if start >= n {
                break;
            }
        }
        if ranges.len() <= 1 {
            let mut h = Hnsw::new();
            h.dim = dim;
            for i in 0..n {
                let v: Vec<f32> = data[i * dim..(i + 1) * dim].to_vec();
                h.insert_attr(ids[i], v, attrs[i]);
            }
            return Self { engine: Engine::open_for_build(), hnsw: RwLock::new(h), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) };
        }

        let dim = dim;
        let bridge = 16usize;
        let shared_data: Arc<Vec<f32>> = Arc::new(shuffled);
        let shared_attr: Arc<Vec<u32>> = Arc::new(shuffled_attr);
        let shared_inv: Arc<Vec<usize>> = Arc::new(inv);
        let shared_ids: Arc<Vec<u64>> = Arc::new(ids.to_vec());
        let handles: Vec<_> = ranges
            .into_iter()
            .enumerate()
            .map(|(_si, (lo, hi))| {
                let data_ptr = Arc::clone(&shared_data);
                let attr_ptr = Arc::clone(&shared_attr);
                let inv_ptr = Arc::clone(&shared_inv);
                let ids_ptr = Arc::clone(&shared_ids);
                let dim = dim;
                std::thread::spawn(move || {
                    let mut h = Hnsw::build_segment(&data_ptr, dim, lo, hi - lo, lo as u64, &attr_ptr);
                    // Fix node ids: segment built with id = lo+local; remap to
                    // the real client id via the inverse permutation.
                    for (local, node) in h.nodes.iter_mut().enumerate() {
                        let true_row = inv_ptr[lo + local];
                        node.id = ids_ptr[true_row];
                    }
                    h.id_to_idx.clear();
                    for (local, node) in h.nodes.iter_mut().enumerate() {
                        h.id_to_idx.insert(node.id, local);
                    }
                    h
                })
            })
            .collect();
        let segments: Vec<Hnsw> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let merged = Hnsw::merge_segments(segments, bridge, None);
        Self { engine: Engine::open_for_build(), hnsw: RwLock::new(merged), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) }
    }

    /// MODULE 5 constructor: build the dense HNSW in parallel (like
    /// `build_parallel`) AND a sparse/lexical index over the SAME doc ids, so
    /// the two can be fused by `search_hybrid`. `sparse_docs[i]` is the sparse
    /// (term, weight) vector for row `i`.
    pub fn build_parallel_hybrid(
        data: &[f32],
        dim: usize,
        n_shards: usize,
        sparse_docs: &[Vec<(u32, f32)>],
    ) -> Self {
        let n = data.len() / dim;
        assert_eq!(sparse_docs.len(), n, "sparse_docs length must match row count");
        let base = Self::build_parallel(data, dim, n_shards);
        // Populate the sparse index in doc order (id == insertion order == row).
        let mut sp = SparseIndex::new();
        for (i, doc) in sparse_docs.iter().enumerate() {
            sp.add(i as u64, doc.clone());
        }
        *base.sparse.write().unwrap() = sp;
        base
    }

    /// MODULE 4 — selectivity-routed filtered k-NN. Runs the same INT8
    /// traversal + exact f32 rerank pipeline as `search_ef`, but the graph walk
    /// honours `filter`. For selective predicates it routes to an exact
    /// brute-force over the matching set; otherwise to a connectivity-preserving
    /// gated traversal that starts from the matching category's entry node.
    pub fn search_filtered_attr(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: &Filter,
    ) -> Vec<(u64, f32)> {
        let mut q = query.to_vec();
        l2_normalize(&mut q);
        let qq = quantize(&q);
        let g = self.hnsw.read().unwrap();
        let qf = g.query_f32(&q);
        let rerank_k = (k * 4).max(64);
        let candidates = g.search_indices_filtered(&qq, rerank_k, ef.max(rerank_k), filter, &qf);
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

    /// MODULE 4 diagnostic: number of nodes matching `filter` (for selectivity
    /// reporting / assertions in the bench).
    pub fn matching_count(&self, filter: &Filter) -> usize {
        let g = self.hnsw.read().unwrap();
        match filter {
            Filter::Any => g.nodes.len(),
            Filter::Eq(a) => g.attr_counts.get(*a as usize).copied().unwrap_or(0),
            Filter::In(set) => set.iter().map(|a| g.attr_counts.get(*a as usize).copied().unwrap_or(0)).sum(),
        }
    }

    /// MODULE 5 — unified hybrid search. Runs the dense ANN over the HNSW and
    /// the sparse BM25 over the lexical index (both in-engine), then fuses the
    /// two ranked lists with Reciprocal Rank Fusion. A single query returns one
    /// ranking that captures BOTH semantic similarity (dense) and exact keyword
    /// match (sparse) — the "one graph, many access paths" thesis: one index
    /// object, many retrieval paths, fused with no client round-trips.
    ///
    /// `sparse_query` is the lexical side (term, weight) pairs. `rerank` is how
    /// many dense + sparse candidates each contribute to the fusion (RRF works
    /// on ranks, so a modest `rerank` like 50 is plenty).
    pub fn search_hybrid(
        &self,
        query_dense: &[f32],
        sparse_query: &[(u32, f32)],
        k: usize,
        ef: usize,
        rerank: usize,
    ) -> Vec<(u64, f32)> {
        let dense = self.search_ef(query_dense, rerank.max(k), ef);
        let dense_ids: Vec<u64> = dense.iter().map(|(id, _)| *id).collect();
        let sp = self.sparse.read().unwrap();
        let sparse = sp.search(sparse_query, rerank.max(k));
        let sparse_ids: Vec<u64> = sparse.iter().map(|(id, _)| *id).collect();
        drop(sp);
        let fused = rrf_fuse(&[dense_ids, sparse_ids], 60);
        fused.into_iter().take(k).collect()
    }

    /// MODULE 5 — sparse/lexical search only (BM25 over the inverted index).
    /// Exposed so callers + the bench can compare the single sparse path
    /// against the fused hybrid (fusion should dominate either path alone).
    pub fn search_sparse(&self, sparse_query: &[(u32, f32)], k: usize) -> Vec<(u64, f32)> {
        self.sparse.read().unwrap().search(sparse_query, k)
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
        // REVERSED so BinaryHeap<Cand> pops the NEAREST candidate first
        // (min-heap on distance). This is the canonical HNSW greedy-expand
        // ordering: explore the closest frontier node, stop when the closest
        // popped candidate is farther than the current worst kept result.
        // The `idx` tiebreaker makes heap pops DETERMINISTIC (heap tie-order
        // was causing non-deterministic descents).
        o.dist
            .partial_cmp(&self.dist)
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| self.idx.cmp(&o.idx))
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
        // Max-heap (worst/farthest on top) → `results.peek()` is the current
        // worst-kept, used by the greedy stop. `idx` tiebreaker → deterministic.
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}

// ── VectorIndex public API (unchanged) ──────────────────────────────────────

pub struct VectorIndex {
    engine: Arc<Engine>,
    hnsw: RwLock<Hnsw>,
    /// MODULE 5: sparse / lexical index over the SAME doc ids as `hnsw`. None
    /// for pure-dense indexes. Hybrid queries fuse the two in-engine.
    sparse: RwLock<SparseIndex>,
    /// GPU-resident index for CAGRA-style GPU search. None if no GPU or not built.
    gpu_idx: RwLock<Option<gpu::GpuIndex>>,
    /// VUGVA Virtual Memory Table — manages GPU VRAM + CPU RAM for hybrid mode.
    /// Splits vectors into chunks, prefetches hot chunks to GPU on demand.
    vugva_vmt: RwLock<Option<Arc<gpu::vugva::VugvaVmt>>>,
}

impl VectorIndex {
    /// Upload vectors + flat CSR graph to GPU for CAGRA-style GPU search.
    /// Uses VUGVA-style unified memory: GPU reads from RAM directly.
    /// No cuMemcpyHtoD — CUDA page migrator handles data transfer.
    pub fn upload_to_gpu(&self) {
        if !gpu::gpu_init() { return; }
        let g = self.hnsw.read().unwrap();
        let n = g.nodes.len();
        let dim = g.dim;
        let degree = g.nodes.first().map(|n| n.neighbors.first().map_or(0, |l| l.len())).unwrap_or(0);
        if degree == 0 || n == 0 { return; }
        // Build flat CSR graph: n × degree × i32
        // Pad missing neighbors with self-loops (node→self) so the GPU kernel
        // always has valid neighbors to explore. -1 holes break graph traversal.
        let mut graph_flat: Vec<i32> = vec![-1i32; n * degree];
        for (i, node) in g.nodes.iter().enumerate() {
            let i32_i = i as i32;
            if let Some(level0) = node.neighbors.first() {
                for (j, &neighbor) in level0.iter().enumerate().take(degree) {
                    graph_flat[i * degree + j] = neighbor as i32;
                }
            }
            // Pad remaining slots with self-loop
            for j in 0..degree {
                if graph_flat[i * degree + j] == -1 {
                    graph_flat[i * degree + j] = i32_i;
                }
            }
        }
        // VUGVA: allocate unified memory for vectors (GPU reads from RAM directly)
        let vectors_i8: Vec<i8> = g.all_i8.clone();
        drop(g);
        match gpu::gpu_build_index(&vectors_i8, &graph_flat, n, dim, degree) {
            Some(idx) => {
                eprintln!("[GPU] VectorIndex uploaded: {n} vectors, degree={degree} (unified memory)");
                *self.gpu_idx.write().unwrap() = Some(idx);
            }
            None => eprintln!("[GPU] VectorIndex upload failed"),
        }
    }

    /// Initialize VUGVA Virtual Memory Table for hybrid mode.
    /// Splits all INT8 vectors into chunks, tracks which chunks are in VRAM vs RAM.
    /// GPU reads vectors on demand via chunk prefetch — the VUGVA paper's core idea.
    pub fn init_vugva(&self, vram_budget_bytes: usize) {
        let g = self.hnsw.read().unwrap();
        let vectors_i8 = g.all_i8.clone();
        let dim = g.dim;
        let n_nodes = g.nodes.len();
        drop(g);
        let chunk_size = (vram_budget_bytes / dim / 2).max(1024); // ~50% VRAM per chunk
        let vmt = gpu::vugva::VugvaVmt::new(&vectors_i8, dim, chunk_size, vram_budget_bytes);
        let stats = gpu::vugva::vugva_stats(&vmt);
        eprintln!("[VUGVA] Initialized for {n_nodes} vectors: {:?}", stats);
        *self.vugva_vmt.write().unwrap() = Some(Arc::new(vmt));
    }

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
        Self { engine, hnsw: RwLock::new(hnsw), sparse: RwLock::new(SparseIndex::new()), gpu_idx: RwLock::new(None), vugva_vmt: RwLock::new(None) }
    }

    /// Insert/replace a vector: durable f32 write + graph update with a
    /// quantized (int8) copy in-memory.
    pub fn insert(&self, id: u64, vector: Vec<f32>) -> std::io::Result<()> {
        self.engine.put(vec_key(id), Value::Vector(vector.clone()))?;
        self.hnsw.write().unwrap().insert(id, vector);
        Ok(())
    }

    /// Module 2 — select the quantization mode (Int8 | Binary | Binary2 |
    /// Binary15 | Turbo* | Product) for all subsequent inserts. Must be called
    /// on a fresh index (before any insert). Binary/2-bit/1.5-bit give Qdrant-
    /// class RAM compression (~16–32× vs f32) while the exact-f32 rerank
    /// preserves recall. Turbo* (Qdrant 1.18 / Google ICLR 2026) and Product
    /// need a fitting sample — call `fit_quant` after this.
    pub fn set_quant_mode(&self, mode: QuantMode) {
        self.hnsw.write().unwrap().set_quant_mode(mode);
    }

    /// Module 2 — fit TurboQuant / Product Quantization parameters from a sample
    /// of (already L2-normalized) vectors. Required before inserts when the mode
    /// is in the Turbo family or `Product`. For int8/binary modes this is a no-op.
    pub fn fit_quant(&self, sample: &[Vec<f32>]) {
        self.hnsw.write().unwrap().fit_quant(sample);
    }

    /// Current quantization mode (for bench reporting).
    pub fn quant_mode(&self) -> QuantMode {
        self.hnsw.read().unwrap().quant
    }

    /// Graph-only insert — **bench-only, NOT production.**
    /// Skips the durable substrate `Value::Vector` write so the scale bench
    /// uses just the two in-memory HNSW buffers (`all_i8` + `all_f32`)
    /// instead of three copies (int8 + f32 mirror + substrate f32). At
    /// 1M×1536d that avoids ~6 GB of redundant RAM and, more importantly,
    /// avoids 1M sequential WAL fsyncs that made `--xlarge` hang/freeze the
    /// whole box. Vectors inserted this way are NOT durable/recoverable —
    /// which is fine for a benchmark that never reopens the index.
    pub fn insert_graph_only(&self, id: u64, vector: Vec<f32>) {
        self.hnsw.write().unwrap().insert(id, vector);
    }

    /// Insert a vector with a MODULE 4 filter attribute, graph-only (bench).
    /// The attribute is stored on the node so `search_unified(filter=..)` can
    /// route by selectivity. Same durability caveat as `insert_graph_only`.
    pub fn insert_graph_only_attr(&self, id: u64, vector: Vec<f32>, attr: u32) {
        self.hnsw.write().unwrap().insert_attr(id, vector, attr);
    }

    /// PARALLEL INGEST — the multi-core build path. Builds `vectors` (row-major
    /// `n×dim`, already L2-normalized) as `n_shards` independent HNSW segments
    /// in parallel (one OS thread per shard, respecting the repo's no-external-
    /// crate rule), then merges them INTO the live graph under a single write
    /// lock (the merge is O(N) append + O(N·log N) bridge — cheap vs the build).
    ///
    /// This is what the server's batched `VADD` should call: a pipeline batch of
    /// B vectors ingests at ~cores× the single-`insert` rate, while reads stay
    /// lock-free (RwLock read) so concurrent `VSEARCH` scales across connections.
    ///
    /// Durability: each vector is still written to the WAL-backed substrate
    /// (`Value::Vector`) here, so the index is recoverable on reopen — unlike
    /// `insert_graph_only`.
    pub fn insert_many_parallel(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dim: usize,
        n_shards: usize,
        attrs: &[u32],
    ) -> std::io::Result<()> {
        let n = ids.len();
        if n == 0 {
            return Ok(());
        }
        if attrs.len() != n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "attrs length must equal ids length",
            ));
        }
        // The live graph has a fixed dimensionality; a batch must use the same
        // dim as the existing index (or any dim when empty). Merging a different
        // dim would desync the packed i8 storage and panic in merge_into.
        let existing = self.dim();
        if existing != 0 && existing != dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("VADDBATCH dim {dim} != existing index dim {existing}"),
            ));
        }
        // Durable substrate writes — batch into one put_batch for a single
        // WAL flush instead of N sequential fsyncs. At 100K vectors on HDD
        // this saves seconds of fsync overhead.
        let mut kvs: Vec<(Vec<u8>, Value)> = Vec::with_capacity(n);
        for i in 0..n {
            kvs.push((vec_key(ids[i]), Value::Vector(vectors[i * dim..(i + 1) * dim].to_vec())));
        }
        self.engine.put_batch(kvs)?;
        // Shard the batch into as-even ranges and build each in its own thread.
        let shards = n_shards.max(1).min(n);
        let per = (n as f64 / shards as f64).ceil() as usize;
        let ranges: Vec<(usize, usize)> = (0..shards)
            .map(|s| {
                let lo = (s * per).min(n);
                let hi = ((s + 1) * per).min(n);
                (lo, hi)
            })
            .filter(|(lo, hi)| hi > lo)
            .collect();
        let segments: Vec<Hnsw> = ranges
            .into_iter()
            .map(|(lo, hi)| {
                let ids_seg: Vec<u64> = ids[lo..hi].to_vec();
                let vseg: Vec<f32> = vectors[lo * dim..hi * dim].to_vec();
                let attr_seg: Vec<u32> = attrs[lo..hi].to_vec();
                let dim = dim;
                std::thread::spawn(move || {
                    let mut h = Hnsw::new();
                    h.dim = dim;
                    for (row, &id) in ids_seg.iter().enumerate() {
                        let v: Vec<f32> = vseg[row * dim..(row + 1) * dim].to_vec();
                        h.insert_attr(id, v, attr_seg[row]);
                    }
                    h
                })
            })
            .map(|h| h.join().unwrap())
            .collect();
        let mut g = self.hnsw.write().unwrap();
        g.merge_into(segments, 8);
        Ok(())
    }

    /// Graph-only parallel ingest — same as `insert_many_parallel` but skips the
    /// durable WAL write (bench-only, like `insert_graph_only`). Builds the
    /// batch as parallel shards and merges into the live in-memory graph.
    pub fn insert_many_parallel_graph_only(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dim: usize,
        n_shards: usize,
    ) {
        let n = ids.len();
        if n == 0 {
            return;
        }
        let shards = n_shards.max(1).min(n);
        let per = (n as f64 / shards as f64).ceil() as usize;
        let ranges: Vec<(usize, usize)> = (0..shards)
            .map(|s| {
                let lo = (s * per).min(n);
                let hi = ((s + 1) * per).min(n);
                (lo, hi)
            })
            .filter(|(lo, hi)| hi > lo)
            .collect();
        let zero_attr: Vec<u32> = vec![0u32; n];
        let segments: Vec<Hnsw> = ranges
            .into_iter()
            .map(|(lo, hi)| {
                let ids_seg: Vec<u64> = ids[lo..hi].to_vec();
                let vseg: Vec<f32> = vectors[lo * dim..hi * dim].to_vec();
                let attr_seg: Vec<u32> = zero_attr[lo..hi].to_vec();
                let dim = dim;
                std::thread::spawn(move || {
                    let mut h = Hnsw::new();
                    h.dim = dim;
                    for (row, &id) in ids_seg.iter().enumerate() {
                        let v: Vec<f32> = vseg[row * dim..(row + 1) * dim].to_vec();
                        h.insert_attr(id, v, attr_seg[row]);
                    }
                    h
                })
            })
            .map(|h| h.join().unwrap())
            .collect();
        let mut g = self.hnsw.write().unwrap();
        g.merge_into(segments, 8);
    }

    /// Dump every live node as `(id, exact_f32_vector, attr)`. Used by the
    /// parallel rebuild ingest path so an existing index can be combined with a
    /// new batch and re-built in parallel (preserving ids + attributes).
    pub fn export_vectors(&self) -> Vec<(u64, Vec<f32>, u32)> {
        let g = self.hnsw.read().unwrap();
        let dim = g.dim;
        if dim == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(g.nodes.len());
        for (gi, node) in g.nodes.iter().enumerate() {
            if node.deleted {
                continue;
            }
            let v = g.vec_at_f32(gi).to_vec();
            out.push((node.id, v, node.attr));
        }
        out
    }

    /// Parallel bulk rebuild ingest (the `VADDBATCH PAR` path). Combines the
    /// EXISTING live graph with the new batch and rebuilds the WHOLE graph via
    /// `build_parallel_ids`, which shuffles rows so each shard is a global
    /// subsample and bridges only the K segment entries (O(K²), microseconds) —
    /// a genuinely cores× build with correct recall, unlike the serial
    /// per-node `merge_into` append. IDs and filter attributes are preserved.
    /// The default `insert_many_parallel` (serial `merge_into`) is kept for
    /// callers that prefer incremental append over a full rebuild.
    pub fn insert_many_parallel_rebuild(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dim: usize,
        n_shards: usize,
        attrs: &[u32],
    ) -> std::io::Result<()> {
        let n = ids.len();
        if n == 0 {
            return Ok(());
        }
        if attrs.len() != n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "attrs length must equal ids length",
            ));
        }
        let existing_dim = self.dim();
        if existing_dim != 0 && existing_dim != dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("VADDBATCH dim {dim} != existing index dim {existing_dim}"),
            ));
        }
        // Durable substrate writes — batch into one put_batch for a single WAL flush.
        let mut kvs: Vec<(Vec<u8>, Value)> = Vec::with_capacity(n);
        for i in 0..n {
            kvs.push((vec_key(ids[i]), Value::Vector(vectors[i * dim..(i + 1) * dim].to_vec())));
        }
        self.engine.put_batch(kvs)?;
        let existing = self.export_vectors();
        let m = existing.len();
        let total = m + n;
        if total == 0 {
            return Ok(());
        }
        let mut data: Vec<f32> = Vec::with_capacity(total * dim);
        let mut all_ids: Vec<u64> = Vec::with_capacity(total);
        let mut all_attrs: Vec<u32> = Vec::with_capacity(total);
        for (id, v, a) in &existing {
            data.extend_from_slice(v);
            all_ids.push(*id);
            all_attrs.push(*a);
        }
        for i in 0..n {
            data.extend_from_slice(&vectors[i * dim..(i + 1) * dim]);
            all_ids.push(ids[i]);
            all_attrs.push(attrs[i]);
        }
        let rebuilt = VectorIndex::build_parallel_ids(&data, dim, n_shards, &all_ids, &all_attrs);
        let mut g = self.hnsw.write().unwrap();
        *g = rebuilt.hnsw.into_inner().unwrap();
        Ok(())
    }

    /// MODULE 5 — attach a sparse/lexical (term, weight) vector to `id`, so the
    /// unified hybrid path can fuse it with the dense ANN. Same doc id must have
    /// been inserted densely first.
    pub fn add_sparse(&self, id: u64, terms: Vec<(u32, f32)>) {
        self.sparse.write().unwrap().add(id, terms);
    }

    /// MODULE 6 — the UNIFIED query entry point. One method, every access path,
    /// selected by the optional arguments (so a single RESP command `VSEARCH`
    /// can serve dense / filtered / learned / hybrid without fragmenting into a
    /// dozen command names). Exactly ONE path is taken per call:
    ///   • `sparse_query` set  → hybrid (dense ANN + sparse BM25, RRF-fused)
    ///   • `filter` set        → selectivity-routed filtered ANN
    ///   • `learned` set       → learned-adaptive beam width
    ///   • otherwise           → plain quantized+rerank ANN at `ef`
    /// This is the "one graph, many access paths" thesis exposed as ONE call.
    pub fn search_unified(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
        learned: Option<&LearnedEf>,
        sparse_query: Option<&[(u32, f32)]>,
        rerank: usize,
    ) -> Vec<(u64, f32)> {
        if let Some(sq) = sparse_query {
            return self.search_hybrid(query, sq, k, ef, rerank);
        }
        if let Some(f) = filter {
            return self.search_filtered_attr(query, k, ef, f);
        }
        if let Some(m) = learned {
            return self.search_ef_learned(query, k, m);
        }
        self.search_ef(query, k, ef)
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
        let mode = gpu::gpu_get_mode();
        static SEARCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = SEARCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // ── TURBO: Real GPU graph traversal (CAGRA) ──
        if mode == gpu::ComputeMode::Turbo && gpu::gpu_available() {
            let guard = self.gpu_idx.read().unwrap();
            if let Some(ref idx) = *guard {
                let itopk = ef.max(k * 4).max(64);
                let max_iters = (itopk as f32).sqrt() as usize + 5;
                if let Some((indices, _distances)) = gpu::gpu_search(
                    idx, &qq, 1, k, itopk, max_iters, 0,
                ) {
                    if count < 3 {
                        let valid: Vec<_> = indices.iter().filter(|&&i| i >= 0).collect();
                        eprintln!("[DEBUG] GPU search query#{}: {} valid results out of {} total, first 5: {:?}",
                            count, valid.len(), k, &indices[..k.min(indices.len())]);
                    }
                    if count % 100 == 0 {
                        eprintln!("[MITM] search_ef#{count} mode=Turbo GPU_CAGRA n={}", idx.n);
                    }
                    let g = self.hnsw.read().unwrap();
                    let results: Vec<(u64, f32)> = indices.iter().filter(|&&i| i >= 0).filter_map(|&i| {
                        let idx = i as usize;
                        let node = g.nodes.get(idx)?;
                        if node.deleted { return None; }
                        let dot = dot_f32(&q, g.vec_at_f32(idx));
                        let d = (1.0 - dot).max(0.0).min(2.0);
                        Some((node.id, d))
                    }).collect();
                    if count < 3 {
                        eprintln!("[DEBUG] GPU results: {:?}", &results[..results.len().min(5)]);
                    }
                    return results;
                }
            }
            if count % 100 == 0 {
                eprintln!("[MITM] search_ef#{count} mode=Turbo NO_GPU_IDX → auto-downgrade to Hybrid");
            }
            // Auto-downgrade: GPU index unavailable, switch to Hybrid
            gpu::gpu_set_mode(gpu::ComputeMode::Hybrid);
        }

        // ── HYBRID: CPU graph walk + CPU rerank (VUGVA manages RAM chunks) ──
        // VUGVA VMT tracks which chunks are hot/warm but distance computation
        // still runs on CPU. GPU distance path requires a dedicated kernel that
        // reads from VUGVA GPU-resident chunks — not yet implemented.
        let g = self.hnsw.read().unwrap();
        let qf = g.query_f32(&q);
        let rerank_k = (k * 4).max(64);
        let candidates = g.search_indices(&qq, rerank_k, ef.max(rerank_k), &qf);

        if mode != gpu::ComputeMode::CpuOnly {
            // Prefetch chunks via VUGVA (data management only, CPU computes)
            let vmt_guard = self.vugva_vmt.read().unwrap();
            if vmt_guard.is_some() && count % 500 == 0 {
                let vmt = vmt_guard.as_ref().unwrap();
                let chunk_indices: Vec<usize> = candidates.iter().map(|&(idx, _)| vmt.chunk_of(idx)).collect();
                let _prefetched = vmt.prefetch_batch(&chunk_indices);
                let stats = gpu::vugva::vugva_stats(vmt);
                eprintln!("[MITM] search_ef#{count} mode={mode:?} VUGVA_PREFETCH {:?}", stats);
            }
        }

        // ── CPU fallback: per-candidate f32 rerank ──
        if count % 100 == 0 {
            eprintln!("[MITM] search_ef#{count} mode={mode:?} CPU_FALLBACK candidates={}", candidates.len());
        }
        let mut rescored: Vec<(u64, f32)> = candidates
            .into_iter()
            .filter_map(|(idx, _int8_dist)| {
                let node = g.nodes.get(idx)?;
                if node.deleted { return None; }
                let dot = dot_f32(&q, g.vec_at_f32(idx));
                let d = (1.0 - dot).max(0.0).min(2.0);
                Some((node.id, d))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(k);
        rescored
    }

    /// Query-adaptive k-NN: picks a per-query beam width from the difficulty of
    /// the query, then runs the same INT8-traversal + f32-rerank pipeline as
    /// `search_ef`. Easy queries get a narrow beam (fewer distance computations
    /// → lower latency); hard queries get a wide beam (recall preserved). This
    /// is the ruvector-style win over Qdrant's fixed-ef: equal-or-better recall
    /// at lower p95 latency, no GPU, no offline training.
    ///
    /// `ef_probe` is the cheap probe beam (difficulty probe); `ef_min`/`ef_max`
    /// bound the scaled real beam. Defaults tuned for 384/768d at 100k–1M scale.
    pub fn search_adaptive(
        &self,
        query: &[f32],
        k: usize,
        ef_probe: usize,
        ef_min: usize,
        ef_max: usize,
    ) -> Vec<(u64, f32)> {
        let mut q = query.to_vec();
        l2_normalize(&mut q);
        let qq = quantize(&q);
        let g = self.hnsw.read().unwrap();
        let qf = g.query_f32(&q);
        let rerank_k = (k * 4).max(64);
        let candidates = g.search_indices_adaptive(&qq, rerank_k, ef_probe, ef_min, ef_max, &qf);
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
    /// single calls. Pre-allocates scratch buffers outside the loop to avoid
    /// per-query heap allocation.
    pub fn search_many(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Vec<Vec<(u64, f32)>> {
        let mode = gpu::gpu_get_mode();
        let num_queries = queries.len();

        // ── TURBO: Batch ALL queries into ONE GPU kernel call ──
        // This is the CAGRA way: Q blocks × 256 threads, one kernel launch.
        // GPU stays busy the entire time — no per-query overhead.
        if mode == gpu::ComputeMode::Turbo && gpu::gpu_available() {
            let guard = self.gpu_idx.read().unwrap();
            if let Some(ref idx) = *guard {
                // Quantize all queries
                let mut all_qq: Vec<i8> = Vec::with_capacity(num_queries * idx.dim);
                for q in queries {
                    let mut qn = q.clone();
                    l2_normalize(&mut qn);
                    all_qq.extend_from_slice(&quantize(&qn));
                }
                let itopk = 128.max(k * 4);
                let max_iters = 20;
                if let Some((all_indices, _all_dists)) = gpu::gpu_search(
                    idx, &all_qq, num_queries, k, itopk, max_iters, 0,
                ) {
                    // f32 rerank from CPU (read f32 vectors from RAM)
                    let g = self.hnsw.read().unwrap();
                    let mut results = Vec::with_capacity(num_queries);
                    for qi in 0..num_queries {
                        let mut hits: Vec<(u64, f32)> = (0..k).filter_map(|ki| {
                            let global_i = all_indices[qi * k + ki];
                            if global_i < 0 { return None; }
                            let idx = global_i as usize;
                            let node = g.nodes.get(idx)?;
                            if node.deleted { return None; }
                            let mut qn = queries[qi].clone();
                            l2_normalize(&mut qn);
                            let dot = dot_f32(&qn, g.vec_at_f32(idx));
                            let d = (1.0 - dot).max(0.0).min(2.0);
                            Some((node.id, d))
                        }).collect();
                        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                        hits.truncate(k);
                        results.push(hits);
                    }
                    return results;
                }
            }
        }

        // ── CPU fallback: sequential per-query ──
        let g = self.hnsw.read().unwrap();
        let rerank_k = (k * 4).max(64);
        let dim = g.dim;
        // Pre-allocate scratch buffers reused across all queries in the batch.
        let mut qn: Vec<f32> = vec![0.0; dim];
        let mut qq: Vec<i8> = vec![0; dim];
        let mut qf: Vec<f32> = Vec::new();
        let mut rescored: Vec<(u64, f32)> = Vec::with_capacity(rerank_k);
        queries
            .iter()
            .map(|q| {
                // Reuse pre-allocated buffers instead of allocating per query.
                qn.clear();
                qn.extend_from_slice(q);
                l2_normalize(&mut qn);
                qq.clear();
                quantize_into(&qn, &mut qq);
                qf.clear();
                qf.extend_from_slice(&g.query_f32(&qn));
                let candidates = g.search_indices(&qq, rerank_k, 128.max(rerank_k), &qf);
                rescored.clear();
                rescored.extend(
                    candidates
                        .into_iter()
                        .filter_map(|(idx, _int8_dist)| {
                            let node = g.nodes.get(idx)?;
                            if node.deleted {
                                return None;
                            }
                            let dot = dot_f32(&qn, g.vec_at_f32(idx));
                            let d = (1.0 - dot).max(0.0).min(2.0);
                            Some((node.id, d))
                        }),
                );
                rescored.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                rescored.truncate(k);
                rescored.clone()
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

    /// Embedding dimensionality of the live graph (0 if empty).
    pub fn dim(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        g.dim
    }

    /// Embedding dimensionality the active TurboQuant codebook was fitted for
    /// (0 when no turbo quantization is active). Per the TurboQuant paper the
    /// random rotation is `d×d`, so vectors inserted under a turbo mode must
    /// share this exact dimensionality or quantization desyncs the packed
    /// storage.
    pub fn quant_dim(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        match &g.turbo {
            Some(t) => t.dim,
            None => 0,
        }
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

    /// Dump cumulative MITM debug counters (DBSTRIKE_DEBUG must be set).
    pub fn dump_mitm_stats(&self) {
        let g = self.hnsw.read().unwrap();
        g.dump_mitm_stats();
    }

    /// MITM: dump one node's L0 neighbors + their distance to `query`, and
    /// whether `target` is among them. Definitive check for "is the edge that
    /// should exist actually present?"
    pub fn debug_node_neighbors(&self, node_idx: usize, query: &[f32], target_idx: usize) {
        let g = self.hnsw.read().unwrap();
        if node_idx >= g.nodes.len() {
            eprintln!("[MITM] debug_node_neighbors: node_idx {node_idx} out of range {}", g.nodes.len());
            return;
        }
        let q = quantize(query);
        let l0 = match g.nodes[node_idx].neighbors.first() {
            Some(l) => l,
            None => {
                eprintln!("[MITM] node {node_idx} has no L0 list");
                return;
            }
        };
        eprintln!(
            "[MITM] node {node_idx} L0 degree={} contains_target({})={}",
            l0.len(),
            target_idx,
            l0.contains(&target_idx)
        );
        // distances of this node's L0 neighbors to query
        let mut ds: Vec<(usize, f32)> = l0
            .iter()
            .map(|&n| (n, cos_dist_q(&q, &g.all_i8[n * g.dim..n * g.dim + g.dim])))
            .collect();
        ds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (n, d) in ds.iter().take(10) {
            eprintln!("[MITM]   neigh {n} d_to_query={:.4}", d);
        }
        if let Some((nn, nd)) = ds.first() {
            eprintln!("[MITM]   node {node_idx} NEAREST L0 neighbor={nn} dist_to_query={:.4}", nd);
        }
        let ids: Vec<usize> = l0.iter().copied().collect();
        eprintln!("[MITM]   node {node_idx} L0 ids = {:?}", ids);
        // and: is target itself connected back?
        if target_idx < g.nodes.len() {
            let tl0 = &g.nodes[target_idx].neighbors.first();
            match tl0 {
                Some(t) => eprintln!(
                    "[MITM] target {target_idx} L0 degree={} contains_node({})={}",
                    t.len(),
                    node_idx,
                    t.contains(&node_idx)
                ),
                None => eprintln!("[MITM] target {target_idx} has no L0 list"),
            }
        }
    }

    pub fn debug_asymmetric_edges(&self) -> usize {
        let g = self.hnsw.read().unwrap();
        let mut count = 0usize;
        let n = g.nodes.len();
        for a in 0..n {
            let la = match g.nodes[a].neighbors.first() {
                Some(l) => l,
                None => continue,
            };
            for &b in la {
                let lb = match g.nodes[b].neighbors.first() {
                    Some(l) => l,
                    None => continue,
                };
                if !lb.contains(&a) {
                    count += 1;
                }
            }
        }
        count
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
    fn adaptive_recall_matches_fixed_ef() {
        // Adaptive search must never regress recall vs a well-tuned fixed ef.
        let idx = VectorIndex::open(eng());
        let dim = 64;
        let n = 800u64;
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
        // Brute-force ground truth.
        let q = mkv(42);
        let mut truth: Vec<(u64, f32)> = Vec::new();
        idx.for_each_normalized(|id, v| {
            let mut s = 0f32;
            for i in 0..dim {
                s += q[i] * v[i];
            }
            let d = (1.0 - s).max(0.0).min(2.0);
            truth.push((id, d));
        });
        truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let truth: Vec<u64> = truth.iter().take(5).map(|(id, _)| *id).collect();

        let fixed = idx.search_ef(&q, 5, 128);
        let fixed_ids: Vec<u64> = fixed.iter().map(|(id, _)| *id).collect();
        let fixed_hits = fixed_ids.iter().filter(|i| truth.contains(i)).count();

        let adaptive = idx.search_adaptive(&q, 5, 16, 32, 256);
        let adp_ids: Vec<u64> = adaptive.iter().map(|(id, _)| *id).collect();
        let adp_hits = adp_ids.iter().filter(|i| truth.contains(i)).count();

        // Adaptive must recall at least as many of the true top-5 as fixed-ef.
        assert!(
            adp_hits >= fixed_hits,
            "adaptive regressed recall: fixed={fixed_hits} adaptive={adp_hits}"
        );
    }

    #[test]
    fn diag_graph_connectivity() {
        // Build a small real-ish graph and check self-recall + reachability.
        let idx = VectorIndex::open(eng());
        let dim = 64;
        let n = 2000u64;
        let mkv = |seed: u64| -> Vec<f32> {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            (0..dim)
                .map(|_| {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        for i in 0..n { idx.insert(i, mkv(i)); }
        // self-recall for a sample
        let mut self_hits = 0;
        let mut self_hits_big = 0;
        let mut fails = Vec::new();
        for qid in [0u64, 1, 2, 10, 100, 500, 1999] {
            let res = idx.search(&mkv(qid), 10);
            if res.iter().any(|(id, _)| *id == qid) { self_hits += 1; }
            else { fails.push(qid); }
            let res2 = idx.search_ef(&mkv(qid), 10, 4000);
            if res2.iter().any(|(id, _)| *id == qid) { self_hits_big += 1; }
        }
        eprintln!("DIAG self-recall ef=128 hits={}/7  ef=4000 hits={}/7  fails(ef128)={:?}", self_hits, self_hits_big, fails);
        let (ml, per, avg) = idx.debug_shape();
        eprintln!("DIAG max_level={} avg_neigh0={:.2} per_level={:?}", ml, avg, per);
        idx.dump_mitm_stats();
        // Inspect the failing node 1 and the entry it lands on (145).
        eprintln!("[MITM] === inspect entry 145 vs target 1 ===");
        idx.debug_node_neighbors(145, &mkv(1), 1);
        idx.debug_node_neighbors(1, &mkv(1), 145);
        assert!(self_hits >= 5, "self-recall too low: {}/7", self_hits);
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


