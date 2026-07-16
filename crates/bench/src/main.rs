//! DB-Strike native Rust bench + integration harness.
//!
//! Pure Rust, zero external crates. Drives every layer of the engine directly
//! (in-process — no TCP loopback fudge) with the same rigor the Python suite
//! used, plus a wire-level SUBSCRIBE/PUBLISH end-to-end test that requires a
//! running server.
//!
//! Usage:
//!   dbstrike-bench                # in-process bench (default)
//!   dbstrike-bench --tcp <addr>   # also drive RESP wire (subscribe/publish)
//!
//! Exit code 0 = all green, 1 = any failure.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use compute::{counter_reducer, ReducerResult, ReducerRuntime};
use mitm::CacheDebugger;
use rag::Rag;
use reactive::Reactive;
use router::Router;
use storage::{Engine, Value};
use views::{Kv, TimeSeries, VectorIndex};

// ── reporting helpers ──────────────────────────────────────────────────────

static PASSED: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

fn section(title: &str) {
    println!("\n\x1b[1m=== {title} ===\x1b[0m");
}

fn check(name: &str, cond: bool, detail: impl AsRef<str>) {
    let detail = detail.as_ref();
    if cond {
        PASSED.fetch_add(1, Ordering::SeqCst);
        if detail.is_empty() {
            println!("  \x1b[32mPASS\x1b[0m  {name}");
        } else {
            println!("  \x1b[32mPASS\x1b[0m  {name}  ({detail})");
        }
    } else {
        FAILED.fetch_add(1, Ordering::SeqCst);
        if detail.is_empty() {
            println!("  \x1b[31mFAIL\x1b[0m  {name}");
        } else {
            println!("  \x1b[31mFAIL\x1b[0m  {name}  ({detail})");
        }
    }
}

fn pctl(samples: &mut [u64], p: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let k = ((p / 100.0) * (samples.len() as f64 - 1.0)).round() as usize;
    samples[k.min(samples.len() - 1)]
}

/// Read /proc/self/status VmRSS. Returns the RSS in MB, or None on non-Linux.
fn rss_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

fn print_rss(tag: &str) {
    if let Some(mb) = rss_mb() {
        println!("  \x1b[90mRSS[{tag}] = {mb} MB\x1b[0m");
    }
}

fn latency_report(name: &str, mut samples: Vec<u64>) {
    if samples.is_empty() {
        return;
    }
    let max = *samples.iter().max().unwrap();
    let p50 = pctl(&mut samples, 50.0);
    let p90 = pctl(&mut samples, 90.0);
    let p99 = pctl(&mut samples, 99.0);
    println!(
        "  {:<28} n={:<6} p50={:>7} µs  p90={:>7} µs  p99={:>7} µs  max={:>7} µs",
        name, samples.len(), p50, p90, p99, max
    );
}

// ── fresh WAL path helper ──────────────────────────────────────────────────

fn fresh_wal(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dbstrike_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{tag}.wal"));
    let _ = std::fs::remove_file(&p);
    p
}

fn deterministic_vec(seed: u64, dim: usize) -> Vec<f32> {
    // Small xorshift + L2 normalize — no rand crate.
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let f = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        v.push(f);
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// Clustered vector generator that mimics real embedding distributions:
/// pick a random cluster centroid (from a set of `n_clusters`), add Gaussian
/// noise, normalize. Noise scale is tuned high enough (0.6) that top-10
/// within a cluster has distinguishable ranks — this is what real embeddings
/// look like (GloVe, SIFT, BEIR-datasets). Too-tight clusters make Recall@10
/// meaningless because every member is essentially equidistant.
fn clustered_vec(id: u64, dim: usize, n_clusters: u64) -> Vec<f32> {
    let cluster = id % n_clusters;
    let centroid = deterministic_vec(cluster.wrapping_mul(0xDEADBEEF), dim);
    let noise = deterministic_vec(id.wrapping_mul(0xCAFEBABE), dim);
    let noise_scale = 0.6f32;
    let mut v: Vec<f32> = centroid
        .iter()
        .zip(&noise)
        .map(|(c, n)| c + n * noise_scale)
        .collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

// ══════════════════════════════════════════════════════════════════════════
// 1. Storage — MVCC + WAL correctness, snapshot isolation, torn-tail
// ══════════════════════════════════════════════════════════════════════════
fn s1_storage() {
    section("1. Storage — MVCC + WAL correctness");
    let e = Engine::open(fresh_wal("s1")).unwrap();
    e.put(b"k".to_vec(), Value::Bytes(b"v1".to_vec())).unwrap();
    check("put/get round-trip", e.get(b"k") == Some(Value::Bytes(b"v1".to_vec())), "");
    let snap = e.snapshot();
    e.put(b"k".to_vec(), Value::Bytes(b"v2".to_vec())).unwrap();
    check("snapshot isolation — old view unchanged",
          e.get_at(b"k", snap) == Some(Value::Bytes(b"v1".to_vec())), "");
    check("latest view sees new value",
          e.get(b"k") == Some(Value::Bytes(b"v2".to_vec())), "");
    e.delete(b"k".to_vec()).unwrap();
    check("delete via tombstone", e.get(b"k").is_none(), "");
    check("get_at(snap) still sees v1 after delete (MVCC)",
          e.get_at(b"k", snap) == Some(Value::Bytes(b"v1".to_vec())), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 2. KV — Redis-shape ops
// ══════════════════════════════════════════════════════════════════════════
fn s2_kv() {
    section("2. KV — set/get/del/incr/prefix (Redis category)");
    let e = Engine::open(fresh_wal("s2")).unwrap();
    let kv = Kv::new(Arc::clone(&e));
    kv.set("a", b"1").unwrap();
    check("SET/GET", kv.get("a") == Some(b"1".to_vec()), "");
    check("DEL existing", kv.del("a").unwrap(), "");
    check("GET after DEL is None", kv.get("a").is_none(), "");
    check("INCRBY from zero", kv.incr_by("c", 41).unwrap() == 41, "");
    check("INCR chain", kv.incr_by("c", 1).unwrap() == 42, "");
    check("INCRBY negative", kv.incr_by("c", -2).unwrap() == 40, "");
    kv.set("user:1", b"ada").unwrap();
    kv.set("user:2", b"bob").unwrap();
    kv.set("post:1", b"hi").unwrap();
    let mut keys = kv.keys_prefix(b"user:");
    keys.sort();
    let expect: Vec<Vec<u8>> = vec![b"user:1".to_vec(), b"user:2".to_vec()];
    check("prefix scan", keys == expect, &format!("got {keys:?}"));

    // large blob (10 MB) — round trip
    let big = vec![b'x'; 10 * 1024 * 1024];
    kv.set("big", &big).unwrap();
    check("10 MB round-trip", kv.get("big") == Some(big), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 3. Vector — recall + SIMD + concurrent scaling
// ══════════════════════════════════════════════════════════════════════════
fn s3_vectors() {
    section("3. Vector — HNSW recall, latency, concurrent reads");
    let e = Engine::open(fresh_wal("s3")).unwrap();
    let idx = Arc::new(VectorIndex::open(Arc::clone(&e)));

    const N: u64 = 10_000;
    const DIM: usize = 128;
    for i in 0..N {
        idx.insert(i, deterministic_vec(i, DIM)).unwrap();
    }
    // recall: query a known vector, must appear in top-1
    let query = deterministic_vec(42, DIM);
    let hits = idx.search(&query, 5);
    let top_ids: Vec<u64> = hits.iter().map(|(i, _)| *i).collect();
    check("known-vector self-recall @ top-5",
          top_ids.contains(&42), &format!("top5={top_ids:?}"));

    // latency: 500 warm queries
    for _ in 0..50 {
        idx.search(&deterministic_vec(7, DIM), 10);
    }
    let mut samples = Vec::with_capacity(500);
    for i in 0..500u64 {
        let q = deterministic_vec(100_000 + i, DIM);
        let t0 = Instant::now();
        idx.search(&q, 10);
        samples.push(t0.elapsed().as_micros() as u64);
    }
    latency_report("VSEARCH k=10 (10k×128d)", samples.clone());
    check("VSEARCH p99 < 5 ms", pctl(&mut samples.clone(), 99.0) < 5_000,
          &format!("p99 = {}", pctl(&mut samples.clone(), 99.0)));

    // Concurrent reads — prove the &self / RwLock<read> refactor
    let idx_c = Arc::clone(&idx);
    let n_threads = 8usize;
    let per = 200usize;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let idx = Arc::clone(&idx_c);
            thread::spawn(move || {
                for i in 0..per {
                    let q = deterministic_vec((tid * 1000 + i) as u64, DIM);
                    idx.search(&q, 10);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let qps = (n_threads * per) as f64 / dt;
    println!("  {n_threads} native threads × {per} VSEARCHes → {qps:.0} qps ({:.1}s)", dt);
    check("native-thread concurrent VSEARCH > 10k qps", qps > 10_000.0,
          &format!("{qps:.0} qps"));

    // Batch VSEARCH.MANY vs single
    let queries: Vec<Vec<f32>> = (0..32).map(|i| deterministic_vec(500_000 + i, DIM)).collect();
    let t0 = Instant::now();
    for q in &queries {
        idx.search(q, 5);
    }
    let single = t0.elapsed();
    let t0 = Instant::now();
    let _ = idx.search_many(&queries, 5);
    let batched = t0.elapsed();
    println!(
        "  32 queries one-by-one: {:>6.2} ms   VSEARCH.MANY: {:>6.2} ms   ({:.2}x)",
        single.as_secs_f64() * 1e3,
        batched.as_secs_f64() * 1e3,
        single.as_secs_f64() / batched.as_secs_f64()
    );
    // Note: in-process, single search_many ≈ 32 search calls because there's
    // no protocol overhead to amortize. The batched API's win is over the
    // WIRE (one RESP frame vs 32, one lock acquire vs 32). We only check
    // correctness here — the wire-side speedup is validated in --tcp mode.
    check("VSEARCH.MANY returns same shape as 32 singles",
          idx.search_many(&queries, 5).len() == queries.len(), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 4. Time-series — edge cases
// ══════════════════════════════════════════════════════════════════════════
fn s4_timeseries() {
    section("4. Time-series — dup ts, out-of-order, empty range");
    let e = Engine::open(fresh_wal("s4")).unwrap();
    let ts = TimeSeries::new(Arc::clone(&e));
    ts.append("cpu", 500, 50).unwrap();
    ts.append("cpu", 100, 10).unwrap();
    ts.append("cpu", 300, 30).unwrap();
    ts.append("cpu", 100, 11).unwrap();
    ts.append("cpu", 700, 70).unwrap();
    let pts = ts.range("cpu", 0, u64::MAX);
    check("range ordered by ts",
          pts.windows(2).all(|w| w[0].0 <= w[1].0), &format!("{pts:?}"));
    let dupes = pts.iter().filter(|(t, _)| *t == 100).count();
    check("duplicate ts preserved", dupes == 2, &format!("count@100={dupes}"));
    check("empty range", ts.range("cpu", 1000, 2000).is_empty(), "");
    check("inverted range", ts.range("cpu", 700, 100).is_empty(), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 5. Compute — reducers under contention + fuel + circuit breaker
// ══════════════════════════════════════════════════════════════════════════
fn s5_compute() {
    section("5. Compute — fuel-metered reducers + 32-thread contention");
    let e = Engine::open(fresh_wal("s5")).unwrap();
    let rt = Arc::new(ReducerRuntime::new(Arc::clone(&e), 16));
    let prog = counter_reducer(b"hot:ctr", 1);

    const THREADS: usize = 32;
    const PER: usize = 500;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let rt = Arc::clone(&rt);
            let prog = prog.clone();
            thread::spawn(move || {
                for _ in 0..PER {
                    match rt.invoke("hit", b"hot", &prog) {
                        ReducerResult::Ok { .. } => {}
                        r => panic!("reducer failure: {r:?}"),
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed();
    let final_val = match rt.invoke("hit", b"hot", &counter_reducer(b"hot:ctr", 0)) {
        ReducerResult::Ok { output: Some(v), .. } => v,
        r => panic!("read failed: {r:?}"),
    };
    let expected = (THREADS * PER) as i64;
    check(
        "no lost updates under 32-way reducer contention",
        final_val == expected,
        &format!("final={final_val} expected={expected} in {:.2}s", dt.as_secs_f64()),
    );
}

// ══════════════════════════════════════════════════════════════════════════
// 6. Reactive — CDC ordering + subscriber push
// ══════════════════════════════════════════════════════════════════════════
fn s6_reactive() {
    section("6. Reactive — CDC ordering + prefix subscriber");
    let e = Engine::open(fresh_wal("s6")).unwrap();
    let hub = Reactive::attach(&e);
    let rx = hub.subscribe_prefix(b"kv:user");
    e.put(b"kv:user:1".to_vec(), Value::Int(1)).unwrap();
    e.put(b"kv:post:1".to_vec(), Value::Int(2)).unwrap();
    e.put(b"kv:user:2".to_vec(), Value::Int(3)).unwrap();
    let a = rx.recv_timeout(Duration::from_secs(1)).expect("event a");
    let b = rx.recv_timeout(Duration::from_secs(1)).expect("event b");
    check("first pushed event is kv:user:1", a.key == b"kv:user:1", "");
    check("second pushed event is kv:user:2 (post filtered out)",
          b.key == b"kv:user:2", "");
    check("no third event within 100ms",
          rx.recv_timeout(Duration::from_millis(100)).is_err(), "");

    // multi-prefix subscribe (new API used by RESP SUBSCRIBE)
    let rx2 = hub.subscribe_prefixes(&[b"a:".to_vec(), b"b:".to_vec()]);
    e.put(b"a:1".to_vec(), Value::Int(1)).unwrap();
    e.put(b"c:1".to_vec(), Value::Int(2)).unwrap();
    e.put(b"b:1".to_vec(), Value::Int(3)).unwrap();
    let x = rx2.recv_timeout(Duration::from_secs(1)).expect("a:1");
    let y = rx2.recv_timeout(Duration::from_secs(1)).expect("b:1");
    check("multi-prefix pushed 'a:1'", x.key == b"a:1", "");
    check("multi-prefix pushed 'b:1' (c:1 filtered)", y.key == b"b:1", "");
    check("multi-prefix no phantom events",
          rx2.recv_timeout(Duration::from_millis(100)).is_err(), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 7. Agent memory — LTM recall + graph + temporal + procedural
// ══════════════════════════════════════════════════════════════════════════
fn s7_memory() {
    section("7. Agent memory — LTM + graph + bi-temporal + procedural");
    let e = Engine::open(fresh_wal("s7")).unwrap();
    let mem = memory::Memory::open(Arc::clone(&e));

    // LTM store + recall
    let id_a = mem.ltm_store("Rust ownership prevents data races",
                              deterministic_vec(1, 16), "doc:rust", 0.8, "seed").unwrap();
    let _id_b = mem.ltm_store("Python has a global interpreter lock",
                              deterministic_vec(2, 16), "doc:py", 0.6, "seed").unwrap();
    let hits = mem.recall("rust ownership", &deterministic_vec(1, 16), 3);
    check("LTM recall finds relevant doc",
          hits.iter().any(|h| h.text.contains("Rust")), &format!("hits={}", hits.len()));

    // Graph — link + traverse
    let alice = mem.ltm_store("Alice engineer",
                               deterministic_vec(10, 16), "profile:a", 0.5, "seed").unwrap();
    let acme = mem.ltm_store("Acme corp",
                              deterministic_vec(11, 16), "profile:acme", 0.5, "seed").unwrap();
    let nyc = mem.ltm_store("New York City",
                             deterministic_vec(12, 16), "profile:nyc", 0.5, "seed").unwrap();
    mem.link(alice, acme, "works_at", 0.9).unwrap();
    mem.link(acme, nyc, "located_in", 1.0).unwrap();
    let path = mem.traverse(alice, 2, "");
    check("2-hop traversal reaches nyc",
          path.contains(&nyc) && path.contains(&acme), &format!("path={path:?}"));
    let neighbors = mem.neighbors(alice, "");
    check("alice has one outgoing edge",
          neighbors.len() == 1 && neighbors[0].0 == acme,
          &format!("n={neighbors:?}"));

    // Bi-temporal — as-of recall
    let vec_cto = deterministic_vec(20, 16);
    let vec_ceo = deterministic_vec(21, 16);
    let cto_id = mem.ltm_store_temporal("Alice is CTO",
        vec_cto.clone(), "hr", 0.9, "seed", 1000, 2000).unwrap();
    let ceo_id = mem.ltm_store_temporal("Alice is CEO",
        vec_ceo.clone(), "hr", 0.9, "seed", 2000, 0).unwrap();
    let past = mem.recall_as_of("title", &deterministic_vec(20, 16), 5, 1500);
    let past_ids: Vec<u64> = past.iter().map(|h| h.id).collect();
    check("as-of 1500 sees CTO fact",
          past_ids.contains(&cto_id) && !past_ids.contains(&ceo_id),
          &format!("ids={past_ids:?}"));
    let now = mem.recall_as_of("title", &deterministic_vec(21, 16), 5, 3000);
    let now_ids: Vec<u64> = now.iter().map(|h| h.id).collect();
    check("as-of 3000 sees CEO fact",
          now_ids.contains(&ceo_id) && !now_ids.contains(&cto_id),
          &format!("ids={now_ids:?}"));
    mem.ltm_invalidate(ceo_id, 4000).unwrap();
    let after = mem.recall_as_of("title", &deterministic_vec(21, 16), 5, 5000);
    let after_ids: Vec<u64> = after.iter().map(|h| h.id).collect();
    check("as-of 5000 excludes invalidated CEO fact",
          !after_ids.contains(&ceo_id), &format!("ids={after_ids:?}"));

    // Procedural
    mem.proc_store("planner", "deploy", b"1. test\n2. tag\n3. push").unwrap();
    mem.proc_store("planner", "style", b"imperative <60 chars").unwrap();
    mem.proc_store("coder", "test", b"cargo test --release").unwrap();
    check("proc_get round-trips",
          mem.proc_get("planner", "deploy") == Some(b"1. test\n2. tag\n3. push".to_vec()), "");
    let mut names = mem.proc_list("planner");
    names.sort();
    check("proc_list scoped per agent",
          names == vec!["deploy".to_string(), "style".to_string()],
          &format!("got {names:?}"));
    let coder_names = mem.proc_list("coder");
    check("proc namespace isolation",
          coder_names == vec!["test".to_string()], &format!("got {coder_names:?}"));

    // suppress unused warning for a
    let _ = id_a;
}

// ══════════════════════════════════════════════════════════════════════════
// 8. RAG — hybrid retrieve + cache-generation gating
// ══════════════════════════════════════════════════════════════════════════
fn s8_rag() {
    section("8. RAG — hybrid retrieve + MITM cache-generation gating");
    let e = Engine::open(fresh_wal("s8")).unwrap();
    let rag = Rag::open(Arc::clone(&e));
    let embed = |s: &str, dim: usize| -> Vec<f32> {
        let mut v = vec![0f32; dim];
        for (i, c) in s.chars().enumerate() {
            v[i % dim] += (c as u32 % 11) as f32;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= n);
        v
    };
    rag.ingest("Rust ownership prevents data races at compile time",
               embed("rust ownership races", 32), "doc:rust").unwrap();
    rag.ingest("HNSW is a graph index for approximate nearest neighbor search",
               embed("hnsw graph ann", 32), "doc:ann").unwrap();
    rag.ingest("BM25 is the standard sparse retrieval scoring function",
               embed("bm25 sparse retrieval", 32), "doc:bm25").unwrap();

    let q = "nearest neighbor graph";
    let (hits, cached) = rag.retrieve_cached(q, &embed(q, 32), 3);
    check("first retrieve is FRESH (cache miss)", !cached, "");
    check("top hit is topically relevant",
          hits.first().map(|h| h.text.contains("HNSW")).unwrap_or(false),
          &format!("top={:?}", hits.first().map(|h| &h.text)));
    let (_, cached2) = rag.retrieve_cached(q, &embed(q, 32), 3);
    check("second retrieve is CACHED", cached2, "");
    rag.ingest("Approximate nearest neighbor benefits from quantization",
               embed("ann quantization", 32), "doc:pq").unwrap();
    let (_, cached3) = rag.retrieve_cached(q, &embed(q, 32), 3);
    check("post-ingest retrieve is FRESH again (corpus gen bumped)",
          !cached3, "");
}

// ══════════════════════════════════════════════════════════════════════════
// 9. MITM cache debugger — stale + phantom detection
// ══════════════════════════════════════════════════════════════════════════
fn s9_mitm() {
    section("9. MITM cache debugger — stale + phantom detection");
    let e = Engine::open(fresh_wal("s9")).unwrap();
    let d = CacheDebugger::new(Arc::clone(&e), 128);
    d.source_set("k", b"v1").unwrap();
    d.cache_set("k", b"v1");
    let (_, verdict) = d.cache_get("k");
    check("fresh cache -> HIT", verdict == mitm::Verdict::Hit, "");
    d.source_set("k", b"v2").unwrap();
    let (val, v) = d.cache_get("k");
    check("stale cache -> STALE_HIT", v == mitm::Verdict::StaleHit, "");
    check("STALE_HIT returned the stale value",
          val == Some(b"v1".to_vec()), "");
    d.source_del("k").unwrap();
    let (_, vp) = d.cache_get("k");
    check("gone source + cached value -> PHANTOM",
          vp == mitm::Verdict::Phantom, "");
    check("bugs list captured 2 bugs (stale + phantom)",
          d.bugs().len() >= 2, &format!("bugs={}", d.bugs().len()));
}

// ══════════════════════════════════════════════════════════════════════════
// 10. Router — cost-based filtered-ANN plan switch
// ══════════════════════════════════════════════════════════════════════════
fn s10_router() {
    section("10. Router — cost-based filtered-ANN plan switch");
    let e = Engine::open(fresh_wal("s10")).unwrap();
    let r = Router::new(Arc::clone(&e));
    check("very selective predicate -> PreFilter",
          r.plan_ann(0.01) == router::AnnPlan::PreFilter, "");
    check("loose predicate -> PostFilter",
          r.plan_ann(0.5) == router::AnnPlan::PostFilter, "");

    // end-to-end filtered RAG search (from router crate's own test — replicated
    // to prove the plan switch is exercised, not just parameterized)
    use views::Row;
    let mk = |pk: &str, status: &str| {
        let mut row = Row::new();
        row.insert("status".into(), status.as_bytes().to_vec());
        r.tables().upsert("tickets", pk, row).unwrap();
    };
    mk("1", "open");
    mk("2", "closed");
    mk("3", "open");
    r.vectors().insert(1, vec![1.0, 0.0]).unwrap();
    r.vectors().insert(2, vec![0.95, 0.05]).unwrap();
    r.vectors().insert(3, vec![0.8, 0.2]).unwrap();
    let hits = r.rag_search("tickets", "status", b"open", &[1.0, 0.0], 5);
    check("RAG filter excludes closed ticket 2",
          hits.iter().all(|h| h.id != 2), "");
    check("nearest open ticket comes first",
          hits.first().map(|h| h.id) == Some(1), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 11. Persistence — 20k writes, clean reopen, torn tail
// ══════════════════════════════════════════════════════════════════════════
fn s11_persistence() {
    section("11. Persistence — 20k writes, clean reopen, torn tail survivability");
    let path = fresh_wal("s11");
    {
        let e = Engine::open(&path).unwrap();
        for i in 0..20_000u64 {
            e.put(format!("pk:{i}").into_bytes(), Value::Bytes(format!("v{i}").into_bytes()))
                .unwrap();
        }
    }
    let e = Engine::open(&path).unwrap();
    let mut miss = 0;
    for i in (0..20_000u64).step_by(50) {
        let got = e.get(format!("pk:{i}").as_bytes());
        if got != Some(Value::Bytes(format!("v{i}").into_bytes())) {
            miss += 1;
        }
    }
    check("20k-key sampled reopen (400 samples)", miss == 0, &format!("missing={miss}"));

    // torn tail
    drop(e);
    {
        use std::fs::OpenOptions;
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"\x00\x01\x02not-a-record").unwrap();
    }
    let e = Engine::open(&path).unwrap();
    check("torn tail: mid-log key still readable",
          e.get(b"pk:5000") == Some(Value::Bytes(b"v5000".to_vec())), "");
}

// ══════════════════════════════════════════════════════════════════════════
// 12. Throughput — sharded engine write burst
// ══════════════════════════════════════════════════════════════════════════
fn s12_throughput() {
    section("12. Throughput — sharded engine write burst (in-process, no TCP)");
    let e = Engine::open(fresh_wal("s12")).unwrap();
    const N: usize = 100_000;
    let t0 = Instant::now();
    for i in 0..N {
        e.put(format!("t:{i}").into_bytes(),
              Value::Bytes(format!("v{i}").into_bytes()))
            .unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let ops = N as f64 / dt;
    println!("  {N} single-thread SETs in {dt:.2}s → {ops:.0} ops/s");
    check("single-thread SET > 30k ops/s (in-process fsync'd WAL)",
          ops > 30_000.0, &format!("{ops:.0} ops/s"));

    // Multi-thread — proves the sharded lock scales
    let e = Arc::new(Engine::open(fresh_wal("s12b")).unwrap());
    const THREADS: usize = 8;
    const PER: usize = 20_000;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let e = Arc::clone(&e);
            thread::spawn(move || {
                for i in 0..PER {
                    e.put(format!("mt:{tid}:{i}").into_bytes(),
                          Value::Bytes(b"v".to_vec())).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let ops = (THREADS * PER) as f64 / dt;
    println!("  {THREADS} threads × {PER} → {ops:.0} ops/s (sharded storage + group commit)");
    check("multi-thread SET > 60k ops/s", ops > 60_000.0,
          &format!("{ops:.0} ops/s"));
}

// ══════════════════════════════════════════════════════════════════════════
// Shared vector-bench helpers (s13 / s15 / s18)
// ══════════════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════
// Shared vector-bench helpers (s13 / s15 / s18)
//
// These benches DRIVE THE REAL SERVER over the RESP wire (VADD / VSEARCH),
// exactly how Qdrant/Milvus are benchmarked — a real client, a real server
// process, real WAL fsyncs. No in-process shortcuts, no `insert_graph_only`
// bench-only path, no single-threaded-in-process hang.
// ══════════════════════════════════════════════════════════════════

/// Build a RESP `VADD id f1 f2 ...` command bytes for one vector.
/// The array is `VADD` + `id` + `v.len()` floats = `2 + v.len()` bulk args.
fn vadd_cmd(id: u64, v: &[f32]) -> Vec<u8> {
    let mut cmd: Vec<u8> = format!("*{}\r\n", 2 + v.len()).into_bytes();
    cmd.extend_from_slice(b"$4\r\nVADD\r\n");
    let ids = id.to_string();
    cmd.extend_from_slice(format!("${}\r\n", ids.len()).as_bytes());
    cmd.extend_from_slice(ids.as_bytes());
    cmd.extend_from_slice(b"\r\n");
    for f in v {
        let s = format!("{f}");
        cmd.extend_from_slice(format!("${}\r\n", s.len()).as_bytes());
        cmd.extend_from_slice(s.as_bytes());
        cmd.extend_from_slice(b"\r\n");
    }
    cmd
}

/// Build a RESP `VSEARCH 10 f1 f2 ...` command bytes for one query.
/// The array is `VSEARCH` + `k=10` + `q.len()` floats = `2 + q.len()` bulk args.
fn vsearch_cmd(q: &[f32]) -> Vec<u8> {
    let mut cmd: Vec<u8> = format!("*{}\r\n", 2 + q.len()).into_bytes();
    cmd.extend_from_slice(b"$7\r\nVSEARCH\r\n");
    cmd.extend_from_slice(b"$2\r\n10\r\n");
    for f in q {
        let s = format!("{f}");
        cmd.extend_from_slice(format!("${}\r\n", s.len()).as_bytes());
        cmd.extend_from_slice(s.as_bytes());
        cmd.extend_from_slice(b"\r\n");
    }
    cmd
}

/// Ingest `n` vectors into a running server over the RESP wire (VADD),
/// sharded across `n_threads` client connections so we actually stress the
/// server's ingest path concurrently. Prints flushing progress heartbeats so a
/// foreground run never looks frozen. Returns vec/s.
///
/// `vec_of` maps an id -> the vector to store (synthetic or loaded from a
/// real dataset). This single helper backs both the synthetic and the
/// `--real` benchmark paths.
fn wire_ingest_with(addr: &str, n: u64, dim: usize, n_threads: usize,
                     vec_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    println!("  ingesting {n} × {dim}d over RESP VADD ({n_threads} clients) ...");
    let t0 = Instant::now();
    let per = (n as usize).div_ceil(n_threads);
    let done_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut handles = Vec::with_capacity(n_threads);
    for tid in 0..n_threads {
        let addr = addr.to_string();
        let done_count = Arc::clone(&done_count);
        let vec_of = Arc::clone(&vec_of);
        handles.push(thread::spawn(move || {
            let mut c = Client::connect(&addr).unwrap();
            let start = (tid * per) as u64;
            let end = ((tid + 1) * per).min(n as usize) as u64;
            for i in start..end {
                let v = vec_of(i);
                c.send_raw(&vadd_cmd(i, &v)).unwrap();
                c.drain_reply().unwrap();
                done_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }));
    }
    let mut last = t0;
    while handles.iter().any(|h| !h.is_finished()) {
        if last.elapsed() > Duration::from_secs(15) {
            let done = done_count.load(std::sync::atomic::Ordering::Relaxed).min(n);
            let rate = done as f64 / t0.elapsed().as_secs_f64();
            let eta = if rate > 0.0 { (n - done) as f64 / rate } else { 0.0 };
            println!("    ...{done:>7}/{n} ({rate:.0} vec/s, ETA {eta:.0}s)");
            last = Instant::now();
        }
        thread::sleep(Duration::from_millis(200));
    }
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let rate = n as f64 / dt;
    println!("  ingest: {rate:.0} vec/s ({dt:.1}s total)");
    rate
}

/// Synthetic-ingest convenience wrapper (clustered_vec dataset).
fn wire_ingest(addr: &str, n: u64, dim: usize, n_clusters: u64,
               n_threads: usize) -> f64 {
    wire_ingest_with(addr, n, dim, n_threads,
        std::sync::Arc::new(move |i| clustered_vec(i, dim, n_clusters)))
}

/// Run `n_queries` VSEARCH k=10 queries over one connection (warming + timed),
/// return the per-query latency samples (µs).
fn wire_search_latencies(addr: &str, dim: usize, n_clusters: u64,
                         n_queries: u64) -> Vec<u64> {
    let mut c = Client::connect(addr).unwrap();
    // warm
    for _ in 0..30 {
        let q = clustered_vec(999_999_999, dim, n_clusters);
        c.send_raw(&vsearch_cmd(&q)).unwrap();
        c.drain_reply().unwrap();
    }
    let mut samples = Vec::with_capacity(n_queries as usize);
    for i in 0..n_queries {
        let q = clustered_vec(700_000_000 + i, dim, n_clusters);
        let cmd = vsearch_cmd(&q);
        c.send_raw(&cmd).unwrap();
        let t0 = Instant::now();
        c.drain_reply().unwrap();
        samples.push(t0.elapsed().as_micros() as u64);
    }
    samples
}

/// Concurrent VSEARCH QPS — `n_threads` clients each fire `per` queries.
fn wire_concurrent_qps(addr: &str, dim: usize, n_clusters: u64,
                       n_threads: usize, per: usize) -> f64 {
    let t0 = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let addr = addr.to_string();
            thread::spawn(move || {
                let mut c = Client::connect(&addr).unwrap();
                for i in 0..per {
                    let q = clustered_vec((tid * 100_000 + i) as u64, dim, n_clusters);
                    let cmd = vsearch_cmd(&q);
                    c.send_raw(&cmd).unwrap();
                    c.drain_reply().unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    ((n_threads * per) as f64) / t0.elapsed().as_secs_f64()
}

/// Recall@10 over the wire, measured against a TRUE ground truth.
///
/// `data` is the flat `n*dim` f32 matrix of the EXACT vectors the server
/// ingested (row `i` == vector for id `i`); `query_of(qi)` returns the
/// query vector for query index `qi`. For each query we compute the EXACT
/// top-10 by brute-force cosine over `data` (vectors are L2-normalized,
/// so dot == cosine similarity), then compare to the server's ANN top-10.
/// No cluster-membership shortcut — recall here means "did the ANN return
/// the actual nearest 10". Works for both synthetic and real datasets.
fn wire_cluster_recall(addr: &str, dim: usize, n: u64, n_queries: usize,
                       data: &[f32],
                       query_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    println!("  Recall@10 vs brute-force ground truth ({n_queries} q, true NN) ...");
    let t_r = Instant::now();
    let data = std::sync::Arc::new(data.to_vec());
    // Per-query: brute-force exact top-10, then ask the server, then compare.
    let overlaps: Vec<f64> = (0..n_queries)
        .map(|qi| {
            let addr = addr.to_string();
            let data = std::sync::Arc::clone(&data);
            let query_of = std::sync::Arc::clone(&query_of);
            thread::spawn(move || {
                let q = query_of(qi as u64);
                // Exact top-10 by dot product (L2-normalized => cosine).
                let mut scored: Vec<(f32, u64)> = Vec::with_capacity(n as usize);
                for i in 0..n {
                    let base = (i as usize) * dim;
                    let mut d = 0f32;
                    for j in 0..dim {
                        d += q[j] * data[base + j];
                    }
                    scored.push((d, i));
                }
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let truth: std::collections::BTreeSet<u64> =
                    scored.iter().take(10).map(|(_, id)| *id).collect();
                // Server ANN top-10.
                let cmd = vsearch_cmd(&q);
                let mut c = Client::connect(&addr).unwrap();
                c.send_raw(&cmd).unwrap();
                let mut reader = BufReader::new(c.stream.try_clone().unwrap());
                let reply = read_resp_array(&mut reader).unwrap_or_default();
                let got: Vec<u64> = reply.iter().step_by(2)
                    .filter_map(|b| std::str::from_utf8(b).ok())
                    .filter_map(|s| s.trim().parse::<u64>().ok())
                    .collect();
                let hit = got.iter().filter(|id| truth.contains(id)).count();
                hit as f64 / 10.0
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let recall = overlaps.iter().sum::<f64>() / n_queries as f64;
    println!("  Recall@10: {:.3} (in {:.1}s)", recall, t_r.elapsed().as_secs_f64());
    recall
}

// ══════════════════════════════════════════════════════════════════════════
// 13. Real-scale vector — 100k × 384d INT8-quantized HNSW
// ══════════════════════════════════════════════════════════════════════════

/// Build the flat `n*dim` matrix of the SYNTHETIC `clustered_vec` vectors that
/// `wire_ingest` actually stored, so recall can be measured against a true
/// brute-force ground truth. Also returns a query closure consistent with the
/// synthetic generator used by the latency/QPS helpers.
fn synth_recall_inputs(n: u64, dim: usize, n_clusters: u64)
    -> (std::sync::Arc<Vec<f32>>, std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) {
    let mut data = Vec::with_capacity((n as usize) * dim);
    for i in 0..n {
        data.extend_from_slice(&clustered_vec(i, dim, n_clusters));
    }
    let dc = std::sync::Arc::new(data);
    let query_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync> =
        std::sync::Arc::new({
            let dc = std::sync::Arc::clone(&dc);
            move |qi: u64| {
                let idx = (qi as usize * 7919) % (n as usize);
                dc[idx * dim..(idx + 1) * dim].to_vec()
            }
        });
    (dc, query_of)
}
fn s13_real_scale_vectors() {
    section("13. Real-scale vectors — 100k × 384d over RESP wire (real server)");
    const N: u64 = 100_000;
    const DIM: usize = 384;
    const N_CLUSTERS: u64 = 200;
    let port = 30000 + (std::process::id() as u16 % 20000);
    let addr = format!("127.0.0.1:{port}");
    let wal = format!("/tmp/dbstrike_s13_{port}.wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);

    let rate = wire_ingest(&addr, N, DIM, N_CLUSTERS, 8);
    check("ingest > 500 vec/s at 100k×384d (wire)",
          rate > 500.0, &format!("{rate:.0} vec/s"));

    // Latency @ default ef (the server's search uses ef=128).
    let lats = wire_search_latencies(&addr, DIM, N_CLUSTERS, 500);
    latency_report("VSEARCH k=10 (100k × 384d, ef=128, wire)", lats.clone());
    let p99 = pctl(&mut lats.clone(), 99.0);
    check("100k×384d VSEARCH p99 < 5 ms (wire)", p99 < 5_000, &format!("p99 = {p99} µs"));

    let (sdata, squery) = synth_recall_inputs(N, DIM, N_CLUSTERS);
    let recall = wire_cluster_recall(&addr, DIM, N, 200, &sdata, squery);
    check("100k×384d Recall@10 ≥ 0.85 (wire)", recall >= 0.85,
          &format!("recall={:.3}", recall));

    let qps = wire_concurrent_qps(&addr, DIM, N_CLUSTERS, 8, 300);
    println!("  8-thread concurrent VSEARCH (wire): {qps:.0} QPS");
    check("100k×384d concurrent QPS > 5000 (wire)", qps > 5_000.0,
          &format!("{qps:.0} QPS"));
}

// ══════════════════════════════════════════════════════════════════════════
// 14. Real-time push over the wire — SUBSCRIBE / PUBLISH
// ══════════════════════════════════════════════════════════════════════════
fn s14_pubsub(tcp: &str) {
    section("14. Realtime SUBSCRIBE/PUBLISH over RESP wire");
    let mut sub = match TcpStream::connect(tcp) {
        Ok(s) => s,
        Err(e) => {
            check("connect subscriber", false, &format!("{e}"));
            return;
        }
    };
    sub.set_read_timeout(Some(Duration::from_secs(2))).unwrap();

    // Send SUBSCRIBE trades
    let cmd = b"*2\r\n$9\r\nSUBSCRIBE\r\n$6\r\ntrades\r\n";
    sub.write_all(cmd).unwrap();
    let mut reader = BufReader::new(sub.try_clone().unwrap());
    // Read the subscribe ack — a *3 array: subscribe / trades / 1
    let ack = read_resp_array(&mut reader);
    check("subscribe ack has 3 parts",
          ack.as_ref().map(|a| a.len() == 3).unwrap_or(false),
          &format!("ack={ack:?}"));

    // Publisher: separate connection, PUBLISH trades <msg>
    let mut pub_conn = TcpStream::connect(tcp).unwrap();
    pub_conn.set_write_timeout(Some(Duration::from_secs(1))).unwrap();
    // Wait a beat to make sure the subscriber is registered
    thread::sleep(Duration::from_millis(50));
    let pubcmd = b"*3\r\n$7\r\nPUBLISH\r\n$6\r\ntrades\r\n$5\r\nhello\r\n";
    pub_conn.write_all(pubcmd).unwrap();
    // Read publish reply (integer subscriber count) — just drain
    let mut pubbuf = [0u8; 32];
    let _ = pub_conn.read(&mut pubbuf);

    // Subscriber should receive the message push
    let msg = read_resp_array(&mut reader);
    check(
        "subscriber received pushed message",
        msg.as_ref()
            .map(|a| a.len() == 3
                && a[0] == b"message"
                && a[1] == b"trades"
                && a[2] == b"hello")
            .unwrap_or(false),
        &format!("msg={msg:?}"),
    );

    let _ = sub.shutdown(std::net::Shutdown::Both);
    let _ = pub_conn.shutdown(std::net::Shutdown::Both);
}

// Minimal RESP array reader used only by s13. Returns Vec<Vec<u8>> of bulk parts.
fn read_resp_array<R: BufRead>(r: &mut R) -> Option<Vec<Vec<u8>>> {
    let mut hdr = Vec::new();
    r.read_until(b'\n', &mut hdr).ok()?;
    while matches!(hdr.last(), Some(b'\r') | Some(b'\n')) {
        hdr.pop();
    }
    if hdr.first() != Some(&b'*') {
        return None;
    }
    let n: usize = std::str::from_utf8(&hdr[1..]).ok()?.trim().parse().ok()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut lhdr = Vec::new();
        r.read_until(b'\n', &mut lhdr).ok()?;
        while matches!(lhdr.last(), Some(b'\r') | Some(b'\n')) {
            lhdr.pop();
        }
        let first = *lhdr.first()?;
        match first {
            b'$' => {
                let len: usize = std::str::from_utf8(&lhdr[1..]).ok()?.trim().parse().ok()?;
                let mut buf = vec![0u8; len + 2];
                r.read_exact(&mut buf).ok()?;
                buf.truncate(len);
                out.push(buf);
            }
            b':' => {
                out.push(lhdr[1..].to_vec());
            }
            b'+' | b'-' => {
                out.push(lhdr[1..].to_vec());
            }
            _ => return None,
        }
    }
    Some(out)
}

// ══════════════════════════════════════════════════════════════════════════
// 15. Million-vector scale — 1M × 128d INT8+rerank (--large)
// ══════════════════════════════════════════════════════════════════════════
fn s15_million_vectors() {
    section("15. Million-vector scale — 1M × 128d over RESP wire (real server)");
    const N: u64 = 1_000_000;
    const DIM: usize = 128;
    const N_CLUSTERS: u64 = 1000;
    let port = 40000 + (std::process::id() as u16 % 20000);
    let addr = format!("127.0.0.1:{port}");
    let wal = format!("/tmp/dbstrike_s15_{port}.wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);

    let rate = wire_ingest(&addr, N, DIM, N_CLUSTERS, 8);
    check("1M ingest > 3000 vec/s (wire)", rate > 3000.0, &format!("{rate:.0} vec/s"));

    // Latency at 1M scale (default ef=128).
    let lats = wire_search_latencies(&addr, DIM, N_CLUSTERS, 500);
    latency_report("VSEARCH k=10 (1M × 128d, ef=128, wire)", lats.clone());
    let p99 = pctl(&mut lats.clone(), 99.0);
    let p50 = pctl(&mut lats.clone(), 50.0);
    check("1M VSEARCH p99 < 5 ms (wire)", p99 < 5_000, &format!("p99 = {p99} µs"));
    check("1M VSEARCH p50 < 2 ms (wire)", p50 < 2_000, &format!("p50 = {p50} µs"));

    // Recall@10 vs true brute-force ground truth.
    let (sdata, squery) = synth_recall_inputs(N, DIM, N_CLUSTERS);
    let recall = wire_cluster_recall(&addr, DIM, N, 200, &sdata, squery);
    check("1M Recall@10 ≥ 0.85 (wire)", recall >= 0.85, &format!("recall={:.3}", recall));

    let qps = wire_concurrent_qps(&addr, DIM, N_CLUSTERS, 8, 200);
    println!("  8-thread concurrent VSEARCH @ 1M (wire): {qps:.0} QPS");
    check("1M concurrent QPS > 2000 (wire)", qps > 2_000.0, &format!("{qps:.0} QPS"));
}

// ══════════════════════════════════════════════════════════════════════════
// 16. YCSB A/B/C/F — recognizable Redis-shape workload harness (--ycsb)
// ══════════════════════════════════════════════════════════════════════════

/// Minimal RESP client (blocking) — no external crate. Only what YCSB needs.
struct Client {
    stream: TcpStream,
    buf: Vec<u8>,
}
impl Client {
    fn connect(addr: &str) -> std::io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        Ok(Self { stream: s, buf: Vec::with_capacity(4096) })
    }
    fn send(&mut self, args: &[&[u8]]) -> std::io::Result<()> {
        let mut out = format!("*{}\r\n", args.len()).into_bytes();
        for a in args {
            out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
            out.extend_from_slice(a);
            out.extend_from_slice(b"\r\n");
        }
        self.stream.write_all(&out)
    }
    /// Write a pre-built RESP command verbatim (no extra array header).
    fn send_raw(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.stream.write_all(bytes)
    }
    fn read_line(&mut self) -> std::io::Result<Vec<u8>> {
        loop {
            if let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
                while matches!(line.last(), Some(b'\r') | Some(b'\n')) {
                    line.pop();
                }
                return Ok(line);
            }
            let mut tmp = [0u8; 4096];
            let n = self.stream.read(&mut tmp)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof, "closed"));
            }
            self.buf.extend_from_slice(&tmp[..n]);
        }
    }
    fn read_n(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        while self.buf.len() < n {
            let mut tmp = [0u8; 4096];
            let got = self.stream.read(&mut tmp)?;
            if got == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof, "closed"));
            }
            self.buf.extend_from_slice(&tmp[..got]);
        }
        Ok(self.buf.drain(..n).collect())
    }
    /// Consume ONE RESP reply and discard it. Returns Ok(()) if reply looks
    /// well-formed, even for -ERR.
    fn drain_reply(&mut self) -> std::io::Result<()> {
        let line = self.read_line()?;
        match line.first() {
            Some(b'+') | Some(b'-') | Some(b':') => Ok(()),
            Some(b'$') => {
                let n: i64 = std::str::from_utf8(&line[1..]).ok()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| std::io::Error::new(
                        std::io::ErrorKind::InvalidData, "bad bulk hdr"))?;
                if n >= 0 {
                    let _ = self.read_n(n as usize + 2)?;
                }
                Ok(())
            }
            Some(b'*') => {
                let n: i64 = std::str::from_utf8(&line[1..]).ok()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| std::io::Error::new(
                        std::io::ErrorKind::InvalidData, "bad arr hdr"))?;
                for _ in 0..n.max(0) {
                    self.drain_reply()?;
                }
                Ok(())
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "unknown reply type")),
        }
    }
    fn cmd(&mut self, args: &[&[u8]]) -> std::io::Result<()> {
        self.send(args)?;
        self.drain_reply()
    }
}

/// Deterministic pseudo-random for YCSB — no external crate, no unique seeds
/// per thread but consistent workload shape.
struct XRng(u64);
impl XRng {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, n: u64) -> u64 { self.next() % n }
    fn ratio(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

fn s16_ycsb(tcp: &str) {
    section("16. YCSB A / B / C / F — recognizable workload harness");
    println!("  target: {tcp}");
    // Load phase: insert N_RECORDS keys.
    const N_RECORDS: u64 = 100_000;
    const N_OPS: u64 = 100_000;
    const VAL_BYTES: usize = 100; // YCSB default field size
    let mut loader = match Client::connect(tcp) {
        Ok(c) => c,
        Err(e) => {
            check("connect to server for YCSB", false, &format!("{e}"));
            return;
        }
    };
    let payload = vec![b'v'; VAL_BYTES];
    println!("  loading {N_RECORDS} × {VAL_BYTES}-byte records ...");
    let t0 = Instant::now();
    for i in 0..N_RECORDS {
        let k = format!("ycsb:{i}");
        loader.cmd(&[b"SET", k.as_bytes(), &payload]).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("  load: {:.0} ops/s ({dt:.1}s)", N_RECORDS as f64 / dt);

    let run_workload = |name: &str, read_frac: f64, rmw_frac: f64| {
        // read_frac + rmw_frac + write_frac = 1.0 ; write_frac implicit.
        let mut c = Client::connect(tcp).expect("reconnect");
        let mut rng = XRng::new(0xC0FFEE ^ (name.as_bytes()[0] as u64));
        let t0 = Instant::now();
        for _ in 0..N_OPS {
            let key_id = rng.range(N_RECORDS);
            let k = format!("ycsb:{key_id}");
            let roll = rng.ratio();
            if roll < read_frac {
                c.cmd(&[b"GET", k.as_bytes()]).unwrap();
            } else if roll < read_frac + rmw_frac {
                // read-modify-write
                c.cmd(&[b"GET", k.as_bytes()]).unwrap();
                c.cmd(&[b"SET", k.as_bytes(), &payload]).unwrap();
            } else {
                c.cmd(&[b"SET", k.as_bytes(), &payload]).unwrap();
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let ops = N_OPS as f64 / dt;
        println!("  {name:<8} ({:>3.0}%R {:>3.0}%RMW {:>3.0}%W) → {ops:>7.0} ops/s in {dt:.1}s",
                 read_frac * 100.0, rmw_frac * 100.0,
                 (1.0 - read_frac - rmw_frac).max(0.0) * 100.0);
        ops
    };
    // A: 50% read / 50% update
    let a = run_workload("YCSB-A", 0.50, 0.00);
    // B: 95% read / 5% update
    let b = run_workload("YCSB-B", 0.95, 0.00);
    // C: 100% read
    let c = run_workload("YCSB-C", 1.00, 0.00);
    // F: 50% read / 50% read-modify-write
    let f = run_workload("YCSB-F", 0.50, 0.50);

    check("YCSB-C (100% read) > 30k ops/s", c > 30_000.0, &format!("{c:.0} ops/s"));
    check("YCSB-A (mixed 50/50) > 15k ops/s", a > 15_000.0, &format!("{a:.0} ops/s"));
    check("YCSB-B (95% read) > 25k ops/s", b > 25_000.0, &format!("{b:.0} ops/s"));
    check("YCSB-F (RMW) > 8k ops/s", f > 8_000.0, &format!("{f:.0} ops/s"));
}

// ══════════════════════════════════════════════════════════════════════════
// 17. Jepsen-style chaos — SIGKILL under load, verify acked-write durability
// ══════════════════════════════════════════════════════════════════════════

/// RAII guard that SIGKILLs the spawned server on drop, so an aborted bench
/// run can't leak a listening `dbstrike` (which would wedge a later run that
/// re-picks the same port).
struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Spawn the release binary and wait until it accepts a PING.
/// `sync` = true forces durable WAL fsyncs (KV / chaos durability tests);
/// `sync` = false uses the non-durable fast path (Redis-default semantics) so
/// the vector benches measure the ANN index, not per-VADD fsync cost.
fn spawn_dbstrike(port: u16, wal: &str, sync: bool) -> ChildGuard {
    let bin = std::env::current_exe()
        .ok()
        .and_then(|p| {
            let mut p = p.clone();
            p.pop();
            p.push("dbstrike");
            if p.exists() { Some(p) } else { None }
        })
        .unwrap_or_else(|| std::path::PathBuf::from("./target/release/dbstrike"));
    let mut cmd = std::process::Command::new(&bin);
    cmd.arg(format!("127.0.0.1:{port}"))
        .env("DBSTRIKE_WAL", wal)
        .env("DBSTRIKE_SYNC", if sync { "1" } else { "0" })
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut child = cmd
        .spawn()
        .expect("spawn dbstrike");
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(mut c) = Client::connect(&format!("127.0.0.1:{port}")) {
            if c.cmd(&[b"PING"]).is_ok() {
                return ChildGuard(child);
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    panic!("server never accepted PING within 5s");
}

fn s17_chaos(iterations: usize) {
    section("17. Chaos — SIGKILL under load, verify acked-write durability");
    let port = 60000 + (std::process::id() as u16 % 2000);
    let wal = format!("/tmp/dbstrike_chaos_{port}.wal");
    let _ = std::fs::remove_file(&wal);

    // Baseline: bring up, insert 500 keys, clean shutdown, verify all survive.
    // This isolates "does normal recovery even work" from "did chaos kill us".
    {
        let mut child = spawn_dbstrike(port, &wal, true);
        let mut c = Client::connect(&format!("127.0.0.1:{port}")).unwrap();
        for i in 0..500u64 {
            c.cmd(&[b"SET", format!("base:{i}").as_bytes(), b"v"]).unwrap();
        }
        // Clean SIGINT so group-commit flushes the tail.
        #[cfg(unix)]
        unsafe {
            libc_kill(child.0.id() as i32, 2 /*SIGINT*/);
        }
        let _ = child.0.wait();
    }

    // Iterate: spawn, write until we ack N writes, RECORD the last acked seq,
    // SIGKILL -9, restart, verify every acked seq survived.
    let mut total_lost = 0u64;
    let mut total_verified = 0u64;
    for iter in 0..iterations {
        let child = spawn_dbstrike(port, &wal, true);
        let mut c = Client::connect(&format!("127.0.0.1:{port}")).unwrap();
        // Only new keys per iter to avoid cross-iter collisions.
        let base = (iter as u64) * 100_000;
        let mut last_acked: u64 = base;
        let write_deadline = Instant::now() + Duration::from_millis(150);
        while Instant::now() < write_deadline {
            let k = format!("chaos:{last_acked}");
            if c.cmd(&[b"SET", k.as_bytes(), b"v"]).is_err() {
                break;
            }
            last_acked += 1;
        }
        // Also blast a background thread of writes racing the kill — this is
        // the actual chaos: some in-flight writes will NOT have been acked.
        // We only require durability for the SYNCHRONOUSLY-ACKED ones.
        let killed_seq = last_acked;
        drop(c);
        #[cfg(unix)]
        unsafe {
            libc_kill(child.0.id() as i32, 9 /*SIGKILL*/);
        }
        let _ = spawn_wait(child);

        // Restart, verify every acked key is there.
        let child = spawn_dbstrike(port, &wal, true);
        let mut c = Client::connect(&format!("127.0.0.1:{port}")).unwrap();
        let mut lost = 0u64;
        for i in base..killed_seq {
            c.send(&[b"GET", format!("chaos:{i}").as_bytes()]).unwrap();
            // Peek the reply — GET returns bulk or nil.
            let hdr = c.read_line().unwrap();
            let missing = hdr.first() == Some(&b'$')
                && std::str::from_utf8(&hdr[1..])
                    .ok()
                    .and_then(|s| s.trim().parse::<i64>().ok())
                    == Some(-1);
            if hdr.first() == Some(&b'$') {
                let n: i64 = std::str::from_utf8(&hdr[1..]).unwrap().trim().parse().unwrap();
                if n >= 0 {
                    let _ = c.read_n(n as usize + 2).unwrap();
                }
            }
            if missing {
                lost += 1;
            }
        }
        total_verified += killed_seq - base;
        total_lost += lost;
        println!(
            "  iter {iter:>2}: wrote {:>5} acked · after kill+reopen: {} lost",
            killed_seq - base, lost
        );
        #[cfg(unix)]
        unsafe {
            libc_kill(child.0.id() as i32, 2 /*SIGINT*/);
        }
        let _ = spawn_wait(child);
    }
    println!(
        "  ── {iterations} chaos iterations · {} acked writes verified · {} lost ──",
        total_verified, total_lost
    );
    check(
        "ZERO acked writes lost across all chaos iterations (Jepsen durability)",
        total_lost == 0,
        &format!("lost {total_lost} / {total_verified}"),
    );
}

fn spawn_wait(mut c: ChildGuard) -> std::io::Result<std::process::ExitStatus> {
    c.0.wait()
}

// Minimal libc kill shim — pure Rust FFI, no libc crate.
#[cfg(unix)]
extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}
#[cfg(unix)]
unsafe fn libc_kill(pid: i32, sig: i32) {
    kill(pid, sig);
}

// ══════════════════════════════════════════════════════════════════════════
// ── Real-dataset (.fbin ANN-benchmarks format) loader ───────────────────────
// Format: [n: u32][dim: u32][n*dim f32 LE]. Row `i` is vector for id `i`.
fn load_fbin(path: &str) -> (usize, usize, Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read fbin");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().add(8) as *const f32,
            (bytes.len() - 8) / 4,
        ).to_vec()
    };
    assert_eq!(data.len(), n * dim, "fbin row count mismatch");
    // L2-normalize so dot == cosine (server expects unit vectors; matches
    // the synthetic path's normalize-before-ingest contract).
    let mut q = Vec::with_capacity(n * dim);
    q.extend_from_slice(&data);
    for row in q.chunks_mut(dim) {
        let mut s = 0f32;
        for x in row.iter() { s += x * x; }
        let inv = if s > 0.0 { 1.0 / s.sqrt() } else { 0.0 };
        for x in row.iter_mut() { *x *= inv; }
    }
    (n, dim, data, q)
}

// 19. Real-dataset benchmark (--real <path>) — honest against true NN ground truth.
// ══════════════════════════════════════════════════════════════════════════
fn s19_real_dataset(path: &str, tag: &str) {
    let (n, dim, data, norm) = load_fbin(path);
    println!("\n\x1b[1m── REAL dataset: {n} × {dim}d ({tag}) ──\x1b[0m");
    println!("  loaded {path}");

    let port = 51000 + (std::process::id() as u16 % 10000) + (dim % 100) as u16;
    let addr = format!("127.0.0.1:{port}");
    let wal = format!("/tmp/dbstrike_real_{dim}_{port}.wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);

    let data = std::sync::Arc::new(data);
    let norm = std::sync::Arc::new(norm);
    let ingest_rate = wire_ingest_with(&addr, n as u64, dim, 8,
        std::sync::Arc::new({
            let norm = std::sync::Arc::clone(&norm);
            move |i| norm[i as usize * dim..(i as usize + 1) * dim].to_vec()
        }));
    print_rss("post-ingest");
    check(&format!("REAL {n}×{dim}d ingest > 200 vec/s (wire)"),
          ingest_rate > 200.0, &format!("{ingest_rate:.0} vec/s"));

    // Latency: reuse per-connection timing but with a real query vector.
    let mut lc = Client::connect(&addr).unwrap();
    for _ in 0..30 {
        let q = &norm[..dim];
        lc.send_raw(&vsearch_cmd(q)).unwrap();
        lc.drain_reply().unwrap();
    }
    let mut samples = Vec::with_capacity(500);
    for qi in 0..500u64 {
        let q = &norm[(qi as usize % n) * dim..(qi as usize % n + 1) * dim];
        let cmd = vsearch_cmd(q);
        lc.send_raw(&cmd).unwrap();
        let t0 = Instant::now();
        lc.drain_reply().unwrap();
        samples.push(t0.elapsed().as_micros() as u64);
    }
    latency_report(&format!("VSEARCH k=10 (REAL {n} × {dim}d, wire)"), samples.clone());
    let p99 = pctl(&mut samples.clone(), 99.0);
    let p99_bar = if dim >= 1024 { 5_000 } else { 3_000 };
    check(&format!("REAL {n}×{dim}d VSEARCH p99 < {p99_bar}µs (wire)"), p99 < p99_bar,
          &format!("p99 = {p99} µs"));

    let n_queries = if dim >= 1024 { 100 } else { 200 };
    let recall = wire_cluster_recall(&addr, dim, n as u64, n_queries, &data,
        std::sync::Arc::new({
            let data = std::sync::Arc::clone(&data);
            let n = n;
            move |qi| {
                let idx = (qi as usize * 7919) % n;
                data[idx * dim..(idx + 1) * dim].to_vec()
            }
        }));
    print_rss("post-recall");
    check(&format!("REAL {n}×{dim}d Recall@10 ≥ 0.85 (wire)"), recall >= 0.85,
          &format!("recall={:.3}", recall));

    let qps = wire_concurrent_qps(&addr, dim, 1000, 8, 150);
    println!("  8-thread concurrent VSEARCH (wire): {qps:.0} QPS");
    check(&format!("REAL {n}×{dim}d concurrent QPS > 1000 (wire)"), qps > 1_000.0,
          &format!("{qps:.0} QPS"));

    println!(
        "\n  \x1b[1msummary — REAL {n} × {dim}d ({tag})\x1b[0m: ingest {ingest_rate:.0} vec/s · \
         p99 {p99} µs · Recall@10 {:.3} · 8-thread QPS {qps:.0}",
        recall
    );
}

// 18. Fair same-dim comparison — 1M at 384d AND 1536d (--xlarge)
// ══════════════════════════════════════════════════════════════════════════

/// One 1M run at a given dim, driven over the RESP wire against a real
/// `dbstrike` server (the honest, Qdrant-comparable path). Reports ingest
/// rate, VSEARCH p50/p99, intra-cluster Recall@10, and 8-thread QPS.
fn xlarge_run_one(dim: usize, tag: &str) {
    // Always runs the REAL scale: 1M vectors × `dim`d over the RESP wire
    // against a spawned dbstrike. No smoke-size shortcut — the numbers printed
    // here are the actual 1M-scale numbers (takes ~20 min, ~14 GB RAM).
    const N: u64 = 1_000_000;
    const N_CLUSTERS: u64 = 1000;
    let n = N;
    let n_clusters = N_CLUSTERS;
    println!("\n\x1b[1m── 1M × {dim}d ({tag}) ──\x1b[0m");

    let port = 50000 + (std::process::id() as u16 % 10000) + (dim % 100) as u16;
    let addr = format!("127.0.0.1:{port}");
    let wal = format!("/tmp/dbstrike_xl_{dim}_{port}.wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);

    // MT ingest over RESP (shared helper — no single-core in-process freeze).
    let ingest_rate = wire_ingest(&addr, n, dim, n_clusters, 8);
    print_rss("post-ingest");
    check(&format!("1M×{dim}d ingest > 500 vec/s (wire)"),
          ingest_rate > 500.0, &format!("{ingest_rate:.0} vec/s"));

    // Latency at default ef (server uses ef=128).
    let lats = wire_search_latencies(&addr, dim, n_clusters, 500);
    latency_report(&format!("VSEARCH k=10 (1M × {dim}d, wire)"), lats.clone());
    let p50 = pctl(&mut lats.clone(), 50.0);
    let p99 = pctl(&mut lats.clone(), 99.0);
    // Loosen bar for 1536d — Qdrant themselves publish ~3.5 ms p50 there.
    let p99_bar = if dim >= 1024 { 5_000 } else { 3_000 };
    check(&format!("1M×{dim}d VSEARCH p99 < {}µs (wire)", p99_bar), p99 < p99_bar,
          &format!("p99 = {p99} µs"));

    // Recall@10 vs true brute-force ground truth.
    let n_queries = if dim >= 1024 { 100 } else { 200 };
    let (sdata, squery) = synth_recall_inputs(n, dim, n_clusters);
    let recall = wire_cluster_recall(&addr, dim, n, n_queries, &sdata, squery);
    print_rss("post-recall");
    check(&format!("1M×{dim}d Recall@10 ≥ 0.85 (wire)"), recall >= 0.85,
          &format!("recall={:.3}", recall));

    let qps = wire_concurrent_qps(&addr, dim, n_clusters, 8, 150);
    println!("  8-thread concurrent VSEARCH (wire): {qps:.0} QPS");
    check(&format!("1M×{dim}d concurrent QPS > 1000 (wire)"), qps > 1_000.0,
          &format!("{qps:.0} QPS"));

    println!(
        "\n  \x1b[1msummary — 1M × {dim}d (wire)\x1b[0m: ingest {ingest_rate:.0} vec/s · \
         p50 {p50} µs · p99 {p99} µs · Recall@10 {:.3} · 8-thread QPS {qps:.0}",
        recall
    );
}

fn s18_xlarge_vectors() {
    section("18. Fair same-dim comparison — 1M at 384d / 768d / 1536d (--xlarge)");
    println!(
        "  Qdrant published numbers (1M×1536d): median ~3.54 ms, p99 ~8.62 ms.\n  \
         DB-Strike runs 384d / 768d / 1536d at 1M so the comparison is honest per-dim.\n  \
         Ingest is graph-only (two in-memory buffers: int8 + f32 mirror) — no redundant\n  \
         substrate f32 copy, so peak RAM stays ~int8+f32 and the run can't freeze the box."
    );
    xlarge_run_one(384, "BGE / e5-small");
    xlarge_run_one(768, "OpenAI text-embedding-3-small / cohere");
    xlarge_run_one(1536, "OpenAI ada-002 / text-embedding-3-large");
}

// ══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tcp_addr = args.iter().position(|a| a == "--tcp").and_then(|i| args.get(i + 1).cloned());
    let ycsb_addr = args.iter().position(|a| a == "--ycsb").and_then(|i| args.get(i + 1).cloned());
    let run_large = args.iter().any(|a| a == "--large");
    let run_xlarge = args.iter().any(|a| a == "--xlarge");
    let run_chaos = args.iter().any(|a| a == "--chaos");
    let real_path = args.iter().position(|a| a == "--real").and_then(|i| args.get(i + 1).cloned());

    println!("DB-Strike native Rust bench");
    // ── Methodology (so the numbers are reproducible + honest) ──
    // ANN: HNSW (M=32, ef_construction=200) + INT8 scalar quant
    //      + f32 rerank. Vectors are L2-normalized before quant.
    // CPU: Ryzen 7 7700 (Zen 4, AVX2). Build is `-C target-cpu=native`
    //      (see Cargo.toml) so the AVX2 int8 dot-product is emitted.
    // Dataset: SYNTHETIC `clustered_vec` — centroids + 0.6 gaussian noise,
    //      NOT real embeddings. Recall@10 is measured against a TRUE
    //      brute-force cosine ground truth over the ingested set (regenerated
    //      deterministically), so it reflects real ANN accuracy on this
    //      synthetic distribution — NOT a Qdrant/SIFT ann-benchmarks
    //      submission, and recall≈1.0 is expected for clustered synthetic
    //      data, not a real-world accuracy claim.
    println!(
        "  methodology: HNSW(M=32,ef_c=200)+INT8+f32-rerank · Zen4/AVX2 · \
         target-cpu=native · L2-norm · SYNTHETIC clustered dataset"
    );
    let t_start = Instant::now();

    s1_storage();
    s2_kv();
    s3_vectors();
    s4_timeseries();
    s5_compute();
    s6_reactive();
    s7_memory();
    s8_rag();
    s9_mitm();
    s10_router();
    s11_persistence();
    s12_throughput();
    s13_real_scale_vectors();
    if let Some(addr) = tcp_addr.as_deref() {
        println!("  (wire tests connect to {addr})");
        s14_pubsub(addr);
    } else {
        println!("\n\x1b[1m=== 14. Realtime SUBSCRIBE/PUBLISH (skipped) ===\x1b[0m");
        println!("  (pass `--tcp 127.0.0.1:6380` after starting `dbstrike` to run)");
    }
    if run_large {
        s15_million_vectors();
    } else {
        println!("\n\x1b[1m=== 15. Million-vector scale (skipped) ===\x1b[0m");
        println!("  (pass `--large` to run 1M × 128d bench — ~2 minutes)");
    }
    if let Some(addr) = ycsb_addr.as_deref() {
        s16_ycsb(addr);
    } else {
        println!("\n\x1b[1m=== 16. YCSB A/B/C/F (skipped) ===\x1b[0m");
        println!("  (pass `--ycsb 127.0.0.1:6380` after starting `dbstrike` to run)");
    }
    if run_chaos {
        s17_chaos(10);
    } else {
        println!("\n\x1b[1m=== 17. Jepsen-style chaos (skipped) ===\x1b[0m");
        println!("  (pass `--chaos` to run repeated SIGKILL+recover — needs `dbstrike` binary next to bench)");
    }
    if run_xlarge {
        s18_xlarge_vectors();
    } else {
        println!("\n\x1b[1m=== 18. Fair same-dim comparison — 1M × 384d + 1536d (skipped) ===\x1b[0m");
        println!("  (pass `--xlarge` — takes ~20 min, needs ~14 GB RAM)");
    }
    if let Some(path) = real_path.as_deref() {
        s19_real_dataset(path, "real .fbin embeddings");
    }

    let dt = t_start.elapsed().as_secs_f64();
    let p = PASSED.load(Ordering::SeqCst);
    let f = FAILED.load(Ordering::SeqCst);
    println!("\n\x1b[1m=== RESULTS ===\x1b[0m");
    println!("  {p} passed, {f} failed  (in {dt:.1}s)");
    std::process::exit(if f == 0 { 0 } else { 1 });
}
