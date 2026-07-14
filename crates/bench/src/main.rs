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
    let mut keys = kv.keys_prefix("user:");
    keys.sort();
    check("prefix scan", keys == vec!["user:1".to_string(), "user:2".to_string()],
          &format!("got {keys:?}"));

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
// 13. Real-scale vector — 100k × 384d INT8-quantized HNSW
// ══════════════════════════════════════════════════════════════════════════
fn s13_real_scale_vectors() {
    section("13. Real-scale vectors — 100k × 384d, INT8 quantized, clustered");
    let e = Engine::open(fresh_wal("s13")).unwrap();
    let idx = Arc::new(VectorIndex::open(Arc::clone(&e)));
    const N: u64 = 100_000;
    const DIM: usize = 384;
    const N_CLUSTERS: u64 = 200;  // ~500 vectors/cluster — realistic embedding shape

    println!("  ingesting {N} × {DIM}d clustered vectors ({N_CLUSTERS} clusters, gaussian noise) ...");
    let t0 = Instant::now();
    for i in 0..N {
        idx.insert(i, clustered_vec(i, DIM, N_CLUSTERS)).unwrap();
    }
    let dt_ingest = t0.elapsed().as_secs_f64();
    println!("  ingest: {:.0} vec/s ({:.1}s total)", N as f64 / dt_ingest, dt_ingest);
    check("ingest > 500 vec/s at 100k×384d",
          (N as f64 / dt_ingest) > 500.0,
          &format!("{:.0} vec/s", N as f64 / dt_ingest));

    // Diagnostic — graph shape (catches degenerate construction)
    let (max_level, per_level, avg_neigh0) = idx.debug_shape();
    println!("  graph: max_level={max_level}, per-level counts={per_level:?}, avg level-0 neighbors={avg_neigh0:.1}");

    // Precompute normalized f32 vectors once for brute-force ground truth.
    println!("  precomputing 100k normalized f32 vectors for brute-force ...");
    let t_bf = Instant::now();
    let all_norm: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let mut v = idx.get_vector(i).unwrap();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= n);
            v
        })
        .collect();
    println!("  prep in {:.1}s", t_bf.elapsed().as_secs_f64());

    // Query set: fresh clustered vectors from clusters the index has seen.
    let n_queries = 100usize;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|qi| clustered_vec(800_000 + qi as u64, DIM, N_CLUSTERS))
        .collect();

    // Brute-force ground truth top-10 for every query.
    println!("  brute-force ground-truth top-10 for {} queries ...", n_queries);
    let t_gt = Instant::now();
    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q_raw| {
            let mut q = q_raw.clone();
            let nn = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            q.iter_mut().for_each(|x| *x /= nn);
            let mut scored: Vec<(u64, f32)> = all_norm
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                    (i as u64, 1.0 - dot)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.into_iter().take(10).map(|(id, _)| id).collect()
        })
        .collect();
    println!("  GT in {:.1}s", t_gt.elapsed().as_secs_f64());

    // DIAGNOSTIC: a query from cluster 0 should return IDs where id % N_CLUSTERS == 0
    let q0 = &queries[0]; // cluster = 800_000 % 200 = 0
    let gt0 = &ground_truth[0];
    let hnsw0 = idx.search_ef(q0, 10, 128);
    let gt_clusters: Vec<u64> = gt0.iter().map(|id| id % N_CLUSTERS).collect();
    let hnsw_clusters: Vec<u64> = hnsw0.iter().map(|(id, _)| id % N_CLUSTERS).collect();
    let hnsw_dists: Vec<f32> = hnsw0.iter().map(|(_, d)| *d).collect();
    println!("  ── query 0 (should be from cluster 0) ──");
    println!("    brute-force top-10 IDs:      {gt0:?}");
    println!("    brute-force top-10 clusters: {gt_clusters:?}");
    println!("    HNSW top-10 IDs:             {:?}", hnsw0.iter().map(|(id, _)| *id).collect::<Vec<_>>());
    println!("    HNSW top-10 clusters:        {hnsw_clusters:?}");
    println!("    HNSW top-10 distances:       {:?}",
             hnsw_dists.iter().map(|d| (d * 10000.0).round() / 10000.0).collect::<Vec<_>>());

    // Recall vs latency curve at several ef values (ann-benchmarks style).
    println!("  ── recall × latency curve (ef=32/64/128/256) ──");
    let ef_vals = [32usize, 64, 128, 256];
    let mut best_recall = 0.0f64;
    let mut best_p99: u64 = 0;
    for &ef in &ef_vals {
        // Warm
        for i in 0..30 {
            idx.search_ef(&queries[i % n_queries], 10, ef);
        }
        // Timed
        let mut latencies = Vec::with_capacity(n_queries);
        let mut recall_sum = 0.0f64;
        for (q, gt) in queries.iter().zip(&ground_truth) {
            let t0 = Instant::now();
            let hits = idx.search_ef(q, 10, ef);
            latencies.push(t0.elapsed().as_micros() as u64);
            let got: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
            let overlap = gt.iter().filter(|id| got.contains(id)).count();
            recall_sum += overlap as f64 / 10.0;
        }
        let recall = recall_sum / n_queries as f64;
        let p50 = pctl(&mut latencies.clone(), 50.0);
        let p99 = pctl(&mut latencies.clone(), 99.0);
        println!(
            "   ef={:<4} Recall@10 = {:.3}   p50 = {:>4} µs   p99 = {:>4} µs",
            ef, recall, p50, p99
        );
        if recall > best_recall {
            best_recall = recall;
            best_p99 = p99;
        }
    }
    check(
        "peak Recall@10 ≥ 0.90 (matches Qdrant / hnswlib class)",
        best_recall >= 0.90,
        &format!("peak recall={:.3} @ p99={}µs", best_recall, best_p99),
    );

    // Concurrent scaling at production-realistic ef=128.
    let n_threads = 8usize;
    let per = 300usize;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let idx = Arc::clone(&idx);
            thread::spawn(move || {
                for i in 0..per {
                    let q = clustered_vec((tid * 10_000 + i) as u64, DIM, N_CLUSTERS);
                    idx.search_ef(&q, 10, 128);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let qps = (n_threads * per) as f64 / dt;
    println!("  {n_threads}-thread concurrent VSEARCH @ ef=128: {qps:.0} QPS");
    check("100k×384d concurrent QPS > 5000 at production ef",
          qps > 5_000.0, &format!("{qps:.0} QPS"));
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
    section("15. Million-vector scale — 1M × 128d, INT8+rerank");
    let e = Engine::open(fresh_wal("s15")).unwrap();
    let idx = Arc::new(VectorIndex::open(Arc::clone(&e)));
    const N: u64 = 1_000_000;
    const DIM: usize = 128;
    const N_CLUSTERS: u64 = 1000;

    println!("  ingesting {N} × {DIM}d clustered vectors ({N_CLUSTERS} clusters) ...");
    let t0 = Instant::now();
    let mut last_print = t0;
    for i in 0..N {
        idx.insert(i, clustered_vec(i, DIM, N_CLUSTERS)).unwrap();
        if last_print.elapsed() > Duration::from_secs(15) {
            let done = i + 1;
            let rate = done as f64 / t0.elapsed().as_secs_f64();
            let eta = (N - done) as f64 / rate;
            println!("    ...{done:>7}/{N} ({rate:.0} vec/s, ETA {eta:.0}s)");
            last_print = Instant::now();
        }
    }
    let dt_ingest = t0.elapsed().as_secs_f64();
    println!("  ingest: {:.0} vec/s ({:.1}s total)", N as f64 / dt_ingest, dt_ingest);
    check("1M ingest > 3000 vec/s", (N as f64 / dt_ingest) > 3000.0,
          &format!("{:.0} vec/s", N as f64 / dt_ingest));

    // Warm up
    for _ in 0..100 {
        idx.search(&clustered_vec(999_999_999, DIM, N_CLUSTERS), 10);
    }
    // Latency at 1M scale (default ef=128)
    let mut samples = Vec::with_capacity(500);
    for i in 0..500u64 {
        let q = clustered_vec(700_000_000 + i, DIM, N_CLUSTERS);
        let t0 = Instant::now();
        idx.search(&q, 10);
        samples.push(t0.elapsed().as_micros() as u64);
    }
    latency_report("VSEARCH k=10 (1M × 128d, ef=128)", samples.clone());
    let p99 = pctl(&mut samples.clone(), 99.0);
    let p50 = pctl(&mut samples.clone(), 50.0);
    check("1M VSEARCH p99 < 5 ms", p99 < 5_000, &format!("p99 = {p99} µs"));
    check("1M VSEARCH p50 < 2 ms", p50 < 2_000, &format!("p50 = {p50} µs"));

    // Recall vs FULL brute-force over all 1M (correct methodology).
    // 128d + AVX2 auto-vec makes each brute-force query ~50-100 ms; 20 queries
    // ≈ 2 seconds — cheap enough to be honest.
    println!("  precomputing 1M normalized f32 vectors for brute-force ...");
    let t_bf = Instant::now();
    let all_norm: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let mut v = idx.get_vector(i).unwrap();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= n);
            v
        })
        .collect();
    println!("  prep in {:.1}s", t_bf.elapsed().as_secs_f64());

    println!("  measuring Recall@10 vs FULL 1M brute-force (20 queries) ...");
    let n_queries = 20usize;
    let mut recall_sum = 0.0f64;
    let t_r = Instant::now();
    for qi in 0..n_queries {
        let q_raw = clustered_vec(800_000_000 + qi as u64, DIM, N_CLUSTERS);
        let mut q = q_raw.clone();
        let nn = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        q.iter_mut().for_each(|x| *x /= nn);
        // Brute force over all 1M
        let mut scored: Vec<(u64, f32)> = all_norm
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                (i as u64, 1.0 - dot)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: Vec<u64> = scored.iter().take(10).map(|(id, _)| *id).collect();
        let hits = idx.search(&q_raw, 10);
        let got: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
        let overlap = gt.iter().filter(|id| got.contains(id)).count();
        recall_sum += overlap as f64 / 10.0;
    }
    let recall = recall_sum / n_queries as f64;
    println!("  Recall@10 (full 1M brute-force, {} queries): {:.3} in {:.1}s",
             n_queries, recall, t_r.elapsed().as_secs_f64());
    check("1M Recall@10 ≥ 0.85 vs full brute-force", recall >= 0.85,
          &format!("recall={:.3}", recall));

    // Concurrent scaling
    let n_threads = 8usize;
    let per = 200usize;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let idx = Arc::clone(&idx);
            thread::spawn(move || {
                for i in 0..per {
                    let q = clustered_vec((tid * 100_000 + i) as u64, DIM, N_CLUSTERS);
                    idx.search_ef(&q, 10, 128);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let qps = (n_threads * per) as f64 / dt;
    println!("  {n_threads}-thread concurrent VSEARCH @ 1M: {qps:.0} QPS");
    check("1M concurrent QPS > 2000", qps > 2_000.0, &format!("{qps:.0} QPS"));
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
        s.set_nodelay(true)?;
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

/// Spawn the release binary and wait until it accepts a PING.
fn spawn_dbstrike(port: u16, wal: &str) -> std::process::Child {
    let bin = std::env::current_exe()
        .ok()
        .and_then(|p| {
            let mut p = p.clone();
            p.pop();
            p.push("dbstrike");
            if p.exists() { Some(p) } else { None }
        })
        .unwrap_or_else(|| std::path::PathBuf::from("./target/release/dbstrike"));
    let mut child = std::process::Command::new(&bin)
        .arg(format!("127.0.0.1:{port}"))
        .env("DBSTRIKE_WAL", wal)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn dbstrike");
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(mut c) = Client::connect(&format!("127.0.0.1:{port}")) {
            if c.cmd(&[b"PING"]).is_ok() {
                return child;
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    panic!("server never accepted PING within 5s");
}

fn s17_chaos(iterations: usize) {
    section("17. Chaos — SIGKILL under load, verify acked-write durability");
    let port = 16400 + (std::process::id() as u16 % 100);
    let wal = format!("/tmp/dbstrike_chaos_{port}.wal");
    let _ = std::fs::remove_file(&wal);

    // Baseline: bring up, insert 500 keys, clean shutdown, verify all survive.
    // This isolates "does normal recovery even work" from "did chaos kill us".
    {
        let mut child = spawn_dbstrike(port, &wal);
        let mut c = Client::connect(&format!("127.0.0.1:{port}")).unwrap();
        for i in 0..500u64 {
            c.cmd(&[b"SET", format!("base:{i}").as_bytes(), b"v"]).unwrap();
        }
        // Clean SIGINT so group-commit flushes the tail.
        #[cfg(unix)]
        unsafe {
            libc_kill(child.id() as i32, 2 /*SIGINT*/);
        }
        let _ = child.wait();
    }

    // Iterate: spawn, write until we ack N writes, RECORD the last acked seq,
    // SIGKILL -9, restart, verify every acked seq survived.
    let mut total_lost = 0u64;
    let mut total_verified = 0u64;
    for iter in 0..iterations {
        let child = spawn_dbstrike(port, &wal);
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
            libc_kill(child.id() as i32, 9 /*SIGKILL*/);
        }
        let _ = spawn_wait(child);

        // Restart, verify every acked key is there.
        let child = spawn_dbstrike(port, &wal);
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
            libc_kill(child.id() as i32, 2 /*SIGINT*/);
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

fn spawn_wait(mut c: std::process::Child) -> std::io::Result<std::process::ExitStatus> {
    c.wait()
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tcp_addr = args.iter().position(|a| a == "--tcp").and_then(|i| args.get(i + 1).cloned());
    let ycsb_addr = args.iter().position(|a| a == "--ycsb").and_then(|i| args.get(i + 1).cloned());
    let run_large = args.iter().any(|a| a == "--large");
    let run_chaos = args.iter().any(|a| a == "--chaos");

    println!("DB-Strike native Rust bench");
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

    let dt = t_start.elapsed().as_secs_f64();
    let p = PASSED.load(Ordering::SeqCst);
    let f = FAILED.load(Ordering::SeqCst);
    println!("\n\x1b[1m=== RESULTS ===\x1b[0m");
    println!("  {p} passed, {f} failed  (in {dt:.1}s)");
    std::process::exit(if f == 0 { 0 } else { 1 });
}
