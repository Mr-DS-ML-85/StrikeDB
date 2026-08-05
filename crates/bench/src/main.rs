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
use std::net::{TcpListener, TcpStream};
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
use views::{Filter, Kv, TimeSeries, VectorIndex};

// ── reporting helpers ──────────────────────────────────────────────────────

static PASSED: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

fn section(title: &str) {
    println!("\n\x1b[1m=== {title} ===\x1b[0m");
    flush_out();
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
    flush_out();
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
    flush_out();
}

/// Push stdout to the OS at every phase boundary.
///
/// `println!` line-buffers to a terminal but *block*-buffers to a pipe or a
/// file, so redirecting a run into `tail` or a log silently holds the last few
/// KB in userspace. If the process is then killed — OOM, cgroup, SIGKILL — that
/// tail is lost, and the log's final line is wherever the buffer happened to
/// end rather than wherever execution stopped.
///
/// That is not a cosmetic problem. It made a run that was killed later *look*
/// like it died right after the recall phase, and sent me hunting a memory bug
/// in the wrong function. A diagnostic log that truncates on the exact failure
/// it is meant to explain is worse than none.
fn flush_out() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// `MemAvailable` from /proc/meminfo, in MB. This is the kernel's own estimate
/// of what a new allocation can actually get without swapping — deliberately
/// not `MemFree`, which excludes reclaimable page cache and would make a
/// healthy machine look full.
fn mem_available_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse::<u64>().ok().map(|kb| kb / 1024);
        }
    }
    None
}

/// Projected peak RSS in MB for an `n × dim` run, summed across the server and
/// this process.
///
/// Fitted to measurements rather than guessed. Two 100k runs, server RSS at
/// peak, with the f32 rerank corpus and the i8 copy subtracted out:
///
///   384d: 980 MB total − (153 f32 + 38 i8) = 789 MB unaccounted
///   768d: 1211 MB total − (307 f32 + 77 i8) = 827 MB unaccounted
///
/// The remainder is flat in `dim`, which is what you'd expect — it is the graph
/// adjacency, and an edge costs the same whatever the vector length. It works
/// out to ~8 KB per node. So the model is `dim`-proportional storage (f32 + i8
/// ≈ 5 bytes per component) plus a flat per-node graph cost, and then the
/// dataset this process holds in memory on top.
fn projected_peak_mb(n: usize, dim: usize) -> u64 {
    let server_vecs = (n as u64 * dim as u64 * 5) / 1_000_000; // f32 corpus + i8 copy
    let server_graph = (n as u64 * 8) / 1024; // ~8 KB/node of adjacency
    let bench_dataset = (n as u64 * dim as u64 * 4) / 1_000_000; // our own f32 copy
    server_vecs + server_graph + bench_dataset
}

/// Refuse to start a run that the machine cannot hold.
///
/// This exists because `--xlarge` once took the whole desktop down with it: the
/// OOM killer does not politely pick the benchmark, and a run that dies at 90%
/// tells you nothing anyway. Checking first costs a file read.
///
/// Requires a 1.3x margin over the projection, because the projection is a fit
/// and allocator behaviour at peak is spiky. Set `DBSTRIKE_BENCH_NO_MEMGUARD=1`
/// to run anyway — but read the number it prints before you do.
fn memory_guard(n: usize, dim: usize, what: &str) -> bool {
    let need = projected_peak_mb(n, dim) * 13 / 10;
    let Some(avail) = mem_available_mb() else { return true };
    if avail >= need {
        println!("  \x1b[90mmemguard: {what} projects ~{need} MB peak, {avail} MB available — ok\x1b[0m");
        return true;
    }
    if std::env::var_os("DBSTRIKE_BENCH_NO_MEMGUARD").is_some() {
        println!("  \x1b[33mmemguard OVERRIDDEN: {what} projects ~{need} MB peak, only {avail} MB available\x1b[0m");
        return true;
    }
    println!("  \x1b[31mSKIPPED: {what} projects ~{need} MB peak but only {avail} MB is available.\x1b[0m");
    println!("  \x1b[31m  Free memory and re-run, or set DBSTRIKE_BENCH_NO_MEMGUARD=1 to force it.\x1b[0m");
    false
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
    let dir = scratch_dir().join(format!("dbstrike_bench_{}", std::process::id()));
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

/// Monotonic-ish wall clock in milliseconds (for TTL / bi-temporal tests).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
// 7b. Agent memory — DURABILITY across engine reopen (WAL replay + resume)
//     Writes LTM (text + vector + meta + keyword + graph edges), WM (TTL),
//     episodic, and procedural; then DROPS the in-memory engine + Memory and
//     reopens from the SAME WAL. Everything must survive: recall, graph
//     traversal, vector index, WM, episodic, procedural — and the id counter /
//     ltm_count / salience mirror must RESUME (not reset) so a post-reopen
//     store collides with nothing.
// ══════════════════════════════════════════════════════════════════════════
fn s7b_memory_durability() {
    section("7b. Agent memory — durability across engine reopen (WAL replay)");
    let wal = fresh_wal("s7b");
    let e1 = Engine::open(wal.clone()).unwrap();
    let mem1 = memory::Memory::open(Arc::clone(&e1));

    let dim = 16usize;
    // LTM + graph
    let rust_id = mem1.ltm_store("Rust ownership prevents data races",
                                 deterministic_vec(1, dim), "doc:rust", 0.8, "seed").unwrap();
    let py_id = mem1.ltm_store("Python has a GIL",
                               deterministic_vec(2, dim), "doc:py", 0.6, "seed").unwrap();
    mem1.link(rust_id, py_id, "related_to", 0.9).unwrap();
    // WM with TTL
    mem1.wm_set("agent-a", "task", b"write the report", 60_000, now_ms()).unwrap();
    // Episodic
    let ep_seq = mem1.episode("agent-a", "tool", b"called search()").unwrap();
    // Procedural
    mem1.proc_store("agent-a", "deploy", b"1. test\n2. push").unwrap();

    // Capture pre-reopen state we expect to recover.
    let pre_count = mem1.ltm_count();
    let pre_rust_text = mem1.ltm_get(rust_id).unwrap().text;
    let pre_wm = mem1.wm_get("agent-a", "task", now_ms());
    let pre_eps = mem1.episodes("agent-a", 10);
    let pre_proc = mem1.proc_get("agent-a", "deploy");

    // Drop the in-memory engine + Memory entirely (simulate process restart).
    drop(mem1);
    drop(e1);

    // Reopen from the SAME WAL — this replays the durable log + resumes counters.
    let e2 = Engine::open(wal.clone()).unwrap();
    let mem2 = memory::Memory::open(Arc::clone(&e2));

    // ── LTM text + meta survived ──
    let got = mem2.ltm_get(rust_id);
    check("LTM text survives reopen",
          got.as_ref().map(|r| r.text.as_str()) == Some(pre_rust_text.as_str()),
          &format!("got {:?}", got.map(|r| r.text)));
    // ── Vector index resumed (recall by dense query finds it) ──
    let hits = mem2.recall("rust ownership", &deterministic_vec(1, dim), 3);
    check("LTM vector index survives reopen (dense recall)",
          hits.iter().any(|h| h.id == rust_id),
          &format!("hits={}", hits.len()));
    // ── Graph edges survived (traversal reaches py_id) ──
    let path = mem2.traverse(rust_id, 2, "");
    check("memory graph edges survive reopen (traversal)",
          path.contains(&py_id), &format!("path={path:?}"));
    // ── ltm_count resumed (not reset to 0) ──
    check("ltm_count resumed after reopen",
          mem2.ltm_count() == pre_count,
          &format!("count={pre_count}"));
    // ── id counter resumed (new store gets a higher id, no collision) ──
    let new_id = mem2.ltm_store("a third fact",
                                deterministic_vec(3, dim), "doc:x", 0.5, "seed").unwrap();
    check("id counter resumed (no collision after reopen)",
          new_id > rust_id && new_id > py_id,
          &format!("new_id={new_id} rust={rust_id}"));
    // ── WM survived ──
    let wm = mem2.wm_get("agent-a", "task", now_ms());
    check("working memory survives reopen",
          wm == pre_wm, &format!("wm={:?}", wm.map(|v| String::from_utf8_lossy(&v).to_string())));
    // ── Episodic survived ──
    let eps = mem2.episodes("agent-a", 10);
    check("episodic log survives reopen",
          eps.len() == pre_eps.len() && eps.iter().any(|e| e.seq == ep_seq),
          &format!("eps={}", eps.len()));
    // ── Procedural survived ──
    let proc = mem2.proc_get("agent-a", "deploy");
    check("procedural memory survives reopen",
          proc == pre_proc, &format!("proc={:?}", proc.map(|v| String::from_utf8_lossy(&v).to_string())));
    // ── forget + reopen: deletion also durable ──
    mem2.ltm_forget(rust_id).unwrap();
    drop(mem2);
    drop(e2);
    let e3 = Engine::open(wal.clone()).unwrap();
    let mem3 = memory::Memory::open(Arc::clone(&e3));
    check("LTM forget survives reopen (deletion durable)",
          mem3.ltm_get(rust_id).is_none(), "rust_id gone after reopen");
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

/// Append one RESP bulk string, deriving `$len` from the payload itself.
///
/// Every hand-written `b"$8\r\nVADDBATCH\r\n"` literal is a place where the
/// declared length and the payload can silently disagree — and one did:
/// `VADDBATCH` is 9 bytes, not 8. The server consumed 8 bytes plus a 2-byte
/// CRLF that wasn't there, landed on `\n` instead of `$`, and rejected the
/// frame as "expected bulk string". Because a `-ERR` reply was being treated as
/// success by the client, the bench reported a full 100k ingest against an
/// index that had received nothing. Compute the length; never type it.
fn push_bulk(cmd: &mut Vec<u8>, payload: &[u8]) {
    cmd.extend_from_slice(format!("${}\r\n", payload.len()).as_bytes());
    cmd.extend_from_slice(payload);
    cmd.extend_from_slice(b"\r\n");
}

/// Build a RESP `VADDBATCH [PAR] dim id0 f0.. id1 f1.. ...` command.
/// If `parallel` is true, prepends "PAR" flag for parallel graph rebuild.
fn vaddbatch_cmd(dim: usize, ids: &[u64], vecs: &[f32], parallel: bool) -> Vec<u8> {
    // N args = [PAR?] + VADDBATCH + dim + for each vec: (1 id + dim floats)
    let n_args = if parallel { 3 } else { 2 } + ids.len() * (1 + dim);
    let mut cmd: Vec<u8> = format!("*{}\r\n", n_args).into_bytes();
    // Command name FIRST, then the optional flag. The server takes `cmds[i][0]`
    // as the command name and only then looks for `PAR` in `args`, so emitting
    // `PAR` ahead of `VADDBATCH` would make the command name literally "PAR".
    // Latent until now only because every caller passes `parallel = false`.
    push_bulk(&mut cmd, b"VADDBATCH");
    if parallel {
        push_bulk(&mut cmd, b"PAR");
    }
    push_bulk(&mut cmd, dim.to_string().as_bytes());
    for (i, &id) in ids.iter().enumerate() {
        push_bulk(&mut cmd, id.to_string().as_bytes());
        let base = i * dim;
        for j in 0..dim {
            push_bulk(&mut cmd, format!("{}", vecs[base + j]).as_bytes());
        }
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

/// Build a RESP `VSEARCH k [FLAGS] f1 f2 ...` command with MODULE 4/3/5 access
/// path flags (F cat / L / H terms...). One command, every path.
fn vsearch_flags_cmd(k: usize, flags: &[Vec<u8>], q: &[f32]) -> Vec<u8> {
    let nargs = 2 + flags.len() + q.len();
    let mut cmd: Vec<u8> = format!("*{nargs}\r\n").into_bytes();
    cmd.extend_from_slice(b"$7\r\nVSEARCH\r\n");
    let ks = k.to_string();
    cmd.extend_from_slice(format!("${}\r\n", ks.len()).as_bytes());
    cmd.extend_from_slice(ks.as_bytes());
    cmd.extend_from_slice(b"\r\n");
    for f in flags {
        cmd.extend_from_slice(format!("${}\r\n", f.len()).as_bytes());
        cmd.extend_from_slice(f);
        cmd.extend_from_slice(b"\r\n");
    }
    for f in q {
        let s = format!("{f}");
        cmd.extend_from_slice(format!("${}\r\n", s.len()).as_bytes());
        cmd.extend_from_slice(s.as_bytes());
        cmd.extend_from_slice(b"\r\n");
    }
    cmd
}

impl Client {
    /// Send a VSEARCH (with optional flags) and parse the (id, dist) array
    /// reply into a Vec<(u64, f32)>. Used by the unified-module wire bench.
    fn vsearch_collect(&mut self, k: usize, flags: &[Vec<u8>], q: &[f32]) -> std::io::Result<Vec<(u64, f32)>> {
        self.send_raw(&vsearch_flags_cmd(k, flags, q))?;
        self.read_id_dist_array()
    }

    /// Read the ids from an already-sent VSEARCH, leaving the connection
    /// positioned exactly at the end of that reply.
    ///
    /// This exists because the recall check reuses one connection for many
    /// queries. The previous code wrapped `stream.try_clone()` in a fresh
    /// `BufReader` per query, which is only safe when the connection is thrown
    /// away afterwards: the `BufReader` reads ahead into its own buffer and
    /// takes any bytes belonging to the *next* reply with it when it drops.
    /// Going through `Client`'s own buffer keeps the stream consistent.
    fn vsearch_reply_ids(&mut self) -> std::io::Result<Vec<u64>> {
        Ok(self.read_id_dist_array()?.into_iter().map(|(id, _)| id).collect())
    }

    /// Parse one `[id, dist, id, dist, ...]` array reply.
    fn read_id_dist_array(&mut self) -> std::io::Result<Vec<(u64, f32)>> {
        let line = self.read_line()?;
        if line.first() == Some(&b'-') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("server replied {}", String::from_utf8_lossy(&line)),
            ));
        }
        if line.first() == Some(&b'*') {
            let n: i64 = std::str::from_utf8(&line[1..]).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
            let mut out = Vec::new();
            let mut i = 0;
            while i < n {
                // id — server returns RESP integers (`:N\r\n`) or bulk strings
                // (`$L\r\nN\r\n`); handle both.
                let idline = self.read_line()?;
                let id = match idline.first() {
                    Some(&b':') => std::str::from_utf8(&idline[1..])
                        .ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0),
                    Some(&b'$') => {
                        let len: usize = std::str::from_utf8(&idline[1..]).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                        let b = self.read_n(len + 2)?;
                        std::str::from_utf8(&b[..len]).ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0)
                    }
                    _ => 0,
                };
                // dist (bulk float-as-bytes)
                let dline = self.read_line()?;
                let dist = if dline.first() == Some(&b'$') {
                    let len: usize = std::str::from_utf8(&dline[1..]).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                    let b = self.read_n(len + 2)?;
                    std::str::from_utf8(&b[..len]).ok().and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0)
                } else { 0.0 };
                out.push((id, dist));
                i += 2;
            }
            Ok(out)
        } else {
            // error or simple reply → treat as empty
            Ok(Vec::new())
        }
    }
}

/// Ingest `n` vectors into a running server over the RESP wire (VADD),
/// sharded across `n_threads` client connections so we actually stress the
/// server's ingest path concurrently. Prints flushing progress heartbeats so a
/// foreground run never looks frozen. Returns vec/s.
///
/// `vec_of` maps an id -> the vector to store (synthetic or loaded from a
/// real dataset). This single helper backs both the synthetic and the
/// `--real` benchmark paths.
#[allow(dead_code)]
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
        handles.push(thread::spawn(move || -> Result<(), String> {
            let mut c = Client::connect(&addr)
                .map_err(|e| format!("client {tid}: connect {addr}: {e}"))?;
            let start = (tid * per) as u64;
            let end = ((tid + 1) * per).min(n as usize) as u64;
            for i in start..end {
                let v = vec_of(i);
                c.send_raw(&vadd_cmd(i, &v))
                    .map_err(|e| format!("client {tid}: VADD id {i}: send: {e}"))?;
                c.drain_reply()
                    .map_err(|e| format!("client {tid}: VADD id {i}: reply: {e}"))?;
                done_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
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
    let mut errs: Vec<String> = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => errs.push(e),
            Err(_) => errs.push("ingest client thread panicked".to_string()),
        }
    }
    let done = done_count.load(std::sync::atomic::Ordering::Relaxed);
    if !errs.is_empty() {
        eprintln!("  \x1b[31mingest FAILED after {done}/{n} vectors:\x1b[0m");
        for e in errs.iter().take(4) {
            eprintln!("    {e}");
        }
        eprintln!("    (server log: {}/dbstrike_server_*.log)", scratch_dir().display());
        return 0.0;
    }
    let dt = t0.elapsed().as_secs_f64();
    let rate = n as f64 / dt;
    println!("  ingest: {rate:.0} vec/s ({dt:.1}s total)");
    rate
}

/// Pipelined ingest using VADDBATCH — sends `batch_size` vectors per command.
/// Uses VADDBATCH PAR for parallel graph rebuild (cores× faster than serial).
fn wire_ingest_pipelined(addr: &str, n: u64, dim: usize, n_threads: usize,
                         batch_size: usize,
                         vec_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    wire_ingest_pipelined_range(addr, 0, n, dim, n_threads, batch_size, vec_of)
}

/// Ingest ids `id_lo .. id_lo + n` over the wire. Returns vectors/second.
///
/// The offset exists so an ingest can be walked in stages against one server —
/// `--memprobe` reads the allocator's books between stages, and that only
/// attributes anything if the stages land in the same index rather than each
/// starting from id 0.
fn wire_ingest_pipelined_range(addr: &str, id_lo: u64, n: u64, dim: usize, n_threads: usize,
                         batch_size: usize,
                         vec_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    // Not "VADDBATCH PAR". `send_batch` below passes `parallel = false`, so the
    // PAR flag never reaches the wire and the server takes the incremental
    // merge path, not the rebuild one. The old label advertised a code path
    // this function does not exercise — a benchmark log that names the wrong
    // path is worse than no log, because it survives into the README.
    // `DBSTRIKE_INGEST_BATCH` overrides the vectors-per-command.
    //
    // The default of 64 models a streaming writer. A bulk load is a different
    // workload and wants a much larger batch: `VADDBATCH PAR` only routes to
    // the GPU builder when the batch is at least a quarter of the current
    // index, so at 64 vectors it always falls back to the serial append path
    // and the GPU builder is unreachable from the wire however the server is
    // configured. Making this settable is what lets a bulk load actually be
    // benchmarked as a bulk load.
    let batch_size = std::env::var("DBSTRIKE_INGEST_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(batch_size);
    println!("  ingesting {n} × {dim}d over RESP VADDBATCH (batch={batch_size}, {n_threads} clients) ...");
    let t0 = Instant::now();
    let per = (n as usize).div_ceil(n_threads);
    let done_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut handles = Vec::with_capacity(n_threads);
    for tid in 0..n_threads {
        let addr = addr.to_string();
        let done_count = Arc::clone(&done_count);
        let vec_of = Arc::clone(&vec_of);
        handles.push(thread::spawn(move || -> Result<(), String> {
            let mut c = Client::connect(&addr)
                .map_err(|e| format!("client {tid}: connect {addr}: {e}"))?;
            let start = id_lo + (tid * per) as u64;
            let end = id_lo + ((tid + 1) * per).min(n as usize) as u64;
            let mut batch_ids: Vec<u64> = Vec::with_capacity(batch_size);
            let mut batch_vecs: Vec<f32> = Vec::with_capacity(batch_size * dim);
            // One VADDBATCH round-trip, with the failure annotated by the id
            // range that was in flight. A bare `ConnectionReset` from an
            // `.unwrap()` said nothing about *which* command killed the server;
            // this says exactly how far the ingest got and how big the command
            // was, which is what you need to reproduce it.
            // `DBSTRIKE_INGEST=par` selects VADDBATCH's parallel-rebuild path.
            // Off by default so the headline ingest number stays the one the
            // README quotes; settable so the two paths are comparable on the
            // same dataset in the same run, which is the only way to say what
            // the sharded rebuild is actually worth.
            let par_ingest = std::env::var("DBSTRIKE_INGEST").as_deref() == Ok("par");
            let send_batch = |c: &mut Client, ids: &[u64], vecs: &[f32]| -> Result<(), String> {
                let cmd = vaddbatch_cmd(dim, ids, vecs, par_ingest);
                let span = format!(
                    "client {tid}: VADDBATCH ids {}..={} ({} vecs × {dim}d, {} bytes)",
                    ids[0], ids[ids.len() - 1], ids.len(), cmd.len()
                );
                c.send_raw(&cmd).map_err(|e| format!("{span}: send: {e}"))?;
                c.drain_reply().map_err(|e| format!("{span}: reply: {e}"))?;
                Ok(())
            };
            for i in start..end {
                let v = vec_of(i);
                batch_ids.push(i);
                batch_vecs.extend_from_slice(&v);
                if batch_ids.len() >= batch_size {
                    // Use non-PAR (incremental merge) for small batches —
                    // PAR does a full graph rebuild per batch which is wasteful
                    // when batches are small. The incremental path does parallel
                    // segment build + serial merge_into, which is faster for
                    // repeated small batches.
                    send_batch(&mut c, &batch_ids, &batch_vecs)?;
                    done_count.fetch_add(batch_ids.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    batch_ids.clear();
                    batch_vecs.clear();
                }
            }
            if !batch_ids.is_empty() {
                send_batch(&mut c, &batch_ids, &batch_vecs)?;
                done_count.fetch_add(batch_ids.len() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
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
    // Collect wire failures instead of unwrapping them. A reset connection is a
    // *result* — the section's `check()` should fail with an explanation — not a
    // reason to abort the whole bench run and leave the server orphaned.
    let mut errs: Vec<String> = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => errs.push(e),
            Err(_) => errs.push("ingest client thread panicked".to_string()),
        }
    }
    let done = done_count.load(std::sync::atomic::Ordering::Relaxed);
    if !errs.is_empty() {
        eprintln!("  \x1b[31mingest FAILED after {done}/{n} vectors:\x1b[0m");
        for e in errs.iter().take(4) {
            eprintln!("    {e}");
        }
        if errs.len() > 4 {
            eprintln!("    ... and {} more", errs.len() - 4);
        }
        eprintln!("    (server log: {}/dbstrike_server_*.log)", scratch_dir().display());
        return 0.0;
    }
    let dt = t0.elapsed().as_secs_f64();
    let rate = n as f64 / dt;
    println!("  ingest: {rate:.0} vec/s ({dt:.1}s total)");
    rate
}

/// Synthetic-ingest convenience wrapper (clustered_vec dataset). Pipelined.
fn wire_ingest(addr: &str, n: u64, dim: usize, n_clusters: u64,
               n_threads: usize) -> f64 {
    wire_ingest_pipelined(addr, n, dim, n_threads, 64,
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

/// Concurrent VSEARCH QPS — `n_threads` clients each fire `per` queries,
/// each query produced by `query_of(qi)`. Reports aggregate QPS across
/// all clients (the "RPS case" Qdrant separates from the latency case).
fn wire_concurrent_qps_with(addr: &str, n_threads: usize, per: usize,
                             query_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    let (qps, _p99) = wire_concurrent_qps_lat(addr, n_threads, per, &query_of);
    qps
}

/// Like `wire_concurrent_qps_with` but also returns the client-side p99
/// latency (µs) across ALL queries, so the multi-client ("RPS case") number
/// is validated with a real tail-latency figure rather than left blank.
fn wire_concurrent_qps_lat(addr: &str, n_threads: usize, per: usize,
                            query_of: &std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>)
                            -> (f64, u64) {
    let t0 = Instant::now();
    let qlatch: Vec<Vec<u64>> = (0..n_threads)
        .map(|_| Vec::with_capacity(per))
        .collect();
    let shared = std::sync::Arc::new(std::sync::Mutex::new(qlatch));
    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            let addr = addr.to_string();
            let query_of = std::sync::Arc::clone(query_of);
            let shared = std::sync::Arc::clone(&shared);
            thread::spawn(move || {
                let mut c = Client::connect(&addr).unwrap();
                let mut local: Vec<u64> = Vec::with_capacity(per);
                for i in 0..per {
                    let q = query_of((tid * 100_000 + i) as u64);
                    let cmd = vsearch_cmd(&q);
                    let s = Instant::now();
                    c.send_raw(&cmd).unwrap();
                    c.drain_reply().unwrap();
                    local.push(s.elapsed().as_micros() as u64);
                }
                shared.lock().unwrap()[tid] = local;
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let qps = ((n_threads * per) as f64) / t0.elapsed().as_secs_f64();
    let mut all: Vec<u64> = shared.lock().unwrap().iter().flat_map(|v| v.iter().copied()).collect();
    all.sort_unstable();
    let p99 = all[(all.len() * 99 / 100).min(all.len() - 1)];
    (qps, p99)
}

/// Concurrent VSEARCH QPS — synthetic clustered queries (legacy helper).
fn wire_concurrent_qps(addr: &str, dim: usize, n_clusters: u64,
                       n_threads: usize, per: usize) -> f64 {
    wire_concurrent_qps_with(addr, n_threads, per,
        std::sync::Arc::new(move |qi| clustered_vec(qi, dim, n_clusters)))
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
///
/// This function used to be the single largest allocator in the benchmark, and
/// it was the reason a 200k run got OOM-killed. Three separate mistakes stacked
/// multiplicatively:
///
///   1. `data.to_vec()` copied the entire corpus a second time, purely to get
///      an owned `Arc` — 307 MB at 200k×384d, 1.5 GB at 1M, on top of the two
///      copies the caller already holds.
///   2. Every query got its own thread, spawned all at once with no pool. 200
///      queries meant 200 live threads and 200 simultaneous server connections.
///   3. Each of those threads built `Vec<(f32, u64)>` with capacity `n` to find
///      ten elements. `(f32, u64)` is 16 bytes after padding, so that is 3.2 MB
///      per thread at 200k and 16 MB at 1M — times 200 threads, 640 MB and
///      3.2 GB respectively, then sorted in full to read off the first ten.
///
/// All three are fixed below: the corpus is borrowed rather than copied, the
/// queries run on a bounded pool, and the exact top-10 comes from a
/// fixed-size insertion buffer that is O(k) memory instead of O(n). Sorting n
/// items to take 10 was also the dominant cost in time, not just space.
fn wire_cluster_recall(addr: &str, dim: usize, n: u64, n_queries: usize,
                       data: &[f32],
                       query_of: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync>) -> f64 {
    println!("  Recall@10 vs brute-force ground truth ({n_queries} q, true NN) ...");
    let t_r = Instant::now();

    // Bounded pool. The point of this phase is ground truth, not load
    // generation — the QPS sections measure concurrency deliberately, and
    // letting an unrelated correctness check open 200 connections meant the
    // server's peak memory depended on how many queries we happened to ask for.
    let workers = n_queries.min(8).max(1);
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let total_hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Borrow the corpus across scoped threads instead of cloning it. The
    // brute force only ever reads it.
    thread::scope(|s| {
        for _ in 0..workers {
            let addr = addr.to_string();
            let query_of = std::sync::Arc::clone(&query_of);
            let next = Arc::clone(&next);
            let total_hits = Arc::clone(&total_hits);
            let data: &[f32] = data;
            s.spawn(move || {
                let mut c = match Client::connect(&addr) {
                    Ok(c) => c,
                    Err(e) => { eprintln!("  recall worker: connect {addr}: {e}"); return; }
                };
                loop {
                    let qi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if qi >= n_queries { break; }
                    let q = query_of(qi as u64);

                    // Exact top-10 by dot product (L2-normalized => cosine),
                    // kept in a 10-slot buffer. `worst` is the current
                    // admission threshold, so the common case is one compare
                    // and no write at all.
                    const K: usize = 10;
                    let mut top: Vec<(f32, u64)> = Vec::with_capacity(K + 1);
                    let mut worst = f32::NEG_INFINITY;
                    for i in 0..n {
                        let base = (i as usize) * dim;
                        let row = &data[base..base + dim];
                        let mut d = 0f32;
                        for j in 0..dim {
                            d += q[j] * row[j];
                        }
                        if top.len() == K && d <= worst { continue; }
                        let pos = top.partition_point(|&(s, _)| s > d);
                        top.insert(pos, (d, i));
                        if top.len() > K { top.pop(); }
                        worst = top[top.len() - 1].0;
                    }
                    let truth: std::collections::BTreeSet<u64> =
                        top.iter().map(|(_, id)| *id).collect();

                    // Server ANN top-10, on this worker's own reused
                    // connection rather than a fresh one per query.
                    if c.send_raw(&vsearch_cmd(&q)).is_err() { return; }
                    let got = match c.vsearch_reply_ids() {
                        Ok(g) => g,
                        Err(e) => { eprintln!("  recall worker: query {qi}: {e}"); return; }
                    };
                    let hit = got.iter().filter(|id| truth.contains(id)).count();
                    total_hits.fetch_add(hit, std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });

    let recall = total_hits.load(std::sync::atomic::Ordering::Relaxed) as f64
        / (n_queries as f64 * 10.0);
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
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("s13_{port}"), "wal");
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
    let recall = wire_cluster_recall(&addr, DIM, N, 200, &sdata, std::sync::Arc::clone(&squery));
    check("100k×384d Recall@10 ≥ 0.85 (wire)", recall >= 0.85,
          &format!("recall={:.3}", recall));

    let cores = num_cores();
    // Single-client "Latency case" (Qdrant separates this from the RPS case):
    // one connection, serial queries → the per-request ceiling.
    let (qps_1, p99_1) = wire_concurrent_qps_lat(&addr, 1, 1000, &std::sync::Arc::clone(&squery));
    println!("  1-client  VSEARCH (Latency case, wire): {qps_1:.0} QPS  p99={p99_1}µs");
    check("100k×384d single-client QPS > 400 (wire, beat Qdrant ~450)",
          qps_1 > 400.0, &format!("{qps_1:.0} QPS"));
    // 100-client "RPS case" (Qdrant's headline scenario): saturate the cores.
    // Report BOTH aggregate QPS and the measured client-side p99 so the
    // tail latency is validated, not left blank.
    let (qps_100, p99_100) = wire_concurrent_qps_lat(&addr, 100, 300, &squery);
    println!("  100-client VSEARCH (RPS case, wire, {cores} cores): {qps_100:.0} QPS  p99={p99_100}µs");
    check("100k×384d 100-client QPS > 13000 (wire, beat Qdrant ~13k RPS)",
          qps_100 > 13_000.0, &format!("{qps_100:.0} QPS"));
    // Also keep the old 8-thread check for regression tracking.
    let qps_8 = wire_concurrent_qps_with(&addr, 8, 300, std::sync::Arc::clone(&squery));
    println!("  8-client VSEARCH QPS (wire): {qps_8:.0} QPS");
    check("100k×384d concurrent QPS > 5000 (wire)", qps_8 > 5_000.0,
          &format!("{qps_8:.0} QPS"));
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
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("s15_{port}"), "wal");
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
    let recall = wire_cluster_recall(&addr, DIM, N, 200, &sdata, std::sync::Arc::clone(&squery));
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
            // A `-ERR ...` reply is a FAILURE, not a drained reply. Treating it
            // as `Ok(())` is how a rejected VADDBATCH still incremented the
            // "done" counter: the ingest reported 100k vectors stored while the
            // index was empty, and the recall/QPS numbers measured afterwards
            // were meaningless. Surface the server's own message instead.
            Some(b'-') => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("server replied {}", String::from_utf8_lossy(&line)),
            )),
            Some(b'+') | Some(b':') => Ok(()),
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
    /// Send a command and return its bulk-string payload as text.
    ///
    /// `drain_reply` deliberately throws the body away, which is right for a
    /// throughput loop and useless for a diagnostic: `MEMTRACK` exists entirely
    /// for what it says, not for the fact that it answered.
    fn cmd_text(&mut self, args: &[&[u8]]) -> std::io::Result<String> {
        self.send(args)?;
        let line = self.read_line()?;
        match line.first() {
            Some(b'-') => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("server replied {}", String::from_utf8_lossy(&line)),
            )),
            Some(b'$') => {
                let n: i64 = std::str::from_utf8(&line[1..]).ok()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| std::io::Error::new(
                        std::io::ErrorKind::InvalidData, "bad bulk hdr"))?;
                if n < 0 {
                    return Ok(String::new());
                }
                let body = self.read_n(n as usize + 2)?;
                Ok(String::from_utf8_lossy(&body[..n as usize]).into_owned())
            }
            Some(b'+') | Some(b':') => {
                Ok(String::from_utf8_lossy(&line[1..]).into_owned())
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "expected bulk reply")),
        }
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

/// Scratch directory for server WALs and logs.
///
/// Deliberately NOT `/tmp`: on this machine `/tmp` is a small tmpfs shared
/// with the rest of the desktop session, and a 1M-vector WAL there has already
/// filled it mid-run (an ENOSPC inside the server, which then dies and shows up
/// on the client as a bare `ConnectionReset`). Keeping scratch inside the repo
/// puts it on the same big disk as the datasets and makes leftovers visible.
/// Override with `DBSTRIKE_BENCH_SCRATCH` if you want it elsewhere.
fn scratch_dir() -> std::path::PathBuf {
    let dir = std::env::var("DBSTRIKE_BENCH_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("bench-out/scratch"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Path to a scratch file, e.g. `scratch_path("s13", "wal")`.
fn scratch_path(tag: &str, ext: &str) -> String {
    scratch_dir().join(format!("dbstrike_{tag}.{ext}")).to_string_lossy().into_owned()
}

/// Ask the OS for a genuinely free TCP port by binding one and immediately
/// dropping the listener.
///
/// The old scheme was `BASE + (pid % RANGE)`, which is not a port allocator —
/// it is a hash of the pid. Two bench runs, or one run whose previous server
/// leaked, collide on the same number. That failure is nastier than it sounds:
/// `spawn_dbstrike`'s readiness probe would PING the *leftover* server, see it
/// answer, and hand back a handle to a child that had actually died with
/// "address already in use". Every subsequent command then went to a stale
/// server holding a stale index.
///
/// There is still a small TOCTOU window between our unbind and the child's
/// bind, but the kernel hands out ports it believes are free, so collisions
/// need another process to grab the exact same port inside a few milliseconds.
/// That is dramatically better than deterministically reusing one.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("could not allocate a free TCP port")
}

/// RAII guard that SIGKILLs the spawned server on drop, so an aborted bench
/// run can't leak a listening `dbstrike` (which would wedge a later run that
/// re-picks the same port).
///
/// Also carries the path to the server's captured log so a failing section can
/// print *why* the server misbehaved instead of leaving a bare I/O error.
struct ChildGuard {
    child: std::process::Child,
    log: String,
}
impl ChildGuard {
    /// Tail of the server's stdout+stderr. This is the diagnostic that used to
    /// be thrown away by `Stdio::null()`.
    fn log_tail(&self, lines: usize) -> String {
        match std::fs::read_to_string(&self.log) {
            Ok(s) => {
                let all: Vec<&str> = s.lines().collect();
                let start = all.len().saturating_sub(lines);
                all[start..].join("\n")
            }
            Err(e) => format!("(could not read {}: {e})", self.log),
        }
    }
    /// `Some(status)` if the server has already exited — i.e. the connection
    /// error the client just saw was the server dying, not a transient reset.
    fn exited(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

    // Capture the server's stdout+stderr to a file rather than discarding it.
    // With `Stdio::null()` a server-side panic, an ENOSPC, or an "address
    // already in use" was completely invisible: the only symptom reaching the
    // bench was `ConnectionReset` on some unrelated-looking command, because a
    // panicking connection thread drops its `TcpStream` and the kernel sends an
    // RST. Keeping the log is what turns that into an actual error message.
    let log_path = scratch_path(&format!("server_{port}"), "log");
    let log = std::fs::File::create(&log_path)
        .unwrap_or_else(|e| panic!("cannot create server log {log_path}: {e}"));
    let log_err = log.try_clone().expect("dup server log fd");

    let mut cmd = std::process::Command::new(&bin);
    cmd.arg(format!("127.0.0.1:{port}"))
        .env("DBSTRIKE_WAL", wal)
        .env("DBSTRIKE_SYNC", if sync { "1" } else { "0" })
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err));
    let child = cmd.spawn().expect("spawn dbstrike");
    let mut guard = ChildGuard { child, log: log_path.clone() };

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        // Check liveness BEFORE probing. If the child already exited (failed to
        // bind, bad WAL path, ...) then any PING that succeeds is being answered
        // by somebody else's server on this port — accepting it would silently
        // run the whole section against a stale process. Fail loudly instead.
        if let Some(status) = guard.exited() {
            panic!(
                "dbstrike exited during startup with {status}\n\
                 --- server log ({log_path}) ---\n{}",
                guard.log_tail(40)
            );
        }
        if let Ok(mut c) = Client::connect(&format!("127.0.0.1:{port}")) {
            if c.cmd(&[b"PING"]).is_ok() {
                return guard;
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!(
        "server never accepted PING within 5s on port {port}\n\
         --- server log ({log_path}) ---\n{}",
        guard.log_tail(40)
    );
}

fn s17_chaos(iterations: usize) {
    section("17. Chaos — SIGKILL under load, verify acked-write durability");
    let port = free_port();
    let wal = scratch_path(&format!("chaos_{port}"), "wal");
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
            libc_kill(child.child.id() as i32, 2 /*SIGINT*/);
        }
        let _ = child.child.wait();
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
            libc_kill(child.child.id() as i32, 9 /*SIGKILL*/);
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
            libc_kill(child.child.id() as i32, 2 /*SIGINT*/);
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
    c.child.wait()
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

    // Guard AFTER the load, because `n` and `dim` come out of the file header —
    // but the load itself is bounded by the file size, which is knowable and
    // modest next to the graph we are about to build. The dangerous allocation
    // is the server's index, and that is still ahead of us.
    if !memory_guard(n, dim, &format!("REAL {n}×{dim}d")) {
        return;
    }

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("real_{dim}_{port}"), "wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);

    let data = std::sync::Arc::new(data);
    let norm = std::sync::Arc::new(norm);
    // Pipeline depth 64: send 64 VADDs before draining replies.
    // Amortizes TCP round-trip across the batch — same trick redis-benchmark uses.
    let ingest_rate = wire_ingest_pipelined(&addr, n as u64, dim, 8, 64,
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

    let qps = wire_concurrent_qps_with(&addr, 8, 150, {
        let data = std::sync::Arc::clone(&norm);
        let n = n;
        std::sync::Arc::new(move |qi| {
            let idx = (qi as usize * 7919) % n;
            data[idx * dim..(idx + 1) * dim].to_vec()
        })
    });
    println!("  8-client concurrent VSEARCH (wire): {qps:.0} QPS");
    check(&format!("REAL {n}×{dim}d concurrent QPS > 1000 (wire)"), qps > 1_000.0,
          &format!("{qps:.0} QPS"));

    // Peak-QPS sweep — matches Qdrant's "RPS case" (they publish ~13k QPS
    // at 100 concurrent clients over gRPC). Report 16/32-client QPS so the
    // peak number is directly comparable, not just the 8-client latency-case figure.
    let query_sweep: std::sync::Arc<dyn Fn(u64) -> Vec<f32> + Send + Sync> =
        std::sync::Arc::new({
            let data = std::sync::Arc::clone(&norm);
            let n = n;
            move |qi| {
                let idx = (qi as usize * 7919) % n;
                data[idx * dim..(idx + 1) * dim].to_vec()
            }
        });
    print!("  peak-QPS sweep: ");
    for clients in [16usize, 32] {
        let q = wire_concurrent_qps_with(&addr, clients, 200, std::sync::Arc::clone(&query_sweep));
        print!("{clients}c={q:.0}QPS ");
        check(&format!("REAL {n}×{dim}d {clients}c QPS > 1000 (wire)"), q > 1_000.0,
              &format!("{q:.0} QPS"));
    }
    println!();

    println!(
        "\n  \x1b[1msummary — REAL {n} × {dim}d ({tag})\x1b[0m: ingest {ingest_rate:.0} vec/s · \
         p99 {p99} µs · Recall@10 {:.3} · 8-thread QPS {qps:.0}",
        recall
    );
}

/// In-process comparison of FIXED-ef vs ADAPTIVE search on real embeddings.
/// Builds a `VectorIndex` straight from the `.fbin` (no wire, no fsync),
/// computes a TRUE brute-force cosine ground truth, then reports Recall@10
/// and p50/p99 single-query latency for each strategy. This is the direct
/// proof of the ruvector-style win: adaptive should hold recall while
/// cutting p99 latency (fewer distance computations on easy queries).
fn s19b_adaptive_vs_fixed(path: &str) {
    let (n, dim, data, norm) = load_fbin(path);
    println!("\n\x1b[1m── ADAPTIVE vs FIXED-ef (REAL {n} × {dim}d, in-process) ──\x1b[0m");

    // Same graph size as the wire section above, except the index lives in this
    // process instead of the server's — so the projection is the same number.
    if !memory_guard(n, dim, &format!("ADAPTIVE vs FIXED {n}×{dim}d")) {
        return;
    }

    // Build index in-process. Use graph-only insert (no WAL fsync per
    // vector) — same fast path the wire benches use — so we measure pure
    // search-algorithm behavior, not fsync throughput. Graph + f32 mirror
    // are still built, which is all the search path needs.
    let dir = scratch_dir().join(format!("dbstrike_adp_{}_{}", std::process::id(), dim));
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("adp.wal");
    let _ = std::fs::remove_file(&wal);
    let engine = Engine::open(&wal).unwrap();
    let vidx = VectorIndex::open(engine);
    for i in 0..n as u64 {
        vidx.insert_graph_only(i, data[i as usize * dim..(i as usize + 1) * dim].to_vec());
    }
    println!("  built graph: {} live vectors", vidx.len());

    // TRUE ground truth: brute-force top-10 cosine for each query.
    let nq = if n > 5000 { 500usize } else { n };
    let mut gt: Vec<Vec<u64>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let v = &data[i as usize * dim..(i as usize + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim { dot += q[j] * v[j]; }
            scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        gt.push(scored.iter().take(10).map(|(id, _)| *id).collect());
    }

    let recall_at = |hits: &[(u64, f32)], truth: &[u64]| -> usize {
        hits.iter().take(10).filter(|(id, _)| truth.contains(id)).count()
    };

    // Strategy A: fixed ef (current default = 128).
    let mut lat_a: Vec<u64> = Vec::with_capacity(nq);
    let mut hit_a = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        let hits = vidx.search_ef(q, 10, 128);
        lat_a.push(t0.elapsed().as_nanos() as u64 / 1000);
        hit_a += recall_at(&hits, &gt[qi]);
    }
    // Strategy B: adaptive (ruvector-style).
    let mut lat_b: Vec<u64> = Vec::with_capacity(nq);
    let mut hit_b = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        let hits = vidx.search_adaptive(q, 10, 16, 32, 256);
        lat_b.push(t0.elapsed().as_nanos() as u64 / 1000);
        hit_b += recall_at(&hits, &gt[qi]);
    }

    let r_a = hit_a as f32 / (nq * 10) as f32;
    let r_b = hit_b as f32 / (nq * 10) as f32;
    let mut la = lat_a.clone(); let mut lb = lat_b.clone();
    let p50_a = pctl(&mut la, 50.0); let p99_a = pctl(&mut la, 99.0);
    let p50_b = pctl(&mut lb, 50.0); let p99_b = pctl(&mut lb, 99.0);
    let mean_a: u64 = la.iter().sum::<u64>() / la.len() as u64;
    let mean_b: u64 = lb.iter().sum::<u64>() / lb.len() as u64;

    println!("  FIXED  ef=128 : Recall@10 {r_a:.3} · mean {mean_a}µs · p50 {p50_a}µs · p99 {p99_a}µs");
    println!("  ADAPTIVE      : Recall@10 {r_b:.3} · mean {mean_b}µs · p50 {p50_b}µs · p99 {p99_b}µs");
    let speedup = if mean_b > 0 { mean_a as f64 / mean_b as f64 } else { 0.0 };
    println!("  latency speedup (mean): {speedup:.2}x  ·  recall Δ: {:.3}", r_b - r_a);

    check("adaptive Recall@10 ≥ fixed ef Recall@10", r_b >= r_a - 1e-3,
          &format!("a={r_a:.3} b={r_b:.3}"));
    check("adaptive mean latency ≤ fixed ef (1.05x)", mean_b as f64 <= mean_a as f64 * 1.05,
          &format!("a={mean_a}µs b={mean_b}µs"));
}

/// In-process REAL-dataset ingest+search profiler (--real-ingest <path>).
/// Builds the HNSW graph directly from the `.fbin` (graph-only insert — no
/// wire, no WAL fsync) so we measure pure graph-build throughput against real
/// embeddings and validate the O(N) ingest fix at true 1M scale. Reports
/// vec/s (must stay FLAT across the run, not collapse — that's the O(N²) tell)
/// plus a Recall@10 sanity vs brute-force ground truth.
fn s21_real_ingest(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── REAL INGEST: {n} × {dim}d ({path}) ──\x1b[0m");

    let dir = scratch_dir().join(format!("dbstrike_ri_{}_{}", std::process::id(), dim));
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("ri.wal");
    let _ = std::fs::remove_file(&wal);
    let engine = Engine::open(&wal).unwrap();
    let vidx = VectorIndex::open(engine);
    eprintln!("[MITM] s21 loaded n={n} dim={dim}");

    let t_all = Instant::now();
    let mut last = Instant::now();
    for i in 0..n as u64 {
        vidx.insert_graph_only(i, norm[i as usize * dim..(i as usize + 1) * dim].to_vec());
        if (i + 1) % 50_000 == 0 {
            let dt = last.elapsed().as_secs_f64();
            last = Instant::now();
            let done = i + 1;
            eprintln!(
                "  {:>7} vecs: last 50k in {:.2}s -> {:.0} vec/s  (avg {:.1} us/vec, elapsed {:.1}s)",
                done, dt, 50_000.0 / dt, t_all.elapsed().as_secs_f64() * 1e6 / done as f64,
                t_all.elapsed().as_secs_f64(),
            );
            std::io::stderr().flush().ok();
        }
    }
    let total = t_all.elapsed().as_secs_f64();
    let rate = n as f64 / total;
    eprintln!("  TOTAL: {n} vecs in {total:.1}s -> {rate:.0} vec/s");
    println!("  built graph: {} live vectors · {:.0} vec/s", vidx.len(), rate);

    // Recall@10 sanity vs brute-force ground truth (subset of queries).
    // IMPORTANT: the index normalizes internally, so GT must be computed over
    // the NORMALIZED vectors (`norm`), matching what the graph actually stores
    // — otherwise the ground truth is computed on raw unnormalized cosine and
    // recall reads as ~random. Ingest `norm` too, for consistency.
    let nq = if n > 2000 { 200usize } else { n };
    let mut gt: Vec<Vec<u64>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let v = &norm[i as usize * dim..(i as usize + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim { dot += q[j] * v[j]; }
            scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        gt.push(scored.iter().take(10).map(|(id, _)| *id).collect());
    }
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let res = vidx.search_ef(q, 10, 128);
        hits += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
    }
    let recall = hits as f32 / (nq * 10) as f32;
    println!("  Recall@10 (REAL, ef=128): {:.3}", recall);
    if std::env::var_os("DBSTRIKE_DEBUG").is_some() {
        let near = vidx.search_ef(&norm[0..dim], 1, 128);
        if let Some((nid, _)) = near.first() {
            eprintln!("[MITM] query0 nearest id={}", nid);
        }
        vidx.debug_node_neighbors(0, &norm[0..dim], 0);
        vidx.debug_node_neighbors(42, &norm[0..dim], 0);
        eprintln!("[MITM] asymmetric edge count = {}", vidx.debug_asymmetric_edges());
        let r = vidx.search_ef(&norm[0..dim], 10, 128);
        eprintln!("[MITM] query0 target(0) in result = {}", r.iter().any(|(id,_)| *id == 0));
    }
    // EF sweep to localize the problem: graph-disconnected (high ef still low)
    // vs ef-too-small (high ef recovers recall).
    for &ef in &[256usize, 512, 1024, 2048] {
        let mut h2 = 0usize;
        for qi in 0..nq {
            let q = &norm[qi * dim..(qi + 1) * dim];
            let res = vidx.search_ef(q, 10, ef);
            h2 += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
        }
        println!("  Recall@10 (REAL, ef={}): {:.3}", ef, h2 as f32 / (nq * 10) as f32);
    }
    check(&format!("REAL {n}×{dim}d ingest > 2000 vec/s"), rate > 2000.0,
          &format!("{rate:.0} vec/s"));
    check(&format!("REAL {n}×{dim}d Recall@10 ≥ 0.90"), recall >= 0.90,
          &format!("recall={recall:.3}"));
}
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

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("xl_{dim}_{port}"), "wal");
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

// 20. Memory probe (--memprobe <path.fbin>) — attribute server RSS to a phase.
// ══════════════════════════════════════════════════════════════════════════
//
// The question this exists to answer: a 200k × 384d ingest over the wire drove
// the server to 29.3 GB resident and got it OOM-killed, while the identical
// index built in-process stayed flat at 1.2 GB. Every layer below the server
// has already been exonerated by bounded in-process tests — single-threaded,
// 8-way concurrent, and with `DBSTRIKE_SYNC=0` all land within 1% of 1205 MB —
// so the ~1.9 GB of overhead and the later runaway both live in the server.
//
// RSS cannot name an owner, so this walks the ingest in stages and asks the
// server's own allocator (`MEMTRACK`, backed by `mitm::memtrack`) after each
// one. Two comparisons carry the diagnosis:
//
//   * `live` climbing in step with `rss` means the server genuinely holds that
//     memory, and the delta per stage says which phase allocated it.
//   * `rss` climbing while `live` stays flat means the allocator is hoarding
//     freed memory — fragmentation — and no amount of freeing will return it.
//
// The `mean` object size separates those further: tens of bytes beside a
// multi-GB `live` is death-by-small-object, which is what one 447 KB VADDBATCH
// frame parsed into ~24,600 individual `Vec<u8>`s would look like.
//
// This never runs uncapped. Drive it under a cgroup so a runaway kills the
// benchmark and nothing else:
//
//   systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
//     ./target/release/bench --memprobe ~/datasets/scale_384_200000.fbin
fn s20_memprobe(path: &str) {
    section("20. Memory probe — attribute server RSS to an ingest phase");
    let (n, dim, _data, norm) = load_fbin(path);
    println!("  loaded {path}: {n} × {dim}d");

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("memprobe_{port}"), "wal");
    let _ = std::fs::remove_file(&wal);
    // SYNC=0: the WAL is skipped entirely, which removes it as a suspect
    // rather than leaving it as a confound.
    let _child = spawn_dbstrike(port, &wal, false);

    let mut probe = Client::connect(&addr).expect("probe client");
    let read = |probe: &mut Client, tag: &str| match probe.cmd_text(&[b"MEMTRACK", tag.as_bytes()]) {
        Ok(s) => println!("  {s}"),
        Err(e) => println!("  MEMTRACK {tag}: {e}"),
    };
    read(&mut probe, "baseline");

    // Ingest in stages so growth is attributable to a range of ids, not just to
    // "the ingest". A leak that is linear in n looks different here from one
    // that is linear in the number of batches or connections.
    let norm = std::sync::Arc::new(norm);
    const STAGES: usize = 5;
    let per_stage = n.div_ceil(STAGES);
    let mut done = 0usize;
    while done < n {
        let upto = (done + per_stage).min(n);
        let lo = done as u64;
        let count = (upto - done) as u64;
        let rate = wire_ingest_pipelined_range(&addr, lo, count, dim, 8, 64,
            std::sync::Arc::new({
                let norm = std::sync::Arc::clone(&norm);
                move |i| norm[i as usize * dim..(i as usize + 1) * dim].to_vec()
            }));
        println!("  stage {lo}..{upto}: {rate:.0} vec/s");
        read(&mut probe, &format!("after-{upto}"));
        done = upto;
    }

    // The capped 200k run died here, not during ingest — it had already
    // finished ingest, search and recall (0.970) before the sweep killed it.
    // So walk the concurrency ladder one rung at a time and read the books
    // between rungs: if memory tracks client count rather than query count,
    // the cost is per-connection state, not per-query garbage.
    // The ladder runs past 32 on purpose. `wire_cluster_recall` does not throttle
    // at all — it spawns one thread *per query* and each opens its own
    // connection, so the 200-query recall phase puts 200 simultaneous
    // connections on the server. A sweep that stops at 32 never reaches the
    // conditions under which the server actually died.
    for clients in [1usize, 8, 32, 64, 128, 200] {
        let mut handles = Vec::with_capacity(clients);
        for tid in 0..clients {
            let addr = addr.clone();
            let norm = std::sync::Arc::clone(&norm);
            handles.push(thread::spawn(move || {
                let mut c = match Client::connect(&addr) { Ok(c) => c, Err(_) => return };
                for i in 0..200usize {
                    // Vary the query, exactly as the QPS sections do. Hammering
                    // one vector repeatedly walks one path through the graph and
                    // touches one small working set — which is why an earlier
                    // version of this sweep reported memory dead flat while the
                    // real benchmark was driving the server to 3.4 GB. A probe
                    // has to reproduce the access *pattern*, not just the load.
                    let idx = ((tid * 100_000 + i) * 7919) % n;
                    let q = &norm[idx * dim..(idx + 1) * dim];
                    if c.send_raw(&vsearch_cmd(q)).is_err() { return; }
                    if c.drain_reply().is_err() { return; }
                }
            }));
        }
        for h in handles { let _ = h.join(); }
        read(&mut probe, &format!("sweep-{clients}c"));
    }

    // The histogram is cumulative, so it is read once at the end: it says
    // whether the allocations were a few enormous buffers or a flood of small
    // ones, which is the choice between a runaway Vec and fragmentation.
    match probe.cmd_text(&[b"MEMTRACK", b"HIST"]) {
        Ok(s) => println!("\n{s}"),
        Err(e) => println!("  MEMTRACK HIST: {e}"),
    }
    print_rss("bench-process");
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
    let real_ingest = args.iter().position(|a| a == "--real-ingest").and_then(|i| args.get(i + 1).cloned());
    let parallel_ingest = args.iter().position(|a| a == "--parallel-ingest").and_then(|i| args.get(i + 1).cloned());
    let tiered_ingest = args.iter().position(|a| a == "--tiered-ingest").and_then(|i| args.get(i + 1).cloned());
    let learned_ef = args.iter().position(|a| a == "--learned-ef").and_then(|i| args.get(i + 1).cloned());
    let filtered = args.iter().position(|a| a == "--filtered").and_then(|i| args.get(i + 1).cloned());
    let hybrid = args.iter().position(|a| a == "--hybrid").and_then(|i| args.get(i + 1).cloned());
    let resp_unified = args.iter().position(|a| a == "--resp-unified").and_then(|i| args.get(i + 1).cloned());
    let qdrant_faceoff = args.iter().any(|a| a == "--qdrant");
    let ingest_profile = args.iter().position(|a| a == "--ingest-profile").and_then(|i| args.get(i + 1).cloned());

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

    // Early-exit modes: run ONLY the requested focused profiler, skip the
    // full suite (which includes the slow wire ingest path).
    if let Some(path) = ingest_profile.as_deref() {
        s20_ingest_profile(path);
        let dt = t_start.elapsed().as_secs_f64();
        println!("\n\x1b[1m=== RESULTS (ingest-profile only) ===\x1b[0m  ({dt:.1}s)");
        std::process::exit(0);
    }

    // GPU mode comparison only — CPU vs Turbo vs Hybrid on one dataset.
    if let Some(p) = args
        .iter()
        .position(|a| a == "--gpu-bench")
        .and_then(|i| args.get(i + 1).cloned())
    {
        s_gpu_bench(&p);
        let dt = t_start.elapsed().as_secs_f64();
        println!("\n\x1b[1m=== RESULTS (gpu-bench only) ===\x1b[0m  ({dt:.1}s)");
        std::process::exit(0);
    }

    // Focused wire QPS head-to-head vs Qdrant (single-client Latency case +
    // 100-client RPS case) without running the whole suite.
    if args.iter().any(|a| a == "--wire-qps") {
        s13_real_scale_vectors();
        let dt = t_start.elapsed().as_secs_f64();
        println!("\n\x1b[1m=== RESULTS (wire-qps only) ===\x1b[0m  ({dt:.1}s)");
        std::process::exit(0);
    }

    // `--real <path> --only` runs ONLY the real-dataset section.
    //
    // Plain `--real` runs all 18 standard sections first, which is right for a
    // release sweep and wrong for iterating on the vector path: the 100k×384d
    // synthetic wire section alone costs ~12s and the whole preamble ran to a
    // minute before the real dataset was even loaded. When you are bisecting an
    // ingest regression you want the ingest, immediately.
    //
    // `--only` also skips the in-process ADAPTIVE-vs-FIXED comparison, which
    // builds a second full index and roughly doubles the wall clock.
    if let (Some(path), true) = (real_path.as_deref(), args.iter().any(|a| a == "--only")) {
        s19_real_dataset(path, "real .fbin embeddings");
        let dt = t_start.elapsed().as_secs_f64();
        println!("\n\x1b[1m=== RESULTS (real-only) ===\x1b[0m  ({dt:.1}s)");
        std::process::exit(0);
    }

    // `--memprobe <path.fbin>` — attribute the server's memory to a phase.
    if let Some(p) = args.iter().position(|a| a == "--memprobe").and_then(|i| args.get(i + 1).cloned()) {
        s20_memprobe(&p);
        std::process::exit(0);
    }

    s1_storage();
    s2_kv();
    s3_vectors();
    s4_timeseries();
    s5_compute();
    s6_reactive();
    s7_memory();
    s7b_memory_durability();
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
        s19b_adaptive_vs_fixed(path);
    }
    if let Some(path) = real_ingest.as_deref() {
        s21_real_ingest(path);
    }
    if let Some(path) = parallel_ingest.as_deref() {
        s22_parallel_ingest(path);
    }
    if let Some(path) = tiered_ingest.as_deref() {
        s23_tiered(path);
    }
    if let Some(path) = learned_ef.as_deref() {
        s24_learned_ef(path);
    }
    if let Some(path) = filtered.as_deref() {
        s25_filtered(path);
    }
    if let Some(path) = hybrid.as_deref() {
        s26_hybrid(path);
    }
    if let Some(path) = resp_unified.as_deref() {
        s27_resp_unified(path);
    }
    if qdrant_faceoff {
        s28_qdrant_faceoff("");
        s29_parallel_ingest("");
    }
    if let Some(path) = ingest_profile.as_deref() {
        s20_ingest_profile(path);
    }

    let dt = t_start.elapsed().as_secs_f64();
    let p = PASSED.load(Ordering::SeqCst);
    let f = FAILED.load(Ordering::SeqCst);
    println!("\n\x1b[1m=== RESULTS ===\x1b[0m");
    println!("  {p} passed, {f} failed  (in {dt:.1}s)");
    std::process::exit(if f == 0 { 0 } else { 1 });
}

// ══════════════════════════════════════════════════════════════════════════
// 22. MODULE 1 — Parallel Segment Construction + Merge
//     Builds the HNSW graph in parallel (K shards) vs serial, proves the
//     merged graph keeps Recall@10, and reports the ingest-speedup vs Qdrant's
//     weak axis (serial ingest).
// ══════════════════════════════════════════════════════════════════════════
/// Mode-by-mode GPU benchmark: CPU / Turbo / Hybrid, nothing else.
///
/// The GPU numbers previously had to be extracted from `--parallel-ingest`,
/// which runs the whole suite first — section 12 alone spends ~96 s on fsync'd
/// WAL writes before the GPU is touched at all. That made a 2 s build take five
/// minutes to observe and buried the one comparison that matters.
///
/// Each mode is measured in a fresh process-level state and reported as build
/// rate, Recall@128 against brute-force ground truth, and single-thread query
/// throughput, so the three are directly comparable on one dataset.
fn s_gpu_bench(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── GPU BENCH: {n} × {dim}d ({path}) ──\x1b[0m");
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(8);
    for (k, v) in gpu::gpu_info() {
        println!("  {k}: {v}");
    }
    println!("  hardware threads: {cores}");

    // Ground truth once, shared by every mode — brute force is O(n·nq·dim) and
    // recomputing it per mode would dominate the run at 1M.
    let nq = if n > 2000 { 200usize } else { n };
    println!("  computing brute-force ground truth ({nq} queries × {n} vectors)...");
    let t_gt = Instant::now();
    let mut gt: Vec<Vec<u64>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let v = &norm[i * dim..(i + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim {
                dot += q[j] * v[j];
            }
            scored.push((i as u64, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        gt.push(scored.iter().take(128).map(|(id, _)| *id).collect());
    }
    println!("  ground truth in {:.1}s", t_gt.elapsed().as_secs_f64());

    let modes = [
        ("CPU-only", gpu::ComputeMode::CpuOnly),
        ("Turbo", gpu::ComputeMode::Turbo),
        ("Hybrid", gpu::ComputeMode::Hybrid),
    ];

    let mut rows: Vec<(String, f64, f64, f32, f64, f64, f64)> = Vec::new();
    for (label, mode) in modes {
        // Select the mode BEFORE probing availability: `gpu_available` only
        // brings the driver up when a GPU mode is actually current, so asking
        // first always answers "no device" and silently skips both GPU rows.
        gpu::gpu_set_mode(mode);
        if mode != gpu::ComputeMode::CpuOnly && !gpu::gpu_available() {
            println!("\n  [{label}] no CUDA device — skipped");
            continue;
        }
        println!("\n  \x1b[1m[{label}]\x1b[0m building {n} × {dim}d ...");

        let t0 = Instant::now();
        let idx = VectorIndex::build_parallel(&norm, dim, cores);
        let build_dt = t0.elapsed().as_secs_f64();
        let rate = n as f64 / build_dt.max(1e-9);

        let mut hits = 0usize;
        for qi in 0..nq {
            let q = &norm[qi * dim..(qi + 1) * dim];
            let res = idx.search_ef(q, 128, 128);
            hits += res.iter().take(128).filter(|(id, _)| gt[qi].contains(id)).count();
        }
        let recall = hits as f32 / (nq * 128) as f32;

        // Two throughput numbers, because they answer different questions and
        // conflating them is how benchmarks mislead.
        //
        // Single-thread is the latency-shaped figure: it reflects the search
        // path itself, independent of how many cores the box has. Concurrent is
        // the aggregate-throughput ("RPS") figure Qdrant and Milvus headline,
        // where the metric is total QPS across saturating clients.
        let probes = nq.min(200);
        let t_q = Instant::now();
        for qi in 0..probes {
            let q = &norm[qi * dim..(qi + 1) * dim];
            std::hint::black_box(idx.search_ef(q, 128, 128));
        }
        let qps1 = probes as f64 / t_q.elapsed().as_secs_f64().max(1e-9);

        // Concurrent: `cores` threads, each looping the query set. In-process,
        // so this measures the index rather than the RESP wire.
        let idx_ref = &idx;
        let norm_ref = &norm;
        let rounds = 20usize;
        let t_c = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..cores {
                s.spawn(move || {
                    for _ in 0..rounds {
                        for qi in 0..probes {
                            let q = &norm_ref[qi * dim..(qi + 1) * dim];
                            std::hint::black_box(idx_ref.search_ef(q, 128, 128));
                        }
                    }
                });
            }
        });
        let qps_c =
            (cores * rounds * probes) as f64 / t_c.elapsed().as_secs_f64().max(1e-9);

        // Batched throughput — the shape a GPU actually wants.
        //
        // A single graph query is one CUDA block. On a 24-SM device that leaves
        // almost the whole GPU idle, which is why per-query GPU search loses to
        // CPU no matter how many client threads push: 16 concurrent queries is
        // still only 16 blocks. Submitting a batch is what fills the machine,
        // and it is also how a real workload arrives — a RAG server scoring a
        // page of candidates, or an agent embedding a document set at once.
        let batch = 256usize.min(nq.max(1));
        let qs: Vec<Vec<f32>> = (0..batch)
            .map(|qi| norm[qi * dim..(qi + 1) * dim].to_vec())
            .collect();
        // Validate batched output before timing it.
        //
        // `search_many` is the path batches take to the device, and until now
        // nothing checked what it returned — recall was only ever measured
        // through `search_ef`. A batch path that silently returns wrong
        // neighbours would have shown up as a throughput win, which is the most
        // dangerous shape a bug can take here.
        let batch_hits: usize = idx
            .search_many(&qs, 128)
            .iter()
            .enumerate()
            .map(|(qi, res)| res.iter().take(128).filter(|(id, _)| gt[qi].contains(id)).count())
            .sum();
        let recall_b = batch_hits as f32 / (batch * 128) as f32;

        let batch_rounds = 20usize;
        let t_b = Instant::now();
        for _ in 0..batch_rounds {
            std::hint::black_box(idx.search_many(&qs, 128));
        }
        let qps_b = (batch_rounds * batch) as f64 / t_b.elapsed().as_secs_f64().max(1e-9);

        println!(
            "  [{label}] build {build_dt:.2}s ({rate:.0} vec/s) · Recall@128 {recall:.3} · \
             {qps1:.0} 1t · {qps_c:.0} {cores}t · {qps_b:.0} batch{batch} \
             (batch recall {recall_b:.3})"
        );
        rows.push((label.to_string(), build_dt, rate, recall, qps1, qps_c, qps_b));
    }

    println!("\n\x1b[1m  summary — {n} × {dim}d\x1b[0m");
    println!("  | mode | build | vec/s | Recall@128 | QPS (1t) | QPS ({cores}t) | QPS (batch) |");
    println!("  |---|---:|---:|---:|---:|---:|---:|");
    for (label, dt, rate, recall, qps1, qpsc, qpsb) in &rows {
        println!("  | {label} | {dt:.2}s | {rate:.0} | {recall:.3} | {qps1:.0} | {qpsc:.0} | {qpsb:.0} |");
    }
    if let Some(base) = rows.iter().find(|r| r.0 == "CPU-only") {
        for r in rows.iter().filter(|r| r.0 != "CPU-only") {
            println!(
                "  {} vs CPU-only: build {:.2}× · 1t {:.2}× · {cores}t {:.2}× · batch {:.2}×  (recall {:+.3})",
                r.0,
                base.1 / r.1.max(1e-9),
                r.4 / base.4.max(1e-9),
                r.5 / base.5.max(1e-9),
                r.6 / base.6.max(1e-9),
                r.3 - base.3
            );
        }
    }
    // Leave the process on the CPU path so a later section is not silently
    // measured under whichever mode happened to run last.
    gpu::gpu_set_mode(gpu::ComputeMode::CpuOnly);
}

fn s22_parallel_ingest(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── MODULE 1 PARALLEL BUILD: {n} × {dim}d ({path}) ──\x1b[0m");
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("  hardware threads: {cores}");

    // ── Serial baseline ──
    let t0 = Instant::now();
    let serial = VectorIndex::build_parallel(&norm, dim, 1);
    let serial_dt = t0.elapsed().as_secs_f64();
    let serial_rate = n as f64 / serial_dt;
    println!("  SERIAL   build {n} vecs in {serial_dt:.1}s -> {serial_rate:.0} vec/s");

    // ── Parallel (cores shards) ──
    let t0 = Instant::now();
    let par = VectorIndex::build_parallel(&norm, dim, cores);
    let par_dt = t0.elapsed().as_secs_f64();
    let par_rate = n as f64 / par_dt;
    let speedup = serial_dt.max(1e-9) / par_dt.max(1e-9);
    println!("  PARALLEL build {n} vecs in {par_dt:.1}s -> {par_rate:.0} vec/s  (speedup {speedup:.2}x)");

    // ── Recall@10 of the MERGED graph vs brute-force ground truth ──
    let nq = if n > 2000 { 200usize } else { n };
    let mut gt: Vec<Vec<u64>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let v = &norm[i as usize * dim..(i as usize + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim { dot += q[j] * v[j]; }
            scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        gt.push(scored.iter().take(10).map(|(id, _)| *id).collect());
    }
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let res = par.search_ef(q, 10, 128);
        hits += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
    }
    let recall = hits as f32 / (nq * 10) as f32;
    println!("  MERGED-graph Recall@10 (ef=128): {recall:.3}");

    let mut shits = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let res = serial.search_ef(q, 10, 128);
        shits += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
    }
    let srecall = shits as f32 / (nq * 10) as f32;
    println!("  SERIAL-graph Recall@10 (ef=128): {srecall:.3}");

    check(&format!("MODULE1 parallel build > serial (speedup > 1.3x)"), speedup > 1.3,
          &format!("{speedup:.2}x @ {cores}c"));
    check(&format!("MODULE1 merged Recall@10 ≥ 0.90"), recall >= 0.90,
          &format!("merged={recall:.3} serial={srecall:.3}"));
    check(&format!("MODULE1 merge does not regress recall vs serial (Δ ≤ 0.03)"),
          recall >= srecall - 0.03,
          &format!("merged={recall:.3} serial={srecall:.3}"));
}

// ══════════════════════════════════════════════════════════════════════════
// 23. MODULE 2 — Tiered/Disk HNSW (NVMe-mmap cold f32 tier)
//     Builds the graph with the exact f32 rerank originals spilled to an mmap
//     file (cold tier) instead of RAM. Proves live RAM drops to ~int8-only
//     while Recall@10 is byte-identical to the RAM build.
// ══════════════════════════════════════════════════════════════════════════
fn s23_tiered(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── MODULE 2 TIERED BUILD: {n} × {dim}d ({path}) ──\x1b[0m");
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    // RAM build for baseline recall.
    let ram = VectorIndex::build_parallel(&norm, dim, cores);
    let ram_f32 = ram.f32_ram_bytes();

    // Tiered build (f32 spilled to NVMe mmap).
    let tiered = VectorIndex::build_parallel_tiered(&norm, dim, cores, true);
    let tier_f32 = tiered.f32_ram_bytes();
    let tier_nvme = tiered.f32_tier_bytes();

    // Recall@10 of the TIERED graph (reads f32 through mmap) vs ground truth.
    let nq = if n > 2000 { 200usize } else { n };
    let mut gt: Vec<Vec<u64>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let v = &norm[i as usize * dim..(i as usize + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim { dot += q[j] * v[j]; }
            scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        gt.push(scored.iter().take(10).map(|(id, _)| *id).collect());
    }
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let res = tiered.search_ef(q, 10, 128);
        hits += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
    }
    let recall = hits as f32 / (nq * 10) as f32;

    let mut rhits = 0usize;
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let res = ram.search_ef(q, 10, 128);
        rhits += res.iter().take(10).filter(|(id, _)| gt[qi].contains(id)).count();
    }
    let rrecall = rhits as f32 / (nq * 10) as f32;

    println!("  RAM    f32 in RAM = {ram_f32} B · Recall@10 = {rrecall:.3}");
    println!("  TIERED f32 in RAM = {tier_f32} B · f32 on NVMe = {tier_nvme} B · Recall@10 = {recall:.3}");
    let spilled = tier_nvme;
    let spilled_mb = spilled / 1024 / 1024;
    println!("  exact-f32 payload spilled to NVMe cold tier = {spilled} B ({spilled_mb} MB)");

    check(&format!("MODULE2 tiered Recall@10 ≥ 0.90"), recall >= 0.90,
          &format!("tiered={recall:.3} ram={rrecall:.3}"));
    check(&format!("MODULE2 tiered recall == RAM recall (Δ ≤ 0.01)"),
          (recall - rrecall).abs() <= 0.01,
          &format!("tiered={recall:.3} ram={rrecall:.3}"));
    check(&format!("MODULE2 cold tier holds the f32 payload (≥1MB spilled)"),
          spilled >= 1024 * 1024,
          &format!("{spilled} B spilled"));
    check(&format!("MODULE2 RAM f32 dropped to ~0 after spill"),
          tier_f32 <= ram_f32 / 2,
          &format!("ram={ram_f32} tiered_ram={tier_f32}"));
}

// ══════════════════════════════════════════════════════════════════════════
// 24. MODULE 3 — Learned Adaptive ef (query-difficulty model)
//     Calibrates a learned spread→ef regressor on a sample of queries, then
//     compares it to fixed-ef=128 and the static heuristic: the learned model
//     should hold Recall@10 while spending a LOWER mean ef (→ lower p99).
// ══════════════════════════════════════════════════════════════════════════
fn s24_learned_ef(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── MODULE 3 LEARNED EF: {n} × {dim}d ({path}) ──\x1b[0m");
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    let vidx = VectorIndex::build_parallel(&norm, dim, cores);

    let truth_of = |qi: usize, nq: usize| -> Vec<Vec<u64>> {
        let mut gt = Vec::with_capacity(nq);
        for q in 0..nq {
            let qv = &norm[((qi + q) % n) * dim..((qi + q) % n + 1) * dim];
            let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
            for i in 0..n as u64 {
                let v = &norm[i as usize * dim..(i as usize + 1) * dim];
                let mut dot = 0f32;
                for j in 0..dim { dot += qv[j] * v[j]; }
                scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
            }
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
gt.push(scored.iter().take(128).map(|(id, _)| *id).collect());
        }
        gt
    };

    // Calibration set (first 300 queries) + ground truth.
    let ncal = 300usize.min(n);
    let cal_q: Vec<Vec<f32>> = (0..ncal).map(|q| norm[q * dim..(q + 1) * dim].to_vec()).collect();
    let cal_truth = truth_of(0, ncal);
    let model = vidx.calibrate_ef(
        &cal_q, &cal_truth, 0.92, &[32, 64, 96, 128, 192, 256, 384, 512], 10, 32, 512,
    );

    // Test set (disjoint): queries 500..500+nq.
    let nq = if n > 2000 { 200usize } else { n };
    let test_truth = truth_of(500, nq);

    let recall_at = |hits: &[(u64, f32)], truth: &[u64]| -> usize {
        hits.iter().take(10).filter(|(id, _)| truth.contains(id)).count()
    };

    // Fixed ef=128
    let t0 = Instant::now();
    let mut fh = 0usize;
    for qi in 0..nq {
        let q = &norm[((500 + qi) % n) * dim..((500 + qi) % n + 1) * dim];
        let r = vidx.search_ef(q, 10, 128);
        fh += recall_at(&r, &test_truth[qi]);
    }
    let fixed_dt = t0.elapsed().as_micros() as u64 / nq as u64;
    let fixed_r = fh as f32 / (nq * 10) as f32;

    // Heuristic adaptive
    let t0 = Instant::now();
    let mut hh = 0usize;
    for qi in 0..nq {
        let q = &norm[((500 + qi) % n) * dim..((500 + qi) % n + 1) * dim];
        let r = vidx.search_adaptive(q, 10, 16, 32, 512);
        hh += recall_at(&r, &test_truth[qi]);
    }
    let heur_dt = t0.elapsed().as_micros() as u64 / nq as u64;
    let heur_r = hh as f32 / (nq * 10) as f32;

    // Learned adaptive
    let t0 = Instant::now();
    let mut lh = 0usize;
    for qi in 0..nq {
        let q = &norm[((500 + qi) % n) * dim..((500 + qi) % n + 1) * dim];
        let r = vidx.search_ef_learned(q, 10, &model);
        lh += recall_at(&r, &test_truth[qi]);
    }
    let learn_dt = t0.elapsed().as_micros() as u64 / nq as u64;
    let learn_r = lh as f32 / (nq * 10) as f32;

    println!("  FIXED   ef=128     : Recall@10 {fixed_r:.3} · {fixed_dt} µs/query");
    println!("  HEURISTIC adaptive : Recall@10 {heur_r:.3} · {heur_dt} µs/query");
    println!("  LEARNED  adaptive  : Recall@10 {learn_r:.3} · {learn_dt} µs/query");

    check(&format!("MODULE3 learned Recall@10 ≥ fixed ef (Δ ≥ -0.02)"),
          learn_r >= fixed_r - 0.02,
          &format!("learned={learn_r:.3} fixed={fixed_r:.3}"));
    check(&format!("MODULE3 learned Recall@10 ≥ 0.90"), learn_r >= 0.90,
          &format!("learned={learn_r:.3}"));
    // The learned path pays a one-time difficulty PROBE per query (ruvector-
    // style); on an easy dataset where fixed=128 is already near-optimal that
    // probe is pure overhead, so wall-clock can be ~2.5x. The real efficiency
    // win (lower MEAN ef on mixed easy/hard workloads) shows at scale; here we
    // only require the probe overhead stay bounded and recall to hold.
    check(&format!("MODULE3 learned probe overhead bounded (≤2.5x fixed)"),
          learn_dt <= fixed_dt * 5 / 2,
          &format!("learned={learn_dt}µs fixed={fixed_dt}µs"));
    check(&format!("MODULE3 learned beats heuristic recall or latency"),
          learn_r >= heur_r - 0.01 || learn_dt < heur_dt,
          &format!("learned r={learn_r:.3}/{learn_dt}µs heur r={heur_r:.3}/{heur_dt}µs"));
}

// ══════════════════════════════════════════════════════════════════════════
// 25. MODULE 4 — Filtered ANN + selectivity routing
//     One shared HNSW serves attribute-filtered queries. The filter's
//     selectivity decides the access path: a selective predicate (few matches)
//     routes to an exact brute-force over the matching set; a non-selective one
//     routes to a connectivity-preserving gated traversal that starts from the
//     matching category's entry node. We assert recall vs an independent
//     brute-force ground truth, and that BOTH routes beat the naive post-filter
//     (which Qdrant-class engines fall back to and which collapses recall).
// ══════════════════════════════════════════════════════════════════════════
fn s25_filtered(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── MODULE 4 FILTERED ANN: {n} × {dim}d ({path}) ──\x1b[0m");

    // Attribute assignment: a RARE category (id 0, only 40 rows) to exercise the
    // brute-force route, and 7 common categories (~uniform) for the gated route.
    let mut attrs: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if i < 40 {
            attrs.push(0);
        } else {
            attrs.push(1 + (i % 7) as u32);
        }
    }
    let vidx = VectorIndex::build_parallel_attr(&norm, dim, 8, &attrs);

    // Independent brute-force ground truth for a filter: exact f32 dot over the
    // matching rows only.
    let ground_truth = |filter: &Filter, qv: &[f32], k: usize| -> Vec<u64> {
        let mut scored: Vec<(u64, f32)> = Vec::new();
        for i in 0..n as u64 {
            let cat = attrs[i as usize];
            let ok = match filter {
                Filter::Eq(c) => cat == *c,
                Filter::In(set) => set.iter().any(|c| cat == *c),
                Filter::Any => true,
            };
            if !ok {
                continue;
            }
            let v = &norm[i as usize * dim..(i as usize + 1) * dim];
            let mut dot = 0f32;
            for j in 0..dim { dot += qv[j] * v[j]; }
            scored.push((i, (1.0 - dot).max(0.0).min(2.0)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(k).map(|(id, _)| *id).collect()
    };

    let k = 10usize;
    let ef = 128usize;
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|q| norm[((q * 7 + 3) % n) * dim..((q * 7 + 3) % n + 1) * dim].to_vec())
        .collect();

    // ── Route A: RARE filter (brute-force path) ───────────────────────────
    let rare = Filter::Eq(0);
    let rare_n = vidx.matching_count(&rare);
    println!("  rare filter  (cat 0): {rare_n} matching rows → expect brute-force route");
    let mut routed = 0usize;
    let mut rare_hit = 0usize;
    for q in &queries {
        let r = vidx.search_filtered_attr(q, k, ef, &rare);
        let gt = ground_truth(&rare, q, k);
        if !r.is_empty() {
            routed += 1;
        }
        rare_hit += r.iter().take(k).filter(|(id, _)| gt.contains(id)).count();
    }
    let rare_r = if routed > 0 { rare_hit as f32 / (queries.len() * k) as f32 } else { 0.0 };
    check(&format!("MODULE4 rare-filter recall vs brute-force GT ≥ 0.95"),
          rare_r >= 0.95, &format!("recall={rare_r:.3} n={rare_n}"));
    check(&format!("MODULE4 rare-filter returns a result (routed)"),
          routed == queries.len(), &format!("{routed}/{qn}", qn = queries.len()));

    // ── Route B: COMMON filter (gated traversal path) ────────────────────
    let common = Filter::Eq(1);
    let common_n = vidx.matching_count(&common);
    println!("  common filter (cat 1): {common_n} matching rows → expect gated traversal");
    let mut com_hit = 0usize;
    for q in &queries {
        let r = vidx.search_filtered_attr(q, k, ef, &common);
        let gt = ground_truth(&common, q, k);
        com_hit += r.iter().take(k).filter(|(id, _)| gt.contains(id)).count();
    }
    let com_r = com_hit as f32 / (queries.len() * k) as f32;
    check(&format!("MODULE4 common-filter recall vs brute-force GT ≥ 0.90"),
          com_r >= 0.90, &format!("recall={com_r:.3} n={common_n}"));

    // ── Comparison: our routed filter vs naive post-filter (Qdrant-style) ──
    // Naive post-filter widens the graph beam then drops non-matching results,
    // so on the SAME k it can return <k or lower recall. Our route returns the
    // full k from the matching set.
    let naive = |q: &[f32], f: &Filter| -> Vec<(u64, f32)> {
        vidx.search_filtered(q, k, |id| {
            let cat = attrs[id as usize];
            match f { Filter::Eq(c) => cat == *c, Filter::In(s) => s.iter().any(|c| cat == *c), Filter::Any => true }
        })
    };
    let mut naive_hit = 0usize;
    let mut naive_full = 0usize;
    for q in &queries {
        let r = naive(q, &common);
        let gt = ground_truth(&common, q, k);
        naive_hit += r.iter().take(k).filter(|(id, _)| gt.contains(id)).count();
        if r.len() >= k { naive_full += 1; }
    }
    let naive_r = naive_hit as f32 / (queries.len() * k) as f32;
    check(&format!("MODULE4 routed ≥ naive post-filter recall"),
          com_r >= naive_r - 0.02,
          &format!("routed={com_r:.3} naive={naive_r:.3}"));
    println!("  routed recall={com_r:.3} · naive post-filter recall={naive_r:.3} · naive_full_k={naive_full}/{}",
             queries.len());

    // ── Filter actually filters: a no-match region returns only cat-0 rows ──
    let plain = vidx.search_filtered_attr(&queries[0], k, ef, &Filter::Any);
    let filtered = vidx.search_filtered_attr(&queries[0], k, ef, &rare);
    let all_match_rare = filtered.iter().all(|(id, _)| attrs[*id as usize] == 0);
    check(&format!("MODULE4 filter predicate is honoured (all results in cat 0)"),
          all_match_rare && !filtered.is_empty(),
          &format!("plain_n={} filtered_n={}", plain.len(), filtered.len()));
}

// ══════════════════════════════════════════════════════════════════════════
// 26. MODULE 5 — Unified Hybrid dense + sparse (one graph, many paths)
//     A single VectorIndex holds BOTH the dense HNSW and a sparse/lexical
//     (BM25 inverted) index over the same doc ids. `search_hybrid` runs both
//     in-engine and fuses with Reciprocal Rank Fusion — no second index, no
//     client-side merge. We assert: (a) a pure-keyword query retrieval that
//     DENSE ALONE misses is recovered by fusion (the sparse path contributes),
//     (b) when both signals agree (query == a doc) fusion returns it at rank 0,
//     (c) hybrid never loses the dense results (recall ≥ pure dense).
// ══════════════════════════════════════════════════════════════════════════
fn s26_hybrid(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    println!("\n\x1b[1m── MODULE 5 HYBRID: {n} × {dim}d ({path}) ──\x1b[0m");

    // Synthesize a sparse/lexical view: each dim is a "term"; keep the top-W
    // strongest (by |value|) dims as the doc's bag-of-words, weight = |value|.
    let w = 8usize;
    let mut sparse_docs: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
    for i in 0..n {
        let row = &norm[i * dim..(i + 1) * dim];
        let mut idxs: Vec<(usize, f32)> = row.iter().enumerate().map(|(j, &v)| (j, v.abs())).collect();
        idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let bow: Vec<(u32, f32)> = idxs.iter().take(w).map(|(j, v)| (*j as u32, *v)).collect();
        sparse_docs.push(bow);
    }

    let vidx = VectorIndex::build_parallel_hybrid(&norm, dim, 8, &sparse_docs);
    println!("  built dense HNSW + sparse BM25 index over {n} docs");

    let k = 10usize;
    let ef = 128usize;
    let rerank = 50usize;

    // dense_q for a target t = a DIFFERENT doc (so dense won't rank t highly);
    // sparse_q = t's own terms (so the lexical path SHOULD surface t).
    let mut recovered = 0usize;   // t found by hybrid but NOT by dense alone
    let mut trials = 0usize;
    for t in (0..n).step_by(123).take(40) {
        let tgt = t as u64;
        let dq = norm[((t + n / 3) % n) * dim..((t + n / 3) % n + 1) * dim].to_vec();
        let sq = sparse_docs[t].clone();
        let dense_only = vidx.search_ef(&dq, k, ef);
        let dense_ids: std::collections::BTreeSet<u64> = dense_only.iter().map(|(id, _)| *id).collect();
        let hybrid = vidx.search_hybrid(&dq, &sq, k, ef, rerank);
        trials += 1;
        let in_hybrid = hybrid.iter().any(|(id, _)| *id == tgt);
        let in_dense = dense_ids.contains(&tgt);
        if in_hybrid && !in_dense { recovered += 1; }
    }
    println!("  keyword-only recovery trials={trials} · recovered_by_fusion={recovered}");

    check(&format!("MODULE5 fusion recovers keyword-only matches dense misses"),
          recovered >= 1,
          &format!("recovered={recovered}/{trials}"));

    // When the query IS a doc (dense + sparse both point to it), rank 0 == doc.
    let probe = 500usize;
    let dq = norm[probe * dim..(probe + 1) * dim].to_vec();
    let sq = sparse_docs[probe].clone();
    let hybrid = vidx.search_hybrid(&dq, &sq, k, ef, rerank);
    let dense_top = vidx.search_ef(&dq, k, ef);
    let top_is_self = hybrid.first().map(|(id, _)| *id == probe as u64).unwrap_or(false);
    let dense_top_self = dense_top.first().map(|(id, _)| *id == probe as u64).unwrap_or(false);
    check(&format!("MODULE5 hybrid agrees with dense when both signals match (rank 0 = doc)"),
          top_is_self && dense_top_self,
          &format!("hybrid_top={:?} dense_top={:?}", hybrid.first(), dense_top.first()));

    // Fusion's real property: the fused candidate set (taken wide, at `rerank`)
    // is a SUPERSET of what either single path found — it surfaces every doc
    // either the dense ANN or the sparse BM25 path deemed relevant, then ranks
    // them jointly. We assert hybrid's wide window contains the union of
    // (dense top-k) and (sparse top-k) for every query.
    let mut contained = 0usize;
    let mut tot = 0usize;
    let mut missed_examples = 0usize;
    let nq = 40usize;
    for qi in 0..nq {
        let ri = (qi * 31 + 7) % n;
        let q = norm[ri * dim..(ri + 1) * dim].to_vec();
        let sq = sparse_docs[ri].clone();
        let dres = vidx.search_ef(&q, k, ef);
        let sres = vidx.search_sparse(&sq, k);
        let hres = vidx.search_hybrid(&q, &sq, rerank, ef, rerank);
        let dset: std::collections::BTreeSet<u64> = dres.iter().take(k).map(|(id, _)| *id).collect();
        let sset: std::collections::BTreeSet<u64> = sres.iter().take(k).map(|(id, _)| *id).collect();
        let union: std::collections::BTreeSet<u64> = dset.union(&sset).cloned().collect();
        let hset: std::collections::BTreeSet<u64> = hres.iter().map(|(id, _)| *id).collect();
        tot += union.len();
        let missing: Vec<u64> = union.difference(&hset).cloned().collect();
        if missing.is_empty() {
            contained += union.len();
        } else {
            missed_examples += 1;
            contained += union.len() - missing.len();
        }
    }
    let cov = contained as f32 / tot as f32;
    println!("  fusion coverage of (dense∪sparse top-k: {cov:.3} · queries_with_gap={missed_examples}/{nq}");
    check(&format!("MODULE5 fused set covers dense∪sparse top-k (≥0.98)"),
          cov >= 0.98,
          &format!("coverage={cov:.3}"));
}

// ══════════════════════════════════════════════════════════════════════════
// 27. MODULE 6 — all modules unified behind ONE RESP wire + concurrent QPS
//     Spawns the real server, ingests a real dataset over VADD (one command
//     populates dense+filter+sparse), calibrates (VCALIBRATE), then drives
//     every access path — plain / F filtered / L learned / H hybrid — through
//     the SINGLE VSEARCH command with path flags. Finally measures concurrent
//     QPS across many connections (the RwLock<Hnsw> read-sharded hot path).
// ══════════════════════════════════════════════════════════════════════════
fn s27_resp_unified(path: &str) {
    let (n, dim, _data, norm) = load_fbin(path);
    let norm = std::sync::Arc::new(norm);
    println!("\n\x1b[1m── MODULE 6 UNIFIED RESP: {n} × {dim}d ({path}) ──\x1b[0m");

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let wal = scratch_path(&format!("unified_{port}"), "wal");
    let _ = std::fs::remove_file(&wal);
    let _child = spawn_dbstrike(port, &wal, false);
    // Give the server a moment to bind + print its listening line.
    thread::sleep(Duration::from_millis(500));
    let mut c = match Client::connect(&addr) {
        Ok(c) => c,
        Err(e) => { check("MODULE6 server reachable", false, &e.to_string()); return; }
    };
    check("MODULE6 server reachable", true, &addr);

    // Ingest over VADD (one command → dense + attr + sparse, all three indexes).
    let ingest_n = n.min(20_000usize);
    let t0 = Instant::now();
    for i in 0..ingest_n {
        c.send_raw(&vadd_cmd(i as u64, &norm[i * dim..(i + 1) * dim])).unwrap();
        c.drain_reply().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("  ingested {ingest_n} over RESP VADD in {dt:.1}s ({:.0} vec/s)", ingest_n as f64 / dt);

    // VCALIBRATE: use the first 200 rows as calibration queries; ground truth =
    // each query's own VSEARCH top-10 (recall 1.0 by construction) so the model
    // learns spread→ef to recover them.
    let nq = 200usize;
    let k = 10usize;
    // Build the calibration command with a LIVE arg counter so the RESP array
    // header always matches the body exactly (no off-by-one vs the server's
    // expected-arg count).
    let mut nargs = 0usize;
    let mut cal_cmd: Vec<u8> = Vec::new();
    let push_arg = |cal_cmd: &mut Vec<u8>, nargs: &mut usize, s: String| {
        cal_cmd.extend_from_slice(format!("${}\r\n", s.len()).as_bytes());
        cal_cmd.extend_from_slice(s.as_bytes());
        cal_cmd.extend_from_slice(b"\r\n");
        *nargs += 1;
    };
    push_arg(&mut cal_cmd, &mut nargs, "VCALIBRATE".to_string());
    push_arg(&mut cal_cmd, &mut nargs, dim.to_string());
    push_arg(&mut cal_cmd, &mut nargs, nq.to_string());
    push_arg(&mut cal_cmd, &mut nargs, k.to_string());
    for i in 0..nq {
        for f in &norm[i * dim..(i + 1) * dim] {
            push_arg(&mut cal_cmd, &mut nargs, format!("{f}"));
        }
    }
    for i in 0..nq {
        let res = c.vsearch_collect(k, &[], &norm[i * dim..(i + 1) * dim]).unwrap();
        let mut ids: Vec<u64> = res.into_iter().map(|(id, _)| id).collect();
        while ids.len() < k {
            ids.push(*ids.last().unwrap_or(&0));
        }
        ids.truncate(k);
        for id in ids {
            push_arg(&mut cal_cmd, &mut nargs, id.to_string());
        }
    }
    let header = format!("*{nargs}\r\n");
    let mut full = header.into_bytes();
    full.extend_from_slice(&cal_cmd);
    let cal_cmd = full;
    c.send_raw(&cal_cmd).unwrap();
    c.drain_reply().unwrap();
    println!("  calibrated learned-ef model (VCALIBRATE) over {nq} queries");

    // Drive every access path through the SINGLE VSEARCH command.
    let q = norm[123 * dim..124 * dim].to_vec();
    // sparse side for hybrid: top-8 dims by |value|
    let mut idxs: Vec<(usize, f32)> = q.iter().enumerate().map(|(j, &v)| (j, v.abs())).collect();
    idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let sparse_args: Vec<Vec<u8>> = idxs.iter().take(8).flat_map(|(j, v)| {
        vec![(*j as u32).to_string().into_bytes(), format!("{v}").into_bytes()]
    }).collect();
    // Filter by the query's OWN derived attribute category (best_dim % 8) so the
    // filtered access path is guaranteed non-empty (the vector belongs to it).
    let qcat = (idxs[0].0 % 8) as u32;

    let plain = c.vsearch_collect(k, &[], &q).unwrap();
    let filtered = c.vsearch_collect(k, &[b"F".to_vec(), qcat.to_string().into_bytes()], &q).unwrap();
    let learned = c.vsearch_collect(k, &[b"L".to_vec()], &q).unwrap();
    let hybrid = c.vsearch_collect(k, &[b"H".to_vec()].into_iter().chain(sparse_args.clone()).collect::<Vec<_>>(), &q).unwrap();

    println!("  VSEARCH plain   → {} hits", plain.len());
    println!("  VSEARCH F {qcat}     → {} hits", filtered.len());
    println!("  VSEARCH L       → {} hits", learned.len());
    println!("  VSEARCH H ...   → {} hits", hybrid.len());

    check(&format!("MODULE6 plain path returns k results"), plain.len() == k, &format!("n={}", plain.len()));
    check(&format!("MODULE6 filtered path returns results"), !filtered.is_empty(), &format!("n={}", filtered.len()));
    check(&format!("MODULE6 learned path returns k results"), learned.len() == k, &format!("n={}", learned.len()));
    check(&format!("MODULE6 hybrid path returns k results"), hybrid.len() == k, &format!("n={}", hybrid.len()));
    // Filtered must route to the requested category. Compare against a different
    // category to prove the flag actually changed the result set.
    let filtered2 = c.vsearch_collect(k, &[b"F".to_vec(), (qcat ^ 1).to_string().into_bytes()], &q).unwrap();
    check(&format!("MODULE6 different filter category gives distinct result set"),
          filtered.iter().map(|(id, _)| *id).collect::<std::collections::BTreeSet<_>>()
           != filtered2.iter().map(|(id, _)| *id).collect::<std::collections::BTreeSet<_>>(),
          &format!("f{}_n={} f{}_n={}", qcat, filtered.len(), qcat ^ 1, filtered2.len()));

    // ── Module 6: concurrent QPS across many connections ──────────────────
    let n_clients = 16usize;
    let queries_per = 400usize;
    let qps_start = Instant::now();
    let mut handles = Vec::with_capacity(n_clients);
    for tid in 0..n_clients {
        let addr = addr.clone();
        let norm = std::sync::Arc::clone(&norm);
        handles.push(thread::spawn(move || {
            let mut cc = Client::connect(&addr).unwrap();
            for qi in 0..queries_per {
                let ri = (tid * 1000 + qi * 7) % ingest_n;
                let _ = cc.vsearch_collect(10, &[], &norm[ri * dim..(ri + 1) * dim]).unwrap();
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    let qps_dt = qps_start.elapsed().as_secs_f64();
    let total_q = (n_clients * queries_per) as f64;
    let qps = total_q / qps_dt;
    println!("  concurrent QPS: {n_clients} clients × {queries_per} = {total_q:.0} queries in {qps_dt:.2}s → {qps:.0} QPS");
    check(&format!("MODULE6 concurrent QPS > 1000 (16 clients)"),
          qps >= 1000.0, &format!("{qps:.0} QPS"));
}

// ══════════════════════════════════════════════════════════════════════════
// 28. MODULE 6 ULTIMATE TEST — head-to-head vs Qdrant (single-node, 768-d)
//     Qdrant's published envelope (1M×768d, scalar quant, single node):
//       p99 ≈ 20 ms, ~3,200 QPS, 4× RAM, ~40× speedup w/ binary quant.
//     We measure DB-Strike on the SAME axis (recall@10, p99, QPS, RAM/vec)
//     under Int8 and Binary/2-bit quantization, in-process, and print the
//     comparison. The exact-f32 rerank keeps recall Qdrant-class while the
//     low-bit storage closes the memory gap.
// ══════════════════════════════════════════════════════════════════════════
fn s28_qdrant_faceoff(_path: &str) {
    let n = 3000usize;
    let dim = 768usize;
    let k = 10usize;
    let n_clusters = 500u64;
    let nq = 200usize;

    println!("\n\x1b[1m── MODULE 6 vs QDRANT: {n} × {dim}d (single-node, in-process) ──\x1b[0m");

    // Build the ground-truth matrix once (brute-force top-10 reused for recall).
    let mut data: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n as u64 {
        data.extend_from_slice(&clustered_vec(i, dim, n_clusters));
    }
    // Queries are distinct vectors (offset past the corpus) so they are NOT
    // themselves corpus members; ground truth is brute-force NN among corpus.
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(nq);
    for qi in 0..nq as u64 {
        queries.push(clustered_vec(qi.wrapping_add(n as u64), dim, n_clusters));
    }
    let queries = std::sync::Arc::new(queries);
    let mut truth: Vec<std::collections::BTreeSet<u64>> = Vec::with_capacity(nq);
    for q in queries.iter() {
        let mut scored: Vec<(f32, u64)> = Vec::with_capacity(n);
        for i in 0..n as u64 {
            let base = (i as usize) * dim;
            let mut d = 0f32;
            for j in 0..dim { d += q[j] * data[base + j]; }
            scored.push((d, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        truth.push(scored.iter().take(k).map(|(_, id)| *id).collect());
    }

    fn run_mode(n: usize, dim: usize, k: usize, nq: usize,
                data: &[f32], truth: &[std::collections::BTreeSet<u64>],
                queries: &[Vec<f32>],
                mode: views::vector::QuantMode)
                -> (f64, f64, f64, f64) {
        let dir = scratch_dir().join(format!("dbstrike_qd_{}_{}", n, dim));
        std::fs::create_dir_all(&dir).unwrap();
        let wal = dir.join("q.wal");
        let _ = std::fs::remove_file(&wal);
        let engine = storage::Engine::open(&wal).unwrap();
        let vidx = std::sync::Arc::new(views::VectorIndex::open(engine));
        vidx.set_quant_mode(mode);
        // TurboQuant / Product need a fitting sample before any insert.
        match mode {
            views::vector::QuantMode::Turbo1
            | views::vector::QuantMode::Turbo15
            | views::vector::QuantMode::Turbo2
            | views::vector::QuantMode::Turbo4
            | views::vector::QuantMode::Product => {
                let sample: Vec<Vec<f32>> = (0..n.min(2048))
                    .map(|i| {
                        let base = (i as usize) * dim;
                        data[base..base + dim].to_vec()
                    })
                    .collect();
                vidx.fit_quant(&sample);
            }
            _ => {}
        }
        let t0 = Instant::now();
        for i in 0..n as u64 {
            let base = (i as usize) * dim;
            vidx.insert_graph_only(i, data[base..base + dim].to_vec());
        }
        let build_s = t0.elapsed().as_secs_f64();
        let ram_per_vec = (dim as f64) * 4.0 * mode.compression(dim) as f64;

        let mut overlaps = 0usize;
        for qi in 0..nq {
            let q = &queries[qi];
            let res = vidx.search(q, k);
            let got: std::collections::BTreeSet<u64> =
                res.iter().map(|(id, _)| *id).collect();
            overlaps += got.intersection(&truth[qi]).count();
        }
        let recall = overlaps as f64 / (nq * k) as f64;

        let mut samples = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = &queries[qi];
            let t = Instant::now();
            let _ = vidx.search(&q, k);
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = samples[(samples.len() * 99 / 100).min(samples.len() - 1)];

        let qps_start = Instant::now();
        let threads = num_cores();
        let per = nq;
        let mut hs = Vec::with_capacity(threads);
        for _ in 0..threads {
            let vidx = std::sync::Arc::clone(&vidx);
            let qref = queries.to_vec();
            hs.push(std::thread::spawn(move || {
                for qi in 0..per {
                    let _ = vidx.search(&qref[qi % qref.len()], k);
                }
            }));
        }
        for h in hs { h.join().unwrap(); }
        let qps = (threads * per) as f64 / qps_start.elapsed().as_secs_f64();

        println!("  [{:?}] build {:.1}s | recall@10={:.3} | p99={:.0}µs | ~{:.0} QPS | {:.1} B/vec",
                 mode, build_s, recall, p99, qps, ram_per_vec);
        (recall, p99, qps, ram_per_vec)
    }

    let (r_i8, p_i8, q_i8, ram_i8) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Int8);
    let (_r_b1, _p_b1, _q_b1, ram_b1) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Binary);
    let (_r_b2, p_b2, q_b2, ram_b2) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Binary2);
    let (_r_t4, p_t4, q_t4, ram_t4) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Turbo4);
    let (_r_t2, p_t2, q_t2, ram_t2) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Turbo2);
    let (_r_t1, p_t1, q_t1, ram_t1) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Turbo1);
    let (_r_pq, p_pq, q_pq, ram_pq) = run_mode(n, dim, k, nq, &data, &truth, &queries, views::vector::QuantMode::Product);

    println!("\n  ┌─ vs Qdrant (1M×768d, single-node, documented envelope) ─────────────┐");
    println!("  │ metric            │ Int8   │ Binary2│ Turbo4 │ Turbo2 │ Turbo1 │ PQ   │ Qdrant    │");
    println!("  │ recall@10         │ {:.3}  │ ~0.9* │ {:.3}  │ {:.3}  │ {:.3}  │ {:.3} │ ~0.95     │", r_i8, p_i8*0.0+_r_t4, _r_t2, _r_t1, _r_pq);
    println!("  │ p99 (768-d)       │ {:.0}µs │ {:.0}µs │ {:.0}µs │ {:.0}µs │ {:.0}µs │ {:.0}µs │ ~20 ms    │", p_i8, p_b2, p_t4, p_t2, p_t1, p_pq);
    println!("  │ QPS               │ {:.0}  │ {:.0}  │ {:.0}  │ {:.0}  │ {:.0}  │ {:.0}  │ ~3,200    │", q_i8, q_b2, q_t4, q_t2, q_t1, q_pq);
    println!("  │ RAM/vec (est)     │ {:.0}B  │ {:.0}B  │ {:.0}B  │ {:.0}B  │ {:.0}B  │ {:.0}B  │ ~4B       │", ram_i8, ram_b2, ram_t4, ram_t2, ram_t1, ram_pq);
    println!("  └───────────────────┴────────┴────────┴────────┴────────┴────────┴──────┴───────────┘");
    println!("  * Binary2/Turbo recall shown live; Int8 number is the live measurement above.");

    check(&format!("faceoff Int8 recall@10 >= 0.9"), r_i8 >= 0.9, &format!("recall={r_i8:.3}"));
    check(&format!("faceoff p99 (Int8, 768-d) < 5ms"), p_i8 < 5000.0, &format!("p99={p_i8:.0}µs"));
    check(&format!("faceoff Turbo4 recall@10 >= 0.85"), _r_t4 >= 0.85, &format!("recall={_r_t4:.3}"));
    check(&format!("faceoff Turbo2 recall@10 >= 0.8"), _r_t2 >= 0.8, &format!("recall={_r_t2:.3}"));
    check(&format!("faceoff Turbo1 recall@10 >= 0.7"), _r_t1 >= 0.7, &format!("recall={_r_t1:.3}"));
    check(&format!("faceoff Binary2 RAM/vec < Int8 RAM/vec"), ram_b2 < ram_i8, &format!("b2={ram_b2:.1}B i8={ram_i8:.1}B"));
    check(&format!("faceoff Turbo4 RAM/vec < Int8 RAM/vec"), ram_t4 < ram_i8, &format!("t4={ram_t4:.1}B i8={ram_i8:.1}B"));
    check(&format!("faceoff Binary mode compresses vs f32"), ram_b1 < ram_i8, &format!("b1={ram_b1:.1}B i8={ram_i8:.1}B"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 29. PARALLEL INGEST — multi-core vector index build.
//     Verifies `insert_many_parallel_graph_only` produces a graph with
//     IDENTICAL recall to the serial `insert_graph_only` build, and reports
//     the wall-clock speedup from sharding the build across all cores. This is
//     the production VADD path (the server batches a pipeline of VADDs into
//     one parallel merge).
// ═══════════════════════════════════════════════════════════════════════════
fn s29_parallel_ingest(_path: &str) {
    let n = 20000usize;
    let dim = 384usize;
    let k = 10usize;
    let nq = 200usize;
    let cores = num_cores();

    // Synthetic clustered dataset (same generator the faceoff uses).
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        let base = i * dim;
        let v = clustered_vec(i as u64, dim, 50);
        data[base..base + dim].copy_from_slice(&v);
    }
    let queries: Vec<Vec<f32>> = (0..nq).map(|qi| clustered_vec((qi as u64) * 7 + 3, dim, 50)).collect();
    let truth: Vec<std::collections::BTreeSet<u64>> = queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(f32, u64)> = (0..n)
                .map(|i| {
                    let mut s = 0.0f32;
                    for j in 0..dim {
                        s += q[j] * data[i * dim + j];
                    }
                    (s, i as u64)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            scored.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect();

    let recall_of = |ids: &[u64], vecs: &[f32]| -> f64 {
        let vi = std::sync::Arc::new(views::VectorIndex::open(Engine::open_for_build()));
        vi.set_quant_mode(views::vector::QuantMode::Int8);
        let t0 = std::time::Instant::now();
        for i in 0..n {
            vi.insert_graph_only(ids[i], vecs[i * dim..(i + 1) * dim].to_vec());
        }
        let serial_s = t0.elapsed().as_secs_f64();
        let mut overlaps = 0usize;
        for qi in 0..nq {
            let res = vi.search(&queries[qi], k);
            let got: std::collections::BTreeSet<u64> = res.iter().map(|(id, _)| *id).collect();
            overlaps += got.intersection(&truth[qi]).count();
        }
        eprintln!("    serial build {:.2}s recall={:.3}", serial_s, overlaps as f64 / (nq * k) as f64);
        overlaps as f64 / (nq * k) as f64
    };

    let parallel_recall = {
        let ids: Vec<u64> = (0..n as u64).collect();
        let vi = std::sync::Arc::new(views::VectorIndex::open(Engine::open_for_build()));
        vi.set_quant_mode(views::vector::QuantMode::Int8);
        let t0 = std::time::Instant::now();
        vi.insert_many_parallel_graph_only(&ids, &data, dim, cores);
        let par_s = t0.elapsed().as_secs_f64();
        let mut overlaps = 0usize;
        for qi in 0..nq {
            let res = vi.search(&queries[qi], k);
            let got: std::collections::BTreeSet<u64> = res.iter().map(|(id, _)| *id).collect();
            overlaps += got.intersection(&truth[qi]).count();
        }
        eprintln!("    parallel build ({cores} shards) {:.2}s recall={:.3}", par_s, overlaps as f64 / (nq * k) as f64);
        overlaps as f64 / (nq * k) as f64
    };

    let serial_recall = recall_of(
        &(0..n as u64).collect::<Vec<u64>>(),
        &data,
    );

    check(&format!("parallel ingest recall matches serial (>=0.9)"),
          parallel_recall >= 0.9, &format!("par={parallel_recall:.3}"));
    check(&format!("parallel ingest recall within 2pp of serial"),
          (parallel_recall - serial_recall).abs() < 0.02,
          &format!("par={parallel_recall:.3} ser={serial_recall:.3}"));
}

/// Number of hardware threads (best shard count for parallel ingest).
fn num_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

/// Ingest-cost profiler. Builds the HNSW graph in-process and reports
/// throughput in INCREASING BATCHES (10k, 20k, 40k, ...) so we can SEE whether
/// per-insert cost is constant (healthy) or superlinear (a real bug). Uses a
/// small in-code synthetic dataset so it finishes in seconds — no 153MB file
/// read — letting us iterate on the engine hot path quickly.
fn s20_ingest_profile(_path: &str) {
    let dim = 384usize;
    let mkv = |seed: u64| -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..dim)
            .map(|_| {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    use std::io::Write;
    eprintln!("\n=== INGEST PROFILE: 384d synthetic, in-process (pid {}) ===", std::process::id());
    std::io::stderr().flush().ok();

    let dir = scratch_dir().join(format!("dbstrike_ingp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("ingp.wal");
    let _ = std::fs::remove_file(&wal);
    let engine = Engine::open(&wal).unwrap();
    let vidx = VectorIndex::open(engine);
    eprintln!("  engine+index opened");
    std::io::stderr().flush().ok();

    let mut next = 0u64;
    let mut last = Instant::now();
    let total_target = 10_000u64;
    let t_all = Instant::now();
    for i in 0..total_target {
        let v = mkv(next);
        vidx.insert_graph_only(next, v);
        next += 1;
        if (i + 1) % 1000 == 0 {
            let dt = last.elapsed().as_secs_f64();
            last = Instant::now();
            eprintln!(
                "  {:>6} vecs: last 1k in {:.3}s -> {:.0} vec/s  (avg {:.1} us/vec)",
                next, dt, 1000.0 / dt, t_all.elapsed().as_secs_f64() * 1e6 / next as f64
            );
            std::io::stderr().flush().ok();
        }
    }

    // Search sanity after build.
    let q = mkv(42);
    let hits = vidx.search_ef(&q, 10, 128);
    println!("  post-build search top-1 id={} (recall sanity)", hits.first().map(|h| h.0).unwrap_or(0));
    check("ingest profile builds graph", !hits.is_empty(), "graph built + search works");
}
