//! Quick in-process ingest + search benchmark on real datasets.
//! No TCP, no RESP — just the engine directly.

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/home/irfan/datasets/real_384_1M.fbin");

    // GPU tier detection
    gpu::gpu_init();
    let gpu_info = gpu::gpu_info();
    println!("[GPU] {}", gpu_info.iter().map(|(k,v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", "));

    // Load .fbin
    println!("Loading {path}...");
    let t0 = Instant::now();
    let bytes = std::fs::read(path).expect("cannot read file");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes[8..].as_ptr() as *const f32, n * dim).to_vec()
    };
    println!("  loaded {n} × {dim}d in {:.1}s", t0.elapsed().as_secs_f64());

    // L2 normalize
    println!("Normalizing...");
    let t1 = Instant::now();
    let mut norm = data.clone();
    for i in 0..n {
        let base = i * dim;
        let mut s = 0.0f32;
        for j in 0..dim { s += norm[base + j] * norm[base + j]; }
        let inv = s.sqrt().recip();
        for j in 0..dim { norm[base + j] *= inv; }
    }
    println!("  normalized in {:.1}s", t1.elapsed().as_secs_f64());

    // Build graph — try parallel first, fall back to serial for perfect recall
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("Building HNSW graph ({cores} cores, {n} vectors, {dim}d)...");
    let t2 = Instant::now();
    let vi = views::VectorIndex::build_parallel(&norm, dim, cores);
    let build_s = t2.elapsed().as_secs_f64();
    println!("  TOTAL build: {build_s:.1}s ({:.0} vec/s)", n as f64 / build_s);
    let (ml, per_level, avg_neigh) = vi.debug_shape();
    println!("  graph: max_level={ml} avg_neigh0={avg_neigh:.1} per_level={per_level:?}");

    // Search benchmarks
    let nq = 200.min(n);
    println!("Search benchmark ({nq} queries)...");

    // Single-thread latency
    let mut samples = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let t = Instant::now();
        let _ = vi.search_ef(q, 10, 128);
        samples.push(t.elapsed().as_micros() as u64);
    }
    samples.sort();
    let p50 = samples[nq / 2];
    let p99 = samples[(nq * 99 / 100).min(nq - 1)];
    println!("  single-thread: p50={p50}µs p99={p99}µs");

    // Multi-thread QPS
    let t3 = Instant::now();
    let qps_n = 1000;
    let vi_arc = std::sync::Arc::new(vi);
    let norm_arc = std::sync::Arc::new(norm);
    let handles: Vec<_> = (0..cores).map(|_| {
        let vi = std::sync::Arc::clone(&vi_arc);
        let norm = std::sync::Arc::clone(&norm_arc);
        let nq = qps_n / cores;
        std::thread::spawn(move || {
            for qi in 0..nq {
                let q = &norm[(qi % n) * dim..(qi % n) * dim + dim];
                let _ = vi.search_ef(q, 10, 128);
            }
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    let qps = qps_n as f64 / t3.elapsed().as_secs_f64();
    println!("  {cores}-thread QPS: {qps:.0}");

    // Recall (limit to 50 queries for speed)
    let n_recall = 50.min(n);
    let mut hits = 0usize;
    for qi in 0..n_recall {
        let q = &norm_arc[qi * dim..(qi + 1) * dim];
        // Brute-force ground truth
        let mut truth: Vec<(f32, u64)> = (0..n as u64).map(|i| {
            let mut dot = 0.0f32;
            for j in 0..dim { dot += q[j] * norm_arc[i as usize * dim + j]; }
            ((1.0 - dot).max(0.0).min(2.0), i)
        }).collect();
        truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt: std::collections::BTreeSet<u64> = truth.iter().take(10).map(|(_, id)| *id).collect();
        let res = vi_arc.search_ef(q, 10, 128);
        hits += res.iter().take(10).filter(|(id, _)| gt.contains(id)).count();
    }
    let recall = hits as f64 / (n_recall * 10) as f64;
    println!("  Recall@10: {recall:.3}");

    println!("\n=== Summary ===");
    println!("  Dataset: {n} × {dim}d");
    println!("  Build: {build_s:.1}s ({:.0} vec/s)", n as f64 / build_s);
    println!("  Search p50/p99: {p50}µs / {p99}µs");
    println!("  {cores}-thread QPS: {qps:.0}");
    println!("  Recall@10: {recall:.3}");
}
