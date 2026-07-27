//! Quick in-process ingest + search benchmark on real datasets.
use std::time::Instant;

fn run_bench(_path: &str, mode: &str, n: usize, dim: usize, norm: Vec<f32>, cores: usize, keep_norm: bool) -> Vec<f32> {
    // norm is OWNED. In single-mode runs we drop it after build to save
    // 1.5-3GB RAM. In `--mode all` runs the caller needs the full dataset
    // back for the next mode (keep_norm=true) — returning only the queries
    // corrupted every mode after the first (index OOB at ground truth).
    let nq = 200.min(n);
    let queries: Vec<f32> = (0..nq).flat_map(|qi| norm[qi * dim..(qi + 1) * dim].iter().copied()).collect();

    match mode {
        "turbo" => {
            let detected = gpu::gpu_auto_mode(n, dim);
            if detected != gpu::ComputeMode::Turbo {
                eprintln!("[GPU] WARNING: data may exceed VRAM. Using {:?}", detected);
            } else {
                gpu::gpu_set_mode(gpu::ComputeMode::Turbo);
            }
        }
        "hybrid" => gpu::gpu_set_mode(gpu::ComputeMode::Hybrid),
        "cpu" => gpu::gpu_set_mode(gpu::ComputeMode::CpuOnly),
        _ => { gpu::gpu_auto_mode(n, dim); }
    }
    let actual_mode = gpu::gpu_get_mode();
    println!("\n{}", "=".repeat(56));
    println!("  MODE: {:?} | {n} × {dim}d | {cores} cores", actual_mode);
    println!("{}", "=".repeat(56));

    // Ground truth FIRST (while norm is fresh).
    // 50 queries × top-10 is only 500 samples, so recall lands on a 0.002
    // grid — 0.996 and 0.998 differ by a single neighbour and mode-to-mode
    // comparisons drown in that noise. 200 queries gives a 0.0005 grid.
    // Brute force is embarrassingly parallel, so widening it is nearly free.
    let n_recall = std::env::var("RECALL_QUERIES").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(200).min(nq);
    println!("Computing recall ground truth ({n_recall} queries, {cores} threads)...");
    let t_bf = Instant::now();
    let mut ground_truth: Vec<Vec<u64>> = vec![Vec::new(); n_recall];
    {
        let norm_ref: &[f32] = &norm;
        let chunk = (n_recall + cores - 1) / cores.max(1);
        std::thread::scope(|s| {
            for (ci, out) in ground_truth.chunks_mut(chunk.max(1)).enumerate() {
                let base = ci * chunk.max(1);
                s.spawn(move || {
                    let mut truth: Vec<(f32, u64)> = Vec::with_capacity(n);
                    for (off, slot) in out.iter_mut().enumerate() {
                        let qi = base + off;
                        let q = &norm_ref[qi * dim..(qi + 1) * dim];
                        truth.clear();
                        for i in 0..n {
                            let v = &norm_ref[i * dim..(i + 1) * dim];
                            let mut dot = 0.0f32;
                            for j in 0..dim { dot += q[j] * v[j]; }
                            truth.push(((1.0 - dot).max(0.0).min(2.0), i as u64));
                        }
                        truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        *slot = truth.iter().take(10).map(|(_, id)| *id).collect();
                    }
                });
            }
        });
    }
    println!("  ground truth: {:.1}s", t_bf.elapsed().as_secs_f64());

    // Build graph — GPU path stores only all_i8 (no f32 copies)
    println!("Building HNSW graph...");
    let t2 = Instant::now();
    let vi = views::VectorIndex::build_parallel_tiered(&norm, dim, cores, true);
    // Single-mode runs: norm is no longer needed after build — free the RAM.
    let saved_norm = if keep_norm { norm } else { Vec::new() };
    let build_s = t2.elapsed().as_secs_f64();
    println!("  build: {build_s:.1}s ({:.0} vec/s)", n as f64 / build_s);

    if matches!(actual_mode, gpu::ComputeMode::Turbo | gpu::ComputeMode::Hybrid) {
        println!("  Uploading graph to GPU...");
        vi.upload_to_gpu();
    }

    let vi = std::sync::Arc::new(vi);

    // Single-thread latency
    let mut samples = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let t = Instant::now();
        let _ = vi.search_ef(q, 10, 128);
        samples.push(t.elapsed().as_micros() as u64);
    }
    samples.sort();
    let p50 = samples[nq / 2];
    let p99 = samples[(nq * 99 / 100).min(nq - 1)];
    println!("  search p50/p99: {p50}µs / {p99}µs");

    // Multi-thread QPS
    let t3 = Instant::now();
    let qps_n = 1000;
    let queries_arc = std::sync::Arc::new(queries);
    let handles: Vec<_> = (0..cores).map(|_| {
        let vi = std::sync::Arc::clone(&vi);
        let q = std::sync::Arc::clone(&queries_arc);
        let nq = qps_n / cores;
        std::thread::spawn(move || {
            for qi in 0..nq {
                let query = &q[(qi % nq) * dim..(qi % nq) * dim + dim];
                let _ = vi.search_ef(query, 10, 128);
            }
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    let qps = qps_n as f64 / t3.elapsed().as_secs_f64();
    println!("  {cores}-thread QPS: {qps:.0}");

    // Batched QPS — the throughput path.
    //
    // The single-query numbers above are LATENCY-bound and cannot saturate a
    // GPU: each caller blocks on its own result, so in a closed loop the
    // in-flight query count equals the thread count. The APGC search kernel is
    // one block per query, so 16 threads means at most 16 resident blocks on a
    // 24-SM device — and with the device mostly idle between micro-launches the
    // clock governor never leaves its lowest P-state (measured: 210 MHz of a
    // 3135 MHz max during search, vs 2805 MHz during the build).
    //
    // search_many submits Q queries as ONE launch of Q blocks, which is the
    // only shape that actually fills the device. Reporting only the
    // single-query number would understate GPU throughput by an order of
    // magnitude; reporting only this one would hide the latency story. Both.
    let batch_sizes = [32usize, 256, 1024];
    for &bs in &batch_sizes {
        let bq: Vec<Vec<f32>> = (0..bs)
            .map(|i| queries_arc[(i % nq) * dim..(i % nq) * dim + dim].to_vec())
            .collect();
        // Warm up so we time steady state, not first-touch/JIT.
        let _ = vi.search_many(&bq, 10);
        // Run for a fixed WALL time, not a fixed rep count. The GPU clock
        // governor needs sustained load before it leaves its idle P-state, so
        // a sub-second sample measures the ramp, not the steady state.
        let secs: f64 = std::env::var("BATCH_SECS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(2.0);
        let t = Instant::now();
        let mut done = 0usize;
        while t.elapsed().as_secs_f64() < secs {
            let _ = vi.search_many(&bq, 10);
            done += bs;
        }
        let el = t.elapsed().as_secs_f64();
        println!("  batch[{bs:>4}] QPS: {:.0}  ({:.1}µs/query)",
            done as f64 / el, el * 1e6 / done as f64);
    }

    // Recall
    let mut hits = 0usize;
    for qi in 0..n_recall {
        let q = &queries_arc[qi * dim..(qi + 1) * dim];
        let gt: std::collections::BTreeSet<u64> = ground_truth[qi].iter().copied().collect();
        let res = vi.search_ef(q, 10, 128);
        hits += res.iter().take(10).filter(|(id, _)| gt.contains(id)).count();
    }
    let recall = hits as f64 / (n_recall * 10) as f64;
    println!("  Recall@10: {recall:.3}");
    println!("  Mode: {actual_mode:?} | Build: {build_s:.1}s | QPS: {qps:.0} | p50: {p50}µs | Recall: {recall:.3}");

    // `--mode all`: hand the full dataset back for the next mode.
    saved_norm
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/home/irfan/datasets/real_384_1M.fbin");
    let mode_idx = args.iter().position(|a| a == "--mode");
    let mode = mode_idx.and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("auto");

    gpu::gpu_init();
    let gpu_info = gpu::gpu_info();
    println!("[GPU] {}", gpu_info.iter().map(|(k,v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", "));

    println!("Loading {path}...");
    let t0 = Instant::now();
    let bytes = std::fs::read(path).expect("cannot read file");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut data: Vec<f32> = unsafe { std::slice::from_raw_parts(bytes[8..].as_ptr() as *const f32, n * dim).to_vec() };
    println!("  loaded {n} × {dim}d in {:.1}s", t0.elapsed().as_secs_f64());
    drop(bytes);

    // L2 normalize in-place
    println!("Normalizing...");
    let t1 = Instant::now();
    for i in 0..n {
        let base = i * dim;
        let mut s = 0.0f32;
        for j in 0..dim { s += data[base + j] * data[base + j]; }
        let inv = s.sqrt().recip();
        for j in 0..dim { data[base + j] *= inv; }
    }
    println!("  normalized in {:.1}s", t1.elapsed().as_secs_f64());

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    if mode == "all" {
        for m in &["cpu", "hybrid", "turbo"] {
            data = run_bench(path, m, n, dim, data, cores, true);
        }
    } else {
        run_bench(path, mode, n, dim, data, cores, false);
    }

    println!("\n{}", "=".repeat(56));
    println!("  Dataset: {n} × {dim}d | Mode: {mode}");
    println!("{}", "=".repeat(56));
}
