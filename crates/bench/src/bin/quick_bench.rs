//! Quick in-process ingest + search benchmark on real datasets.
//! No TCP, no RESP — just the engine directly.
//!
//! Usage:
//!   quick_bench <dataset.fbin>                    — auto mode (default)
//!   quick_bench <dataset.fbin> --mode cpu         — CPU-only
//!   quick_bench <dataset.fbin> --mode turbo       — full GPU
//!   quick_bench <dataset.fbin> --mode hybrid      — GPU+RAM+CPU
//!   quick_bench <dataset.fbin> --mode all         — all three modes

use std::time::Instant;

fn run_bench(_path: &str, mode: &str, n: usize, dim: usize, norm: &[f32], cores: usize) {
    let nq = 200.min(n);
    let queries: Vec<f32> = (0..nq).flat_map(|qi| norm[qi * dim..(qi + 1) * dim].iter().copied()).collect();

    // Set compute mode
    match mode {
        "turbo" => {
            let detected = gpu::gpu_auto_mode(n, dim);
            if detected != gpu::ComputeMode::Turbo {
                eprintln!("[GPU] WARNING: --mode turbo requested but data may exceed VRAM build budget. Using {:?} (auto-detected)", detected);
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

    // Build graph
    let tiered = matches!(actual_mode, gpu::ComputeMode::Hybrid);
    println!("Building HNSW graph...");
    let t2 = Instant::now();
    let vi = views::VectorIndex::build_parallel_tiered(norm, dim, cores, tiered);
    let build_s = t2.elapsed().as_secs_f64();
    println!("  build: {build_s:.1}s ({:.0} vec/s)", n as f64 / build_s);

    // VUGVA: Initialize Virtual Memory Table for hybrid mode.
    // GPU VRAM budget = 70% of free VRAM (leave headroom for output buffers).
    if matches!(actual_mode, gpu::ComputeMode::Hybrid | gpu::ComputeMode::Turbo) {
        let vram_budget = 7000 * 1024 * 1024; // ~7 GB for RTX 4060
        println!("  Initializing VUGVA Virtual Memory Table ({:.0} MB VRAM budget)...", vram_budget as f64 / 1024.0 / 1024.0);
        vi.init_vugva(vram_budget);
    }

    // Ground truth for recall
    let n_recall = 50.min(nq);
    let ground_truth: Vec<Vec<u64>> = (0..n_recall).map(|qi| {
        let q = &norm[qi * dim..(qi + 1) * dim];
        let mut truth: Vec<(f32, u64)> = (0..n as u64).map(|i| {
            let mut dot = 0.0f32;
            for j in 0..dim { dot += q[j] * norm[i as usize * dim + j]; }
            ((1.0 - dot).max(0.0).min(2.0), i)
        }).collect();
        truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        truth.iter().take(10).map(|(_, id)| *id).collect()
    }).collect();

    let vi = std::sync::Arc::new(vi);
    // Upload to GPU for turbo/hybrid modes
    if matches!(actual_mode, gpu::ComputeMode::Turbo | gpu::ComputeMode::Hybrid) {
        println!("  Uploading graph to GPU...");
        vi.upload_to_gpu();
    }

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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/home/irfan/datasets/real_384_1M.fbin");

    // Parse --mode flag
    let mode_idx = args.iter().position(|a| a == "--mode");
    let mode = mode_idx.and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("auto");

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
    drop(data);

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

    if mode == "all" {
        // Run all three modes
        for m in &["cpu", "hybrid", "turbo"] {
            // Clone norm for each run (some modes may consume it)
            let norm_copy = norm.clone();
            run_bench(path, m, n, dim, &norm_copy, cores);
        }
    } else {
        run_bench(path, mode, n, dim, &norm, cores);
    }

    println!("\n{}", "=".repeat(56));
    println!("  Dataset: {n} × {dim}d | Mode: {mode}");
    println!("{}", "=".repeat(56));
}
