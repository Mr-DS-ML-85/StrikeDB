/// Real dataset recall benchmarks for the APGC paper.
/// Loads .fbin datasets and measures actual Recall@10.

use std::time::Instant;

pub fn run_recall_benchmarks() {
    println!("── Real Dataset Recall Benchmarks ──");
    println!();

    let datasets = [
        ("real_384_100k", "/home/irfan/datasets/real_384_100k.fbin", 100_000, 384),
        ("real_384_1M", "/home/irfan/datasets/real_384_1M.fbin", 1_000_000, 384),
        ("real_768_100k", "/home/irfan/datasets/real_768_100k.fbin", 100_000, 768),
        ("real_768_1M", "/home/irfan/datasets/real_768_1M.fbin", 1_000_000, 768),
    ];

    for (name, path, expected_n, expected_dim) in &datasets {
        match load_fbin(path) {
            Ok(data) => {
                if data.n != *expected_n || data.dim != *expected_dim {
                    eprintln!("  SKIP {}: expected {}x{}, got {}x{}", name, expected_n, expected_dim, data.n, data.dim);
                    continue;
                }
                measure_recall(name, &data);
            }
            Err(e) => {
                eprintln!("  SKIP {}: {}", name, e);
            }
        }
    }
}

struct Dataset {
    vectors: Vec<Vec<f32>>,
    n: usize,
    dim: usize,
}

fn load_fbin(path: &str) -> Result<Dataset, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    
    // fbin format: 4-byte header (n: int32, dim: int32), then n*dim floats
    let mut header = [0u8; 8];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let mut raw = vec![0u8; n * dim * 4];
    file.read_exact(&mut raw).map_err(|e| e.to_string())?;
    
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            let offset = (i * dim + j) * 4;
            let bytes = [raw[offset], raw[offset+1], raw[offset+2], raw[offset+3]];
            v.push(f32::from_le_bytes(bytes));
        }
        vectors.push(v);
    }
    
    Ok(Dataset { vectors, n, dim })
}

fn measure_recall(name: &str, data: &Dataset) {
    let n = data.n;
    let dim = data.dim;
    let nq = 50.min(n);
    
    println!("Dataset: {} ({} × {}d)", name, n, dim);
    
    // Ground truth: brute-force top-10 for first nq queries
    let t0 = Instant::now();
    let ground_truth: Vec<Vec<usize>> = (0..nq).map(|qi| {
        let q = &data.vectors[qi];
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != qi)
            .map(|j| {
                let mut dot = 0.0f32;
                for d in 0..dim { dot += q[d] * data.vectors[j][d]; }
                (1.0 - dot, j)
            }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.iter().take(10).map(|(_, j)| *j).collect()
    }).collect();
    let gt_time = t0.elapsed().as_secs_f64();
    println!("  Ground truth: {:.1}s", gt_time);
    
    // Test different k values for precision levels
    let gpu = apgc::precision::GpuCaps::RTX4060;
    let precisions = [
        ("FP32 (seeds only)", apgc::precision::PrecisionLevel::Fp32),
        ("FP16 (majority)", apgc::precision::PrecisionLevel::Fp16),
        ("BF16 (majority)", apgc::precision::PrecisionLevel::Bf16),
        ("FP8 (near-outliers)", apgc::precision::PrecisionLevel::Fp8),
        ("INT8 (outliers)", apgc::precision::PrecisionLevel::Int8),
    ];
    
    for (label, prec) in &precisions {
        if !gpu.supports(*prec) {
            println!("  {}: N/A (not supported on RTX 4060)", label);
            continue;
        }
        
        let t1 = Instant::now();
        let mut hits = 0usize;
        for qi in 0..nq {
            let q = &data.vectors[qi];
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != qi)
                .map(|j| {
                    (apgc::precision::MixedPrecisionBuilder::distance(q, &data.vectors[j], *prec), j)
                }).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let result_set: std::collections::HashSet<usize> = dists.iter().take(10).map(|(_, j)| *j).collect();
            hits += ground_truth[qi].iter().filter(|id| result_set.contains(id)).count();
        }
        let recall = hits as f64 / (nq * 10) as f64;
        let search_time = t1.elapsed().as_secs_f64();
        println!("  {}: Recall@10={:.4} ({:.1}s, {:.0} QPS)", label, recall, search_time, nq as f64 / search_time);
    }
    
    // Mixed-precision (APGC)
    let t2 = Instant::now();
    let mut hits = 0usize;
    let config = apgc::precision::GraphConfig {
        k: 10, n, dim, seed_ratio: 0.01, outlier_ratio: 0.10,
        gpu_caps: Some(gpu),
    };
    let mut graph = apgc::precision::ApgcGraph::new(config);
    graph.assign_precision();
    
    for qi in 0..nq {
        let q = &data.vectors[qi];
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != qi)
            .map(|j| {
                let prec = graph.precision_map[j];
                (apgc::precision::MixedPrecisionBuilder::distance(q, &data.vectors[j], prec), j)
            }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let result_set: std::collections::HashSet<usize> = dists.iter().take(10).map(|(_, j)| *j).collect();
        hits += ground_truth[qi].iter().filter(|id| result_set.contains(id)).count();
    }
    let recall = hits as f64 / (nq * 10) as f64;
    let search_time = t2.elapsed().as_secs_f64();
    println!("  APGC Mixed: Recall@10={:.4} ({:.1}s, {:.0} QPS)", recall, search_time, nq as f64 / search_time);
    println!();
}
