//! BUG-1 authoritative tight loop: REAL 100k×384 dataset, wire-equivalent
//! pipeline (build_parallel_ids → upload_to_gpu → search_ef), recall measured
//! as top-10 OVERLAP vs CPU brute-force ground truth — not self-hit, which
//! is meaningless on tie-heavy corpora.
//! Skips silently if the dataset file is absent.

use views::VectorIndex;

fn load_fbin(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 8 {
        return None;
    }
    let n = u32::from_le_bytes(b[0..4].try_into().ok()?) as usize;
    let dim = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
    let mut data = vec![0f32; n * dim];
    let raw = &b[8..8 + n * dim * 4];
    for (dst, src) in data.iter_mut().zip(raw.chunks_exact(4)) {
        *dst = f32::from_le_bytes(src.try_into().unwrap());
    }
    for row in data.chunks_mut(dim) {
        let nr = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in row.iter_mut() {
            *x /= nr;
        }
    }
    Some((data, n, dim))
}

fn overlap_recall(vi: &VectorIndex, data: &[f32], dim: usize, k: usize) -> f32 {
    let n = data.len() / dim;
    let nq = 20;
    let mut hits = 0usize;
    let mut total = 0usize;
    for qi in 0..nq {
        let row = qi * 3793 % n;
        let q = &data[row * dim..(row + 1) * dim];
        // Brute-force ground truth over the whole corpus.
        let mut truth: Vec<(u64, f32)> = (0..n as u64)
            .map(|i| {
                let v = &data[i as usize * dim..(i as usize + 1) * dim];
                let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                (i, 1.0 - dot)
            })
            .collect();
        truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let truth_ids: Vec<u64> = truth.iter().take(k).map(|(id, _)| *id).collect();
        let got: Vec<u64> = vi.search_ef(q, k, 128).into_iter().map(|(id, _)| id).collect();
        hits += got.iter().filter(|g| truth_ids.contains(g)).count();
        total += k;
    }
    hits as f32 / total as f32
}

#[test]
fn wire_real_data_fallback_vs_success() {
    let path = "/home/irfan/datasets/real_384_100k.fbin";
    if std::fs::metadata(path).is_err() {
        eprintln!("dataset absent — skipping");
        return;
    }
    if !gpu::gpu_init() {
        eprintln!("no CUDA device — skipping");
        return;
    }
    gpu::gpu_set_mode(gpu::ComputeMode::Turbo);
    let (data, n, dim) = load_fbin(path).expect("valid fbin");
    eprintln!("dataset {n}x{dim}");

    let ids: Vec<u64> = (0..n as u64).collect();
    let attrs = vec![0u32; n];

    std::env::set_var("DBSTRIKE_FORCE_APGC_FALLBACK", "1");
    let fb = VectorIndex::build_parallel_ids(&data, dim, 16, &ids, &attrs);
    fb.upload_to_gpu();
    let r_fb = overlap_recall(&fb, &data, dim, 10);
    eprintln!("FALLBACK overlap-recall@10 = {r_fb:.3}");

    std::env::remove_var("DBSTRIKE_FORCE_APGC_FALLBACK");
    let ok = VectorIndex::build_parallel_ids(&data, dim, 16, &ids, &attrs);
    ok.upload_to_gpu();
    let r_ok = overlap_recall(&ok, &data, dim, 10);
    eprintln!("SUCCESS  overlap-recall@10 = {r_ok:.3}");

    // Observed post-fix: fallback 0.995 / success 0.985 @100k×384d.
    // Gate well below that so build noise never trips it, far above the
    // collapsed 1/12 signature this exists to catch.
    assert!(
        r_fb >= 0.8,
        "BUG-1 regression: fallback recall {r_fb:.3} vs success {r_ok:.3} on real data"
    );
}
