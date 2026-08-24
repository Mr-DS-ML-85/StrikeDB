//! Structural diagnostics: forced-fallback vs success-path graphs on the
//! SAME corpus. Answers WHY the fallback collapses instead of guessing.

use views::VectorIndex;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn f32(&mut self) -> f32 {
        (self.next() % 20_000) as f32 / 10_000.0 - 1.0
    }
}

fn make_corpus(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let clusters = 40;
    let mut centroids = Vec::with_capacity(clusters * dim);
    for _ in 0..clusters * dim {
        centroids.push(rng.f32());
    }
    let sigma = 0.05f32;
    let mut data = vec![0f32; n * dim];
    for i in 0..n {
        let c = (i % clusters) * dim;
        for d in 0..dim {
            data[i * dim + d] = centroids[c + d] + rng.f32() * sigma;
        }
    }
    for row in data.chunks_mut(dim) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    data
}

fn report(label: &str, vi: &VectorIndex, data: &[f32], dim: usize) {
    let n = data.len() / dim;
    let (nodes, zero_deg, reach, bad) = vi.graph_health();
    eprintln!(
        "[{label}] nodes={nodes} zero-deg={zero_deg} bfs-reachable={reach}/{n} bad-edges={bad}"
    );
    let mut hits = 0;
    for qi in 0..25 {
        let row = qi * 733 % n;
        let q = &data[row * dim..(row + 1) * dim];
        let got: Vec<u64> = vi.search(q, 10).into_iter().map(|(id, _)| id).collect();
        if got.contains(&(row as u64)) {
            hits += 1;
        }
    }
    eprintln!("[{label}] self-hit@10 = {hits}/25");
}

#[test]
fn diagnose_fallback_vs_success() {
    if !gpu::gpu_init() {
        eprintln!("no CUDA device — skipping");
        return;
    }
    gpu::gpu_set_mode(gpu::ComputeMode::Turbo);
    let (n, dim) = (20_000usize, 64usize);
    let data = make_corpus(n, dim);

    std::env::set_var("DBSTRIKE_FORCE_APGC_FALLBACK", "1");
    let fb = VectorIndex::build_parallel_tiered(&data, dim, 8, false);
    report("FALLBACK", &fb, &data, dim);

    std::env::remove_var("DBSTRIKE_FORCE_APGC_FALLBACK");
    let ok = VectorIndex::build_parallel_tiered(&data, dim, 8, false);
    report("SUCCESS", &ok, &data, dim);
}
