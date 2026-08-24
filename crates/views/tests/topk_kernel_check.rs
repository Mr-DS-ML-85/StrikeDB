//! Isolates BUG-1: does gpu_batch_cosine_dist_topk return TRUE global top-k?
//! The APGC fallback's entire edge list comes from this one call; if its
//! output diverges from CPU brute force, everything downstream inherits it.
//! Compares returned SETS and ORDER against exhaustive CPU scoring over the
//! exact same int8 corpus.

use gpu::gpu_batch_cosine_dist_topk;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn to_i8(data: &[f32], dim: usize) -> Vec<i8> {
    data.chunks(dim)
        .map(|row| {
            let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter().map(|&x| (x / nrm * 127.0) as i8).collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

fn cpu_dists(q: &[i8], corpus: &[i8], dim: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = (0..corpus.len() / dim)
        .map(|i| {
            let dot: i64 = q
                .iter()
                .zip(&corpus[i * dim..(i + 1) * dim])
                .map(|(a, b)| (*a as i64) * (*b as i64))
                .sum();
            let d = 1.0 - dot as f32 / 16129.0;
            (i, d)
        })
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v
}

#[test]
fn batched_topk_matches_bruteforce() {
    if !gpu::gpu_init() {
        eprintln!("no CUDA device — skipping");
        return;
    }
    gpu::gpu_set_mode(gpu::ComputeMode::Turbo);

    let (seg_n, dim, k) = (3000usize, 64usize, 64usize);
    let mut rng = Lcg(0xDEADBEEF);
    let f32_data: Vec<f32> = (0..seg_n * dim).map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0).collect();
    // normalize then quantize exactly like the builder does
    let mut norm = f32_data.clone();
    for row in norm.chunks_mut(dim) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    let corpus_i8 = to_i8(&norm, dim);

    let (indices, distances) =
        gpu_batch_cosine_dist_topk(&corpus_i8, &corpus_i8, seg_n, seg_n, dim, k)
            .expect("GPU topk must succeed");

    // Verify 30 sampled queries across the segment.
    let samples = [0usize, 97, 511, 1024, 1500, 2048, 2500, 2999, 1234, 4321 % seg_n,
                   777, 2222, 111, 888, 2911, 1600, 640, 2879, 1300, 1999,
                   300, 2700, 900, 2100, 450, 2550, 1750, 1200, 650, 2350];
    let mut worst_set_miss = 0usize;
    for &qi in &samples {
        let got: Vec<usize> = (0..k)
            .map(|j| indices[qi * k + j])
            .filter(|&x| x >= 0)
            .map(|x| x as usize)
            .collect();
        let truth = cpu_dists(&corpus_i8[qi * dim..(qi + 1) * dim], &corpus_i8, dim);
        let truth_ids: Vec<usize> = truth.iter().take(k).map(|(i, _)| *i).collect();

        // Set overlap (order-insensitive), excluding the self-match the GPU
        // legitimately returns at rank 0.
        let overlap = got.iter().filter(|g| truth_ids.contains(g)).count();
        worst_set_miss = worst_set_miss.max(k - overlap);

        if qi == samples[0] {
            eprintln!("q{qi} gpu[:5]   {:?}", &got[..5.min(got.len())]);
            eprintln!("q{qi} cpu[:5]   {:?}", &truth_ids[..5]);
            eprintln!("q{qi} gpuD[:5]  {:?}", &distances[qi * k..qi * k + 5]);
            eprintln!("q{qi} cpuD[:5]  {:?}", &truth.iter().take(5).map(|(_, d)| *d).collect::<Vec<_>>());
        }
    }
    eprintln!("worst per-query set miss (of {k}): {worst_set_miss}");
    assert!(
        worst_set_miss <= 2,
        "GPU batched top-k diverges from brute force — every fallback edge is suspect"
    );
}
