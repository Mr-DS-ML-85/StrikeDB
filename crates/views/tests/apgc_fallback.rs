//! Regression harness for BUG-1: the APGC GPU branch's "segment kNN"
//! fallback built a COLLAPSED graph at 1M×384d (self-recall@50 = 1/12,
//! repro 2026-08-24) while the same data through the successful GPU build
//! scored 12/12. These tests hold the fallback to an honest bar at unit
//! scale: navigable graph, verified against brute-force ground truth on
//! clustered (non-degenerate) data.
//!
//! Runs only with a CUDA device present; skips silently otherwise (same
//! convention as the gpu crate's hardware tests).
//!
//! Seam: DBSTRIKE_FORCE_APGC_FALLBACK=1 routes build_parallel_tiered's GPU
//! branch straight to the fallback kNN path even when the device build
//! would succeed — the only way to reach this code deterministically,
//! since in production it is entered exclusively via VRAM-refusal/budget
//! aborts whose timing varies with machine load (which is exactly why the
//! original collapse went misdiagnosed for so long).

use views::VectorIndex;

/// Deterministic LCG so failures reproduce bit-for-bit.
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

/// Clustered corpus: 40 gaussian blobs, sigma wide enough that nearest
/// neighbours are non-trivial (honest-benchmark rule: no near-duplicate
/// degenerate data that makes any graph look perfect).
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
    // L2-normalize every row (ingest contract everywhere else).
    for row in data.chunks_mut(dim) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    data
}

fn brute_force_topk(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = (0..data.len() / dim)
        .map(|i| {
            let dot: f32 = data[i * dim..(i + 1) * dim]
                .iter()
                .zip(q)
                .map(|(a, b)| a * b)
                .sum();
            (i as u64, 1.0 - dot)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

// STATUS: [ignore]d while BUG-1's full fix is in flight — the seam, loud
// per-segment failure recovery, per-node bridging and self-padding are
// landed but wire-level A/B still shows collapsed recall (1/12 @100k), so
// more of the pipeline is implicated (see apgc_diag.rs evidence). Un-ignore
// and keep it green once the fallback produces navigable graphs end-to-end.
#[test]
#[ignore = "BUG-1 open: fallback graph still collapses on the wire (1/12 @100k) — re-enable when fixed"]
fn apgc_fallback_graph_is_navigable() {
    if !gpu::gpu_init() {
        eprintln!("no CUDA device — skipping");
        return;
    }
    gpu::gpu_set_mode(gpu::ComputeMode::Turbo);
    std::env::set_var("DBSTRIKE_FORCE_APGC_FALLBACK", "1");

    let (n, dim) = (20_000usize, 64usize);
    let data = make_corpus(n, dim);

    let vi = VectorIndex::build_parallel_tiered(&data, dim, 8, false);

    // Honest recall gate: brute-force ground truth over the SAME distance fn,
    // queries drawn from the corpus itself (each has its self-match, which
    // makes this test GENEROUS — a fail here is unambiguous).
    let nq = 25;
    let k = 10;
    let mut hits = 0;
    let mut total = 0;
    for qi in 0..nq {
        let row = qi * 733 % n; // spread across clusters
        let q = &data[row * dim..(row + 1) * dim];
        let truth = brute_force_topk(&data, dim, q, k);
        let got: Vec<u64> = vi.search(q, k).into_iter().map(|(id, _)| id).collect();
        hits += truth.iter().filter(|t| got.contains(t)).count();
        total += k;
        if qi < 3 {
            eprintln!("row {row}: truth {:?} got {:?}", &truth[..3], &got[..3.min(got.len())]);
        }
    }
    let recall = hits as f32 / total as f32;
    eprintln!("fallback self-recall@{k} = {recall:.3}");
    assert!(
        recall >= 0.5,
        "APGC fallback produced a collapsed/near-useless graph: \
         recall@{k}={recall:.3} — BUG-1 class regression"
    );
}
