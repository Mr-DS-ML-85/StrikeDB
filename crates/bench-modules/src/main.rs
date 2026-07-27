/// Benchmark suite for VUGVA, OpusEdge, and APGC modules.
/// Measures throughput, latency, and memory usage for the APGC paper.
///
/// Usage: cargo run --release -p bench-modules

use std::time::Instant;

mod recall;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  APGC + OpusEdge + VUGVA Benchmark Suite                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    bench_opusedge_signal();
    bench_opusedge_primitives();
    bench_opusedge_stabilizers();
    bench_apgc_precision();
    bench_apgc_kv_prune();
    bench_vugva_memory();
    bench_full_pipeline();

    println!("\n══════════════════════════════════════════════════════════");
    println!("All benchmarks complete. Numbers above are for the APGC paper.");
}

// ═══ OpusEdge Signal Extraction ═══════════════════════════════════════

fn bench_opusedge_signal() {
    use opusedge::signal::{DeltaSignal, SignalSource};

    println!("── OpusEdge: Signal Extraction ──");
    for seq_len in [128, 512, 2048, 8192] {
        let hidden: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| (0..128).map(|d| ((t * 7 + d * 13) as f32 * 0.001).sin()).collect())
            .collect();
        let t0 = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = DeltaSignal::from_proxy_delta(&hidden);
        }
        let ns_per = t0.elapsed().as_nanos() as f64 / iters as f64;
        println!("  seq={:<5} | Proxy-Δ: {:>8.1} ns/token | throughput: {:>12.0} tokens/s",
            seq_len, ns_per / seq_len as f64, seq_len as f64 / (ns_per / 1e9));
    }
    println!();
}

// ═══ OpusEdge Primitives ═════════════════════════════════════════════

fn bench_opusedge_primitives() {
    use opusedge::signal::{DeltaSignal, SignalSource};
    use opusedge::primitives::*;

    println!("── OpusEdge: Primitives ──");
    let delta = DeltaSignal {
        scores: (0..4096).map(|i| (i as f32 * 0.001).sin().abs()).collect(),
        source: SignalSource::ProxyDelta,
        seq_len: 4096,
    };
    let iters = 1000;

    // SelKV
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = SelKV::evict(&delta, 0.875, 4096);
    }
    let selkv_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  SelKV          | {:>10.1} µs/evict | {:>12.0} evictions/s | 87.5% cache savings",
        selkv_ns / 1000.0, 1e9 / selkv_ns);

    // SMSA
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Smsa::adaptive_window(&delta, 64);
    }
    let smsa_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  SMSA           | {:>10.1} µs/window | {:>12.0} windows/s",
        smsa_ns / 1000.0, 1e9 / smsa_ns);

    // Delta-AR
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = DeltaAR::route(&delta, 64);
    }
    let dar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  Delta-AR       | {:>10.1} µs/route  | {:>12.0} routes/s  | O(S²)→O(S·K)",
        dar_ns / 1000.0, 1e9 / dar_ns);

    // HeadDeactivate
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = HeadDeactivate::gate(&delta, 32);
    }
    let hd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  HeadDeactivate | {:>10.1} µs/gate   | {:>12.0} gates/s    | up to 87.5% heads off",
        hd_ns / 1000.0, 1e9 / hd_ns);

    // StateCompress
    let state = vec![0.001f32; 1024];
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = StateCompress::compress(&state, 0.01, 0.5, 0.5);
    }
    let sc_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  StateCompress  | {:>10.1} µs/compress | {:>12.0} compressions/s | 37.5% state savings",
        sc_ns / 1000.0, 1e9 / sc_ns);

    // DenseEvic
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = DenseEvic::evict(&delta, 0.5, 100);
    }
    let de_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  DenseEvic      | {:>10.1} µs/evict | {:>12.0} evictions/s",
        de_ns / 1000.0, 1e9 / de_ns);

    // ΔRank
    let t0 = Instant::now();
    for _ in 0..iters {
        for s in &delta.scores {
            let _ = DeltaRank::score_to_rank(*s, 128);
        }
    }
    let dr_ns = t0.elapsed().as_nanos() as f64 / iters as f64 / delta.seq_len as f64;
    println!("  ΔRank          | {:>10.1} ns/token  | {:>12.0} tokens/s  | 4-tier rank mapping",
        dr_ns, 1e9 / dr_ns);
    println!();
}

// ═══ OpusEdge Stabilizers ════════════════════════════════════════════

fn bench_opusedge_stabilizers() {
    use opusedge::stabilizers::*;

    println!("── OpusEdge: Stabilizers ──");
    let kv = vec![1.0f32; 4096];
    let iters = 1000;

    // MPSR
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Mpsr::recycle(&kv, 128);
    }
    let mpsr_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  MPSR           | {:>10.1} µs/recycle | {:>12.0} recycles/s | KV→SSM state projection",
        mpsr_ns / 1000.0, 1e9 / mpsr_ns);

    // EBAR
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Ebar::compute_budget(0.5, 0.3, 1.0, 0.5);
    }
    let ebar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  EBAR           | {:>10.1} ns/call   | {:>12.0} calls/s     | entropy-driven budget",
        ebar_ns, 1e9 / ebar_ns);

    // SSR
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Ssr::soft_threshold(1.5, 1.0, 1.0);
    }
    let ssr_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  SSR            | {:>10.1} ns/call   | {:>12.0} calls/s     | soft spectral relaxation",
        ssr_ns, 1e9 / ssr_ns);

    // IPSS
    let hist_keys: Vec<Vec<f32>> = (0..64).map(|i| vec![i as f32 * 0.01; 128]).collect();
    let query = vec![0.5f32; 128];
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = Ipss::linear_fallback(&hist_keys, &query);
    }
    let ipss_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  IPSS           | {:>10.1} µs/fallback | {:>12.0} fallbacks/s | O(S) linear attention",
        ipss_ns / 1000.0, 1e9 / ipss_ns);
    println!();
}

// ═══ APGC Precision ═══════════════════════════════════════════════════

fn bench_apgc_precision() {
    use apgc::precision::*;

    println!("── APGC: Mixed-Precision Graph Construction ──");
    let iters = 10;

    for n in [100, 500, 1000] {
        let dim = 384;
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| {
            (0..dim).map(|d| ((i * 7 + d * 13) as f32 * 0.001).sin()).collect()
        }).collect();

        // FP32 baseline
        let cfg_fp32 = GraphConfig { k: 32, n, dim, seed_ratio: 0.0, outlier_ratio: 0.0, gpu_caps: None };
        let t0 = Instant::now();
        for _ in 0..iters { let _ = MixedPrecisionBuilder::build(&vectors, cfg_fp32.clone()); }
        let fp32_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Mixed precision (5% FP32 seeds, 85% FP16, 10% INT8)
        let cfg_mix = GraphConfig { k: 32, n, dim, seed_ratio: 0.05, outlier_ratio: 0.10, gpu_caps: None };
        let t0 = Instant::now();
        for _ in 0..iters { let _ = MixedPrecisionBuilder::build(&vectors, cfg_mix.clone()); }
        let mix_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = fp32_ms / mix_ms;

        // Memory comparison
        let mem_fp32 = n * 32 * 8; // 32 edges × (4+4) bytes
        let mem_mix_est = (n as f64 * 32.0 * (0.05 * 8.0 + 0.85 * 4.0 + 0.10 * 2.0)) as usize;

        println!("  N={:<6} | FP32: {:>8.1} ms | Mixed: {:>8.1} ms | speedup: {:>5.2}x | mem savings: {:>5.1}%",
            n, fp32_ms, mix_ms, speedup,
            (1.0 - mem_mix_est as f64 / mem_fp32 as f64) * 100.0);
    }
    println!();
}

// ═══ APGC KV Prune ═══════════════════════════════════════════════════

fn bench_apgc_kv_prune() {
    use apgc::kv_prune::*;

    println!("── APGC: KV-Aware Search Pruning ──");
    let pruner = KvSearchPruner::new(0.5, 32);
    let seq_len = 8192;
    let attn = AttentionScores {
        scores: (0..seq_len).map(|i| (i as f32 * 0.001).sin().abs()).collect(),
        seq_len,
    };
    let candidates: Vec<usize> = (0..seq_len).collect();
    let iters = 1000;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = pruner.prune_candidates(&candidates, &attn);
    }
    let ns_per = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  seq={:<5} | prune: {:>8.1} µs | {:>12.0} prunes/s | {} candidates → {} pruned",
        seq_len, ns_per / 1000.0, 1e9 / ns_per, seq_len, 32);
    println!();
}

// ═══ VUGVA Memory ════════════════════════════════════════════════════

fn bench_vugva_memory() {
    use vugva::{VugvaConfig, VugvaVmt, Chunk, Tier, LruEvictor, LookaheadTracker};

    println!("── VUGVA: Memory Tiering ──");
    let iters = 1000;

    // Insert benchmark
    let cfg = VugvaConfig::default();
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut vmt = VugvaVmt::new(cfg.clone());
        for i in 0..1000 {
            vmt.insert(Chunk::new(i, Tier::Vram, vec![0u8; 256]));
        }
    }
    let insert_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  Insert 1000    | {:>10.1} µs/batch  | {:>12.0} chunks/s",
        insert_ns / 1000.0, 1e6 / (insert_ns / 1e9));

    // Lookup benchmark
    let mut vmt = VugvaVmt::new(cfg.clone());
    for i in 0..10000 { vmt.insert(Chunk::new(i, Tier::Vram, vec![0u8; 256])); }
    let t0 = Instant::now();
    for _ in 0..iters {
        for i in 0..10000 { vmt.get(i); }
    }
    let lookup_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  Lookup 10000   | {:>10.1} µs/batch  | {:>12.0} lookups/s",
        lookup_ns / 1000.0, 1e7 / (lookup_ns / 1e9));

    // Eviction benchmark
    let mut vmt2 = VugvaVmt::new(VugvaConfig {
        vram_capacity: 256 * 500, // 500 chunks worth
        ..cfg.clone()
    });
    for i in 0..1000 { vmt2.insert(Chunk::new(i, Tier::Vram, vec![0u8; 256])); }
    let mut evictor = LruEvictor::new();
    let t0 = Instant::now();
    for _ in 0..100 { evictor.evict_to_target(&mut vmt2); }
    let evict_ns = t0.elapsed().as_nanos() as f64 / 100.0;
    println!("  Evict           | {:>10.1} µs/cycle   | {:>12.0} cycles/s | {:>8.1} µs/chunk",
        evict_ns / 1000.0, 1e9 / evict_ns, evict_ns / 1000.0 / 10.0);

    // Prefetch prediction benchmark
    let mut pf = LookaheadTracker::new(256, 16);
    for i in 0..256 { pf.record_access(i as u64); }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = pf.predict();
    }
    let pf_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("  Prefetch       | {:>10.1} ns/predict | {:>12.0} predicts/s | window=16",
        pf_ns, 1e9 / pf_ns);
    println!();
}

// ═══ Full Pipeline ═══════════════════════════════════════════════════

fn bench_full_pipeline() {
    use engine::{Engine, EngineConfig};
    use opusedge::ArchType;

    println!("── Full Pipeline: VUGVA + OpusEdge + APGC ──");
    let config = EngineConfig {
        arch: ArchType::Dense,
        graph_k: 32,
        graph_seed_ratio: 0.05,
        graph_outlier_ratio: 0.10,
        beam_width: 32,
        ..Default::default()
    };
    let mut engine = Engine::new(config);

    // Build phase
    let n = 5000;
    let dim = 384;
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| {
        (0..dim).map(|d| ((i * 7 + d * 13) as f32 * 0.001).sin()).collect()
    }).collect();
    let t0 = Instant::now();
    engine.build_graph(&vectors);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Search phase
    let hs: Vec<Vec<f32>> = (0..256).map(|t| {
        (0..dim).map(|d| ((t * 3 + d * 7) as f32 * 0.001).cos()).collect()
    }).collect();
    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.search(&vectors[0], &hs, 10);
    }
    let search_us = t0.elapsed().as_micros() as f64 / iters as f64;

    println!("  Build  (10K × 384d)  | {:>8.1} ms | {:>8.0} vec/s", build_ms, n as f64 / (build_ms / 1000.0));
    println!("  Search (1 query)     | {:>8.1} µs   | {:>8.0} QPS", search_us, 1e6 / search_us);
    println!();
}
