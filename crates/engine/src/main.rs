/// Unified Inference Engine — Main Entry Point
///
/// Demonstrates the full VUGVA + OpusEdge + APGC pipeline:
/// 1. Build a mixed-precision kNN graph (APGC)
/// 2. Extract Δ signals from LLM hidden states (OpusEdge)
/// 3. Run KV-aware search with memory tiering (VUGVA)
///
/// Usage: cargo run --release -p engine

use engine::{Engine, EngineConfig};
use opusedge::ArchType;

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Unified Inference Engine — VUGVA + OpusEdge + APGC ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // ── Configuration ──
    let config = EngineConfig {
        arch: ArchType::Dense,
        graph_k: 16,
        graph_seed_ratio: 0.05,
        graph_outlier_ratio: 0.10,
        beam_width: 16,
        kv_prune_threshold: 0.5,
        ..Default::default()
    };

    let mut engine = Engine::new(config);

    // ── Step 1: Build mixed-precision kNN graph (APGC) ──
    println!("[1/4] Building mixed-precision kNN graph (APGC)...");
    let n = 1000;
    let dim = 128;
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| {
        (0..dim).map(|d| {
            ((i * 7 + d * 13) as f32 * 0.001).sin()
        }).collect()
    }).collect();
    engine.build_graph(&vectors);
    println!("  ✓ Graph built: {} nodes, k=16", n);

    // ── Step 2: Extract Δ signals (OpusEdge) ──
    println!("[2/4] Extracting Δ signals from hidden states (OpusEdge)...");
    let seq_len = 256;
    let hidden_dim = 128;
    let hidden_states: Vec<Vec<f32>> = (0..seq_len).map(|t| {
        (0..hidden_dim).map(|d| {
            (t as f32 * 0.01 + d as f32 * 0.001).sin() * 0.1
        }).collect()
    }).collect();
    let delta = engine.extract_signal(&hidden_states);
    let avg_delta = delta.scores.iter().sum::<f32>() / delta.seq_len as f32;
    println!("  ✓ Δ signal extracted: {} tokens, avg={:.4}, entropy={:.4}",
        delta.seq_len, avg_delta, delta.entropy());

    // ── Step 3: Run SelKV eviction (OpusEdge) ──
    println!("[3/4] Running SelKV KV cache eviction (OpusEdge)...");
    let eviction = opusedge::primitives::SelKV::evict(&delta, 0.875, seq_len);
    println!("  ✓ SelKV: {} retained, {} evicted ({:.1}% savings)",
        eviction.retained_indices.len(), eviction.evicted_indices.len(),
        eviction.memory_savings * 100.0);

    // ── Step 4: Run unified search (APGC + VUGVA) ──
    println!("[4/4] Running unified search (APGC + VUGVA)...");
    let hidden_states_short = vec![vec![0.5; 128]; 32];
    let result = engine.search(&vectors[0], &hidden_states_short, 10);
    println!("  ✓ Search complete:");
    println!("    Top-10: {:?}", &result.top_k[..5.min(result.top_k.len())]);
    println!("    Active heads: {}/32", result.active_heads);
    println!("    ΔRank: {}", result.rank);
    println!("    State compression: {:.1}%", result.compression_ratio * 100.0);
    println!("    VRAM: {} KB, RAM: {} KB", result.vram_usage / 1024, result.ram_usage / 1024);

    // ── Summary ──
    println!();
    println!("┌──────────────────────────────────────────────────────┐");
    println!("│ Pipeline Summary                                    │");
    println!("├──────────────────────────────────────────────────────┤");
    println!("│ VUGVA  : {} chunks, VRAM {:.1} KB, RAM {:.1} KB │",
        n, result.vram_usage as f64 / 1024.0, result.ram_usage as f64 / 1024.0);
    println!("│ OpusEdge: SelKV 87.5%, ΔRank {}, Heads {}/32   │",
        result.rank, result.active_heads);
    println!("│ APGC   : Mixed-precision kNN, {} nodes, k={}      │", n, 16);
    println!("└──────────────────────────────────────────────────────┘");
}
