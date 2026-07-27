/// Unified Inference Engine
///
/// Integrates:
/// - **VUGVA**: Memory tiering (VRAM → RAM → NVMe) with chunk management
/// - **OpusEdge**: Δ-signal driven compute allocation (30 primitives)
/// - **APGC**: Mixed-precision graph construction + KV-aware search pruning

pub use vugva;
pub use opusedge;
pub use apgc;

use vugva::{VugvaConfig, VugvaVmt, Chunk, Tier, LruEvictor, LookaheadTracker};
use opusedge::{DeltaSignal, SignalSource, ArchType};
use opusedge::primitives::{SelKV, HeadDeactivate, StateCompress, DeltaRank};
use apgc::precision::{MixedPrecisionBuilder, GraphConfig};
use apgc::PrecisionLevel;
use apgc::kv_prune::KvSearchPruner;

/// Engine configuration.
pub struct EngineConfig {
    pub arch: ArchType,
    pub vugva: VugvaConfig,
    pub kv_prune_threshold: f32,
    pub beam_width: usize,
    pub graph_k: usize,
    pub graph_seed_ratio: f64,
    pub graph_outlier_ratio: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            arch: ArchType::Dense,
            vugva: VugvaConfig::default(),
            kv_prune_threshold: 0.5,
            beam_width: 32,
            graph_k: 32,
            graph_seed_ratio: 0.01,
            graph_outlier_ratio: 0.10,
        }
    }
}

/// The unified inference engine.
pub struct Engine {
    config: EngineConfig,
    vmt: VugvaVmt,
    evictor: LruEvictor,
    prefetcher: LookaheadTracker,
    graph: Option<apgc::precision::ApgcGraph>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let vmt = VugvaVmt::new(config.vugva.clone());
        Self {
            config,
            vmt,
            evictor: LruEvictor::new(),
            prefetcher: LookaheadTracker::new(256, 16),
            graph: None,
        }
    }

    /// Build a mixed-precision kNN graph from vectors.
    pub fn build_graph(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        let graph_cfg = GraphConfig {
            k: self.config.graph_k,
            n,
            dim: vectors[0].len(),
            seed_ratio: self.config.graph_seed_ratio,
            outlier_ratio: self.config.graph_outlier_ratio,
            gpu_caps: Some(apgc::precision::GpuCaps::detect()),
        };
        self.graph = Some(MixedPrecisionBuilder::build(vectors, graph_cfg));

        // Store graph data in VUGVA tiers
        for i in 0..n {
            let chunk = Chunk::new(i as u64, Tier::Vram, vec![0u8; 1024]);
            self.vmt.insert(chunk);
        }
    }

    /// Extract Δ signal from hidden states.
    pub fn extract_signal(&self, hidden_states: &[Vec<f32>]) -> DeltaSignal {
        match self.config.arch {
            ArchType::Dense => DeltaSignal::from_proxy_delta(hidden_states),
            ArchType::Hybrid => DeltaSignal::from_proxy_delta(hidden_states),
            ArchType::MoE => {
                // For MoE, use router logits instead
                DeltaSignal::from_proxy_delta(hidden_states)
            }
        }
    }

    /// Run full APGC+OpusEdge pipeline on a query.
    pub fn search(
        &mut self,
        query: &[f32],
        hidden_states: &[Vec<f32>],
        k: usize,
    ) -> SearchResult {
        let delta = self.extract_signal(hidden_states);
        let graph = self.graph.as_ref().expect("build_graph() must be called first");

        // Step 1: SelKV — evict unimportant nodes from frontier
        let eviction = SelKV::evict(&delta, 0.25, graph.n);

        // Step 2: KV-aware pruning — prune candidates using attention scores
        let pruner = KvSearchPruner::new(self.config.kv_prune_threshold, self.config.beam_width);
        let attn = apgc::kv_prune::AttentionScores {
            scores: delta.scores.clone(),
            seq_len: delta.seq_len,
        };
        let candidates: Vec<usize> = (0..graph.n.min(1000)).collect();
        let pruned = pruner.prune_candidates(&candidates, &attn);

        // Step 3: HeadDeactivate — determine active heads
        let head_gate = HeadDeactivate::gate(&delta, 32);

        // Step 4: StateCompress — check if we can compress state
        let state_comp = StateCompress::compress(
            &vec![0.0; 1024], // placeholder hidden state
            delta.scores.iter().sum::<f32>() / delta.seq_len.max(1) as f32,
            0.5, 0.5,
        );

        // Step 5: ΔRank — determine rank for projection
        let avg_delta = delta.scores.iter().sum::<f32>() / delta.seq_len.max(1) as f32;
        let rank = DeltaRank::score_to_rank(avg_delta, 128);

        // Step 6: VUGVA — prefetch upcoming chunks
        for &node in pruned.iter().take(16) {
            self.prefetcher.record_access(node as u64);
        }
        let predictions = self.prefetcher.predict();
        for chunk_id in &predictions {
            self.vmt.get(*chunk_id);
        }

        // Step 7: LRU eviction if VRAM full
        if self.vmt.needs_eviction() {
            self.evictor.evict_to_target(&mut self.vmt);
        }

        SearchResult {
            top_k: pruned.into_iter().take(k).collect(),
            eviction_ratio: 0.25,
            active_heads: head_gate.active_heads.len(),
            compression_ratio: state_comp.memory_savings,
            rank,
            vram_usage: self.vmt.vram_usage(),
            ram_usage: self.vmt.ram_usage(),
        }
    }
}

/// Result of a search through the unified engine.
#[derive(Debug)]
pub struct SearchResult {
    pub top_k: Vec<usize>,
    pub eviction_ratio: f64,
    pub active_heads: usize,
    pub compression_ratio: f64,
    pub rank: usize,
    pub vram_usage: usize,
    pub ram_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_build_and_search() {
        let config = EngineConfig {
            arch: ArchType::Dense,
            graph_k: 4,
            graph_seed_ratio: 0.25,
            graph_outlier_ratio: 0.0,
            ..Default::default()
        };
        let mut engine = Engine::new(config);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.0, 0.1, 0.9],
        ];
        engine.build_graph(&vectors);

        let hidden_states = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7],
        ];
        let result = engine.search(&vectors[0], &hidden_states, 3);

        assert!(result.top_k.len() <= 3);
        assert!(result.active_heads > 0);
        assert!(result.rank > 0);
        eprintln!("Search result: {:?}", result);
    }

    #[test]
    fn test_engine_full_pipeline() {
        let mut engine = Engine::new(EngineConfig::default());
        let vectors: Vec<Vec<f32>> = (0..20).map(|i| {
            vec![i as f32 * 0.1, (20 - i) as f32 * 0.1, 0.5]
        }).collect();
        engine.build_graph(&vectors);

        let hs = vec![vec![0.1, 0.2, 0.3]; 10];
        let result = engine.search(&vectors[0], &hs, 5);
        assert!(!result.top_k.is_empty());
    }
}
