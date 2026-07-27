/// KV-Aware Search Pruning — uses attention patterns to guide graph traversal.

/// Attention scores per token from an LLM inference engine.
pub struct AttentionScores {
    pub scores: Vec<f32>,  // per-token importance
    pub seq_len: usize,
}

/// KV-aware search pruner that uses attention patterns to select
/// which graph nodes to expand during search.
pub struct KvSearchPruner {
    /// Precision threshold: nodes above this get FP32 distance
    pub precision_threshold: f32,
    /// Beam width for graph traversal
    pub beam_width: usize,
}

impl KvSearchPruner {
    pub fn new(precision_threshold: f32, beam_width: usize) -> Self {
        Self { precision_threshold, beam_width }
    }

    /// Given a set of candidate node indices, prune them based on attention scores.
    /// High-attention nodes are kept at full precision.
    /// Low-attention nodes are deprioritized.
    pub fn prune_candidates(
        &self,
        candidates: &[usize],
        attention: &AttentionScores,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates.iter()
            .filter_map(|&node| {
                if node < attention.seq_len {
                    Some((attention.scores[node], node))
                } else {
                    None
                }
            })
            .collect();
        // Sort by attention score descending (highest attention = most important)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(self.beam_width).map(|(_, n)| n).collect()
    }

    /// Determine precision for a node based on its attention score.
    pub fn node_precision(&self, node: usize, attention: &AttentionScores) -> &str {
        if node < attention.seq_len && attention.scores[node] > self.precision_threshold {
            "FP32"
        } else {
            "INT8"
        }
    }

    /// Selective search: expand only high-attention nodes at FP32,
    /// skip low-attention nodes entirely.
    pub fn selective_expand(
        &self,
        frontier: &[usize],
        attention: &AttentionScores,
    ) -> Vec<usize> {
        frontier.iter()
            .filter(|&&node| {
                node < attention.seq_len && attention.scores[node] > self.precision_threshold * 0.5
            })
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_candidates() {
        let pruner = KvSearchPruner::new(0.5, 3);
        let attn = AttentionScores {
            scores: vec![0.1, 0.9, 0.3, 0.8, 0.2],
            seq_len: 5,
        };
        let candidates = vec![0, 1, 2, 3, 4];
        let pruned = pruner.prune_candidates(&candidates, &attn);
        assert_eq!(pruned.len(), 3);
        // Highest attention nodes should be kept
        assert!(pruned.contains(&1)); // score 0.9
        assert!(pruned.contains(&3)); // score 0.8
    }

    #[test]
    fn test_node_precision() {
        let pruner = KvSearchPruner::new(0.5, 3);
        let attn = AttentionScores {
            scores: vec![0.1, 0.9, 0.3],
            seq_len: 3,
        };
        assert_eq!(pruner.node_precision(1, &attn), "FP32"); // high attention
        assert_eq!(pruner.node_precision(0, &attn), "INT8"); // low attention
    }

    #[test]
    fn test_selective_expand() {
        let pruner = KvSearchPruner::new(0.5, 10);
        let attn = AttentionScores {
            scores: vec![0.1, 0.9, 0.3, 0.8, 0.2],
            seq_len: 5,
        };
        let frontier = vec![0, 1, 2, 3, 4];
        let expanded = pruner.selective_expand(&frontier, &attn);
        // Only high-attention nodes should be expanded
        assert!(expanded.contains(&1));
        assert!(expanded.contains(&3));
        assert!(!expanded.contains(&0)); // below threshold * 0.5
    }
}
