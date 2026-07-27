/// Δ Signal extraction — the core importance signal for all OpusEdge primitives.
///
/// Three architecture families, three signal sources:
/// - Hybrid (Falcon-H1, Jamba): native SSM selectivity Δ (O(1) cost)
/// - Dense (Qwen, LLaMA): Proxy-Δ via RMS hidden-state drift (O(L) cost)
/// - MoE (Mixtral, OLMoE): Router-Gated Importance via softmax entropy

/// Per-token importance signal.
#[derive(Debug, Clone)]
pub struct DeltaSignal {
    /// Importance scores per token, shape [seq_len]
    pub scores: Vec<f32>,
    /// Signal source used
    pub source: SignalSource,
    /// Sequence length
    pub seq_len: usize,
}

/// How the Δ signal was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSource {
    /// Native SSM selectivity (O(1) per token, zero-cost)
    NativeDelta,
    /// Proxy-Δ via RMS hidden-state drift (O(L) per token)
    ProxyDelta,
    /// Router-Gated Importance via softmax entropy
    RouterIR,
}

impl DeltaSignal {
    /// Compute Proxy-Δ for dense models.
    /// Proxy-Δ = RMS(hidden_state[t] - hidden_state[t-1]) / sqrt(hidden_dim)
    /// This is the O(L) approximation of native SSM selectivity.
    pub fn from_proxy_delta(hidden_states: &[Vec<f32>]) -> Self {
        let seq_len = hidden_states.len();
        if seq_len < 2 {
            return Self { scores: vec![0.0; seq_len], source: SignalSource::ProxyDelta, seq_len };
        }
        let hidden_dim = hidden_states[0].len() as f32;
        let scale = 1.0 / hidden_dim.sqrt();

        let mut scores = Vec::with_capacity(seq_len);
        scores.push(0.0); // first token has no previous state

        for t in 1..seq_len {
            let mut ssd = 0.0f32;
            for d in 0..hidden_states[t].len() {
                let diff = hidden_states[t][d] - hidden_states[t - 1][d];
                ssd += diff * diff;
            }
            scores.push((ssd * scale).sqrt());
        }

        Self { scores, source: SignalSource::ProxyDelta, seq_len }
    }

    /// Compute Router-IR for MoE models.
    /// Router-IR = 1 - entropy(softmax(router_logits)) / log(num_experts)
    /// Range: [0, 1] where 1 = high confidence (specialized), 0 = generic
    pub fn from_router_ir(router_logits: &[Vec<f32>]) -> Self {
        let seq_len = router_logits.len();
        let mut scores = Vec::with_capacity(seq_len);

        for logits in router_logits {
            let num_experts = logits.len() as f32;
            if num_experts <= 1.0 {
                scores.push(0.0);
                continue;
            }
            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
            let entropy: f32 = logits.iter()
                .map(|l| {
                    let p = (l - max_logit).exp() / exp_sum;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            let max_entropy = num_experts.ln();
            let ir = if max_entropy > 0.0 { 1.0 - entropy / max_entropy } else { 0.0 };
            scores.push(ir);
        }

        Self { scores, source: SignalSource::RouterIR, seq_len }
    }

    /// Normalize scores to [0, 1] range.
    pub fn normalize(&mut self) {
        let max_val = self.scores.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for s in &mut self.scores {
                *s /= max_val;
            }
        }
    }

    /// Get the top-k most important token indices.
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(f32, usize)> = self.scores.iter().enumerate()
            .map(|(i, &s)| (s, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Get the bottom-p% least important token indices (eviction candidates).
    pub fn bottom_percent(&self, p: f32) -> Vec<usize> {
        assert!(p >= 0.0 && p <= 1.0, "p must be in [0, 1]");
        let k = (self.seq_len as f32 * p) as usize;
        let mut indexed: Vec<(f32, usize)> = self.scores.iter().enumerate()
            .map(|(i, &s)| (s, i)).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Compute Shannon entropy of the signal distribution.
    pub fn entropy(&self) -> f32 {
        let sum: f32 = self.scores.iter().sum();
        if sum <= 0.0 { return 0.0; }
        self.scores.iter()
            .map(|&s| {
                if s > 0.0 {
                    let p = s / sum;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_delta() {
        let states = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.2, 3.3],
            vec![1.5, 2.5, 3.5],
        ];
        let delta = DeltaSignal::from_proxy_delta(&states);
        assert_eq!(delta.seq_len, 3);
        assert_eq!(delta.scores[0], 0.0); // first token has no drift
        assert!(delta.scores[1] > 0.0);   // second token has drift
        assert!(delta.scores[2] > delta.scores[1]); // larger drift
    }

    #[test]
    fn test_router_ir() {
        let logits = vec![
            vec![10.0, 0.0, 0.0], // high confidence → IR near 1
            vec![1.0, 1.0, 1.0],  // uniform → IR near 0
        ];
        let ir = DeltaSignal::from_router_ir(&logits);
        assert!(ir.scores[0] > 0.9); // highly peaked
        assert!(ir.scores[1] < 0.1); // uniform
    }

    #[test]
    fn test_top_k() {
        let delta = DeltaSignal { scores: vec![0.1, 0.9, 0.5, 0.3], source: SignalSource::ProxyDelta, seq_len: 4 };
        let top2 = delta.top_k(2);
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    fn test_bottom_percent() {
        let delta = DeltaSignal { scores: vec![0.1, 0.9, 0.5, 0.3], source: SignalSource::ProxyDelta, seq_len: 4 };
        let bottom = delta.bottom_percent(0.5);
        assert_eq!(bottom.len(), 2);
    }
}
