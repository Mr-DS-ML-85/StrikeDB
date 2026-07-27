/// OpusEdge stabilizers — prevent quality cliffs under aggressive compute reduction.

// ─── MPSR: Manifold-Preserving State Recycling ───────────────────────

pub struct Mpsr;

impl Mpsr {
    /// Project evicted KV tensor into compressed state vector
    /// and inject into subsequent SSM block.
    /// Returns the compressed state update.
    pub fn recycle(evicted_kv: &[f32], projection_dim: usize) -> Vec<f32> {
        // Lightweight projection: sum blocks of evicted KV
        let _block_size = (evicted_kv.len() / projection_dim).max(1);
        let mut compressed = vec![0.0f32; projection_dim];
        for (i, &v) in evicted_kv.iter().enumerate() {
            compressed[i % projection_dim] += v;
        }
        // Normalize
        let norm: f32 = compressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for c in &mut compressed {
                *c /= norm;
            }
        }
        compressed
    }
}

// ─── EBAR: Entropy-Buffered Autoregression ───────────────────────────

pub struct Ebar;

impl Ebar {
    /// Scale compute budget based on output entropy.
    /// When entropy is low (high confidence), reduce compute.
    /// When entropy is high, restore full compute.
    pub fn compute_budget(
        current_entropy: f32,
        prev_buffer: f32,
        base_budget: f32,
        sensitivity: f32,
    ) -> f32 {
        let scaled = base_budget * (1.0 + sensitivity * (current_entropy - prev_buffer));
        scaled.clamp(0.1 * base_budget, base_budget)
    }
}

// ─── SSR: Curvature-Adaptive Soft Spectral Relaxation ────────────────

pub struct Ssr;

impl Ssr {
    /// Soft thresholding operator for SVD truncation.
    /// Prevents perplexity explosions by smoothly decaying
    /// low-magnitude singular values instead of hard zeroing.
    pub fn soft_threshold(singular_value: f32, threshold: f32, elasticity: f32) -> f32 {
        let ratio = (singular_value - threshold) / (elasticity * threshold + 1e-10);
        let sigmoid = 1.0 / (1.0 + (-ratio).exp());
        singular_value * sigmoid
    }

    /// Apply soft spectral relaxation to a vector of singular values.
    pub fn apply(singular_values: &mut [f32], base_threshold: f32, layer_entropy: f32) {
        // Elasticity inversely proportional to layer entropy
        let elasticity = if layer_entropy > 0.01 { 1.0 / layer_entropy } else { 100.0 };
        for sv in singular_values.iter_mut() {
            *sv = Self::soft_threshold(*sv, base_threshold, elasticity);
        }
    }
}

// ─── IPSS: Information-Preserving Salience Smoothing ─────────────────

pub struct Ipss;

impl Ipss {
    /// Compute head salience as running temporal variance of activations.
    /// When salience falls below threshold, use linear-time fallback
    /// instead of quadratic attention.
    pub fn salience(
        current_activation: &[f32],
        historical_mean: &[f32],
        _ema_decay: f32,
    ) -> f32 {
        let dim = current_activation.len().max(1) as f32;
        let mut variance = 0.0f32;
        for (c, h) in current_activation.iter().zip(historical_mean.iter()) {
            let diff = c - *h;
            variance += diff * diff;
        }
        variance / dim
    }

    /// Check if head should use linear fallback.
    pub fn should_fallback(salience: f32, threshold: f32) -> bool {
        salience < threshold
    }

    /// Linear-time fallback for sub-threshold heads.
    /// Replace O(S²) attention with O(S) weighted average.
    pub fn linear_fallback(
        historical_keys: &[Vec<f32>],
        query: &[f32],
    ) -> Vec<f32> {
        if historical_keys.is_empty() {
            return vec![0.0; query.len()];
        }
        let mut result = vec![0.0f32; query.len()];
        let mut total_weight = 0.0f32;
        for k in historical_keys {
            let mut dot = 0.0f32;
            for (q, k_val) in query.iter().zip(k.iter()) {
                dot += q * k_val;
            }
            let weight = dot.exp(); // unnormalized softmax-like weight
            total_weight += weight;
            for (r, k_val) in result.iter_mut().zip(k.iter()) {
                *r += weight * k_val;
            }
        }
        if total_weight > 0.0 {
            for r in &mut result {
                *r /= total_weight;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpsr_recycle() {
        let kv = vec![1.0, 2.0, 3.0, 4.0];
        let compressed = Mpsr::recycle(&kv, 2);
        assert_eq!(compressed.len(), 2);
        // Should be normalized
        let norm: f32 = compressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ebar_budget() {
        let budget = Ebar::compute_budget(0.1, 0.5, 1.0, 0.5);
        assert!(budget < 1.0); // low entropy → reduced budget
        let budget2 = Ebar::compute_budget(2.0, 0.5, 1.0, 0.5);
        assert!(budget2 >= 1.0); // high entropy → full budget (clamped)
    }

    #[test]
    fn test_ssr_soft_threshold() {
        let sv = Ssr::soft_threshold(0.5, 1.0, 1.0);
        assert!(sv < 0.5); // below threshold → decayed
        let sv2 = Ssr::soft_threshold(2.0, 1.0, 1.0);
        assert!(sv2 > 1.0); // above threshold → mostly preserved (sigmoid ≈ 1.0)
    }

    #[test]
    fn test_ipss_fallback() {
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query = vec![0.5, 0.5];
        let result = Ipss::linear_fallback(&keys, &query);
        assert_eq!(result.len(), 2);
        // Should return weighted average
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
