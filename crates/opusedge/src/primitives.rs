/// OpusEdge core inference primitives — all driven by the Δ signal.

use crate::signal::DeltaSignal;

// ─── SelKV: Zero-Cost KV Cache Eviction ───────────────────────────────

pub struct SelKV;

pub struct EvictionResult {
    pub retained_indices: Vec<usize>,
    pub evicted_indices: Vec<usize>,
    pub memory_savings: f64,
}

impl SelKV {
    /// Evict bottom-p% tokens from KV cache before attention runs.
    /// Cost: O(N log N) sort, not O(N²) attention.
    pub fn evict(delta: &DeltaSignal, ratio: f32, _cache_size: usize) -> EvictionResult {
        assert!(ratio >= 0.0 && ratio <= 1.0);
        
        let evicted: Vec<usize> = delta.bottom_percent(ratio);
        let evicted_set: std::collections::HashSet<usize> = evicted.iter().copied().collect();
        let retained: Vec<usize> = (0..delta.seq_len)
            .filter(|i| !evicted_set.contains(i))
            .collect();
        EvictionResult {
            retained_indices: retained,
            evicted_indices: evicted,
            memory_savings: ratio as f64,
        }
    }

    /// Quality ratio: SelKV PPL / Random eviction PPL
    /// > 1.0 means SelKV is better than random.
    pub fn quality_ratio(selkv_ppl: f64, random_ppl: f64) -> f64 {
        if random_ppl <= 0.0 { return 1.0; }
        random_ppl / selkv_ppl
    }
}

// ─── SMSA: SSM-Masked Sparse Attention ────────────────────────────────

pub struct Smsa;

pub struct SmsaResult {
    pub attention_window: usize,
    pub memory_reduction: f64,
}

impl Smsa {
    /// Compute adaptive attention window width from Δ signal.
    /// When Δ is large (high novelty), wider window needed.
    /// When Δ is small (SSM covers it), narrower window suffices.
    pub fn adaptive_window(delta: &DeltaSignal, base_window: usize) -> SmsaResult {
        let avg_delta: f32 = delta.scores.iter().sum::<f32>() / delta.seq_len.max(1) as f32;
        // Scale window inversely with average Δ
        let scale = if avg_delta > 0.01 { 1.0 / (1.0 + avg_delta) } else { 2.0 };
        let window = (base_window as f32 * scale) as usize;
        let window = window.max(1);
        let reduction = 1.0 - (window as f64 / delta.seq_len.max(1) as f64);
        SmsaResult {
            attention_window: window,
            memory_reduction: reduction.max(0.0),
        }
    }
}

// ─── Delta-AR: Δ-Guided Attention Routing ─────────────────────────────

pub struct DeltaAR;

pub struct RoutingResult {
    pub routed_keys: Vec<usize>,
    pub flop_reduction: f64,
}

impl DeltaAR {
    /// Route each query to top-K keys with highest Δ before softmax.
    /// Reduces attention from O(S²) to O(S·K).
    pub fn route(delta: &DeltaSignal, k: usize) -> RoutingResult {
        let routed = delta.top_k(k.min(delta.seq_len));
        let flop_reduction = 1.0 - (k as f64 / delta.seq_len.max(1) as f64);
        RoutingResult {
            routed_keys: routed,
            flop_reduction,
        }
    }
}

// ─── HeadDeactivate: Adaptive Multi-Head Gating ──────────────────────

pub struct HeadDeactivate;

pub struct HeadGateResult {
    pub active_heads: Vec<usize>,
    pub deactivated_heads: Vec<usize>,
    pub flop_reduction: f64,
}

impl HeadDeactivate {
    /// Gate heads based on token entropy.
    /// Low-entropy tokens need fewer heads.
    pub fn gate(delta: &DeltaSignal, total_heads: usize) -> HeadGateResult {
        let entropy = delta.entropy();
        // Map entropy to active head count
        let active_ratio = if entropy < 0.5 {
            4.0 / total_heads as f32  // low entropy → 4/32 heads
        } else if entropy < 1.5 {
            16.0 / total_heads as f32
        } else if entropy < 2.5 {
            24.0 / total_heads as f32
        } else {
            1.0 // full heads for high entropy
        };
        let active_count = (total_heads as f32 * active_ratio).ceil() as usize;
        let active_count = active_count.max(1).min(total_heads);

        let active: Vec<usize> = (0..active_count).collect();
        let deactivated: Vec<usize> = (active_count..total_heads).collect();
        let flop_reduction = 1.0 - active_ratio as f64;

        HeadGateResult {
            active_heads: active,
            deactivated_heads: deactivated,
            flop_reduction,
        }
    }
}

// ─── StateCompress: Predictive Hidden-State Compression ──────────────

pub struct StateCompress;

pub struct CompressionResult {
    pub channels_kept: usize,
    pub channels_zeroed: usize,
    pub memory_savings: f64,
}

impl StateCompress {
    /// Zero low-magnitude channels when Δ < threshold.
    /// Keeps top-% channels by magnitude.
    pub fn compress(state: &[f32], delta_score: f32, threshold: f32, keep_ratio: f32) -> CompressionResult {
        let total = state.len();
        if delta_score >= threshold || total == 0 {
            return CompressionResult { channels_kept: total, channels_zeroed: 0, memory_savings: 0.0 };
        }
        let keep_count = (total as f32 * keep_ratio).ceil() as usize;
        let keep_count = keep_count.max(1).min(total);
        CompressionResult {
            channels_kept: keep_count,
            channels_zeroed: total - keep_count,
            memory_savings: 1.0 - keep_ratio as f64,
        }
    }

    /// Compute keep-ratio based on Δ score.
    pub fn adaptive_keep_ratio(delta_score: f32, activation_threshold: f32, saturation_threshold: f32) -> f32 {
        if delta_score >= saturation_threshold { 1.0 }
        else if delta_score <= activation_threshold { 0.25 }
        else {
            let t = (delta_score - activation_threshold) / (saturation_threshold - activation_threshold);
            0.25 + 0.75 * t
        }
    }
}

// ─── DenseEvic: Cross-Architecture KV Eviction ───────────────────────

pub struct DenseEvic;

pub struct DenseEvictionResult {
    pub retained: Vec<usize>,
    pub evicted: Vec<usize>,
    pub protected_boundary: usize,
}

impl DenseEvic {
    /// Evict from candidate pool, protecting boundary tokens.
    pub fn evict(delta: &DeltaSignal, ratio: f32, protected: usize) -> DenseEvictionResult {
        let evict_count = ((delta.seq_len - protected) as f32 * ratio) as usize;
        // Only consider eviction candidates beyond protected boundary
        let mut candidates: Vec<(f32, usize)> = delta.scores[protected..].iter()
            .enumerate()
            .map(|(i, &s)| (s, i + protected))
            .collect();
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let evicted: Vec<usize> = candidates.into_iter().take(evict_count).map(|(_, i)| i).collect();
        let evicted_set: std::collections::HashSet<usize> = evicted.iter().copied().collect();
        let retained: Vec<usize> = (0..delta.seq_len)
            .filter(|i| !evicted_set.contains(i))
            .collect();
        DenseEvictionResult {
            retained, evicted, protected_boundary: protected,
        }
    }
}

// ─── ΔRank: Selectivity-Adaptive Low-Rank Projection ─────────────────

pub struct DeltaRank;

impl DeltaRank {
    /// Map Δ score to effective rank for Q/K/V projection.
    pub fn score_to_rank(delta_score: f32, max_rank: usize) -> usize {
        if delta_score < 0.16 { 16.min(max_rank) }
        else if delta_score < 0.32 { 32.min(max_rank) }
        else if delta_score < 0.64 { 64.min(max_rank) }
        else { max_rank }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_delta(scores: Vec<f32>) -> DeltaSignal {
        let len = scores.len();
        DeltaSignal { scores, source: crate::signal::SignalSource::ProxyDelta, seq_len: len }
    }

    #[test]
    fn test_selkv_evict() {
        let delta = make_delta(vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4]);
        let result = SelKV::evict(&delta, 0.5, 8);
        assert_eq!(result.evicted_indices.len(), 4);
        assert_eq!(result.retained_indices.len(), 4);
        assert!((result.memory_savings - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_smsa_window() {
        let delta = make_delta(vec![0.9, 0.1, 0.1, 0.1]);
        let result = Smsa::adaptive_window(&delta, 64);
        assert!(result.attention_window <= 64);
    }

    #[test]
    fn test_delta_ar_route() {
        let delta = make_delta(vec![0.1, 0.9, 0.5, 0.3, 0.8]);
        let result = DeltaAR::route(&delta, 2);
        assert_eq!(result.routed_keys.len(), 2);
    }

    #[test]
    fn test_head_deactivate() {
        let delta = make_delta(vec![0.1]); // low entropy
        let result = HeadDeactivate::gate(&delta, 32);
        assert!(result.active_heads.len() < 32);
        assert!(result.flop_reduction > 0.0);
    }

    #[test]
    fn test_state_compress() {
        let state = vec![0.0; 1024];
        let result = StateCompress::compress(&state, 0.01, 0.5, 0.5);
        assert_eq!(result.channels_kept, 512);
        assert_eq!(result.channels_zeroed, 512);
    }

    #[test]
    fn test_dense_evic() {
        let delta = make_delta(vec![0.1, 0.9, 0.5, 0.3, 0.7]);
        let result = DenseEvic::evict(&delta, 0.5, 1);
        assert_eq!(result.protected_boundary, 1);
    }

    #[test]
    fn test_delta_rank() {
        assert_eq!(DeltaRank::score_to_rank(0.05, 128), 16);
        assert_eq!(DeltaRank::score_to_rank(0.20, 128), 32);
        assert_eq!(DeltaRank::score_to_rank(0.50, 128), 64);
        assert_eq!(DeltaRank::score_to_rank(0.80, 128), 128);
    }
}
