/// Mixed-precision graph construction for kNN graphs.
/// Supports FP32, FP16, BF16, FP8, and INT8 precision levels.

/// Precision levels for mixed-precision graph construction.
/// OpusEdge applies to FP32, FP16, BF16, and FP8 — all natively supported.
/// GPU capability detection determines which formats are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrecisionLevel {
    Int8,   // 1 byte  — outlier nodes (bottom 10%)
    Fp8,    // 1 byte  — near-outlier nodes (10-20%)
    Fp16,   // 2 bytes — majority of vectors
    Bf16,   // 2 bytes — alternative to FP16 (wider range)
    Fp32,   // 4 bytes — seed vectors (top 1%)
}

impl PrecisionLevel {
    /// Bytes per element at this precision.
    pub fn bytes(&self) -> usize {
        match self {
            PrecisionLevel::Int8 => 1,
            PrecisionLevel::Fp8 => 1,
            PrecisionLevel::Fp16 => 2,
            PrecisionLevel::Bf16 => 2,
            PrecisionLevel::Fp32 => 4,
        }
    }

    /// Name for display.
    pub fn name(&self) -> &'static str {
        match self {
            PrecisionLevel::Int8 => "INT8",
            PrecisionLevel::Fp8 => "FP8",
            PrecisionLevel::Fp16 => "FP16",
            PrecisionLevel::Bf16 => "BF16",
            PrecisionLevel::Fp32 => "FP32",
        }
    }

    /// Whether this format requires tensor cores (FP16/BF16/FP8).
    pub fn needs_tensor_cores(&self) -> bool {
        matches!(self, PrecisionLevel::Fp16 | PrecisionLevel::Bf16 | PrecisionLevel::Fp8)
    }
}

/// GPU capability detection for precision format support.
/// Different GPU generations support different precision formats:
/// - V100 (Volta, sm_70): FP32, FP16, INT8. NO FP8, NO BF16.
/// - A100 (Ampere, sm_80): FP32, FP16, BF16, INT8. FP8 limited.
/// - H100 (Hopper, sm_90): FP32, FP16, BF16, INT8, FP8 (full).
/// - RTX 4060/4090 (Ada, sm_89): FP32, FP16, BF16, INT8, FP8.
#[derive(Debug, Clone, Copy)]
pub struct GpuCaps {
    pub compute_capability: (u32, u32), // (major, minor)
    pub name: &'static str,
}

impl GpuCaps {
    pub const V100: Self = Self { compute_capability: (7, 0), name: "Tesla V100" };
    pub const A100: Self = Self { compute_capability: (8, 0), name: "A100 SXM" };
    pub const RTX4060: Self = Self { compute_capability: (8, 9), name: "RTX 4060" };
    pub const RTX4090: Self = Self { compute_capability: (8, 9), name: "RTX 4090" };
    pub const H100: Self = Self { compute_capability: (9, 0), name: "H100 SXM" };

    /// Check if a precision level is supported on this GPU.
    pub fn supports(&self, p: PrecisionLevel) -> bool {
        match p {
            PrecisionLevel::Fp32 => true, // always
            PrecisionLevel::Int8 => true, // sm_70+
            PrecisionLevel::Fp16 => self.compute_capability >= (7, 0), // sm_70+
            PrecisionLevel::Bf16 => self.compute_capability >= (8, 0), // sm_80+
            PrecisionLevel::Fp8 => self.compute_capability >= (8, 9),  // sm_89+ (Ada)
        }
    }

    /// Return all supported precision levels for this GPU.
    pub fn supported_precisions(&self) -> Vec<PrecisionLevel> {
        let all = [
            PrecisionLevel::Fp32,
            PrecisionLevel::Fp16,
            PrecisionLevel::Bf16,
            PrecisionLevel::Fp8,
            PrecisionLevel::Int8,
        ];
        all.iter().copied().filter(|p| self.supports(*p)).collect()
    }

    /// Auto-detect GPU from compute capability (0,0 = unknown/CPU).
    pub fn detect() -> Self {
        // In real implementation: query CUDA driver for device info.
        // For now, default to RTX 4060 (Ada Lovelace).
        Self::RTX4060
    }
}

/// Configuration for the mixed-precision builder.
#[derive(Clone)]
pub struct GraphConfig {
    pub k: usize,
    pub n: usize,
    pub dim: usize,
    pub seed_ratio: f64,
    pub outlier_ratio: f64,
    pub gpu_caps: Option<GpuCaps>,
}

/// The kNN graph built with mixed precision.
pub struct ApgcGraph {
    pub edges: Vec<Vec<Edge>>,
    pub precision_map: Vec<PrecisionLevel>,
    pub n: usize,
    pub k: usize,
    pub config: GraphConfig,
}

/// Built kNN graph edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub distance: f32,
    pub precision: PrecisionLevel,
}

impl ApgcGraph {
    pub fn new(config: GraphConfig) -> Self {
        let n = config.n;
        let k = config.k;
        Self {
            edges: vec![Vec::new(); n],
            precision_map: vec![PrecisionLevel::Fp16; n],
            n,
            k,
            config,
        }
    }

    /// Assign precision levels using 5-tier hierarchy.
    /// Adapts to GPU capabilities — unsupported formats fall back to nearest supported.
    pub fn assign_precision(&mut self) {
        let gpu = self.config.gpu_caps.unwrap_or(GpuCaps::RTX4060);
        let seed_count = (self.config.n as f64 * self.config.seed_ratio) as usize;
        let fp8_start = (self.config.n as f64 * 0.90) as usize;
        let outlier_start = self.config.n - (self.config.n as f64 * self.config.outlier_ratio) as usize;

        for i in 0..self.config.n {
            self.precision_map[i] = if i < seed_count {
                PrecisionLevel::Fp32 // seeds always FP32
            } else if i < fp8_start {
                // Majority: prefer BF16 if supported, else FP16
                if gpu.supports(PrecisionLevel::Bf16) { PrecisionLevel::Bf16 }
                else { PrecisionLevel::Fp16 }
            } else if i < outlier_start {
                // Near-outliers: prefer FP8 if supported, else FP16
                if gpu.supports(PrecisionLevel::Fp8) { PrecisionLevel::Fp8 }
                else if gpu.supports(PrecisionLevel::Fp16) { PrecisionLevel::Fp16 }
                else { PrecisionLevel::Fp32 }
            } else {
                // Bottom outliers: INT8
                PrecisionLevel::Int8
            };
        }
    }
}

/// Builds a kNN graph using mixed-precision distance computation.
pub struct MixedPrecisionBuilder;

impl MixedPrecisionBuilder {
    /// Compute distance between two vectors at a given precision.
    pub fn distance(a: &[f32], b: &[f32], precision: PrecisionLevel) -> f32 {
        match precision {
            PrecisionLevel::Fp32 => Self::l2_fp32(a, b),
            PrecisionLevel::Fp16 => Self::l2_fp16(a, b),
            PrecisionLevel::Bf16 => Self::l2_bf16(a, b),
            PrecisionLevel::Fp8 => Self::l2_fp8(a, b),
            PrecisionLevel::Int8 => Self::l2_int8(a, b),
        }
    }

    fn l2_fp32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }

    fn l2_fp16(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| {
            let dx = Self::to_fp16_approx(*x) - Self::to_fp16_approx(*y);
            dx * dx
        }).sum::<f32>().sqrt()
    }

    fn l2_bf16(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| {
            let dx = Self::to_bf16_approx(*x) - Self::to_bf16_approx(*y);
            dx * dx
        }).sum::<f32>().sqrt()
    }

    fn l2_fp8(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| {
            let dx = Self::to_fp8_approx(*x) - Self::to_fp8_approx(*y);
            dx * dx
        }).sum::<f32>().sqrt()
    }

    fn l2_int8(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| {
            let xi = (x * 127.0).round() as i8;
            let yi = (y * 127.0).round() as i8;
            let d = xi as f32 - yi as f32;
            d * d
        }).sum::<f32>().sqrt()
    }

    fn to_fp16_approx(x: f32) -> f32 {
        let bits = x.to_bits();
        let sign = bits & 0x8000_0000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = bits & 0x007F_FFFF;
        let truncated = mantissa & 0x007FE000;
        f32::from_bits(sign | (((exp + 127) as u32) << 23) | truncated)
    }

    fn to_bf16_approx(x: f32) -> f32 {
        let bits = x.to_bits();
        let sign = bits & 0x8000_0000;
        let exp = bits & 0x7F80_0000;
        let mantissa = bits & 0x007F_FFFF;
        let truncated = mantissa & 0x007F_E000;
        f32::from_bits(sign | exp | truncated)
    }

    fn to_fp8_approx(x: f32) -> f32 {
        let bits = x.to_bits();
        let sign = bits & 0x8000_0000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = bits & 0x007F_FFFF;
        let truncated = mantissa & 0x0060_0000;
        f32::from_bits(sign | (((exp + 127) as u32) << 23) | truncated)
    }

    /// Build kNN graph with GPU-aware precision assignment.
    pub fn build(vectors: &[Vec<f32>], config: GraphConfig) -> ApgcGraph {
        let n = config.n;
        let k = config.k;
        let seed_count = (n as f64 * config.seed_ratio) as usize;
        let gpu = config.gpu_caps.unwrap_or(GpuCaps::RTX4060);

        let mut graph = ApgcGraph::new(config.clone());
        graph.assign_precision();

        // Phase 1: Seeds on FP32
        for i in 0..seed_count.min(n) {
            let mut candidates: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (Self::distance(&vectors[i], &vectors[j], PrecisionLevel::Fp32), j))
                .collect();
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            graph.edges[i] = candidates.into_iter().take(k)
                .map(|(dist, j)| Edge { from: i, to: j, distance: dist, precision: PrecisionLevel::Fp32 })
                .collect();
        }

        // Phase 2: Remaining nodes at their assigned precision
        for i in seed_count..n {
            let prec = graph.precision_map[i];
            let mut candidates: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (Self::distance(&vectors[i], &vectors[j], prec), j))
                .collect();
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            graph.edges[i] = candidates.into_iter().take(k)
                .map(|(dist, j)| Edge { from: i, to: j, distance: dist, precision: prec })
                .collect();
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_preserves_ordering() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let d_fp32 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Fp32);
        let s_fp32 = MixedPrecisionBuilder::distance(&a, &c, PrecisionLevel::Fp32);
        assert!(d_fp32 > s_fp32);
        assert!((s_fp32).abs() < 1e-6);
    }

    #[test]
    fn test_all_precisions() {
        let a = vec![1.0, 0.5, 0.0];
        let b = vec![0.0, 0.5, 1.0];
        let d_fp32 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Fp32);
        let d_bf16 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Bf16);
        let d_fp16 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Fp16);
        let d_fp8 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Fp8);
        let d_int8 = MixedPrecisionBuilder::distance(&a, &b, PrecisionLevel::Int8);
        // All should be positive
        assert!(d_fp32 > 0.0);
        assert!(d_bf16 > 0.0);
        assert!(d_fp16 > 0.0);
        assert!(d_fp8 > 0.0);
        assert!(d_int8 > 0.0);
        // FP32 should be closest to true distance
        let true_dist = ((1.0_f32) + (0.0) + (1.0_f32 * 1.0)).sqrt();
        assert!((d_fp32 - true_dist).abs() < 0.01);
    }

    #[test]
    fn test_gpu_caps_v100() {
        let caps = GpuCaps::V100;
        assert!(caps.supports(PrecisionLevel::Fp32));
        assert!(caps.supports(PrecisionLevel::Fp16));
        assert!(caps.supports(PrecisionLevel::Int8));
        assert!(!caps.supports(PrecisionLevel::Bf16));
        assert!(!caps.supports(PrecisionLevel::Fp8));
        assert_eq!(caps.supported_precisions().len(), 3);
    }

    #[test]
    fn test_gpu_caps_ada() {
        let caps = GpuCaps::RTX4060;
        assert!(caps.supports(PrecisionLevel::Fp32));
        assert!(caps.supports(PrecisionLevel::Fp16));
        assert!(caps.supports(PrecisionLevel::Bf16));
        assert!(caps.supports(PrecisionLevel::Fp8));
        assert!(caps.supports(PrecisionLevel::Int8));
        assert_eq!(caps.supported_precisions().len(), 5);
    }

    #[test]
    fn test_graph_build() {
        let gpu = GpuCaps::RTX4060;
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0],
        ];
        let config = GraphConfig { k: 2, n: 4, dim: 3, seed_ratio: 0.5, outlier_ratio: 0.0, gpu_caps: Some(gpu) };
        let graph = MixedPrecisionBuilder::build(&vectors, config);
        assert_eq!(graph.edges.len(), 4);
        for edges in &graph.edges {
            assert!(edges.len() <= 2);
        }
    }
}
