/// OpusEdge — Telemetry-Guided Dynamic Compute Allocation
///
/// Pure Rust reimplementation of the C++20 OpusEdge engine.
/// One Δ signal drives 30 primitives for dense, MoE, and hybrid SSM-attention LLMs.
///
/// Core primitives:
/// - SelKV: Δ-guided KV cache eviction (87.5% cache reduction)
/// - SMSA: SSM-Masked Sparse Attention (4.98× speedup)
/// - Delta-AR: Δ-guided attention routing (O(S²)→O(S·K))
/// - HeadDeactivate: Adaptive multi-head gating
/// - StateCompress: Predictive hidden-state compression
/// - DenseEvic: Cross-architecture KV eviction
/// - ΔRank: Selectivity-adaptive low-rank projection

pub mod signal;
pub mod primitives;
pub mod stabilizers;

pub use signal::{DeltaSignal, SignalSource};
pub use primitives::{SelKV, Smsa, DeltaAR, HeadDeactivate, StateCompress, DenseEvic, DeltaRank};
pub use stabilizers::{Mpsr, Ebar, Ssr, Ipss};

/// Architecture type for signal extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchType {
    Dense,
    Hybrid,
    MoE,
}
