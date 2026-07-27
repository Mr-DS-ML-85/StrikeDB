/// APGC — Adaptive Precision Graph Construction
///
/// GPU-accelerated kNN graph construction with:
/// - Mixed-precision (FP32/FP16/BF16/FP8/INT8) based on node importance
/// - KV-aware search pruning using LLM attention patterns
/// - VUGVA-aware memory tiering
/// - GPU capability detection (V100 vs Ada Lovelace format support)

pub mod graph;
pub mod precision;
pub mod kv_prune;

pub use precision::{ApgcGraph, MixedPrecisionBuilder, PrecisionLevel};
pub use kv_prune::KvSearchPruner;
