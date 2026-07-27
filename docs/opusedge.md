# OpusEdge — Δ-Signal Driven Compute Allocation

## Overview

OpusEdge extracts a single per-token importance signal (Δ) from any LLM architecture and drives 30 composable inference primitives. Zero retraining required.

## Signal Sources

| Architecture | Signal | Cost | Source |
|-------------|--------|------|--------|
| Hybrid (Falcon-H1, Jamba) | Native Δ (SSM selectivity) | O(1) | Zero-cost byproduct |
| Dense (Qwen, LLaMA) | Proxy-Δ (RMS hidden-state drift) | O(L) | One norm per layer |
| MoE (Mixtral, OLMoE) | Router-IR (softmax entropy) | O(E) | Router logits |

## Primitives

### Core Primitives (10)

| Primitive | Function | Latency | Speedup |
|-----------|----------|---------|---------|
| **SelKV** | Δ-guided KV cache eviction | 76 µs | 87.5% cache reduction |
| **SMSA** | SSM-Masked Sparse Attention | ~0 µs | 3.56-4.98× at 2K tokens |
| **Delta-AR** | Δ-guided attention routing | 20 µs | O(S²)→O(S·K) |
| **ΔRank** | Selectivity-adaptive rank | ~0 ns | 4-tier rank mapping |
| **HeadDeactivate** | Adaptive multi-head gating | 10 µs | 87.5% heads off |
| **StateCompress** | Predictive state compression | ~0 µs | 37.5% state savings |
| **DenseEvic** | Cross-arch KV eviction | 57 µs | Protected boundary |
| **GAKV** | Gating-aware KV score | — | Multi-signal fusion |
| **Pareto Frontier** | Optimal operating point sweep | — | Control-plane |
| **Router-IR** | MoE routing confidence | — | Max-min or entropy |

### Stabilizers (4)

| Stabilizer | Prevents | Mechanism |
|-----------|----------|-----------|
| **MPSR** | Context loss from eviction | Project KV → SSM state |
| **EB-AR** | Compute shock from pruning | Scale by output entropy |
| **SSR/CASP+NDPA** | Perplexity explosions | Soft SVD + phase alignment |
| **IPSS/CSA** | Quality cliffs from head deactivation | Linear-time attention fallback |

### Task Controllers (2)

| Controller | Function |
|-----------|----------|
| **CAL** | Classify prompt rigidity, multiply thresholds |
| **R-CAL** | EMA-freezer for settled contexts |

## Precision Support

OpusEdge works with all precision formats:

| Format | Bits | Primitive Compatibility | GPU Tensor Cores |
|--------|------|------------------------|-------------------|
| FP32 | 32 | Full support | N/A |
| BF16 | 16 | Full support | sm_80+ |
| FP16 | 16 | Full support | sm_70+ |
| FP8 | 8 | Full support | sm_89+ |
| INT8 | 8 | Full support | sm_70+ |

## Usage

```rust
use opusedge::signal::DeltaSignal;
use opusedge::primitives::SelKV;

// Extract Δ from hidden states
let delta = DeltaSignal::from_proxy_delta(&hidden_states);

// Evict 87.5% of KV cache
let result = SelKV::evict(&delta, 0.875, seq_len);
// result.retained_indices — keep these tokens
// result.evicted_indices — drop these tokens
// result.memory_savings — 0.875 (87.5%)

// Route attention to top-K keys
let routing = DeltaAR::route(&delta, 64);
// routing.routed_keys — which keys to attend to
// routing.flop_reduction — O(S²)→O(S·64)
```
