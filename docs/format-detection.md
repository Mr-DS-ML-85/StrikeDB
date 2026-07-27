# GPU Format Detection

## Overview

DB-Strike auto-detects GPU capabilities and loads only compatible precision formats. This ensures optimal performance across GPU generations.

## GPU Capability Matrix

| GPU Architecture | Compute Capability | FP32 | FP16 | BF16 | FP8 | INT8 |
|-----------------|-------------------|------|------|------|-----|------|
| **Tesla V100** (Volta) | sm_70 | ✅ | ✅ Tensor | ❌ | ❌ | ✅ Tensor |
| **A100 SXM** (Ampere) | sm_80 | ✅ | ✅ Tensor | ✅ Tensor | ⚠️ Limited | ✅ Tensor |
| **H100 SXM** (Hopper) | sm_90 | ✅ | ✅ Tensor | ✅ Tensor | ✅ Tensor | ✅ Tensor |
| **RTX 4060** (Ada) | sm_89 | ✅ | ✅ Tensor | ✅ Tensor | ✅ Tensor | ✅ |
| **RTX 4090** (Ada) | sm_89 | ✅ | ✅ Tensor | ✅ Tensor | ✅ Tensor | ✅ |
| **RTX 3090** (Ampere) | sm_86 | ✅ | ✅ Tensor | ❌ | ❌ | ✅ |

## Detection API

```rust
use apgc::precision::GpuCaps;

let caps = GpuCaps::detect();  // Auto-detect from CUDA driver
println!("GPU: {}", caps.name);
println!("Supported: {:?}", caps.supported_precisions());

// Manual specification
let v100 = GpuCaps::V100;
assert!(!v100.supports(PrecisionLevel::Bf16));
assert!(!v100.supports(PrecisionLevel::Fp8));
assert!(v100.supports(PrecisionLevel::Fp16));  // Tensor Cores

let ada = GpuCaps::RTX4060;
assert!(ada.supports(PrecisionLevel::Fp8));   // Ada Lovelace
assert!(ada.supports(PrecisionLevel::Bf16));
```

## Behavior on Unsupported Formats

When a GPU doesn't support a format, APGC automatically falls back:

```rust
// On V100: Fp8 → falls back to Fp16 (nearest supported)
// On V100: Bf16 → falls back to Fp16
// On RTX 4060: all 5 formats supported, no fallback needed

let mut graph = ApgcGraph::new(config);
graph.assign_precision();  // Uses GPU caps for automatic fallback
```

## V100 Cluster (32GB × 24 = 768GB VRAM)

For the V100 cluster:
- **Available formats**: FP32, FP16, INT8
- **NOT available**: BF16, FP8
- **Recommended**: FP16 for majority, FP32 for seeds, INT8 for outliers
- **Kernel loading**: Only FP32/FP16/INT8 CUDA kernels are compiled
- **Memory budget**: 768GB total VRAM, ~600GB usable after overhead

## Ada Lovelace (RTX 4060/4090)

For Ada GPUs:
- **Available formats**: All 5 (FP32, FP16, BF16, FP8, INT8)
- **Recommended**: BF16 for majority (wider range than FP16), FP8 for near-outliers
- **Maximum optimization**: 5-tier hierarchy with full format coverage
