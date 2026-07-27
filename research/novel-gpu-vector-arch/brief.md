# Research Brief: Novel GPU Vector Search Architecture

**Date**: 2026-07-21
**Depth**: deep (5-8 sub-agents, 2 follow-up rounds, 25+ sources)

## Question
Design a novel GPU-accelerated vector search architecture that fills gaps between existing approaches (CAGRA, Jasper, ETALE, GGNN, GRNND, CMANNS, etc.) to achieve 0.97+ recall@10 on 1M×384d INT8 vectors with build time under 20s.

## Context
- Working system: DB-Strike — Rust data engine with GPU-accelerated HNSW/CAGRA
- Current GPU build achieves 0.000 recall because graph is built on INT8 vectors (poor distance discrimination)
- CPU HNSW achieves 0.992 recall at 170s build time
- GPU: RTX 4060 (8GB VRAM, 24 SMs)
- Zero external dependencies constraint

## Scope
**In**: GPU graph construction algorithms, GPU-native HNSW/Vamana/CAGRA, quantization-aware graph building, Rust GPU programming, novel hybrid CPU+GPU architectures
**Out**: Multi-GPU scaling, disk-based indexing, search-only optimizations (we need BUILD speed)

## Key Gaps to Investigate
1. CAGRA uses NN-Descent which doesn't converge well on INT8 — is there a better GPU-native initialization?
2. Jasper uses Vamana (not NN-Descent) — what's Vamana's GPU construction algorithm?
3. All papers evaluate on FP32 — nobody characterizes INT8 graph construction quality
4. ETALE's lock-free slab graph — can it be combined with NN-Descent for faster convergence?
5. GGNN does GPU traversal of CPU-built graphs — can we do GPU traversal of GPU-built graphs?
6. No paper combines: (a) GPU-native graph construction, (b) INT8 quantization-aware, (c) dynamic updates
7. Rust-CUDA exists but nobody has benchmarked it for vector search kernels

## Assumptions
- Single GPU (RTX 4060), not multi-GPU
- INT8 quantized vectors (user's data format)
- Cosine similarity metric
- Must work within 8GB VRAM
- Zero external dependencies
