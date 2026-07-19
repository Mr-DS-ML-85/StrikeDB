# TurboQuant & Qdrant: Deep Research Report

**Date:** 2026-07-18 | **Mode:** standard | **Sources:** 45+

---

## Executive Summary

TurboQuant is a **data-oblivious vector quantization** technique from Google Research (arXiv:2504.19874, April 2025). It achieves near-optimal compression by combining a Hadamard rotation, Lloyd-Max scalar quantization, and QJL residual correction — with **zero training time** and formal theoretical guarantees. Qdrant v1.18 (May 2026) implements a **hybrid TurboQuant+RaBitQ** variant with production extensions that boost recall by +14-18pp on anisotropic data over vanilla TurboQuant. No Qdrant competitor (Milvus, pgvector, Weaviate) currently offers TurboQuant-equivalent quantization.

---

## 1. What Is TurboQuant

**The paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" by Zandieh et al. (Google Research / DeepMind), submitted April 2025 [1].

**The algorithm (two stages):**

1. **Stage 1 (MSE):** L2-normalize the vector → apply random rotation Θ via Walsh-Hadamard Transform (O(D log D)) → each coordinate follows a known Beta distribution ≈ N(0, 1/d) → Lloyd-Max quantize each coordinate to nearest centroid (b-1 bits per coord)
2. **Stage 2 (PROD):** Compute residual r = x - dequant(quant(x)) → apply QJL sketch: sign(S·r) where S is a random projection matrix → store (b-1)·D code bits + D QJL sign bits + 1 residual norm scalar

**Per-vector storage:** ~b·D + 32 bits (code bits + QJL bits + norm)

**Theoretical guarantee:** MSE ≤ 3π² / 4^b — within **2.7× of the information-theoretic lower bound**. At b=1, only 1.45× from optimal [1].

**Why "data-oblivious":** After random rotation, coordinates become approximately N(0, 1/d) regardless of input distribution. A SINGLE universal Lloyd-Max codebook, derived analytically once, works for ALL datasets. No k-means training needed.

---

## 2. How Qdrant Implements TurboQuant

Qdrant's v1.18 implementation is **NOT vanilla TurboQuant** — it's a hybrid with three production extensions [2]:

### 2a. MSE variant, not PROD
The paper's PROD variant spends 1 bit on QJL correction. Qdrant rejects this because:
- PROD is asymmetric-only (can't serve both HNSW build and user queries)
- PROD requires an extra O(D log D) QJL projection per query
- PROD splits the bit budget, giving fewer bits to the codebook

Instead, Qdrant uses the MSE variant + RaBitQ-style length renormalization (4 bytes per vector).

### 2b. RaBitQ length renormalization
Qdrant stores `l2/cn` (original L2 norm / centroid-norm ratio) per vector to correct Lloyd-Max shrinkage. At b=1, MSE loses ~36% of every dot product — renormalization recovers this.

### 2c. Per-coordinate anisotropy compensation
Real embeddings are anisotropic (not uniformly distributed on the unit sphere). Qdrant does a per-segment pre-pass to estimate `(shift, scale)` per coordinate using P-Square quantile estimation, then applies compensation on the query side at scoring time — **free at search time**.

**Recall improvement from extensions (4-bit, 100K brute-force):**

| Dataset | Vanilla TQ | Qdrant impl | Delta |
|---------|-----------|-------------|-------|
| arxiv-instr (aniso) | 0.815 | 0.954 | **+13.9pp** |
| gte-mul-ads (aniso) | 0.791 | 0.967 | **+17.6pp** |
| openai-3-l (iso) | 0.941 | 0.970 | +2.9pp |

---

## 3. Qdrant's HNSW Integration

**Critical detail:** Qdrant does NOT "navigate with cheap distance and rerank with Turbo score." Instead:

- **HNSW build:** Uses **symmetric scoring** (quantized-vs-quantized) — both stored vectors and build-time queries use TurboQuant. Graph topology is optimized for "encoding-compensated space."
- **User queries:** Uses **asymmetric scoring** (float32 query vs quantized stored vectors). Anisotropy compensation is applied ONLY to asymmetric scoring, not to HNSW build.
- **Anisotropy compensation deliberately NOT applied to symmetric scoring** — it would amplify quantization noise from both sides.

HNSW parameters: `m=16`, `ef_construct=128`.

---

## 4. Competitive Landscape

### Qdrant vs Milvus
- Milvus uses FAISS/Knowhere (PQ, SQ8, IVF) — requires per-dataset training
- Qdrant's TurboQuant is training-free with universal codebooks
- No TurboQuant equivalent in Milvus

### Qdrant vs pgvector
- pgvector only supports halfvec, binary quantization, HNSW/IVFFlat
- No scalar, product, or rotation-based quantization
- pgvector caps dimensions at 2000/4000; Qdrant has no cap

### Qdrant vs Weaviate
- Weaviate supports PQ, BQ, SQ — no TurboQuant equivalent
- Qdrant's benchmarks show Weaviate improved least among competitors

### Quantization comparison (Qdrant's 10-dataset benchmark, HNSW m=16, ef_c=128):

| Mode | Compression | Best recall range |
|------|------------|-------------------|
| SQ (INT8) | 4x | 0.88-0.98 |
| **TQ 4-bit** | **8x** | **0.82-0.97** (competitive with SQ at half storage) |
| TQ 2-bit | 16x | 0.82-0.92 (beats BQ 2-bit by 9-24pp) |
| BQ 2-bit | 16x | 0.60-0.78 |
| TQ 1-bit | 32x | 0.63-0.85 (beats BQ 1-bit by 9-21pp) |
| BQ 1-bit | 32x | 0.48-0.70 |
| PQ | up to 64x | requires training |

---

## 5. Key Technical Details for DB-Strike

### 5a. Hadamard rotation — NOT padding to power-of-2
Qdrant uses **greedy decomposition** into sum-of-powers-of-two blocks (e.g., D=300 → [256, 32, 8, 4]). Each block goes through its own WHT. Then 3 rounds of (permute + WHT) using a reversible LCG + Fisher-Yates permutation. The rotation is deterministic from a seed, not persisted to disk.

**Bug they caught:** Must use upper 32 bits of LCG state for modulo, not lower bits (lower bits have short periods).

### 5b. RAM per vector (1536-dim)
| Encoding | Storage |
|----------|---------|
| bits4 (8x) | 772 bytes |
| bits2 (16x) | 388 bytes |
| bits1_5 (24x) | 292 bytes |
| bits1 (32x) | 196 bytes |
| f32 (1x) | 6,144 bytes |

### 5c. SIMD scoring kernels
The 4-bit kernel: `pshufb` (codebook lookup) + `maddubs` + `madd_epi16` (multiply-accumulate) — fits into a handful of integer SIMD instructions per 16-dimension chunk. Works on AVX-VNNI, AVX-512, ARM SDOT.

### 5d. No QPS/latency benchmarks published
Qdrant focuses entirely on recall comparisons. No concrete QPS or p99 latency numbers for TurboQuant at 1M scale.

---

## 6. What DB-Strike Already Has vs What Qdrant Does

| Feature | DB-Strike | Qdrant |
|---------|-----------|--------|
| INT8 scalar quant | ✅ AVX2 int8 dot | ✅ SQ |
| TurboQuant | ✅ Hadamard + Lloyd-Max + QJL | ✅ TQ + RaBitQ + anisotropy |
| Binary quant | ✅ 1/2-bit | ✅ 1/1.5/2-bit |
| Product quant | ✅ k-means PQ | ✅ PQ |
| f32 rerank | ✅ exact f32 on ef candidates | ✅ rescoring |
| NVMe cold tier | ✅ mmap f32 tier | ✅ disk-based |
| Anisotropy compensation | ❌ **MISSING** | ✅ per-segment P-Square |
| RaBitQ renormalization | ❌ **MISSING** | ✅ length renorm |
| HNSW m=16, ef_c=128 | Uses m=32, ef_c=200 | m=16, ef_c=128 |

---

## 7. Opportunities for DB-Strike to Beat Qdrant

### 7a. Add anisotropy compensation (highest impact)
Qdrant's biggest recall wins come from per-coordinate `(shift, scale)` compensation on anisotropic data (+14-18pp on real embeddings). DB-Strike currently assumes isotropic data. Adding P-Square quantile calibration per segment would close this gap.

### 7b. Add RaBitQ length renormalization
Store `l2/cn` (4 bytes per vector) to correct Lloyd-Max shrinkage. This is simple and recovers ~3-4pp on isotropic data, much more on anisotropic.

### 7c. Tune HNSW parameters
Qdrant uses m=16, ef_c=128. DB-Strike uses m=32, ef_c=200 — twice the graph density, which helps recall but hurts memory and insert speed. At TurboQuant's compression levels, the graph overhead dominates, so m=16 with anisotropy compensation may match recall at lower memory.

### 7d. SIMD scoring for TurboQuant
DB-Strike's TurboQuant scoring uses f32 dot products (Hadamard-rotated query vs dequantized centroids). Qdrant uses integer SIMD kernels (`pshufb + maddubs + madd_epi16`) that are ~2-4× faster per operation. Implementing these would close the throughput gap.

### 7e. Greedy WHT decomposition (no padding)
DB-Strike pads to power-of-2 for Hadamard. Qdrant decomposes into sum-of-powers-of-two blocks, saving memory and avoiding waste on non-power-of-2 dimensions.

---

## Sources

[1] Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv:2504.19874 (April 2025)
[2] Ivan Pleshkov, "TurboQuant: The Story Behind Qdrant's New Quantization", https://ivanpleshkov.dev/blog/turboquant/ (May 2026)
[3] Qdrant TurboQuant announcement, https://qdrant.tech/articles/turboquant-quantization/ (May 2026)
[4] Google Research blog, "TurboQuant: Redefining AI Efficiency", https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ (March 2026)
[5] Qdrant quantization documentation, https://qdrant.tech/documentation/manage-data/quantization/
[6] Qdrant vs Elastic benchmark, https://qdrant.tech/blog/benchmark-elastic-diskbbq/ (July 2026)
[7] Qdrant benchmarks, https://qdrant.tech/benchmarks/
[8] RaBitQ paper, Gao & Long, SIGMOD 2024, arXiv:2405.12497
[9] QJL paper, arXiv:2406.03482, AAAI 2025
[10] PolarQuant paper, arXiv:2502.02617, AISTATS 2026
[11] ScaNN paper, Guo et al., arXiv:1908.10396 (2019)

*Accessed: 2026-07-18*
