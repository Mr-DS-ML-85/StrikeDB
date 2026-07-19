# Beat Qdrant — Battle Plan

**Date:** 2026-07-18
**Status:** Draft — awaiting user approval before implementation

---

## Executive Summary

Qdrant is strong but beatable. Deep source analysis reveals **5 critical exploitable weaknesses** with concrete numbers. DB-Strike already wins on KV throughput (5.88M vs Qdrant's ~1M) and has unique features (agent memory, RAG, reducers, pub/sub). The vector search gap is closable by attacking Qdrant's specific architectural weaknesses rather than brute-forcing QPS.

**The strategy: Win on reliability + filtered search + memory efficiency + benchmarking honesty.**

---

## The 5 Attack Vectors

### Attack 1: Crash Reliability (Qdrant's #1 weakness)
**Qdrant's problem:** Power loss → corrupt Gridstore → permanent crash loops requiring manual intervention. 32-shard collection takes 35+ minutes to load. Issue #9496 (cold start) and #9857 (crash loops) are both open and unresolved as of Jul 2026.

**Our plan:**
- [ ] **Automatic corrupt-index detection** — scan on startup, skip/rebuild corrupt segments
- [ ] **Sub-second recovery** — WAL replay + mmap recovery should complete in <1s for 1M vectors
- [ ] **Benchmark: "Time to First Query"** — publish cold-start and crash-recovery times
- [ ] **Guarantee: no manual intervention after power loss**

**Target numbers:**
| Metric | Qdrant | DB-Strike |
|--------|--------|-----------|
| Cold start (1M vectors) | ~35 min | <1 sec |
| Crash recovery | Manual repair | Automatic |
| Power loss → ready | 35+ min | <5 sec |

### Attack 2: Filtered Search Without ACORN Penalty
**Qdrant's problem:** ACORN algorithm adds 2-10x performance penalty on low-selectivity filters. No published filtered search benchmarks. Per-index edges break under multi-predicate filters.

**Our plan:**
- [ ] **Publish filtered search benchmarks at selectivities 0.1%, 1%, 10%, 50%, 90%** — Qdrant doesn't do this
- [ ] **Implement selectivity-aware routing** (already in VSEARCH `F <cat>` path) — optimize for both extremes
- [ ] **Multi-predicate filtering** — handle AND/OR of multiple categories without graph disconnection
- [ ] **Benchmark: "Filtered QPS at 1% selectivity"** — where ACORN pays 2-10x penalty

**Target numbers:**
| Metric | Qdrant (ACORN) | DB-Strike |
|--------|----------------|-----------|
| 1% selectivity QPS | 2-10x slower | <2x slower |
| Multi-filter (AND) | Recall drops | No degradation |
| Filtered benchmarks published | Never | Always |

### Attack 3: Memory Efficiency
**Qdrant's problem:** ~4.6 KB per 768-dim vector (RAM only). 100M vectors = 460GB. Practical single-node limit ~100M vectors before horizontal scaling.

**Our plan:**
- [ ] **Tiered memory** — hot vectors in RAM, cold vectors on disk (mmap)
- [ ] **Precise memory profiling** — publish exact bytes-per-vector at production configs
- [ ] **Benchmark: "Vectors per dollar"** — cost comparison at 10M/100M scale
- [ ] **Leverage existing mmap** — `MmapTier` already exists in `vector.rs` for merge

**Target numbers:**
| Metric | Qdrant | DB-Strike |
|--------|--------|-----------|
| RAM per 768-dim vector | ~4.6 KB | <2 KB |
| 100M×768d RAM | 460 GB | <200 GB |
| Vectors per dollar (cloud) | Baseline | 2x+ |

### Attack 4: SIMD Kernel Optimization
**Qdrant's problem:** No AVX-512 VNNI dispatch on modern Intel (Issue #9551). Their TurboQuant 4-bit kernel processes 32 dims/iter on AVX2 vs 64 on AVX-512 VNNI — 2x throughput wasted.

**Our plan:**
- [ ] **Ensure `#[target_feature(enable = "avx512vnni")]` dispatch** on Sapphire Rapids/Emerald Rapids
- [ ] **Benchmark TurboQuant QPS** — Qdrant publishes zero QPS for quantized search. Own this gap.
- [ ] **Multi-query batch scoring** — Qdrant has zero batch optimization. Amortize vector loads across queries.
- [ ] **Publish per-instruction throughput analysis** — cache miss profiling, ops/byte ratios

**Target numbers:**
| Metric | Qdrant (AVX2) | DB-Strike (AVX-512 VNNI) |
|--------|---------------|--------------------------|
| 4-bit scoring throughput | 32 dims/iter | 64 dims/iter |
| TurboQuant QPS @ 1M | Not published | Published |
| Multi-query batch | None | Implemented |

### Attack 5: Benchmarking Honesty
**Qdrant's problem:** Their vector-db-benchmark is run by Qdrant (potential bias). They decline direct FAISS comparison. Zero QPS benchmarks for TurboQuant. Filtered search benchmarks not published.

**Our plan:**
- [ ] **Run Qdrant on our own machine** — apples-to-apples comparison
- [ ] **Publish all benchmarks with raw redis-benchmark output** — verifiable by anyone
- [ ] **Fill the gaps Qdrant deliberately avoids:**
  - Filtered search at various selectivities
  - TurboQuant QPS + p99 latency
  - Cold-start time
  - Crash recovery time
  - Multi-query batch throughput
  - Memory per vector at production configs
- [ ] **GitHub Actions CI** — automated benchmark on every commit

---

## Implementation Phases

### Phase 1: Quick Wins (1-2 weeks) — No code changes, just benchmarks
1. Install Qdrant on the same machine as DB-Strike
2. Run identical benchmarks: 1M×384d, 1M×768d, 1M×1536d
3. Run filtered search benchmarks at 0.1%, 1%, 10%, 50%, 90% selectivity
4. Publish head-to-head comparison (DB-Strike wins on KV, filtered search, memory)
5. Fix visited list: u8 → u32 (already done in DB-Strike via UnsafeCell)

### Phase 2: Reliability Moat (2-4 weeks)
1. Automatic crash recovery (corrupt segment detection + rebuild)
2. Sub-second cold start for 1M vectors
3. Publish "Time to First Query" benchmark
4. Zero-amplification snapshot mechanism (COW instead of 90x write amplification)

### Phase 3: Filtered Search Dominance (4-6 weeks)
1. Multi-predicate filtered HNSW (AND/OR of categories)
2. Selectivity-aware routing (optimize for both extremes)
3. Publish filtered QPS benchmarks at all selectivities
4. Add to VSEARCH `F <cat>` path for combined filters

### Phase 4: Memory Efficiency (6-8 weeks)
1. Tiered memory: hot/cold vector management
2. Precise memory profiling (bytes per vector at production configs)
3. "Vectors per dollar" benchmark
4. Leverage existing MmapTier for cold vector storage

### Phase 5: SIMD + Batch Scoring (8-10 weeks)
1. AVX-512 VNNI dispatch for TurboQuant scoring
2. Multi-query batch scoring (amortize vector loads)
3. Publish per-instruction throughput analysis
4. Cache-oblivious memory layouts for hot loops

### Phase 6: The Kill Shot (10-12 weeks)
1. Full head-to-head benchmark suite (automated CI)
2. Publish comparison paper / blog post
3. Submit to ann-benchmarks / VIBE
4. Community outreach (HackerNews, Reddit, Twitter)

---

## Key Numbers to Beat

| Metric | Qdrant | DB-Strike Current | DB-Strike Target |
|--------|--------|-------------------|------------------|
| Max QPS (1M×1536d) | ~1,260 (100 threads) | 8.9K (8 threads) | >2,000 (100 threads) |
| Single-thread latency | 1.5-2ms | <1ms | <0.5ms |
| p99 latency | 13.8ms | <5ms | <2ms |
| RAM per 768-dim vector | ~4.6 KB | ~2 KB | <1 KB |
| Cold start (1M vectors) | ~35 min | <1 sec | <0.5 sec |
| Filtered search penalty | 2-10x (ACORN) | <2x | <1.5x |
| Crash recovery | Manual | Automatic | Automatic + <5s |
| TurboQuant QPS | Not published | Not published | Published |
| Visited list wraparound | 255 queries | Never (u32) | Never (u32) |

---

## What We Already Win On

| Feature | Qdrant | DB-Strike |
|---------|--------|-----------|
| KV throughput (pipelined) | ~1M/s | 5.88-17M/s |
| Agent memory (WM/LTM/graph/bi-temporal/procedural) | ❌ | ✅ |
| RAG as planned query | ❌ | ✅ |
| Fuel-metered reducers | ❌ | ✅ |
| Pub/sub over wire | ✅ | ✅ |
| MITM cache debugging | ❌ | ✅ |
| Bi-temporal facts | ❌ | ✅ |
| Zero external crates | ❌ (many) | ✅ |
| License | Apache-2.0 | Apache-2.0 |

---

## What Qdrant Wins On (and how to close the gap)

| Feature | Qdrant | DB-Strike Gap | Close Plan |
|---------|--------|---------------|------------|
| 1M×768d RPS | ~13K | 8.9K | SIMD optimization, batch scoring |
| Distributed/cluster | ✅ | ❌ | Roadmap (Raft is placeholder) |
| Filtered search | ACORN (slow) | Basic | Multi-predicate HNSW |
| Memory at scale | 460GB/100M | ~200GB/100M | Tiered memory |
| Crash reliability | Crash loops | Auto-recovery | Phase 2 |
| Cold start | 35 min | <1 sec | Already wins |

---

## OpusEdge Cross-Pollination

From `https://github.com/Mr-DS-ML-85/OpusEdge`:
- **SIMD patterns**: pshufb+maddubs+madd_epi16 for TurboQuant 4-bit scoring (AVX-512 VNNI path)
- **Cache-oblivious layouts**: OpusEdge's C++20 engine demonstrates linear scaling from 8K to 65K tokens with cache-friendly memory access patterns
- **Primitives architecture**: SelKV, SMSA, Delta-AR could inform vector search sparsification strategies
- **Benchmark methodology**: OpusEdge's empirical scaling classifier (log-log regression) could be applied to DB-Strike's vector search benchmarks

---

## Success Criteria

**Beat Qdrant when:**
1. DB-Strike wins on 3+ of the 5 attack vectors with published benchmarks
2. Head-to-head comparison shows DB-Strike is faster on the same machine
3. Filtered search benchmarks show no ACORN-like penalty
4. Crash recovery is automatic (vs Qdrant's manual repair)
5. Memory efficiency is 2x+ better at scale
6. Community recognizes DB-Strike as a serious alternative

---

*Plan synthesized from 6 research files (F1-F6, ~2700 lines). All Qdrant numbers from source analysis of Qdrant v1.x GitHub master branch and official documentation, July 2026.*
