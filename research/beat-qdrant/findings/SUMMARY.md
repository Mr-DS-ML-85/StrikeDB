# DB-Strike vs Qdrant: Consolidated Research Summary

**Date:** 2026-07-18
**Sources:** F1-F6 research findings (6 files, ~2,700 lines of analysis)
**Purpose:** Prioritized actionable insights for beating Qdrant

---

## Executive Summary

Qdrant is a strong Rust-based vector database with ~33.4K GitHub stars, achieving ~1,260 QPS on 1M x 1536d vectors at 0.97 recall (100 threads, 8 vCPU). However, deep source analysis reveals **30 specific exploitable weaknesses** across architecture, performance, reliability, and operational complexity. The most impactful opportunities lie in: (1) cold-start/recovery reliability, (2) filtered search performance without ACORN overhead, (3) memory efficiency at scale, (4) SIMD kernel optimizations, and (5) operational simplicity gaps.

---

## TOP 15 PRIORITIZED ACTIONABLE INSIGHTS

### TIER 1: High-Impact / Directly Exploitable

---

#### 1. [RELIABILITY] Crash Recovery = Permanent Crash Loops (CRITICAL)
**Impact:** Data loss / extended downtime
**Details:** After power loss, corrupt Gridstore payload indexes cause permanent crash loops requiring manual intervention. A 32-shard collection takes ~35 minutes to load; each corrupt shard discovered costs another full crash cycle.
**Concrete Number:** 35+ minutes startup for 32-shard collection; crash loop per corrupt shard.
**Opportunity:** Build automatic corrupt-index detection and repair. Guarantee sub-minute recovery. Position as "crash-proof" vs Qdrant's "crash-loops."
**Source:** F3 (Issue #9857, #9496)

#### 2. [PERFORMANCE] ACORN Filtered Search = 2-10x Performance Penalty
**Impact:** Filtered search throughput
**Details:** Qdrant's ACORN algorithm (used for low-selectivity filters) is 2-10x slower than standard HNSW. It explores 2-hop neighbors of filtered-out nodes. Activates when estimated_selectivity < 0.4 (default). No filtered search benchmarks are published — a gap we can own.
**Concrete Number:** 2-10x slowdown on ACORN path; selectivity threshold default = 0.4.
**Opportunity:** Implement a better filtered-search strategy that avoids the 2-10x ACORN penalty. Publish filtered search benchmarks at various selectivities (Qdrant doesn't).
**Source:** F4 (Documentation, ACORN algorithm)

#### 3. [SCALABILITY] 460 GB RAM for 100M x 768d Vectors
**Impact:** Cost at scale
**Details:** Qdrant stores ALL vectors in RAM by default. Memory formula: vector_dim x 4 bytes x 1.5 (overhead). At 100M vectors x 768 dim = ~460 GB RAM. Practical single-node limit is ~100M vectors before horizontal scaling required.
**Concrete Number:** ~4.6 KB per 768-dim vector; ~9.2 KB per 1536-dim vector (with HNSW index overhead).
**Opportunity:** Build tiered memory architecture (hot/cold vectors) or native disk-based HNSW with acceptable latency. Be the "10x more vectors per dollar" solution.
**Source:** F3 (Memory Limitations section)

#### 4. [PERFORMANCE] No AVX-512 Usage on Modern Intel Hardware
**Impact:** 2x SIMD throughput wasted
**Details:** Qdrant's simsimd dispatches to AVX2 instead of AVX-512 on Intel Xeon 6 hardware (Issue #9551). Their TurboQuant 4-bit kernel on AVX-512 VNNI processes 64 dims/iteration vs 32 on AVX2 — a 2x throughput gain they're leaving on the table.
**Concrete Number:** AVX-512 VNNI = 64 dims/iter, AVX2 = 32 dims/iter. 2x throughput difference. VPDPBUSD instruction = single-cycle fused u8xi8 dot product.
**Opportunity:** Ensure our SIMD dispatch correctly detects and uses AVX-512 VNNI on Sapphire Rapids/Emerald Rapids. Benchmark and publish the 2x advantage.
**Source:** F6 (SIMD dispatch, Issue #9551)

#### 5. [ARCHITECTURE] No Incremental HNSW Updates — Full Rebuild Required
**Impact:** Write throughput and operational cost
**Details:** HNSW indexes are rebuilt from scratch during segment optimization. No incremental insert/delete support. Deleted points are merely marked and filtered during search. Segment merging triggers graph rebuild because point IDs are renumbered. The GraphLayersHealer only fixes connectivity, not optimality.
**Concrete Number:** HNSW graph must be rebuilt when adding payload indexes (requires toggling m=0 then back). 256-point single-threaded warm-up phase.
**Opportunity:** Implement true incremental HNSW with WAL-based updates. This is a major architectural differentiator.
**Source:** F1 (Section 8.2, 8.3)

---

### TIER 2: Medium-High Impact / Differentiating

---

#### 6. [RELIABILITY] Visited List u8 Counter Overflow Under High Query Rates
**Impact:** Latency spikes at high QPS
**Details:** Qdrant uses a `u8` generation counter (0-255) for visited list tracking. After 255 consecutive queries, the counter wraps and forces a full array clear (`fill(0)`), causing latency spikes. Under sustained high query rates this is a measurable problem.
**Concrete Number:** 255-query wraparound; full O(N) array clear on wrap.
**Opportunity:** Use u64 generation counter or bitset. Eliminates the periodic O(N) reset entirely.
**Source:** F1 (Section 5, visited_pool.rs)

#### 7. [SCALABILITY] Cold-Start Time = 35+ Minutes (Actively Unresolved)
**Impact:** Serverless / auto-scaling viability
**Details:** Issue #9496 is a tracking issue for cold-start optimization as of Jun 2026, still incomplete. Current approach loads segments component-by-component with blocking reads. Two-pass optimization (prefetch + cache) is in progress but many sub-tasks remain open.
**Concrete Number:** 32-shard collection = ~35 minutes startup.
**Opportunity:** Build sub-second cold start. This is the #1 blocker for Qdrant's serverless ambitions and a massive competitive moat.
**Source:** F3 (Section 5, Issue #9496)

#### 8. [PERFORMANCE] Snapshot Creation = 90x Write Amplification
**Impact:** Write throughput during maintenance
**Details:** Issue #9858: Snapshot creation causes ~90x write amplification and stalls background snapshots. This means write throughput drops dramatically during any snapshot/backup operation.
**Concrete Number:** 90x write amplification factor during snapshots.
**Opportunity:** Implement copy-on-write snapshots or incremental snapshots with zero write amplification.
**Source:** F3 (Issue #9858)

#### 9. [ARCHITECTURE] Extra HNSW Edges Are Per-Index, Not Per-Combination
**Impact:** Multi-filter search degradation
**Details:** Qdrant's filterable HNSW adds extra edges per payload index separately, not for combinations of payload indices. When 2+ strict filters are applied simultaneously, graph components can disconnect. The query planner heuristics also vary between Qdrant versions, making performance unpredictable across upgrades.
**Concrete Number:** Each payload index adds extra edges independently. Combined strict filters still degrade recall.
**Opportunity:** Build a combination-aware payload graph or a filter-native graph construction that handles multi-predicate filtering without degradation.
**Source:** F4 (Section 1, 8)

#### 10. [SIMD] TurboQuant 4-bit: No QPS Benchmarks Published
**Impact:** Marketing/competitive positioning
**Details:** Qdrant publishes recall comparisons for TurboQuant but zero QPS benchmarks at 1M+ scale. No p99 latency measurements for quantized search. No per-instruction throughput analysis. No cache miss profiling.
**Concrete Number:** Recall-only benchmarks: TQ4 ~0.92 avg across datasets, similar to SQ at 2x compression. But actual QPS unknown.
**Opportunity:** Publish concrete QPS + p99 numbers for our quantized search. Fill the benchmark gap Qdrant deliberately avoids.
**Source:** F6 (Section 9)

---

### TIER 3: Medium Impact / Operational Advantages

---

#### 11. [OPERATIONS] No Auto-Scaling or Shard Rebalancing in OSS
**Impact:** Operational burden / cost
**Details:** Self-hosted Qdrant requires manual shard management. Must plan shard count upfront (no resharding). Recommendations contradict: "start with 12 shards" vs "set shard_number = node count for optimal throughput." Minimum 3 nodes for Raft consensus + RF>=2 for production.
**Concrete Number:** Minimum 3 nodes + RF=2 for production. 12-shard recommendation for flexibility.
**Opportunity:** Build automatic shard rebalancing, resharding, and auto-scaling as core OSS features (not gated behind paid tier).
**Source:** F3 (Section 3)

#### 12. [PERFORMANCE] Memory-Bound Hot Loop in Quantized Scoring
**Impact:** Throughput ceiling on modern hardware
**Details:** The 4-bit TQ kernel has arithmetic intensity of ~0.175 ops/byte — firmly memory-bound. Total memory traffic per 16 dims = 40 bytes (8 vector + 32 query). For 1536-dim vectors: SSE = ~672 instructions, AVX2 = ~336, AVX-512 = ~120.
**Concrete Number:** 0.175 ops/byte arithmetic intensity. 40 bytes per 16-dim chunk. L2 cache fits ~340-1380 TQ4 vectors (1536d).
**Opportunity:** Optimize memory access patterns: prefetch hints, SoA layout for better SIMD traversal, explore cache-oblivious layouts.
**Source:** F6 (Section 7)

#### 13. [RELIABILITY] Multiple Critical Open Bugs (Memory Safety + Data Corruption)
**Impact:** Production reliability
**Details:** Open critical bugs as of Jul 2026: use-after-free in EncodedVectorsU8 (Issue #9799), queue-proxifying shard can drop shard (Issue #9665), on-disk map index over-counts deletions (Issue #9802), gRPC timeout cancels receiver initialization (Issue #9745).
**Concrete Number:** 4 critical + 5 high-severity open bugs.
**Opportunity:** Position reliability story: "zero critical bugs" / formal verification / crash-safe by design.
**Source:** F3 (Section 7)

#### 14. [MARKET] Single-Company Backing vs Foundation Governance
**Impact:** Enterprise adoption risk perception
**Details:** Qdrant is a single Berlin-based company (Series A). Milvus is under LF AI & Data Foundation. Enterprise procurement teams prefer foundation-backed projects for long-term stability. Key operational features (monitoring, auto-scaling, backups, RBAC) are gated behind paid tiers.
**Concrete Number:** Qdrant free cloud tier = 0.5 vCPU, 1GB RAM, 4GB disk (essentially unusable).
**Opportunity:** If building open-source, get foundation backing. If building commercial, ensure all operational features are in OSS tier.
**Source:** F5 (Section 7, 9)

#### 15. [SCORING] No Multi-Query Batch Optimization
**Impact:** Multi-tenant / batch search throughput
**Details:** Each query creates its own RawScorer with precomputed query. Parallelism is at segment level, not query level. No optimization for scoring multiple different queries against the same vector set simultaneously.
**Concrete Number:** Zero multi-query batch optimization in current codebase.
**Opportunity:** Implement batch query scoring that amortizes vector loads across multiple queries. Especially impactful for multi-tenant search.
**Source:** F6 (Section 8)

---

## BENCHMARKS TO PUBLISH (Qdrant Doesn't)

| Metric | Qdrant Published | Gap We Can Fill |
|--------|-----------------|-----------------|
| QPS at 1M+ vectors with TurboQuant | Recall only | Full QPS + p99 latency |
| Filtered search at various selectivities | Not published | Selectivity sweep: 0.1%, 1%, 10%, 50%, 90% |
| Cold-start time | Not a benchmark | Time-to-first-query from zero |
| Crash recovery time | Not a benchmark | Recovery from power loss / corruption |
| Multi-query batch throughput | Not measured | Concurrent query optimization |
| Memory per vector at production configs | Rough estimates | Precise measurements with profiling |

---

## KEY NUMBERS TO REMEMBER

| Metric | Qdrant Value | DB-Strike Target |
|--------|-------------|-----------------|
| Max QPS (1M x 1536d, 100 threads) | ~1,260 | >2,000 |
| Single-thread latency (1536d, 0.95 recall) | 1.5-2ms | <1ms |
| p99 latency (100 threads, 0.95 recall) | 13.8ms | <5ms |
| RAM per 768-dim vector | ~4.6 KB | <1 KB |
| Cold-start (32-shard) | ~35 min | <30 sec |
| ACORN filtered search penalty | 2-10x | 1x (no penalty) |
| Snapshot write amplification | 90x | <2x |
| Visited list wraparound | 255 queries | Never (u64/bitset) |
| SIMD throughput (4-bit, AVX-512 vs AVX2) | 64 vs 32 dims/iter | Always AVX-512 VNNI |
| Crash recovery | Manual intervention | Automatic repair |

---

## IMPLEMENTATION PRIORITY MATRIX

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Fix visited list: u8 → u64 counter (F1)
- [ ] Ensure AVX-512 VNNI dispatch works correctly (F6)
- [ ] Publish filtered search benchmarks at various selectivities (F4)
- [ ] Publish TurboQuant QPS + p99 numbers (F6)

### Phase 2: Core Differentiators (1-2 months)
- [ ] Build automatic crash recovery / corrupt-index repair (F3)
- [ ] Implement sub-second cold start (F3)
- [ ] Design filter-native HNSW for multi-predicate search (F4)
- [ ] Add zero-amplification snapshot mechanism (F3)

### Phase 3: Architectural Advantages (3-6 months)
- [ ] True incremental HNSW updates (F1)
- [ ] Tiered memory: hot/cold vector management (F3)
- [ ] Multi-query batch scoring optimization (F6)
- [ ] Auto-scaling + shard rebalancing in OSS (F3)

---

*Consolidated from F1 (HNSW internals), F2 (benchmarks), F3 (weaknesses), F4 (filtering/hybrid), F5 (landscape), F6 (SIMD/scoring). All sources from Qdrant v1.x GitHub master branch and official documentation, July 2026.*
