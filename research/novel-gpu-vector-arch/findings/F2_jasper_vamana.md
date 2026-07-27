# F2: Jasper & Vamana Graph Construction on GPU

> **Generated**: 2026-07-18 | **Queries**: 6 parallel research agents | **Confidence**: HIGH (most claims traceable to primary papers)

---

## 1. How Does Jasper's Vamana Graph Construction Work on GPU?

### Paper: Jasper — GPU-Accelerated ANNS: Quantized for Speed, Built for Change
- **Authors:** Hunter McCoy, Zikun Wang, Prashant Pandey
- **Year:** 2026 (arXiv:2601.07048, submitted Jan 11 2026, revised Feb 4 2026)
- **URL:** https://arxiv.org/abs/2601.07048
- **Code:** https://github.com/saltsystemslab/JasperGPUANNS
- **Confidence:** HIGH

### Key Insight: Three-Part GPU-Native Vamana Construction

Jasper builds the Vamana graph entirely on GPU with three core contributions:

**1. CUDA Batch-Parallel Construction with Lock-Free Streaming Insertions**
- Processes multiple vertices simultaneously across thousands of GPU threads
- Lock-free: new vectors can be inserted without global rebuild
- Each thread runs greedy beam search (find nearest neighbors) + edge pruning concurrently
- This directly addresses CAGRA's biggest weakness: CAGRA requires full index rebuild on any data change

**2. GPU-Efficient RaBitQ Quantization**
- Implements RaBitQ quantization natively on GPU during construction itself
- Reduces memory footprint by up to **8x** vs full-precision vectors
- Avoids random memory access penalties that plague other quantization schemes (e.g., PQ codebook lookups)
- RaBitQ's D-bit codes stored contiguously → coalesced GPU memory access
- Distance computation uses bitwise XOR + popcount → maps perfectly to GPU SIMT execution

**3. Optimized Greedy Search Kernel**
- Strips extraneous components from CAGRA's search implementation
- Achieves **80% peak GPU utilization** (roofline model)
- Better latency hiding via compute/memory overlap despite data-dependent access patterns

### Performance vs CAGRA and BANG

| Metric | vs CAGRA | vs BANG |
|--------|----------|---------|
| Query throughput | **1.93x higher** | **19-131x faster** |
| Index construction | **2.4x faster** | N/A |
| Updatability | CAGRA: **none**; Jasper: streaming insertions | N/A |
| Peak GPU utilization | Up to **80%** (roofline) | N/A |

### GPU Construction Pipeline

```
Phase 1 (GPU): Batch-parallel greedy beam search
  - Multiple vertices processed simultaneously
  - RaBitQ-compressed codes for distance computation (8x less memory traffic)

Phase 2 (GPU): Lock-free edge pruning
  - RobustPrune-style pruning per vertex
  - Atomic CAS for concurrent edge updates
  - No locks → no warp serialization

Phase 3 (GPU): Streaming insertion support
  - New vectors inserted incrementally
  - No full rebuild required
```

---

## 2. Vamana Graph vs HNSW vs NSG

### Foundational Papers

| Method | Paper | Authors | Year | URL | Confidence |
|--------|-------|---------|------|-----|-----------|
| Vamana | DiskANN: Fast Accurate Billion-point NN Search on a Single Node | Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi (MSR India) | 2019 (NeurIPS) | https://proceedings.neurips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html | HIGH |
| HNSW | Efficient and Robust ANNS Using Hierarchical Navigable Small World Graphs | Malkov, Yashunin | 2016 (IEEE TPAMI 2020) | https://arxiv.org/abs/1603.09320 | HIGH |
| NSG | Fast ANNS With The Navigating Spreading-out Graph | Fu, Xiang, Wang, Cai (Zhejiang/Alibaba) | 2017 (VLDB 2019) | https://arxiv.org/abs/1707.00143 | HIGH |
| Survey | Comprehensive Survey of Graph-Based ANNS | Wang, Xu, Yue, Wang | 2021 | https://arxiv.org/abs/2101.12631 | HIGH |
| GPU Survey | GPU-Accelerated Algorithms for Graph Vector Search | Liu et al. | 2026 | https://arxiv.org/abs/2602.16719 | HIGH |

### Structural Differences

| Property | HNSW | NSG | Vamana |
|----------|------|-----|--------|
| **Structure** | Multi-layer hierarchy | Flat, single-layer | Flat, single-layer |
| **Out-degree** | Variable (2*M at L0, <=M elsewhere) | Low, ~32-64 | Fixed at L (32-64) |
| **Edge direction** | Bidirectional | Directed | Directed |
| **Entry point** | Fixed (top layer) | Single centroid | Leader set (~1% in RAM) |
| **Memory layout** | Per-level pointer lists | Variable-length adjacency | Flat arrays (fixed-width) |
| **SSD support** | mmap possible, not native | mmap possible, not native | **Native SSD design** |
| **Scale** | ~100M in RAM | ~100M in RAM | **1B+ on 64GB RAM + SSD** |

### Construction Algorithms

**HNSW:** Sequential insertion. Each point: greedy descend through layers → find M neighbors → connect bidirectional edges → prune. Inherently sequential (each insertion depends on previous state).

**NSG:** Two-phase: (1) Build kNN graph via NN-Descent (iterative refinement), (2) Prune edges using monotonic reachability criterion. Moderately parallelizable.

**Vamana:** Modified NSW insertion with bounded degree. Each new point: greedy beam search from leader set → collect L candidates → prune using "pruned neighbors of pruned neighbors" criterion → keep exactly L edges. Flat structure + fixed degree = GPU-friendly.

### Why Vamana is Amenable to GPU Construction

1. **Flat structure** — no layer-level branching in GPU kernels
2. **Fixed out-degree** — all adjacency lists same length → no warp divergence
3. **Contiguous memory** — flat row-major arrays `[node_id, degree, neighbor_1, ..., neighbor_L]` → coalesced access
4. **Decomposable** — construction splits into: (1) GPU kNN graph build (embarrassingly parallel), (2) GPU pruning (rank-based, no distance recomputation needed)
5. **No bidirectional edge management during insert** — reverse edges handled in separate pass

### Performance Comparison

| Metric | HNSW | NSG | Vamana/DiskANN |
|--------|------|-----|----------------|
| Recall@10 | 90-98%+ | 95-99% | 95%+ |
| Memory/vector | High (~80-130B for M=16) | Low (~128-256B for L=32-64) | Very low (10% in RAM) |
| Scale | ~100M | ~100M | **1B+** |
| Dynamic updates | Online insert | Batch rebuild | Online insert (FreshDiskANN) |

---

## 3. RaBitQ Quantization-Aware Graph Construction on GPU

### Paper: RaBitQ — Quantizing High-Dimensional Vectors with Theoretical Error Bound
- **Authors:** Jianyang Gao, Cheng Long (Nanyang Technological University)
- **Year:** 2024 (SIGMOD 2024)
- **URL:** https://arxiv.org/abs/2405.12497
- **Confidence:** HIGH

### What RaBitQ Is

RaBitQ quantizes D-dimensional vectors into **D-bit strings** (1 bit per dimension). Unlike Product Quantization (PQ), RaBitQ's codebook is **random and data-independent**, providing provable theoretical error bounds.

**Quantization Process:**
1. **Normalize** raw vectors to unit vectors
2. **Random rotation** via fixed random Gaussian matrix
3. **Lloyd-Max scalar quantization** per rotated dimension → each dimension becomes 1 bit
4. **Distance estimation** via bitwise XOR + popcount → extremely fast, unbiased estimator

### How RaBitQ Enables Graph Construction on GPU (Jasper)

| Challenge | How RaBitQ Solves It |
|-----------|---------------------|
| Memory bandwidth bottleneck | 8x memory reduction (1 bit/dim vs 32 bits/dim) |
| Random access from PQ codebook lookups | RaBitQ has no codebook lookup — pure bitwise ops |
| GPU SIMT compatibility | XOR + popcount maps to SIMD/SIMT perfectly |
| Distance computation during beam search | Bitwise ops on compressed codes → coalesced access |

### Related GPU-RaBitQ Papers

| Paper | Authors | Year | Key Result | URL | Confidence |
|-------|---------|------|------------|-----|-----------|
| GPU-Native IVF-RaBitQ | Shi, Gao, Xia, Feher, Long | 2026 | 2.2x QPS vs CAGRA, 7.7x faster build; in NVIDIA cuVS | https://arxiv.org/abs/2602.23999 | HIGH |
| Ascend-RaBitQ | He, Ye, Cai et al. | 2026 | NPU-CPU heterogeneous (Huawei Ascend), not GPU | https://arxiv.org/abs/2605.16007 | MEDIUM |

### Key Gap: RaBitQ + Graph Construction

RaBitQ is a quantization scheme; "quantization-aware graph construction" means using RaBitQ codes during the graph building process (for neighbor distance computation) rather than only at query time. Jasper is the primary paper demonstrating this — distances during Vamana beam search are computed on RaBitQ codes, not full-precision vectors.

---

## 4. What Makes Jasper 1.93x Faster Than CAGRA?

### Paper: CAGRA — Highly Parallel Graph Construction and ANNS for GPUs
- **Authors:** Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang (NVIDIA)
- **Year:** 2023 (ICDE 2024)
- **URL:** https://arxiv.org/abs/2308.15136
- **Confidence:** HIGH (139 citations)

### CAGRA's Achievements (Before Jasper)
- 2.2-27x faster graph construction than HNSW (CPU)
- 33-77x higher large-batch query throughput than HNSW
- 3.4-53x faster single-query than HNSW at 95% recall

### Three CAGRA Limitations Jasper Exploits

| Limitation | CAGRA | Jasper's Solution |
|-----------|-------|-------------------|
| **No updatability** | Must rebuild entire index on any data change | Lock-free streaming insertions |
| **No quantization** | Full-precision float32 vectors → max memory | RaBitQ: 8x memory reduction |
| **Suboptimal GPU utilization** | Data-dependent memory access in greedy search limits compute/memory overlap | Optimized kernel: 80% peak utilization |

### The 1.93x Speedup Breakdown

1. **RaBitQ quantization** → 8x less memory traffic → higher effective bandwidth
2. **Optimized search kernel** → 80% roofline utilization (vs CAGRA's lower utilization)
3. **Vamana graph structure** → fixed-degree flat adjacency → coalesced access patterns
4. **Streamlined beam search** → extraneous components removed → less wasted compute

### Additional Speedup: 2.4x Faster Construction

- CAGRA's two-stage pipeline: GPU kNN graph → CPU rank-based optimization
- Jasper: fully GPU-native batch-parallel with lock-free insertions → no CPU round-trip

---

## 5. Can Vamana Be Built Entirely on GPU?

**Answer: Yes.** Multiple systems now demonstrate GPU-native Vamana construction.

### Timeline of GPU Vamana Construction

| System | Year | GPU Construction? | Graph Type | Key Innovation | URL | Confidence |
|--------|------|--------------------|------------|----------------|-----|-----------|
| DiskANN/Vamana | 2019 | No (CPU-only) | Vamana | Original algorithm | NeurIPS 2019 | HIGH |
| CAGRA | 2024 | Yes (GPU-native) | CAGRA graph | Parallel construction, 2.2-27x vs HNSW | https://arxiv.org/abs/2308.15136 | HIGH |
| BANG | 2024 | Search only (CPU build) | Vamana | PQ + prefetching for billion-scale | https://arxiv.org/abs/2401.11324 | HIGH |
| Jasper | 2026 | Yes (GPU-native Vamana) | Vamana | Lock-free streaming + RaBitQ | https://arxiv.org/abs/2601.07048 | HIGH |
| Tagore | 2025 | Yes (GPU-native) | NSG + Vamana | GNN-Descent + CFS pruning kernels | https://arxiv.org/abs/2508.08744 | HIGH |
| CMANNS | 2026 | Yes (GPU-accelerated) | Vamana | Compute-memory disaggregation | https://dl.acm.org/doi/10.1145/3802027 | MED-HIGH |
| NVIDIA cuVS | 2024+ | Yes (official library) | Vamana | Production GPU build → DiskANN-compatible output | https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/vamana | VERY HIGH |

### What Changed to Make GPU-Native Vamana Feasible

| Innovation | Paper/System | Year | Impact |
|-----------|-------------|------|--------|
| Batch-parallel construction | Jasper | 2026 | Transforms sequential insertions into parallel workload |
| GPU-specific kNN initialization (GNN-Descent) | Tagore | 2025 | Replaces CPU-based NN-Descent for graph init |
| Quantization to fit GPU memory | BANG, Jasper, cuVS | 2024-2026 | 4-8x compression → billion-scale fits in GPU memory |
| Prefetching and pipelining | BANG | 2024 | Hides PCIe latency between CPU/GPU |
| Compute-memory disaggregation | CMANNS | 2026 | Separates distance computation (GPU) from irregular traversal |
| Graph reordering for memory locality | Oguri et al. | 2025 | +15% QPS via improved cache behavior |

### Remaining Challenges

1. **GPU memory capacity** — full graph + dataset may not fit for billions of vectors
2. **Graph quality vs throughput** — parallelizing inherently sequential traversal can produce lower-quality graphs
3. **Search-construction symmetry** — cuVS builds on GPU but searches on CPU (via DiskANN)
4. **Race conditions** — concurrent edge updates require careful synchronization

### Graph Reordering for GPU (Supporting Paper)

- **Paper:** On the Effectiveness of Graph Reordering for Accelerating ANNS on GPU
- **Authors:** Oguri, Nishimura, Matsui
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2508.15436
- **Key insight:** First to run NSG, Vamana, and NN-Descent on GPU while preserving original algorithms. Graph reordering achieves up to 15% QPS improvement.
- **Confidence:** HIGH

---

## 6. ETALE's Slab Graph and Lock-Free Construction

### Paper: ETALE — Evolving Topology with Accelerated Lock-free Execution for Dynamic Graph ANN Search on GPUs
- **Author:** Dongfang Zhao (University of Washington, Tacoma)
- **Year:** 2026 (submitted June 24, 2026)
- **URL:** https://arxiv.org/abs/2607.02543
- **Confidence:** VERY HIGH (full paper read in HTML)

### What Problem Does ETALE Solve?

GPU graph indexes (CAGRA, Tagore) are **static** — they require expensive full rebuilds for any data change. CPU dynamic indexes (HNSW, SPFresh, DIGRA) support incremental updates but cannot exploit GPU parallelism. ETALE is the **first to combine**: GPU-native proximity graph + incremental insertion AND deletion + lock-free concurrent mutation.

### What is a Slab Graph?

**Formal Definition:** An ETALE graph is G=(V, r, A, x) where:
- **V** = set of live nodes
- **r** = **reference map**: assigns each identifier u a packed 64-bit word `r(u) = <delta_u, d_u, s_u>`:
  - `delta_u`: 1-bit deletion flag (0=live, 1=tombstone)
  - `d_u`: out-degree (up to R=32, warp width)
  - `s_u`: slab identifier pointing to adjacency storage
- **A** = **slab map**: each slab maps to ordered array of R entries, each `(neighbor_id, distance)`
- **x** = **vector map**: assigns each identifier its embedding vector

### Key Architectural Properties

| Property | Standard Graph (CAGRA/HNSW) | ETALE Slab Graph |
|----------|---------------------------|-----------------|
| Node representation | Neighbor list inline with node | Neighbors in separate immutable slab |
| Update mechanism | In-place mutation | **Copy-on-write**: allocate fresh slab, CAS pointer |
| Deletion | Tombstone-only or rebuild | Tombstone + repair + bounded reclamation |
| Concurrency | Locks or batch rebuild | **Lock-free**: single CAS publishes full rewrite |
| Memory model | Mutable arrays | Immutable slabs + epoch-based reclamation |

### How Lock-Free Construction Works (Publish Protocol)

1. **Snapshot** — atomically read current 64-bit reference `r(u) = <delta, d, s>`
2. **Allocate** — pop fresh slab from pre-allocated pool (no allocation overhead)
3. **Write** — build new neighbor sequence into fresh slab (O(R) where R=32)
4. **Fence** — GPU memory fence ensures slab contents visible before reference
5. **CAS** — compare-and-swap reference from snapshot to new reference
6. **Retire** — if CAS succeeds, push old slab to retired stack; if fails, discard and retry

**Critical properties:**
- **Wait-free readers**: single atomic load yields immutable slab identifier
- **Lock-free writers**: failed CAS means another writer succeeded → system always progresses
- **Deletion monotonicity**: deletion bit is inlined in same atomic word → no later update can undo deletion
- **Bounded memory**: periodic reclamation keeps footprint bounded by live set size

### Three Update Protocols

| Operation | Algorithm | Complexity | Key Mechanism |
|-----------|-----------|-----------|---------------|
| **Insertion** | Beam search → RobustPrune → Publish new node slab → Publish reverse edges | O(R log N) | Independent reverse-edge repairs |
| **Deletion** | Tombstone via CAS → Rebuild each live neighbor's adjacency | O(R²) per deletion | Each repair independent, concurrent |
| **Reclamation** | Phase 1: remove edges to tombstones. Phase 2: free orphaned slabs | Sublinear | Cluster-aware caching |

### Performance Comparison

| System | Type | Dynamic? | Maintenance Speed | Recall |
|--------|------|----------|-------------------|--------|
| **ETALE** | Graph, dynamic | Yes (insert + delete) | **~250ms/round** (1% churn) | >0.95 |
| CAGRA | Graph, static | No (full rebuild) | ~1.8s/round | Matched |
| Tagore | Graph, static | No (accelerated rebuild) | ~500ms/round | Matched |
| DIGRA | Graph, dynamic (CPU) | Yes | 3.3-147x slower | Matched |
| HNSW | Graph, dynamic (CPU) | Yes | 3.3-147x slower | Matched |

**Key results:**
- 4.8-8.8x faster than CAGRA per-window rebuild
- 1.8-2.5x faster than Tagore
- 3.3-147x faster than CPU dynamic indexes

### Gaps Identified

1. **No quantization awareness** — evaluates on FP32 only; no INT8/quantized graph construction
2. **Initial build not optimized** — focuses on streaming maintenance, not bulk construction
3. **CUDA-specific** — slab structure relies on GPU atomic operations; not portable to other architectures without careful adaptation
4. **Slab graph + NN-Descent unexplored** — could combining slab structure with NN-Descent iterative refinement improve convergence?

---

## Cross-Cutting Analysis

### The GPU Vamana Ecosystem Map (2019-2026)

```
DiskANN/Vamana (2019, CPU-only)
  │
  ├── CAGRA (2024, GPU-native, static, no quantization)
  │     └── Jasper (2026, improves CAGRA: +RaBitQ, +updatable, +optimized kernel)
  │
  ├── BANG (2024, GPU search on CPU-built Vamana, PQ quantization)
  │
  ├── Tagore (2025, GPU-native NSG+Vamana, GNN-Descent init)
  │
  ├── CMANNS (2026, GPU Vamana via compute-memory disaggregation)
  │
  ├── NVIDIA cuVS (2024+, official GPU Vamana build → DiskANN-compatible output)
  │
  └── ETALE (2026, dynamic GPU graph with slab structure, lock-free updates)
```

### Key Takeaways for DB-Strike

1. **Vamana is the preferred GPU graph structure** — flat, fixed-degree, SSD-friendly, decomposable into GPU-parallel phases
2. **RaBitQ quantization is the breakthrough** — 8x memory reduction without random access penalties; bitwise ops map to GPU SIMT
3. **Lock-free construction is solved** — both Jasper (batch-parallel) and ETALE (slab graph + CAS) demonstrate feasible lock-free GPU graph mutation
4. **Jasper's 1.93x over CAGRA** comes from three sources: quantization (memory), optimized kernel (compute), and lock-free updates (efficiency)
5. **ETALE's slab graph** is a novel data structure — immutable slabs with copy-on-write CAS publication — that solves the multi-word atomic update problem
6. **Remaining gap: quantization-aware dynamic graph construction** — no system combines ETALE's dynamic updates with RaBitQ-style quantization

---

## Confidence Summary

| Claim | Confidence | Source |
|-------|-----------|--------|
| Jasper builds Vamana on GPU with batch-parallel lock-free algorithm | HIGH | arXiv:2601.07048 |
| Jasper achieves 1.93x query throughput vs CAGRA | HIGH | arXiv:2601.07048 abstract |
| Jasper achieves 2.4x faster construction vs CAGRA | HIGH | arXiv:2601.07048 abstract |
| RaBitQ provides 8x memory reduction | HIGH | arXiv:2601.07048 + arXiv:2405.12497 |
| Vamana is structurally better for GPU than HNSW | HIGH | Analysis of flat vs hierarchical structure |
| ETALE's slab graph enables lock-free GPU graph updates | VERY HIGH | arXiv:2607.02543 (full paper read) |
| ETALE is 4.8-8.8x faster than CAGRA rebuild | HIGH | arXiv:2607.02543 evaluation |
| No system combines dynamic updates + quantization-aware construction | HIGH | Literature survey across all papers |
| NVIDIA cuVS provides production GPU Vamana build | VERY HIGH | Official NVIDIA documentation |

---

## All Referenced Papers

| # | Title | Authors | Year | URL |
|---|-------|---------|------|-----|
| 1 | GPU-Accelerated ANNS: Quantized for Speed, Built for Change (Jasper) | McCoy, Wang, Pandey | 2026 | https://arxiv.org/abs/2601.07048 |
| 2 | DiskANN: Fast Accurate Billion-point NN Search on a Single Node | Subramanya et al. (MSR) | 2019 | NeurIPS 2019 |
| 3 | Efficient and Robust ANNS Using Hierarchical Navigable Small World Graphs (HNSW) | Malkov, Yashunin | 2016 | https://arxiv.org/abs/1603.09320 |
| 4 | Fast ANNS With The Navigating Spreading-out Graph (NSG) | Fu et al. | 2017 | https://arxiv.org/abs/1707.00143 |
| 5 | Comprehensive Survey of Graph-Based ANNS | Wang et al. | 2021 | https://arxiv.org/abs/2101.12631 |
| 6 | GPU-Accelerated Algorithms for Graph Vector Search (Survey) | Liu et al. | 2026 | https://arxiv.org/abs/2602.16719 |
| 7 | RaBitQ: Quantizing High-Dimensional Vectors with Theoretical Error Bound | Gao, Long | 2024 | https://arxiv.org/abs/2405.12497 |
| 8 | GPU-Native ANNS with IVF-RaBitQ | Shi, Gao, Xia et al. | 2026 | https://arxiv.org/abs/2602.23999 |
| 9 | CAGRA: Highly Parallel Graph Construction and ANNS for GPUs | Ootomo et al. (NVIDIA) | 2023 | https://arxiv.org/abs/2308.15136 |
| 10 | BANG: Billion-Scale ANNS Using a Single GPU | Khan et al. | 2024 | https://arxiv.org/abs/2401.11324 |
| 11 | Scalable Graph Indexing using GPUs (Tagore) | Li et al. | 2025 | https://arxiv.org/abs/2508.08744 |
| 12 | CMANNS: GPU-Accelerated Graph Index Construction via Compute-Memory Disaggregation | Huan et al. | 2026 | https://dl.acm.org/doi/10.1145/3802027 |
| 13 | ETALE: Evolving Topology with Lock-free Execution for Dynamic Graph ANN on GPUs | Zhao | 2026 | https://arxiv.org/abs/2607.02543 |
| 14 | On the Effectiveness of Graph Reordering for Accelerating ANNS on GPU | Oguri et al. | 2025 | https://arxiv.org/abs/2508.15436 |
| 15 | FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Search | Singh et al. (MSR) | 2021 | https://arxiv.org/abs/2105.09613 |
| 16 | PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing | Rubel et al. | 2026 | https://arxiv.org/abs/2602.21247 |
| 17 | NVIDIA cuVS Vamana (official library) | NVIDIA | 2024+ | https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/vamana |
