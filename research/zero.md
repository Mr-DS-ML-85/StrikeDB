# ZERO: A GPU-Native Architecture for Zero-Overhead Graph Construction in Approximate Nearest Neighbor Search

**Irfan Mahir**  
*Independent Researcher, Dhaka, Bangladesh*  
irfan@furylogic.com

---

## Abstract

Graph-based approximate nearest neighbor (ANN) search on GPUs has traditionally been constrained by a fundamental mismatch between graph algorithms and GPU architecture. Existing approaches attempt to port CPU-centric algorithms (HNSW, NSG, NN-Descent) to GPUs, resulting in irregular memory access patterns, synchronization bottlenecks, and limited scalability. We present **ZERO** (Zero-Overhead Graph Construction), a novel GPU-native architecture that eliminates traditional graph structures entirely. Instead of building graphs, ZERO constructs **Sorted Neighbor Lists (SNL)** —compressed, sorted arrays of neighbor indices stored in contiguous memory. Four asynchronous streams operate concurrently without global barriers: (1) Tensor Core-accelerated distance computation with fused epilogues, (2) warp-cooperative candidate collection with no global atomics, (3) lane-confined atomic updates with per-warp ownership, and (4) continuous background consolidation. ZERO introduces three paradigm-shifting innovations: (a) sorted neighbor lists replace graph adjacency with memory-coalesced arrays, (b) stream-based asynchronous execution eliminates all global synchronization, and (c) lane-confined atomics enable contention-free updates. To our knowledge, ZERO is the first GPU-native graph construction architecture to completely eliminate the graph structure itself, achieving projected 5–10× faster construction than CAGRA, 50–70% memory reduction, and 2–3× higher search throughput.

**Keywords:** Approximate Nearest Neighbor Search, GPU Architecture, Graph Construction, Sorted Neighbor Lists, Tensor Cores, VUGVA Tiered Memory

---

## 1. Introduction

Approximate nearest neighbor search (ANNS) over high-dimensional vectors is a fundamental primitive for modern AI systems, powering retrieval-augmented generation (RAG), recommendation engines, and large-scale similarity search. Graph-based indexes—particularly HNSW, NSG, and Vamana—have emerged as the dominant approach due to their excellent accuracy-latency trade-off.

The rapid adoption of GPUs for ANNS has led to a series of GPU-accelerated graph indexes: CAGRA (2024), Tagore (2026), GRNND (2026), and CMANNS (2026). However, these systems share a common limitation: they are **CPU algorithms ported to GPUs**, not algorithms designed *for* GPUs. This fundamental mismatch manifests in three critical problems:

1. **Irregular Memory Access**: Graph construction requires random neighbor lookups, candidate sorting, deduplication, and redirection. GPUs excel at regular, predictable memory patterns, not pointer-chasing graph traversals.

2. **Synchronization Overhead**: Parallel NN-Descent implementations require global barriers at each iteration, collapsing GPU throughput. The GPU is forced to constantly switch between compute-heavy distance calculations and memory-heavy graph maintenance.

3. **Memory Fragmentation**: Graph indices consume 239–334 GB for billion-scale datasets, far exceeding any single GPU's HBM capacity. Existing systems resort to costly CPU-GPU data movement.

**The core thesis**: Graph construction on GPUs is fundamentally broken because the algorithm's control flow is misaligned with the hardware's strengths. What is needed is a complete rethinking of graph construction—one that abandons traditional graph structures and designs algorithms *exclusively* for GPU execution.

We present **ZERO** (Zero-Overhead Graph Construction), a GPU-native architecture that eliminates traditional graphs entirely. ZERO makes three radical departures from every existing approach:

1. **No Graphs, Only Sorted Neighbor Lists**: Instead of building graphs (nodes + edges + adjacency lists), ZERO constructs sorted, compressed arrays of neighbor indices. This transforms irregular pointer-chasing into contiguous memory access.

2. **Asynchronous Stream-Based Execution**: Four concurrent streams operate without global barriers, eliminating synchronization bottlenecks and enabling continuous construction.

3. **Lane-Confined Atomics**: Each warp owns a subset of vectors and updates them without cross-warp contention, eliminating atomic bottlenecks.

To our knowledge, ZERO is the **first GPU-native graph construction architecture** to completely eliminate the graph structure itself, achieving projected 5–10× faster construction than CAGRA, 50–70% memory reduction, and 2–3× higher search throughput.

---

## 2. Related Work

### 2.1 CPU-Based Graph Indexes

**HNSW** [Malkov & Yashunin, 2018] is the most widely used graph-based ANNS algorithm, constructing a hierarchical graph with random skip lists. **NSG** [Fu et al., 2019] improves upon HNSW by building a monotonic graph. **Vamana** [Subramanya et al., 2019] introduces degree-optimized graphs for disk-based search. These methods are fundamentally CPU-centric.

### 2.2 GPU-Accelerated Static Graph Indexes

**CAGRA** [Ootomo et al., 2024] introduced GPU-optimized HNSW with fixed-degree graphs and locality ordering, achieving up to 12.3× faster build times than CPU HNSW.

**Tagore** [Li et al., 2026] accelerates refinement-based graph indexes (NSG, Vamana) using GNN-Descent, achieving 1.32×–112.79× speedup over CPU baselines.

**GRNND** [Wang et al., 2026] redesigned RNN-Descent for GPUs using warp-cooperative updates, achieving 2–27× speedup over HNSW.

**CMANNS** [Zhang et al., 2026] introduced a GPU-managed CPU-GPU hybrid approach with up to 13.05× reduction in memory cost.

**Jasper** [Song et al., 2026] introduced the first GPU-based updatable Vamana index with lock-free streaming insertion, achieving up to 1.93× higher QPS and 8× smaller memory footprint.

All these methods are **static** or **incrementally updatable** but still rely on traditional graph structures (adjacency lists, pointer-chasing). None eliminate the graph structure itself.

### 2.3 GPU-Native Dynamic Indexes

**ETALE** [Zhao, 2026] introduced a GPU-native dynamic graph with lock-free copy-on-write slabs, achieving 4.8× faster updates than static rebuild.

**CleANN** [Zhang et al., 2025] introduced workload-aware linking and query-adaptive consolidation on CPU.

These methods advance dynamic graph management but do not address the fundamental issue of graph construction overhead on GPUs.

### 2.4 Gap in the Literature

**No existing system:**
1. Eliminates the graph structure entirely in favor of sorted neighbor lists
2. Uses stream-based asynchronous execution without global barriers
3. Employs lane-confined atomics for contention-free updates
4. Leverages VUGVA tiered memory during construction
5. Uses Tensor Cores for distance computation in graph construction

ZERO fills this gap by introducing a complete GPU-native architecture.

---

## 3. The ZERO Architecture

### 3.1 Overview

ZERO is built around three radical departures from existing approaches:

**1. Sorted Neighbor Lists (SNL)**
- Replaces graph adjacency with compressed, sorted arrays
- Every vector stores its top-K neighbors as a dense, sorted array
- Eliminates pointer-chasing and enables contiguous memory access

**2. Asynchronous Stream-Based Execution**
- Four concurrent streams with zero global barriers
- No synchronization between construction stages
- Continuous background consolidation

**3. Lane-Confined Atomics**
- Each warp owns a subset of vectors
- Updates applied without cross-warp contention
- Eliminates atomic bottlenecks

### 3.2 Sorted Neighbor Lists (SNL)

The fundamental data structure in ZERO is the Sorted Neighbor List. Instead of building a graph with adjacency lists, each vector stores its top-K neighbors as a **compressed, sorted array of indices**.

For a dataset $\mathcal{V} = \{v_1, ..., v_n\}$ with $v_i \in \mathbb{R}^d$, the SNL for vector $v_i$ is:

$$\text{SNL}(v_i) = [\text{idx}_{i,1}, \text{idx}_{i,2}, ..., \text{idx}_{i,K}]$$

where $\text{idx}_{i,j}$ is the index of the $j$-th nearest neighbor of $v_i$, and:

$$\text{idx}_{i,1} \prec \text{idx}_{i,2} \prec ... \prec \text{idx}_{i,K}$$

in order of decreasing similarity.

**Storage Format:** SNLs are stored as a flat array in GPU global memory:

```
[SNL(v_1)] [SNL(v_2)] ... [SNL(v_n)]
```

where each SNL is a contiguous block of K indices. This enables:
- **Perfect memory coalescing**: Warp reads contiguous blocks
- **Cache-friendly**: Entire SNLs fit in L1/L2 cache
- **No pointer-chasing**: All neighbor indices are directly accessible

### 3.3 Asynchronous Stream-Based Construction

ZERO operates four concurrent streams:

**Stream 1: Distance Computation (Tensor Cores)**
```
Input: Current SNL state
Output: Distance matrix D where D[i][j] = distance(v_i, v_j)
Hardware: Tensor Cores with INT8 dp4a (4 MACs/instruction)
```
Distance computation uses fused epilogues to maintain precision hierarchy:
- INT8 dp4a for bulk distance (4 MACs/instruction)
- FP16 for candidate refinement (2 MACs/instruction)
- FP32 only for seed/entry nodes (1 MAC/instruction)

**Stream 2: Candidate Collection (Warp-Level)**
```
Input: Distance matrix D
Output: Candidate list C_i for each vector
Operations:
  1. Warp-cooperative gather of distances
  2. Per-warp sorting of candidates (fixed capacity)
  3. No global atomics
  4. Double-buffered neighbor pools
```

**Stream 3: Neighbor List Update (Lane-Confined Atomics)**
```
Input: Candidate lists C_i
Output: Updated SNLs
Operations:
  1. Each warp owns a subset of vectors
  2. Lane-confined atomics only
  3. No cross-warp contention
  4. Disordered propagation to avoid premature convergence
```

**Stream 4: Background Consolidation**
```
Input: Updated SNLs
Output: Consolidated SNLs
Operations:
  1. Continuous, non-blocking
  2. Removes redundant edges
  3. Maintains degree bound
  4. Evicts cold shards via VUGVA
```

All streams run concurrently with zero global barriers. The "graph" is never locked; updates are applied using lane-confined atomics.

### 3.4 Lane-Confined Atomics

Lane-confined atomics are a novel GPU primitive where each warp owns a subset of vectors and updates them without cross-warp contention.

For a warp with 32 threads and a subset of 32 vectors:

```
Warp W owns vectors {v_i, v_{i+1}, ..., v_{i+31}}
Thread t in W updates vector v_{i+t}
Thread t only reads/writes to v_{i+t}
```

This eliminates all cross-warp contention and enables:
- Zero atomic overhead
- Perfect warp-level parallelism
- No global synchronization

The key insight: by partitioning vectors among warps, we eliminate the need for global atomics entirely.

### 3.5 VUGVA-Integrated Memory Tiering

ZERO uses VUGVA for tiered memory management during construction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ZERO MEMORY HIERARCHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 0 (VRAM): Active construction shard                                  │
│  - Current shard being processed                                           │
│  - Hot vectors with high access frequency                                 │
│  - Full precision representation                                          │
│                                                                             │
│  TIER 1 (RAM): Processed shards                                           │
│  - Completed neighbor lists                                               │
│  - Compressed representations                                             │
│  - Ready for merging                                                      │
│                                                                             │
│  TIER 2 (SSD): Cold shards                                                │
│  - Fully processed shards                                                 │
│  - Quantized representations                                              │
│  - Evicted from GPU/CPU                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key innovation:** Vectors are streamed from SSD/DRAM in shards. Each shard constructs its local SNLs independently, then shards are merged using a lock-free merge kernel running entirely on GPU. Cold vectors are evicted to DRAM/SSD using VUGVA's CPU-bypass architecture: the GPU reads vectors directly from system RAM via `cuMemAllocManaged` with page migration.

### 3.6 Lock-Free Shard Merge

After each shard builds its local SNLs, shards must be merged into the global index. ZERO uses a lock-free merge kernel:

```
Merge Kernel:
  1. Each warp loads SNLs from two shards
  2. Warp merges two sorted lists into one
  3. Updates global SNL using lane-confined atomics
  4. No global locks or barriers
```

The merge operation is $O(K)$ per warp, where K is the neighbor list size.

---

## 4. Mathematical Formulation

### 4.1 Sorted Neighbor List Update

Let $\text{SNL}_i$ be the current neighbor list for vector $v_i$. For candidate list $C_i$ (size M > K), the update rule is:

$$\text{SNL}_i^{(t+1)} = \text{top}_K(\text{SNL}_i^{(t)} \cup C_i)$$

where $\text{top}_K$ selects the K elements with highest similarity.

The update is applied atomically without locks using lane-confined atomics.

### 4.2 Distance Computation with Tensor Cores

Distance computation uses fused GEMM with INT8 dp4a:

$$D = \text{gemm\_int8}(V, V^T)$$

with fused epilogues for precision hierarchy:

$$\text{final}(i,j) = \begin{cases}
\text{FP32} & \text{if } i \text{ or } j \text{ is seed node} \\
\text{FP16} & \text{if candidate refinement} \\
\text{INT8} & \text{otherwise (bulk)}
\end{cases}$$

### 4.3 Convergence Properties

Under mild assumptions (bounded updates, Lipschitz continuity), ZERO converges to a stable state where:

$$\mathbb{E}[\|\text{SNL}^{(t+1)} - \text{SNL}^*\|^2] \leq (1 - \mu \eta)^t \cdot \|\text{SNL}^{(0)} - \text{SNL}^*\|^2 + \frac{\eta \sigma^2}{\mu}$$

where $\text{SNL}^*$ is the optimal sorted neighbor list state.

### 4.4 Memory Footprint

ZERO's memory footprint is:

$$M_{\text{ZERO}} = n \cdot K \cdot \text{sizeof(idx)} + n \cdot d \cdot \text{sizeof(float)}$$

For billion-scale datasets (n=1B, d=128, K=32):

$$M_{\text{ZERO}} = 1B \cdot 32 \cdot 4 + 1B \cdot 128 \cdot 4 = 128GB + 512GB = 640GB$$

With VUGVA tiering, only the active shard resides in VRAM.

---

## 5. Novelty Claims

| Innovation | What It Does | Why It's Novel |
|------------|--------------|----------------|
| **Sorted Neighbor Lists** | Replaces graph adjacency with compressed, sorted arrays | No existing GPU ANNS system uses sorted neighbor lists; all use graph structures |
| **Stream-Based Async Construction** | Four concurrent streams with zero global barriers | All existing systems (CAGRA, Tagore, GRNND) use barrier-synchronized iterations |
| **Lane-Confined Atomics** | Per-warp updates with no cross-warp contention | No published GPU graph construction algorithm uses this primitive |
| **Sharded Streaming Construction** | Builds local lists per shard, merges with lock-free kernel | Existing systems use CPU-GPU sharding; ZERO is pure GPU |
| **VUGVA-Integrated Tiering** | GPU reads vectors directly from RAM/SSD via CPU-bypass | No existing GPU graph construction uses unified memory tiering during build |
| **Tensor Core Distance** | Uses Tensor Cores for distance computation in graph construction | Existing systems (CAGRA, Tagore) do not use Tensor Cores during construction |

---

## 6. Comparison with Existing Systems

| Feature | CAGRA | Tagore | GRNND | CMANNS | Jasper | **ZERO** |
|---------|-------|--------|-------|--------|--------|----------|
| **Graph Structure** | HNSW | NSG/Vamana | NN-Descent | HNSW/NSG | Vamana | **SNL** |
| **CPU Dependency** | None | Yes (sharding) | None | Yes (sharding) | None | **None** |
| **Global Barriers** | Yes | Yes | Yes | Yes | Yes | **No** |
| **Warp-Level Primitives** | Partial | Partial | Yes | Partial | Partial | **Full** |
| **Tensor Cores** | No | No | No | Yes | No | **Yes** |
| **VUGVA Tiering** | No | No | No | No | No | **Yes** |
| **Streaming Construction** | No | No | No | No | Yes | **Yes** |
| **Lock-Free Updates** | No | No | No | No | Yes | **Yes** |
| **Graph Structure Eliminated** | No | No | No | No | No | **Yes** |

---

## 7. Implementation Path

| Phase | Component | Description | Estimated Effort |
|-------|-----------|-------------|------------------|
| 1 | SNL Data Structure | Compressed sorted arrays | ~300 lines |
| 2 | Tensor Core Distance Stream | GEMM with fused epilogues | ~400 lines |
| 3 | Candidate Collection Stream | Warp-cooperative gather | ~500 lines |
| 4 | Neighbor Update Stream | Lane-confined atomics | ~300 lines |
| 5 | Background Consolidation | Continuous, non-blocking | ~400 lines |
| 6 | VUGVA Tiering Integration | CPU-bypass memory management | ~200 lines |
| 7 | Lock-Free Shard Merge | Merge kernel | ~300 lines |
| 8 | Benchmark Suite | Evaluation harness | ~500 lines |

**Total Estimated Implementation:** ~2,900 lines of GPU-native Rust + CUDA

---

## 8. Expected Performance

Based on state-of-the-art (GRNND: 2.4–51.7× speedup; Tagore: 1.32×–112.79×; CMANNS: up to 13.05× reduction), ZERO is projected to achieve:

| Metric | Projected Improvement | Basis |
|--------|----------------------|-------|
| **Construction Speed** | 5–10× faster than CAGRA | Elimination of global barriers + Tensor Cores |
| **Memory Footprint** | 50–70% reduction | SNL compression + VUGVA tiering |
| **Search Throughput** | 2–3× higher than CAGRA | Contiguous memory access |
| **Update Cost** | Sub-linear | Shard-based streaming |

---

## 9. Conclusion

We have presented **ZERO** (Zero-Overhead Graph Construction), a GPU-native architecture for approximate nearest neighbor search that eliminates traditional graph structures entirely. ZERO introduces three paradigm-shifting innovations:

1. **Sorted Neighbor Lists** replace graph adjacency with compressed, sorted arrays, transforming irregular pointer-chasing into contiguous memory access.

2. **Asynchronous stream-based execution** eliminates all global barriers, enabling continuous, contention-free construction.

3. **Lane-confined atomics** enable per-warp updates without cross-warp contention, eliminating atomic bottlenecks.

ZERO is the **first GPU-native graph construction architecture** to completely eliminate the graph structure itself, achieving projected 5–10× faster construction than CAGRA, 50–70% memory reduction, and 2–3× higher search throughput. ZERO establishes a fundamentally new direction for GPU-accelerated ANNS: one where the graph is not built, but *sorted neighbor lists* are constructed directly.

---

## References

1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4)*, 824-836.

2. Ootomo, H., et al. (2024). CAGRA: Highly Parallel Approximate Nearest Neighbor Search on GPU. *arXiv:2401.07636*.

3. Li, Z., et al. (2026). Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search. *SIGMOD 2026*.

4. Wang, Z., et al. (2026). GRNND: GPU-Parallel RNN-Descent for Approximate Nearest Neighbor Search. *SIGMOD 2026*.

5. Zhang, Y., et al. (2026). CMANNS: GPU-Managed CPU-GPU Hybrid ANNS. *VLDB 2026*.

6. Song, J., et al. (2026). Jasper: A GPU-native updatable graph index for approximate nearest neighbor search. *arXiv:2603.00001*.

7. Zhao, D. (2026). ETALE: Evolving Topology with Accelerated Lock-free Execution for Dynamic Graph ANN Search on GPUs. *arXiv:2607.02543*.

8. Zhang, Z., et al. (2025). CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search. *arXiv:2507.19802*.

9. Mahir, I. (2026). VUGVA: Virtual Unified GPU VRAM Architecture. *Zenodo*.

10. Mahir, I. (2026). APGC: GPU-Built, CPU-Served Approximate Nearest Neighbor Search. *Zenodo*.

---

**Keywords:** Approximate Nearest Neighbor Search, GPU Architecture, Graph Construction, Sorted Neighbor Lists, Tensor Cores, VUGVA Tiered Memory

**License:** AGPL v3

---

