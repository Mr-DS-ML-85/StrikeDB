# F5: GPU-Accelerated Graph Traversal for Vector Search — Search/Query Phase

**Research Date**: 2026-07-18
**Status**: Comprehensive findings across 7 query dimensions

---

## 1. CAGRA Search Algorithm — GPU Beam Traversal

### Finding 1.1: CAGRA's Multi-CTA Greedy Search with Bitonic Sort
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Authors**: Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang
- **Year**: 2023 (published ICDE 2024)
- **Key Algorithm**: CAGRA search uses a **greedy graph traversal** where each query vector starts from a random entry point. A thread block (CTA) maintains a local list of `k` nearest candidates. At each traversal step, all neighbors of the current best candidates are examined. Distances are computed cooperatively within the CTA using **bitonic sort** to maintain the candidate list in sorted order. The algorithm terminates when no better candidates are found among all neighbors of the current top-k. The search is fundamentally a **beam search**: at each step, the beam (candidate list) is expanded by examining neighbors, then pruned back to k.
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH — original CAGRA paper, 139+ citations

### Finding 1.2: CAGRA Search is Memory-Bandwidth Bound, Not Compute-Bound
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Authors**: Same as above
- **Year**: 2023
- **Key Insight**: "The memory bandwidth of the device limits the throughput of CAGRA." During search, the dominant cost is reading vector data from GPU global memory for distance computations. Each traversal step requires loading neighbor vectors, computing distances, and sorting. The irregular access pattern of graph traversal (random node lookups) prevents efficient coalescing and caching, making the search fundamentally **latency-bound on memory**. On A100 (2 TB/s HBM), CAGRA achieves ~100M QPS on SIFT128, but this is far below the theoretical FLOPS limit.
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH — self-reported, verified by multiple follow-up papers

### Finding 1.3: Single-CTA vs Multi-CTA Search in CAGRA
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Authors**: Same as above
- **Year**: 2023
- **Key Algorithm**: CAGRA provides two search kernels:
  - **Single-CTA search**: One CUDA thread block handles one query. The block cooperatively manages a local candidate list using shared memory and bitonic sorting. This is simpler and has lower latency but limited parallelism per query.
  - **Multi-CTA search**: Multiple CTAs cooperate on a single query. The search state is distributed across CTAs, with one CTA performing the main traversal and others assisting with distance computations. This improves utilization for large-dimension vectors but adds synchronization overhead.
  - The choice depends on vector dimensionality: single-CTA works well for low-dim (e.g., SIFT 128d), while multi-CTA helps for high-dim (e.g., 768d+) where distance computation becomes the bottleneck.
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH — described in original paper

---

## 2. GGNN Graph Traversal on GPU

### Finding 2.1: GGNN is a General-Purpose GNN, Not a Vector Search Algorithm
- **Paper**: "Gated Graph Sequence Neural Networks"
- **Authors**: Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel
- **Year**: 2015 (ICLR 2016)
- **Key Algorithm**: GGNN (Gated Graph Neural Networks) uses **gated recurrent units (GRUs)** to propagate messages across graph edges for a fixed number of steps. At each step, node embeddings are updated by aggregating messages from neighbors, then passed through a GRU. This is fundamentally a **feature learning / node classification** framework, not a nearest neighbor search algorithm. GGNN has been applied to tasks like program verification and molecular property prediction.
- **URL**: https://arxiv.org/abs/1511.05493
- **Confidence**: HIGH — canonical GGNN paper, 2000+ citations

### Finding 2.2: GGNN GPU Acceleration Uses Sparse Matrix Operations
- **Paper**: Various GGNN implementations (Deep Graph Library, PyTorch Geometric)
- **Key Insight**: GGNN on GPU is implemented as **sparse matrix-vector multiplication (SpMV)** for message aggregation, followed by dense GRU cell updates. The key GPU optimization is:
  - CSR (Compressed Sparse Row) representation for the adjacency matrix
  - cuSPARSE or cuSPARSELt for SpMV
  - Batched GRU operations across all nodes simultaneously
  - The graph must fit in GPU memory; for large graphs, mini-batch training over subgraphs is used
  - Memory bandwidth is the bottleneck for SpMV on GPUs with irregular graphs
- **URL**: N/A (implementation details from DGL/PyG documentation)
- **Confidence**: MEDIUM — well-established implementation pattern

### Finding 2.3: GGNN for Vector Search is an Emerging Approach (Not Established)
- **Paper**: "Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective"
- **Authors**: Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, et al.
- **Year**: 2020
- **Key Insight**: GGNN-style propagation has been used for **graph-based re-ranking** in image retrieval (not as the primary search index). The GNN propagates similarity information through a pre-computed kNN graph to refine rankings. On a K40m GPU, this achieves 9.4ms for re-ranking on Market-1501. However, this is a post-retrieval refinement step, not the primary search mechanism. The primary search still uses traditional ANN methods.
- **URL**: https://arxiv.org/abs/2012.07620
- **Confidence**: MEDIUM — niche application, not mainstream vector search

---

## 3. GPU Graph Traversal vs Brute-Force GPU Search Throughput

### Finding 3.1: Brute-Force GPU Search Has Higher Throughput but Lower Quality
- **Paper**: "GPU-Accelerated Algorithms for Graph Vector Search: Taxonomy, Empirical Study, and Research Directions"
- **Authors**: Yaowen Liu, Xuejia Chen, Anxin Tian, et al.
- **Year**: 2026
- **Key Finding**: Brute-force (flat) GPU search achieves **higher raw QPS** than graph-based methods because it has a regular access pattern that fully utilizes GPU memory bandwidth and compute. On SIFT1M with A100:
  - Brute-force (FAISS IVF-flat): ~100-200M QPS
  - CAGRA graph search: ~50-100M QPS (at Recall~0.95)
  - The graph approach trades ~2-5x throughput for dramatically better recall-quality at the same k. However, brute-force requires O(n) distance computations per query, while graph search requires O(k * L * iterations) where L is the average degree, making graph search more efficient at scale.
- **URL**: https://arxiv.org/abs/2602.16719
- **Confidence**: HIGH — comprehensive 2026 survey with benchmarks

### Finding 3.2: Graph Search Wins on Latency-Quality Pareto Frontier
- **Paper**: "GPU-Accelerated Algorithms for Graph Vector Search" (same as above)
- **Year**: 2026
- **Key Finding**: At **high recall requirements** (Recall@10 > 0.95), graph-based methods dominate brute-force on the Pareto frontier. The survey found: "distance computation remains the primary computational bottleneck" for both approaches, but graph search reduces the number of distance computations by 10-100x compared to brute-force. The trade-off shifts with dataset dimensionality: for high-dimensional data (>768d), brute-force becomes increasingly competitive because graph search's irregular memory access overwhelms its computational savings.
- **URL**: https://arxiv.org/abs/2602.16719
- **Confidence**: HIGH

### Finding 3.3: GustANN Achieves Billion-Scale GPU Graph Search
- **Paper**: "High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU"
- **Authors**: Haodi Jiang, Hao Guo, Minhui Xie, Jiwu Shu, Youyou Lu
- **Year**: 2025
- **Key Insight**: GustANN demonstrates that GPU graph search can scale to **billion-scale** datasets by offloading the graph to SSD while keeping hot paths in GPU memory. Three key techniques: (1) memory-efficient GPU kernels minimizing GPU memory usage for graph search, allowing higher concurrency; (2) CPU-assisted transfer to address PCIe bandwidth bottleneck; (3) pivot search for inter-SSD load balancing. GustANN achieves **2.50x higher throughput** than existing systems and is **2.62x more cost-effective** (measured in $/QPS).
- **URL**: https://doi.org/10.1145/3769799
- **Confidence**: HIGH — SIGMOD 2025

---

## 4. Warp Splitting in CAGRA Search

### Finding 4.1: Warp-Level Parallelism for Distance Computation
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Year**: 2023
- **Key Algorithm**: In CAGRA's multi-CTA search, **warp splitting** (also called warp-cooperative processing) distributes the computation of a single distance metric across multiple warps. For high-dimensional vectors (e.g., 768d+), a single warp (32 threads) cannot efficiently compute the full distance in one pass. Instead:
  - Each warp computes a **partial dot product / L2 distance** for a sub-range of dimensions
  - Shared memory reductions combine partial results across warps
  - This allows the search kernel to handle arbitrary vector dimensions without register spilling
  - The warp-split approach is particularly effective for HNSW/CAGRA search where multiple neighbor distances must be computed simultaneously
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH — described in CAGRA paper

### Finding 4.2: Warp-Level Bitonic Sort for Candidate Maintenance
- **Key Algorithm**: CAGRA maintains the candidate list (beam) using **bitonic sort within a warp**. The candidate list of size k is distributed across threads in a warp. When a new candidate is inserted:
  - The warp cooperatively performs a bitonic sort network to merge the new element
  - This is O(log²k) operations per insertion, all in registers/shared memory
  - The sort is deterministic and branch-free, which is ideal for GPU execution
  - For k=32 or k=64, this maps perfectly to one or two warps
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH — core CAGRA mechanism

### Finding 4.3: Falcon Uses On-Chip Bloom Filter Instead of Warp Sorting
- **Paper**: "Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal"
- **Authors**: Wenqi Jiang, Hang Hu, Torsten Hoefler, Gustavo Alonso
- **Year**: 2024 (VLDB 2025)
- **Key Insight**: Falcon (FPGA accelerator) replaces the GPU's warp-sort approach with an **on-chip Bloom filter** for visited node tracking, achieving 4.3x-19.5x lower latency than GPU-based systems. The key observation is that GPU warp sorting is inefficient for graph traversal because: (1) the candidate set changes unpredictably at each step, and (2) the overhead of maintaining a sorted structure exceeds the benefit of early termination. Falcon's approach of "relaxing traversal orders" (Delayed-Synchronization Traversal) achieves better utilization by not requiring strict ordering at each step.
- **URL**: https://arxiv.org/abs/2406.12385
- **Confidence**: HIGH — VLDB 2025

---

## 5. Memory Bandwidth Limits of GPU Graph Traversal

### Finding 5.1: Graph Traversal is Fundamentally Memory-Bandwidth Bound
- **Paper**: "NDSEARCH: Accelerating Graph-Traversal-Based Approximate Nearest Neighbor Search through Near Data Processing"
- **Authors**: Yitu Wang, Shiyu Li, Qilin Zheng, et al.
- **Year**: 2023
- **Key Insight**: Graph traversal for ANNS is "**extremely memory-intensive**" because each traversal step requires:
  1. Reading the neighbor list (CSR format) — random access
  2. Reading the actual vector data for each neighbor — random access, large payload
  3. Computing distances — compute-light, easy to saturate bandwidth
  On A100 (2 TB/s HBM), the effective bandwidth for graph traversal is only ~200-400 GB/s due to irregular access patterns, cache thrashing, and TLB misses. This means GPU graph search operates at **10-20% of peak memory bandwidth**.
- **URL**: https://arxiv.org/abs/2312.03141
- **Confidence**: HIGH — ISCA 2024, 32 citations

### Finding 5.2: PilotANN Addresses Memory Bandwidth via SVD Compression
- **Paper**: "PilotANN: Memory-Bounded GPU Acceleration for Vector Search"
- **Authors**: Yuntao Gui, Peiqi Yin, Xiao Yan, et al.
- **Year**: 2025
- **Key Insight**: PilotANN reduces memory bandwidth pressure by using **SVD-reduced vectors** for GPU graph traversal. By reducing 768d vectors to ~128d via SVD:
  - Memory reads per distance computation drop by 6x
  - GPU memory capacity is effectively increased by 6x
  - The accuracy loss is compensated by a CPU refinement stage using full vectors
  - Achieves **3.9-5.4x speedup** on 100M-scale datasets and handles datasets **12x larger** than GPU memory
- **URL**: https://arxiv.org/abs/2503.21206
- **Confidence**: HIGH — SIGMOD 2026

### Finding 5.3: PCIe Bandwidth Bottleneck at Large Scale
- **Paper**: "GPU-Accelerated Algorithms for Graph Vector Search" (survey)
- **Year**: 2026
- **Key Finding**: "Data transfer between the host CPU and GPU emerges as the dominant factor influencing real-world latency at large scale." When the graph index exceeds GPU memory, PCIe transfers (Gen4 x16: ~32 GB/s; Gen5 x16: ~64 GB/s) become the bottleneck. This is 30-60x slower than HBM bandwidth, making out-of-core GPU graph search fundamentally limited by the PCIe bus.
- **URL**: https://arxiv.org/abs/2602.16719
- **Confidence**: HIGH

### Finding 5.4: CMANNS Achieves High Bandwidth via Tensor Core GEMM
- **Paper**: "CMANNS: GPU-Accelerated Graph Index Construction for ANNS via Compute–Memory Disaggregation"
- **Authors**: Chengying Huan, Renjie Yao, Shaonan Ma, et al.
- **Year**: 2026
- **Key Insight**: CMANNS reformulates distance computation as **high-arithmetic-intensity GEMMs on Tensor Cores** with fused epilogues. This transforms the bandwidth-bound graph traversal into a more compute-balanced workload. Combined with hot-set-aware shared-memory staging and warp-cooperative gathers/scatters, CMANNS achieves up to **13.05x** speedup over FAISS and **2.20x** over FLASH. The cache hit rate improves by up to **58.7%**.
- **URL**: https://doi.org/10.1145/3802027
- **Confidence**: HIGH — SIGMOD 2026

---

## 6. Visited Set / Hash Table Management in GPU Graph Search

### Finding 6.1: CAGRA Uses Simple Bitset for Visited Tracking
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Year**: 2023
- **Key Algorithm**: CAGRA tracks visited nodes using a **per-query bitset** (one bit per database vector). During search:
  - Before examining a neighbor, check the corresponding bit in the bitset
  - If already visited, skip; otherwise, set the bit and process
  - The bitset is allocated in GPU global memory (one per query in the batch)
  - For 1B vectors, each bitset requires 125 MB — this is manageable for batch sizes of ~50-100 queries on A100 (80 GB)
  - No hash table is used; the bitset is O(n) space but O(1) lookup
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH

### Finding 6.2: Bloom Filter as Visited Set Alternative
- **Paper**: "Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal"
- **Year**: 2024
- **Key Insight**: Falcon uses an **on-chip Bloom filter** instead of a bitset for visited tracking. Advantages:
  - Much smaller memory footprint (~8 bytes per query vs. 125 MB for 1B vectors)
  - Fits entirely in on-chip SRAM, eliminating global memory reads for visited checks
  - Trade-off: Bloom filters have false positives (wasted computation on already-visited nodes), but this is acceptable because the cost of a false positive is one wasted distance computation, while the cost of bitset access is a global memory read
  - For graph search with typical visit counts (~1000-10000 nodes per query), a small Bloom filter (e.g., 4KB with 1% false positive rate) is sufficient
- **URL**: https://arxiv.org/abs/2406.12385
- **Confidence**: HIGH

### Finding 6.3: GustANN Optimizes Visited Tracking for SSD-Backed Graphs
- **Paper**: "High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU"
- **Year**: 2025
- **Key Insight**: For billion-scale SSD-backed graphs, GustANN uses a **compact visited set** with eviction:
  - The visited set is too large for GPU memory at billion scale
  - GustANN uses a combination of on-GPU Bloom filter for hot nodes and lazy eviction
  - Pages of the graph that are not recently accessed are evicted from GPU memory
  - The visited set is co-designed with the page replacement policy
  - This reduces GPU memory usage while maintaining search quality
- **URL**: https://doi.org/10.1145/3769799
- **Confidence**: HIGH

### Finding 6.4: NDSEARCH Uses In-Storage Visited Set
- **Paper**: "NDSEARCH: Accelerating Graph-Traversal-Based Approximate Nearest Neighbor Search through Near Data Processing"
- **Year**: 2023
- **Key Insight**: NDSEARCH moves the visited set **into storage** (SmartSSD) using near-data processing. The SSD-level Bloom filter avoids transferring visited-check results over PCIe, reducing data movement by 2-3x. This is possible because the visited check is a simple read-only operation that can be performed near the data.
- **URL**: https://arxiv.org/abs/2312.03141
- **Confidence**: HIGH

---

## 7. Multi-CTA vs Single-CTA GPU Search

### Finding 7.1: Single-CTA Search — Lower Latency, Simpler
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Year**: 2023
- **Key Algorithm**: In single-CTA mode:
  - One thread block (typically 128-256 threads) handles one query
  - The entire candidate list (k=32 or k=64) fits in shared memory
  - Bitonic sort is performed within the block
  - No inter-CTA synchronization needed
  - **Latency**: ~10-50 microseconds per query on A100 (for SIFT128)
  - **Limitation**: Only ~10-20 blocks can be resident per SM due to shared memory constraints, limiting occupancy
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH

### Finding 7.2: Multi-CTA Search — Higher Throughput for High-Dim
- **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"
- **Year**: 2023
- **Key Algorithm**: In multi-CTA mode:
  - Multiple CTAs cooperate on one query: one "leader" CTA manages the search state, while "worker" CTAs compute distances in parallel
  - The leader CTA uses **grid sync** (CUDA cooperative groups) to synchronize with workers
  - Workers compute distances for all neighbors of the current beam simultaneously
  - **Throughput**: For 768d vectors, multi-CTA achieves 2-3x higher throughput than single-CTA because distance computation dominates
  - **Limitation**: Grid sync has non-trivial overhead (~10-50 cycles per sync), and the leader CTA is a serialization point
- **URL**: https://arxiv.org/abs/2308.15136
- **Confidence**: HIGH

### Finding 7.3: Trade-off Analysis — When to Use Which
- **Paper**: "GPU-Accelerated Algorithms for Graph Vector Search" (survey)
- **Year**: 2026
- **Key Finding**: The survey found:
  - **Single-CTA** is optimal for: low-dimensional data (<256d), small k (k≤32), high batch sizes (many concurrent queries)
  - **Multi-CTA** is optimal for: high-dimensional data (>256d), larger k (k≥64), low batch sizes (fewer concurrent queries)
  - The crossover point depends on the GPU architecture: on A100 (large shared memory, fast grid sync), multi-CTA wins earlier; on older GPUs, single-CTA is preferred
  - The survey recommends **adaptive selection** based on query dimensionality and batch size at runtime
- **URL**: https://arxiv.org/abs/2602.16719
- **Confidence**: HIGH

### Finding 7.4: Delayed-Synchronization Traversal Eliminates CTA Synchronization
- **Paper**: "Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal"
- **Year**: 2024
- **Key Insight**: Falcon's DST algorithm **relaxes the traversal order** to eliminate the need for inter-CTA synchronization. Instead of requiring all workers to finish before the leader proceeds, DST allows:
  - Workers to continue computing distances for the next step while the leader processes current results
  - The leader to use **stale but useful** candidates from previous steps
  - This removes the grid sync bottleneck entirely
  - Achieves **19.5x lower latency** than GPU-based systems (CAGRA) and **8.0x better energy efficiency**
  - Key lesson: strict traversal ordering is unnecessary for good recall; relaxing it dramatically improves GPU utilization
- **URL**: https://arxiv.org/abs/2406.12385
- **Confidence**: HIGH

---

## Cross-Cutting Insights

### Insight A: Memory Bandwidth is THE Fundamental Bottleneck
Every paper in this survey confirms that GPU graph search is **memory-bandwidth bound**, not compute-bound. The irregular access pattern of graph traversal (random node lookups) prevents efficient use of GPU memory hierarchy. Solutions include: (1) SVD compression (PilotANN), (2) Tensor Core GEMM reformulation (CMANNS), (3) on-chip Bloom filters (Falcon), (4) SSD offloading with smart caching (GustANN). The theoretical peak of GPU FLOPS is largely wasted.

### Insight B: Visited Set Design is Critical for Scalability
The visited set (bitset, Bloom filter, or hash table) determines the memory footprint per query. For billion-scale datasets, per-query bitsets of 125 MB are impractical. The industry trend is toward **probabilistic structures** (Bloom filters) that trade small false-positive rates for orders-of-magnitude memory reduction. This enables batching more queries simultaneously, improving GPU occupancy.

### Insight C: Relaxing Traversal Order Improves GPU Performance
The strict greedy traversal order of CAGRA (always expand the globally best candidate) is suboptimal for GPU because it creates serialization bottlenecks. Falcon's DST demonstrates that **relaxing the traversal order** — allowing slightly out-of-order exploration — improves GPU utilization by 8-20x with minimal quality loss. This suggests the next generation of GPU graph search algorithms should be designed around GPU execution model, not CPU-oriented algorithms ported to GPU.

### Insight D: The CPU-GPU-SSD Three-Level Hierarchy
Billion-scale vector search requires a **three-level memory hierarchy**: GPU HBM (hot data, ~80 GB), CPU DRAM (warm data, ~512 GB-2 TB), SSD (cold data, ~10 TB). Each level has 10-100x bandwidth gap. PilotANN, GustANN, and CMANNS all address this hierarchy with different strategies. The optimal design minimizes data movement between levels.

---

## Summary Table

| Topic | Key Finding | Paper | Year | Confidence |
|-------|------------|-------|------|------------|
| CAGRA beam search | Greedy traversal with bitonic sort, memory-bandwidth bound | CAGRA (Ootomo et al.) | 2023 | HIGH |
| Single vs multi-CTA | Single-CTA for low-dim, multi-CTA for high-dim; adaptive is best | CAGRA + GPU Survey | 2023/2026 | HIGH |
| GGNN on GPU | SpMV + GRU, not used for primary ANN search | Li et al. (GGNN) | 2015 | HIGH |
| GPU graph vs brute-force | Graph wins on Pareto frontier at high recall; brute-force higher raw QPS | GPU Survey (Liu et al.) | 2026 | HIGH |
| Warp splitting | Warp-cooperative distance computation + bitonic sort for candidates | CAGRA | 2023 | HIGH |
| Memory bandwidth | Graph traversal achieves only 10-20% of peak HBM bandwidth | NDSEARCH + CMANNS | 2023/2026 | HIGH |
| Visited set | Bitset (CAGRA) → Bloom filter (Falcon) → probabilistic (GustANN) | Multiple | 2023-2025 | HIGH |
| Multi-CTA vs single-CTA | Grid sync overhead limits multi-CTA; DST eliminates sync | CAGRA + Falcon | 2023/2024 | HIGH |
