# CAGRA: GPU-Native ANN Index — Technical Deep Dive

> **Paper**: "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs" (arXiv:2308.15136, ICDE 2024)
> **Authors**: Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang (NVIDIA)
> **Open-source**: [cuVS](https://github.com/NVIDIA/cuvs) (part of NVIDIA RAPIDS)

---

## Table of Contents

1. [Overview](#1-overview)
2. [How CAGRA Builds kNN Graph on GPU](#2-how-cagra-builds-knn-graph-on-gpu)
3. [How CAGRA Searches on GPU](#3-how-cagra-searches-on-gpu)
4. [CAGRA vs HNSW Key Differences](#4-cagra-vs-hnsw-key-differences)
5. [CAGRA Memory Layout for GPU Coalesced Access](#5-cagra-memory-layout-for-gpu-coalesced-access)
6. [Key CUDA Kernels](#6-key-cuda-kernels)
7. [Open-Source Implementation in cuVS](#7-open-source-implementation-in-cuvs)
8. [Pseudocode](#8-pseudocode)
9. [Performance Numbers](#9-performance-numbers)
10. [References](#10-references)

---

## 1. Overview

CAGRA (**C**UDA **A**NN **GRA**ph-based) is a graph-based ANN index designed from the ground up for GPU parallelism. Unlike CPU-oriented methods (HNSW, NSG) that were later ported to GPU, CAGRA is a GPU-native design where both the graph structure and the search algorithm are co-designed with the GPU architecture in mind.

### Core Design Principles

1. **Fixed out-degree** — every node has exactly `d` outgoing edges (unlike HNSW's variable degree). This enables uniform GPU workload distribution.
2. **No hierarchy** — flat graph (no HNSW-style layers). Initial nodes found via parallel random sampling instead of hierarchical navigation.
3. **Directional edges** — directed graph, edges point from a node to its neighbors.
4. **Rank-based optimization** — graph pruning uses rank positions (not distance computations) to avoid expensive distance tables.

### Two-Phase Architecture

```
Phase 1: Build
  ┌─────────────────────────────┐
  │ 1. Build kNN graph (GPU)   │  ← NN-Descent or IVF-PQ
  │ 2. Optimize/prune graph     │  ← Rank-based reordering + reverse edges
  └─────────────────────────────┘

Phase 2: Search
  ┌─────────────────────────────┐
  │ 1. Random seed sampling     │  ← Parallel across queries
  │ 2. Graph traversal loop     │  ← Wavefront/batch expansion
  │ 3. Priority queue update    │  ← Top-M maintenance
  └─────────────────────────────┘
```

---

## 2. How CAGRA Builds kNN Graph on GPU

### 2.1 Initial kNN Graph Construction

CAGRA supports multiple GPU-native algorithms for building the initial kNN graph:

#### Option A: NN-Descent (GPU-accelerated)

NN-Descent is an iterative algorithm where each node maintains a small local neighborhood that is repeatedly refined.

```
Algorithm: GPU NN-Descent
─────────────────────────
Input: Dataset D[0..N-1], intermediate degree d_init = 3*d

1. For each node i, randomly select d_init neighbors → graph[i]
2. Repeat until convergence (or max iterations):
   a. For each node i (parallel across all N nodes):
      - Sample neighbors of neighbors → candidate set C
      - Compute distances from D[i] to all candidates in C
      - Merge graph[i] with C, keep top-d_init by distance
   b. Check convergence: if edge changes < threshold, stop

Output: kNN graph G[N][d_init]
```

**GPU parallelism**: Each of the N nodes is assigned to a CUDA thread block. Within a block, threads cooperate to compute distances to candidates using warp-level parallelism.

**Memory**: NN-Descent on GPU uses:
- Device memory: `N × (dims × 2 + 276)` bytes (vectors + working buffers)
- Host memory: `N × (13 × d_init + 912)` bytes (full graph + bloom filter + sample buffers)

#### Option B: IVF-PQ (supports out-of-core)

For datasets larger than GPU memory, IVF-PQ builds the kNN graph in batches:

```
Algorithm: IVF-PQ-based kNN Graph Build
────────────────────────────────────────
1. Train IVF-PQ index on a subset of data
2. For each batch of query vectors:
   a. Query the IVF-PQ index for approximate nearest neighbors
   b. Store results in the kNN graph
3. The resulting graph serves as input to CAGRA optimization
```

#### Option C: ACE (Accelerated CAGRA build)

A newer algorithm that uses CAGRA's own search to iteratively refine the graph during construction.

### 2.2 Graph Optimization (Pruning)

After building the initial kNN graph with degree `d_init` (typically 2× or 3× the final degree `d`), CAGRA optimizes it to reduce degree while preserving reachability.

#### Step 1: Rank-Based Reordering

**Key insight**: Instead of computing exact distances for detourability checks (O(N × d_init³) distance computations), CAGRA uses the **rank position** of edges as a proxy for distance.

```
Algorithm: Rank-Based Reordering
────────────────────────────────
Input: kNN graph G with sorted neighbor lists (by distance)
Output: Reordered graph G' with top-d edges per node

For each node X (parallel across N):
  For each neighbor Y of X (parallel across d_init):
    Count = 0
    For each neighbor Z of X:
      If Z is also a neighbor of Y:
        // Check detourability using rank positions:
        If max(rank(X→Z), rank(Z→Y)) < rank(X→Y):
          Count += 1  // X→Y is "detourable" via X→Z→Y
    detourable_count[X][Y] = Count
  
  // Sort neighbors by detourable_count (ascending)
  // Edges with fewer detour routes are MORE important
  Sort neighbors of X by detourable_count
  Keep top-d neighbors
```

**Why rank-based is faster**: Avoids N × d_init × (d_init - 1) distance computations. Uses pre-computed rank positions (integer comparisons) instead.

#### Step 2: Reverse Edge Addition

Add reverse edges to improve graph connectivity and reduce strongly connected components.

```
Algorithm: Reverse Edge Addition
────────────────────────────────
1. Create reversed graph: flip all edge directions
2. Sort reversed edges by rank (ascending) — "someone who
   considers you more important is also more important to you"
3. Cap each node's reversed-degree at d
4. Merge: interleave d/2 forward edges + d/2 reverse edges
   (compensate if reversed graph has fewer than d/2 edges)
```

**Effect**: Strongly connected components decrease dramatically; 2-hop node count increases (more exploration per iteration).

---

## 3. How CAGRA Searches on GPU

### 3.1 Search Algorithm Overview

CAGRA search is a **wavefront traversal** algorithm that explores the graph in batches.

```
Search Data Structures
──────────────────────
┌─────────────────────────────────────────┐
│ Internal Top-M List: [M entries]         │  ← sorted by distance (priority queue)
│ Candidate List: [p × d entries]          │  ← neighbors of p source nodes
│ Visited Hash Table: [O(p × d × iterations)] │ ← dedup visited nodes
└─────────────────────────────────────────┘

Where:
  M ≥ k (intermediate top-k size, typically 64-128)
  p = search_width (number of source nodes explored per iteration)
  d = graph degree (typically 32-64)
```

### 3.2 Search Loop

```
Algorithm: CAGRA Search (Single Query)
──────────────────────────────────────
Input: Query q, graph G, dataset D, parameters (M, iterations)

Step 0: INITIALIZATION
  Initialize top-M list with dummy entries (distance = FLT_MAX)
  Generate p × d random node indices → candidate_list
  Mark all as "unvisited"

For iter = 0 to max_iterations:

  Step 1: DISTANCE COMPUTATION
    For each node in candidate_list where distance not yet computed:
      Compute distance(q, D[node])  // GPU: many in parallel
      Mark as visited in hash table
  
  Step 2: UPDATE TOP-M
    Merge candidate_list with internal top-M list
    Sort combined buffer, keep top-M by distance
  
  Step 3: GRAPH TRAVERSAL
    Select top-p nodes from internal top-M list
    For each selected node X:
      For each neighbor Y of X:
        If Y not yet visited:
          Add Y to candidate_list
    (Distance not computed yet — deferred to next iteration)

Return top-k from internal top-M list
```

### 3.3 GPU-Parallel Execution Model

The critical innovation is how this maps to GPU execution:

**Warp Splitting**: Multiple queries (or multiple source nodes within a query) are processed concurrently within a single warp.

**Team Size**: A "team" of threads cooperatively computes one distance. Common values: 4, 8, 16, or 32 threads per distance computation. This enables coalesced vector loads.

```
GPU Execution Layout (Multi-CTA)
─────────────────────────────────
  CTA 0: Process Query 0 (wavefront iteration)
  CTA 1: Process Query 1 (wavefront iteration)
  CTA 2: Process Query 2 (wavefront iteration)
  ...
  CTA N: Process Query N (wavefront iteration)

  Within each CTA:
    - team_size threads cooperate on one distance computation
    - Multiple distances computed in parallel via warp-level parallelism
    - Shared memory holds top-M list and candidate buffer
```

**Single-CTA Mode**: For large batches — each CTA handles one query independently. Multiple CTAs process different queries in parallel.

**Multi-CTA Mode**: For small batches — multiple CTAs cooperate on a single query for higher per-query throughput. Uses cooperative groups for synchronization.

### 3.4 Forgettable Hash Table

To track visited nodes efficiently on GPU:

```
Algorithm: Forgettable Hash Table
──────────────────────────────────
Purpose: Track visited nodes without full allocation for all N nodes

Implementation:
  - Open-address hash table in GPU shared memory or registers
  - Key = node index, Value = "visited" flag
  - When hash table fills up → evict oldest entries (forget)
  - Re-visit penalty is bounded: a node whose distance was
    already small enough is already in the top-M list; one
    that was too large won't help

Trade-off: Small memory footprint vs. occasional re-visits
```

### 3.5 1-Bit Parented Node Management

To track which nodes have already been used as traversal sources (preventing redundant exploration):

```
Each node has a 1-bit flag:
  0 = not yet used as a source (can still be expanded)
  1 = already used as a source (neighbors already added to candidates)

During Step 3 (graph traversal):
  For top-p nodes:
    If node is not parented (bit = 0):
      Add neighbors to candidate list
      Set bit = 1
    Else:
      Skip (neighbors already explored)
```

This eliminates redundant graph exploration and bounds the total work per query.

---

## 4. CAGRA vs HNSW Key Differences

| Aspect | CAGRA | HNSW |
|--------|-------|------|
| **Design target** | GPU-native (designed for massive parallelism) | CPU-native (designed for sequential traversal) |
| **Graph structure** | Flat, single-layer, directed, fixed out-degree | Multi-layer hierarchical, variable out-degree |
| **Out-degree** | Fixed `d` for all nodes (e.g., 32, 64) | Variable: 2×M at layer 0, ≤M at upper layers |
| **Entry point selection** | Random parallel sampling (GPU bandwidth) | Hierarchical descent (sequential) |
| **Distance computation** | Many distances computed in parallel per iteration | Single-threaded sequential |
| **Graph construction** | GPU-accelerated NN-Descent + rank-based pruning | CPU sequential insertion with backlinks |
| **Memory layout** | Flat arrays (row-major), GPU coalesced | Pointer-based adjacency lists, CPU cache-optimized |
| **Batch throughput** | Excellent: queries parallelized across CTAs | Good: batch queries parallelized with OpenMP |
| **Single-query latency** | Excellent on GPU (3.4–53× faster than HNSW @95% recall) | Good on CPU |
| **Recall@10** | 90-95%+ (tunable via `itopk_size`) | 90-98%+ (tunable via `ef`) |
| **Build speed** | 2.2–27× faster than HNSW | Moderate |
| **Interoperability** | Can export to HNSW format for CPU search | Native CPU format |
| **Filtering** | Native filtered search support | Separate implementation needed |
| **Insertion** | Batch-only (extend API) | Online (insert one at a time) |
| **GPU memory** | Requires dataset + graph on GPU | CPU-only |

### Key Trade-offs

1. **HNSW advantage**: Online insertion of new points (CAGRA requires batch rebuild or extend)
2. **HNSW advantage**: CPU-only deployment (no GPU needed for search)
3. **CAGRA advantage**: Orders-of-magnitude higher throughput on GPU
4. **CAGRA advantage**: Simpler graph construction (no hierarchical layer management)
5. **CAGRA advantage**: Flat structure enables more efficient GPU memory access patterns

---

## 5. CAGRA Memory Layout for GPU Coalesced Access

### 5.1 Dataset Storage

```
Dataset Layout (Row-Major, 16-byte Aligned)
────────────────────────────────────────────
Address:  base + row_idx * dim * sizeof(T)

  Row 0: [v0_0, v0_1, v0_2, ..., v0_{dim-1}]
  Row 1: [v1_0, v1_1, v1_2, ..., v1_{dim-1}]
  ...
  Row N: [vN_0, vN_1, vN_2, ..., vN_{dim-1}]

  T = float (4B), half (2B), or int8_t (1B)
  Rows padded to 16-byte boundary for vectorized loads
```

**Coalesced access pattern**: When a team of threads computes distance(q, D[node]), consecutive threads read consecutive 4-byte floats from the same row, achieving 128-byte cache line utilization.

### 5.2 Graph Storage

```
Graph Layout (Row-Major Adjacency Matrix)
──────────────────────────────────────────
Address:  base + row_idx * graph_degree * sizeof(IdxT)

  Node 0: [neighbor_0, neighbor_1, ..., neighbor_{d-1}]
  Node 1: [neighbor_0, neighbor_1, ..., neighbor_{d-1}]
  ...
  Node N: [neighbor_0, neighbor_1, ..., neighbor_{d-1}]

  IdxT = uint32_t (4B) or uint16_t (2B)
  All rows have identical length (fixed degree)
```

**Why fixed degree matters**: Every thread block processes the same number of neighbors, eliminating divergence. Neighbors are stored in row-major order for coalesced reads.

### 5.3 Search Buffer Layout

```
Search Buffer (Per Query, in Shared Memory / Registers)
───────────────────────────────────────────────────────

┌─────────────────────────────────────────────────┐
│ Internal Top-M List (sorted by distance)        │
│  [idx_0:dist_0, idx_1:dist_1, ..., idx_M:dist_M] │
│  → Contiguous in memory, merge-sort friendly    │
├─────────────────────────────────────────────────┤
│ Candidate List (unsorted)                       │
│  [idx_0, idx_1, ..., idx_{p×d-1}]              │
│  → Node indices only (distances computed later)  │
├─────────────────────────────────────────────────┤
│ Visited Hash Table (open-addressing)            │
│  [slot_0, slot_1, ..., slot_H-1]                │
│  → Power-of-2 size for fast modular hashing     │
└─────────────────────────────────────────────────┘
```

### 5.4 Coalesced Access Patterns

```
Pattern 1: Graph Traversal (reading neighbors)
  Thread 0 reads G[node0][0..d-1]  → d consecutive IdxT values
  Thread 1 reads G[node1][0..d-1]  → d consecutive IdxT values
  ...
  Coalesced: Yes (each thread reads its own row, d × 4B per thread)

Pattern 2: Distance Computation (reading dataset)
  Team of 32 threads reads D[node][0..dim-1] together
  Thread i reads D[node][i*4 .. i*4+3]  → 16 bytes (4 floats)
  ...
  Coalesced: Yes (128B = full cache line per 32 threads × 4B)

Pattern 3: Top-M Update (sorting)
  Parallel bitonic sort on 128 elements (registers/shared memory)
  No global memory traffic during sort
```

### 5.5 Memory Footprint

```
Total GPU Memory = Dataset + Graph + Search Workspace

Dataset:
  N × dim × sizeof(T)
  Example: 1M vectors × 1024 dim × 4B = 3,906 MB

Graph:
  N × graph_degree × sizeof(IdxT)
  Example: 1M × 64 × 4B = 244 MB

Search Workspace (per batch):
  batch_size × dim × sizeof(float)  →  query vectors
  batch_size × k × (sizeof(IdxT) + sizeof(float))  →  results
  Example: 100 queries × 1024 × 4B ≈ 0.4 MB

Total ≈ 4,150 MB (for 1M vectors, dim=1024, fp32, degree=64)
```

---

## 6. Key CUDA Kernels

### 6.1 Graph Construction Kernels

```
Kernel 1: nn_descent_iterate
  Purpose: One iteration of NN-Descent
  Grid: N / blockDim.x blocks
  Block: 256 threads
  Shared memory: candidate buffer
  Global read: D[0..N-1] (dataset), G[0..N-1] (current graph)
  Global write: G[0..N-1] (updated graph)

Kernel 2: rank_based_reorder
  Purpose: Count detourable routes and reorder edges
  Grid: N / blockDim.x blocks
  Block: 256 threads
  Shared memory: detour counts for d_init neighbors
  Global read: G[0..N-1] (sorted kNN graph)
  Global write: G[0..N-1] (reordered, pruned to d edges)

Kernel 3: reverse_edge_merge
  Purpose: Add reverse edges and merge with forward edges
  Grid: N / blockDim.x blocks
  Block: 256 threads
  Global read: G_forward, G_reversed
  Global write: G_final[0..N-1][d]
```

### 6.2 Search Kernels

```
Kernel 4: cagra_search_single_cta
  Purpose: Complete search for one query in one CTA
  Grid: batch_size / blockDim.x blocks (one CTA per query)
  Block: 128-1024 threads
  Shared memory: top-M list, candidate buffer, hash table
  Global read: G[0..N-1] (graph), D[0..N-1] (dataset), q (query)
  Global write: results[query_id][0..k-1]

Kernel 5: cagra_search_multi_cta
  Purpose: Search for one query across multiple CTAs
  Grid: (batch_size × CTAs_per_query) / blockDim.x blocks
  Block: 128-256 threads
  Shared memory: partial results per CTA
  Global read: same as single_cta
  Global write: partial results → merge kernel → final results

Kernel 6: cagra_distance_compute
  Purpose: Compute distance between query and candidate vectors
  Grid: (batch_size × candidates_per_batch) / blockDim.x blocks
  Block: team_size (4, 8, 16, or 32) threads
  Team-cooperative: team_size threads compute ONE distance together
  Uses vectorized loads (float4) for coalesced access

Kernel 7: cagra_topk_merge
  Purpose: Merge candidate distances with top-M list
  Grid: batch_size / blockDim.x blocks
  Block: 256 threads
  Shared memory: combined list of size M + p×d
  Uses parallel merge sort or bitonic sort

Kernel 8: cagra_hash_table_ops
  Purpose: Insert/check visited nodes in hash table
  Grid: batch_size / blockDim.x blocks
  Block: 256 threads
  Shared memory: hash table (power-of-2 size)
  Operations: insert, lookup, evict (forgettable)
```

### 6.3 Support Kernels

```
Kernel 9: random_seed_selection
  Purpose: Generate random initial nodes for search
  Grid: batch_size / blockDim.x blocks
  Block: 256 threads
  Uses: XOR-shift PRNG for reproducible random numbers

Kernel 10: select_k (raft primitive)
  Purpose: Top-K selection on GPU
  Used in: top-M update, distance-based sorting
  Implementation: Parallel radix select or warp-level sort

Kernel 11: mst_optimize (optional)
  Purpose: Degree-constrained MST for guaranteed connectivity
  Grid: N / blockDim.x blocks
  Used when: guarantee_connectivity = true
```

---

## 7. Open-Source Implementation in cuVS

### 7.1 Repository Structure

```
cuvs/
├── cpp/src/neighbors/
│   ├── cagra/
│   │   ├── build.cuh          # Index build entry point
│   │   ├── search.cuh         # Search entry point
│   │   ├── optimize.cuh       # Graph optimization (pruning)
│   │   ├── detail/
│   │   │   ├── graph_op.cuh   # Graph construction operations
│   │   │   ├── hashmap.cuh    # Forgettable hash table
│   │   │   ├── search_single_cta.cuh  # Single-CTA search kernel
│   │   │   ├── search_multi_cta.cuh   # Multi-CTA search kernel
│   │   │   └── search_multi_kernel.cuh # Multi-kernel search
│   │   └── cagra.cuh          # Main include
│   ├── nn_descent/            # NN-Descent graph builder
│   ├── ivf_pq/                # IVF-PQ for initial graph
│   └── ...                    # Other ANN indices
├── python/cuvs/
│   └── neighbors/
│       └── cagra.py           # Python bindings
└── examples/
    └── c/neighbors/cagra_c.cu # C API example
```

### 7.2 Python API

```python
from cuvs.neighbors import cagra

# Build index
dataset = load_data()  # numpy array or cupy array
index_params = cagra.IndexParams(
    graph_degree=64,                    # Final graph degree
    intermediate_graph_degree=128,       # Initial kNN degree (2x)
    graph_build_algo='nn_descent',      # or 'ivf_pq', 'ace'
    guarantee_connectivity=False,
)
index = cagra.build(index_params, dataset)

# Search
search_params = cagra.SearchParams(
    itopk_size=64,          # Intermediate top-k
    max_iterations=0,       # Auto-select
    search_width=1,         # Source nodes per iteration
    team_size=0,            # Auto-select (4, 8, 16, or 32)
)
neighbors, distances = cagra.search(search_params, index, queries, k=10)
```

### 7.3 C++ API

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_resources.hpp>

using namespace cuvs::neighbors;

raft::resources res;
raft::device_matrix_view<const float, int64_t> dataset = load_dataset(res);

// Build
cagra::index_params index_params;
index_params.graph_degree = 64;
index_params.intermediate_graph_degree = 128;
index_params.graph_build_params = cagra::graph_build_params::nn_descent_params(128);

auto index = cagra::build(res, index_params, dataset);

// Search
cagra::search_params search_params;
search_params.itopk_size = 64;
search_params.algo = cagra::search_algo::SINGLE_CTA;

auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float>(res, n_queries, k);

cagra::search(res, search_params, index, queries, 
               neighbors.view(), distances.view());
```

### 7.4 C API

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraIndexParams_t index_params;
cuvsCagraIndex_t index;
cuvsCagraSearchParams_t search_params;

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&index_params);
cuvsCagraIndexCreate(&index);

cuvsCagraBuild(res, index_params, dataset_tensor, index);

cuvsCagraSearchParamsCreate(&search_params);
cuvsCagraSearch(res, search_params, index, queries_tensor, 
                neighbors_tensor, distances_tensor);

cuvsCagraIndexDestroy(index);
cuvsCagraIndexParamsDestroy(index_params);
cuvsCagraSearchParamsDestroy(search_params);
cuvsResourcesDestroy(res);
```

### 7.5 Key Build Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph_degree` | 64 | Final graph degree (all nodes) |
| `intermediate_graph_degree` | 128 | Degree of initial kNN graph before pruning |
| `graph_build_algo` | `ivf_pq` | Algorithm for initial graph: `nn_descent`, `ivf_pq`, `ace`, `brute_force` |
| `guarantee_connectivity` | false | Use MST to ensure connected graph |
| `attach_dataset_on_build` | true | Attach dataset vectors to index |

### 7.6 Key Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `itopk_size` | 64 | Intermediate top-k size (≥ k) — main accuracy knob |
| `max_iterations` | 0 | Max search iterations (0 = auto) |
| `search_width` | 1 | Number of source nodes explored per iteration |
| `team_size` | 0 | Threads per distance computation (4, 8, 16, 32; 0 = auto) |
| `algo` | `auto` | `single_cta`, `multi_cta`, `multi_kernel`, `auto` |
| `hashmap_mode` | `auto` | `hash`, `small`, `auto` — hash table strategy |
| `persistent` | false | Use persistent kernel (low-latency) |

---

## 8. Pseudocode

### 8.1 Complete Graph Build Pipeline

```
function CAGRA_BUILD(dataset, params):
    N, dim = dataset.shape
    d_init = params.intermediate_graph_degree   // e.g., 128
    d = params.graph_degree                     // e.g., 64
    
    // Phase 1: Build initial kNN graph
    if params.build_algo == nn_descent:
        G = GPU_NN_DESCENT(dataset, d_init, max_iters=50)
    else if params.build_algo == ivf_pq:
        G = GPU_IVF_PQ_KNN(dataset, d_init, batch_size=1024)
    
    // Sort each node's neighbors by distance
    parallel_for i in 0..N:
        sort(G[i] by distance_to(dataset[i]))
    
    // Phase 2: Optimize graph
    // Step 1: Rank-based reordering + pruning
    parallel_for i in 0..N:
        for j in 0..d_init:
            count = 0
            for z in 0..d_init:
                if G[i][z] in neighbors_of(G[i][j]):
                    if max(rank(G[i][z]), rank_of(G[i][j], G[i][z])) < rank(G[i][j]):
                        count += 1
            detourable_count[i][j] = count
        
        sort G[i] by detourable_count ascending
        G_optimized[i] = G[i][0:d]  // keep top-d
    
    // Step 2: Reverse edges
    G_reversed = transpose(G_optimized)
    parallel_for i in 0..N:
        sort G_reversed[i] by rank ascending
        G_reversed[i] = G_reversed[i][0:min(d, |G_reversed[i]|)]
    
    // Step 3: Merge forward + reverse
    parallel_for i in 0..N:
        forward = G_optimized[i][0:d/2]
        reverse = G_reversed[i][0:d/2]
        G_final[i] = interleave(forward, reverse)
        // Compensate if reverse has fewer than d/2 edges
    
    return CAGRAIndex(dataset, G_final)
```

### 8.2 Complete Search Loop

```
function CAGRA_SEARCH(query, index, params):
    G = index.graph           // N × d adjacency matrix
    D = index.dataset         // N × dim vectors
    M = params.itopk_size     // e.g., 64
    p = params.search_width   // e.g., 1
    d = index.graph_degree    // e.g., 64
    
    // Allocate buffers (in shared memory)
    top_M[0..M-1] = {(idx: -1, dist: FLT_MAX) × M}
    candidates[0..p×d-1] = empty
    hash_table[0..H-1] = empty  // forgettable hash
    
    // Step 0: Random initialization
    seeds = random_sample(0..N-1, p×d)
    for i in 0..p×d-1:
        candidates[i] = seeds[i]
    
    for iter = 0..max_iterations:
        // Step 1: Compute distances for unvisited candidates
        for each node in candidates:
            if node not in hash_table:
                dist = L2(query, D[node])
                candidates[node] = dist
                hash_table.insert(node)
        
        // Step 2: Merge candidates into top_M
        combined = merge(top_M, candidates)  // parallel merge
        top_M = sorted(combined)[0:M]         // keep top-M
        
        // Step 3: Expand frontier
        new_candidates = empty
        source_nodes = top_M[0:p]  // top-p nodes as sources
        for each src in source_nodes:
            for each neighbor in G[src]:
                if neighbor not in hash_table:
                    new_candidates.append(neighbor)
            mark src as "parented"
        
        candidates = new_candidates
        
        // Early termination check
        if no_new_candidates or iter >= min_iterations:
            break
    
    return top_M[0:k]  // return top-k results
```

### 8.3 Distance Computation Kernel (Team-Cooperative)

```
__global__ void distance_kernel(
    float* query,       // [dim]
    float* dataset,     // [N × dim]
    uint32_t* nodes,    // [num_nodes] — node indices to compute
    float* distances,   // [num_nodes] — output distances
    int dim
) {
    int node_idx = blockIdx.x;
    int tid = threadIdx.x;
    int team_size = blockDim.x;  // e.g., 32
    
    uint32_t node = nodes[node_idx];
    
    // Each team of threads cooperatively computes one distance
    float sum = 0.0f;
    for (int d = tid; d < dim; d += team_size) {
        float diff = query[d] - dataset[node * dim + d];
        sum += diff * diff;
    }
    
    // Warp-level reduction
    for (int offset = team_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    if (tid == 0) {
        distances[node_idx] = sum;
    }
}
```

### 8.4 Top-M Update (Parallel Merge)

```
function PARALLEL_TOP_M_UPDATE(top_M, candidates, M):
    // Combined size: M + p×d (e.g., 64 + 1×64 = 128)
    combined = allocate(M + p×d)
    
    // Copy both into combined buffer
    parallel_copy top_M → combined[0:M]
    parallel_copy candidates → combined[M:M+p×d]
    
    // Parallel bitonic sort (on shared memory)
    // For 128 elements: 7 stages of compare-and-swap
    for stage = 0 to log2(M + p×d) - 1:
        for step = 0 to stage:
            parallel_for pair in bitonic_pairs(stage, step):
                if combined[pair.a].dist > combined[pair.b].dist:
                    swap(combined[pair.a], combined[pair.b])
    
    // Keep top-M
    parallel_copy combined[0:M] → top_M
```

---

## 9. Performance Numbers

From the paper (DGX A100, A100 80GB GPU):

### Graph Construction Time

| Dataset | CAGRA | HNSW (CPU) | Speedup |
|---------|-------|-----------|---------|
| SIFT-1M | 1.2s | 13.4s | 11.2× |
| GIST-1M | 6.8s | 150.2s | 22.1× |
| GloVe-200 | 2.1s | 45.3s | 21.6× |

### Large-Batch Search Throughput (90-95% recall@10)

| Dataset | CAGRA | HNSW (CPU) | Speedup |
|---------|-------|-----------|---------|
| SIFT-1M | 330K QPS | 4.3K QPS | 77× |
| DEEP-1M | 180K QPS | 5.2K QPS | 34× |

### Single-Query Latency (95% recall@10)

| Dataset | CAGRA | HNSW (CPU) | Speedup |
|---------|-------|-----------|---------|
| SIFT-1M | 0.15ms | 0.52ms | 3.4× |
| DEEP-1M | 0.08ms | 4.2ms | 53× |

### Scaling to 100M Vectors

| Dataset | Build Time | Search QPS | Memory |
|---------|-----------|------------|--------|
| DEEP-100M | 32s | 85K QPS | ~40 GB |

---

## 10. References

1. **CAGRA Paper**: Ootomo et al., "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs," ICDE 2024. arXiv:2308.15136

2. **cuVS Library**: https://github.com/NVIDIA/cuvs — Apache 2.0 licensed, part of NVIDIA RAPIDS

3. **cuVS CAGRA Docs**: https://docs.rapids.ai/api/cuvs/stable/neighbors/cagra/

4. **NN-Descent**: Dong et al., "Fast K-NN Graph Construction by GPU Based NN-Descent," SIGMOD 2022. acm.org/doi/10.1145/3459637.3482344

5. **HNSW**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs," IEEE TPAMI 2018.

6. **NSG**: Fu et al., "Accelerating Approximate Nearest Neighbor Search on Large Datasets via Graph Supersampling," KDD 2019.

7. **Top-K on GPU**: Johnson et al., "Top-K Algorithms on GPU: A Comprehensive Study and New Methods," SIGMOD 2023. acm.org/doi/10.1145/3581784.3607062

---

*Last updated: 2026-07-18*
*Source: NVIDIA CAGRA paper (ICDE 2024) + cuVS documentation + cuVS source code analysis*
