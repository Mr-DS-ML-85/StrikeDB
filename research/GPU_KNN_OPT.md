# GPU kNN Graph Optimization for 1M Vectors

## Problem Statement

**Current approach**: Brute-force kNN → O(N²) complexity  
**Scale**: 1M vectors, likely 128-768 dimensions  
**Current performance**: ~500 seconds for brute-force  
**Goal**: Dramatically reduce graph construction time via GPU-accelerated NN-Descent

---

## Key Findings from Research

### 1. NN-Descent Algorithm (Original CPU)

NN-Descent is an iterative graph construction algorithm with complexity O(N * k * iters) where:
- N = number of vectors (1M)
- k = neighbors per node (typically 32-128)
- iters = iterations to convergence (typically 10-30)

**Key Insight**: Instead of comparing all pairs O(N²), NN-Descent maintains a candidate neighbor list per node and iteratively refines it by sampling neighbors-of-neighbors.

**Core Operations Per Iteration**:
1. For each node, sample candidates from current neighbor lists
2. Compute distances to new candidates
3. Merge candidates with existing neighbor list
4. Keep top-k neighbors

### 2. GRNND - GPU-Parallel Relative NN-Descent (arXiv:2510.02774)

**Publication**: October 2025, Nanjing University / Institute of Science Tokyo

**Key Contributions**:
- First GPU-parallel implementation of Relative NN-Descent (RNN-Descent)
- **2.4x to 51.7x speedup** over existing GPU methods
- **17.8x to 49.8x speedup** over CPU methods

**GPU Optimizations**:
1. **Disordered Neighbor Propagation**: Mitigates synchronized update traps, prevents premature convergence
2. **Warp-Level Cooperative Operations**: Efficient memory access patterns
3. **Double-Buffered Neighbor Pool**: Fixed capacity, eliminates contention, enables parallelized updates

### 3. Tagore/GNN-Descent (arXiv:2508.08744)

**Publication**: SIGMOD 2026 (Accepted)

**Key Contributions**:
- **GNN-Descent**: GPU-specific k-NN graph initialization algorithm
- Two-phase descent procedure for similarity comparison
- Highly parallelized neighbor updates
- **1.32x to 112.79x speedup** over existing methods

**Novel Approach**:
- Two-phase descent: coarse-grained + fine-grained neighbor search
- Universal computing procedure (CFS) for complex dependencies
- Asynchronous GPU-CPU-disk framework for out-of-GPU-memory datasets

### 4. Wang et al. GPU NN-Descent (arXiv:2103.15386)

**Publication**: March 2021

**Key Contributions**:
- First major GPU redesign of NN-Descent
- **100-250x faster** than single-thread NN-Descent
- **2.5-5x faster** than existing GPU approaches

**GPU-Specific Optimizations**:
1. **Reduced Memory Accesses**: Critical for GPU performance
2. **Full Parallelism Exploitation**: All operations parallelized
3. **k-NN Graph Merge**: Enables out-of-GPU-memory datasets via graph merging

---

## cuVS (RAPIDS) Implementation

### Architecture

cuVS provides production-grade NN-Descent implementation as part of the RAPIDS ecosystem:

```
cuVS Stack:
├── cuVS (Vector Search Library)
│   ├── NN-Descent (k-NN graph construction)
│   ├── CAGRA (Graph-based ANN search)
│   ├── IVF-Flat / IVF-PQ (Inverted file indexes)
│   └── HNSW (Hierarchical Navigable Small World)
├── RAFT (ML Primitives)
│   ├── Distance computations
│   ├── Selection (top-k)
│   └── Matrix operations
└── CUDA / cuBLAS / cuDNN
```

### API Usage (Python)

```python
from cuvs.neighbors import nn_descent

# Build k-NN graph
index_params = nn_descent.IndexParams(
    n_neighbors=64,        # k neighbors per node
    intermediate_graph_degree=128,  # candidates per iteration
    graph_degree=64,       # final graph degree
    max_iterations=20,     # convergence iterations
    termination_threshold=0.001
)

# dataset: shape (N, D) on GPU
graph = nn_descent.build(index_params, dataset)

# Returns: (edges, distances) arrays
# edges: shape (N, k) - neighbor indices
# distances: shape (N, k) - neighbor distances
```

### API Usage (C++)

```cpp
#include <cuvs/neighbors/nn_descent.hpp>

using namespace cuvs::neighbors;

raft::handle_t handle;
auto dataset_view = raft::make_device_matrix_view<float, int64_t>(
    dataset_ptr, n_rows, n_dims);

nn_descent::index_params params;
params.n_neighbors = 64;
params.intermediate_graph_degree = 128;
params.max_iterations = 20;

auto graph = nn_descent::build(handle, params, dataset_view);
```

---

## GPU NN-Descent Pseudocode

### Phase 0: Initialization

```
function GPU_NN_DESCENT(dataset[N][D], k=64, max_iters=20):
    // dataset: N vectors of dimension D (on GPU)
    // k: target neighbors per node
    
    // Allocate per-node structures
    neighbors[N][k]          // current k-NN for each node (index + distance)
    candidates[N][2k]        // candidate pool (2x for buffer)
    nn_new[N]                // flag: has new neighbors?
    nn_old[N]                // flag: had new neighbors last iter
    
    // Initialize with random neighbors
    parallel for i in 0..N:
        neighbors[i] = random_sample(all_indices, k, exclude=i)
        candidates[i] = neighbors[i]
        nn_new[i] = True
    
    return {neighbors, candidates, nn_new}
```

### Phase 1: Distance Computation (GPU-Parallel)

```
function COMPUTE_DISTANCES_kernel(dataset, query_idx, cand_idx, D):
    // Each thread block handles one (query, candidate) pair
    // Uses shared memory for vector tiles
    
    tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid >= N * 2k: return
    
    i = tid / (2k)      // query node index
    j = tid % (2k)      // candidate index within pool
    
    // Cooperative vector loading
    __shared__ float tile_q[TILE_SIZE][D]
    __shared__ float tile_c[TILE_SIZE][D]
    
    // Load query and candidate vectors in tiles
    for tile in range(0, D, TILE_SIZE):
        if threadIdx.x < TILE_SIZE:
            tile_q[threadIdx.x] = dataset[i][tile + threadIdx.x]
            tile_c[threadIdx.x] = dataset[candidates[i][j]][tile + threadIdx.x]
        __syncthreads()
        
        // Compute partial distance in registers
        dist_tile = 0
        for d in range(tile, min(tile+TILE_SIZE, D)):
            diff = tile_q[threadIdx.x][d] - tile_c[threadIdx.x][d]
            dist_tile += diff * diff
    
    // Warp-level reduction (shuffle)
    for offset in [16, 8, 4, 2, 1]:
        dist_tile += __shfl_down(dist_tile, offset)
    
    if threadIdx.x % 32 == 0:
        atomicAdd(&partial_dist[i][j], dist_tile)
```

### Phase 2: Neighbor Sampling (GPU-Parallel)

```
function SAMPLE_NEIGHBORS_kernel(neighbors, candidates, nn_new, N, k):
    // Each thread block handles one node
    i = blockIdx.x
    tid = threadIdx.x
    
    __shared__ bool has_new
    has_new = False
    __syncthreads()
    
    if not nn_new[i]: return
    
    // Each thread in warp processes one neighbor
    for j in range(tid, 2k, blockDim.x):
        cand = candidates[i][j]
        
        // Check if candidate is already in neighbor list
        already_present = False
        for n in range(k):
            if neighbors[i][n].idx == cand:
                already_present = True
                break
        
        if not already_present:
            // Insert into sorted neighbor list (maintain k-NN)
            INSERT_SORTED(neighbors[i], cand, dist[i][j], k)
            has_new = True
    
    __syncthreads()
    nn_new[i] = has_new
```

### Phase 3: Warp-Level Merge

```
function MERGE_NEIGHBORS_warp(neighbors_i, candidates_i, k):
    // Each warp handles one node
    // Uses warp shuffle for efficient merge of sorted lists
    
    lane = threadIdx.x % 32
    
    // Cooperative merge of two sorted lists
    // neighbors_i: current k-NN (sorted by distance)
    // candidates_i: candidate pool (sorted by distance)
    
    local_top_k = []
    i = 0  // pointer to neighbors_i
    j = 0  // pointer to candidates_i
    
    while len(local_top_k) < k:
        if lane == 0:
            // Thread 0 selects next best
            if i < k and (j >= 2k or dist_neighbors[i] < dist_candidates[j]):
                choice = ('neighbor', i)
                i += 1
            else:
                choice = ('candidate', j)
                j += 1
        
        // Broadcast choice to all threads
        choice = __shfl(choice, 0)
        
        // Add to local top-k
        local_top_k.append(choice)
    
    // Write back to global memory
    if lane < k:
        neighbors_i[lane] = local_top_k[lane]
```

### Phase 4: Main Iteration Loop

```
function GPU_NN_DESCENT_MAIN(dataset, k=64, max_iters=20):
    N, D = dataset.shape
    
    // Phase 0: Initialize
    neighbors, candidates, nn_new = INITIALIZE(dataset, k)
    
    // Main loop
    for iteration in 0..max_iters:
        // Check convergence
        active_nodes = GPU_SUM(nn_new)
        if active_nodes / N < 0.001:  // 0.1% threshold
            break
        
        // Copy new flags to old
        nn_old = nn_new.copy()
        nn_new.fill(False)
        
        // Phase 1: Compute distances (parallel over all edges)
        // Launch: N * 2k threads
        COMPUTE_DISTANCES_kernel<<<N*2k / 256, 256>>>(
            dataset, neighbors, candidates, D
        )
        
        // Phase 2: Sample and merge neighbors (parallel over nodes)
        // Launch: N blocks, 256 threads each
        SAMPLE_NEIGHBORS_kernel<<<N, 256>>>(
            neighbors, candidates, nn_old, nn_new, k
        )
        
        // Phase 3: Update candidate lists
        // For each node, add new neighbors of neighbors to candidates
        UPDATE_CANDIDATES_kernel<<<N, 256>>>(
            neighbors, candidates, nn_new, k
        )
    
    return neighbors  // Final k-NN graph
```

### Phase 5: Double-Buffered Pool (GRNND Optimization)

```
function DOUBLE_BUFFER_POOL(candidates, neighbors, k):
    // From GRNND paper: eliminate contention with double buffering
    
    // Buffer 0: active candidates (being processed)
    // Buffer 1: staging area (accumulating new candidates)
    
    buffer_idx = 0
    
    for iteration in 0..max_iters:
        // Swap buffers
        active = candidates[buffer_idx]
        staging = candidates[1 - buffer_idx]
        
        // Process active buffer (no contention with staging)
        PARALLEL_MERGE(neighbors, active, k)
        
        // Stage new candidates from neighbor-of-neighbor sampling
        SAMPLE_NEIGHBORS_OF_NEIGHBORS<<<N, 256>>>(
            neighbors, staging, k
        )
        
        // Atomic swap at end of iteration
        buffer_idx = 1 - buffer_idx
    
    return neighbors
```

---

## Performance Comparison

| Method | Complexity | 1M Vectors (Est.) | Speedup vs Brute-Force |
|--------|-----------|-------------------|------------------------|
| Brute-Force | O(N² * D) | ~500s | 1x (baseline) |
| CPU NN-Descent | O(N * k * iters * D) | ~20-30s | 15-25x |
| GPU NN-Descent (Wang) | O(N * k * iters * D / P) | ~2-4s | 125-250x |
| GPU NN-Descent (GRNND) | O(N * k * iters * D / P) | ~0.5-1s | 500-1000x |
| cuVS NN-Descent | Optimized GPU kernel | ~1-3s | 150-500x |

Where:
- P = parallelism factor (GPU cores / warp size)
- k = 64 (typical)
- iters = 15-20 (typical convergence)
- D = dimension (128-768)

---

## Implementation Recommendations

### 1. Use cuVS for Production

```python
# Recommended approach using cuVS
from cuvs.neighbors import nn_descent

index_params = nn_descent.IndexParams(
    n_neighbors=64,
    intermediate_graph_degree=128,
    graph_degree=64,
    max_iterations=20,
    termination_threshold=0.001
)

graph = nn_descent.build(index_params, dataset_on_gpu)
```

### 2. Memory Layout Optimization

```
// Structure of Arrays (SoA) for GPU coalescing
struct NodeNeighbors {
    int32_t indices[K];      // Neighbor indices
    float distances[K];       // Neighbor distances
    uint8_t flags;            // Update flags
} __align__(32);

// Align to 32 bytes for warp access
```

### 3. Tile-Based Distance Computation

```
// Process vectors in tiles for shared memory efficiency
#define TILE_SIZE 32  // Match warp size

for (int tile = 0; tile < D; tile += TILE_SIZE) {
    __shared__ float q_tile[TILE_SIZE];
    __shared__ float c_tile[TILE_SIZE];
    
    // Load tiles cooperatively
    q_tile[threadIdx.x] = dataset[q][tile + threadIdx.x];
    c_tile[threadIdx.x] = dataset[c][tile + threadIdx.x];
    __syncthreads();
    
    // Compute partial distance
    for (int d = 0; d < TILE_SIZE; d++) {
        float diff = q_tile[d] - c_tile[d];
        partial_dist += diff * diff;
    }
}
```

### 4. Convergence Detection

```
// Early stopping with GPU atomic counter
__device__ int active_count = 0;

// Kernel: atomicAdd(&active_count, 1) when node has updates
// Host: if active_count / N < threshold, stop iterations
```

---

## Integration with CAGRA

For ANN search (not just graph construction), use CAGRA which builds on NN-Descent:

```python
from cuvs.neighbors import cagra

# Build CAGRA index (includes graph construction)
index_params = cagra.IndexParams(
    metric="euclidean",
    graph_degree=64,
    intermediate_graph_degree=128,
    nn_descent_niter=20
)

index = cagra.build(index_params, dataset_on_gpu)

# Search
search_params = cagra.SearchParams(
    itopk=64,
    search_width=2,
    num_threads=1
)

distances, neighbors = cagra.search(search_params, index, queries, k=10)
```

---

## References

1. **GRNND** (arXiv:2510.02774) - GPU-Parallel Relative NN-Descent, 2025
2. **Tagore/GNN-Descent** (arXiv:2508.08744) - Scalable GPU Graph Indexing, SIGMOD 2026
3. **Wang et al.** (arXiv:2103.15386) - Large-Scale GPU k-NN Graph Construction, 2021
4. **cuVS Documentation** - https://docs.rapids.ai/api/cuvs/stable/
5. **CAGRA Paper** - Highly Parallel Graph Construction and ANN Search

---

## Quick Start

```bash
# Install cuVS
conda install -c rapidsai -c conda-forge -c nvidia cuvs pylibraft

# Or with pip
pip install cuvs-cu12  # For CUDA 12
```

```python
import numpy as np
from cuvs.neighbors import nn_descent

# Generate random dataset (replace with your data)
N, D = 1_000_000, 128
dataset = np.random.randn(N, D).astype(np.float32)

# Move to GPU
import cupy as cp
dataset_gpu = cp.asarray(dataset)

# Build k-NN graph
index_params = nn_descent.IndexParams(
    n_neighbors=64,
    intermediate_graph_degree=128,
    max_iterations=20
)

graph = nn_descent.build(index_params, dataset_gpu)

# graph.edges: shape (N, 64) - neighbor indices
# graph.distances: shape (N, 64) - neighbor distances
print(f"Graph built: {graph.edges.shape}")
```

---

*Research compiled: July 2026*
*Sources: arXiv, cuVS/RAPIDS documentation, NVIDIA developer resources*
