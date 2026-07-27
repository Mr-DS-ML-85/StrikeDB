# F3: GPU-Native HNSW Graph Construction Research

> Focus: Can HNSW's hierarchical structure run natively on GPU? Who has tried?
> Researched: 2026-07-18

---

## 1. GANNS: GPU-Accelerated Proximity Graph ANN Search & Construction

| Field | Detail |
|-------|--------|
| **Title** | GPU-accelerated proximity graph approximate nearest Neighbor search and construction |
| **Authors** | Y Yu, D Wen, Y Zhang, L Qin, W Zhang, X Lin (University of Technology Sydney) |
| **Year / Venue** | 2022, IEEE ICDE 2022 |
| **Citations** | 60 |
| **URL** | https://ieeexplore.ieee.org/document/ (IEEE ICDE) |

**What they did:**
- GPU-accelerated algorithm for both ANN search and proximity graph construction
- Novel GPU-friendly search framework exploiting massive parallelism at key search steps
- GPU-accelerated proximity graph construction algorithms with efficient parallel implementations
- Addressed bottleneck of prior GPU work (which only accelerated distance computation) by tackling expensive data structure operations on GPU

**What's MISSING:**
- Graph quality may lag CPU-based HNSW (Jiang & Chen, ICDE 2025 found GPU-constructed graph can have lower quality)
- GPU performance not always superior — SAGA (MICRO 2025) observed GPU performance sometimes falls below CPU baselines for construction
- No HNSW-specific optimizations — focused on general proximity graphs, not HNSW's hierarchical structure or dynamic insertions
- Memory-bound bottlenecks not addressed (identified by CMANNS, 2026)
- No open-source implementation found

**Confidence:** HIGH — 60 citations, foundational GPU graph ANN paper, successor works (SAGA, CMANNS, GRAB-ANNS) explicitly build upon it.

---

## 2. GGNN: Graph-Based GPU Nearest Neighbor Search

| Field | Detail |
|-------|--------|
| **Title** | Graph-based GPU Nearest Neighbor Search |
| **Authors** | Groh, Ruppert, Wieschollek et al. |
| **Year / Venue** | 2022, IEEE TPAMI |
| **Citations** | 138 |
| **URL** | https://ieeexplore.ieee.org (search title) |

**What they did:**
- Built a GPU-native graph where each node has exactly **k outgoing edges**
- Enables regular, parallelized greedy traversal with constant workload per step
- Focuses on the traversal engine, not graph construction
- Canonical reference for GPU graph traversal

**What's MISSING:**
- Graph construction is separate — relies on CPU-side HNSW or similar
- No dynamic insertions support
- No multi-GPU scaling
- No discussion of hierarchical layers on GPU

**Confidence:** HIGH — 138 citations, core reference for GPU graph traversal. The k-outgoing-edge regular graph design is the key insight enabling constant-workload parallel steps.

---

## 3. CMANNS: GPU-Accelerated Graph Index Construction via Compute-Memory Disaggregation

| Field | Detail |
|-------|--------|
| **Title** | CMANNS: GPU-Accelerated Graph Index Construction for ANNS via Compute-Memory Disaggregation |
| **Authors** | C Huan, R Yao, S Ma, R Gu, Z Yang, L Chen et al. |
| **Year / Venue** | 2026, ACM Proceedings |
| **URL** | https://dl.acm.org/doi/abs/10.1145/3802027 |

**What they did:**
- Accelerates graph index construction for ANNS on GPUs using compute-memory disaggregation architecture
- Handles large-scale vector datasets with 100M and 200M vectors (up to 95 GB)
- Compares to FAISS at matched recall and identical search budgets
- Addresses memory-bound bottlenecks identified in GANNS

**What's MISSING:**
- No arXiv preprint found — only ACM DL publication (limited public details)
- No comparison with other GPU-based ANNS systems mentioned
- Algorithm details and code availability unknown

**Confidence:** HIGH — confirmed via Google Scholar with matching title, authors, venue. 12 citation results indicate active research area.

---

## 4. Full GPU HNSW Implementations

### 4a. cuHNNSW (The Only Full GPU HNSW Attempt)

| Field | Detail |
|-------|--------|
| **Title** | cuHNNSW |
| **Authors** | js1010 (GitHub) |
| **Year** | 2019-2021 |
| **URL** | https://github.com/js1010/cuhnsw |

**What they did:**
- CUDA implementation of full HNSW (build + search)
- Build: **8-9x faster** than hnswlib
- Search: **3-4x faster** than hnswlib
- hnswlib-compatible format

**What's MISSING:**
- Single GPU only
- **Abandoned** (last commit 2021)
- Multi-device noted as "very hard"
- No paper published
- Community concluded the approach doesn't scale

**Confidence:** HIGH — only known attempt at full HNSW on GPU.

### 4b. SVFusion (CPU-GPU Co-Processing)

| Field | Detail |
|-------|--------|
| **Title** | SVFusion |
| **Authors** | Peng et al. |
| **Year** | 2026 |

**What they did:**
- CPU-GPU co-processing for hierarchical vector search
- Real-time coordination via CUDA

**What's MISSING:**
- CPU-dependent — not purely GPU
- Hierarchical indexing still split across CPU/GPU

**Confidence:** MEDIUM — limited public details.

### 4c. ON-NSW (GPU for Edge Devices)

| Field | Detail |
|-------|--------|
| **Title** | ON-NSW |
| **Authors** | Park et al. |
| **Year** | 2025 |

**What they did:**
- GPU-optimized HNSW for edge devices
- Entire bottom-layer search on GPU

**What's MISSING:**
- Top-layer traversal still on CPU
- Edge-focused, not general-purpose

**Confidence:** MEDIUM — edge-specific, not datacenter-scale.

---

## 5. Challenges of HNSW's Hierarchical Structure on GPU

The community has largely **pivoted away from full GPU HNSW** due to fundamental parallelization challenges:

### Challenge 1: Sequential Layer Traversal (THE BIGGEST BLOCKER)
Search must go **top-down through layers**; each layer transition depends on the previous result. This is inherently serial and anti-GPU. GPU thrives on independent parallel work — HNSW search is the opposite.

### Challenge 2: Strong Inter-Node Dependencies
Graph construction inserts nodes that modify neighbor lists, requiring synchronization that conflicts with GPU parallelism. Insertions at one node affect its neighbors, creating cascade effects.

### Challenge 3: Irregular Memory Access
Pointer-chasing graph traversal causes **warp divergence** and **poor memory coalescing**. GPU memory system expects regular, coalesced access patterns — graph traversal gives you random pointer following.

### Challenge 4: Variable Workload Per Layer
Upper layers are tiny (few nodes), lower layers are massive. Load balancing across GPU SMs is poor — most threads idle while a few do all the work on upper layers.

### Challenge 5: Multi-Device Graph Sharing
The graph structure must be shared across threads. cuHNNSW author notes this as "very hard" — graph mutation during construction creates coherence problems.

### Community Response: Reject HNSW's Hierarchy Entirely
- **CAGRA (NVIDIA, 2024)** deliberately replaced HNSW's hierarchy with a flat proximity graph optimized for GPU parallelism. 2.2-27x faster than HNSW build; 33-77x faster batch search. 139 citations.
- **BANG (2024-2025)** found CPU/GPU coordination during iterative graph traversals is essential at billion scale.
- **Consensus**: HNSW's hierarchical design is fundamentally mismatched to GPU architecture. CPU-GPU co-processing (SVFusion, PathWeaver) is the practical compromise.

---

## 6. PathWeaver: Multi-GPU Graph ANNS

| Field | Detail |
|-------|--------|
| **Title** | PathWeaver: A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search |
| **Authors** | Sukjin Kim, Seongyeon Park, Si Ung Noh, Junguk Hong, Taehee Kwon, Hunseong Lim, Jinho Lee |
| **Year / Venue** | 2025, USENIX ATC 2025 |
| **Citations** | 9 (in <1 year) |
| **URL** | https://arxiv.org/abs/2507.17094 |

**What they did:**
- Multi-GPU framework for graph-based ANNS
- Three key techniques:
  1. **Pipelining-based path extension** — GPU-aware pipelining leveraging GPU-to-GPU communication to eliminate redundant search iterations across GPUs
  2. **Ghost staging** — uses representative dataset to identify optimal query starting points, shrinking search space for hard queries
  3. **Direction-guided selection** — filters irrelevant graph points early, cutting unnecessary memory accesses and distance computations
- **3.24x geomean speedup**, up to **5.30x at 95% recall** over state-of-the-art multi-GPU ANNS baselines

**What's MISSING:**
- No public code or dataset release
- Evaluation scope unclear (specific GPU types, interconnect topology, dataset sizes not detailed in abstract)
- Focus is throughput-oriented; query latency / tail-latency behavior not discussed
- No comparison to emerging GPU-native indexes like CAGRA or IVF-RaBitQ on GPU
- Single-architecture evaluation likely; generalization to heterogeneous GPU clusters unaddressed

**Confidence:** HIGH — USENIX ATC is a top systems venue, 9 citations in under a year, confirmed on both Google Scholar and arXiv.

---

## Summary: State of GPU HNSW

| System | Year | Full HNSW on GPU? | Status | Key Insight |
|--------|------|-------------------|--------|-------------|
| **GGNN** | 2022 | No (traversal only) | Active, 138 citations | k-outgoing-edge regular graph enables parallel traversal |
| **GANNS** | 2022 | No (general proximity graph) | Active, 60 citations | Foundational GPU graph ANN, but graph quality lags CPU |
| **cuHNNSW** | 2019-2021 | **YES** (all levels) | **Abandoned** | 8-9x build speedup but community concluded it doesn't scale |
| **CAGRA** | 2024 | No (replaced hierarchy with flat graph) | NVIDIA state-of-the-art | Deliberately rejected HNSW's hierarchy for GPU parallelism |
| **BANG** | 2024-2025 | No (CPU/GPU coordination) | Active, 32 citations | CPU/GPU coordination essential at billion scale |
| **PathWeaver** | 2025 | No (multi-GPU graph search) | USENIX ATC 2025 | 3.24x speedup via pipelining + ghost staging |
| **CMANNS** | 2026 | No (compute-memory disaggregation) | ACM 2026 | Handles 200M vectors via disaggregated architecture |
| **SVFusion** | 2026 | Partial (CPU-GPU split) | Preprint | Real-time CPU-GPU coordination |
| **ON-NSW** | 2025 | Partial (bottom layer GPU only) | Preprint | Edge-focused, top layers still CPU |

### Bottom Line for DB-Strike

**Nobody has successfully built full GPU HNSW at scale.** cuHNNSW (2019) was the only attempt — it achieved 8-9x build speedup but was abandoned because the hierarchical structure fundamentally conflicts with GPU parallelism. The industry consensus (led by NVIDIA's CAGRA) is to **replace HNSW's hierarchy entirely** with flat proximity graphs optimized for GPU. For DB-Strike, this means:

1. **GPU HNSW build is not a viable path** — the community has tried and abandoned it
2. **Flat GPU graphs (CAGRA-style) are the proven approach** — but they require different search characteristics
3. **CPU-GPU hybrid (PathWeaver-style) is the practical compromise** — if you must have hierarchical structure
4. **The real opportunity** may be in novel graph structures designed for GPU from scratch (not retrofitting HNSW)
