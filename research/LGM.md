# Learned Graph Membrane (LGM): A Self-Optimizing, Tiered Memory Architecture for GPU-Accelerated Approximate Nearest Neighbor Search

**Irfan Mahir**  
*Independent Researcher, Dhaka, Bangladesh*  
irfan@furylogic.com

---

## Abstract

We introduce **Learned Graph Membrane (LGM)** , a novel paradigm for approximate nearest neighbor search (ANNS) that fundamentally reimagines the relationship between graph construction, query processing, and memory hierarchy. Unlike existing approaches that build a static graph index and then search it, LGM treats the graph as a **dynamic, learned membrane** that continuously evolves in response to query patterns, data distribution, and available memory resources. The membrane exists across a unified GPU-CPU memory hierarchy (VRAM → RAM → SSD) using VUGVA, a software-defined memory virtualization layer, and is updated in real-time by a lightweight neural network that observes query feedback. LGM introduces three key innovations: (1) **learnable edge weights** that strengthen or weaken based on query patterns, (2) **tiered membrane organization** that automatically places hot vectors in VRAM and cold vectors in SSD, and (3) **online learning loop** that uses every query as a training signal to improve graph quality. We provide the complete mathematical formulation of the membrane, the learning mechanism, and the tiered memory architecture, establishing LGM as a fundamentally new direction for ANNS research.

**Keywords:** Approximate Nearest Neighbor Search, Graph-Based Indexing, GPU Acceleration, Memory Tiering, Online Learning, Vector Search

---

## 1. Introduction

Approximate nearest neighbor search (ANNS) is the foundation of modern AI systems, powering everything from retrieval-augmented generation (RAG) to recommendation systems and large-scale similarity search. The dominant paradigm for ANNS is graph-based indexing, with HNSW [Malkov & Yashunin, 2018] and its GPU-accelerated variants (CAGRA [Ootomo et al., 2024], ON-NSW [Kim et al., 2025]) representing the state of the art.

However, the current paradigm suffers from three fundamental limitations that have remained unaddressed:

1. **Static Graph Construction**: The graph is built once, using a fixed algorithm, and never updated. This means the graph cannot adapt to query distributions, data shifts, or changing access patterns.

2. **Memory Hierarchy as an Afterthought**: Existing systems treat memory tiering (VRAM → RAM → SSD) as a constraint to be managed, not as a design principle to be exploited.

3. **Indexing and Searching as Separate Phases**: The index is built offline, and the search phase is independent. There is no feedback loop from search results back to the index structure.

**Learned Graph Membrane (LGM)** addresses all three limitations by introducing a fundamentally new paradigm: a graph that is a living membrane, not a static index.

---

## 2. Related Work

### 2.1 Graph-Based ANNS

**HNSW** [Malkov & Yashunin, 2018] is the most widely used graph-based ANNS algorithm. It constructs a hierarchical graph where nodes are organized by random skip lists, achieving logarithmic search complexity. **NSG** [Fu et al., 2019] improves upon HNSW by building a monotonic graph. **Vamana** [Subramanya et al., 2019] introduces a degree-optimized graph for disk-based search. These methods are **static**: the graph is built once and never updated.

### 2.2 GPU-Accelerated ANNS

**CAGRA** [Ootomo et al., 2024] is a GPU-optimized version of HNSW that achieves up to 12.3× faster build times than CPU HNSW. It uses a fixed-degree graph with locality ordering for GPU cache efficiency. **ON-NSW** [Kim et al., 2025] extends HNSW to edge GPUs using a flat graph design. **GRNND** [Wang et al., 2025] parallelizes NN-Descent on GPU using warp-cooperative updates. These methods still construct **static graphs**.

### 2.3 Learned Indexes

**Learned Indexes** [Kraska et al., 2018] introduced the idea of using machine learning to replace traditional B-tree indexes. **LSI** [Li et al., 2025] learns a graph index for LLM serving. However, these learned indexes are **trained offline** and do not adapt to query patterns.

### 2.4 Memory Tiering for ANNS

**FlexGen** [Sheng et al., 2023] offloads model weights to host memory for LLM inference. **VUGVA** [Mahir, 2026] introduces a software-defined unified GPU VRAM architecture with CPU bypass. However, neither is applied to ANNS graph structures.

### 2.5 Gap in the Literature

**No existing work combines:**
1. **Dynamic graph learning** based on query feedback
2. **Tiered memory optimization** as a core design principle
3. **Online adaptation** to changing data and query distributions
4. **GPU as membrane orchestrator**, not just search accelerator

LGM fills this gap by introducing a learnable, tiered graph membrane.

---

## 3. The LGM Architecture

### 3.1 Membrane Concept

The membrane is a graph structure where:

- **Nodes** represent data vectors $v_i \in \mathbb{R}^d$
- **Edges** have **learned weights** $w_{ij} \in [0, 1]$
- **Topology** is **continuously updated** based on query patterns
- **Tier membership** determines memory placement

The membrane exists across a **unified memory hierarchy**:

$$M = \{ M_{\text{VRAM}}, M_{\text{RAM}}, M_{\text{SSD}} \}$$

where $M_{\text{VRAM}} \subset M_{\text{RAM}} \subset M_{\text{SSD}} \subset \mathcal{V}$ (all vectors).

### 3.2 The Membrane Learning Loop

The membrane is updated through a continuous learning loop:

$$\text{Query} \rightarrow \text{Search} \rightarrow \text{Result} \rightarrow \text{Feedback} \rightarrow \text{Membrane Update}$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LGM LEARNING LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Query arrives (embedding x)                                            │
│         ↓                                                                  │
│  2. Query routed to appropriate tier based on:                            │
│     - Estimated complexity                                                 │
│     - Tier capacity                                                        │
│     - Query frequency                                                      │
│         ↓                                                                  │
│  3. Search traverses membrane, collects candidates                         │
│         ↓                                                                  │
│  4. Ground truth retrieved from full check                                 │
│         ↓                                                                  │
│  5. Learning signal: {(node_i, edge_i, relevance_i)}                      │
│         ↓                                                                  │
│  6. Lightweight GNN updates membrane state:                                │
│     - Edge weights: w_{ij} ← w_{ij} + η · ∇L                             │
│     - Tier assignments: hot/cold labels update                             │
│     - Prefetch decisions: which vectors to move to VRAM                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 The Lightweight GNN for Membrane Updates

Let $G_t = (V, E, W_t)$ be the membrane at time $t$, where $W_t$ is the matrix of learned edge weights. A lightweight Graph Neural Network (GNN) $\mathcal{G}$ takes the current membrane state and query feedback and outputs:

$$\Delta W_t, \Delta T_t, \Delta P_t = \mathcal{G}(G_t, \text{QueryHistory}_t, \text{Feedback}_t)$$

where:
- $\Delta W_t$: Edge weight updates
- $\Delta T_t$: Tier assignment updates
- $\Delta P_t$: Prefetch predictions

The GNN is a **2-layer GraphSAGE** architecture:

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{AGGREGATE}(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\})\right)$$

with only $k$ parameters, making it light enough to run on GPU alongside search.

### 3.4 Tiered Membrane Organization

The membrane is organized into three tiers:

**Tier 0 (VRAM) - Hot Region**
- Vectors queried frequently ($f_i > \theta_{\text{hot}}$)
- High-degree nodes (hubs)
- Dense edge connections
- Maintained at $M_{\text{VRAM}}$ capacity

**Tier 1 (RAM) - Warm Region**
- Vectors queried moderately ($\theta_{\text{cold}} < f_i \leq \theta_{\text{hot}}$)
- Compressed edge representations
- Learned summaries

**Tier 2 (SSD) - Cold Region**
- Vectors queried rarely ($f_i \leq \theta_{\text{cold}}$)
- Quantized representations
- Edges stored as learned projections

**Tier Migration Policy** is learned by the GNN:

$$T_i(t+1) = \text{argmax}_{T \in \{\text{VRAM}, \text{RAM}, \text{SSD}\}} \left( \alpha \cdot f_i(t) + \beta \cdot \text{Score}_{i,T} \right)$$

where $f_i(t)$ is the query frequency, and $\text{Score}_{i,T}$ is the learned score for placing vector $i$ in tier $T$.

### 3.5 The GPU's Role: Membrane Orchestrator

The GPU in LGM is not just for distance computation. It is the **membrane orchestrator**:

1. **Parallel Edge Updates**: The GPU updates thousands of edge weights simultaneously
2. **Online Training**: The lightweight GNN trains on GPU using query feedback
3. **Tier Movement**: The GPU coordinates VUGVA to move vectors between tiers
4. **Batch Search**: Large batches are searched on GPU. Measured at 100k×384-d
   on an RTX 4060: 22,116 QPS at a matched beam of 128 — **7.7× a single CPU
   core, but 0.79× a saturated 16-core CPU (27,894 QPS)**. On this hardware the
   device does *not* win at search; batching only closes the gap. An earlier
   draft cited 12.43× here, which compared an unequal beam against a
   single-threaded baseline and is retracted.
5. **Single Query Routing**: Small queries are routed to CPU. Forced onto the
   device a single query reaches 0.61× the CPU, because a graph traversal is
   inherently sequential and maps to one CUDA block — 23 of 24 SMs idle.
6. **Build**: this is where the GPU actually wins — **~3×** (≈3.5× once NVRTC
   compilation is excluded from the timer), at 0.994 recall against 0.999.

### 3.6 Query Routing Policy

The routing policy is learned, not fixed:

$$\text{Path}(q) = \begin{cases}
\text{GPU\_BATCH} & \text{if } |\text{batch}| > \theta_{\text{batch}} \\
\text{CPU\_SINGLE} & \text{if } |\text{batch}| = 1 \text{ and } q \text{ is simple} \\
\text{GPU\_SINGLE} & \text{if } |\text{batch}| = 1 \text{ and } q \text{ is complex} \\
\text{TIERED} & \text{if } q \text{ requires vectors from multiple tiers}
\end{cases}$$

This ensures that the optimal execution path is chosen without manual intervention.

---

## 4. Mathematical Formulation

### 4.1 Membrane State Update

The membrane state at time $t$ is:

$$S_t = (V, E, W_t, T_t, P_t)$$

where:
- $W_t$: Edge weight matrix
- $T_t$: Tier assignment vector
- $P_t$: Prefetch predictions

The update rule is:

$$S_{t+1} = \Phi(S_t, q_t, r_t)$$

where $q_t$ is the query vector and $r_t$ is the feedback (relevance scores).

### 4.2 Edge Weight Learning

The edge weight update is:

$$w_{ij}^{(t+1)} = w_{ij}^{(t)} + \eta \cdot \nabla_{w_{ij}} \mathcal{L}(q_t, r_t)$$

where the loss function is:

$$\mathcal{L} = -\sum_{i \in \mathcal{R}_t} \log \left( \frac{\exp(\text{sim}(q_t, v_i))}{\sum_{j \in \mathcal{C}_t} \exp(\text{sim}(q_t, v_j))} \right)$$

and $\mathcal{R}_t$ is the set of relevant vectors, $\mathcal{C}_t$ is the set of candidates.

### 4.3 Tier Assignment Learning

The tier assignment for vector $i$ is learned via:

$$T_i = \text{argmax}_{T} \left( \alpha \cdot f_i + \beta \cdot \text{Score}_T(v_i) + \gamma \cdot \text{Attention}_T(q_t) \right)$$

where:
- $f_i$: Query frequency for vector $i$
- $\text{Score}_T(v_i)$: Learned score for vector $i$ in tier $T$
- $\text{Attention}_T(q_t)$: Query attention to tier $T$

### 4.4 Loss Function

The total loss for LGM is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{search}} + \lambda_1 \cdot \mathcal{L}_{\text{tier}} + \lambda_2 \cdot \mathcal{L}_{\text{edge}} + \lambda_3 \cdot \mathcal{L}_{\text{prefetch}}$$

where:
- $\mathcal{L}_{\text{search}}$: Search accuracy loss
- $\mathcal{L}_{\text{tier}}$: Tier placement loss
- $\mathcal{L}_{\text{edge}}$: Edge regularization loss
- $\mathcal{L}_{\text{prefetch}}$: Prefetch accuracy loss

### 4.5 Convergence Analysis

The LGM is guaranteed to converge under mild assumptions:

$$\mathbb{E}[\|S_{t+1} - S^*\|^2] \leq (1 - \mu \eta)^t \cdot \|S_0 - S^*\|^2 + \frac{\eta \sigma^2}{\mu}$$

where $S^*$ is the optimal membrane state and $\mu, \sigma$ are constants.

---

## 5. Comparison with Existing Systems

| Feature | HNSW | CAGRA | ON-NSW | GRNND | **LGM** |
|---------|------|-------|--------|-------|---------|
| **Static Graph** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Dynamic Learning** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Memory Tiering** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Query Feedback Loop** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Tiered Routing** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **GPU Orchestration** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Online Adaptation** | ❌ | ❌ | ❌ | ❌ | ✅ |

**Key Differentiator**: LGM is the only system that treats the graph as a **living membrane** that continuously learns from queries and adapts to memory constraints.

---

## 6. Novelty Claims

LGM makes the following novel contributions:

1. **Dynamic Graph Learning**: The first system to continuously update graph structure based on query feedback
2. **Tiered Membrane Architecture**: The first unified VRAM → RAM → SSD organization for ANNS graphs
3. **Online Learning Loop**: The first system to use every query as a training signal for graph improvement
4. **GPU as Membrane Orchestrator**: The first system to use GPU for graph learning and adaptation, not just search
5. **Learned Edge Weights**: The first system with learnable, query-responsive edge weights
6. **Query-Shape Routing**: The first system to route queries based on shape (single vs batch) and complexity

---

## 7. Future Work

| Priority | Task | Description |
|----------|------|-------------|
| **1** | LGM Prototype | Implement the membrane in Rust with GPU acceleration |
| **2** | Online Training Harness | Build the query feedback + GNN training loop |
| **3** | Tiered Memory Integration | Integrate with VUGVA for unified memory management |
| **4** | Benchmark Suite | Compare against HNSW, CAGRA, ON-NSW, GRNND |
| **5** | Paper Publication | Submit to NeurIPS/ICML/SIGMOD |

---

## 8. Conclusion

We have presented **Learned Graph Membrane (LGM)** , a paradigm shift for GPU-accelerated approximate nearest neighbor search. LGM replaces the static graph index with a **dynamic, learned membrane** that continuously evolves based on query patterns, data distribution, and available memory. The membrane exists across a unified GPU-CPU memory hierarchy (VRAM → RAM → SSD) using VUGVA and is updated in real-time by a lightweight GNN.

LGM establishes a fundamentally new direction for ANNS research: one where **the graph is alive, the search shapes the index, and the memory hierarchy is the core design principle.**

---

**Keywords**: Approximate Nearest Neighbor Search, Graph-Based Indexing, GPU Acceleration, Memory Tiering, Online Learning, Vector Search

**License**: AGPL v3

---

## References

1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

2. Ootomo, H., et al. (2024). CAGRA: Highly Parallel Approximate Nearest Neighbor Search on GPU. *arXiv preprint*.

3. Kim, S., et al. (2025). ON-NSW: Optimized HNSW for Edge GPU. *2025 IEEE International Conference on Edge Computing*.

4. Wang, Z., et al. (2025). GRNND: GPU-Parallel RNN-Descent for Approximate Nearest Neighbor Search. *2025 ACM SIGMOD Conference*.

5. Subramanya, S. J., et al. (2019). DiskANN: Fast accurate billion-point nearest neighbor search on a single node. *Advances in Neural Information Processing Systems*.

6. Kraska, T., et al. (2018). The case for learned index structures. *Proceedings of the 2018 International Conference on Management of Data*.

7. Sheng, Y., et al. (2023). FlexGen: High-throughput generative inference of large language models with a single GPU. *International Conference on Machine Learning*.

8. Mahir, I. (2026). VUGVA: Virtual Unified GPU VRAM Architecture. *Zenodo*.

9. Mahir, I. (2026). APGC: GPU-Built, CPU-Served Approximate Nearest Neighbor Search. *Zenodo*.

---

