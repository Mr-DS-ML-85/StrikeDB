# Early Development — Open Work, Gaps, and Hard-Won Findings

**Last updated:** 2026-07-27
**Purpose:** Survive context loss. Every entry carries the file, the line, the
evidence, and the reason — so work can resume cold without re-deriving anything.

**Hardware these measurements come from:** AMD Ryzen 7 7700 (Zen 4, 16 threads,
AVX-512 VNNI), 32 GB DDR5, NVIDIA RTX 4060 (8 GB VRAM, 24 SMs, compute 8.9),
NVMe Gen4. Datasets in `/home/irfan/datasets/real_{384,768}_{100k,1M}.fbin`.

---

## 0. Read this first — the three things that mislead

1. **The GPU was never running.** Until `6e5c1b8`, `gpu_available()` only read a
   flag set by `gpu_init()`, and nothing called `gpu_init()` outside the
   `GPU.LOAD` / `GPU.MODE` RESP handlers. Every benchmark therefore measured the
   CPU path while appearing to exercise the GPU. Any GPU number predating that
   commit is CPU-path data. Fixed: lazy init in `gpu_available()`, gated on mode.

2. **Ingest, not search, is the competitive gap.** Search is already strong
   (0.985 recall @ 416 µs p99, 25k QPS on 100k×384d). Build is the weak axis and
   the one Qdrant/Milvus compete on.

3. **The distance kernel is not the bottleneck.** Forcing the scalar int8 dot
   instead of AVX-512 VNNI (~8× slower in isolation) changes end-to-end search by
   **~0%** and ingest by **6%**. `DBSTRIKE_DOT=scalar|avx2|vnni` exists so anyone
   can re-verify. Do not spend time on SIMD here; the "Attack 4 / 2× SIMD"
   premise in `research/beat-qdrant/` is real for Qdrant and false for us.

---

## 1. Blocking — item 1, the biggest single win

### 1.1 Route bulk `VADDBATCH` ingest to the GPU builder
**Status:** attempted, reverted, root cause found.
**Value:** wire ingest 5,812 vec/s → ~42,000 vec/s (the fast path is proven:
`build_parallel_tiered` measures 9.29× serial at 100k×384d, recall 0.994).

**Why the obvious fix fails.** Delegating `build_parallel_ids`
(`crates/views/src/vector.rs:4124`) to `build_parallel_tiered` and remapping
`node.id` through `ids` *looks* exact — that builder does label nodes with the
original row index — but it fails
`parallel_build_labels_each_vector_with_its_own_id` with **599/600 wrong**.

The id remap is correct. The **input contract** is not:

- `build_parallel_tiered`'s GPU path quantizes with a bare
  `(data[row * dim + d] * 127.0) as i8` — it requires **L2-normalized** input
  (its own doc says "row-major L2-normalized").
- `build_parallel_ids` is called from `insert_many_parallel_rebuild`
  (`vector.rs:~4784`) with **raw client vectors** straight off `VADDBATCH`.
- Un-normalized coordinates saturate the `i8` cast → the graph is built on
  garbage. Not mislabelled: genuinely wrong neighbours.

**Survey done — the diagnosis is worse than "normalize before routing".**
`l2_normalize` is called *inside* `Hnsw::insert_attr` (`vector.rs:1988`,
`:2021`), so every CPU insert path normalizes internally and **all**
`build_parallel*` callers legitimately pass raw data. The GPU block is the only
place that quantizes without normalizing first.

That makes this a **latent bug in the GPU build path itself**, not merely a
routing mismatch:

- It works on `real_*.fbin` only because sentence-transformer embeddings happen
  to arrive L2-normalized already. The 9.29× / recall-0.994 result is therefore
  correct *for that data* and would silently degrade on any un-normalized corpus.
- Two sites need it, both in `build_parallel_tiered`'s GPU block:
  - `let i8_all = ... (data[row * dim + d] * 127.0) as i8` (~`:3786`)
  - `for d in 0..dim { h.all_i8.push((data[base + d] * 127.0) as i8) }` (~`:3889`)
- `h.all_f32` (~`:3890`) must store the **normalized** vector too, since the
  exact-f32 rerank dots it against a normalized query — check what the CPU path
  puts in `all_f32` and match it exactly, or rerank scores will disagree with
  traversal scores.

**To finish:**
1. Normalize per row inside the GPU block (all three sites above), matching what
   `insert_attr` does. Do **not** normalize at the call site — that would leave
   the latent bug in place for every other GPU-build caller.
2. Re-run `parallel_build_labels_each_vector_with_its_own_id` under
   `DBSTRIKE_GPU=turbo`; must pass at 1, 4, **and** 8 shards (1 shard skips the
   shuffle, so it passes even when broken — 4/8 are the meaningful cases).
3. Then measure the wire ingest number.

A note explaining all this is already in the source at the call site.

---

## 2. GPU / APGC

### 2.1 `#77` — `CorpusTier` → `GpuIndex`: **done**, but search does not use it
`TieredPool::write_page` → `CorpusTier::upload` → `gpu_build_index` serves the
Hybrid corpus from the tier. Confirmed live by the log line
`[VUGVA] 36 MB corpus served via TieredPool (T0/T1/T2)`. `GpuIndex` owns the
tier (it frees its pages on drop, so `d_vectors` would otherwise dangle) and
`GpuIndex::free` skips the corpus pointer when the tier owns it.

**The remaining gap is on the search side, not the memory side.**
`search_ef:5003` branches to the device only under `Turbo`:

```rust
if mode == gpu::ComputeMode::Turbo { ... gpu_idx ... }
```

So Hybrid uploads its corpus through VUGVA and then searches on **CPU**. Its
QPS column measures graph quality, not tiering. Either extend that branch to
Hybrid, or accept that Hybrid means "GPU build + CPU search" and say so.

Also still true: `gpu_should_use_gpu:459` keeps the binary fits-in-VRAM check.

### 2.2 `#67` — fuse f32 rerank into `apgc_search`
In progress before this session. Turbo currently emits the int8 ranking and the
host reranks unless `d_vec_f32` is populated.

### 2.3 `#73` — prefetch: **done at the pool level**; ring still advisory
`TieredPool::prefetch(name, gpu_idx)` issues the DRAM→VRAM copy on the prefetch
stream and records an event; `access` claims it via `claim_prefetch` and only
waits on the event. Measured (8 × 32 MiB, RTX 4060):

```
issue 7.47 ms · claim 12.69 ms   vs   cold 40.81 ms   → 3.2× on the claim
```

Covered by `paper_prefetch_overlaps_transport_with_compute`, which asserts both
halves — that the timing improves *and* that each page carries its own bytes
(distinct per-page contents, so a prefetch returning the wrong page's pointer
fails rather than passing).

Still open here:
- `prefetch.rs::prefetch_ahead` remains the old layer-schedule API whose `Dram`
  and `Ssd` arms are empty (`// here we record the intent`). Only its
  VRAM→VRAM peer copy is real. The pool-level `prefetch` above supersedes it for
  every caller that has a `TieredPool`; `prefetch_ahead` should either be
  rewritten on top of it or removed.
- Cold (T2) pages are **not** prefetched — a file read would have to happen
  before any device copy, and blocking there defeats the purpose. They promote
  synchronously.
- `dma.rs`'s descriptor ring is still bookkeeping: real transfers go through
  `cuMemcpy*Async` on the stream pool, not the ring.

### 2.4 Remaining CPU round-trips in the build
`#78` fixed Phase-1 seeding (single launch when VRAM allows; log line
`single-launch seeding`). **Not yet examined:** Phase 2 (locality ordering) and
the graph-wiring stage, both of which run on CPU. Build log at 100k×384d:
```
pivot assign  0.9s   (GPU, now single-launch)
locality order 0.02s (CPU)
refine        0.8s   (GPU, 12 NN-descent passes — already correct pattern)
total         1.7s
```
Phase 2 is cheap at 100k; re-check at 1M before optimizing.

---

## 3. Build time — the axis where we lose

### 3.1 Global graph write lock serializes concurrent ingest
`crates/views/src/vector.rs:4384` — `hnsw: RwLock<Hnsw>`, and `insert` takes
`.write()` (`:4502`). Measured: one in-process thread does 3,204 vec/s; **eight
wire clients do 5,879 vec/s — 1.8× from 8× the clients** on 16 cores.

Per-vector cost also grows with the graph (inherent to HNSW):
| vectors so far | rate | avg |
|---|---|---|
| 1k | 16,067 vec/s | 62 µs |
| 9k | 3,366 vec/s | 199 µs |
| 10k | 3,204 vec/s | 210 µs |

**Fix:** sharded build with per-shard locks, then bridge-merge. The machinery
exists and is proven (`build_parallel_ids`, `merge_segments`, recall 0.994 vs
0.999 serial); the wire path just doesn't use it. Largely subsumed by §1.1 if
that lands.

---

### 2.5 GPU search is slower than CPU search — the biggest open perf question
Once `upload_to_gpu` was actually called (it had one caller in the tree, so
`GpuIndex` was never built outside `quick_bench`), the real device numbers
appeared and Turbo is **0.62× single-thread / 0.77× concurrent** against CPU
search at 100k×384d — ~545 µs/query vs ~340 µs.

Per-query launch + PCIe round-trip exceed the graph walk they replace.
`QueryCoalescer` exists to amortize exactly this and evidently is not engaging
on this path — that is the first thing to check. Until it is understood, the
defensible claim is **GPU-accelerated *build***, not GPU-accelerated search.

## 4. Unproven / unmeasured — needed before publishing

- [ ] **1M×384d GPU** build + recall + QPS
- [ ] **1M×768d GPU** build + recall + QPS.
      Watch VRAM: the single-launch seeding guard in `gpu_build_knn_graph`
      requires `full_bytes × 1.5 < vram_free`; at 1M×768d that's ~1 GB of
      buffers. Confirm it doesn't silently fall back to batched (the log says
      which path it took).
- [ ] **Filtered-search selectivity sweep** (0.1 / 1 / 10 / 50 / 90%). Free
      ground — Qdrant pays 2–10× on ACORN and publishes nothing here.
- [ ] **Hybrid larger-than-VRAM** demonstration (blocked on §2.1)
- [ ] Cold-start and crash-recovery timings as formal benchmarks (we claim to
      win big here vs Qdrant's 35 min / manual repair, but haven't published a
      harness)

---

## 5. Docs still carrying stale claims

- [x] `README.md` — corrected this session (unreproducible GPU table removed,
      VUGVA described accurately, APGC stated to replace CAGRA, ingest
      bottleneck and GPU limitations documented)
- [ ] `docs/` — **not reviewed**, likely still claims CAGRA and cuMemAllocManaged
- [ ] `index.html` (landing page) — **not reviewed**, same risk
- [ ] Any GPU perf table anywhere must be regenerated from §4

---

## 6. Lower priority / stale backlog

- `#27` consolidation background reducer (architecture DD2)
- `#34` Product Quantization with learned codebooks
- `#40` `--xlarge` 1M×1536d bench section
- `#43` pipeline the demos for real speedup
- `#47` reduce per-op allocations in the demos' RESP client

---

## 7. Landed this session (for context, `6e5c1b8`)

**Correctness — both silent, both now covered by regression tests:**

- **`build_parallel_ids` id scramble → recall 0.000.** Mapped segment-local rows
  back through `inv` where the shuffle needs `perm`. Because `shuffled_attr`
  already indexed through `perm`, the graph and attributes were *correct* and
  only labels were wrong — so search stayed fast and returned plausible
  neighbours, and nothing looked broken until recall was measured against ground
  truth. The sibling `build_parallel_attr` had it right all along; that
  inconsistency is what exposed it. Test verified to fail without the fix
  (598/600 wrong at 4 shards, clean at 1).
- **`VADDBATCH PAR` quadratic cliff.** Rebuilt the whole index per call →
  106 s vs 17 s for plain append over 100k in 64-vector batches. Batches under a
  quarter of the index now take the append path.

**VUGVA T2 implemented.** The cold tier existed only as a comment: `TieredPool`
had no spill member, `allocate()` backed every page with DRAM *before* consulting
`initial_tier`, and `access()` read the DRAM chunk without opening the file — so
the DRAM pool bounded the whole corpus and the capacity cliff had merely moved
down one tier. Now: `allocate()` honours `Tier::Ssd` by reserving spill space,
`access()` genuinely stages SSD→DRAM→VRAM with DRAM attached on demand as read
staging, `SpillFile` gained `reserve()`/`write_at()`. Verified on hardware by
`paper_ssd_tier_promotes_cold_pages` (1 MiB cold page vs a 4 MiB DRAM pool and a
64 MiB spill file — a two-tier cache cannot allocate that at all).

**GPU build 9.29×** — see §0.1 and §2.4. 100k×384d: serial 21.9 s (4,559 vec/s)
→ parallel 2.4 s (42,366 vec/s), recall 0.994 merged vs 0.999 serial.

**Also:** re-vendored `vugva-core` (the vendored copy was missing `context.rs`,
`range_alloc.rs`, `spill.rs` and 700 lines of `tiered.rs`); `CorpusTier` adapter;
AVX-512 VNNI int8 dot (exact, tested, honestly ~0%/6%); `DBSTRIKE_GPU` and
`DBSTRIKE_DOT` env switches; `bench-out/` gitignored (159 MB).

---

## 8. Strategic read — how to actually win

**Do not claim to beat Qdrant and Milvus on every axis.** They are mature
distributed systems; someone will find the axis where we lose, and that is what
destroys credibility. Claim what is measured and reproducible.

Defensible today, all measured on this box:

> Zero-dependency Rust vector engine — 0.985 Recall@10 at 416 µs p99 and
> 25k QPS on 100k×384d, sub-second cold start, automatic crash recovery.

Genuine differentiators, in order of strength:
1. **Zero external crates.** Pure stdlib + raw FFI; NVRTC means no CUDA build
   dependency, no cuBLAS, no BLAS. Milvus links the entire RAFT/CUDA stack.
   Nobody else in this space can say this.
2. **VUGVA larger-than-VRAM-and-RAM.** Memory layer done and hardware-verified;
   search path pending (§2.1). Neither competitor has it.
3. **Cold start / crash recovery.** Qdrant has open issues: #9496 (~35 min for a
   32-shard collection) and #9857 (crash loops needing manual repair).
4. **Filtered search without an ACORN-style 2–10× penalty** (§4).

The one number to fix before claiming anything about scale is **ingest** (§1.1,
§3.1).
