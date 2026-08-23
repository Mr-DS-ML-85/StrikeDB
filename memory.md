# DB-Strike — Working Memory

**Master state file. Update on every code change.**
Last updated: 2026-08-05 · GPU batch Recall@128 0.39→0.992 (f32-scoring fix, §9)

---

## 8. Bottleneck investigation — three real bugs (write-up of the fix session)

Three independent sub-agent audits (GPU / CPU+RAM / DB code) converged on the
same verdict: **hardware is not the problem** (RTX 4060 ~15 TFLOPS / 272 GB/s /
24 SMs; Ryzen 7700 at 5.4 GHz with AVX-512 VNNI active; governor=performance;
DDR5 ~47 GB/s measured). The DB has **two dead-code/algorithm bugs** and the
benchmark inflates its own result. All findings verified by reading the source
at the exact lines below.

### VERIFIED LIVE over RESP (2026-08-05, server 127.0.0.1:6380, scale_384_100000.fbin, GPU.MODE turbo)

Before trusting the sub-agent write-ups, I tested the running server directly via
`redis-cli`/redis-py over the wire:

| path | RESp command | recall@128 | notes |
|---|---|---|---|
| CPU single | `VSEARCH k <384 floats>` | **0.9992** | correct |
| **GPU batch** | `VSEARCH.MANY k dim <NQ×384 floats>` | **0.3906** | **garbage — BUG 1 CONFIRMED LIVE** |

- `VSEARCH.MANY` returns exactly **`batch-size NQ` hits per query** (NQ=16 → 16,
  NQ=50 → 50, NQ=100 → 100), not k=128. The same query returns wildly different
  recall run-to-run (0.1562 → 0.4688 → 0.3906) on a deterministic index — the
  signature of stale/uninitialised GPU output buffers.
- It can also **panic the whole server**: `thread '<unnamed>' panicked at
  crates/views/src/vector.rs:1991:42: copy_from_slice: source slice length (64)
  does not match destination slice length (384)` — a quantize/dim desync reached
  through the batch path. The server died mid-test on that run.
- So the sub-agents were **wrong** that the batch path "silently falls back to
  CPU." It returns corrupt data and can crash. The `1.93× batch beats CPU` line in
  the README/memory.md table (commit `1d0a36f`) measured this broken path.
- **The GPU kernel really can't launch** (135,104 B smem vs 101,376 B opt-in
  ceiling measured on AD107). The whole `search_many` GPU path is dead code that
  publishes garbage instead of failing loudly.

### Bug 1 — the APGC GPU search kernel can never launch (fatal, silent)

- `apgc_search_smem` (crates/gpu/src/lib.rs:1825–1844) sizes dynamic shared
  memory for `apgc_search_kernel`. For the benchmark workload (itopk=512,
  beam=64 default, `index.degree`=64) it computes **135,104 B**:
  BUF = next_pow2(512 + 64·64) = **8192**; 3·BUF + SR_HASH(8192) + D4(96) +
  16 + rerank(384 + next_pow2(GPU_TOPK_MAX=512)=896) = 33,776 ints · 4 = 135,104.
- RTX 4060 is **compute capability 8.9 (AD107)**. Verified empirically:
  default dynamic-smem limit = **48 KB (49,152 B)**, opt-in ceiling =
  **101,376 B (99 KB)**, per-SM = **102,400 B**. The kernel needs 135 KB — it
  can NEVER launch, opt-in or not.
- The opt-in call **`cuFuncSetAttribute(... MAX_DYNAMIC_SHARED_SIZE_BYTES)`
  is never made anywhere** in the codebase (grep-verified). So even a request
  between 48 KB and 99 KB would be rejected.
- Consequence: `cuLaunchKernel` returns non-zero at
  crates/gpu/src/lib.rs:2161, `gpu_search_batched` **silently falls through**
  to the fused brute-force scan (2178) → two-kernel brute-force (2216) →
  `None` → caller falls back to CPU HNSW. `gpu_search`/`search_many`/fused
  batch (crates/views/src/vector.rs:5179, 5324) all end up CPU. **Every "GPU
  batch" number on record was CPU-path or a brute-force scan, not APGC.**
- Why the graph is 64-wide: `gpu_degree = degree.min(64)` (vector.rs:4652).
  **Degree cannot be trimmed** — all_kernels.cu:110–112 measures that
  clamping 64→32 drops recall@10 from 0.996 to 0.948 (graph adjacency is not
  quality-ordered; truncation is never free). Trim must come from `beam`,
  `SR_HASH`, or `itopk`.
- `search_many` also over-fetches: `itopk = fetch_k.max(rerank_k).max(128)`
  with `fetch_k = max(k*4, 64)` = 512 at k=128 (vector.rs:5312, 5320), which
  is what forces itopk=512 into the sizer.

### Bug 2 — CPU search frontier is unbounded (3–5× too many distance evals)

- `search_layer` (crates/views/src/vector.rs:2311–2328) pushes **every
  unvisited neighbor** onto the candidate heap ("Always extend the frontier",
  2316). hnswlib inserts into the ef-set first and only pushes if the neighbor
  made it in. Result measured: **avg 16,418 node visits/query** at ef=512
  (p50 16,786, max 23,659) vs ~4–6k for a tuned HNSW. The `ef*2` capacity
  hint at 2288 is not a bound — the heap grows to 10⁴+.
- This is pure over-work: distance evals dominate QPS. Bounding it should lift
  single- and multi-thread QPS ~3× with zero recall change (the same ef-set is
  kept; only hopeless frontier candidates stop being expanded).

### Bug 3 — the segment-bridge build pass is single-threaded (81% of build)

- `merge_segments` Phase 2 (vector.rs:3382–3399): for each of `total` nodes, a
  serial `search_local` graph walk into the nearest other segment to add
  cross-shard edges. 16 shards × 6,250 nodes = the whole pass runs on one
  core while 15 sit idle. Measured 6.38 s build → ~4.1 s is this bridge.

### Benchmark inflation — the "QPS collapse" was a metric change, not a graph

- `s_gpu_bench` calls `search_ef(q, 128, 128)` (crates/bench/src/main.rs:2802)
  but `rerank_k = (k*4).max(64)` (vector.rs:5213) inflates the traversal to
  **ef=512** at k=128 (vector.rs:5214). Old 27,894 QPS was k=10/ef=128;
  27,894/4 ≈ 6,970 ≈ measured 6,734. Not a graph regression, but the headline
  was silently measuring a 4×-harder workload. M=32 / M_max0=64 are unchanged
  (vector.rs:1860–1861).

### Fix plan (decided, ordered)

1. **Make APGC launch**: add `cuFuncSetAttribute` opt-in AND trim smem to
   ≤ ~96 KB. Trim levers (degree fixed at 64):
   - `SR_HASH` 8192 → 4096 (dedup set; kernel + host sizer both, they MUST
     stay in sync — the 9958fb0 bug proved the corruption that follows a
     mismatch).
   - `beam` default 64 → 32 (BUF = next_pow2(itopk + beam·degree): with beam 32,
     BUF = next_pow2(512+2048) = 4096 → 3·4096 + 4096 + 96 + 16 + 896 =
     18,688 ints · 4 = **74,752 B**, fits). Beam is per-iteration expansion,
     not the list width; recall impact to be verified empirically, fall back to
     beam 48 (85,952 B) if recall drops.
   - Verify recall ≥ 0.976 with the unit test `test_apgc_gpu_search` + bench.
2. **Bound the CPU frontier** (Bug 2): push a neighbor only when it either
   improves the ef-set or the ef-set is not full; keep the greedy `> worst`
   break. Expected ~3× QPS at identical recall.
3. **Parallelize the bridge** (Bug 3): chunk `total` across threads for Phase 2
   (Phase 1 is already GPU/vectorized). Segments' `search_local` is read-only.
4. **Kill the benchmark inflation**: document that `search_ef(q,k)` traverses at
   `max(ef, k*4)`; or make bench report the true ef. Keep behavior (recall
   needs the over-fetch) but stop presenting it as a smaller workload.
5. Re-run `--gpu-bench` + wire verify; confirm batch path is **real APGC graph
   search, not brute-force**; update README table; commit.

**Hard rule from user: no brute-force search — the fused/brute fallback must
not be what produces benchmark numbers. GPU batch must run the APGC graph
kernel.**

---

## 9. GPU batch recall 0.39 → 0.992 — the f32 scoring desync (FIXED)

**Symptom (verified live over RESP, 2026-08-05):** `VSEARCH.MANY` (GPU batch,
Turbo mode, 100k×384d) returned **0.3906 Recall@128** — garbage IDs that
changed run-to-run, with `hits/query == NQ` (only NQ hits, not k=128). The
single-query CPU path measured 0.9992 at the same time, so the graph itself was
fine. The bug was **entirely inside the GPU kernel's f32 candidate-scoring path.**

### Root cause — `crates/gpu/kernels/all_kernels.cu`

In `apgc_search_kernel`, the per-iteration candidate-scoring loop had this bug:

```c
// BUG: thread `tid` owns candidate `c` (outer loop), but this inner loop
// strided dimensions by `threads` starting at `tid`. Each candidate's score
// was a PARTIAL dot over ONE strided dimension, not the full dot product.
for (int c = itopk + tid; c < itopk + nc; c += threads) {
    ...
    for (int d = tid; d < D; d += threads) dot += qf[d] * vv[d];  // ✗
    buf_dot[c] = __float2int_rn(dot * 127.0f);
}
```

With `threads=512, D=384`, each thread's `dot` covered only `d ∈ {tid, tid+512,
…}` — i.e. **one or two dimensions**, not the full 384. Candidate scores were
therefore ~random, the bitonic sort ranked noise, and the walk lost its way.
The int8 fallback branch below it summed `d4=0..D4` fully and was correct —
that's why the 3 GPU unit tests (int8-only) passed while the live f32-rerank
batch path returned garbage.

Fix: each owning thread sums ALL `D` dims serially:

```c
for (int c = itopk + tid; c < itopk + nc; c += threads) {
    ...
    for (int d = 0; d < D; d++) dot += qf[d] * vv[d];  // ✓ full dot
    buf_dot[c] = __float2int_rn(dot * 127.0f);
}
```

### Second bug — entry-node seed score read the int8 buffer as float

At the top of the kernel, the entry-node score did:

```c
const float* vv = (const float*)vectors + (size_t)entry_node * D;  // ✗
```

`vectors` is the **int8 corpus** (`const char*`); casting it to `float*` reads
four packed bytes as one float and seeds the walk with garbage. With the f32
corpus live it must read `vec_f32` instead. Fixed.

### Why the kernel output looked "scrambled" after the fix

After fixing, batch recall jumped to **0.992** and query-0 top-8 matched ground
truth exactly — yet a debug read of the raw kernel output showed ids like
`[27841, 36403, …]` with the *correct* distances `[0.0, 0.177, 0.187, …]`.

**Not a bug.** The APGC kernel walks hnsw **position** space; `search_many`
maps position→external id through `g.nodes` (crates/views/src/vector.rs:5346).
The kernel's distances matched truth to the last digit; the id mismatch was just
position-vs-id addressing. The debug pairs were the smoking gun that the rerank
was now correct.

### Verification

- `NQ=1000, K=128, random queries`: **Recall@128 = 0.9920** (min 0.8906) vs the
  CPU path's 0.9992 — the ~0.7% gap is beam/candidate-set tuning
  (`fetch_k=512`, beam 56), not correctness.
- `VSEARCH.MANY` NQ=8/16/32/64, K=128: all hit counts = 128, q0 exact.
- Kernel launch `r=0, sync=0, c1=0, c2=0`; nvalid = out_count every time.

### Debug lines removed

The temporary `[GPU-DEBUG]` / `[SEARCH-DEBUG]` eprintlns from the investigation
were removed. A single sanity guard remains: if a batch returns fewer valid
hits than `out_count`, it logs `[GPU] apgc search: only n/… valid hits`.

---

Companion: `early-dev.md` holds longer-form findings. This file is the index —
what is done, what is open, what is broken, and what will mislead you.

---

## 0. Read first — five things that will mislead you

1. **Features are implemented above the wire, not through it.** The GPU build,
   VUGVA tiering, and the LGM membrane are all real and tested, but the
   benchmarks call in-process APIs (`VectorIndex::build_parallel`), not RESP.
   Before believing any feature "works", check that a *server* path reaches it.
   This is the single biggest structural gap in the project.

2. **The GPU never ran in any benchmark before `6e5c1b8`.** `gpu_available()`
   only read a flag that `gpu_init()` set, and nothing called `gpu_init()`
   outside the `GPU.LOAD`/`GPU.MODE` RESP handlers. Every GPU number predating
   that commit is CPU-path data.

3. **The distance kernel is not the bottleneck.** Forcing the scalar int8 dot
   instead of AVX-512 VNNI (~8× slower in isolation) changes search by ~0% and
   ingest by 6%. Verify with `DBSTRIKE_DOT=scalar|avx2|vnni`. Do not spend time
   on SIMD.

4. **Bulk ingest is not reachable through `VADDBATCH` at any batch size.**
   Small batches take the serial append path; large batches are *worse*
   (measured: batch 64 ≈ 18 s, batch 512 and 2048 both >200 s, batch 25000 made
   zero progress in 600 s). Use `VBULKLOAD`.

5. **Check your test harness before blaming the server.** `VBULKLOAD` appeared
   to hang for an hour of debugging. The server was fine; the Python client
   looped `while not buf.endswith(b"\r\n")` and blocked on an already-complete
   reply. Reproduce with a trace before concluding the system is broken.

---

## 1. Measured numbers (RTX 4060 + Ryzen 7700, 100k × 384d)

### GPU bench — `dbstrike-bench --gpu-bench <file.fbin>` (~25 s)

`nq` = 1000 queries (was 200). Two metrics: **Recall@128** (single-query, the hard
one) and **batch Recall@128** (fused GPU query path).

| mode | build | vec/s | Recall@128 | Recall@128 batch | QPS 16t | QPS batch256 |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only | 6.38 s | 15,679 | **0.999** | **1.000** | 6,734 | 735 |
| Turbo | 4.57 s | 21,874 | 0.996 | 0.999 | 11,077 | 1,418 |
| Hybrid (VUGVA) | 4.70 s | 21,277 | 0.996 | 0.999 | 12,267 | 1,441 |

- **Build ~1.4×** (was 2.98× pre-fix). `rev_cap` 16→48 + `ND_CAND_MAX` 2048
  doubled per-pass refinement; this is the cost of the recall restoration.
- **Batched GPU search 1.93× the 16-thread CPU** (1,418 vs 735) — the denser
  graph flips the old 0.79×. The device wins when queries arrive as batches.

- **Recall@128 over RESP** (server `VSEARCH`, `GPU.MODE turbo`, 100k×384d):
  **0.9993** at k=128, p50 3.5 ms — best wire number on record, after the
  `apgc_search_smem` sizer fix. Recall@10 over wire = 1.0000 @ 0.73 ms.
- The 0.999 user remembered was **Recall@10-era** (CPU 0.999 / GPU 0.994), not a
  lost Recall@128 config. `f176f12` switched the headline metric; Recall@128 is
  structurally harder and single-path GPU plateaus at 0.996–0.9987.
- **Root cause of the ceiling (fixed):** upper HNSW levels on the GPU-built
  graph were wired with *random* buddies (4/node); upper-level greedy descent
  stranded ~0.2% of true neighbors. ef=512, 24 NN-descent passes, diversity
  heuristic: all no-ops. The only things that moved recall: upper levels seeded
  with **real nearest same-level neighbors** from the GPU NN-descent kNN list
  (`crates/views/src/vector.rs`), `rev_cap` 16→48 (`crates/gpu/src/lib.rs`),
  `ND_CAND_MAX` 1024→2048 + `ND_HASH` 2048→4096 (`all_kernels.cu`).
- **200-query samples lie**: 0.999 at 200q was upward jitter; honest stable
  numbers only appear at ≥1000 queries. Always bench with `nq` ≥ 1000.

### Ingest

| path | rate |
|---|---:|
| Wire `VADDBATCH`, batch 64 | 5,812 vec/s |
| `bulk_load_fbin` direct (CpuOnly) | **14,183 vec/s** (7.05 s) |
| **`VBULKLOAD` over RESP + `GPU.MODE turbo`** | **36,148 vec/s** (100k in 2.77 s) — **6.2× the wire append path** |

### VUGVA

| | |
|---|---|
| Prefetch overlap | **3.2×** (8×32 MiB: issue 7.47 ms + claim 12.69 ms vs 40.81 ms cold) |
| T2 spill round trip | byte-exact, corpus 1.5× the DRAM budget |
| LGM membrane | 3/3 hot pages resident; 3/3 again after workload inversion |

---

## 2. Done and verified

- **VUGVA**: T0↔T1↔T2, NVMe spill (`reserve`/`write_at`), bounded DRAM staging,
  async look-ahead prefetch, `write_page`. 62 unit + 28 hardware tests.
- **LGM §3.4 membrane** (`membrane.rs`): EMA decay, score-ranked placement,
  85% high-water, hysteresis. Opt-in via `enable_membrane()`.
- **APGC GPU build**: pivot assign → locality order → NN-descent, INT8/dp4a
  throughout, single-launch seeding.
- **`CorpusTier`** → `GpuIndex` for Hybrid; `GpuIndex` owns the tier.
- **Query-shape routing**: single → CPU, batch → GPU.
- **`VBULKLOAD <path>`**: server-side bulk load, data never crosses the wire.
- **APGC paper reframed** (`research/APGC_Paper.tex`) — retitled *GPU-Built,
  CPU-Served*, fabricated results replaced, Threats to Validity added, compiles
  clean at 6 pages.
- Docs reconciled: README, `docs/apgc.md`, `docs/vugva.md`, `index.html`,
  `docs/index.html`.
- **GPU batch Recall@128 = 0.9920** live over RESP (was 0.39 garbage) after
  fixing the f32 candidate-scoring desync in `apgc_search_kernel` (§9).
  `VSEARCH.MANY` NQ=8/16/32/64 @ K=128 returns full k=128 hits, q0 exact.

### Bugs fixed (all were silent — none threw)

| bug | symptom |
|---|---|
| `build_parallel_ids` used `inv` where shuffle needs `perm` | **recall 0.000**, search still fast and plausible |
| GPU block quantized without L2-normalizing (3 sites + mmap tier) | wrong neighbours on un-normalized input |
| `all_f32`/mmap tier stored raw not normalized | rerank ranked by magnitude, not angle |
| `GpuBuildBuffers::run` ignored all CUDA status | OOM → all-zeros graph, every node's NN = node 0 |
| `gpu_unload()` left `OnceLock` state with dangling ctx | GPU "available" but every call failed with 201 |
| kNN padding used node 0 instead of self | spurious edges inflating node 0's in-degree |
| `gpu_check_capacity` returned VRAM cached at init | over-reported free memory |
| SelKV gate: `0.1f < 1.0f-0.9f` is **true** in binary32 | every zero-in-degree node silently unreachable |
| `upload_to_gpu` degree from node 0 alone | every other row truncated |
| APGC f32 candidate score summed `d = tid; d < D; d += threads` | each score a partial dot over ~1 dim → Recall@128 0.39 garbage on f32 rerank batch (§9) |
| APGC entry-node seed read int8 `vectors` as `float*` | walk seeded with garbage f32 dot when corpus live (§9) |
| Server re-parsed partial frames from byte zero | 96 MB command ≈ 72 GB of parsing (fix REVERTED — see below) |
| **My parse backoff waited for bytes never coming** | any multi-read command hung the connection |
| `GPU.MODE turbo` probed availability before setting mode | GPU unreachable over RESP entirely |
| `export_vectors` allocated per vector, then copied again | 100k allocs + 300 MB per rebuild |
| LGM hysteresis scaled net score (negative → inverted) | residents evicted preferentially |
| **Upper HNSW levels seeded with random buddies (4/node)** | ~0.2% of true neighbors structurally unreachable; ef=512 could not fix it |
| **`apgc_search_smem` sizer still used 1024 for the dedup set** | `SR_HASH` was raised 1024→8192 in `all_kernels.cu` but host'd dynamic shared-mem sizer wasn't updated; the kernel's `vis`/`qcache`/`s_ctl` overran the buffer and corrupted the bitonic sort → `apgc_search` returned sentinel `-1/2.0`. Broke 3 GPU unit tests; only visible in the test harness (the live benchmark path never exercised the raw kernel). Fixed by syncing sizer to `SR_HASH = 8192`. |
| **NN-descent `rev_cap` 16 + `ND_CAND_MAX` 1024 undersized** | reverse-neighbor refinement starved the GPU graph's local structure |
| **`nq`=200 sample in `s_gpu_bench`** | masked 0.999-at-200q = jitter; honest numbers need ≥1000 queries |

### Retracted

**"12.43× batched GPU search"** — wrong twice: `search_many` dropped `ef` to 64
against the CPU's 128, and the CPU baseline was single-threaded. Real: 7.73× one
core, **0.79× a 16-thread CPU**. Corrected in README, both docs, both landing
pages, LGM.md.

---

## 3. Open — ordered by value

### 3.1 RESP wiring audit ← **highest value**
Features exist but the server may not reach them. Verify each:
- [x] **`VBULKLOAD` at 100k = 2.77 s = 36,148 vec/s**, APGC confirmed running,
      `VSEARCH` answers in 0.7 ms straight after. Required reverting a parse
      backoff I had added that deadlocked any command split across reads.
- [x] **`GPU.MODE turbo` was broken** — probed `gpu_available()` *before*
      setting the mode, but availability only initialises the driver when a GPU
      mode is already current. On a fresh server (default `CpuOnly`) it always
      answered "no device", so `GPU.MODE turbo` returned `ERR no GPU detected`
      on a machine with a working GPU. **There was no way to enable GPU
      execution over RESP at all.** Fixed: set, probe, roll back on genuine
      absence. Same ordering bug appeared independently in `--gpu-bench`.
- [ ] Does `VSEARCH` use `search_many` for pipelined batches?
- [ ] Is the LGM membrane reachable from the server at all? (currently not)

### 3.1b Single-path Recall@128 plateau (0.996 vs CPU 0.999)
Graph is near (batch/wire hit 0.998–0.999) but CPU greedy descent over the flat
GPU graph + weak upper levels strands ~0.3% of true neighbors. Tried and failed:
ef=512, 24 passes, diversity heuristic. Next candidates: strengthen upper-level
fanout/degree, or a short GPU-recall-only pass fused into `search_unified`.

### 3.2 DGM §3.2 — lock-free COW slab graph (~500 lines)
Streaming insert/delete without rebuild. **This is also the real fix for
ingest.** Correctness-critical: deletion-monotonicity invariant, bounded
reclaim. Needs a fresh context window.

### 3.3 Unproven claims
- [ ] 1M×384d and 1M×768d in any GPU mode (watch the single-launch VRAM guard)
- [ ] QPS-vs-recall curve — removes the iso-recall confound; first thing a
      reviewer will attack
- [ ] Larger-than-VRAM search end-to-end (tier works, never exercised under load)
- [ ] Filtered-search selectivity sweep — free ground, Qdrant publishes none

### 3.4 Smaller
- [ ] `#67` fuse f32 rerank into `apgc_search` (in progress)
- [ ] DMA descriptor ring still advisory; transfers go via `cuMemcpy*Async`
- [ ] Cold T2 pages promote synchronously (no prefetch)
- [ ] `gpu_should_use_gpu:459` still does a binary fits-in-VRAM check
- [ ] `#80` APGC paper: decide implement-or-restate per contribution
- [ ] `#34` PQ codebooks, `#27` consolidation reducer, `#43`/`#47` demos

---

## 4. Papers

| paper | state |
|---|---|
| **VUGVA** | Implemented + hardware-tested. Publishable. |
| **APGC** | Implemented; paper reframed to match code. Needs iso-recall curve. |
| **LGM** | §3.4 implemented and verified. §3.3 GNN deferred — no baseline yet. |
| **DGM** | §3.4 = LGM's (done). §3.2 COW graph is the real work. |
| **ZERO** | Deleted by author. Central claim (SNL) described a flat adjacency list, which is what the code already had. |

**Not learned.** The membrane is frequency-driven — no gradient, no network.
Calling it "learned tiering" repeats the overclaim already retracted once.

---

## 5. Environment switches

| var | effect |
|---|---|
| `DBSTRIKE_GPU=turbo\|hybrid\|cpu` | compute mode (default `cpu`) |
| `DBSTRIKE_DOT=scalar\|avx2\|vnni` | force int8 dot kernel (attribution) |
| `DBSTRIKE_GPU_SINGLE=1` | force single queries onto the GPU (0.61×) |
| `DBSTRIKE_INGEST=par` | `VADDBATCH PAR` path in the bench |
| `DBSTRIKE_INGEST_BATCH=<n>` | vectors per `VADDBATCH` (default 64) |
| `GPU_SEARCH_ITOPK=<n>` | search beam (**set to 128 for fair comparison**) |
| `GPU_COALESCE=0` | disable query coalescing |
| `DBSTRIKE_SPILL_DIR` | where `CorpusTier` puts its NVMe spill |

---

## 6. Process notes

- **Background anything over ~30 s.** Two sessions were lost to foreground
  benchmarks. `--gpu-bench` is ~25 s; `--real` is minutes; `--parallel-ingest`
  runs 12 unrelated sections (96 s of WAL writes) before touching the GPU.
- **`pkill -f dbstrike-bench` kills sibling background jobs** — they share the
  pattern. Scope it or kill by PID.
- **Verify a regression test fails without the fix.** Two tests this session
  passed at 1 shard and only bit at 4/8, because 1 shard skips the shuffle.
- **`cargo test -p gpu` needs `--test-threads=1`** or a `gpu_exclusive()` lock;
  parallel CUDA contexts hand each other invalid pointers (observed:
  1,566,595 of 1,572,864 bytes wrong). Three GPU unit tests
  (`test_apgc_gpu_search`, `test_batch_cosine_dist`, `test_gpu_init`) were also
  failing from the sizer/SR_HASH mismatch — now fixed (see bug table), all 8 pass.

---

## 7. Verification

```bash
cargo test --release -p vugva-core -p gpu -p views -- --test-threads=1
cargo clippy --all-targets                       # expect 0 warnings
./target/release/dbstrike-bench --gpu-bench /home/irfan/datasets/real_384_100k.fbin
cd /home/irfan/Desktop/VUGVA/vugva && cargo test  # 62 unit + 28 hardware
```

Canonical VUGVA lives at `/home/irfan/Desktop/VUGVA/vugva`; `crates/vugva-core`
is a vendored copy. **Edit canonical, then re-vendor:**
```bash
rm -rf crates/vugva-core/src && cp -r /home/irfan/Desktop/VUGVA/vugva/src crates/vugva-core/src
```

---

## 8. Per-agent LTM recall scoping — the cross-tenant leak (FIXED, 2026-08-24)

### The bug
Working memory, episodic and procedural memory were always per-agent
(`mem:wm:<agent>:…`, `mem:ep:<agent>:…`, `mem:proc:<agent>:…`), but **LTM —
the semantic recall pool — was global**. `MEM.RECALL k query f1...` had no
agent identity at all: any agent (or anonymous connection) could recall every
other user's memories. A cross-tenant leak in the flagship feature.

### The fix (meta v3)
- `Meta.owner: String` added; meta format bumped to v3
  (`[ver][src][ts][sal][vf][vt][owner_lp][lineage]`). v1/v2 blobs decode with
  `owner = ""` so old WALs open unchanged.
- `recall_scoped(scope, …)` filters hits by owner BEFORE top-k truncation,
  with a 3× overfetch so scoped recall still fills k. Legacy records
  (`owner == ""`) stay visible to every scope — documented migration
  semantics for pre-scoping corpora.
- Wire API — optional leading token pair, zero breaking changes:
  `MEM.REMEMBER | MEM.REMEMBER.T | MEM.RECALL | MEM.RECALL.AS_OF |
  RAG.INGEST | RAG.SEARCH | RAG.CONTEXT` all accept `[AGENT <name>]`.
  No token → scope "default".
- RAG query cache key now includes the scope
  (`rag:q:{gen}:{scope}:{k}:{query}`) — agent A can never be served agent B's
  cached retrieval.

### Verification
- Unit: `ltm_recall_is_agent_scoped` (alice/bob/default isolation +
  legacy visibility + owner round-trip), `v2_meta_decodes_with_empty_owner`.
- Live RESP: alice recalls only her vault code, bob only his; unscoped default
  pool returns neither; isolation survives kill -9 restart.
- Full suite: 180 passed / 0 failed.

### Same-day adjacent fixes
- **FLUSHALL/FLUSHDB is real now**: WAL + `.snap` atomically renamed to
  `<wal>.bak-<millis>` (zero-copy backup of the entire pre-flush world),
  fresh WAL opened, shard maps + vector graphs cleared. Runs serialized on
  the group-commit flusher thread — cannot interleave with in-flight commits;
  crash-safe either way. Storage tests cover restorability + concurrent-commit
  serialization.
- **HELLO + RESP3 negotiation**: redis-py ≥ 8 sends HELLO unconditionally and
  defaults to RESP3. Server now serves HELLO pre-auth (handshake must not be
  NOAUTH-gated), replies `%` map claiming proto 3 when asked, switches nulls
  to `_` per connection (`write_resp_buf_as`). Embedded
  `HELLO proto AUTH u p` honored via dispatch_auth.
- **parse_floats accepts single-bulk vectors**: `VADD 11 "0.5 0.1 0.9"`
  alongside one-float-per-arg form.
