# DB-Strike — Working Memory

**Master state file. Update on every code change.**
Last updated: 2026-07-27 · 22 commits ahead of `origin/main`, **none pushed**

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

All search columns at matched beam 128.

| mode | build | vec/s | Recall@10 | QPS 1t | QPS 16t | QPS batch |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only | 7.31 s | 13,683 | **0.999** | 2,781 | **27,894** | 2,860 |
| Turbo | **2.46 s** | **40,723** | 0.994 | 8,590 | 72,044 | 22,116 |
| Hybrid (VUGVA) | 2.70 s | 37,081 | 0.994 | 7,576 | 76,983 | 18,542 |

- **Build ~3×** (≈3.5× steady-state; NVRTC compile ~0.39 s sits inside the timer)
- **Batched GPU search 0.79× a saturated 16-core CPU** — 7.73× one core. The GPU
  does *not* win at search on this hardware.
- 1t/16t columns run identical CPU code in all rows; the 2.6–3.1× gap is
  **graph structure**, not hardware, and survives CPU-only deployment.
- Batch recall validated: matches single-query recall exactly.

### Ingest

| path | rate |
|---|---:|
| Wire `VADDBATCH`, batch 64 | 5,812 vec/s |
| `bulk_load_fbin` direct (CpuOnly) | **14,183 vec/s** (7.05 s) |
| `VBULKLOAD` over RESP | **works** — 1k verified; 100k not yet timed |

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
| Server re-parsed partial frames from byte zero | 96 MB command ≈ 72 GB of parsing |
| `export_vectors` allocated per vector, then copied again | 100k allocs + 300 MB per rebuild |
| LGM hysteresis scaled net score (negative → inverted) | residents evicted preferentially |

### Retracted

**"12.43× batched GPU search"** — wrong twice: `search_many` dropped `ef` to 64
against the CPU's 128, and the CPU baseline was single-threaded. Real: 7.73× one
core, **0.79× a 16-thread CPU**. Corrected in README, both docs, both landing
pages, LGM.md.

---

## 3. Open — ordered by value

### 3.1 RESP wiring audit ← **highest value**
Features exist but the server may not reach them. Verify each:
- [ ] `VBULKLOAD` at 100k with `DBSTRIKE_GPU=turbo` — time it, confirm APGC runs
- [ ] Does `GPU.MODE turbo` over RESP actually change the server's build path?
- [ ] Does `VSEARCH` use `search_many` for pipelined batches?
- [ ] Is the LGM membrane reachable from the server at all? (currently not)

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
  1,566,595 of 1,572,864 bytes wrong).

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
