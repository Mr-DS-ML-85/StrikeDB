# StrikeDB — Agent Status / Handoff Document

> **Read this first.** This file is the single source of truth for any AI agent
> continuing work on StrikeDB. It contains the current state, frozen wire
> contracts, open bugs with repro recipes, hard-won operational rules, and the
> prioritized roadmap. Last updated: 2026-08-24 (post BUG-1 fix d389d43, memgent scope/time fixes 436f656+18f02e8).

---

## 0. What StrikeDB Is

A unified Rust database engine where **KV, vectors (HNSW ANN), relational tables,
time-series, pub/sub, CRDTs, AI agent memory (LTM/STM/episodic/graph/bi-temporal/
procedural), and RAG** are views over ONE MVCC+WAL storage substrate. RESP wire
(Redis-compatible, RESP2+RESP3), zero external crates, Apache-2.0.

Core differentiators nobody else ships: one WAL across every model, MVCC
point-in-time reads over vectors (`GETAT`), agent-scoped memory recall,
bi-temporal facts, FLUSHALL-with-undo, MITM cache debugger, `make prove-it`
durability harness.

Server default port in docs: `6380`. Dev/audit runs use `6565`. A second,
unrelated instance may run at `:6380` from `/run/media/irfan/models/StrikeDB`.

---

## 1. Current State (all committed through `69a3078`)

### Working & verified
| Area | State |
|---|---|
| KV SET/GET/INCR/MSET/MGET/DEL | ✅ durable, group-commit, 17M ops/s @P1024 |
| INCRBY concurrency | ✅ fixed (`Kv.txn_lock`) — was losing updates under ≥2 clients |
| Vector HNSW INT8+f32-rerank | ✅ 100k×384d self-recall 14/14 @ 0.5 ms |
| Quant modes (8) | ✅ INT8/BINARY/BINARY2/BINARY15/TURBO1/TURBO15/TURBO2/TURBO4/PRODUCT — all 100% self-recall@5 probe; TurboQuant = Zandieh et al. ICLR 2026 rotation-based VQ (Qdrant shipped it as 1.18 too — we are NOT behind) |
| Namespaces | ✅ per-ns dim, VADDNS/VSEARCHNS/VBULKLOADNS/VLISTNS/VDELNS/VSETQUANTNS/VFITQUANTNS/VADDBATCHNS |
| Bulk load | ✅ VBULKLOAD(NS): mmap file input, parallel sharded build, ONE atomic put_batch, auto-checkpoint >8 MB |
| Agent memory | ✅ LTM/WM/episodic/graph/procedural/bi-temporal, owner-scoped recall (`[AGENT name]` on all MEM/RAG cmds), reserved `vec:__ltm__:` namespace |
| FLUSHALL | ✅ real wipe w/ zero-copy backup `<wal>.bak-<ms>` + `.snap` twin; undo documented; serialized on flusher thread; resets Memory mirrors + RAG query cache |
| HELLO | ✅ served pre-auth; RESP2 flat array / RESP3 `%` map negotiation; `_` nulls per-connection after proto-3; embedded AUTH honored |
| Durability proofs | ✅ `make prove-it` — 39 checks / 6 phases, exit 0 = all held |
| GPU APGC build | ✅ ~1.4–3× faster than CPU build (recall-restored config), NVRTC runtime compile, zero CUDA deps |
| GPU batch search | ✅ 1.93× CPU (was 0.79× before f32-scoring kernel fix); single queries always CPU (routing by shape) |

### Test counts (last full run)
cargo --release green (47 ok suites incl. new gates) · prove-it **39/39** ·
wire-stride python **16/16** · MEM.INVALIDATE dual-client matrix green
(python RESP 9/9 + redis-cli). New regression tests: apgc_wire_real,
topk_kernel_check, ltm_invalidate_is_owner_scoped,
recall_applies_bitemporal_window_at_now, recall_honours_unix_ms_windows.

### Commits this arc (newest last)
`9a0c1f3` FLUSHALL/scoping/HELLO → `08fd7c8` mirror reset → `6d34eac` undo docs
→ `f3e0ce0` INCRBY+LTM-namespace+harness → `9123bb1` gitignore → `7b10b82`
harness docs → `e2e4d14` bi-temporal hits + locomo_diag → `ccbe4c6` WITHMETA
opt-in → `69a3078` GPU fixes (this doc's parent).

---

## 2. ⚠️ OPEN BUGS (prioritized)

### ~~BUG-1~~ ✅ FIXED (d389d43) — APGC fallback lacked global bridge routing
- Root cause was NEVER the tier flag. Controlled wire A/B: GPU-build success =
  12/12 self-recall@50, forced "segment kNN" fallback = 1/12, same tier setting.
- Chain: segment-local kNN honest-but-useless on shuffled clustered data
  (best in-segment d≈0.054 vs global truth d≈0.000); old bridge wired nodes
  only to random entry HUBS (merge_segments' foreign-segment beam search was
  never ported); aggravated by silent per-segment topk failure under VRAM
  pressure, entries×64 bridge coverage cliff, i32::MAX garbage weights, and
  zero-padded flat-kNN phantom edges to node 0.
- Fix: Phase 2b beam walks over nearest foreign segment's local graph + loud
  failures w/ CPU brute-force recovery + per-node bridges + self-padding.
  Post-fix: forced-fallback wire recall 12/12 @100k; overlap@10 0.995 vs
  success 0.985.
- Gates: `apgc_wire_real.rs` (brute-force-verified, real data), seam
  `DBSTRIKE_FORCE_APGC_FALLBACK=1`, diagnostics `DBSTRIKE_APGC_DEBUG=1`.
- `tiered=false` still forced in production pending a 1M tiered=true rerun.

### BUG-2: VUGVA runtime tiering incomplete  *(P1 — partially stale)*
- STALE part: Hybrid upload DOES route through CorpusTier/TieredPool since the
  VUGVA commit (gpu_build_index hybrid branch). The old fits-or-CPU branch is
  gone; `gpu_should_use_gpu` is dead code awaiting deletion.
- REMAINING: no runtime promote/demote during search (device_ptr promotes once;
  demote()/sweep() have zero production callers), no live tier stats in
  GPU.INFO (only static gpu_tier_strategy strings), and the >VRAM end-to-end
  benchmark remains undone (APGC paper admits the same).

### BUG-3: OpusEdge partially wired  *(P2)*
- Kernels (`opusedge_selkv_prune/delta_ar_route/head_gate/state_compress/
  proxy_delta`) launch in the APGC batch-search path; δ-presort applied at
  upload. BUT `GPU_DELTA_AR_K` **defaults to 0 = disabled**, and the entire
  `crates/opusedge` Rust crate (signal/stabilizers/primitives) is dead code —
  imported by nothing outside its own crate. No benchmark proves SelKV/Delta-AR
  helps ANN search (they're LLM-KV primitives repurposed).
- Reference repo: github.com/Mr-DS-ML-85/OpusEdge (C++20, PolyForm NC license —
  **do not copy code into Apache-2.0 StrikeDB**; concepts only).
- Paper: `research/OpusEdge_Paper.txt` (Δ-signal: SSM selectivity / RMS drift /
  router entropy → SelKV eviction, SMSA windows, Delta-AR routing). The ANN
  analogue: Δ = graph hubness (in-degree), SelKV = cold-node pruning.

### GAP-4: Competitor parity (see §5 roadmap)
- JSON payloads + rich filters (flagship gap — only u32 attr Eq/In today)
- Client-selectable hybrid fusion (RRF/DBSF) — dense+BM25 exists internally
- VSNAPSHOT CREATE/LIST/RESTORE API (rename machinery exists in FLUSHALL)
- Recommend(pos/neg), MMR, range search(radius), faceting
- Metrics beyond cosine; gRPC; distributed Raft; multimodal roadmap

---

## 3. 🔒 FROZEN WIRE CONTRACTS (never widen defaults silently)

| Command | Frozen reply | Opt-in extension |
|---|---|---|
| `MEM.RECALL [AGENT n] k q f…` | 4 fields/hit: `id score source text` | trailing `WITHMETA` → 7 (+created_ts valid_from valid_to) |
| `MEM.RECALL.AS_OF [AGENT n] k q t f…` | same 4-field | same WITHMETA |
| `HELLO` | RESP2 array (proto=2) | proto-3 arg → `%` map + `_` nulls on that connection |

Rule: extend reply shapes ONLY by appending NEW opt-in flags. Positional
parsers must never see a stride change. Guarded by prove-it phase 6.

Other stable behaviors: `FLUSHALL` = rename wal+snap to `.bak-<ms>` → wipe →
undoable offline; `GETAT`/`SCAN` read raw engine keys (no kv: prefix);
`vec:` keys store ids as 8-byte big-endian binary, values as string-float lists.

---

## 4. 🧯 OPERATIONAL RULES (violating these cost us hours — do not relearn)

1. **WAL isolation**: ALWAYS set `DBSTRIKE_WAL=<path>` per server run. Default
   is `dbstrike.wal` in cwd — shared across everything.
2. **Cleanup includes `.snap`**: `rm -f dbstrike.wal dbstrike.wal.snap
   dbstrike.wal.bak-*`. Deleting only the WAL resurrects the checkpoint world
   on next boot (caused an entire day of phantom state: ids at 52, dbsize
   200588, ghost namespaces).
3. **Kill strays by port→pid**, never pkill: `ss -tlnp | grep <port> | grep -oE
   "pid=[0-9]+"`. `pgrep -f dbstrike` matches your own command line.
4. **Background daemons**: `(setsid nohup cmd > log 2>&1 < /dev/null &)` — the
   bash tool WAITS on background children otherwise. Poll logs in separate
   short commands.
5. **No pipes mid-run**: `| grep`/`| tail` block-buffer output — healthy runs
   look hung. Run bare or poll log files.
6. **`/tmp` dies on shutdown** — venvs/logs go in repo `.prove-it/`.
7. **Patch, don't rewrite** user code; python-EOF heredocs only on scratch
   scripts, never on their Rust.
8. **Check your harness BEFORE blaming the DB**: today's biggest false alarms
   were (a) missing `DBSTRIKE_WAL` isolation, (b) bytes args stringified into
   literals, (c) `.snap` resurrection. Prove rig on known-empty state first.
9. **VRAM squatters**: local `llama-server` holds ~7 of 8 GB → GPU.MODE fails
   intermittently. Now reports honestly: `ERR no usable GPU — <device> present
   (…) but CUDA context creation failed — VRAM almost certainly exhausted…`.
10. **Server env knobs**: `DBSTRIKE_WAL`, `DBSTRIKE_SYNC=1`(default durable),
    `DBSTRIKE_PASS`, `GPU_SELKV_RATIO`(0.9), `GPU_DELTA_AR_K`(0=off),
    `DBSTRIKE_GPU_SINGLE`, `DBSTRIKE_TRACE`.
11. **memgent repo is OFF-LIMITS** (`/run/media/irfan/models/memgent`) — owned
    by another AI agent. OmniMemEval data is read-only:
    `/run/media/irfan/models/OmniMemEval/data/locomo/locomo10.json`.
12. **Datasets**: `/home/irfan/datasets/*.fbin` — format `[n:u32][dim:u32]
    [n*dim f32 LE]`. Available: real_{384,768}_{100k,1M}, scale_384_{50k,100k,
    200k,5000000}.
13. Nomic embedder endpoint: `http://192.168.0.110:8374/v1/embeddings`, model
    `nomic-embed-text-v1.5`, 768-dim, OpenAI-compatible, ~0.37s/8 texts.

---

## 4b. 🔒 MEMORY TIME-DOMAIN CONTRACT (18f02e8)
valid_from/valid_to are WORLD times. Callers speak unix-ms; benchmarks may use
small synthetic values. Present-tense recall evaluates windows against
`max(logical_clock, wall-clock-ms)` (`Memory::effective_now`). AS_OF stays
caller-supplied verbatim. MEM.INVALIDATE is owner-scoped
(`ltm_invalidate_scoped`): fact owner must match requesting agent ("default"
for bare form); unknown ids error. Do NOT evaluate validity against the bare
logical clock — that hid every wall-clock-dated fact after 436f656.

## 5. 🛠️ THE HARNESSES (use these; extend, don't fork)

### `make prove-it` → `scripts/prove_it.py`
Public durability/correctness evidence engine. stdlib-only, RESP-only, ~11s,
exit 0 = all held. Phases:
1. Acked durability — 6× randomized SIGKILL, mixed workload, every acked
   write verified post-restart (KV + MEM.COUNT exact)
2. Bulk-load atomicity — SIGKILL mid-VBULKLOADNS → namespace is 0 or full
3. Concurrency — 8 clients × 250 INCRBY, zero lost updates
4. Invariant fuzz — 400 random mixed ops + crash: scoping intact, MEM.COUNT exact
4b. Per-subsystem durability — one acked write each: KV/vector/TS/table/CRDT/memory
5. FLUSHALL undo — backup restore round-trip
6. Wire-contract strides — MEM.RECALL 4-field frozen / WITHMETA 7-field

Rules when extending: isolated `DBSTRIKE_WAL` per phase (start_server enforces
port-free assertion), atexit kills strays, check() appends don't abort. It has
caught 2 REAL product bugs (INCRBY races, LTM/user vector namespace collision)
plus the FLUSHALL mirror-reset bug — trust it over unit-green feelings.

### `scripts/locomo_diag.py` — retrieval-only LoCoMo diagnostic
Pure-retrieval truth on LoCoMo (no LLM judge): ingests turns with real
embeddings (nomic HTTP or fastembed bge), session dates grounded in text +
`valid_from`, `dia_id` as source provenance; measures gold-evidence hit@k per
category. Flags: `--convs N --k K --embed-url … --embed-model … --granularity
turn|sentence --qprefix`.
Results (conv1, k=10): temporal **86.5%** (was 5.9% pre-wiring), adversarial
72%, open_domain 67%, multi_hop ~50%, single_hop 44%, overall ~67%. Embedder-
invariant (bge≈nomic), granularity-invariant → bottleneck = atomic-fact
extraction, which lives in memgent (other agent).

---

## 6. 🗺️ ROADMAP (priority order)

1. **BUG-1 root cause** — tiered merge bridge collapse. Start by reading
   `merge_segments` bridge wiring when `tier: Option<MmapTier>` is Some vs
   None, and the APGC-failure fallback ("segment kNN") in the GPU branch.
2. **P0 payloads + filters** — `VSETPAYLOAD ns id json` / filtered search with
   `field=val AND price<100` grammar, durable `vp:` keys, payload index +
   planner. THE flagship gap vs Qdrant.
3. **Fusion flags** — `VSEARCH … FUSION RRF|DBSF|DENSE|BM25` exposing the
   existing SparseIndex.
4. **VSNAPSHOT CREATE|LIST|RESTORE** — reuse FLUSHALL rename machinery; S3 later.
5. **Wire VUGVA** — route search through corpus_tier when corpus > free VRAM;
   tier stats in GPU.INFO; then the >VRAM end-to-end benchmark (paper's
   admitted gap).
6. **OpusEdge decision** — either bench SelKV/Delta-AR benefit on ANN (env
   knobs exist) or strip the dead crate; enable-by-default only with proof.
7. **P1 small wins** — VRECOMMEND pos/neg, MMR flag, range search (radius),
   VFACET field.
8. **P2 Qdrant-compatible REST shim** — `/collections/{name}/points/search`
   subset → instant LangChain/PixelRAG ecosystem compatibility.
9. **Graph persistence** — serialize HNSW to disk; restarts currently rebuild
   (167s @1M even parallel). Competitors mmap graphs.
10. **Metrics beyond cosine** (euclidean/dot), gRPC, distributed Raft — far.
11. **Multimodal** — images/video/files/code/audio/3D = payload/file storage +
    named multi-vectors per point + external embedder pipelines; DB-side needs
    are P0 payloads + named vectors, not new indexes.
12. **LoCoMo end-to-end** — after memgent lands atomic-fact extraction, rerun
    `locomo_diag.py` unchanged; prediction: single_hop 43→65%+, overall 80%+.

---

## 7. QUICK COMMAND REFERENCE

```bash
# build + full tests + evidence
cargo build --release && cargo test --release && make prove-it

# clean server (ALWAYS isolate the WAL, ALWAYS remove .snap)
rm -f dbstrike.wal dbstrike.wal.snap dbstrike.wal.bak-*
(setsid nohup ./target/release/dbstrike 127.0.0.1:6565 > /tmp/s.log 2>&1 < /dev/null &)

# key RESP surface (full list in README)
VADDNS ns id f…        VSEARCHNS ns k f…        VBULKLOADNS ns file.fbin
VLISTNS                MEM.RECALL [AGENT n] k q f… [WITHMETA]
RAG.CONTEXT [AGENT n] k q f…                    FLUSHALL
GPU.MODE turbo|hybrid|cpu                       GETAT key snap

# commit discipline: cargo green + prove-it green before every commit
```

## 8. KEY FILE MAP

| Path | Contents |
|---|---|
| `crates/server/src/main.rs` | RESP dispatch (~2100 lines of handlers), HELLO, FLUSHALL arm, GPU.MODE, parse_agent_scope, parse_floats |
| `crates/views/src/vector.rs` | HNSW core: open_ns (parallel rebuild), bulk_load_fbin, merge_segments, quant modes, upload_to_gpu(_if_enabled), vec_at_f32/tier reads, tiered=false mitigation comment (~4728) |
| `crates/storage/src/engine.rs` | MVCC+WAL, flushall_with_backup, perform_flush_all, PendingWrite.flush_all |
| `crates/memory/src/lib.rs` | Meta.owner v3, recall_scoped, reset_volatile, __ltm__ namespace |
| `crates/router/src/lib.rs` | namespaces map, materialize-on-upload, flush_all_with_backup |
| `crates/gpu/src/lib.rs` | NVRTC kernels incl. opusedge_* names, init_ctx honest errors, corpus_tier (VUGVA), opusedge_knobs |
| `crates/kv/src/kv.rs` | txn_lock INCRBY fix |
| `scripts/prove_it.py` / `scripts/locomo_diag.py` | the two harnesses |
| `research/APGC_Paper.tex`, `research/VUGVA_Paper.txt`, `research/OpusEdge_Paper.txt` | project papers (APGC paper contains its own honest retraction section — read it) |
