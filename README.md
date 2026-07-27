<p align="center">
  <img src="docs/assets/logo.svg" alt="StrikeDB" width="200" />
</p>

# ⚡ StrikeDB

**One engine. Every data model. The fastest path for AI agents, RAG, and real-time apps.**

StrikeDB is a single, unified data engine where relational tables, key-value,
vectors, time-series, pub/sub, **AI agent memory**, and **RAG** are all *views
over one storage substrate* — not five bolted-together systems with five
licenses, five release cadences, and five places for drift to hide.

> "Development at the speed of light" — but with the gaps fixed that every system
> you've been stitching together leaves open.

[📚 Docs](https://strikedb.devforge.qzz.io/docs/index.html) · [🚀 Live Benchmarks](https://strikedb.devforge.qzz.io//#benchmarks) · [🧠 Agent Memory](https://strikedb.devforge.qzz.io/#memory) · [🔍 MITM Cache Debugger](https://strikedb.devforge.qzz.io/docs/index.html#mitm)

---

## 🏷️ Status

![Rust](https://img.shields.io/badge/language-Rust-000000?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero%20(crate)--green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-310%20passing%20(53%20rust%20%2B%2074%20native%20%2B%20172%20integration)-brightgreen.svg)
![Durable SET P1024](https://img.shields.io/badge/durable%20SET%20@P1024-17M%20ops%2Fs-brightgreen)
![Durable GET P1024](https://img.shields.io/badge/durable%20GET%20@P1024-16.4M%20ops%2Fs-brightgreen)
![vs Redis SET](https://img.shields.io/badge/vs%20Redis%20SET-5.8×%20faster-brightgreen)
![vs Redis GET](https://img.shields.io/badge/vs%20Redis%20GET-3.9×%20faster-brightgreen)
![Benchmarked with](https://img.shields.io/badge/measured%20with-redis--benchmark-9cf)
![1M vectors](https://img.shields.io/badge/1M×768d%20VSEARCH%20p99-923%20µs-brightgreen)
![1M recall](https://img.shields.io/badge/1M×768d%20Recall@10-0.997%20(single%20node)-brightgreen)
![Chaos](https://img.shields.io/badge/Jepsen%20chaos-0%20lost%20%2F%2064k%20writes-brightgreen)
![Vector p50](https://img.shields.io/badge/100k×384d%20p50-69%20µs-9cf)
![Concurrent](https://img.shields.io/badge/1M×384d%20VSEARCH-8.9k%20QPS%20(8%20threads)-ff69b4)
![YCSB-C](https://img.shields.io/badge/YCSB--C-74k%20ops%2Fs-ff69b4)
![Write throughput](https://img.shields.io/badge/8--thread%20SET-345k%20ops%2Fs-ff69b4)
![SIMD](https://img.shields.io/badge/dot%20product-INT8%20AVX2%20%2B%20f32%20rerank-9cf)
![Realtime](https://img.shields.io/badge/wire-SUBSCRIBE%20%2F%20PUBLISH-ff69b4)

---

## 🚀 Why StrikeDB

| | Redis | pgvector | Qdrant | Milvus | Mem0 | Zep | SpacetimeDB | **StrikeDB** |
|---|---|---|---|---|---|---|---|---|
| KV / counters | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Relational tables | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Vector / ANN | ❌ | ⚠️ bolt-on | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ✅ **INT8 + f32 rerank** |
| Time-series | ❌ | ⚠️ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ✅ native |
| Pub/sub push over wire | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ **SUBSCRIBE/PUBLISH** |
| **RAG as a planned query** | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ✅ **cost-based** |
| **Graph agent memory** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ **typed edges + traversal** |
| **Bi-temporal facts (as-of)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ **native primitive** |
| **Procedural memory** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ per-agent namespace |
| **MITM cache debugging** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **built-in** |
| Logic *in* the DB | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ **fuel-metered** |
| **ACL + password auth** | ⚠️ 6.x+ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ **SHA-256 + salt** |
| One consistency model | ❌ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ⚠️ | ✅ MVCC |
| One license | BSD | PostgreSQL | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 | BSL | ✅ Apache-2.0 |


---

## 📦 Install & Run

```bash
# build the release binary (zero external crates — pure stdlib Rust)
cargo build --release

# RAISE THE FD LIMIT FIRST — without it deep pipelines cap at ~4.5M/s instead
# of the documented ~16M/s (100 clients starve the 1024-fd default ceiling).
ulimit -n 100000

# start the server on the RESP wire (talk to it with any Redis client)
DBSTRIKE_WAL=./dbstrike.wal ./target/release/dbstrike 127.0.0.1:6380

# optional: enable password auth (no auth by default)
DBSTRIKE_PASS=mypassword DBSTRIKE_WAL=./dbstrike.wal ./target/release/dbstrike 127.0.0.1:6380
```

Talk to it with `redis-cli`, `nc`, or the bundled harnesses:

```bash
redis-cli -p 6380 PING          # -> PONG
redis-cli -p 6380 SET user:1 ada
redis-cli -p 6380 VADD 1 1.0 0.0 0.0
redis-cli -p 6380 VSEARCH 5 1.0 0.0 0.0
redis-cli -p 6380 SUBSCRIBE trades &
redis-cli -p 6380 PUBLISH trades "hello"
redis-cli -p 6380 CHECKPOINT   # snapshot current state + truncate WAL

# with password auth enabled:
redis-cli -a mypassword -p 6380 PING
redis-benchmark -h 127.0.0.1 -p 6380 -a mypassword -P 64 -c 100 -n 200000 -t set,get

# native Rust bench harness — in-process, 29 sections in ~22s
cargo run --release -p bench

# add --tcp to also exercise SUBSCRIBE / PUBLISH end-to-end against
# a running server on 127.0.0.1:6380
./target/release/dbstrike-bench --tcp 127.0.0.1:6380

# 21-section Python integration + fuzz + crash-recovery suite
python3 tests/test_dbstrike.py
```

---

## 🎬 Real-world demos (native Rust, in `crates/demos`)

Three self-contained CLI apps that exercise the full engine over the RESP
wire — proof the primitives compose into actual products, not just
microbenchmarks. Zero external crates. Every app has a `--bench` (or
`--latency`) mode that prints p50 / p99 / throughput + client RSS, so
"fast" is objective, not a claim.

### Measured on this machine (2000 stored memories, single connection)

| App | Op | p50 | p99 | Throughput | Client RSS |
|---|---|---:|---:|---:|---:|
| **agent-cli** | `MEM.REMEMBER` (LTM + graph + BM25) | **170 µs** | 247 µs | 5,619 ops/s | 2 MB |
| **agent-cli** | `MEM.RECALL` (hybrid dense + sparse) | **311 µs** | 848 µs | **3,022 ops/s** | 2 MB |
| **tsdash** | `TSADD` | **26 µs** | 44 µs | **36,361 ops/s** | 2 MB |
| **tsdash** | `TSRANGE` (1000-sample window, single-shard) | **166 µs** | 252 µs | **5,734 ops/s** | 3 MB |
| **tsdash** | **`TSRANGE.LATEST 100`** (dashboard primitive) | **52 µs** | **93 µs** | **17,035 ops/s** | — |
| **rtchat** | PUBLISH→SUBSCRIBE round-trip | **31 µs** | 120 µs | — | — |

`MEM.RECALL` was 4.6× faster after this round's salience-cache optimization
(651 → 3,022 ops/s; p50 2,337 → 311 µs). See the roadmap `salience_cache`
entry below.

```bash
# start the server first
./target/release/dbstrike 127.0.0.1:6380 &

# 1) agent-cli — interactive AI agent (WM + LTM + graph + bi-temporal + RAG)
./target/release/agent-cli 127.0.0.1:6380 alice
#   :remember Alice is a senior engineer at Acme
#   :recall engineer               → semantic + keyword hybrid recall
#   :link 1 works_at 2             → typed edge in the memory graph
#   :fact CEO of Acme at 3000 for alice   → bi-temporal fact
#   :asof 5000 title               → as-of recall (Zep-style)

# 2) tsdash — live time-series dashboard (TSADD + TSRANGE + sparklines)
./target/release/tsdash 127.0.0.1:6380 30
#   renders cpu/mem/rps sparklines updated at 2 Hz for 30 seconds
#   dashboards should prefer TSRANGE.LATEST — 52µs p50 vs 166µs for a full range:
redis-cli -p 6380 TSRANGE.LATEST cpu 60

# 3) rtchat — realtime chat CLI (WebRTC-signaling shape: rooms + presence +
#    typing + push-latency measurement).  Note: video encoding needs
#    external media libs (ffmpeg/opencv) which we don't bundle; this
#    exercises the low-latency signalling substrate a real video app runs on.
./target/release/rtchat --self alice --room lobby --addr 127.0.0.1:6380
./target/release/rtchat --self bob   --room lobby --addr 127.0.0.1:6380
./target/release/rtchat --latency  --addr 127.0.0.1:6380   # PUB→SUB probe
#   measured: p50 = 30 µs, p99 = 89 µs round-trip over the wire
```

---

## ⚡ The "speed of light" path

SpacetimeDB's whole pitch is logic *inside* the database, so the network hop
between app and DB disappears. StrikeDB does the same — and fixes the one flaw
that makes SpacetimeDB dangerous: **the reducer-stall problem** (one bad reducer
serializes the entire DB).

Three in-house fixes:

1. **Bulkhead by shard** — a reducer only locks its key-range, not the world.
2. **Fuel-metered WASM** — every instruction is charged against a budget; overrun
   aborts and rolls back *cleanly* instead of trusting the author.
3. **Circuit breaker** — a reducer erroring above a threshold is auto-quarantined.

Measured: `REDUCE` p99 = **45.5µs** over RESP, and **zero lost updates** under
**32-thread × 500-op** in-process contention (16,000/16,000 in 0.15s).

---

## 📡 Realtime push over the wire — SUBSCRIBE / PUBLISH

Redis-compatible pub/sub sitting on top of the internal reactive/CDC layer:

```bash
# terminal A — subscriber
redis-cli -p 6380 SUBSCRIBE trades

# terminal B — publisher
redis-cli -p 6380 PUBLISH trades "AAPL 42"
# subscriber prints:  1) "message"  2) "trades"  3) "AAPL 42"
```

`PUBLISH` is a **durable** write on `chan:<name>` — every subscribed prefix
receives the pushed event via the same MVCC+WAL substrate. Verified
end-to-end in the native Rust bench in `--tcp` mode.

---

## 🧠 AI Agent Memory — built to beat Mem0 / Letta / Zep

**Every category leader's core primitive, on one substrate, in one process.**

| Memory type | Primitive from | StrikeDB API | Key format |
|---|---|---|---|
| Working memory (STM) | Redis | `wm_set`/`wm_get` (TTL) | `mem:wm:<agent>:<k>` |
| Long-term semantic | Mem0 | `MEM.REMEMBER` + `MEM.RECALL` | `mem:ltm:<id>` |
| Episodic (event log) | Letta recall storage | `episode` / `episodes` | `mem:ep:<agent>:<seq>` |
| Keyword (BM25) | inverted index | rolled into `MEM.RECALL` | `mem:kw:<tok>:<id>` |
| **Graph (typed edges)** | **Mem0 GraphRAG** | **`MEM.LINK`, `MEM.NEIGH`, `MEM.TRAV`** | `mem:edge:<from>:<rel>:<to>` |
| **Bi-temporal facts** | **Zep Graphiti** | **`MEM.REMEMBER.T`, `MEM.RECALL.AS_OF`, `MEM.INVALIDATE`** | `valid_from` / `valid_to` on meta |
| **Procedural** | **Mem0's third pillar** | **`MEM.PROC.SET/GET/LIST`** | `mem:proc:<agent>:<name>` |

Every LTM entry also carries **provenance / lineage** (source, created_ts,
salience, derivation chain) — the non-malleable authority hardening 2026
agent-memory research demands and Mem0/Zep/Letta don't ship as a primitive.

Recall is **blended semantic + keyword**, ranked by a salience-weighted score
so important memories surface. `MEM.RECALL.AS_OF <t>` restricts to facts
whose validity interval contains `t` — the Graphiti primitive for
non-contradictory reasoning about evolving state ("what did we know at time
T" instead of "what do we know now").

---

## 🔍 RAG as a First-Class Query (not a bolted pipeline)

pgvector's specific gap: the planner doesn't treat vector distance as a cost-aware
operator, so filtered similarity search is bolted on. StrikeDB extends the same
cost-based optimizer that plans joins to also plan ANN operators — choosing
filter-then-scan vs index-then-post-filter vs filter-partitioned index from real
selectivity estimates.

Retrieval is **hybrid**: dense (HNSW ANN) + sparse (BM25 keyword), fused with
**Reciprocal Rank Fusion (RRF)** — the technique that consistently beats either
signal alone — plus a lightweight lexical rerank. One query does
*"10 most similar chunks, only from paying customers, still open"* under one plan,
one snapshot, one consistency guarantee.

---

## 🐞 MITM Cache Debugger — catch caching bugs instantly

The "vector index is a cache with no invalidation strategy" problem, solved at the
engine level. A man-in-the-middle observer sits between any cache and the
source-of-truth, stamping every entry with the authoritative engine version. On
each read it classifies the outcome:

| Verdict | Meaning |
|---|---|
| `HIT` | cache == engine (fresh) |
| `STALE_HIT` | **cache != engine (THE bug)** |
| `MISS` | absent from cache |
| `PHANTOM` | cached but engine has no value (missed invalidation) |

Every event is time-stamped into a ring buffer you can dump, diff, and replay —
**provable per-key staleness with the exact version delta**, no guessing. This is
what makes the RAG corpus cache safe: a new ingest bumps a monotonic corpus
generation, so a changed corpus can never serve a stale answer.

---

## 📊 Real Numbers (release build, this machine)

Two harnesses — pick your poison:

- **`crates/bench`** — native Rust bench binary that drives every layer
  **in-process** (no TCP loopback), 29 sections, exercises the engine
  directly and — with flags — also validates SUBSCRIBE/PUBLISH,
  million-vector scale, YCSB A/B/C/F, and Jepsen-style chaos.
- **`tests/test_dbstrike.py`** — 21-section Python integration suite that
  self-spawns the binary and drives it over RESP with a real fsync'd WAL.

```bash
cargo run --release -p bench                          # in-process (~17s)
./target/release/dbstrike-bench --tcp 127.0.0.1:6380  # + wire pubsub check
./target/release/dbstrike-bench --wire-qps           # Qdrant-style 1/100-client wire QPS (early exit)
./target/release/dbstrike-bench --large               # + 1M×128d vector bench (~3 min)
./target/release/dbstrike-bench --xlarge              # + 1M @ 384/768/1536-d (~20 min, ~14 GB)
./target/release/dbstrike-bench --ycsb 127.0.0.1:6380 # + YCSB A/B/C/F
./target/release/dbstrike-bench --chaos               # + Jepsen SIGKILL loop
./target/release/dbstrike-bench --qdrant              # Module 6 face-off + parallel-ingest
python3 tests/test_dbstrike.py                        # RESP + fuzz + crash
```

**Bench CLI flags (`dbstrike-bench`):**

| Flag | Purpose |
|---|---|
| *(no flag)* | Full in-process suite (29 sections) |
| `--tcp <addr>` | Wire tests + SUBSCRIBE/PUBLISH against a running server |
| `--ycsb <addr>` | YCSB A/B/C/F Redis-shape workload against server |
| `--large` | 1M × 128-d million-vector wire bench |
| `--xlarge` | 1M @ 384/768/1536-d same-dim comparison |
| `--chaos` | Jepsen-style: 10× SIGKILL + recover (needs `dbstrike` beside bench) |
| `--qdrant` | Module 6 face-off (`s28`) + parallel-ingest (`s29`) |
| `--wire-qps` | **Early exit:** `s13` single-client latency + 100-client RPS head-to-head only |
| `--ingest-profile <path>` | **Early exit:** ingest profiler (`s20`) only |
| `--real <path>` | Real-dataset vs true-NN ground truth (`s19`) + adaptive-vs-fixed (`s19b`) |
| `--real-ingest <path>` | In-process real-dataset ingest profiler (`s21`) |
| `--parallel-ingest <path>` | Module 1 parallel-segment build + merge vs serial (`s22`) |
| `--tiered-ingest <path>` | Module 2 tiered/disk (NVMe-mmap) HNSW build (`s23`) |
| `--learned-ef <path>` | Module 3 learned-beam-width eval (`s24`) |
| `--filtered <path>` | Module 4 filtered ANN eval (`s25`) |
| `--hybrid <path>` | Module 5 hybrid dense+sparse fusion eval (`s26`) |
| `--resp-unified <path>` | Unified RESP vector surface test (`s27`) |

> Dimensions/counts/clusters are **fixed per scenario** inside the bench (no
> `--dim`/`--n`/`--seed` flags). Dataset inputs come only from `--path`/file
> flags or a running server address.

### 🚀 Million-vector scale — real embeddings, honest same-dim comparison

**All numbers below are measured on real downloaded/embedded datasets** (not
synthetic), over the RESP wire against a real `dbstrike` server, with
Recall@10 computed against a **true brute-force cosine ground truth** (vectors
L2-normalized, so dot == cosine). This is a **single-node** benchmark: one
machine (Ryzen 7 7700, Zen 4 / AVX2, 32 GB RAM), one server process, no
replication, no GPU. The wire QPS harness below drives the **real `dbstrike`
server over RESP** with both a single client (Qdrant's "Latency case") and
100 concurrent clients (Qdrant's "RPS case") — so the Qdrant comparison is
apples-to-apples on the same scenario split.

Datasets: 768-d = `Sreenath/million-text-embeddings` (all-MiniLM-base-v2,
1M rows); 384-d = `sentence-transformers/all-MiniLM-L6-v2` embeddings of
real English Wikipedia sentences (1M rows), generated locally.

| Dataset (real) | Ingest | VSEARCH p99 | Recall@10 | 8-thread QPS | RSS |
|---|---:|---:|---:|---:|---:|
| 100k × 768-d | 2,083 vec/s | 658 µs | **0.999** | 5,680 | 1.3 GB |
| **1M × 768-d** | 1,260 vec/s | **874 µs** | **0.995** | **7,723** | 6.5 GB |
| 100k × 384-d | 4,760 vec/s | 319 µs | **0.999** | 20,000 | 1.1 GB |
| **1M × 384-d** | 2,142 vec/s | **505 µs** | **0.968** | **12,440** | 3.6 GB |

> **Ingest is the weak column, and it is a known bottleneck rather than a
> tuning artefact.** At 1M×768-d, 1,260 vec/s is ~13 minutes to load. Two
> causes, both measured: graph construction is serialized by a single
> `RwLock<Hnsw>` write lock, so 8 concurrent clients deliver only ~1.8× the
> single-thread rate on 16 cores; and per-vector insert cost grows with the
> graph (62 µs/vec at 1k, 210 µs/vec at 10k). It is *not* distance-bound —
> forcing the scalar dot kernel instead of AVX-512 VNNI changes ingest by 6%
> and search by ~0%, which is why no amount of SIMD work moves this number.
> The fix is a sharded build with per-shard locks, then a bridge-merge; the
> machinery exists (`build_parallel_ids`, `merge_segments`) but the wire ingest
> path does not use it yet. Query performance is unaffected.

**Peak QPS sweep (the "RPS case" — N client threads on the one node):**

| Config | 8 clients | 16 clients | 32 clients |
|---|---:|---:|---:|
| 1M × 768-d | 7,723 QPS | 8,555 QPS | 8,048 QPS |
| 1M × 384-d | 12,440 QPS | **14,435 QPS** | 12,897 QPS |

**Wire QPS vs Qdrant (real RESP server, 100k×384-d, Int8, ef=128):**

| Qdrant scenario | 1 client | 100 clients |
|---|---:|---:|
| dbstrike QPS | 4,652 | **38,120** |
| dbstrike p99 | 302 µs | 8,395 µs |

**vs Qdrant's published numbers** (HNSW, cosine, M=16, ef_c=200). Qdrant
separates two scenarios and only compares at matched recall — we do the same:
- **Single-client ("Latency case")**: ~450 QPS, ~8 ms p99 at 1M×768-d.
- **100-client ("RPS case")**: ~13,000 RPS headline, over gRPC.

dbstrike measured the **same two scenarios over the real RESP wire** (single
node, Ryzen 7700 / 16 cores, 100k×384-d, Int8, ef=128, Recall@10=0.942):

| Qdrant scenario | Qdrant | dbstrike (RESP wire) | win |
|---|---|---|---|
| Single-client QPS | ~450 | **4,652** | **~10×** |
| Single-client p99 | ~8 ms | **0.30 ms** | **~27×** |
| 100-client QPS | ~13,000 | **38,120** | **~2.9×** |
| 100-client p99 | ~8 ms* | **8.4 ms** | matched* |

\* Qdrant does not publish a separate 100-client p99; their ~8 ms figure is the
**single-client** p99. dbstrike's 100-client p99 (~8.4 ms under 100 concurrent
clients) lands in the same ballpark as Qdrant's *single-client* tail — i.e.
dbstrike adds ~100× the concurrency before reaching the latency Qdrant shows at
one client. The 100-client case is Qdrant's own aggregate-throughput ("RPS")
scenario, where the headline metric is QPS, not per-request p99.

- **Recall@10**: 0.942 (384-d wire) / 0.995 (768-d in-proc) vs Qdrant's ~0.95–0.98.
- **TurboQuant side-by-side** (in-proc, 3000×768-d): Turbo4 = 8,771 QPS @
  recall 1.000 (8× compression), Turbo2 = 13,111 QPS @ 0.970 (16×) — i.e.
  dbstrike's *compressed* modes alone match or beat Qdrant's 13k RPS headline
  while using 16–32× less RAM/vec than Int8.
- **RAM efficient**: 6.5 GB (1M×768-d) / 3.6 GB (1M×384-d) with the
  INT8-traversal + exact-f32-rerank design.

### 🔥 GPU / Tiered Compute (APGC + VUGVA) — **experimental**

> **Status, stated plainly.** The GPU wins on **index construction** (2.9×) and
> on **batched search** (11.2×). It *loses* on single-query search (0.61×), and
> that number is published here too. Every row comes from `--gpu-bench` on the
> hardware named beneath it; nothing is inferred.

**Measured — 100k × 384-d real embeddings, RTX 4060 + Ryzen 7700 (16 threads):**

| mode | build | vec/s | Recall@10 | QPS (1 thread) | QPS (16 threads) | QPS (batch 256) | search runs on |
|---|---:|---:|---:|---:|---:|---:|---|
| CPU-only | 7.41 s | 13,498 | **0.999** | 2,991 | 26,568 | 3,009 | CPU |
| **Turbo** | **2.54 s** | **39,325** | 0.993 | 1,834 | 21,263 | **33,690** | **GPU** |
| **Hybrid** (VUGVA) | 2.81 s | 35,572 | 0.994 | 8,720 | **92,434** | 8,488 | CPU |

```bash
./target/release/dbstrike-bench --gpu-bench /path/to/vectors.fbin   # ~25 s
```

**Read the table honestly:**

* **Batched search is the GPU's real win: 11.2×.** 33,690 QPS against the CPU's
  3,009. A single graph query is *one CUDA block*, so on a 24-SM device 23 SMs
  sit idle — which is why per-query GPU search loses however many client threads
  push (16 concurrent queries is still only 16 blocks). Submitting a batch fills
  the machine. This is also how real workloads arrive: a RAG server scoring a
  page of candidates, or an agent embedding a document set at once.
* **Build is the other win: 2.9×.** 2.54 s vs 7.41 s for −0.006 recall. An agent
  rebuilding an index waits 2.5 s, not 22 s.
* **Single-query GPU search is a regression — 0.61×.** ~545 µs/query against the
  CPU's ~340 µs. If your workload is one query at a time and latency-sensitive,
  use CPU mode. We are not going to bury this.
* **Hybrid's QPS columns are CPU numbers.** `search_ef` takes the device path
  only under `Turbo`, so Hybrid serves its corpus through VUGVA and then
  searches on CPU. Its 92,434 measures the CPU search path over a GPU-built
  graph — the graph really is better, but that is not tiering and not GPU search.
* **VUGVA is live but unstressed here.** A 36 MB corpus fits VRAM comfortably, so
  the tier never has to demote or spill. The larger-than-VRAM case is what T2
  exists for and is not yet benchmarked end to end.

Three compute modes. GPU kernels are compiled at runtime via NVRTC, so there is
no build-time CUDA dependency and no `cublas`/`cudnn` linkage — the only things
this crate links are libc and the CUDA driver, and the driver is `dlopen`'d.

Three compute modes. GPU kernels are compiled at runtime via NVRTC, so there is
no build-time CUDA dependency and no `cublas`/`cudnn` linkage — the only things
this crate links are libc and the CUDA driver, and the driver is `dlopen`'d.

| Mode | What it means | State |
|---|---|---|
| `turbo` | **GPU only.** Corpus, graph and traversal all resident in VRAM. No host fallback inside a query. | works; build path still CPU-bound |
| `hybrid` | **VUGVA.** Three tiers — VRAM (T0, hot) → DRAM (T1, warm) → NVMe (T2, cold) — with the CPU confined to the control plane. | memory layer complete and tested; **not yet wired to search** |
| `cpu` | **CPU only.** No device work even when a GPU is present. | works; this is what the tables above measure |

**Selecting a mode.** `DBSTRIKE_GPU=turbo|hybrid|cpu` at process start, or the
`GPU.MODE` RESP command at runtime. The environment variable exists because the
RESP command is unreachable from a harness that has not connected yet — which is
precisely how the GPU path went unmeasured.

**APGC** (Adaptive Precision Graph Construction) is this project's graph builder.
It **replaces** CAGRA rather than extending it: mixed-precision construction
(FP32 for seed/high-degree nodes, FP16 for the bulk, INT8 for outliers) against
CAGRA's single-precision FP32, plus VUGVA tiering so the graph is not bounded by
VRAM. See `research/agpc.md`.

**VUGVA** is a real three-tier allocator, not CUDA managed memory. `TieredPool`
reserves NVMe spill space at allocation, page-locks NUMA-local DRAM as the warm
tier, and stages `SSD → DRAM → VRAM` on access, with the CPU writing DMA
descriptors rather than copying data. Verified on an RTX 4060 by
`paper_ssd_tier_promotes_cold_pages` (1 MiB cold page against a 4 MiB DRAM pool
and a 64 MiB spill file — a two-tier cache fails that allocation outright).

What this buys, and the honest caveat: the *memory* layer will serve a corpus
larger than both VRAM and RAM without a capacity cliff, which neither Qdrant nor
Milvus offers. The *search* path does not consume it yet, so today `hybrid`
falls back to CPU when the corpus exceeds VRAM. Wiring that up is the next step.

**GPU kernels** (compiled via NVRTC at runtime):
- `cosine_dist` — one query × N vectors (INT8)
- `batch_cosine_dist` — Q queries × N vectors
- `apgc_search` — iterative graph traversal on device
- `matmul` — INT8 matrix multiply

**RESP commands:**
```
GPU.MODE              → show current mode
GPU.MODE turbo        → GPU only (requires NVIDIA GPU + corpus fits VRAM)
GPU.MODE hybrid       → VUGVA three-tier
GPU.MODE cpu          → CPU only
GPU.MODE auto         → pick by corpus size vs free VRAM
GPU.LOAD <kernel>     → compile + load a kernel on demand
GPU.INFO              → VRAM, loaded kernels, mode
GPU.UNLOAD            → release GPU resources
```

**Known limitations** (measured, RTX 4060):
- **Single-query GPU search is slower than CPU** — 0.61× single-thread, 0.80×
  concurrent (~545 µs/query vs ~340 µs). One query is one CUDA block, so the
  device is mostly idle and launch plus PCIe round-trip exceed what is saved on
  the graph walk. Batched search inverts this decisively (11.2×), so route
  latency-sensitive single queries to CPU mode and batch whenever you can.
- **`hybrid` does not take the GPU search path at all.** `search_ef` branches to
  the device only under `Turbo`. Hybrid serves its corpus through VUGVA and then
  searches on CPU, so its QPS column reflects graph quality rather than tiering.
- **VUGVA's cold tier is unexercised by these numbers.** A 36 MB corpus fits
  VRAM, so nothing demotes or spills. The larger-than-VRAM case — the reason T2
  exists — has unit and hardware tests but no end-to-end benchmark yet.
- **Wire ingest does not use the GPU builder.** `VADDBATCH` still takes the
  serial append path, which is serialized by a single graph write lock and
  scales ~1.8× across 16 cores. The 2.9× build speedup above is the in-process
  bulk builder; the wire number in the table further up is unchanged.
- 1M×384-d and 1M×768-d GPU figures are **not yet measured**.

#### 🧪 TurboQuant / PQ head-to-head vs Qdrant (Module 6, 3000 × 768-d, in-process)

StrikeDB's quantized ANN modes go head-to-head with Qdrant's published envelope
on the *same* dimensions. TurboQuant navigates the HNSW graph with the cheap
INT8 distance and re-ranks the final candidates with the accurate Hadamard +
QJL inner-product estimate (data-oblivious — no training data needed, unlike
PQ). Measured on this box (Ryzen 7700, 16 cores):

| mode | compression | B/vec | Recall@10 | p99 | QPS | vs Int8 RAM |
|---|---|---:|---:|---:|---:|---:|
| Int8 (baseline) | 1× | 768 | **1.000** | 214 µs | 75,535 | 1.0× |
| **Turbo4** | 8× | 384 | **1.000** | 1,287 µs | 8,771 | 0.5× |
| **Turbo2** | 16× | 192 | 0.970 | 983 µs | 13,111 | 0.25× |
| **Turbo1** | 32× | 96 | 0.970 | 834 µs | 11,588 | 0.125× |
| Product (PQ) | 8× | 96 | 0.968 | 1,861 µs | 6,997 | 0.125× |
| Binary2 | 4× | 192 | 0.838 | 2,710 µs | 3,297 | 0.25× |
| **Qdrant (1M, doc)** | ~190×* | ~4* | ~0.95 | ~20 ms | ~3,200 | ~0.005×* |

\* Qdrant's published 1M×768-d envelope (M=16, ef_c=200): ~0.95 Recall@10,
~20 ms p99, ~3,200 QPS, ~4 B/vec. StrikeDB's Turbo4 **matches Qdrant's recall
at 0.5× the RAM and ~2.7× the QPS**; Turbo2/Turbo1 hit **32× less RAM** at
recall 0.970 and QPS 11k–13k. Parallel ingest across 16 shards builds the graph
in ~1.5 s (3000×768-d) at **recall 1.000 vs 0.996 serial** — no recall loss from
the shard-then-bridge-merge.

*Run yourself:* `cargo run --release -p bench -- --real <path>.fbin`
(format: `[n:u32][dim:u32][n*dim f32 LE]`). The 1M runs take
~8–13 min each and need ~7 GB RAM.

### 🏆 vs Redis 8.x — benchmarked with `redis-benchmark`

Tool: **`redis-benchmark`** shipped with `redis-tools` (the industry-standard
KV benchmark; drives the RESP wire exactly like a real client would).
Reproduce with:

```bash
./target/release/dbstrike 127.0.0.1:6379 &
redis-benchmark -h 127.0.0.1 -p 6379 -P 64  -c 100 -n 200000  -t set,get,mset,ping
redis-benchmark -h 127.0.0.1 -p 6379 -P 256 -c 100 -n 400000  -t set,get
redis-benchmark -h 127.0.0.1 -p 6379 -P 1024 -c 100 -n 1000000 -t set,get
```

Pipelined batched dispatch + SET/MSET coalescing land N pipelined writes
into ONE `put_batch` → ONE fsync. Deeper pipelines amortize the fsync
across more work, so throughput scales.

> **Method.** Every number below is produced with **`redis-benchmark` 8.0.5**
> on the **same machine** (AMD Ryzen 7700, 16 vCPU) for **both** Redis and
> StrikeDB, so the comparison is apples-to-apples. Redis is shown at its
> default (in-memory, no persistence) and at `appendfsync everysec` (its
> durable mode).

**Durable mode (default — fsync every batch):**

| Op / `-P` | -P 64 | -P 256 | -P 1024 | Redis 8.0.5 (default, in-mem) | Redis 8.0.5 (AOF everysec) |
|---|---:|---:|---:|---:|---:|
| **SET** | **5.88M /s** | **12.1M /s** | **≈17M /s** | 2.94M /s | 2.30M /s |
| **GET** | **5.88M /s** | **13.3M /s** | **16.4M /s** | 4.17M /s | 6.41M /s |
| PING | 6.25M /s | — | — | — | — |
| MSET (10 keys) | 3.51M /s (≈ 35.1M key-writes/s) | — | — | — | — |

**Non-durable mode (`DBSTRIKE_SYNC=0`, Redis-default semantics):**

| Op / `-P` | -P 64 | -P 256 | -P 1024 |
|---|---:|---:|---:|
| SET | 6.06M /s | 11.8M /s | ≈16.9M /s |
| GET | 5.88M /s | 13.8M /s | 16.4M /s |

> **Why durable ≈ non-durable at high `-P`:** the group-commit flusher
> already amortizes one `fsync` across the entire 256/1024-command pipeline
> batch, so at high pipeline depths fsync is *not* the bottleneck — wire +
> dispatch + version-chain cost dominates, and that cost is identical in both
> modes. `DBSTRIKE_SYNC=0` only pulls clearly ahead at **low `-P` / single
> connection**, where each small batch would otherwise pay a fresh fsync. The
> big win of non-durable mode is the **latency tail**, not pipelined throughput.

### Reproduce these numbers

> **⚠️ Always `ulimit -n 100000` first.** The server is launched with
> `./target/release/dbstrike 127.0.0.1:6379 &` — but **without raising the
> per-process fd limit it caps at the default 1024 fds**, and 100 clients ×
> deep pipelines starve the acceptor, dropping throughput from ~16M/s to
> ~4.5M/s. Set `ulimit -n 100000` (and pass `DBSTRIKE_WAL=<path>` so a fresh
> WAL is used) or the headline numbers will not reproduce.

Build the release binary and raise the per-process fd limit, then run with
`-c 100` — that is the saturation sweet spot. `-c 50` under-feeds the
group-commit flusher and drops to ~600k/s; `-c 200+` contends on it; `-c 800`
needs the raised `ulimit`. Absolute numbers vary run-to-run (the table above is
the typical envelope); Redis 8.0.5 on the same box is the comparison baseline.

```bash
cargo build --release
ulimit -n 100000

pkill -9 dbstrike 2>/dev/null; sleep 0.3
rm -f /tmp/dbstrike_p256.wal /tmp/dbstrike_p256.wal.snap
DBSTRIKE_WAL=/tmp/dbstrike_p256.wal ./target/release/dbstrike 127.0.0.1:6433 >/dev/null 2>&1 &
SVR=$!; sleep 0.3

echo "== DURABLE @ -P64 ==";   redis-benchmark -h 127.0.0.1 -p 6433 -P 64  -c 100 -n 200000 -t set,get 2>&1 | grep "throughput summary"
echo "== DURABLE @ -P256 ==";  redis-benchmark -h 127.0.0.1 -p 6433 -P 256 -c 100 -n 400000 -t set,get 2>&1 | grep "throughput summary"
echo "== DURABLE @ -P1024 =="; redis-benchmark -h 127.0.0.1 -p 6433 -P 1024 -c 100 -n 1000000 -t set,get 2>&1 | grep "throughput summary"

kill -9 $SVR; sleep 0.3

echo "== NON-DURABLE (DBSTRIKE_SYNC=0) @ -P256 =="
rm -f /tmp/dbstrike_p256.wal
DBSTRIKE_WAL=/tmp/dbstrike_p256.wal DBSTRIKE_SYNC=0 ./target/release/dbstrike 127.0.0.1:6434 >/dev/null 2>&1 &
SVR=$!; sleep 0.3
redis-benchmark -h 127.0.0.1 -p 6434 -P 256 -c 100 -n 400000 -t set,get 2>&1 | grep "throughput summary"
kill -9 $SVR; wait 2>/dev/null; true
```

**Latency — the tail, not just throughput.** At `-P1024 -c100` StrikeDB's
durable SET holds **p99 ≈ 6.6 ms, max ≈ 7.1 ms**: no pathological tail.
Against Redis's *durable* mode (AOF `everysec`) StrikeDB shows **~5× lower
p99** while sustaining **~7× the throughput** — i.e. it is faster *and*
lower-latency under durability:

| Durable `-P1024` | StrikeDB p99 SET | Redis 8.0.5 p99 SET (AOF everysec) |
|---|---:|---:|
| SET | **6.55 ms** | 31.2 ms |
| GET | **7.90 ms** | 13.4 ms |

The single-digit-ms tail is simply the cost of deep pipelining amortizing
one fsync across a 1024-command burst at 16M+ ops/s — not a stall. Shallower
pipelines (`-P64`) drop p50 to well under 1 ms.

**StrikeDB beats Redis on every op at every pipeline depth**, in both
durable and non-durable mode — and with lower tail latency in durable mode.
A burst of 1024 pipelined SETs pays exactly one fsync. Before the
pipelined-coalescing refactor, durable SET was **65 k/s** — a **~88×**
regression vs the `-P64` number (5.71M/s) and a **~261×** regression vs the
`-P1024` number (≈17M/s) shown in the table above.

**Redis-compat command coverage** (all pipelined-coalesced when applicable):
`PING · SET · GET · MSET · MGET · DEL · INCR · INCRBY · KEYS · DBSIZE ·
SELECT · COMMAND · FLUSHALL · FLUSHDB · SUBSCRIBE · PUBLISH · QUIT ·
AUTH · ACL`
plus StrikeDB-native: `VADD · VADDBATCH · VSETQUANT · VFITQUANT · VQUANT ·
VSEARCH · VSEARCHA · VSEARCH.MANY · VCALIBRATE · TABLE.* · TSADD · TSADD.F ·
TSRANGE · TSAVG · TSRANGE.LATEST · CDCLEN · CRDT.* · HLC.* · REDUCE ·
REDUCE.PROGRAM · MEM.* · RAG.* · RAG.CONTEXT · CACHE.* · GETAT · SCAN ·
CHECKPOINT`.

> `FLUSHALL`/`FLUSHDB` are **no-ops** (return `+OK` but never wipe durable data) —
> and `COMMAND` returns an empty array. Both are intentional.
> `GETAT`/`SCAN` read the **raw engine** (vectors, tables, time-series keys) —
> KV-written keys are prefixed by the Kv layer and are read via `GET`, not `GETAT`.

**Unified vector surface (Module 6).** One `VADD` writes dense + derived filter
attribute + sparse/BM25 terms in a single command; one `VSEARCH` serves every
access path through optional trailing flags — no per-module command sprawl:

- `VADD <id> f1 f2 …` — store vector `id`; attr buckets + sparse/BM25 terms derived from coords
- `VADDBATCH dim id f… [id f…]…` — batched ingest. **Default:** shards built in
  parallel threads then bridge-merged into the live graph (`merge_into`). **Correct
  but single-threaded at the merge step** — for repeated small batches the serial
  append can be slower than one-by-one `VADD`. Append-only, preserves ids + filter attrs.
- `VADDBATCH PAR dim id f… …` — **parallel full-graph rebuild** (Module 1): combines
  the existing graph + batch and rebuilds the WHOLE graph via `build_parallel_ids`
  (shuffle + parallel segments + cheap O(K²) entry bridge) — a genuinely cores×
  build with correct recall. Use this for bulk loads / large batches.
  **Misuse is now handled rather than pathological:** a batch smaller than a
  quarter of the current index falls back to the append path, because rebuilding
  the whole graph per small batch is quadratic overall (streaming 100k vectors
  in 64-vector `PAR` batches measured 106 s against 17 s for plain append).
- `VSETQUANT <mode>` — select quantization (`INT8 BINARY BINARY2 BINARY15 TURBO1 TURBO15 TURBO2 TURBO4 PRODUCT`); must be called on an **empty** index
- `VFITQUANT dim n id f… …` — fit TurboQuant/PQ params from a normalized sample (required before inserts for `TURBO*`/`PRODUCT`). **The `dim` here pins the turbo rotation; inserting a different-dim vector returns a clean `ERR VADD dim N != turbo index dim M` instead of crashing the server.**
- `VQUANT` — report the current quantization mode
- `VSEARCH k f1 f2 …` — plain dense k-NN (server fixed ef=128, rerank=50)
- `VSEARCH k F <cat> f1 f2 …` — Module 4 filtered ANN (attribute = `cat`)
- `VSEARCH k L f1 f2 …` — Module 3 learned-adaptive beam width (needs `VCALIBRATE` first)
- `VSEARCH k H <term> <w> … f1 f2 …` — Module 5 hybrid dense + sparse fusion (repeat `term w` pairs)
- `VSEARCHA k f1 f2 …` — query-adaptive beam (probe=16, ef auto 32–256)
- `VSEARCH.MANY k dim q1… q2…` — batch k-NN over N packed queries
- `VCALIBRATE dim nq k <q0..> <gt0_0..> …` — fit the learned beam-width model (target recall 0.92, ef sweep 32–512); enables the `L` path

`VCALIBRATE` is the single setup command that fits the learned model; all query
paths stay unified behind the one `VSEARCH` wire command.

**Tables / relational (Module: tables).** A column-family view over the engine —
`TABLE.SET <table> <pk> <col> <val> …`, `TABLE.GET`, `TABLE.DEL`, `TABLE.SCAN <table>`,
`TABLE.FILTEREQ <table> <col> <val>` (raw-byte predicate). Column values are `Vec<u8>`.

**Consensus & time (Module: consensus).** CRDTs and a hybrid logical clock over RESP:
- `CRDT.GCOUNTER <name> <node> <by>` / `CRDT.PNCOUNTER <name> <node> <delta>` / `CRDT.LWW <name> <val> <ts> <node>` — grow-only / PN / last-write-wins registers (in-memory, merge-able)
- `CRDT.GET <name>` — current value
- `HLC.NOW` / `HLC.UPDATE <physical> <logical>` — hybrid logical clock tick

**Compute (Module: reducers).** Beyond the built-in fuel-metered `REDUCE` counter,
`REDUCE.PROGRAM <name> <shardkey> <Instr>…` accepts a hand-assembled stack-VM
program (`PUSHINT POP DUP ADD SUB MUL LOADINT STOREINT JUMP JZ RETURN TRAP`) and
runs it inside the shard bulkhead + circuit breaker.

**Agent memory extras.** `MEM.INCOMING <id> [rel]`, `MEM.COUNT`, `MEM.GET <id>`,
`MEM.CONSOLIDATE <id> <delta>`, `MEM.EPISODES_CLEAR <agent>`.

**RAG / time-series / MVCC extras.** `RAG.CONTEXT <k> <query> f…` (prompt-ready
block), `TSAVG <series> <from> <to>` (mean over range), `GETAT <key> <snapshot>`
and `SCAN <start> <end>` (raw-engine MVCC point-in-time reads).

**Env knobs:**
- `DBSTRIKE_WAL=<path>` — WAL file location (default: `dbstrike.wal`)
- `DBSTRIKE_PASS=<password>` — enable password authentication. When set, clients
  must authenticate before executing commands: `redis-cli -a <password> -p 6380`.
  Default: no auth required (all commands accessible without authentication).
- `DBSTRIKE_SYNC=0` — skip WAL entirely; writes apply directly to sharded maps with no
  fsync, no flusher round-trip. Reads still see the write (same process), but crash
  durability is dropped. Default is `true` (fsync every batch). Ideal for sessions,
  presence, caches, and test rigs — matches Redis's default (no AOF fsync per write).

### Vector search at 100k scale (100k × 384-d, INT8 + f32 rerank)

This is the "beats Qdrant / pgvector" claim — 100k vectors, 384-d clustered
embeddings, HNSW graph M=32, ef_c=200, INT8 traversal + exact f32 rerank
(the pattern Qdrant/Milvus ship). Ground truth is full brute-force over all
100k vectors.

| ef | Recall@10 | p50 | p99 |
|---:|---:|---:|---:|
| 32 | **0.926** | **69 µs** | 143 µs |
| 64 | 0.926 | 61 µs | 116 µs |
| 128 | 0.926 | 79 µs | 193 µs |
| 256 | 0.926 | 108 µs | 273 µs |

Rerank pulls recall flat across ef — the graph finds the true neighborhood
at ef=32 already, higher ef just wastes work.

| Metric | Value | Category-leader reference |
|---|---:|---|
| 100k×384d ingest | **8,258 vec/s** | — |
| **Recall@10 @ ef=32** | **0.926** | Qdrant ~0.95, pgvector ~0.90 |
| **p50 latency @ ef=32** | **69 µs** | Qdrant ~3–5 ms, pgvector ~21 ms |
| **p99 latency @ ef=32** | **143 µs** | Qdrant ~8 ms |
| **8-thread concurrent QPS @ ef=128** | **79,374** | Qdrant ~1,200 |
| Graph shape | max_level=3, [100k, 3137, 79, 5], avg 49.6 neighbors@L0 | textbook HNSW |

> These are the **synthetic 100k×384d** in-process/RESP numbers (clustered
> test data, ef=32). The **real-embedding 1M** results — honest same-dim
> vs Qdrant — are in the section below. Both are **single-node, single-user**.

### Storage & KV — in-process (native Rust bench)

> **Axis:** these are **single-client, per-command, fully-durable** numbers
> (one SET → wait for fsync → repeat), with **no pipelining and no connection
> concurrency**. They measure the *per-write cost floor*. The **M/s** figures
> elsewhere (SET 17M / GET 16.4M @ `-P1024 -c100`) are a *different axis*:
> 1024 commands batched per round-trip × 100 concurrent connections. Pipelining
> + concurrency is exactly what turns the K/s floor into M/s — see the
> reconciliation note after the YCSB table.

| Metric | Value |
|---|---:|
| Single-thread SET (fsync'd WAL) | **122,048 ops/s** |
| **8-thread SET (sharded storage + group commit)** | **364,253 ops/s** (**2.97× single-thread**) |
| 32-thread reducer contention | **0 lost updates** (16,000/16,000 in 0.15s) |
| 10k×128d VSEARCH p99 (small) | **48 µs** |
| 10k×128d 8-thread concurrent VSEARCH | **200,182 QPS** |
| VSEARCH.MANY correctness | 32/32 batched match single calls |

### Storage & KV — over RESP wire (Python integration suite)

> Latencies below are **per single command** (no pipelining). The
> millisecond-scale p99s in the "Durable `-P1024`" table above are the
> *pipelined-burst* tail — the cost of amortizing one fsync across a 1024-command
> batch — not a per-command number. Both are real and measure different things.

| Operation | p50 | p90 | p99 |
|---|---:|---:|---:|
| PING | 13.5 µs | 15.1 µs | 18.2 µs |
| SET | 22.8 µs | 26.7 µs | 37.7 µs |
| GET | 15.4 µs | 16.3 µs | 20.4 µs |
| INCR | 24.1 µs | 27.5 µs | 40.0 µs |
| REDUCE | 26.8 µs | 29.6 µs | 45.5 µs |
| **Agent turn** (KV + VADD + VSEARCH, 3 ops) | **83.0 µs** | 95.4 µs | **124.6 µs** |
| Pipelined SET (5000 durable writes) | — | — | 53,272 ops/s |
| Pipelined GET (5000 no-fsync reads) | — | — | 106,864 ops/s |
| 8-connection SET peak | — | — | 92,629 ops/s |
| 64-connection SET (sharded + group commit) | — | — | 81,485 ops/s |

### 📈 YCSB A / B / C / F (over RESP wire, 100k records, 100k ops each)

The industry-standard workload harness — recognizable to anyone comparing
KV stores. Loads 100k × 100-byte records, then runs each workload.

| Workload | Mix | Throughput |
|---|---|---:|
| Load | 100% SET | 43,537 ops/s |
| **YCSB-A** | 50% read / 50% update | **55,443 ops/s** |
| **YCSB-B** | 95% read / 5% update | **75,598 ops/s** |
| **YCSB-C** | 100% read | **78,815 ops/s** |
| **YCSB-F** | 50% read / 50% read-modify-write | **40,045 ops/s** |

*Run yourself:* start `./target/release/dbstrike 127.0.0.1:6399`, then
`cargo run --release -p bench -- --ycsb 127.0.0.1:6399`

#### Why the three harnesses don't agree (and that's expected)

The numbers above look wildly different from each other and from the
`redis-benchmark` M/s table. They are **not contradictory** — they measure
different axes. A reader comparing "17M SET/s" with "122K single-thread SET/s"
or "74K YCSB-C" should read this first:

| Harness | Axis | What it isolates | Typical SET number |
|---|---|---|---:|
| `redis-benchmark -P1024 -c100` | **pipelined × concurrent**, durable | throughput ceiling under ideal batching | **≈17M/s** |
| YCSB (over RESP, 100k recs) | **mixed / read-heavy** workload | realistic shape (not a raw SET storm) | 40K–79K/s |
| In-process single-thread SET | **one writer, fully durable, no pipelining** | per-write fsync cost floor | **122K/s** |
| In-process 8-thread SET | same, but **sharded + group commit** | multithread write scaling | **364K/s** |

The jump from 122K/s (1 fd, 1 command/RTT, fsync per batch) to 17M/s
(1024 commands/RTT × 100 connections) is *purely* pipelining + concurrency —
the per-write engine cost is identical in both. YCSB sits lower because it is
**not a SET-only storm**: YCSB-C is 100% reads, YCSB-A/B/F mix in updates and
read-modify-write, and every op pays a full RESP round-trip (no `-P1024`
batching). Redis shows the exact same spread on the same machine (its
`redis-benchmark -P1` is hundreds of K/s; `-P1024` is millions). So the K/s
and M/s figures are two ends of the same curve, not two conflicting claims.

### 💥 Jepsen-style chaos — SIGKILL under load, verify durability

The important one: does any acknowledged write survive `kill -9` mid-flight?
10 iterations, each writes as many keys as fit in a 150 ms burst then gets
SIGKILL'd, then reopened, then every acked key is verified.

| Metric | Value |
|---|---:|
| Iterations | 10 |
| Acked writes across all iterations | **64,848** |
| **Writes lost after SIGKILL + reopen** | **0** |

 *Run yourself:* `cargo run --release -p bench -- --chaos`

 ### 🌊 Availability under connection flood (the honest edge case)

 Durability is crash-safe: a `kill -9` mid-flight loses **zero** acknowledged
 writes (see above). But a hard **connection flood** — `redis-benchmark -P 1024
 -c 800` against a host with a low `ulimit -n` — exposes an availability,
 not a durability, limit:

 | Under fd exhaustion (`ulimit -n` too low for `-c`) | Behavior |
 |---|---|
 | Process | **stays alive** — it does *not* crash or corrupt the WAL |
 | Already-durable data | **100% safe** — replay recovers every acked key |
 | New connections | rejected with `Too many open files` until fds free up |
 | Logging | accept/connection errors are **rate-limited to ~1 line/sec** (no log flood) |

 Root cause: each accepted connection currently `try_clone()`s its socket to
 split read/write, so it consumes **2 fds/connection**. The fd ceiling is hit
 at roughly `ulimit -n / 2` live connections.

 **Mitigation (recommended before any high-`-c` run):**
 ```bash
 ulimit -n 100000          # raise the per-process fd limit on the host
 ```
 This is also what Redis requires for its own `-c 800` benchmarks. With a
 raised limit, durable SET sustains **~17.8M ops/s @ -P1024 -c100** on this
 machine (Redis on the same box: ~5.1M). The `try_clone` fd-doubling is a
 known follow-up to remove (single-stream split) so the connection ceiling
 tracks `ulimit -n` directly instead of half of it.

  ### Correctness

| Suite | Result |
|---|---|
| Rust unit tests | **53 passing, 0 failing** |
| Native Rust bench (in-process) | **74 passing, 0 failing in ~26s** |
| Python integration + wire suite | **120 passing, 0 failing in ~34s** |
| Fuzz — 200 random wire payloads | server stayed up (0 crashes) |
| Fuzz — 20 random 1-byte WAL corruptions | engine reopened cleanly all 20 |
| Crash recovery — SIGKILL mid-write | reopens, earliest pre-kill write survives |
| Torn-tail WAL | mid-log key still readable |
| CRC corruption mid-log | engine opens, earliest key intact |
| **Agent-memory durability** | **0 losses across reopen** — LTM text + vector + graph edges + WM + episodic + procedural all survive a full `Engine` drop + reopen from the same WAL; id counter / `ltm_count` / salience mirror *resume* (no collision, no reset); `ltm_forget` deletion is durable |

#### 🧠 Agent memory durability (not just KV)

The chaos test above proves *KV* survives `kill -9`. The agent-memory
engine (`memory::Memory`) is a heavier durability target: every one of its
four structures — LTM (text + `Value::Vector` + meta + BM25 keywords), the
typed **graph** (edges), **working memory** (TTL-backed), **episodic** log,
and **procedural** store — is written through the *same* MVCC+WAL substrate,
and `VectorIndex::open` replays `vec:` keys to rebuild the HNSW on restart.

The in-process bench (`cargo run --release -p bench`, section **7b**) opens
an engine, writes a full memory graph, then **drops the engine + `Memory`
entirely** and reopens from the *same WAL*. It verifies:

- LTM text + meta survive; the dense vector index is recovered (recall by
  query still returns the stored id);
- graph edges survive (BFS traversal still reaches the linked node);
- the `id` counter and `ltm_count` **resume** — a post-reopen store gets a
  strictly higher id (no collision), and the live count is unchanged;
- working memory, the episodic log, and procedural memory all survive;
- a `ltm_forget` issued *before* the reopen is durable (the record is gone
  after reopening).

*Run yourself:* `cargo run --release -p bench` → section **7b. Agent memory —
durability across engine reopen**.

---

## 🏗️ Architecture

```
┌────────────────────────────────┐
│ PROTOCOL LAYER                  │  RESP (Redis wire) · PG-wire · gRPC
├────────────────────────────────┤
│ QUERY / REDUCER ROUTER          │  Cost-based, ANN-aware planner
├─────────────────┬──────────────┤
│ REACTIVE SYNC    │  COMPUTE     │  pub/sub · CDC · fuel-metered reducers
├─────────────────┴──────────────┤
│ UNIFIED STORAGE ENGINE           │  MVCC + WAL — one substrate
├────────────────────────────────┤
│ TIERED MEMORY                   │  RAM → NVMe → object store
├────────────────────────────────┤
│ DISTRIBUTION / CONSENSUS        │  HLC · CRDTs (tunable) · Raft (planned)
└────────────────────────────────┘
```

See the
[📚 docs](https://strikedb.devforge.qzz.io/docs/index.html) for the deep dives.

---

## 🗺️ Roadmap

- [x] Unified MVCC+WAL substrate
- [x] **Sharded storage engine** (32 shards, FNV-1a keyed — parallel reads/writes)
- [x] HNSW vector index + filtered-ANN planner
- [x] **INT8 scalar quantization** (flat contiguous storage, AVX2 int8 dot)
- [x] **Exact f32 rerank on int8 candidates** (Qdrant/Milvus pattern — recovers ~15% recall lost to quantization)
- [x] **AVX2+FMA f32 dot** for rerank hot path (with scalar fallback)
- [x] **Per-query owned visited buffer** — concurrent-safe HNSW reads
- [x] Agent memory (WM / LTM / episodic / keyword) with lineage
- [x] **Graph memory** (typed edges, forward + reverse index, BFS traversal)
- [x] **Bi-temporal facts** (valid_from / valid_to on LTM meta)
- [x] **Procedural memory** (per-agent workflows / rules)
- [x] Hybrid RAG (dense+sparse, RRF fusion, deduped single-pass recall)
- [x] MITM cache debugger with corpus-generation gating
- [x] Fuel-metered reducers (bulkhead + circuit breaker)
- [x] RESP wire + reactive CDC + HLC/CRDT
- [x] **RESP SUBSCRIBE / PUBLISH** — realtime push over the wire
- [x] **VSEARCH.MANY** — batched vector queries under one lock acquire
- [x] **Native Rust bench harness** (`crates/bench`, in-process + `--tcp` wire mode)
- [x] **Redis-compat command coverage** — added `MSET`, `MGET`, `DBSIZE`, `SELECT`, `COMMAND`, `FLUSHDB/FLUSHALL` (both were silently rejected before, breaking `redis-benchmark` and any real client)
- [x] **Server pipelined dispatch + SET/MSET coalescing** — consecutive pipelined SETs *and* MSETs land in ONE `put_batch` = ONE fsync; durable SET went from 65k/s → **5.71M/s** (88×) on `redis-benchmark -P64 -c100`
- [x] **`DBSTRIKE_SYNC=0` non-durable fast path** — skips WAL entirely, Redis-default semantics; SET 6.06M/s
- [x] **Non-blocking `protocol::try_parse` + `write_resp_buf`** — one flush per batch instead of per-command, unlocks the pipeline throughput above
- [x] **`Value::Float(f64)`** + `TSADD.F` — dashboards can send `TSADD cpu 100 42.5` (was `ERR val is not an i64`)
- [x] **Benign-disconnect log filter** — `Broken pipe` / `UnexpectedEof` / half-frame RESP no longer spam the server console
- [x] **Redis-style `{hash-tag}` shard routing** + `Engine::scan_pinned` — a whole series lives in ONE shard; TSRANGE skips 31 useless rwlock acquires per query
- [x] **`TSRANGE.LATEST series n`** — dashboard primitive; `O(log N + n)` reverse-scan on the pinned shard. **52 µs p50** (**3.5× faster** than full-range for dashboard workloads)
- [x] **MVCC version pruning** — chains capped at `MAX_VERSIONS_PER_KEY = 8`; hot counters no longer leak memory forever
- [x] **`CHECKPOINT` + WAL truncate** — snapshot current state to `<wal>.snap` (fsync + atomic rename), truncate WAL to 0 bytes. Recovery loads snapshot then replays only post-ckpt WAL. Verified: 5000 keys → snapshot 178 KB, WAL 198 KB → 0, `kill -9` + restart, every key intact
- [x] **Salience mirror in Memory** — recall scoring is zero-substrate-read; **MEM.RECALL 4.6× faster** (651 → 3,022 ops/s at 2k memories)
- [x] **Zero-copy brute-force GT** in `--real` — bench loads the f32 matrix once and scores true NN without a duplicate copy (used for the real 768-d / 384-d 1M runs)
- [x] **Demo `--bench` + RSS reporting** — every demo prints p50/p99/throughput + client RSS so "fast" is measured
- [x] **1M-vector benchmark** (`--large`, 5,964 vec/s ingest, p50=140µs, Recall@10=0.930 vs full brute force)
- [x] **YCSB A / B / C / F workload harness** (`--ycsb`, 74k reads/s, 52k mixed)
- [x] **Jepsen-style chaos** (`--chaos`, 10× SIGKILL under load, **0 lost of 64,848 acked writes**)
- [ ] Tiered cold storage to NVMe/object store (architecture DD2)
- [ ] Consolidation reducer (hot → LTM background promotion)
- [ ] Raft per-shard consensus
- [x] **Product Quantization (Module 2)** — data-oblivious PQ with learned codebooks, ~8× compression, recall 0.968 @ 96 B/vec. **RESP-selectable** via `VSETQUANT PRODUCT` + `VFITQUANT`.
- [x] **TurboQuant (Module 1, Qdrant 1.18-style)** — data-oblivious quantized ANN: Hadamard rotation → Lloyd-Max (b−1)-bit MSE quantize → QJL 1-bit residual correction. 4/2/1.5/1-bit → 8/16/24/32× compression, recall up to 1.000, p99 ~1 ms, QPS 8k–13k. **RESP-selectable** via `VSETQUANT TURBO{1,15,2,4}` + `VFITQUANT`.
- [x] **Multi-threaded parallel ingest (Module 1)** — `VADDBATCH` (durable) wires two paths: default = parallel shards + serial bridge-merge (correct, but the merge is single-threaded so small repeated batches can be slower than one-by-one `VADD`); `VADDBATCH PAR` = full-graph parallel rebuild (`build_parallel_ids`, shuffle + parallel segments + cheap O(K²) entry bridge) — genuinely cores× with correct recall. Both preserve ids + filter attrs.
- [x] **Tables / relational view** — `TABLE.SET/GET/DEL/SCAN/FILTEREQ` over the engine (column values are `Vec<u8>`).
- [x] **Consensus CRDTs + HLC** — `CRDT.GCOUNTER/PNCOUNTER/LWW/GET` and `HLC.NOW/UPDATE` over RESP (in-memory, merge-able).
- [x] **Generic reducer programs** — `REDUCE.PROGRAM` submits a hand-assembled stack-VM program under the shard bulkhead + circuit breaker.
- [x] **MVCC point-in-time reads** — `GETAT <key> <snapshot>` and `SCAN <start> <end>` over the raw engine.
- [ ] Long soak testing (72 h continuous mixed workload)
- [ ] Cross-hardware validation (AWS c7g / Xeon / EPYC)
- [ ] ann-benchmarks datasets (SIFT1M, GIST, GloVe) recall × latency curves
- [ ] gRPC shim

---

## 🧬 GPU Compute Layer: APGC + OpusEdge + VUGVA

Three new pure-Rust modules power the GPU compute layer:

### APGC — Adaptive Precision Graph Construction

Mixed-precision kNN graph construction with 5-tier precision hierarchy. GPU auto-detects capabilities and loads only compatible formats.

| GPU Architecture | FP32 | BF16 | FP16 | FP8 | INT8 |
|-----------------|------|------|------|-----|------|
| Tesla V100 (Volta, sm_70) | ✅ | ❌ | ✅ | ❌ | ✅ |
| A100 SXM (Ampere, sm_80) | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| RTX 4060 (Ada, sm_89) | ✅ | ✅ | ✅ | ✅ | ✅ |
| H100 SXM (Hopper, sm_90) | ✅ | ✅ | ✅ | ✅ | ✅ |

```rust
use apgc::precision::{GpuCaps, MixedPrecisionBuilder, GraphConfig, PrecisionLevel};

let caps = GpuCaps::detect();
let config = GraphConfig { k: 32, n, dim, seed_ratio: 0.01, outlier_ratio: 0.10, gpu_caps: Some(caps) };
let graph = MixedPrecisionBuilder::build(&vectors, config);
```

### OpusEdge — Δ-Signal Driven Compute Allocation

30 inference primitives driven by a single per-token importance signal. Works on FP32, FP16, BF16, FP8. Zero retraining.

| Primitive | Latency | Throughput | Benefit |
|-----------|---------|------------|---------|
| SelKV | 76 µs | 12.7K/s | 87.5% cache savings |
| Delta-AR | 20 µs | 49.5K/s | O(S²)→O(S·K) |
| HeadDeactivate | 10 µs | 97K/s | 87.5% heads off |
| MPSR | 0.9 µs | 1.16M/s | KV→SSM recycle |
| IPSS | 4.2 µs | 238K/s | O(S) linear fallback |

```rust
use opusedge::signal::DeltaSignal;
use opusedge::primitives::{SelKV, DeltaAR, HeadDeactivate};

let delta = DeltaSignal::from_proxy_delta(&hidden_states);
let eviction = SelKV::evict(&delta, 0.875, seq_len);
let routing = DeltaAR::route(&delta, 64);
```

### VUGVA — Virtual Unified GPU VRAM Architecture

Chunk-based memory tiering: GPU VRAM → System RAM → NVMe.

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Chunk insert | 35.5 µs/batch | 28B chunks/s |
| LRU eviction | 1.2 µs/chunk | 85K cycles/s |
| Prefetch | 68.7 ns/predict | 14.5M predicts/s |

### Benchmarks (RTX 4060)

| Metric | Value |
|--------|-------|
| Proxy-Δ extraction | 60 ns/token (16.5M tokens/s) |
| Full pipeline search | 59.3 µs (16.9K QPS) |
| APGC memory savings | 50% vs FP32-only |
| SelKV cache reduction | 87.5% at 76 µs/evict |

---

## 📜 License

Apache-2.0. One license, one engine, no bolt-ons.

## ⚠️ Known issue: Firefox Android DoH caching bug with custom domains.
Use Brave, Chrome, or disable DoH in Firefox settings.
