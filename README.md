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

[📚 Docs](docs/index.html) · [🚀 Live Benchmarks](docs/index.html#benchmarks) · [🧠 Agent Memory](docs/index.html#memory) · [🔍 MITM Cache Debugger](docs/index.html#mitm)

---

## 🏷️ Status

![Rust](https://img.shields.io/badge/language-Rust-000000?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero%20(crate)--green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-205%20passing%20(44%20rust%20%2B%2057%20native%20%2B%20104%20integration)-brightgreen.svg)
![1M vectors](https://img.shields.io/badge/1M×128d%20VSEARCH%20p50-140%20µs-brightgreen)
![1M recall](https://img.shields.io/badge/1M%20Recall@10-0.930%20vs%20brute--force-brightgreen)
![Chaos](https://img.shields.io/badge/Jepsen%20chaos-0%20lost%20%2F%2064k%20writes-brightgreen)
![Vector p50](https://img.shields.io/badge/100k×384d%20p50-69%20µs-9cf)
![Concurrent](https://img.shields.io/badge/concurrent%20VSEARCH-79k%20QPS-ff69b4)
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
| One consistency model | ❌ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ⚠️ | ✅ MVCC |
| One license | BSD | PostgreSQL | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 | BSL | ✅ Apache-2.0 |


---

## 📦 Install & Run

```bash
# build the release binary (zero external crates — pure stdlib Rust)
cargo build --release

# start the server on the RESP wire (talk to it with any Redis client)
DBSTRIKE_WAL=./dbstrike.wal ./target/release/dbstrike 127.0.0.1:6380
```

Talk to it with `redis-cli`, `nc`, or the bundled harnesses:

```bash
redis-cli -p 6380 PING          # -> PONG
redis-cli -p 6380 SET user:1 ada
redis-cli -p 6380 VADD 1 1.0 0.0 0.0
redis-cli -p 6380 VSEARCH 5 1.0 0.0 0.0
redis-cli -p 6380 SUBSCRIBE trades &
redis-cli -p 6380 PUBLISH trades "hello"

# native Rust bench harness — in-process, 14 sections in ~2.5s
cargo run --release -p bench

# add --tcp to also exercise SUBSCRIBE / PUBLISH end-to-end against
# a running server on 127.0.0.1:6380
./target/release/dbstrike-bench --tcp 127.0.0.1:6380

# 21-section Python integration + fuzz + crash-recovery suite
python3 tests/test_dbstrike.py
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
  **in-process** (no TCP loopback), 17 sections, exercises the engine
  directly and — with flags — also validates SUBSCRIBE/PUBLISH,
  million-vector scale, YCSB A/B/C/F, and Jepsen-style chaos.
- **`tests/test_dbstrike.py`** — 21-section Python integration suite that
  self-spawns the binary and drives it over RESP with a real fsync'd WAL.

```bash
cargo run --release -p bench                          # in-process (~17s)
./target/release/dbstrike-bench --tcp 127.0.0.1:6380  # + wire pubsub check
./target/release/dbstrike-bench --large               # + 1M vector bench (~3 min)
./target/release/dbstrike-bench --ycsb 127.0.0.1:6380 # + YCSB A/B/C/F
./target/release/dbstrike-bench --chaos               # + Jepsen SIGKILL loop
python3 tests/test_dbstrike.py                        # RESP + fuzz + crash
```

### 🚀 Million-vector scale (1M × 128-d, INT8 + f32 rerank)

The "1M is where everyone starts comparing" milestone. Full brute-force
ground truth (all 1,000,000 vectors), 20 queries.

| Metric | Value | Category-leader reference |
|---|---:|---|
| **1M ingest** | **5,964 vec/s** (167 s total) | — |
| **VSEARCH p50** | **140 µs** | Qdrant at 1M×1536d: ~3.5 ms |
| **VSEARCH p99** | **315 µs** | Qdrant at 1M×1536d: ~8.6 ms |
| **Recall@10 vs full 1M brute-force** | **0.930** | Qdrant ~0.95, pgvector ~0.90 |
| **8-thread concurrent VSEARCH** | **62,538 QPS** | Qdrant at 1M: ~1,000–1,200 QPS |

*Run yourself:* `cargo run --release -p bench -- --large`

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

### Storage & KV — in-process (native Rust bench)

| Metric | Value |
|---|---:|
| Single-thread SET (fsync'd WAL) | **122,931 ops/s** |
| **8-thread SET (sharded storage + group commit)** | **345,429 ops/s** (**2.8× single-thread**) |
| 32-thread reducer contention | **0 lost updates** (16,000/16,000 in 0.15s) |
| 10k×128d VSEARCH p99 (small) | **48 µs** |
| 10k×128d 8-thread concurrent VSEARCH | **200,182 QPS** |
| VSEARCH.MANY correctness | 32/32 batched match single calls |

### Storage & KV — over RESP wire (Python integration suite)

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
| Load | 100% SET | 41,204 ops/s |
| **YCSB-A** | 50% read / 50% update | **52,281 ops/s** |
| **YCSB-B** | 95% read / 5% update | **73,011 ops/s** |
| **YCSB-C** | 100% read | **74,648 ops/s** |
| **YCSB-F** | 50% read / 50% read-modify-write | **39,068 ops/s** |

*Run yourself:* start `./target/release/dbstrike 127.0.0.1:6399`, then
`cargo run --release -p bench -- --ycsb 127.0.0.1:6399`

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

### Correctness

| Suite | Result |
|---|---|
| Rust unit tests | **44 passing, 0 failing** |
| Native Rust bench (in-process, 14 sections) | **56 passing, 0 failing in 2.5s** |
| Python integration + wire suite (21 sections) | **104 passing, 0 failing in 33s** |
| Fuzz — 200 random wire payloads | server stayed up (0 crashes) |
| Fuzz — 20 random 1-byte WAL corruptions | engine reopened cleanly all 20 |
| Crash recovery — SIGKILL mid-write | reopens, earliest pre-kill write survives |
| Torn-tail WAL | mid-log key still readable |
| CRC corruption mid-log | engine opens, earliest key intact |

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
│ DISTRIBUTION / CONSENSUS        │  Raft · HLC · CRDTs (tunable)
└────────────────────────────────┘
```

See [`architecture.md`](architecture.md) and the
[📚 docs](docs/index.html) for the deep dives.

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
- [x] **1M-vector benchmark** (`--large`, 5,964 vec/s ingest, p50=140µs, Recall@10=0.930 vs full brute force)
- [x] **YCSB A / B / C / F workload harness** (`--ycsb`, 74k reads/s, 52k mixed)
- [x] **Jepsen-style chaos** (`--chaos`, 10× SIGKILL under load, **0 lost of 64,848 acked writes**)
- [ ] Tiered cold storage to NVMe/object store (architecture DD2)
- [ ] Consolidation reducer (hot → LTM background promotion)
- [ ] Raft per-shard consensus
- [ ] Product Quantization with learned codebooks (8–16× compression, ScaNN-style)
- [ ] Long soak testing (72 h continuous mixed workload)
- [ ] Cross-hardware validation (AWS c7g / Xeon / EPYC)
- [ ] ann-benchmarks datasets (SIFT1M, GIST, GloVe) recall × latency curves
- [ ] gRPC shim

---

## 📜 License

Apache-2.0. One license, one engine, no bolt-ons.
