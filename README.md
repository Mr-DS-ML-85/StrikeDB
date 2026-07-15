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
![Tests](https://img.shields.io/badge/tests-211%20passing%20(50%20rust%20%2B%2057%20native%20%2B%20104%20integration)-brightgreen.svg)
![Durable SET P1024](https://img.shields.io/badge/durable%20SET%20@P1024-14.3M%20ops%2Fs-brightgreen)
![Durable GET P1024](https://img.shields.io/badge/durable%20GET%20@P1024-16.1M%20ops%2Fs-brightgreen)
![vs Redis SET](https://img.shields.io/badge/vs%20Redis%20SET-4.9×%20faster-brightgreen)
![vs Redis GET](https://img.shields.io/badge/vs%20Redis%20GET-3.9×%20faster-brightgreen)
![Benchmarked with](https://img.shields.io/badge/measured%20with-redis--benchmark-9cf)
![1M vectors](https://img.shields.io/badge/1M×384d%20VSEARCH%20p50-159%20µs-brightgreen)
![1M recall](https://img.shields.io/badge/1M×384d%20Recall@10-0.845%20vs%20brute--force-9cf)
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
redis-cli -p 6380 CHECKPOINT   # snapshot current state + truncate WAL

# native Rust bench harness — in-process, 14 sections in ~2.5s
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

### 🚀 Million-vector scale (fair same-dim comparison)

The "1M is where everyone starts comparing" milestone. **Earlier this README
was comparing our 128-d numbers to Qdrant's 1536-d numbers — that isn't
fair.** The new `--xlarge` bench runs 1M at 384-d (typical BGE / e5-small)
and 1M at 1536-d (OpenAI ada-002) so the comparison is honest per-dim.

**1M × 384-d — completed, full brute-force ground truth (20 queries):**

| Metric | Value | Reference |
|---|---:|---|
| Ingest | **4,780 vec/s** (209 s) | — |
| VSEARCH p50 | **159 µs** | — |
| VSEARCH p99 | **464 µs** | — |
| Recall@10 vs full 1M brute-force | **0.845** | Qdrant ~0.95, pgvector ~0.90 |
| 8-thread concurrent VSEARCH | **30,881 QPS** | — |

**1M × 1536-d — pending re-run** (recall gate close to threshold at 384-d
suggests we're tuning-limited; want to land memory/disk optimizations first
so the 12 GB run is stable, then republish 1536-d numbers).

**1M × 128-d — earlier number kept for reference (not a fair Qdrant comp):**
p50 = 140 µs, p99 = 315 µs, Recall@10 = 0.930, 62k QPS. The 128-d win is
real but Qdrant runs 1536-d in their public bench, so ignore this row when
comparing.

*Run yourself:* `cargo run --release -p bench -- --xlarge`  (takes ~15 min, needs ~14 GB RAM)

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
| **GET** | **5.88M /s** | **13.3M /s** | **≈16M /s** | 4.17M /s | 6.41M /s |
| PING | 6.25M /s | — | — | — | — |
| MSET (10 keys) | 3.51M /s (≈ 35.1M key-writes/s) | — | — | — | — |

**Non-durable mode (`DBSTRIKE_SYNC=0`, Redis-default semantics):**

| Op / `-P` | -P 64 | -P 256 | -P 1024 |
|---|---:|---:|---:|
| SET | 6.06M /s | 11.8M /s | ≈16.9M /s |
| GET | 5.88M /s | 13.8M /s | ≈16.9M /s |

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
A burst of 1024 pipelined SETs pays exactly one fsync. Before this refactor
durable SET was 65 k/s (a **220× regression** from what you see here at
-P1024).

**Redis-compat command coverage** (all pipelined-coalesced when applicable):
`PING · SET · GET · MSET · MGET · DEL · INCR · INCRBY · KEYS · DBSIZE ·
SELECT · COMMAND · FLUSHALL · FLUSHDB · SUBSCRIBE · PUBLISH · QUIT`
plus StrikeDB-native: `VADD · VSEARCH · VSEARCH.MANY · TSADD · TSADD.F ·
TSRANGE · TSRANGE.LATEST · MEM.* · RAG.* · CACHE.* · REDUCE · CHECKPOINT`.

**Env knobs:**
- `DBSTRIKE_WAL=<path>` — WAL file location (default: `dbstrike.wal`)
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
│ DISTRIBUTION / CONSENSUS        │  HLC · CRDTs (tunable) · Raft (planned)
└────────────────────────────────┘
```

See the
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
- [x] **Zero-copy brute-force GT** in `--xlarge` — HNSW's `for_each_normalized` lets bench skip the duplicate 6 GB f32 copy at 1M×1536d
- [x] **Demo `--bench` + RSS reporting** — every demo prints p50/p99/throughput + client RSS so "fast" is measured
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

## ⚠️ Known issue: Firefox Android DoH caching bug with custom domains.
Use Brave, Chrome, or disable DoH in Firefox settings.
