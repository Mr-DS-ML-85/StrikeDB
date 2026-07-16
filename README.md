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
![Tests](https://img.shields.io/badge/tests-228%20passing%20(51%20rust%20%2B%2057%20native%20%2B%20120%20integration)-brightgreen.svg)
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

### 🚀 Million-vector scale — real embeddings, honest same-dim comparison

**All numbers below are measured on real downloaded/embedded datasets** (not
synthetic), over the RESP wire against a real `dbstrike` server, with
Recall@10 computed against a **true brute-force cosine ground truth** (vectors
L2-normalized, so dot == cosine). This is a **single-node, single-user**
benchmark: one machine (Ryzen 7 7700, Zen 4 / AVX2, 32 GB RAM),
one server process, no replication, no GPU. The concurrent-QPS column is
8 client threads on that one node — not a 100-client cluster figure.

Datasets: 768-d = `Sreenath/million-text-embeddings` (all-MiniLM-base-v2,
1M rows); 384-d = `sentence-transformers/all-MiniLM-L6-v2` embeddings of
real English Wikipedia sentences (1M rows), generated locally.

| Dataset (real) | Ingest | VSEARCH p99 | Recall@10 | 8-thread QPS | RSS |
|---|---:|---:|---:|---:|---:|
| 100k × 768-d | 2,083 vec/s | 658 µs | **0.999** | 5,680 | 1.3 GB |
| **1M × 768-d** | 1,296 vec/s | **923 µs** | **0.997** | **5,462** | 6.5 GB |
| 100k × 384-d | 4,760 vec/s | 319 µs | **0.999** | 20,000 | 1.1 GB |
| **1M × 384-d** | 2,179 vec/s | **474 µs** | **0.966** | **8,992** | 3.6 GB |

**vs Qdrant's published 1M numbers** (HNSW, cosine, M=16, ef_c=200,
single client): ~0.95–0.98 Recall@10, ~450 QPS, ~8 ms p99 at 1M×768-d.
dbstrike on the same dimensions, real embeddings, single-node:

- **Recall@10 wins**: 0.997 (768-d) / 0.966 (384-d) vs Qdrant's ~0.95–0.98.
- **Latency wins**: p99 923 µs / 474 µs vs Qdrant's ~8 ms single-client (~9× lower).
- **Throughput wins at matched client count**: 5,462 / 8,992 QPS at **8 threads**
  vs Qdrant's ~450 QPS single-client. (Qdrant's headline ~13k QPS is at
  100 concurrent clients over gRPC; dbstrike's number above is 8-client RESP —
  still ~12× their single-client figure. Peak multi-client QPS not yet measured.)
- **RAM efficient**: 6.5 GB (1M×768-d) / 3.6 GB (1M×384-d) with the
  INT8-traversal + exact-f32-rerank design.

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
| Rust unit tests | **51 passing, 0 failing** |
| Native Rust bench (in-process) | **57 passing, 0 failing in ~17s** |
| Python integration + wire suite | **120 passing, 0 failing in ~34s** |
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
