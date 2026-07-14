# DB-Strike Demos

Three runnable example apps that use DB-Strike as a **real backend** over the
RESP wire — no mocks, no stubs. Each one runs against the actual release
binary (`target/release/dbstrike`).

## What's here

| File | What it shows |
|---|---|
| `dbstrike_client.py` | Thin RESP client (KV / vector / memory / RAG / cache) you can import. |
| `agent_memory_demo.py` | An AI agent ("Atlas") with persistent **working + long-term + graph + bi-temporal + procedural** memory — all in one engine. |
| `realtime_dashboard_demo.py` | A **realtime metrics dashboard**: time-series ingest (8 devices), a live window read, pub/sub fan-out, and WAL-backed crash replay. |
| `rag_demo.py` | **Hybrid RAG** (dense + sparse, RRF-fused) with the **MITM cache debugger** proving a stale corpus can never be served. |
| `run_demos.sh` | One-shot: builds (if needed), starts the server, runs all three, shuts down. |

## Run everything

```bash
cargo build --release
bash demos/run_demos.sh
```

## Run one app manually

```bash
# terminal 1 — start the server
DBSTRIKE_WAL=./demo.wal ./target/release/dbstrike 127.0.0.1:6380

# terminal 2 — any demo (uses DBSTRIKE_PORT if set, else 6380)
python3 demos/agent_memory_demo.py
python3 demos/realtime_dashboard_demo.py
python3 demos/rag_demo.py
```

You can also poke the server with a normal Redis client:

```bash
redis-cli -p 6380 PING
redis-cli -p 6380 SET user:1 ada
redis-cli -p 6380 VADD 1 1.0 0.0 0.0
redis-cli -p 6380 VSEARCH 5 1.0 0.0 0.0
```

## About the embeddings

The demos use a **deterministic pseudo-embedding** (`embed()` in
`dbstrike_client.py`) so they run with zero external dependencies. To use real
embeddings, replace `embed()` with a call to an embedding endpoint
(sentence-transformers, OpenAI, a local llama.cpp embedder, etc.) — the wire
protocol and all commands are identical.

## Verified results

All three demos complete against the release binary. Sample output (abridged):

- Agent demo: stores 4 LTM facts, recalls the release blocker (score 1.26),
  builds an Alice→Acme→NYC graph and traverses it in 2 hops, demonstrates
  bi-temporal `as-of` recall (CTO → CEO → invalidate), and an agent turn
  (KV + vector recall) in ~266 µs.
- Realtime demo: 8 devices × 300 samples written at ~2,867 points/s, live
  window read per device, 2,400 durable writes in the CDC log, time-ordered
  series verified across the full range.
- RAG demo: first query computes (`fresh`), repeat serves from cache
  (`cached`), a new ingest bumps the corpus generation and forces a recompute
  (`fresh`); the MITM debugger flags an injected stale read as `STALE_HIT`.

See [`../BENCHMARKS.md`](../BENCHMARKS.md) for the full measured numbers.
