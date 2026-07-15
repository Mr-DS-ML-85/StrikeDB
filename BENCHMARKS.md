# ⚡ StrikeDB Benchmarks

Full verified numbers, including 50k-scale vectors and concurrency scaling, measured by the bundled production harness (`tests/test_dbstrike.py`, 96 checks) against the release binary.

### 📊 Real Numbers (release build, this machine)

| Operation | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| PING | 13.4µs | 13.7µs | 18.3µs | 32.3µs |
| SET | 24.0µs | 27.8µs | 36.5µs | 600µs |
| GET | 14.2µs | 16.7µs | 20.1µs | 40.8µs |
| VSEARCH k=10 (8-d) | 16.0µs | 16.1µs | 18.5µs | 26.2µs |
| VSEARCH k=10 (768-d) | 530µs | 622µs | 728µs | 816µs |
| REDUCE | 24.1µs | 24.8µs | 29.0µs | 58.7µs |
| Agent turn (3 ops) | 78.9µs | 89.9µs | 120.6µs | 270µs |

| Metric | Value |
|---|---|
| Single-connection SET | 39,318 ops/s |
| 8-connection SET peak | 96,249 ops/s |
| 64-connection SET | 87,497 ops/s |
| 50k × 128-d HNSW recall | 1.00 Recall@10 vs brute-force |
| Unit + integration tests | 140 passing, 0 failing |

### 🚀 Million-vector scale (fair same-dim comparison)

**1M × 384-d — completed, full brute-force ground truth (20 queries):**

| Metric | Value |
|---|---:|
| Ingest | 4,780 vec/s (209 s) |
| VSEARCH p50 | 159 µs |
| VSEARCH p99 | 464 µs |
| Recall@10 vs full 1M brute-force | 0.845 |
| 8-thread concurrent VSEARCH | 30,881 QPS |

### 📈 YCSB A / B / C / F (over RESP wire, 100k records, 100k ops each)

| Workload | Mix | Throughput |
|---|---|---:|
| Load | 100% SET | 41,204 ops/s |
| YCSB-A | 50% read / 50% update | 52,281 ops/s |
| YCSB-B | 95% read / 5% update | 73,011 ops/s |
| YCSB-C | 100% read | 74,648 ops/s |
| YCSB-F | 50% read / 50% read-modify-write | 39,068 ops/s |

### 💥 Jepsen-style chaos

| Metric | Value |
|---|---|
| Iterations | 10 |
| Acked writes across all iterations | 64,848 |
| Writes lost after SIGKILL + reopen | 0 |
