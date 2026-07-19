#!/bin/bash
# Fast ingest benchmark — tests VADDBATCH PAR speed on real datasets
set -e

BIN=./target/release/dbstrike
BENCH=./target/release/dbstrike-bench
PORT=52000
WAL=/tmp/dbstrike_ingest_bench.wal
DATASET=${1:-/home/irfan/datasets/real_384_1M.fbin}

pkill -9 dbstrike 2>/dev/null; sleep 0.3
rm -f ${WAL} ${WAL}.snap

echo "=== Ingest Benchmark ==="
echo "Dataset: $DATASET"
echo "Server: $BIN"
echo ""

# Start server
DBSTRIKE_WAL=$WAL $BIN 127.0.0.1:$PORT &>/dev/null &
SERVER_PID=$!
sleep 0.5

# Verify server is up
if ! redis-cli -p $PORT PING >/dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "Server started on port $PORT"

# Run the real dataset benchmark (only s19 — skips all other sections)
echo ""
echo "Running benchmark..."
time $BENCH --real $DATASET 2>&1 | grep -E "ingest|VSEARCH|Recall|QPS|PASS|FAIL|summary|REAL|RSS"

# Cleanup
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
rm -f ${WAL} ${WAL}.snap
echo ""
echo "=== Done ==="
