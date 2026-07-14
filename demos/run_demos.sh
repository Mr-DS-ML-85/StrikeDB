#!/usr/bin/env bash
# Run all DB-Strike demos end-to-end.
# Builds (if needed), starts the release server on a WAL, runs the three apps,
# then shuts the server down.
set -e
cd "$(dirname "$0")/.."

BIN=target/release/dbstrike
WAL=/tmp/dbstrike_demos.wal
PORT=6390

if [ ! -x "$BIN" ]; then
  echo "== building release binary =="
  cargo build --release
fi

rm -f "$WAL"
export DBSTRIKE_WAL="$WAL"

echo "== starting server on 127.0.0.1:$PORT =="
"$BIN" "127.0.0.1:$PORT" > /tmp/dbstrike_demos.log 2>&1 &
SRV=$!
# wait for PONG
for i in $(seq 1 50); do
  if python3 - <<PY 2>/dev/null
import socket
s=socket.create_connection(("127.0.0.1",$PORT),timeout=1)
s.sendall(b"*1\r\n\$4\r\nPING\r\n")
print("PONG" if s.recv(16).startswith(b"+PONG") else "NO")
PY
  then break; fi
  sleep 0.1
done

cleanup() { kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT

export DBSTRIKE_PORT=$PORT
run_demo() {
  echo
  echo "###########################################################"
  echo "# $1"
  echo "###########################################################"
  PORT=$PORT python3 "$2"
}

run_demo "Demo 1 — AI Agent with persistent multi-type memory" demos/agent_memory_demo.py
run_demo "Demo 2 — Realtime metrics + pub/sub dashboard" demos/realtime_dashboard_demo.py
run_demo "Demo 3 — RAG (hybrid) + MITM cache debugger" demos/rag_demo.py

echo
echo "All demos completed."
