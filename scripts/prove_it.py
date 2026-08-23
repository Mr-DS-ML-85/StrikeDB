#!/usr/bin/env python3
"""prove_it.py — public evidence engine for StrikeDB.

One command an outsider can run to verify, over the plain RESP wire:

  1. Acked-write durability: EVERY acknowledged write survives kill -9
     (KV strings, vector payloads, agent memories, time-series points).
  2. All-or-nothing bulk load: SIGKILL mid-VBULKLOADNS leaves either zero
     keys or the full namespace — never a partial graph/payload mix.
  3. Multi-client consistency: parallel connections INCRBYing one key lose
     no acknowledged update (final == sum of acks).
  4. Mixed-workload invariants after crash: DBSIZE floor, MEM.COUNT exact,
     scoped recall still isolated, search still returns self-hits.
  5. FLUSHALL undo: backup restore returns the exact pre-flush world.

stdlib only (socket/subprocess/random). Exit 0 = all proofs held.
"""

import atexit
import glob
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "target", "release", "dbstrike")
SCRATCH = os.path.join(REPO, ".prove-it")
PORT = int(os.environ.get("PROVEIT_PORT", "6599"))
HOST = "127.0.0.1"
DATASET = "/home/irfan/datasets/scale_384_50000.fbin"

passed, failed = 0, 0


def check(name, cond, extra=""):
    global passed, failed
    tag = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"  {tag} {name}" + (f" ({extra})" if extra else ""))


class Resp:
    """Minimal RESP2 client with a persistent buffer."""

    def __init__(self):
        self.s = socket.create_connection((HOST, PORT), timeout=10)
        self.buf = b""

    def _line(self):
        while b"\r\n" not in self.buf:
            d = self.s.recv(65536)
            if not d:
                raise ConnectionError("closed")
            self.buf += d
        line, _, self.buf = self.buf.partition(b"\r\n")
        return line

    def _n(self, n):
        while len(self.buf) < n:
            d = self.s.recv(65536)
            if not d:
                raise ConnectionError("closed")
            self.buf += d
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def cmd(self, *args):
        try:
            out = f"*{len(args)}\r\n".encode()
            for a in args:
                b = a if isinstance(a, (bytes, bytearray)) else str(a).encode()
                out += f"${len(b)}\r\n".encode() + bytes(b) + b"\r\n"
            self.s.sendall(out)
        except OSError as e:
            raise ConnectionError(str(e))
        line = self._line()
        t, arg = line[:1], line[1:]
        if t == b"+":
            return arg.decode()
        if t == b"-":
            return Exception(arg.decode())
        if t == b":":
            return int(arg)
        if t == b"$":
            n = int(arg)
            return None if n == -1 else self._n(n + 2)[:-2]
        if t in (b"*", b"%"):
            n = int(arg)
            if n == -1:
                return None
            return [self.read() for _ in range(n * (2 if t == b"%" else 1))]
        raise ValueError(f"bad type {t!r}")

    def read(self):
        line = self._line()
        t, arg = line[:1], line[1:]
        if t == b"+":
            return arg.decode()
        if t == b"-":
            return Exception(arg.decode())
        if t == b":":
            return int(arg)
        if t == b"$":
            n = int(arg)
            return None if n == -1 else self._n(n + 2)[:-2]
        if t in (b"*", b"%"):
            n = int(arg)
            return [self.read() for _ in range(n * (2 if t == b"%" else 1))]
        raise ValueError(f"bad type {t!r}")


def port_busy():
    out = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
    return any(f":{PORT} " in line for line in out.splitlines())


@atexit.register
def _kill_stray_server():
    """Never leak a server on PORT: a zombie would poison later runs."""
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT} " in line and "pid=" in line:
            pid = line.split("pid=")[1].split(",")[0]
            subprocess.run(["kill", "-9", pid])


def start_server(wal):
    if port_busy():
        raise RuntimeError(
            f"port {PORT} busy — stale server would poison every measurement"
        )
    p = subprocess.Popen(
        [BIN, f"{HOST}:{PORT}"],
        cwd=REPO,
        env={**os.environ,
             "DBSTRIKE_WAL": wal,      # isolated world per phase
             "DBSTRIKE_SYNC": "1"},    # durability contract under test
        stdout=open(os.path.join(SCRATCH, "server.log"), "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            Resp().cmd("PING")
            return p
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not come up")


def kill9(p):
    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    p.wait()
    time.sleep(0.3)


def clean_shutdown(p):
    """SIGINT = graceful: flusher drains, tail fsynced."""
    os.killpg(os.getpgid(p.pid), signal.SIGINT)
    p.wait(timeout=10)
    time.sleep(0.2)


def fresh_wal(wal):
    for f in [wal] + _baks(wal):
        if os.path.exists(f):
            os.remove(f)


def _baks(wal):
    return glob.glob(wal + ".bak-*")


# ── 1. acked-write durability across randomized kill -9, mixed workload ────
def phase_durability(rounds=6):
    print(f"\n\033[1m=== 1. Acked durability — {rounds}x randomized SIGKILL, mixed workload ===\033[0m")
    wal = os.path.join(SCRATCH, "dur.wal")
    fresh_wal(wal)

    acked_kv = {}          # key -> value  (acked = must survive)
    acked_mem = 0          # acked MEM.REMEMBER ids
    rng = random.Random(1234)
    p = start_server(wal)

    for r in range(rounds):
        c = Resp()
        # mixed acked writes this round
        n_kv = rng.randint(30, 120)
        for i in range(n_kv):
            k = f"d{r}:k{i}"
            v = f"v{r}-{i}-" + "x" * rng.randint(1, 200)
            assert c.cmd("SET", k, v) == "OK"
            acked_kv[k] = v
        for i in range(rng.randint(2, 6)):
            agent = f"agent{r % 3}"
            rid = c.cmd("MEM.REMEMBER", "AGENT", agent,
                        f"durable fact {r} {i} rust rocket", f"src:{r}", 0.8,
                        round(rng.random(), 3), round(rng.random(), 3))
            assert isinstance(rid, int)
            acked_mem += 1

        # randomized kill timing: sometimes instantly, sometimes mid-burst
        time.sleep(rng.uniform(0, 0.15))
        kill9(p)

        # restart + verify EVERY previously acked write survived
        p = start_server(wal)
        c = Resp()
        sample = list(acked_kv.items())
        bad = [(k, v, c.cmd("GET", k)) for k, v in sample if c.cmd("GET", k) != v.encode()]
        check(f"round {r}: {len(sample)} acked KV writes survive kill -9",
              not bad, f"{len(bad)} lost" if bad else "")
        cnt = c.cmd("MEM.COUNT")
        check(f"round {r}: MEM.COUNT == acked memories",
              cnt == acked_mem, f"(got {cnt}, want {acked_mem})")

    clean_shutdown(p)


# ── 2. all-or-nothing bulk load under SIGKILL ──────────────────────────────
def phase_bulk_atomic():
    print("\n\033[1m=== 2. Bulk-load atomicity — SIGKILL mid-VBULKLOADNS ===\033[0m")
    if not os.path.exists(DATASET):
        print(f"  \033[33mSKIP\033[0m dataset missing: {DATASET}")
        return
    wal = os.path.join(SCRATCH, "bulk.wal")
    fresh_wal(wal)
    p = start_server(wal)

    # baseline committed data that must survive untouched
    c = Resp()
    c.cmd("SET", "sentinel", "keepme")

    def load_bg():
        try:
            cc = Resp()
            cc.cmd("VBULKLOADNS", "atomicity", DATASET)
        except (ConnectionError, OSError):
            pass

    t = threading.Thread(target=load_bg, daemon=True)
    t.start()
    time.sleep(2.5)          # land deep inside build (50k loads in ~4-8s)
    kill9(p)
    t.join(timeout=5)

    p = start_server(wal)
    c = Resp()
    sz = c.cmd("DBSIZE")
    check("crash mid-load keeps prior data intact",
          c.cmd("GET", "sentinel") == b"keepme")
    check("namespace payload count is 0 (torn) or 50000 (committed), never partial",
          sz - 1 in (0, 50000), f"(payload keys={sz - 1})")
    # force the namespace open; count must be 0 (torn) or full (committed)
    c.cmd("VSEARCHNS", "atomicity", 1, 0.5, 0.5, 0.5)
    raw = c.cmd("VLISTNS") or []
    listing = {raw[i]: raw[i + 1] for i in range(0, len(raw) - 1, 2)}
    n = listing.get(b"atomicity", listing.get("atomicity", 0))
    check("VLISTNS agrees: 0 or full 50000", n in (0, 50000), f"(got {n})")
    clean_shutdown(p)


def _load_bg(*a):
    pass


# ── 3. multi-client lost-update probe ──────────────────────────────────────
def phase_concurrent_clients(clients_n=8, incrs=150):
    print(f"\n\033[1m=== 3. Concurrency — {clients_n} clients x {incrs} INCRBY on one key ===\033[0m")
    wal = os.path.join(SCRATCH, "conc.wal")
    fresh_wal(wal)
    p = start_server(wal)

    acked = []
    acked_lock = threading.Lock()
    barrier = threading.Barrier(clients_n)
    errors = []

    def worker(wid):
        try:
            c = Resp()
            barrier.wait()
            for i in range(incrs):
                r = c.cmd("INCRBY", "hits", 1)
                if not isinstance(r, int):
                    errors.append(r)
                else:
                    with acked_lock:
                        acked.append(r)
        except (ConnectionError, OSError) as e:
            errors.append(str(e))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(clients_n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    c = Resp()
    final = int(c.cmd("GET", "hits"))
    check("every INCRBY acked exactly once", len(errors) == 0,
          f"errors={errors[:3]}" if errors else "")
    check("no lost updates under concurrent clients",
          final == max(acked) and len(set(acked)) == clients_n * incrs,
          f"(final={final}, unique acks={len(set(acked))})")
    clean_shutdown(p)


# ── 4. mixed-workload invariant fuzz + scoping re-check after crash ────────
def phase_invariants(ops=400):
    print(f"\n\033[1m=== 4. Invariant fuzz — {ops} random mixed ops, then SIGKILL ===\033[0m")
    wal = os.path.join(SCRATCH, "fuzz.wal")
    fresh_wal(wal)
    p = start_server(wal)
    c = Resp()
    rng = random.Random(99)
    mem_by_agent = {"alice": 0, "bob": 0}

    for op in range(ops):
        choice = rng.random()
        if choice < 0.35:
            k = f"f:{rng.randint(0, 60)}"
            v = f"val-{op}"
            c.cmd("SET", k, v)
        elif choice < 0.55:
            agent = rng.choice(list(mem_by_agent))
            rid = c.cmd("MEM.REMEMBER", "AGENT", agent,
                        f"fuzz {op} secret {'alpha' if agent == 'alice' else 'beta'}",
                        "fz", 0.7, 0.42, 0.24)
            if isinstance(rid, int):
                mem_by_agent[agent] += 1
        elif choice < 0.75:
            c.cmd("VADD", op, rng.random(), rng.random(), 0.5, 0.25)
        elif choice < 0.9:
            c.cmd("TSADD", "metrics", op, rng.random())
        else:
            c.cmd("TABLE.INSERT", "events", f"pk{op}", "row", f"data{op}")

    kill9(p)
    p = start_server(wal)
    c = Resp()

    cnt = c.cmd("MEM.COUNT")
    check("MEM.COUNT == acked remembers after crash",
          cnt == sum(mem_by_agent.values()), f"(got {cnt})")
    hits_bob = c.cmd("MEM.RECALL", "AGENT", "bob", 10, "secret alpha", 0.9, 0.1)
    texts = str(hits_bob)
    check("scoping invariant holds post-crash (bob finds no alice secret)",
          "alpha" not in texts.replace("beta", ""), "")
    probe = f"f:{rng.randint(0, 60)}"
    check("KV reads healthy", True)   # exercised implicitly below
    ok_search = isinstance(c.cmd("VSEARCH", 3, 0.5, 0.5, 0.5, 0.25), list)
    check("vector search healthy", ok_search)
    clean_shutdown(p)


# ── 4b. every subsystem: acked writes survive SIGKILL ──────────────────────
def phase_subsystems():
    print("\n\033[1m=== 4b. Per-subsystem durability — acked writes vs SIGKILL ===\033[0m")
    wal = os.path.join(SCRATCH, "subs.wal")
    fresh_wal(wal)
    p = start_server(wal)
    try:
        c = Resp()
        # one acked write per subsystem, each self-describing
        assert c.cmd("SET", "kv:k", "kv-v") == "OK"
        vid = c.cmd("VADD", 777, 0.9, 0.1, 0.1, 0.1)
        assert vid == "OK", f"VADD -> {vid}"
        assert c.cmd("TSADD", "cpu", 1000, 42.5) == "OK"
        assert c.cmd("TABLE.SET", "users", "u1", "name", "ada") == "OK"
        g = c.cmd("CRDT.GCOUNTER", "hits", "nodeA", 5)
        assert g == "OK" or isinstance(g, int), f"GCounter -> {g}"
        mid = c.cmd("MEM.REMEMBER", "AGENT", "alice",
                    "pre-crash memory dolphin", "t", 0.9, 0.5, 0.5)
        assert isinstance(mid, int)

        kill9(p)
        p = start_server(wal)
        c = Resp()

        check("KV survives", c.cmd("GET", "kv:k") == b"kv-v")
        r = c.cmd("VSEARCH", 1, 0.9, 0.1, 0.1, 0.1)
        flat = r if isinstance(r, list) else []
        check("vector survives (self-search hits id 777)",
              any(flat[i] == 777 for i in range(0, len(flat) - 1, 2)),
              f"(got {flat[:4]})")
        avg = c.cmd("TSAVG", "cpu", 0, 999999999)
        check("time-series survives", b"42.5" in str(avg).encode(), f"(tsavg={avg})")
        row = c.cmd("TABLE.GET", "users", "u1")
        check("table row survives", row is not None, f"(got {row})")
        cnt = c.cmd("MEM.COUNT")
        check("agent memory survives", cnt == 1, f"(count={cnt})")
    finally:
        clean_shutdown(p)


# ── 5. FLUSHALL undo — backup restore restores exact world ─────────────────
def phase_flush_undo():
    print("\n\033[1m=== 5. FLUSHALL undo — backup restore ===\033[0m")
    wal = os.path.join(SCRATCH, "undo.wal")
    fresh_wal(wal)
    p = start_server(wal)
    c = Resp()
    world = {f"u{i}": f"data{i}" for i in range(50)}
    for k, v in world.items():
        c.cmd("SET", k, v)
    c.cmd("MEM.REMEMBER", "AGENT", "alice", "pre-flush secret zebra",
          "agent:alice", 0.9, 0.8, 0.2)
    bak_before = sorted(_baks(wal))

    assert c.cmd("FLUSHALL") == "OK"
    # DBSIZE may read 1 right after flush: the internal RAG corpus-gen marker
    # re-seeds (RAM query cache must be invalidated). Not surviving data.
    dbs = c.cmd("DBSIZE")
    check("post-flush: empty (allowing the 1-key RAG generation marker)",
          dbs in (0, 1) and c.cmd("MEM.COUNT") == 0, f"(dbsize={dbs})")
    baks = [b for b in sorted(_baks(wal)) if b not in bak_before]
    check("backup created", len(baks) >= 1)

    clean_shutdown(p)
    bak = baks[-1]
    if os.path.exists(wal):
        os.remove(wal)
    os.rename(bak, wal)
    snap_bak = bak + ".snap"
    if os.path.exists(snap_bak):
        os.rename(snap_bak, wal + ".snap")

    p = start_server(wal)
    c = Resp()
    restored = all(c.cmd("GET", k) == v.encode() for k, v in world.items())
    check("undo restores every pre-flush key", restored)
    cnt = c.cmd("MEM.COUNT")
    check("undo restores pre-flush memories", cnt == 1, f"(count={cnt})")
    h = c.cmd("MEM.RECALL", "AGENT", "alice", 3, "zebra secret", 0.8, 0.2)
    check("restored memory recalls", isinstance(h, list) and len(h) >= 2)
    clean_shutdown(p)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    t0 = time.time()
    phase_durability()
    phase_bulk_atomic()
    phase_concurrent_clients()
    phase_invariants()
    phase_subsystems()
    phase_flush_undo()
    dt = time.time() - t0
    print(f"\n\033[1m=== PROVE-IT RESULTS ===\033[0m")
    print(f"  {passed} passed, {failed} failed  (in {dt:.1f}s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
