#!/usr/bin/env python3
"""Redis Streams on dbstrike: blocking reads, consumer groups under
concurrency, MULTI/WATCH, trimming, and durability across kill -9.

Exact reply shapes are covered by the differential test against
redis-server; this suite covers what a diff cannot (timing, concurrency,
crash recovery).

    cargo build --release -p server --bin dbstrike
    python3 tests/test_streams.py
"""
import os, socket, subprocess, sys, tempfile, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("DBSTRIKE_TEST_PORT", "6582"))
B = os.path.join(REPO, "target", "release", "dbstrike")
WAL = os.path.join(tempfile.mkdtemp(prefix="dbstrike_streams_"), "s.wal")


def enc(a):
    a = [x if isinstance(x, bytes) else str(x).encode() for x in a]
    return b"*%d\r\n" % len(a) + b"".join(b"$%d\r\n%s\r\n" % (len(x), x) for x in a)


class C:
    def __init__(s):
        s.s = socket.create_connection(("127.0.0.1", PORT))
        s.f = s.s.makefile("rb")

    def rd(s):
        l = s.f.readline(); t = l[:1]; b = l[1:-2]
        if t == b"+": return b.decode()
        if t == b"-": return "ERR:" + b.decode()
        if t == b":": return int(b)
        if t == b"$":
            n = int(b); return None if n < 0 else s.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(b); return None if n < 0 else [s.rd() for _ in range(n)]
        if t == b"_": return None
        raise RuntimeError(f"bad reply {l!r}")

    def cmd(s, *a):
        s.s.sendall(enc(a)); return s.rd()


P = F = 0


def check(name, ok, info=""):
    global P, F
    if ok: P += 1; print("  PASS", name)
    else: F += 1; print("  FAIL", name, info)


def start():
    p = subprocess.Popen([B, f"127.0.0.1:{PORT}"], env={**os.environ, "DBSTRIKE_WAL": WAL},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            C().cmd("PING"); return p
        except OSError:
            time.sleep(0.05)
    raise SystemExit("server did not start")


p = start(); c = C()

print("== IDs and ranges")
a = c.cmd("XADD", "s", "*", "f", "1"); b = c.cmd("XADD", "s", "*", "f", "2")
ms_a, seq_a = map(int, a.split(b"-")); ms_b, seq_b = map(int, b.split(b"-"))
check("auto IDs strictly increase", (ms_b, seq_b) > (ms_a, seq_a))
check("auto ID uses wall-clock ms", abs(ms_a - time.time() * 1000) < 5000)
check("XRANGE returns both in order", [e[0] for e in c.cmd("XRANGE", "s", "-", "+")] == [a, b])
check("TYPE stream", c.cmd("TYPE", "s") == "stream")

print("== Blocking reads")
res = []
def blocker(cmd):
    cc = C(); res.append(cc.cmd(*cmd))
t = threading.Thread(target=blocker, args=(["XREAD", "BLOCK", "2000", "STREAMS", "s", "$"],)); t.start()
time.sleep(0.2); new = c.cmd("XADD", "s", "*", "f", "3"); t.join()
check("XREAD BLOCK $ wakes on XADD with only the new entry", res and res[0] == [[b"s", [[new, [b"f", b"3"]]]]], res)
t0 = time.time(); r = c.cmd("XREAD", "BLOCK", "300", "STREAMS", "s", "$")
check("XREAD BLOCK times out with a null reply", r is None and 0.25 < time.time() - t0 < 2)
r = c.cmd("XREAD", "BLOCK", "1000", "STREAMS", "s", "0")
check("XREAD BLOCK with data available returns at once", r and len(r[0][1]) == 3)
c.cmd("XGROUP", "CREATE", "s", "g", "$")
res.clear()
t = threading.Thread(target=blocker, args=(["XREADGROUP", "GROUP", "g", "w1", "BLOCK", "0", "STREAMS", "s", ">"],)); t.start()
time.sleep(0.3)
check("blocked consumer group read is not spinning writes", c.cmd("XPENDING", "s", "g")[0] == 0)
gid = c.cmd("XADD", "s", "*", "job", "x"); t.join(5)
check("XREADGROUP BLOCK 0 wakes on XADD", res and res[0][0][1][0][0] == gid, res)
check("...and the entry is pending for w1", c.cmd("XPENDING", "s", "g", "-", "+", "10")[0][:2] == [gid, b"w1"])

print("== Consumer group: exactly-once across concurrent consumers")
c.cmd("DEL", "jobs")
c.cmd("XGROUP", "CREATE", "jobs", "workers", "$", "MKSTREAM")
N = 2000
pipe = b"".join(enc(["XADD", "jobs", "*", "n", str(i)]) for i in range(N))
c.s.sendall(pipe); ids = [c.rd() for _ in range(N)]
check("2000 pipelined XADDs, all distinct", len(set(ids)) == N)
got = {}; lock = threading.Lock()
def consumer(name):
    cc = C()
    while True:
        r = cc.cmd("XREADGROUP", "GROUP", "workers", name, "COUNT", "37", "STREAMS", "jobs", ">")
        if r is None: return
        for eid, fields in r[0][1]:
            with lock: got.setdefault(eid, []).append(name)
            cc.cmd("XACK", "jobs", "workers", eid)
ths = [threading.Thread(target=consumer, args=(f"c{i}",)) for i in range(8)]
[t.start() for t in ths]; [t.join() for t in ths]
check("every entry delivered exactly once", len(got) == N and all(len(v) == 1 for v in got.values()),
      (len(got), sum(len(v) > 1 for v in got.values())))
check("nothing left pending after XACK", c.cmd("XPENDING", "jobs", "workers")[0] == 0)
check("work spread over several consumers", len({v[0] for v in got.values()}) > 1)
info = {g[1]: g for g in c.cmd("XINFO", "GROUPS", "jobs")}
check("lag 0 and entries-read = 2000", info[b"workers"][9] == N and info[b"workers"][11] == 0, info)

print("== Concurrent producers")
c.cmd("DEL", "multi")
def producer(i):
    cc = C()
    for j in range(200): cc.cmd("XADD", "multi", "*", "p", str(i), "j", str(j))
ths = [threading.Thread(target=producer, args=(i,)) for i in range(16)]
[t.start() for t in ths]; [t.join() for t in ths]
es = c.cmd("XRANGE", "multi", "-", "+")
keys = [tuple(map(int, e[0].split(b"-"))) for e in es]
check("16x200 concurrent XADD: 3200 entries", len(es) == 3200, len(es))
check("IDs strictly increasing, no duplicates", keys == sorted(set(keys)))
per = {}
for e in es:
    f = dict(zip(e[1][::2], e[1][1::2])); per.setdefault(f[b"p"], []).append(int(f[b"j"]))
check("each producer's entries in its own order", all(v == list(range(200)) for v in per.values()))

print("== MULTI / WATCH")
c.cmd("DEL", "tx")
c.cmd("MULTI"); c.cmd("XADD", "tx", "1-1", "a", "1"); c.cmd("XADD", "tx", "2-1", "b", "2"); c.cmd("XLEN", "tx")
check("XADD inside MULTI/EXEC is atomic", c.cmd("EXEC") == [b"1-1", b"2-1", 2])
c.cmd("WATCH", "tx"); C().cmd("XADD", "tx", "3-1", "c", "3")
c.cmd("MULTI"); c.cmd("XADD", "tx", "4-1", "d", "4")
check("WATCH aborts after another client's XADD", c.cmd("EXEC") is None and c.cmd("XLEN", "tx") == 3)

print("== Trimming")
c.cmd("DEL", "tr")
c.s.sendall(b"".join(enc(["XADD", "tr", "MAXLEN", "~", "1000", "*", "i", str(i)]) for i in range(1500)))
[c.rd() for _ in range(1500)]
n = c.cmd("XLEN", "tr")
check("MAXLEN ~ keeps at least the threshold, trims whole nodes", 1000 <= n < 1100, n)
check("XTRIM MAXLEN = exact", c.cmd("XTRIM", "tr", "MAXLEN", "10") == n - 10 and c.cmd("XLEN", "tr") == 10)

print("== Durability across kill -9")
c.cmd("DEL", "d")
for i in range(1, 6): c.cmd("XADD", "d", f"{i}-0", "v", str(i))
c.cmd("XDEL", "d", "3-0")
c.cmd("XGROUP", "CREATE", "d", "g", "0")
c.cmd("XREADGROUP", "GROUP", "g", "alice", "COUNT", "2", "STREAMS", "d", ">")
before = (c.cmd("XRANGE", "d", "-", "+"), c.cmd("XPENDING", "d", "g", "-", "+", "10"), c.cmd("XINFO", "STREAM", "d"))
p.kill(); p.wait(); p = start(); c = C()
after = (c.cmd("XRANGE", "d", "-", "+"), c.cmd("XPENDING", "d", "g", "-", "+", "10"), c.cmd("XINFO", "STREAM", "d"))
check("entries survive", after[0] == before[0])
check("PEL survives (owner, delivery count)", [e[:2] + e[3:] for e in after[1]] == [e[:2] + e[3:] for e in before[1]])
check("stream metadata survives (last id, max deleted, entries added)", after[2][6:12] == before[2][6:12], (before[2], after[2]))
check("XADD continues after the last ID", c.cmd("XADD", "d", "5-*", "v", "6") == b"5-1")
check("group continues where it stopped", c.cmd("XREADGROUP", "GROUP", "g", "bob", "STREAMS", "d", ">")[0][1][0][0] == b"4-0")

p.kill()
print(f"{P} passed, {F} failed")
sys.exit(1 if F else 0)
