"""Primary -> replica replication, end to end (stdlib only).

Starts a primary and a replica as real processes and checks: full sync of
every data model, the live stream (updates, deletes, TTL, FLUSHALL), WAIT,
read-only replicas, ROLE/INFO, replica restart (full resync), primary
restart (reconnect) and promotion with REPLICAOF NO ONE.

    cargo build --release && python3 tests/test_replication.py
"""
import os, socket, subprocess, sys, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "target", "release", "dbstrike")
PP, RP = int(os.environ.get("REPL_PRIMARY_PORT", "6590")), int(os.environ.get("REPL_REPLICA_PORT", "6591"))
DIR = tempfile.mkdtemp(prefix="dbstrike_repl_")

def enc(a):
    a = [x if isinstance(x, bytes) else str(x).encode() for x in a]
    return b"*%d\r\n" % len(a) + b"".join(b"$%d\r\n%s\r\n" % (len(x), x) for x in a)

class C:
    def __init__(s, port):
        s.s = socket.create_connection(("127.0.0.1", port)); s.f = s.s.makefile("rb")
    def rd(s):
        l = s.f.readline(); t = l[:1]; b = l[1:-2]
        if t == b"+": return b.decode()
        if t == b"-": return "ERR:" + b.decode()
        if t == b":": return int(b)
        if t == b"$":
            n = int(b); return None if n < 0 else s.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(b); return None if n < 0 else [s.rd() for _ in range(n)]
        raise RuntimeError(repr(l))
    def cmd(s, *a):
        s.s.sendall(enc(a)); return s.rd()

P = F = 0
def check(name, ok, info=""):
    global P, F
    if ok: P += 1; print("  PASS", name)
    else: F += 1; print("  FAIL", name, info)

def start(port, wal, env=None):
    p = subprocess.Popen([BIN, f"127.0.0.1:{port}"], env={**os.environ, "DBSTRIKE_WAL": wal, **(env or {})},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            C(port).cmd("PING"); return p
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")

def until(fn, timeout=10.0):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if fn(): return True
        except Exception:
            pass
        time.sleep(0.05)
    return False

pw, rw = os.path.join(DIR, "p.wal"), os.path.join(DIR, "r.wal")
prim = start(PP, pw)
p = C(PP)
print("== seed the primary (every data model)")
p.cmd("SET", "s", "hello"); p.cmd("SET", "n", "41"); p.cmd("INCR", "n")
p.cmd("SET", "ttl", "v", "EX", 1000)
p.cmd("HSET", "h", "a", "1", "b", "2"); p.cmd("RPUSH", "l", "x", "y", "z"); p.cmd("SADD", "st", "m1", "m2")
p.cmd("ZADD", "z", 1, "one", 2, "two")
for i in range(20):
    p.cmd("VADD", 1000 + i, 1.0, i / 20, 0.0, 0.0)
p.cmd("VADDNS", "faces", 7, 0.1, 0.9)
p.cmd("TABLE.SET", "users", "u1", "name", "ada")
p.cmd("TSADD", "cpu", 100, 42.5)
p.cmd("CRDT.GCOUNTER", "hits", "n1", 5)
p.cmd("MEM.REMEMBER", "AGENT", "alice", "replicated memory dolphin", "src", 0.9, 0.5, 0.5)
p.s.sendall(b"".join(enc(["SET", f"bulk:{i}", i]) for i in range(3000)))  # pipelined
for _ in range(3000):
    p.rd()

print("== start the replica")
rep = start(RP, rw, {"DBSTRIKE_REPLICAOF": f"127.0.0.1:{PP}"})
r = C(RP)
check("replica reaches link up", until(lambda: r.cmd("ROLE")[3] == b"connected"), r.cmd("ROLE"))
check("full sync: strings", r.cmd("GET", "s") == b"hello" and r.cmd("GET", "n") == b"42")
check("full sync: TTL carried", 900 < r.cmd("TTL", "ttl") <= 1000, r.cmd("TTL", "ttl"))
check("full sync: hash/list/set/zset", sorted(r.cmd("HGETALL", "h")) == sorted([b"a", b"1", b"b", b"2"])
      and r.cmd("LRANGE", "l", 0, -1) == [b"x", b"y", b"z"] and sorted(r.cmd("SMEMBERS", "st")) == [b"m1", b"m2"]
      and r.cmd("ZRANGE", "z", 0, -1, "WITHSCORES") == [b"one", b"1", b"two", b"2"])
hit = r.cmd("VSEARCH", 1, 1.0, 0.25, 0.0, 0.0)
check("full sync: vector index rebuilt (VSEARCH)", isinstance(hit, list) and hit[:1] == [1005], hit)
check("full sync: vector namespace", (r.cmd("VSEARCHNS", "faces", 1, 0.1, 0.9) or [None])[0] == 7)
check("full sync: table + time series", r.cmd("TABLE.GET", "users", "u1") is not None and b"42.5" in str(r.cmd("TSAVG", "cpu", 0, 10**12)).encode())
check("full sync: CRDT", r.cmd("CRDT.GET", "hits") == b"5")
check("full sync: agent memory", r.cmd("MEM.COUNT") == 1)
check("full sync: bulk keys", r.cmd("DBSIZE") == p.cmd("DBSIZE"), (r.cmd("DBSIZE"), p.cmd("DBSIZE")))

print("== live stream")
p.cmd("SET", "s", "world"); p.cmd("DEL", "bulk:0"); p.cmd("HSET", "h", "c", "3"); p.cmd("LPOP", "l")
p.cmd("ZINCRBY", "z", 10, "one"); p.cmd("VADD", 5000, 0.0, 0.0, 1.0, 0.0); p.cmd("CRDT.GCOUNTER", "hits", "n2", 2)
p.cmd("SET", "short", "v", "PX", 300)
p.cmd("MULTI"); p.cmd("INCR", "tx"); p.cmd("RPUSH", "l", "w"); p.cmd("EXEC")
check("WAIT 1 replica", p.cmd("WAIT", 1, 2000) == 1)
check("live: overwrite + delete", r.cmd("GET", "s") == b"world" and r.cmd("EXISTS", "bulk:0") == 0)
check("live: collections", r.cmd("HGET", "h", "c") == b"3" and r.cmd("LRANGE", "l", 0, -1) == [b"y", b"z", b"w"]
      and r.cmd("ZSCORE", "z", "one") == b"11")
check("live: vector insert visible to VSEARCH", (r.cmd("VSEARCH", 1, 0.0, 0.0, 1.0, 0.0) or [None])[0] == 5000)
check("live: CRDT", r.cmd("CRDT.GET", "hits") == b"7")
check("live: MULTI block", r.cmd("GET", "tx") == b"1")
p.cmd("MEM.REMEMBER", "AGENT", "alice", "second memory", "src", 0.9, 0.5, 0.5)
check("live: agent memory mirrors refresh", until(lambda: r.cmd("MEM.COUNT") == 2, 5))
check("live: TTL expiry on replica", until(lambda: r.cmd("EXISTS", "short") == 0, 5))

print("== convergence under concurrent load")
import random, threading
def load(seed):
    rnd = random.Random(seed); c = C(PP)
    for _ in range(1200):
        k = f"cc:{rnd.randint(0, 40)}"
        op = rnd.randint(0, 9)
        if op == 0: c.cmd("SET", k, rnd.randint(0, 99))
        elif op == 1: c.cmd("DEL", k)
        elif op == 2: c.cmd("HSET", "cch:" + k, rnd.randint(0, 5), rnd.randint(0, 9))
        elif op == 3: c.cmd("RPUSH", "ccl:" + k, rnd.randint(0, 9))
        elif op == 4: c.cmd("LPOP", "ccl:" + k)
        elif op == 5: c.cmd("ZADD", "ccz:" + k, rnd.randint(0, 9), rnd.randint(0, 9))
        elif op == 6: c.cmd("INCR", "cci:" + k)
        elif op == 7: c.cmd("SADD", "ccs:" + k, rnd.randint(0, 9))
        elif op == 8: c.cmd("EXPIRE", k, 1000)
        else: c.cmd("MULTI"); c.cmd("INCR", "cct"); c.cmd("RPUSH", "cctl", seed); c.cmd("EXEC")
ts = [threading.Thread(target=load, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
check("WAIT after concurrent load", p.cmd("WAIT", 1, 5000) == 1)
def dump(c):
    out = {}
    for k in sorted(c.cmd("KEYS", "cc*")):
        t = c.cmd("TYPE", k)
        v = {"string": lambda: c.cmd("GET", k), "hash": lambda: sorted(c.cmd("HGETALL", k)),
             "list": lambda: c.cmd("LRANGE", k, 0, -1), "set": lambda: sorted(c.cmd("SMEMBERS", k)),
             "zset": lambda: c.cmd("ZRANGE", k, 0, -1, "WITHSCORES")}[t]()
        out[k] = (t, v, c.cmd("TTL", k) > 0)
    return out
dp, dr = dump(p), dump(r)
check("replica converges to the primary exactly", dp == dr and len(dp) > 50,
      (len(dp), len(dr), [k for k in set(dp) | set(dr) if dp.get(k) != dr.get(k)][:5]))

print("== replica is read-only; ROLE/INFO")
check("write refused with READONLY", str(r.cmd("SET", "x", "1")).startswith("ERR:READONLY"))
check("MULTI with a write is aborted", (r.cmd("MULTI"), r.cmd("SET", "x", "1"), str(r.cmd("EXEC")).startswith("ERR:EXECABORT"))[2])
check("reads still work", r.cmd("GET", "s") == b"world")
check("ROLE on primary lists the replica", len(p.cmd("ROLE")[2]) == 1)
check("INFO replication", b"role:slave" in r.cmd("INFO") and b"connected_slaves:1" in p.cmd("INFO"))

print("== FLUSHALL propagates")
p.cmd("FLUSHALL"); p.cmd("SET", "after", "1")
check("replica wiped then follows", until(lambda: r.cmd("DBSIZE") == 1 and r.cmd("GET", "after") == b"1"))
check("replica vector index wiped", r.cmd("VSEARCH", 1, 1.0, 0.25, 0.0, 0.0) in ([], None))

print("== replica restart → full resync")
rep.kill(); rep.wait()
for i in range(100):
    p.cmd("SET", f"while_down:{i}", i)
rep = start(RP, rw, {"DBSTRIKE_REPLICAOF": f"127.0.0.1:{PP}"}); r = C(RP)
check("resynced keys written while down", until(lambda: r.cmd("GET", "while_down:99") == b"99"))

print("== primary restart → replica reconnects")
prim.kill(); prim.wait()
check("replica notices link down", until(lambda: r.cmd("ROLE")[3] == b"connect", 10))
prim = start(PP, pw); p = C(PP)
p.cmd("SET", "after_restart", "yes")
check("replica reconnects and catches up", until(lambda: r.cmd("GET", "after_restart") == b"yes", 15))

print("== promotion")
check("REPLICAOF NO ONE", r.cmd("REPLICAOF", "NO", "ONE") == "OK")
check("promoted node accepts writes", r.cmd("SET", "promoted", "1") == "OK" and r.cmd("ROLE")[0] == b"master")
check("promoted node keeps data", r.cmd("GET", "after_restart") == b"yes")

for x in (prim, rep):
    x.kill()
print(f"{P} passed, {F} failed")
sys.exit(1 if F else 0)
