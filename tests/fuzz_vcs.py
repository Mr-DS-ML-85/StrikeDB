#!/usr/bin/env python3
"""Randomized VC.* invariant test.

Random writes over all data types interleaved with commits, branches,
checkouts, merges (STRATEGY THEIRS/OURS), reverts and resets. Every time a
commit is created, the full logical state of the keyspace is recorded;
later `VC.RESET <id> HARD` to any recorded commit must reproduce that
state exactly, and `VC.STATUS` must then be clean.

    python3 tests/fuzz_vcs.py [seed] [steps]
"""
import os, random, sys, tempfile, time, subprocess
os.environ.setdefault("DBSTRIKE_TEST_PORT", "6585")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ["DBSTRIKE_TEST_PORT"])
B = os.path.join(REPO, "target", "release", "dbstrike")
WAL = os.path.join(tempfile.mkdtemp(prefix="dbstrike_vcsfuzz_"), "f.wal")

import socket
def enc(a):
    a = [x if isinstance(x, bytes) else str(x).encode() for x in a]
    return b"*%d\r\n" % len(a) + b"".join(b"$%d\r\n%s\r\n" % (len(x), x) for x in a)
class C:
    def __init__(s): s.s = socket.create_connection(("127.0.0.1", PORT)); s.f = s.s.makefile("rb")
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
        raise RuntimeError(repr(l))
    def cmd(s, *a): s.s.sendall(enc(a)); return s.rd()

KEYS = [f"k{i}" for i in range(8)]
def state(c):
    out = {}
    for k in KEYS:
        t = c.cmd("TYPE", k)
        if t == "string": v = c.cmd("GET", k)
        elif t == "hash": v = sorted(c.cmd("HGETALL", k))
        elif t == "list": v = c.cmd("LRANGE", k, 0, -1)
        elif t == "set": v = sorted(c.cmd("SMEMBERS", k))
        elif t == "zset": v = c.cmd("ZRANGE", k, 0, -1, "WITHSCORES")
        elif t == "stream": v = c.cmd("XRANGE", k, "-", "+")
        else: v = None
        out[k] = (t, repr(v))
    return out

def rand_write(c, r):
    k = r.choice(KEYS); m = r.choice("abcde"); v = str(r.randint(0, 9))
    op = r.randrange(11)
    if op == 0: c.cmd("SET", k, v)
    elif op == 1: c.cmd("DEL", k)
    elif op == 2: c.cmd("HSET", k, m, v)
    elif op == 3: c.cmd("HDEL", k, m)
    elif op == 4: c.cmd("RPUSH", k, v)
    elif op == 5: c.cmd("LPOP", k)
    elif op == 6: c.cmd("SADD", k, m)
    elif op == 7: c.cmd("SREM", k, m)
    elif op == 8: c.cmd("ZADD", k, v, m)
    elif op == 9: c.cmd("XADD", k, "*", m, v)
    else: c.cmd("APPEND", k, v)

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    r = random.Random(seed)
    p = subprocess.Popen([B, f"127.0.0.1:{PORT}"], env={**os.environ, "DBSTRIKE_WAL": WAL}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try: C().cmd("PING"); break
        except OSError: time.sleep(0.05)
    c = C()
    recorded = {}
    def commit_point(cid):
        if isinstance(cid, bytes):
            recorded[cid] = state(c)
    commit_point(c.cmd("VC.INIT"))
    branches = ["main"]
    failures = 0
    def clean():
        st = c.cmd("VC.STATUS")
        return dict(zip(st[::2], st[1::2]))[b"changes"] == []
    for step in range(steps):
        x = r.random()
        if x < 0.55:
            rand_write(c, r)
        elif x < 0.70:
            commit_point(c.cmd("VC.COMMIT", f"s{step}", "ALLOWEMPTY"))
        elif x < 0.75:
            name = f"b{step}"
            if c.cmd("VC.BRANCH", name) == "OK": branches.append(name)
        elif x < 0.83:
            commit_point(c.cmd("VC.COMMIT", f"pre-co {step}", "ALLOWEMPTY"))
            c.cmd("VC.CHECKOUT", r.choice(branches))
        elif x < 0.91:
            commit_point(c.cmd("VC.COMMIT", f"pre-merge {step}", "ALLOWEMPTY"))
            res = c.cmd("VC.MERGE", r.choice(branches), "STRATEGY", r.choice(["THEIRS", "OURS"]))
            d = dict(zip(res[::2], res[1::2])) if isinstance(res, list) else {}
            if isinstance(d.get(b"commit"), bytes):
                commit_point(d[b"commit"])
            if not clean():
                failures += 1; print("FAIL: dirty after merge", step, c.cmd("VC.STATUS"))
        elif x < 0.95:
            commit_point(c.cmd("VC.COMMIT", f"pre-revert {step}", "ALLOWEMPTY"))
            log = c.cmd("VC.LOG", "COUNT", "5")
            if len(log) > 1:
                rv = c.cmd("VC.REVERT", dict(zip(log[0][::2], log[0][1::2]))[b"id"])
                commit_point(rv)
        else:
            # Check a random recorded commit reproduces exactly.
            cid = r.choice(list(recorded))
            c.cmd("VC.COMMIT", f"pre-check {step}", "ALLOWEMPTY")
            c.cmd("VC.RESET", cid, "HARD")
            got = state(c)
            if got != recorded[cid]:
                failures += 1
                diff = {k: (recorded[cid][k], got[k]) for k in KEYS if recorded[cid][k] != got[k]}
                print("FAIL: commit", cid, "does not reproduce at step", step, diff)
            if not clean():
                failures += 1; print("FAIL: dirty after reset", step)
    # Final sweep over every recorded commit.
    for cid, want in recorded.items():
        c.cmd("VC.RESET", cid, "HARD")
        if state(c) != want:
            failures += 1; print("FAIL (sweep):", cid)
    p.kill()
    print(f"seed {seed}: {len(recorded)} commits checked, {failures} failures")
    sys.exit(1 if failures else 0)

main()
