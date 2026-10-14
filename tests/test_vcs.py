#!/usr/bin/env python3
"""Git-style version control (VC.*) on dbstrike.

Covers every data type through commit/checkout, diffs, fast-forward and
three-way merges (clean, conflicting, STRATEGY, DRYRUN), revert, reset,
restore, tags, revision syntax, time-travel reads, crash recovery, FLUSHALL,
and commits racing concurrent writers (nothing a commit should contain may
be missed).

    cargo build --release -p server --bin dbstrike
    python3 tests/test_vcs.py
"""
import os, socket, subprocess, sys, tempfile, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("DBSTRIKE_TEST_PORT", "6584"))
B = os.path.join(REPO, "target", "release", "dbstrike")
WAL = os.path.join(tempfile.mkdtemp(prefix="dbstrike_vcs_"), "v.wal")


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


def pairs(r):
    return dict(zip(r[::2], r[1::2])) if isinstance(r, list) else {}


def changes(c):
    st = pairs(c.cmd("VC.STATUS"))
    return {x[0]: x[1] for x in st.get(b"changes", [])}


def snapshot(c, keys):
    """Full logical state of some keys (type + contents + TTL presence)."""
    out = {}
    for k in keys:
        t = c.cmd("TYPE", k)
        if t == "string": v = c.cmd("GET", k)
        elif t == "hash": v = sorted(c.cmd("HGETALL", k))
        elif t == "list": v = c.cmd("LRANGE", k, 0, -1)
        elif t == "set": v = sorted(c.cmd("SMEMBERS", k))
        elif t == "zset": v = c.cmd("ZRANGE", k, 0, -1, "WITHSCORES")
        elif t == "stream": v = c.cmd("XRANGE", k, "-", "+")
        else: v = None
        out[k] = (t, v, c.cmd("TTL", k) > 0)
    return out


p = start(); c = C()
ALL = ["s", "h", "l", "st", "z", "x", "ttl", "gone"]

print("== Init, status, commit")
check("not initialized before VC.INIT", str(c.cmd("VC.STATUS")).startswith("ERR"))
c.cmd("SET", "s", "v1"); c.cmd("HSET", "h", "a", "1", "b", "2"); c.cmd("RPUSH", "l", "x", "y")
c.cmd("SADD", "st", "m1", "m2"); c.cmd("ZADD", "z", "1", "one", "2", "two"); c.cmd("XADD", "x", "1-1", "f", "v")
c.cmd("SET", "ttl", "t", "EX", "1000"); c.cmd("SET", "gone", "bye")
root = c.cmd("VC.INIT")
check("VC.INIT returns a commit id", isinstance(root, bytes) and len(root) == 16)
check("VC.INIT twice is an error", str(c.cmd("VC.INIT")).startswith("ERR"))
check("clean after init", changes(c) == {})
state_root = snapshot(c, ALL)
c.cmd("SET", "s", "v2"); c.cmd("HSET", "h", "c", "3"); c.cmd("LPOP", "l"); c.cmd("SREM", "st", "m1")
c.cmd("ZINCRBY", "z", "5", "one"); c.cmd("XADD", "x", "2-1", "g", "w"); c.cmd("PERSIST", "ttl"); c.cmd("DEL", "gone")
c.cmd("SET", "new", "n")
ch = changes(c)
check("status lists every changed key with its kind",
      ch == {b"s": b"modified", b"h": b"modified", b"l": b"modified", b"st": b"modified", b"z": b"modified",
             b"x": b"modified", b"ttl": b"modified", b"gone": b"deleted", b"new": b"added"}, ch)
c2 = c.cmd("VC.COMMIT", "second", "AUTHOR", "irfan")
check("commit after changes", isinstance(c2, bytes) and changes(c) == {})
check("nothing to commit is an error", str(c.cmd("VC.COMMIT", "empty")).startswith("ERR"))
check("ALLOWEMPTY commits anyway", isinstance(c.cmd("VC.COMMIT", "empty", "ALLOWEMPTY"), bytes))
state_c2 = snapshot(c, ALL + ["new"])

print("== Every type survives checkout round trips")
c.cmd("VC.BRANCH", "old", root)
r = c.cmd("VC.CHECKOUT", "old")
check("checkout old branch", pairs(r).get(b"branch") == b"old", r)
check("all types restored to the root commit (incl. TTL)", snapshot(c, ALL) == state_root, (snapshot(c, ALL), state_root))
check("key added later is absent", c.cmd("EXISTS", "new") == 0)
check("expiry index rebuilt: TTL still counts down", 990 < c.cmd("TTL", "ttl") <= 1000)
c.cmd("VC.CHECKOUT", "main")
check("back on main: state of the second commit", snapshot(c, ALL + ["new"]) == state_c2)
check("stream continues after checkout", c.cmd("XADD", "x", "3-*", "a", "b") == b"3-0")
c.cmd("VC.RESET", "HEAD", "HARD")
check("RESET HARD discards working changes", changes(c) == {} and c.cmd("XLEN", "x") == 2)

print("== Dirty working copy")
c.cmd("SET", "s", "dirty")
check("checkout refuses with uncommitted changes", str(c.cmd("VC.CHECKOUT", "old")).startswith("ERR"))
check("...and changed nothing", c.cmd("GET", "s") == b"dirty")
c.cmd("VC.CHECKOUT", "old", "FORCE")
check("FORCE discards them", c.cmd("GET", "s") == b"v1" and changes(c) == {})
c.cmd("VC.CHECKOUT", "main")

print("== Revisions, time travel, diff, history")
check("VC.GET at root", c.cmd("VC.GET", root, "s") == b"v1")
check("VC.GET HEAD~2 (first-parent walk)", c.cmd("VC.GET", "HEAD~2", "s") == b"v1")
check("VC.GET by id prefix", c.cmd("VC.GET", c2[:8], "s") == b"v2")
check("VC.GET of a collection is a type error", str(c.cmd("VC.GET", root, "h")).startswith("ERR"))
check("VC.GET of a missing key is nil", c.cmd("VC.GET", root, "new") is None)
check("unknown revision is an error", str(c.cmd("VC.GET", "nope", "s")).startswith("ERR"))
keys = c.cmd("VC.KEYS", root)
check("VC.KEYS at root", sorted(keys) == sorted(k.encode() for k in ALL), keys)
check("VC.KEYS with a pattern", c.cmd("VC.KEYS", "HEAD", "s*") == [b"s", b"st"])
d = {x[0]: x[1] for x in c.cmd("VC.DIFF", root, c2)}
check("VC.DIFF between commits", d.get(b"new") == b"added" and d.get(b"gone") == b"deleted" and d.get(b"s") == b"modified", d)
check("VC.DIFF with no changes", c.cmd("VC.DIFF", "HEAD", "HEAD") == [])
c.cmd("SET", "s", "wip")
check("VC.DIFF against the working copy", c.cmd("VC.DIFF") == [[b"s", b"modified", b"string"]])
c.cmd("VC.RESET", "HEAD", "HARD")
h = c.cmd("VC.HISTORY", "s")
check("VC.HISTORY lists the commits that changed a key", [pairs(e)[b"id"] for e in h] == [c2, root], h)
log = c.cmd("VC.LOG", "COUNT", "2")
check("VC.LOG newest first with COUNT", len(log) == 2 and pairs(log[1])[b"id"] == c2 and pairs(log[1])[b"author"] == b"irfan")

print("== Merges")
c.cmd("VC.BRANCH", "ff")
c.cmd("VC.CHECKOUT", "ff"); c.cmd("SET", "ffkey", "1"); ffc = c.cmd("VC.COMMIT", "ff work"); c.cmd("VC.CHECKOUT", "main")
r = pairs(c.cmd("VC.MERGE", "ff"))
check("fast-forward merge", r.get(b"result") == b"fast-forward" and r.get(b"commit") == ffc and c.cmd("GET", "ffkey") == b"1", r)
check("merging again is up to date", pairs(c.cmd("VC.MERGE", "ff")).get(b"result") == b"up-to-date")
# Three-way, clean: disjoint keys + same change on both sides.
c.cmd("VC.BRANCH", "f1")
c.cmd("SET", "mainonly", "m"); c.cmd("SET", "both", "same"); c.cmd("HSET", "h", "main", "1"); c.cmd("VC.COMMIT", "main side")
c.cmd("VC.CHECKOUT", "f1"); c.cmd("SET", "f1only", "f"); c.cmd("SET", "both", "same"); c.cmd("SADD", "st", "fromf1")
c.cmd("SET", "flip", "x"); c.cmd("DEL", "flip"); c.cmd("VC.COMMIT", "f1 side"); c.cmd("VC.CHECKOUT", "main")
dry = pairs(c.cmd("VC.MERGE", "f1", "DRYRUN"))
check("DRYRUN previews and changes nothing", dry.get(b"result") == b"merge" and c.cmd("EXISTS", "f1only") == 0, dry)
r = pairs(c.cmd("VC.MERGE", "f1"))
check("clean three-way merge commits", r.get(b"result") == b"merge" and isinstance(r.get(b"commit"), bytes), r)
check("both sides' changes present",
      c.cmd("GET", "mainonly") == b"m" and c.cmd("GET", "f1only") == b"f" and c.cmd("GET", "both") == b"same"
      and b"fromf1" in c.cmd("SMEMBERS", "st") and c.cmd("HGET", "h", "main") == b"1")
mlog = c.cmd("VC.LOG", "COUNT", "1")[0]
check("merge commit has two parents", len(pairs(mlog)[b"parents"]) == 2)
check("clean after merge", changes(c) == {})
# Conflicts.
c.cmd("VC.BRANCH", "f2")
c.cmd("SET", "s", "main-version"); c.cmd("VC.COMMIT", "main s")
c.cmd("VC.CHECKOUT", "f2"); c.cmd("SET", "s", "f2-version"); c.cmd("SET", "f2only", "y"); c.cmd("VC.COMMIT", "f2 s")
c.cmd("VC.CHECKOUT", "main")
before = c.cmd("VC.LOG", "COUNT", "1")
r = pairs(c.cmd("VC.MERGE", "f2"))
check("conflict reported, nothing changed",
      r.get(b"result") == b"conflict" and r.get(b"conflicts")[0][0] == b"s"
      and c.cmd("GET", "s") == b"main-version" and c.cmd("EXISTS", "f2only") == 0
      and c.cmd("VC.LOG", "COUNT", "1") == before, r)
r = pairs(c.cmd("VC.MERGE", "f2", "STRATEGY", "OURS"))
check("STRATEGY OURS keeps ours, takes the rest",
      r.get(b"result") == b"merge" and c.cmd("GET", "s") == b"main-version" and c.cmd("GET", "f2only") == b"y")
check("read through the merge commit sees the merged value", c.cmd("VC.GET", "HEAD", "s") == b"main-version")
c.cmd("VC.RESET", "HEAD~1", "HARD")
r = pairs(c.cmd("VC.MERGE", "f2", "STRATEGY", "THEIRS"))
check("STRATEGY THEIRS takes theirs", c.cmd("GET", "s") == b"f2-version" and c.cmd("VC.GET", "HEAD", "s") == b"f2-version")
c.cmd("SET", "s", "dirty")
check("merge refuses with uncommitted changes", str(c.cmd("VC.MERGE", "f1")).startswith("ERR"))
c.cmd("VC.RESET", "HEAD", "HARD")

print("== Revert, reset, restore, tags, branches")
c.cmd("SET", "rv", "1"); c.cmd("VC.COMMIT", "rv 1")
c.cmd("SET", "rv", "2"); c.cmd("SET", "rv2", "x"); bad = c.cmd("VC.COMMIT", "bad change")
rc = c.cmd("VC.REVERT", bad)
check("VC.REVERT undoes a commit as a new commit", isinstance(rc, bytes) and c.cmd("GET", "rv") == b"1" and c.cmd("EXISTS", "rv2") == 0)
check("revert commit message", pairs(c.cmd("VC.LOG", "COUNT", "1")[0])[b"message"].startswith(b"Revert"))
head = c.cmd("VC.LOG", "COUNT", "1")
c.cmd("VC.RESET", "HEAD~3")  # before "rv 1": rv did not exist
ch = changes(c)
check("soft RESET keeps the working copy, shows it as uncommitted", c.cmd("GET", "rv") == b"1" and ch.get(b"rv") == b"added", ch)
c.cmd("VC.RESET", pairs(head[0])[b"id"], "HARD")
check("RESET back to the newest commit", changes(c) == {})
c.cmd("SET", "s", "oops"); c.cmd("DEL", "h")
check("VC.RESTORE puts keys back from a revision", c.cmd("VC.RESTORE", "HEAD", "s", "h") == 2 and changes(c) == {})
check("VC.TAG and resolve by tag", c.cmd("VC.TAG", "v1.0") == "OK" and c.cmd("VC.GET", "v1.0", "rv") == b"1")
check("tag list", c.cmd("VC.TAG")[0][0] == b"v1.0")
br = {b[0]: b[2] for b in c.cmd("VC.BRANCH")}
check("branch list marks the current branch", br.get(b"main") == 1 and br.get(b"old") == 0)
check("cannot delete the current branch", str(c.cmd("VC.BRANCH", "DELETE", "main")).startswith("ERR"))
check("delete a branch", c.cmd("VC.BRANCH", "DELETE", "ff") == 1 and b"ff" not in {b[0] for b in c.cmd("VC.BRANCH")})
check("invalid branch name", str(c.cmd("VC.BRANCH", "bad name")).startswith("ERR"))

print("== Commits racing writers")
# Writes pipelined right before VC.COMMIT on the same connection must be in it.
c.s.sendall(b"".join(enc(["SET", f"race{i}", i]) for i in range(200)) + enc(["VC.COMMIT", "pipelined"]))
[c.rd() for _ in range(200)]; cid = c.rd()
check("pipelined writes before VC.COMMIT are all in the commit", changes(c) == {} and c.cmd("VC.GET", cid, "race199") == b"199")
stop = False
def writer(i):
    cc = C(); n = 0
    while not stop:
        cc.cmd("SET", f"w{i}", n); n += 1
ths = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
[t.start() for t in ths]
for j in range(10):
    c.cmd("VC.COMMIT", f"concurrent {j}", "ALLOWEMPTY")
stop = True; [t.join() for t in ths]
c.cmd("VC.COMMIT", "final", "ALLOWEMPTY")
mism = [i for i in range(8) if c.cmd("VC.GET", "HEAD", f"w{i}") != c.cmd("GET", f"w{i}")]
check("after concurrent writers + commit, HEAD equals the live data", changes(c) == {} and not mism, mism)

print("== Crash recovery")
c.cmd("SET", "afterlast", "uncommitted")
p.kill(); p.wait(); p = start(); c = C()
ch = changes(c)
check("repo survives kill -9; uncommitted change found after restart", ch == {b"afterlast": b"added"}, ch)
check("history readable after restart", c.cmd("VC.GET", root, "s") == b"v1")
check("commit after restart", isinstance(c.cmd("VC.COMMIT", "after restart"), bytes) and changes(c) == {})
check("branches survive", b"old" in {b[0] for b in c.cmd("VC.BRANCH")})

print("== ACL key patterns")
c.cmd("SET", "secret", "s3cr3t"); c.cmd("SET", "app:x", "1"); c.cmd("VC.COMMIT", "acl data")
check("ACL SETUSER", c.cmd("ACL", "SETUSER", "bob", "on", ">pw", "~app:*", "+@read", "+@write") == "OK")
bob = C(); bob.cmd("AUTH", "bob", "pw")
check("restricted user reads allowed keys through VC.GET", bob.cmd("VC.GET", "HEAD", "app:x") == b"1")
check("...but not other keys", str(bob.cmd("VC.GET", "HEAD", "secret")).startswith("ERR:NOPERM"))
check("VC.RESTORE checks keys too", str(bob.cmd("VC.RESTORE", "HEAD", "secret")).startswith("ERR:NOPERM"))
check("keyspace-wide rewrites need admin", str(bob.cmd("VC.CHECKOUT", "old")).startswith("ERR:NOPERM")
      and str(bob.cmd("VC.RESET", "HEAD~1", "HARD")).startswith("ERR:NOPERM"))
c.cmd("ACL", "DELUSER", "bob")

print("== FLUSHALL")
c.cmd("FLUSHALL")
check("FLUSHALL removes the repository too", str(c.cmd("VC.STATUS")).startswith("ERR"))
c.cmd("SET", "fresh", "1")
check("re-init after FLUSHALL", isinstance(c.cmd("VC.INIT"), bytes) and c.cmd("VC.KEYS", "HEAD") == [b"fresh"])

p.kill()
print(f"{P} passed, {F} failed")
sys.exit(1 if F else 0)
