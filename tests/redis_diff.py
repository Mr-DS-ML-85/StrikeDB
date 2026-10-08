"""Differential test against a real redis-server.

Generates random command sequences over a tiny key space (so types collide,
keys expire and edge-case indexes come up constantly), sends each command to
both servers and compares replies. Reply order Redis leaves unspecified
(SMEMBERS, HGETALL, KEYS, ...) is normalized; errors compare by code.

    redis-server --port 6390 --save "" --daemonize yes
    DBSTRIKE_WAL=/tmp/d.wal ./target/release/dbstrike 127.0.0.1:6577 &
    python3 tests/redis_diff.py 6390 6577 [seeds] [commands-per-seed]
"""
import random, shlex, socket, sys

K=["k1","k2","k3","k4"]; M=["a","b","c","d","e"]; V=["1","2","-3","x","10","0","3.5"]
def k(): return random.choice(K)
def m(): return random.choice(M)
def v(): return random.choice(V)
def i(): return str(random.randint(-6,6))
def sc(): return random.choice(["1","2","-1","0","2.5","+inf","-inf","(1","(2","3"])
T=[
 lambda: f"SET {k()} {v()}", lambda: f"GET {k()}", lambda: f"DEL {k()} {k()}", lambda: f"EXISTS {k()}",
 lambda: f"TYPE {k()}", lambda: f"INCR {k()}", lambda: f"INCRBY {k()} {i()}", lambda: f"APPEND {k()} {v()}",
 lambda: f"STRLEN {k()}", lambda: f"GETRANGE {k()} {i()} {i()}", lambda: f"SETNX {k()} {v()}", lambda: f"GETSET {k()} {v()}",
 lambda: f"MSET {k()} {v()} {k()} {v()}", lambda: f"MGET {k()} {k()}", lambda: f"RENAME {k()} {k()}",
 lambda: f"HSET {k()} {m()} {v()}", lambda: f"HGET {k()} {m()}", lambda: f"HDEL {k()} {m()}", lambda: f"HGETALL {k()}",
 lambda: f"HLEN {k()}", lambda: f"HINCRBY {k()} {m()} {i()}", lambda: f"HEXISTS {k()} {m()}", lambda: f"HSETNX {k()} {m()} {v()}",
 lambda: f"LPUSH {k()} {v()} {v()}", lambda: f"RPUSH {k()} {v()}", lambda: f"LPOP {k()}", lambda: f"RPOP {k()}",
 lambda: f"LPOP {k()} {random.randint(0,3)}", lambda: f"LRANGE {k()} {i()} {i()}", lambda: f"LLEN {k()}", lambda: f"LINDEX {k()} {i()}",
 lambda: f"LSET {k()} {i()} {v()}", lambda: f"LTRIM {k()} {i()} {i()}", lambda: f"LREM {k()} {i()} {v()}",
 lambda: f"LINSERT {k()} {random.choice(['BEFORE','AFTER'])} {v()} {v()}", lambda: f"RPOPLPUSH {k()} {k()}",
 lambda: f"SADD {k()} {m()} {m()}", lambda: f"SREM {k()} {m()}", lambda: f"SISMEMBER {k()} {m()}", lambda: f"SMEMBERS {k()}",
 lambda: f"SCARD {k()}", lambda: f"SINTER {k()} {k()}", lambda: f"SUNION {k()} {k()}", lambda: f"SDIFF {k()} {k()}",
 lambda: f"SUNIONSTORE {k()} {k()} {k()}", lambda: f"SMOVE {k()} {k()} {m()}",
 lambda: f"ZADD {k()} {random.choice(['','NX','XX','GT','LT','CH'])} {random.choice(['1','2','-1','0','2.5'])} {m()}",
 lambda: f"ZINCRBY {k()} {i()} {m()}", lambda: f"ZREM {k()} {m()}", lambda: f"ZSCORE {k()} {m()}", lambda: f"ZCARD {k()}",
 lambda: f"ZRANGE {k()} {i()} {i()} WITHSCORES", lambda: f"ZREVRANGE {k()} {i()} {i()}", lambda: f"ZRANK {k()} {m()}",
 lambda: f"ZREVRANK {k()} {m()}", lambda: f"ZRANGEBYSCORE {k()} {sc()} {sc()} WITHSCORES", lambda: f"ZREVRANGEBYSCORE {k()} {sc()} {sc()}",
 lambda: f"ZCOUNT {k()} {sc()} {sc()}", lambda: f"ZPOPMIN {k()}", lambda: f"ZPOPMAX {k()} 2", lambda: f"ZREMRANGEBYRANK {k()} {i()} {i()}",
 lambda: f"ZREMRANGEBYSCORE {k()} {sc()} {sc()}", lambda: f"ZRANGEBYSCORE {k()} -inf +inf LIMIT {random.randint(0,3)} {random.randint(-1,3)}",
 lambda: f"EXPIRE {k()} 100", lambda: f"TTL {k()}", lambda: f"PERSIST {k()}", lambda: f"SET {k()} {v()} EX 100",
 lambda: f"PEXPIRE {k()} {random.randint(200, 10**6) * 1000} {random.choice(['','NX','XX','GT','LT'])}", lambda: f"SET {k()} {v()} KEEPTTL",
]

def generate(seed, n):
    random.seed(seed)
    return ["FLUSHALL"] + [random.choice(T)() for _ in range(n)]

def enc(a):
    a=[x.encode() if isinstance(x,str) else x for x in a]
    return b"*%d\r\n"%len(a)+b"".join(b"$%d\r\n%s\r\n"%(len(x),x) for x in a)
class C:
    def __init__(s,port): s.s=socket.create_connection(("127.0.0.1",port)); s.f=s.s.makefile("rb")
    def rd(s):
        l=s.f.readline(); t=l[:1]; b=l[1:-2]
        if t in (b"+",): return ("S",b.decode())
        if t==b"-": return ("E",b.decode().split()[0])
        if t==b":": return int(b)
        if t==b"$":
            n=int(b)
            if n<0: return None
            d=s.f.read(n+2)[:-2]; return d
        if t in (b"*",b">"):
            n=int(b)
            if n<0: return None
            return [s.rd() for _ in range(n)]
        if t==b"_": return None
        if t==b",": return float(b)
        raise Exception("bad "+repr(l))
    def cmd(s,*a): s.s.sendall(enc(a)); return s.rd()
UNORDERED={"SMEMBERS","SINTER","SUNION","SDIFF","KEYS","HKEYS","HVALS","SPOP","SRANDMEMBER"}
PAIRS={"HGETALL"}
def norm(cmd,r):
    c=cmd[0].upper()
    if isinstance(r,list) and c in UNORDERED: return sorted(map(repr,r))
    if isinstance(r,list) and c in PAIRS: return sorted(zip(r[::2],r[1::2]))
    return r

def main():
    rport, sport = int(sys.argv[1]), int(sys.argv[2])
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    # TTL durations are drawn from a wide range (whole seconds, never the
    # EX 100 base) so GT/LT outcomes can't hinge on which
    # millisecond each server happened to execute in.
    # PIPELINE=k sends DB-Strike the sequence in pipelined chunks of k
    # (exercising its batched write paths); Redis still gets it one by one.
    import os
    pipe = int(os.environ.get("PIPELINE", "1"))
    a, b = C(rport), C(sport)
    bad = total = 0
    for seed in range(1, seeds + 1):
        lines = generate(seed, n)
        for off in range(0, len(lines), pipe):
            chunk = [shlex.split(l) for l in lines[off:off + pipe]]
            ras = [a.cmd(*p) for p in chunk]
            b.s.sendall(b"".join(enc(p) for p in chunk))
            rbs = [b.rd() for _ in chunk]
            for line, p, ra, rb in zip(lines[off:off + pipe], chunk, ras, rbs):
                ra, rb = norm(p, ra), norm(p, rb)
                total += 1
                if ra != rb:
                    bad += 1
                    print(f"DIFF seed={seed}  {line}\n   redis   : {ra!r}\n   strikedb: {rb!r}")
            if bad >= 20:
                break
    print(f"{total} commands, {bad} differences")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
