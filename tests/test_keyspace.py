"""Keyspace semantics: TTL expiry (lazy + active), MULTI/EXEC/WATCH
atomicity under concurrency, collection correctness under concurrent
clients, BLPOP, and durability of collections/TTLs/EXEC blocks across
kill -9. Stdlib only.

    cargo build --release && python3 tests/test_keyspace.py
"""
import socket, time, threading, subprocess, os, sys, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("DBSTRIKE_TEST_PORT", "6578"))
B = os.path.join(REPO, "target", "release", "dbstrike")
WAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.mkdtemp(prefix="dbstrike_ks_"), "ks.wal")
def enc(a):
    a=[str(x).encode() if not isinstance(x,bytes) else x for x in a]
    return b"*%d\r\n"%len(a)+b"".join(b"$%d\r\n%s\r\n"%(len(x),x) for x in a)
class C:
    def __init__(s): s.s=socket.create_connection(("127.0.0.1",PORT)); s.f=s.s.makefile("rb")
    def rd(s):
        l=s.f.readline(); t=l[:1]; b=l[1:-2]
        if t==b"+": return b.decode()
        if t==b"-": return "ERR:"+b.decode()
        if t==b":": return int(b)
        if t==b"$":
            n=int(b); return None if n<0 else s.f.read(n+2)[:-2]
        if t==b"*":
            n=int(b); return None if n<0 else [s.rd() for _ in range(n)]
    def cmd(s,*a): s.s.sendall(enc(a)); return s.rd()
P=F=0
def check(name, ok, info=""):
    global P,F
    if ok: P+=1; print("  PASS", name)
    else: F+=1; print("  FAIL", name, info)
def start():
    p=subprocess.Popen([B,f"127.0.0.1:{PORT}"],env={**os.environ,"DBSTRIKE_WAL":WAL},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for _ in range(100):
        try: C().cmd("PING"); return p
        except OSError: time.sleep(0.05)
for f in (WAL, WAL+".snap"):
    try: os.remove(f)
    except OSError: pass
p=start(); c=C()
print("== TTL timing")
c.cmd("SET","t1","v","PX",300); c.cmd("HSET","t2","f","v"); c.cmd("PEXPIRE","t2",300); c.cmd("RPUSH","t3","a"); c.cmd("PEXPIRE","t3",300)
check("alive before deadline", c.cmd("EXISTS","t1","t2","t3")==3)
check("PTTL counts down", 0 < c.cmd("PTTL","t1") <= 300)
time.sleep(0.45)
check("GET after deadline is nil (lazy)", c.cmd("GET","t1") is None)
check("EXISTS after deadline 0", c.cmd("EXISTS","t1","t2","t3")==0)
check("expired keys gone from KEYS", c.cmd("KEYS","t*")==[])
c.cmd("SET","t4","v","PX",200); time.sleep(0.5)
check("active expiry reaped untouched key (DBSIZE)", c.cmd("DBSIZE")==0, c.cmd("DBSIZE"))
c.cmd("SET","t5","v","EX",1000); c.cmd("SET","t5","w")
check("plain SET clears TTL", c.cmd("TTL","t5")==-1)
c.cmd("SET","c","5","PX",200); time.sleep(0.3)
check("INCR on expired key restarts at 1", c.cmd("INCR","c")==1)
check("...and has no TTL", c.cmd("TTL","c")==-1)
print("== MULTI/EXEC/WATCH")
check("MULTI", c.cmd("MULTI")=="OK")
check("queued", c.cmd("SET","m","1")=="QUEUED" and c.cmd("INCR","m")=="QUEUED" and c.cmd("LPUSH","ml","a","b")=="QUEUED")
r=c.cmd("EXEC"); check("EXEC results", r==["OK",2,2], r)
check("EXEC without MULTI", str(c.cmd("EXEC")).startswith("ERR:"))
c.cmd("MULTI"); c.cmd("SET","x","1"); c.cmd("DISCARD"); check("DISCARD drops queue", c.cmd("GET","x") is None)
c.cmd("MULTI"); r=c.cmd("VADD","1","0.1"); c.cmd("SET","x","1"); r2=c.cmd("EXEC")
check("non-keyspace cmd refused in MULTI → EXECABORT", str(r).startswith("ERR:") and "EXECABORT" in str(r2), (r,r2))
check("aborted EXEC wrote nothing", c.cmd("GET","x") is None)
c.cmd("MULTI"); c.cmd("SET","s","str"); c.cmd("LPUSH","s","x"); c.cmd("SET","after","1"); r=c.cmd("EXEC")
check("runtime error doesn't abort the rest (Redis semantics)", r[0]=="OK" and "WRONGTYPE" in r[1] and r[2]=="OK", r)
c2=C()
c.cmd("SET","w","1"); c.cmd("WATCH","w"); c2.cmd("SET","w","2"); c.cmd("MULTI"); c.cmd("SET","w","3"); r=c.cmd("EXEC")
check("WATCH: concurrent change aborts EXEC (nil)", r is None, r)
check("...and value is the other client's", c.cmd("GET","w")==b"2")
c.cmd("WATCH","w"); c.cmd("MULTI"); c.cmd("SET","w","4"); r=c.cmd("EXEC")
check("WATCH: untouched key → EXEC runs", r==["OK"], r)
c.cmd("WATCH","w"); c.cmd("UNWATCH"); c2.cmd("SET","w","5"); c.cmd("MULTI"); c.cmd("SET","w","6"); r=c.cmd("EXEC")
check("UNWATCH clears watches", r==["OK"], r)
c.cmd("WATCH","nk"); c2.cmd("HSET","nk","f","v"); c.cmd("MULTI"); c.cmd("GET","x"); r=c.cmd("EXEC")
check("WATCH a missing key that gets created → abort", r is None, r)
# pipelined MULTI/EXEC in one write
c.s.sendall(enc(["MULTI"])+enc(["INCR","pc"])+enc(["INCR","pc"])+enc(["EXEC"]))
r=[c.rd() for _ in range(4)]; check("pipelined MULTI block", r==["OK","QUEUED","QUEUED",[1,2]], r)
print("== Atomicity under concurrency")
c.cmd("DEL","acct:a","acct:b"); c.cmd("SET","acct:a","1000"); c.cmd("SET","acct:b","0")
def mover():
    cc=C()
    for _ in range(100):
        cc.cmd("MULTI"); cc.cmd("DECRBY","acct:a","1"); cc.cmd("INCRBY","acct:b","1"); cc.cmd("EXEC")
seen_bad=[]
def auditor():
    cc=C()
    for _ in range(300):
        cc.cmd("MULTI"); cc.cmd("GET","acct:a"); cc.cmd("GET","acct:b"); r=cc.cmd("EXEC")
        if int(r[0])+int(r[1])!=1000: seen_bad.append(r)
ts=[threading.Thread(target=mover) for _ in range(5)]+[threading.Thread(target=auditor)]
[t.start() for t in ts]; [t.join() for t in ts]
check("transfers conserve the total under concurrency", int(c.cmd("GET","acct:a"))+int(c.cmd("GET","acct:b"))==1000 and c.cmd("GET","acct:b")==b"500")
check("readers in EXEC never see a half-applied transfer", not seen_bad, seen_bad[:3])
c.cmd("DEL","q")
def pusher(n):
    cc=C()
    for i in range(200): cc.cmd("RPUSH","q",f"{n}-{i}")
ts=[threading.Thread(target=pusher,args=(n,)) for n in range(5)]; [t.start() for t in ts]; [t.join() for t in ts]
check("concurrent RPUSH: no lost elements", c.cmd("LLEN","q")==1000)
got=[]
def popper():
    cc=C()
    while True:
        v=cc.cmd("LPOP","q")
        if v is None: return
        got.append(v)
ts=[threading.Thread(target=popper) for _ in range(5)]; [t.start() for t in ts]; [t.join() for t in ts]
check("concurrent LPOP: every element exactly once", len(got)==1000 and len(set(got))==1000, len(got))
def hinc():
    cc=C()
    for _ in range(200): cc.cmd("HINCRBY","hc","f","1")
ts=[threading.Thread(target=hinc) for _ in range(5)]; [t.start() for t in ts]; [t.join() for t in ts]
check("concurrent HINCRBY exact", c.cmd("HGET","hc","f")==b"1000")
print("== BLPOP")
res=[]
def blocker():
    cc=C(); res.append(cc.cmd("BLPOP","bq","2"))
t=threading.Thread(target=blocker); t.start(); time.sleep(0.2); c.cmd("RPUSH","bq","hello"); t.join()
check("BLPOP wakes on push", res==[[b"bq",b"hello"]], res)
t0=time.time(); r=c.cmd("BLPOP","nothing","0.3"); check("BLPOP times out → nil", r is None and 0.25<time.time()-t0<1.5)
print("== Durability across kill -9")
c.cmd("FLUSHALL")
c.cmd("HSET","dh","a","1","b","2"); c.cmd("RPUSH","dl","x","y","z"); c.cmd("SADD","ds","m1","m2"); c.cmd("ZADD","dz","1","one","2","two")
c.cmd("SET","dt","v","EX","1000"); c.cmd("SET","dshort","v","PX","300"); c.cmd("MULTI"); c.cmd("SET","dm","1"); c.cmd("RPUSH","dl","w"); c.cmd("EXEC")
p.kill(); p.wait(); time.sleep(0.4); p=start(); c=C()
check("hash survives", sorted(c.cmd("HGETALL","dh"))==sorted([b"a",b"1",b"b",b"2"]))
check("list survives in order", c.cmd("LRANGE","dl","0","-1")==[b"x",b"y",b"z",b"w"])
check("set survives", sorted(c.cmd("SMEMBERS","ds"))==[b"m1",b"m2"])
check("zset survives", c.cmd("ZRANGE","dz","0","-1","WITHSCORES")==[b"one",b"1",b"two",b"2"])
check("TTL survives restart", 990 < c.cmd("TTL","dt") <= 1000, c.cmd("TTL","dt"))
check("key that expired while down is gone", c.cmd("GET","dshort") is None and c.cmd("EXISTS","dshort")==0)
check("EXEC block durable", c.cmd("GET","dm")==b"1")
check("TYPE after restart", [c.cmd("TYPE",k) for k in ("dh","dl","ds","dz","dt")]==["hash","list","set","zset","string"])
print("== Mixed: SET over a collection via pipelined fast path")
c.cmd("RPUSH","mix","a","b")
c.s.sendall(enc(["SET","mix","str"])+enc(["SET","other","1"])); r=[c.rd(),c.rd()]
check("pipelined SET over list replaces it", r==["OK","OK"] and c.cmd("GET","mix")==b"str" and c.cmd("TYPE","mix")=="string")
c.cmd("RPUSH","mix2","a"); c.cmd("SET","mix2","s")
check("no orphaned elements after overwrite (RPUSH starts fresh)", c.cmd("DEL","mix2")==1 and c.cmd("RPUSH","mix2","n")==1 and c.cmd("LRANGE","mix2","0","-1")==[b"n"])
p.kill()
print(f"{P} passed, {F} failed")
sys.exit(1 if F else 0)
