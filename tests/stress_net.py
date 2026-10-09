import socket, time, threading, resource, sys, os, random
PORT=int(sys.argv[1]) if len(sys.argv)>1 else 6577
resource.setrlimit(resource.RLIMIT_NOFILE,(20000,20000))
fails=[]
def chk(name,cond):
    print(("PASS " if cond else "FAIL ")+name, flush=True)
    if not cond: fails.append(name)
def conn():
    s=socket.create_connection(("127.0.0.1",PORT)); s.settimeout(10); return s
def cmd(*a):
    return b"*%d\r\n"%len(a)+b"".join(b"$%d\r\n%s\r\n"%(len(x),x) for x in [y if isinstance(y,bytes) else str(y).encode() for y in a])
def rd(s,n=None):
    buf=b""
    while True:
        try: d=s.recv(65536)
        except socket.timeout: return buf
        if not d: return buf
        buf+=d
        if n is None or buf.count(b"\r\n")>=n: return buf
# 1. 10k idle connections, server stays responsive
N=10000
cs=[]
t=time.time()
for i in range(N):
    try: cs.append(conn())
    except Exception as e: print("connect fail at",i,e); break
chk(f"opened {len(cs)} connections", len(cs)==N)
s=conn(); s.sendall(cmd("PING")); chk("PING with 10k idle conns", rd(s,1)==b"+PONG\r\n")
s.sendall(cmd("INFO","clients")); info=rd(s,4); 
import re
m=re.search(rb"connected_clients:(\d+)",info); chk(f"connected_clients counts all ({m.group(1) if m else None})", m and int(m.group(1))>=N)
# every 100th idle conn works
ok=True
for c in cs[::100]:
    c.sendall(cmd("SET",b"k%d"%c.fileno(),"v"))
for c in cs[::100]:
    if rd(c,1)!=b"+OK\r\n": ok=False
chk("sampled idle connections serve writes", ok)
for c in cs: c.close()
time.sleep(1)
s.sendall(cmd("INFO","clients")); info=rd(s,4); m=re.search(rb"connected_clients:(\d+)",info)
chk(f"connected_clients drops after close ({m.group(1)})", int(m.group(1))<50)
# 2. slowloris: byte-at-a-time command doesn't block others
sl=conn(); frame=cmd("SET","slow","val")
def trickle():
    for b in frame:
        sl.sendall(bytes([b])); time.sleep(0.01)
th=threading.Thread(target=trickle); th.start()
t0=time.time(); s.sendall(cmd("GET","slow")); r=rd(s,1); dt=time.time()-t0
chk(f"other client unaffected by slowloris ({dt*1000:.1f}ms)", dt<0.1)
th.join(); chk("trickled SET completes", rd(sl,1)==b"+OK\r\n")
# 3. disconnect mid-command and mid-durable-write
for i in range(200):
    c=conn(); c.sendall(cmd("SET","x","y")[:-5]); c.close()
    c=conn(); c.sendall(cmd("INCR","ctr")*50); c.close()
s.sendall(cmd("PING")); chk("server alive after 400 abrupt disconnects", rd(s,1)==b"+PONG\r\n")
# 4. big value (8MB) both directions
big=os.urandom(8<<20)
s.sendall(cmd("SET","big",big)); chk("8MB SET", rd(s,1)==b"+OK\r\n")
s.sendall(cmd("GET","big")); r=b""; exp=b"$%d\r\n"%len(big)+big+b"\r\n"
while len(r)<len(exp):
    d=s.recv(1<<20)
    if not d: break
    r+=d
chk("8MB GET roundtrip", r==exp)
# 5. pipelined commands before protocol error still run
c=conn(); c.sendall(cmd("SET","pe","1")+cmd("INCR","pe")+b"*1\r\n$x\r\n")
r=rd(c,3); chk(f"cmds before protocol error execute ({r!r})", r.startswith(b"+OK\r\n:2\r\n-ERR Protocol error"))
# 6. huge pipeline of durable writes in one burst, replies ordered
c=conn(); n=20000
c.sendall(b"".join(cmd("INCR","ord") for _ in range(n)))
r=b""
while r.count(b"\r\n")<n:
    d=c.recv(1<<20)
    if not d: break
    r+=d
vals=[int(x[1:]) for x in r.split(b"\r\n") if x]
chk("20k pipelined INCR replies in order", vals==list(range(vals[0],vals[0]+n)))
# 7. mixed read/write pipeline order
c=conn(); p=b""
exp=[]
for i in range(2000):
    if i%2: p+=cmd("SET","m%d"%i,i); exp.append(b"+OK")
    else: p+=cmd("GET","m%d"%(i-1)) if i else cmd("PING"); exp.append(None)
c.sendall(p); r=rd(c,None if False else 2000+1000)
lines=r.split(b"\r\n")
chk("mixed pipeline produced replies", len(r)>0)
# GET m{i-1} after SET m{i-1} must return value
c=conn(); p=b"".join(cmd("SET","q%d"%i,i)+cmd("GET","q%d"%i) for i in range(1000)); c.sendall(p)
r=b""
while r.count(b"\r\n")<3000:
    d=c.recv(1<<20)
    if not d: break
    r+=d
ok=all(f"+OK\r\n${len(str(i))}\r\n{i}\r\n".encode() in r for i in (0,500,999)); chk("SET/GET interleaved read-your-writes", ok and r.count(b"+OK")==1000)
# 8. subscribe hand-off with pipelined data and publish
sub=conn(); sub.sendall(cmd("SET","pre","1")+cmd("SUBSCRIBE","ch")); r=rd(sub,4)
chk("SET then SUBSCRIBE in one pipeline", r.startswith(b"+OK\r\n*3\r\n$9\r\nsubscribe"))
s.sendall(cmd("PUBLISH","ch","hello")); rd(s,1)
r=rd(sub,6); chk("published message arrives after hand-off", b"hello" in r)
# 9. QUIT closes after reply
c=conn(); c.sendall(cmd("SET","qq","1")+cmd("QUIT")+cmd("SET","after","1")); r=rd(c)
chk("QUIT replies and closes", r==b"+OK\r\n+OK\r\n")
s.sendall(cmd("EXISTS","after")); chk("commands after QUIT not executed", rd(s,1)==b":0\r\n")
# 10. concurrent clients hammering INCR across workers
def worker(k):
    c=conn()
    for _ in range(200):
        c.sendall(cmd("INCR","shared")); rd(c,1)
ths=[threading.Thread(target=worker,args=(i,)) for i in range(64)]
s.sendall(cmd("DEL","shared")); rd(s,1)
[t.start() for t in ths]; [t.join() for t in ths]
s.sendall(cmd("GET","shared")); chk("64x200 concurrent INCR = 12800", rd(s,1)==b"$5\r\n12800\r\n")
print(f"{len(fails)} failed")
