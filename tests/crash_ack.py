import socket,subprocess,os,time,threading,random,sys,tempfile
B="/home/user/StrikeDB/target/release/dbstrike"; PORT=6581
D=tempfile.mkdtemp(); WAL=D+"/w.wal"
def start():
    p=subprocess.Popen([B,"127.0.0.1:%d"%PORT],env=dict(os.environ,DBSTRIKE_WAL=WAL,DBSTRIKE_CHECKPOINT_MB="1"),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for _ in range(100):
        try: socket.create_connection(("127.0.0.1",PORT)).close(); return p
        except OSError: time.sleep(0.05)
    raise SystemExit("noup")
def enc(*a): return b"*%d\r\n"%len(a)+b"".join(b"$%d\r\n%s\r\n"%(len(x),x) for x in [y if isinstance(y,bytes) else str(y).encode() for y in a])
def q(s,*a):
    s.sendall(enc(*a)); return s.recv(1<<20)
bad=0
hbase={}
for rnd in range(6):
    p=start()
    acked={}; stop=False
    def w(i):
        s=socket.create_connection(("127.0.0.1",PORT)); f=s.makefile("rb")
        n=hbase.get(i,0)
        while not stop:
            try:
                pipe=random.choice([1,1,8])
                s.sendall(b"".join(enc("RPUSH","L%d"%i,"x")+enc("INCR","C%d"%i)+enc("HSET","H%d"%i,"f%d"%(n+j),"v") for j in range(pipe)))
                for _ in range(pipe):
                    l=f.readline(); c=f.readline(); h=f.readline()
                    if not c.startswith(b":"): raise Exception
                    n+=1; acked[i]=(int(l[1:]),int(c[1:]),n)
            except Exception: return
    ths=[threading.Thread(target=w,args=(i,)) for i in range(32)]
    [t.start() for t in ths]
    time.sleep(random.uniform(0.5,2.0))
    p.kill(); stop=True; [t.join() for t in ths]; p.wait()
    snap=dict(acked)
    for i,v in snap.items(): hbase[i]=v[2]
    p=start(); s=socket.create_connection(("127.0.0.1",PORT)); f=s.makefile("rb")
    for i,(llen,ctr,n) in snap.items():
        s.sendall(enc("LLEN","L%d"%i)+enc("GET","C%d"%i)+enc("HLEN","H%d"%i))
        L=int(f.readline()[1:]); f.readline(); C=int(f.readline()); H=int(f.readline()[1:])
        if L<llen or C<ctr or H<n or L<C-0 and False: bad+=1; print("LOST",i,(llen,ctr,n),(L,C,H))
    print("round",rnd,"clients",len(snap),"total acked",sum(v[1] for v in snap.values()),"bad",bad,flush=True)
    p.kill(); p.wait()
print("BAD",bad)
