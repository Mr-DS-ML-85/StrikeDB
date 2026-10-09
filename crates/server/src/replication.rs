//! Asynchronous primary → replica replication.
//!
//! Wire protocol (after the replica sends `REPLSYNC` as a normal RESP
//! command and the primary replies `+FULLSYNC`), every message is a frame
//! `[kind u8][len u32 LE][payload]`:
//!
//!   1 SNAP   one key of the initial snapshot (`Mutation` bytes)
//!   2 READY  end of snapshot (payload: snapshot ts, u64 LE)
//!   3 MUT    a committed mutation newer than the snapshot
//!   4 PING   heartbeat (payload: primary visible ts, u64 LE)
//!   5 FLUSH  FLUSHALL happened at this point in the stream
//!
//! The replica answers with text lines `ACK <primary-ts>\n` after applying
//! data and on every heartbeat; `WAIT n timeout` on the primary counts the
//! replicas whose ack has reached the primary's current commit point.
//!
//! The commit hook runs on the engine's commit thread, so it only clones
//! the mutation into each replica's bounded queue; a replica that falls
//! too far behind is disconnected (and does a full resync on reconnect)
//! rather than making the primary buffer without limit.
//!
//! Replicas are read-only, apply the stream durably through the engine,
//! and keep their derived in-memory state (TTL index, collections flag,
//! CRDT map, vector graphs, agent-memory mirrors) in step with it.

use crate::Db;
use protocol::Resp;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use storage::{Mutation, Value};

const K_SNAP: u8 = 1;
const K_READY: u8 = 2;
const K_MUT: u8 = 3;
const K_PING: u8 = 4;
const K_FLUSH: u8 = 5;

/// Mutations a replica may lag behind before it is disconnected.
const REPLICA_QUEUE: usize = 1_000_000;
const HEARTBEAT: Duration = Duration::from_millis(100);

enum Frame {
    Mut(Mutation),
    Flush,
}

struct ReplicaLink {
    tx: SyncSender<Frame>,
    sock: TcpStream,
    addr: String,
    acked: Arc<AtomicU64>,
}

/// Primary-side registry of connected replicas.
#[derive(Default)]
pub struct Hub {
    active: AtomicBool,
    replicas: RwLock<HashMap<u64, ReplicaLink>>,
}

impl Hub {
    /// Commit hook: forward one mutation to every replica.
    pub fn on_commit(&self, m: &Mutation) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        self.broadcast(|| Frame::Mut(m.clone()));
    }

    pub fn on_flush(&self) {
        if self.active.load(Ordering::Relaxed) {
            self.broadcast(|| Frame::Flush);
        }
    }

    fn broadcast(&self, mk: impl Fn() -> Frame) {
        let mut evict = Vec::new();
        for (id, r) in self.replicas.read().unwrap().iter() {
            match r.tx.try_send(mk()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => evict.push(*id),
            }
        }
        for id in evict {
            if let Some(r) = self.replicas.write().unwrap().remove(&id) {
                eprintln!("[REPL] replica {} fell too far behind; disconnecting (it will resync)", r.addr);
                let _ = r.sock.shutdown(std::net::Shutdown::Both);
            }
        }
        self.active.store(!self.replicas.read().unwrap().is_empty(), Ordering::Relaxed);
    }

    fn register(&self, id: u64, link: ReplicaLink) {
        self.replicas.write().unwrap().insert(id, link);
        self.active.store(true, Ordering::Relaxed);
    }

    fn unregister(&self, id: u64) {
        let mut g = self.replicas.write().unwrap();
        g.remove(&id);
        self.active.store(!g.is_empty(), Ordering::Relaxed);
    }

    /// Replicas whose ack has reached `ts`.
    pub fn acked_at_least(&self, ts: u64) -> usize {
        self.replicas.read().unwrap().values().filter(|r| r.acked.load(Ordering::Relaxed) >= ts).count()
    }

    pub fn describe(&self) -> Vec<(String, u64)> {
        self.replicas.read().unwrap().values().map(|r| (r.addr.clone(), r.acked.load(Ordering::Relaxed))).collect()
    }
}

fn write_frame(w: &mut impl Write, kind: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut hdr = [0u8; 5];
    hdr[0] = kind;
    hdr[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    w.write_all(&hdr)?;
    w.write_all(payload)
}

fn read_frame(r: &mut impl Read) -> std::io::Result<(u8, Vec<u8>)> {
    let mut hdr = [0u8; 5];
    r.read_exact(&mut hdr)?;
    let len = u32::from_le_bytes(hdr[1..].try_into().unwrap()) as usize;
    if len > 1 << 30 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "replication frame too large"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok((hdr[0], buf))
}

/// Primary: serve one replica on this (hijacked) connection until it drops.
pub fn serve_replica(db: &Arc<Db>, stream: TcpStream, id: u64) -> std::io::Result<()> {
    let addr = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let (tx, rx) = sync_channel::<Frame>(REPLICA_QUEUE);
    let acked = Arc::new(AtomicU64::new(0));
    // Register BEFORE taking the dump snapshot: every commit after the
    // snapshot is then guaranteed to be in the queue (commits at or before
    // it are filtered out below, since the dump already contains them).
    db.repl_hub.register(id, ReplicaLink { tx, sock: stream.try_clone()?, addr: addr.clone(), acked: Arc::clone(&acked) });
    let result = (|| -> std::io::Result<()> {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, stream.try_clone()?);
        w.write_all(b"+FULLSYNC\r\n")?;
        let mut snap_ts = 0;
        let mut io_err: Option<std::io::Error> = None;
        db.engine.dump(
            |s| snap_ts = s,
            |m| {
                if io_err.is_none() {
                    if let Err(e) = write_frame(&mut w, K_SNAP, &m.to_bytes()) {
                        io_err = Some(e);
                    }
                }
            },
        );
        if let Some(e) = io_err {
            return Err(e);
        }
        write_frame(&mut w, K_READY, &snap_ts.to_le_bytes())?;
        w.flush()?;
        eprintln!("[REPL] replica {addr} synced (snapshot ts {snap_ts}); streaming");
        // Acks come back on the same socket.
        {
            let acked = Arc::clone(&acked);
            let mut r = BufReader::new(stream.try_clone()?);
            std::thread::spawn(move || {
                let mut line = String::new();
                while r.read_line(&mut line).map(|n| n > 0).unwrap_or(false) {
                    if let Some(ts) = line.trim().strip_prefix("ACK ").and_then(|t| t.parse::<u64>().ok()) {
                        acked.fetch_max(ts, Ordering::Relaxed);
                    }
                    line.clear();
                }
            });
        }
        stream_live(db, &rx, &mut w, snap_ts)
    })();
    db.repl_hub.unregister(id);
    let _ = stream.shutdown(std::net::Shutdown::Both);
    eprintln!("[REPL] replica {addr} disconnected");
    result
}

fn stream_live(db: &Db, rx: &Receiver<Frame>, w: &mut impl Write, snap_ts: u64) -> std::io::Result<()> {
    let mut last_ping = Instant::now();
    loop {
        match rx.recv_timeout(HEARTBEAT) {
            Ok(Frame::Mut(m)) => {
                if m.ts > snap_ts {
                    write_frame(w, K_MUT, &m.to_bytes())?;
                }
                // Drain whatever else is queued before flushing the socket.
                while let Ok(f) = rx.try_recv() {
                    match f {
                        Frame::Mut(m) if m.ts > snap_ts => write_frame(w, K_MUT, &m.to_bytes())?,
                        Frame::Mut(_) => {}
                        Frame::Flush => write_frame(w, K_FLUSH, &[])?,
                    }
                }
                w.flush()?;
            }
            Ok(Frame::Flush) => {
                write_frame(w, K_FLUSH, &[])?;
                w.flush()?;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
        if last_ping.elapsed() >= HEARTBEAT {
            // Everything committed up to `visible` has been queued (the hook
            // runs before visibility is published), so after the replica has
            // applied every frame before this ping it is caught up to here.
            write_frame(w, K_PING, &db.engine.snapshot().to_le_bytes())?;
            w.flush()?;
            last_ping = Instant::now();
        }
    }
}

// ── Replica side ───────────────────────────────────────────────────────────

#[derive(Default)]
pub struct ReplicaState {
    /// (host, port) of the primary while this node is a replica.
    master: Mutex<Option<(String, u16)>>,
    /// Bumped on every REPLICAOF so a stale sync thread notices and exits.
    epoch: AtomicU64,
    link_up: AtomicBool,
    /// Primary commit ts this replica has fully applied.
    applied: AtomicU64,
    last_io_ms: AtomicU64,
}

impl ReplicaState {
    pub fn is_replica(&self) -> bool {
        self.master.lock().unwrap().is_some()
    }

    pub fn master(&self) -> Option<(String, u16)> {
        self.master.lock().unwrap().clone()
    }

    pub fn link_up(&self) -> bool {
        self.link_up.load(Ordering::Relaxed)
    }

    pub fn applied(&self) -> u64 {
        self.applied.load(Ordering::Relaxed)
    }

    pub fn last_io_secs(&self) -> u64 {
        let t = self.last_io_ms.load(Ordering::Relaxed);
        if t == 0 {
            return u64::MAX;
        }
        crate::keyspace::now_ms().saturating_sub(t) / 1000
    }
}

/// `REPLICAOF host port` / `REPLICAOF NO ONE`.
pub fn replicaof(db: &Arc<Db>, args: &[Vec<u8>]) -> Resp {
    if args.len() != 2 {
        return crate::err("wrong number of arguments for 'replicaof' command");
    }
    let a0 = String::from_utf8_lossy(&args[0]).to_string();
    let a1 = String::from_utf8_lossy(&args[1]).to_string();
    let epoch = db.replica.epoch.fetch_add(1, Ordering::SeqCst) + 1;
    if a0.eq_ignore_ascii_case("NO") && a1.eq_ignore_ascii_case("ONE") {
        let was = db.replica.master.lock().unwrap().take();
        db.replica.link_up.store(false, Ordering::Relaxed);
        if was.is_some() {
            // Promotion: make sure every derived index reflects the data.
            rebuild_derived(db);
            eprintln!("[REPL] promoted to primary");
        }
        return Resp::Simple("OK".into());
    }
    let Ok(port) = a1.parse::<u16>() else { return crate::err("Invalid master port") };
    *db.replica.master.lock().unwrap() = Some((a0.clone(), port));
    db.replica.link_up.store(false, Ordering::Relaxed);
    let db2 = Arc::clone(db);
    std::thread::spawn(move || replica_loop(db2, a0, port, epoch));
    Resp::Simple("OK".into())
}

fn current(db: &Db, epoch: u64) -> bool {
    db.replica.epoch.load(Ordering::SeqCst) == epoch
}

fn replica_loop(db: Arc<Db>, host: String, port: u16, epoch: u64) {
    while current(&db, epoch) {
        match sync_once(&db, &host, port, epoch) {
            Ok(()) => {}
            Err(e) => {
                if current(&db, epoch) {
                    eprintln!("[REPL] link to {host}:{port} down: {e}; retrying in 1s");
                }
            }
        }
        db.replica.link_up.store(false, Ordering::Relaxed);
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn resp_cmd(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for a in args {
        out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
        out.extend_from_slice(a);
        out.extend_from_slice(b"\r\n");
    }
    out
}

fn sync_once(db: &Arc<Db>, host: &str, port: u16, epoch: u64) -> std::io::Result<()> {
    let sock = TcpStream::connect((host, port))?;
    sock.set_nodelay(true)?;
    sock.set_read_timeout(Some(Duration::from_secs(5)))?; // heartbeats every 100 ms
    let mut out = sock.try_clone()?;
    let mut r = BufReader::with_capacity(1 << 20, sock);
    let mut line = String::new();
    if let Ok(pw) = std::env::var("DBSTRIKE_MASTERAUTH") {
        let user = std::env::var("DBSTRIKE_MASTERUSER").unwrap_or_else(|_| "default".into());
        out.write_all(&resp_cmd(&[b"AUTH", user.as_bytes(), pw.as_bytes()]))?;
        r.read_line(&mut line)?;
        if !line.starts_with('+') {
            return Err(std::io::Error::other(format!("primary AUTH failed: {}", line.trim())));
        }
        line.clear();
    }
    out.write_all(&resp_cmd(&[b"REPLSYNC"]))?;
    r.read_line(&mut line)?;
    if line.trim() != "+FULLSYNC" {
        return Err(std::io::Error::other(format!("primary refused REPLSYNC: {}", line.trim())));
    }
    // Full resync: start from an empty keyspace.
    wipe_local(db)?;
    let mut batch: Vec<(Vec<u8>, Value)> = Vec::new();
    let mut max_ts = 0u64;
    loop {
        if !current(db, epoch) {
            return Ok(());
        }
        let (kind, payload) = read_frame(&mut r)?;
        db.replica.last_io_ms.store(crate::keyspace::now_ms(), Ordering::Relaxed);
        match kind {
            K_SNAP => {
                let m = Mutation::from_bytes(&payload).ok_or_else(|| std::io::Error::other("bad snapshot frame"))?;
                max_ts = max_ts.max(m.ts);
                batch.push((m.key, m.value));
                if batch.len() >= 4096 {
                    db.engine.put_batch(std::mem::take(&mut batch))?;
                }
            }
            K_READY => {
                if !batch.is_empty() {
                    db.engine.put_batch(std::mem::take(&mut batch))?;
                }
                let snap = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0; 8]));
                rebuild_derived(db);
                db.replica.applied.store(snap.max(max_ts), Ordering::Relaxed);
                db.replica.link_up.store(true, Ordering::Relaxed);
                writeln!(out, "ACK {}", db.replica.applied())?;
                eprintln!("[REPL] full sync from {host}:{port} complete");
            }
            K_MUT => {
                // Apply this and every frame already buffered as one durable
                // batch (one fsync), preserving order.
                let mut muts = vec![payload];
                while muts.len() < 4096 && r.buffer().len() >= 5 {
                    let (k, p) = read_frame(&mut r)?;
                    if k != K_MUT {
                        // Put it back in order: handle after applying this batch.
                        apply_muts(db, &muts)?;
                        muts.clear();
                        handle_control(db, &mut out, k, p)?;
                        break;
                    }
                    muts.push(p);
                }
                if !muts.is_empty() {
                    apply_muts(db, &muts)?;
                }
                writeln!(out, "ACK {}", db.replica.applied())?;
            }
            k => handle_control(db, &mut out, k, payload)?,
        }
    }
}

fn handle_control(db: &Arc<Db>, out: &mut TcpStream, kind: u8, payload: Vec<u8>) -> std::io::Result<()> {
    match kind {
        K_PING => {
            // Every frame before this ping has been applied.
            let ts = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0; 8]));
            db.replica.applied.fetch_max(ts, Ordering::Relaxed);
            writeln!(out, "ACK {}", db.replica.applied())
        }
        K_FLUSH => {
            wipe_local(db)?;
            rebuild_derived(db);
            Ok(())
        }
        K_MUT => apply_muts(db, &[payload]),
        _ => Err(std::io::Error::other(format!("unexpected replication frame kind {kind}"))),
    }
}

fn apply_muts(db: &Arc<Db>, frames: &[Vec<u8>]) -> std::io::Result<()> {
    let mut batch = Vec::with_capacity(frames.len());
    let mut max_ts = 0;
    for p in frames {
        let m = Mutation::from_bytes(p).ok_or_else(|| std::io::Error::other("bad mutation frame"))?;
        max_ts = max_ts.max(m.ts);
        batch.push((m.key, m.value));
    }
    db.engine.put_batch(batch.clone())?;
    for (k, v) in &batch {
        observe(db, k, v);
    }
    db.replica.applied.fetch_max(max_ts, Ordering::Relaxed);
    Ok(())
}

/// Keep derived in-memory state in step with one replicated write.
fn observe(db: &Arc<Db>, key: &[u8], value: &Value) {
    db.ks.observe(key, value);
    if key.starts_with(b"crdt:") {
        *db.crdt.lock().unwrap() = crate::load_crdts(&db.engine);
    } else if let Some(rest) = key.strip_prefix(b"vec:") {
        // `vec:<id8>` = default index; `vec:<ns>:<id8>` = namespace.
        if rest.len() < 8 {
            return;
        }
        let id = u64::from_be_bytes(rest[rest.len() - 8..].try_into().unwrap());
        let vi = if rest.len() == 8 {
            Some(db.router.vectors())
        } else {
            let ns = String::from_utf8_lossy(&rest[..rest.len() - 9]).to_string();
            if ns == "__ltm__" {
                db.mem_dirty.store(true, Ordering::Relaxed);
                None
            } else {
                Some(db.router.vectors_ns(&ns))
            }
        };
        if let Some(vi) = vi {
            match value {
                Value::Vector(v) => vi.insert_graph_only(id, v.clone()),
                Value::Tombstone => vi.forget_graph_only(id),
                _ => {}
            }
        }
    } else if key.starts_with(b"mem:") {
        db.mem_dirty.store(true, Ordering::Relaxed);
    }
}

fn wipe_local(db: &Arc<Db>) -> std::io::Result<()> {
    let bak = db.router.flush_all_with_backup()?;
    // A replica's previous world is reproducible from the primary: don't
    // accumulate a backup per resync.
    let _ = std::fs::remove_file(&bak);
    let _ = std::fs::remove_file(format!("{bak}.snap"));
    db.rag.memory().reset_volatile();
    db.rag.invalidate_query_cache();
    *db.crdt.lock().unwrap() = crate::load_crdts(&db.engine);
    db.ks.reload();
    Ok(())
}

/// Rebuild every derived in-memory index from the (replicated) substrate.
pub fn rebuild_derived(db: &Arc<Db>) {
    db.router.reload();
    db.rag.memory().reload();
    db.rag.invalidate_query_cache();
    *db.crdt.lock().unwrap() = crate::load_crdts(&db.engine);
    db.ks.reload();
    db.mem_dirty.store(false, Ordering::Relaxed);
}

/// Commands a replica refuses (it only changes through the stream).
pub fn is_write_command(name: &str) -> bool {
    use crate::acl::{command_categories, PermCategory};
    if matches!(name, "FLUSHALL" | "FLUSHDB" | "VSNAPSHOT" | "BLPOP" | "BRPOP") {
        return true;
    }
    command_categories(name).contains(&PermCategory::Write)
}
