//! Event-loop networking: one epoll loop per core.
//!
//! The previous server spawned an OS thread per connection. Every idle
//! client cost a thread (stack + scheduler entry), 10k connections meant 10k
//! threads, and a read-heavy workload paid a context switch per request.
//!
//! Here, `DBSTRIKE_IO_THREADS` (default: one per core) worker threads each
//! run an epoll loop and own the connections they accept. All workers share
//! one non-blocking listener registered with `EPOLLEXCLUSIVE`, so the kernel
//! wakes exactly one worker per incoming connection — no acceptor thread, no
//! cross-thread hand-off.
//!
//! A batch of parsed commands is executed:
//!   * INLINE on the loop when nothing in it can block: reads, vector search,
//!     PING/INFO/CLIENT, MULTI queuing — and every write when the engine is
//!     non-durable (`DBSTRIKE_SYNC=0`), since then nothing waits on a disk.
//!   * on the BLOCKING POOL otherwise (a write waiting for its group-commit
//!     fsync, BLPOP, WAIT, AUTH's key stretching, CHECKPOINT, ...). The
//!     connection is paused — its later input stays buffered — until the
//!     result comes back through the worker's eventfd, so per-connection
//!     ordering is exactly that of the threaded server. The pool grows with
//!     the number of IN-FLIGHT blocking batches (which is also what lets
//!     group commit batch concurrent writers), never with connections.
//!     While it holds the connection, the pool thread writes the replies
//!     itself and keeps serving the client's next commands as long as they
//!     arrive promptly (socket events are muted meanwhile), so a
//!     request/response writer pays no loop round trip per command. It hands
//!     the connection back once the client goes quiet or the pool is busy.
//!
//! Long-lived connection modes (SUBSCRIBE, the replication stream) are
//! handed to a dedicated thread, as before.
//!
//! The per-command semantics are not here: both this loop and the threaded
//! fallback (`DBSTRIKE_NET=threads`) call the same `process_batch`.

use super::{err, parse_commands, process_batch, replication, BatchEnd, ConnState, Db, QUERY_BUFFER_LIMIT};
use protocol::write_resp_buf_as;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// ── epoll / eventfd FFI (Linux; std only, no external crates) ───────────────

#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

const POLLIN: i16 = 0x1;

#[cfg_attr(target_arch = "x86_64", repr(C, packed))]
#[cfg_attr(not(target_arch = "x86_64"), repr(C))]
#[derive(Clone, Copy)]
struct EpollEvent {
    events: u32,
    data: u64,
}

extern "C" {
    fn epoll_create1(flags: i32) -> i32;
    fn epoll_ctl(epfd: i32, op: i32, fd: i32, event: *mut EpollEvent) -> i32;
    fn epoll_wait(epfd: i32, events: *mut EpollEvent, maxevents: i32, timeout: i32) -> i32;
    fn eventfd(initval: u32, flags: i32) -> i32;
    fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
    fn read(fd: i32, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
}

const EPOLLIN: u32 = 0x001;
const EPOLLOUT: u32 = 0x004;
const EPOLLERR: u32 = 0x008;
const EPOLLHUP: u32 = 0x010;
const EPOLLRDHUP: u32 = 0x2000;
const EPOLLEXCLUSIVE: u32 = 1 << 28;
const EPOLLET: u32 = 1 << 31;
const EPOLL_CTL_ADD: i32 = 1;
const EPOLL_CTL_DEL: i32 = 2;
const EPOLL_CTL_MOD: i32 = 3;
/// Interest set of a client connection the loop is serving.
const CONN_EVENTS: u32 = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET;
const EPOLL_CLOEXEC: i32 = 0x80000;
const EFD_NONBLOCK: i32 = 0x800;
const EFD_CLOEXEC: i32 = 0x80000;

const TOKEN_LISTENER: u64 = u64::MAX;
const TOKEN_WAKE: u64 = u64::MAX - 1;

/// Stop parsing more input while this much output is waiting for a slow
/// reader (resumes once the socket drains).
const OUTPUT_HIGH_WATER: usize = 8 * 1024 * 1024;

fn ctl(epfd: RawFd, op: i32, fd: RawFd, events: u32, token: u64) -> std::io::Result<()> {
    let mut ev = EpollEvent { events, data: token };
    // SAFETY: valid epoll fd + fd; `ev` lives for the call.
    if unsafe { epoll_ctl(epfd, op, fd, &mut ev) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

// ── Blocking pool ───────────────────────────────────────────────────────────

type Job = Box<dyn FnOnce() + Send>;

/// Threads for batches that must wait (fsync, BLPOP, WAIT, ...). Grows on
/// demand up to `max`; idle threads retire after a while.
struct Pool {
    q: Mutex<PoolQ>,
    cv: Condvar,
    max: usize,
}

struct PoolQ {
    jobs: VecDeque<Job>,
    idle: usize,
    threads: usize,
}

impl Pool {
    fn new(max: usize) -> Arc<Self> {
        Arc::new(Pool { q: Mutex::new(PoolQ { jobs: VecDeque::new(), idle: 0, threads: 0 }), cv: Condvar::new(), max })
    }

    fn submit(self: &Arc<Self>, job: Job) {
        let mut q = self.q.lock().unwrap();
        q.jobs.push_back(job);
        if q.idle > q.jobs.len() - 1 {
            self.cv.notify_one();
        } else if q.threads < self.max {
            q.threads += 1;
            let me = Arc::clone(self);
            std::thread::Builder::new()
                .name("dbstrike-blocking".into())
                .spawn(move || me.work())
                .expect("spawn blocking thread");
        } // else: queued; the next free thread takes it
    }

    /// Jobs are waiting for a free thread.
    fn backlogged(&self) -> bool {
        !self.q.lock().unwrap().jobs.is_empty()
    }

    fn work(&self) {
        let mut q = self.q.lock().unwrap();
        loop {
            if let Some(job) = q.jobs.pop_front() {
                drop(q);
                job();
                q = self.q.lock().unwrap();
                continue;
            }
            q.idle += 1;
            let (g, t) = self.cv.wait_timeout(q, Duration::from_secs(30)).unwrap();
            q = g;
            q.idle -= 1;
            if t.timed_out() && q.jobs.is_empty() {
                q.threads -= 1;
                return;
            }
        }
    }
}

// ── Workers ─────────────────────────────────────────────────────────────────

/// Result of a batch run on the pool, delivered back to its worker.
struct Completion {
    slot: usize,
    gen: u32,
    st: ConnState,
    out: Vec<u8>,
    end: std::io::Result<BatchEnd>,
    /// Input the pool thread read but did not run (a partial command).
    rbuf: Vec<u8>,
    /// The client closed its side while the pool owned the connection.
    eof: bool,
}

/// What other threads need to reach a worker: pool completions, and
/// connections accepted by another worker and assigned to this one.
struct WorkerShared {
    wake_fd: RawFd,
    done: Mutex<Vec<Completion>>,
    inbox: Mutex<Vec<TcpStream>>,
}

impl WorkerShared {
    fn wake(&self) {
        let one: u64 = 1;
        // SAFETY: eventfd write of 8 bytes.
        unsafe {
            write(self.wake_fd, &one as *const u64 as *const c_void, 8);
        }
    }
}

struct Conn {
    /// Shared with a pool thread that writes replies directly; the fd stays
    /// open until the last holder drops it, so it can never be reused under
    /// a running job even if the loop closes the connection meanwhile.
    stream: Arc<TcpStream>,
    fd: RawFd,
    gen: u32,
    rbuf: Vec<u8>,
    wbuf: Vec<u8>,
    wpos: usize,
    /// `None` while a batch of this connection runs on the pool.
    st: Option<ConnState>,
    /// Close once `wbuf` is flushed (QUIT, protocol error).
    close_after_flush: bool,
    /// Edge-triggered: we stopped reading while paused with bytes possibly
    /// still in the socket, so read again when resuming.
    more_to_read: bool,
    /// Socket events are muted while a pool thread owns the connection
    /// (otherwise every byte the client sends wakes the loop for nothing).
    muted: bool,
}

impl Conn {
    fn pending_out(&self) -> usize {
        self.wbuf.len() - self.wpos
    }

    /// Accepting new commands: not paused on the pool, not closing, not
    /// backpressured by a slow reader.
    fn ready(&self) -> bool {
        self.st.is_some() && !self.close_after_flush && self.pending_out() <= OUTPUT_HIGH_WATER
    }
}

/// While a connection is paused, buffer at most this much further input
/// before leaving the rest in the socket.
const PAUSED_INPUT_LIMIT: usize = 16 * 1024 * 1024;

/// Round-robin cursor for placing accepted connections on workers.
static NEXT_WORKER: AtomicUsize = AtomicUsize::new(0);

/// Live client connections across all workers (INFO connected_clients).
pub static CONNECTED: AtomicUsize = AtomicUsize::new(0);

struct Worker {
    db: Arc<Db>,
    epfd: RawFd,
    listener: Arc<TcpListener>,
    shared: Arc<WorkerShared>,
    /// Every worker's shared half, for round-robin connection placement.
    all: Arc<Vec<Arc<WorkerShared>>>,
    index: usize,
    pool: Arc<Pool>,
    conns: Vec<Option<Conn>>,
    gens: Vec<u32>,
    free: Vec<usize>,
}

/// Serve `listener` with one epoll loop per worker. Never returns.
pub fn run(db: Arc<Db>, listener: TcpListener, workers: usize) -> std::io::Result<()> {
    listener.set_nonblocking(true)?;
    let listener = Arc::new(listener);
    let max_blocking: usize = std::env::var("DBSTRIKE_BLOCKING_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let pool = Pool::new(max_blocking.max(1));
    let n = workers.max(1);
    let mut epfds = Vec::new();
    let mut shareds = Vec::new();
    for _ in 0..n {
        // SAFETY: plain syscalls; results checked.
        let epfd = unsafe { epoll_create1(EPOLL_CLOEXEC) };
        let wake_fd = unsafe { eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC) };
        if epfd < 0 || wake_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        ctl(epfd, EPOLL_CTL_ADD, listener.as_raw_fd(), EPOLLIN | EPOLLEXCLUSIVE, TOKEN_LISTENER)?;
        ctl(epfd, EPOLL_CTL_ADD, wake_fd, EPOLLIN, TOKEN_WAKE)?;
        epfds.push(epfd);
        shareds.push(Arc::new(WorkerShared { wake_fd, done: Mutex::new(Vec::new()), inbox: Mutex::new(Vec::new()) }));
    }
    let all = Arc::new(shareds);
    let mut handles = Vec::new();
    for (i, epfd) in epfds.into_iter().enumerate() {
        let mut w = Worker {
            db: Arc::clone(&db),
            epfd,
            listener: Arc::clone(&listener),
            shared: Arc::clone(&all[i]),
            all: Arc::clone(&all),
            index: i,
            pool: Arc::clone(&pool),
            conns: Vec::new(),
            gens: Vec::new(),
            free: Vec::new(),
        };
        handles.push(std::thread::Builder::new().name(format!("dbstrike-io-{i}")).spawn(move || w.run())?);
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

impl Worker {
    fn run(&mut self) {
        let mut events = vec![EpollEvent { events: 0, data: 0 }; 1024];
        loop {
            // SAFETY: buffer of `events.len()` entries.
            let n = unsafe { epoll_wait(self.epfd, events.as_mut_ptr(), events.len() as i32, -1) };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == ErrorKind::Interrupted {
                    continue;
                }
                eprintln!("[NET] epoll_wait failed: {e}");
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            for ev in &events[..n as usize] {
                let (token, flags) = (ev.data, ev.events);
                match token {
                    TOKEN_LISTENER => self.accept_all(),
                    TOKEN_WAKE => self.completions(),
                    t => self.conn_event(t, flags),
                }
            }
        }
    }

    fn accept_all(&mut self) {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    // EPOLLEXCLUSIVE wakes ONE worker, which then drains the
                    // whole accept queue — so placement by "who accepted"
                    // piled a burst of clients onto a single core. Spread them
                    // round-robin across all workers instead.
                    let target = NEXT_WORKER.fetch_add(1, Ordering::Relaxed) % self.all.len();
                    if target == self.index {
                        if let Err(e) = self.register(stream) {
                            eprintln!("[NET] cannot register connection: {e}");
                        }
                    } else {
                        self.all[target].inbox.lock().unwrap().push(stream);
                        self.all[target].wake();
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    // EMFILE etc.: back off briefly instead of spinning.
                    eprintln!("[NET] accept error: {e} (raise ulimit -n for more clients)");
                    std::thread::sleep(Duration::from_millis(10));
                    return;
                }
            }
        }
    }

    fn register(&mut self, stream: TcpStream) -> std::io::Result<()> {
        stream.set_nonblocking(true)?;
        let _ = stream.set_nodelay(true);
        let fd = stream.as_raw_fd();
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                self.conns.push(None);
                self.gens.push(0);
                self.conns.len() - 1
            }
        };
        let gen = self.gens[slot];
        // Edge-triggered, registered ONCE for both directions: no epoll_ctl
        // per command (pausing a connection just stops consuming its input).
        ctl(self.epfd, EPOLL_CTL_ADD, fd, CONN_EVENTS, token(slot, gen))?;
        self.conns[slot] = Some(Conn {
            stream: Arc::new(stream),
            fd,
            gen,
            rbuf: Vec::new(),
            wbuf: Vec::new(),
            wpos: 0,
            st: Some(ConnState::new(&self.db)),
            close_after_flush: false,
            more_to_read: false,
            muted: false,
        });
        CONNECTED.fetch_add(1, Ordering::Relaxed);
        // Data may have arrived before registration (no edge for it).
        self.readable(slot);
        Ok(())
    }

    /// Remove a connection from the loop. Its socket closes once every
    /// holder (possibly a running pool job) has dropped it.
    fn release(&mut self, slot: usize) -> Option<Conn> {
        let c = self.conns[slot].take()?;
        let _ = ctl(self.epfd, EPOLL_CTL_DEL, c.fd, 0, 0);
        // Bump the generation: a pool completion for the old occupant of
        // this slot must not be applied to a new one.
        self.gens[slot] = self.gens[slot].wrapping_add(1);
        self.free.push(slot);
        CONNECTED.fetch_sub(1, Ordering::Relaxed);
        Some(c)
    }

    fn close(&mut self, slot: usize) {
        if let Some(c) = self.release(slot) {
            // Shut down now (a pool job may still hold the Arc).
            let _ = c.stream.shutdown(std::net::Shutdown::Both);
        }
    }

    fn live(&self, slot: usize) -> bool {
        matches!(self.conns.get(slot), Some(Some(_)))
    }

    fn conn_event(&mut self, t: u64, flags: u32) {
        let (slot, gen) = untoken(t);
        if !matches!(self.conns.get(slot), Some(Some(c)) if c.gen == gen) {
            return;
        }
        if flags & EPOLLOUT != 0 {
            self.flush(slot);
        }
        if flags & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR) != 0 && self.live(slot) {
            self.readable(slot);
        }
    }

    /// Drain the socket (edge-triggered: until WouldBlock), then run what
    /// can run.
    fn readable(&mut self, slot: usize) {
        let mut tmp = [0u8; 64 * 1024];
        let mut eof = false;
        {
            let Some(c) = self.conns[slot].as_mut() else { return };
            if c.st.is_none() {
                // A pool thread owns the input while it runs a batch (it
                // may keep reading follow-up commands itself); pick up
                // whatever is left once it hands the connection back.
                c.more_to_read = true;
                return;
            }
            c.more_to_read = false;
            loop {
                if !c.ready() && c.rbuf.len() > PAUSED_INPUT_LIMIT {
                    c.more_to_read = true; // resume reading when unpaused
                    break;
                }
                match (&*c.stream).read(&mut tmp) {
                    Ok(0) => {
                        eof = true;
                        break;
                    }
                    Ok(n) => {
                        c.rbuf.extend_from_slice(&tmp[..n]);
                        if c.rbuf.len() > QUERY_BUFFER_LIMIT {
                            break;
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(_) => {
                        eof = true;
                        break;
                    }
                }
            }
        }
        if eof {
            // Client gone. A batch still running on the pool finishes and its
            // completion is discarded (the slot generation moved on).
            self.close(slot);
            return;
        }
        let c = self.conns[slot].as_mut().unwrap();
        if c.rbuf.len() > QUERY_BUFFER_LIMIT {
            let resp3 = c.st.as_ref().is_some_and(|s| s.resp3);
            let _ = write_resp_buf_as(&mut c.wbuf, &err("Protocol error: query buffer limit exceeded"), resp3);
            c.close_after_flush = true;
            c.rbuf.clear();
            self.flush(slot);
            return;
        }
        if c.ready() {
            self.process(slot);
        }
    }

    /// Execute every complete command buffered for `slot` (inline, or by
    /// handing the batch to the pool), then flush.
    fn process(&mut self, slot: usize) {
        loop {
            let Some(c) = self.conns[slot].as_mut() else { return };
            if !c.ready() {
                break;
            }
            let cmds = match parse_commands(&mut c.rbuf) {
                Ok(v) => v,
                Err(e) => {
                    let resp3 = c.st.as_ref().is_some_and(|s| s.resp3);
                    let _ = write_resp_buf_as(&mut c.wbuf, &err(&format!("Protocol error: {e}")), resp3);
                    c.close_after_flush = true;
                    break;
                }
            };
            if cmds.is_empty() {
                break;
            }
            if batch_inline(&self.db, c.st.as_ref().unwrap(), &cmds) {
                let mut st = c.st.take().unwrap();
                let end = process_batch(&self.db, &mut st, cmds, &mut c.wbuf);
                c.st = Some(st);
                if !self.finish_batch(slot, end) {
                    return;
                }
            } else {
                let st = c.st.take().unwrap();
                let rbuf = std::mem::take(&mut c.rbuf);
                let (db, shared, gen, pool) = (Arc::clone(&self.db), Arc::clone(&self.shared), c.gen, Arc::clone(&self.pool));
                // Nothing queued ahead of this batch's replies: let the pool
                // thread write them straight to the socket (and keep serving
                // the connection while it stays busy) instead of bouncing
                // through the loop. The loop neither writes to nor reads
                // from a connection while its batch runs, so order holds.
                let direct = (c.pending_out() == 0).then(|| Arc::clone(&c.stream));
                if direct.is_some() {
                    // EPOLLET alone: hang-ups still report, input does not.
                    let _ = ctl(self.epfd, EPOLL_CTL_MOD, c.fd, EPOLLET, token(slot, c.gen));
                    c.muted = true;
                }
                self.pool.submit(Box::new(move || {
                    let comp = run_owned(&db, &pool, slot, gen, st, cmds, rbuf, direct);
                    let mut done = shared.done.lock().unwrap();
                    let first = done.is_empty();
                    done.push(comp);
                    drop(done);
                    if first {
                        shared.wake(); // the worker drains the whole list
                    }
                }));
                break;
            }
        }
        self.flush(slot);
    }

    /// Apply how a batch ended. `false` = the connection left this loop.
    fn finish_batch(&mut self, slot: usize, end: std::io::Result<BatchEnd>) -> bool {
        match end {
            Ok(BatchEnd::Continue) => true,
            Ok(BatchEnd::Quit) | Err(_) => {
                if let Some(c) = self.conns[slot].as_mut() {
                    c.close_after_flush = true;
                }
                self.flush(slot);
                false
            }
            Ok(BatchEnd::Subscribe(cmds)) => {
                self.hand_off(slot, Some(cmds));
                false
            }
            Ok(BatchEnd::ReplSync) => {
                self.hand_off(slot, None);
                false
            }
        }
    }

    /// Move a connection to a dedicated thread (subscribe mode, or the
    /// replication stream), after writing whatever replies are pending.
    fn hand_off(&mut self, slot: usize, subscribe: Option<Vec<Vec<Vec<u8>>>>) {
        let Some(mut c) = self.release(slot) else { return };
        CONNECTED.fetch_add(1, Ordering::Relaxed); // still a client, elsewhere
        let db = Arc::clone(&self.db);
        std::thread::spawn(move || {
            let stream = match Arc::try_unwrap(c.stream) {
                Ok(s) => s,
                Err(shared) => match shared.try_clone() {
                    Ok(s) => s,
                    Err(_) => {
                        CONNECTED.fetch_sub(1, Ordering::Relaxed);
                        return;
                    }
                },
            };
            let mut stream = stream;
            let _ = stream.set_nonblocking(false);
            if stream.write_all(&c.wbuf[c.wpos..]).is_err() {
                CONNECTED.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            let st = c.st.take().expect("state present at hand-off");
            let r = match subscribe {
                Some(cmds) => super::continue_subscribed(stream, db, st, cmds, std::mem::take(&mut c.rbuf)),
                None => replication::serve_replica(&db, stream, st.conn_id),
            };
            CONNECTED.fetch_sub(1, Ordering::Relaxed);
            if let Err(e) = r {
                if !super::is_benign_disconnect(&e) {
                    eprintln!("[NET] connection error: {e}");
                }
            }
        });
    }

    /// Wake-ups: connections assigned by other workers, and pool batches
    /// that finished (restore state, queue replies, continue).
    fn completions(&mut self) {
        let mut buf = [0u8; 8];
        // SAFETY: drain the eventfd counter (non-blocking).
        unsafe {
            read(self.shared.wake_fd, buf.as_mut_ptr() as *mut c_void, 8);
        }
        let arrived: Vec<TcpStream> = std::mem::take(&mut *self.shared.inbox.lock().unwrap());
        for stream in arrived {
            if let Err(e) = self.register(stream) {
                eprintln!("[NET] cannot register connection: {e}");
            }
        }
        let done: Vec<Completion> = std::mem::take(&mut *self.shared.done.lock().unwrap());
        for comp in done {
            let live = matches!(self.conns.get(comp.slot), Some(Some(c)) if c.gen == comp.gen);
            if !live {
                continue; // client disconnected meanwhile
            }
            {
                let c = self.conns[comp.slot].as_mut().unwrap();
                c.st = Some(comp.st);
                c.wbuf.extend_from_slice(&comp.out);
                debug_assert!(c.rbuf.is_empty());
                c.rbuf = comp.rbuf;
                if c.muted {
                    c.muted = false;
                    c.more_to_read = true; // input may have arrived unannounced
                    let _ = ctl(self.epfd, EPOLL_CTL_MOD, c.fd, CONN_EVENTS, token(comp.slot, c.gen));
                }
                if comp.eof {
                    c.close_after_flush = true;
                }
            }
            if self.finish_batch(comp.slot, comp.end) {
                self.resume(comp.slot);
            }
        }
    }

    /// A paused connection can take commands again: run what is buffered and
    /// pick up input the edge-triggered socket may still hold.
    fn resume(&mut self, slot: usize) {
        let more = self.conns[slot].as_ref().is_some_and(|c| c.more_to_read);
        if more {
            self.readable(slot); // reads, then processes
        } else {
            self.process(slot);
        }
    }

    /// Write pending output until the socket would block (edge-triggered:
    /// EPOLLOUT fires again once it drains).
    fn flush(&mut self, slot: usize) {
        let mut failed = false;
        let was_backpressured;
        {
            let Some(c) = self.conns[slot].as_mut() else { return };
            was_backpressured = c.pending_out() > OUTPUT_HIGH_WATER;
            while c.wpos < c.wbuf.len() {
                match (&*c.stream).write(&c.wbuf[c.wpos..]) {
                    Ok(0) => break,
                    Ok(n) => c.wpos += n,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
        }
        if failed {
            self.close(slot);
            return;
        }
        let c = self.conns[slot].as_mut().unwrap();
        if c.wpos == c.wbuf.len() {
            c.wbuf.clear();
            c.wpos = 0;
            if c.close_after_flush {
                self.close(slot);
                return;
            }
        } else if c.wpos > 1 << 20 {
            c.wbuf.drain(..c.wpos);
            c.wpos = 0;
        }
        // A slow reader caught up: resume its paused input.
        if was_backpressured && c.ready() {
            self.resume(slot);
        }
    }
}

/// A pool thread waits on an owned connection for the client's next
/// command in polls of 1, 2, 4, ... ms (about 63 ms in all) before handing
/// it back to the loop, checking between polls whether other batches need
/// the thread. Deep-pipeline clients (redis-benchmark -P1024) take over
/// 10 ms to send their next pipeline; handing those back each time cost a
/// fifth of the throughput. A parked thread costs no CPU.
const OWNED_WAITS: u32 = 6;
/// Reads that may go by without completing a command before hand-back.
const OWNED_READS: usize = 8;

/// Run a batch on a pool thread. With `direct`, write the replies straight
/// to the socket and, like a dedicated connection thread, keep reading and
/// running the client's next batches while they arrive promptly — a
/// request/response client issuing durable writes then pays no extra
/// thread hops per command. Hands the connection back (by returning) once
/// it goes quiet, the socket would block, the pool has a backlog, or the
/// batch ends anything but `Continue`.
#[allow(clippy::too_many_arguments)]
fn run_owned(
    db: &Arc<Db>,
    pool: &Pool,
    slot: usize,
    gen: u32,
    mut st: ConnState,
    mut cmds: Vec<Vec<Vec<u8>>>,
    mut rbuf: Vec<u8>,
    direct: Option<Arc<TcpStream>>,
) -> Completion {
    let mut out = Vec::new();
    let mut eof = false;
    loop {
        let end = process_batch(db, &mut st, cmds, &mut out);
        let Some(mut s) = direct.as_deref() else {
            return Completion { slot, gen, st, out, end, rbuf, eof };
        };
        let mut sent = 0;
        while sent < out.len() {
            match s.write(&out[sent..]) {
                Ok(0) => break,
                Ok(n) => sent += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(_) => break, // WouldBlock or gone: the loop takes the rest
            }
        }
        out.drain(..sent);
        if !out.is_empty() || !matches!(end, Ok(BatchEnd::Continue)) || rbuf.len() > QUERY_BUFFER_LIMIT {
            drop(direct); // hand-off may need sole ownership of the stream
            return Completion { slot, gen, st, out, end, rbuf, eof };
        }
        // Next batch: already buffered, or arriving within a short wait.
        let (mut waits, mut reads) = (0, 0);
        cmds = loop {
            match parse_commands(&mut rbuf) {
                Ok(c) if !c.is_empty() => break c,
                Ok(_) => {}
                Err(_) => break Vec::new(), // the loop reports it
            }
            if eof {
                break Vec::new();
            }
            // poll first: at one request per round trip the next command
            // is rarely there yet, and poll returns at once when it is.
            // Quiet client, busy pool, or a frame trickling in slowly / too
            // big to be worth it here: the loop serves it from now on.
            if waits == OWNED_WAITS || reads == OWNED_READS || pool.backlogged() {
                break Vec::new();
            }
            let mut p = PollFd { fd: s.as_raw_fd(), events: POLLIN, revents: 0 };
            // SAFETY: one valid pollfd.
            if unsafe { poll(&mut p, 1, 1 << waits) } == 0 {
                waits += 1;
                continue;
            }
            reads += 1;
            // Drain everything the client has sent before parsing: a deep
            // pipeline cut into socket-read-sized pieces became several
            // batches, each waiting out its own group-commit fsync.
            let mut tmp = [0u8; 64 * 1024];
            loop {
                match s.read(&mut tmp) {
                    Ok(0) => {
                        eof = true;
                        break;
                    }
                    Ok(n) => {
                        rbuf.extend_from_slice(&tmp[..n]);
                        if n < tmp.len() || rbuf.len() > QUERY_BUFFER_LIMIT {
                            break;
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(_) => {
                        eof = true;
                        break;
                    }
                }
            }
        };
        if cmds.is_empty() {
            drop(direct);
            return Completion { slot, gen, st, out, end: Ok(BatchEnd::Continue), rbuf, eof };
        }
    }
}

fn token(slot: usize, gen: u32) -> u64 {
    ((gen as u64) << 32) | slot as u64
}

fn untoken(t: u64) -> (usize, u32) {
    ((t & 0xffff_ffff) as usize, (t >> 32) as u32)
}

/// Can this whole batch run on the event loop without blocking it?
fn batch_inline(db: &Db, st: &ConnState, cmds: &[Vec<Vec<u8>>]) -> bool {
    let durable = db.engine.is_durable();
    let mut in_multi = st.multi.is_some();
    for c in cmds {
        let name = String::from_utf8_lossy(&c[0]).to_ascii_uppercase();
        if in_multi {
            match name.as_str() {
                "EXEC" => return false, // commits (and fsyncs) the queue
                "DISCARD" => in_multi = false,
                _ => {} // queued, not executed
            }
            continue;
        }
        match name.as_str() {
            "MULTI" => in_multi = true,
            // Always potentially slow or blocking, durable or not.
            "AUTH" | "BLPOP" | "BRPOP" | "WAIT" | "CHECKPOINT" | "FLUSHALL" | "FLUSHDB" | "VBULKLOAD"
            | "VBULKLOADNS" | "VSNAPSHOT" | "GPU.LOAD" | "GPU.MODE" | "GPU.UNLOAD" | "GPU.SWEEP" | "VFITQUANT"
            | "VFITQUANTNS" | "VCALIBRATE" | "VADDBATCH" | "VADDBATCHNS" | "MEM.CONSOLIDATE"
            // promotion rebuilds every derived index
            | "REPLICAOF" | "SLAVEOF" => return false,
            "HELLO" if c.iter().any(|a| a.eq_ignore_ascii_case(b"AUTH")) => return false,
            "ACL" if !c.get(1).is_some_and(|s| {
                let s = String::from_utf8_lossy(s).to_ascii_uppercase();
                matches!(s.as_str(), "WHOAMI" | "LIST" | "GETUSER")
            }) => return false,
            _ if !durable => {} // nothing waits on a disk
            n if is_read_only(n) => {}
            _ => return false,
        }
    }
    true
}

/// Commands that never write (so never wait for an fsync).
fn is_read_only(name: &str) -> bool {
    if super::keyspace::is_keyspace_cmd(name) {
        return !super::keyspace::is_keyspace_write(name);
    }
    matches!(
        name,
        "PING" | "ECHO" | "TIME" | "QUIT" | "HELLO" | "CLIENT" | "SELECT" | "DBSIZE" | "INFO" | "COMMAND" | "CONFIG"
            | "ROLE" | "PUBSUB" | "KEYS" | "SCAN" | "SCANRANGE" | "GETAT" | "VSEARCH" | "VSEARCHNS" | "VSEARCHA"
            | "VSEARCHANS" | "VSEARCH.MANY" | "VSEARCH.MANYNS" | "VGETPAYLOAD" | "VLISTNS" | "VQUANT" | "VQUANTNS"
            | "VFACET" | "VRECOMMEND" | "TSRANGE" | "TSRANGE.LATEST" | "TSLATEST" | "TSAVG" | "TABLE.GET"
            | "TABLE.SCAN" | "TABLE.FILTEREQ" | "CRDT.GET" | "HLC.NOW" | "MEM.RECALL" | "MEM.RECALL.AS_OF"
            | "MEM.GET" | "MEM.COUNT" | "MEM.NEIGH" | "MEM.TRAV" | "MEM.PROC.GET" | "MEM.PROC.LIST" | "MEM.EPISODES"
            | "MEM.INCOMING" | "CDCLEN" | "GPU.INFO" | "MULTI" | "DISCARD" | "WATCH" | "UNWATCH" | "SUBSCRIBE"
            | "PSUBSCRIBE" | "REPLSYNC" | "MEMTRACK"
    )
}
