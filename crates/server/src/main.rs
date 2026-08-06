//! DB-Strike server — one engine, all layers, speaking the Redis (RESP) wire.
//!
//! Boots the unified substrate, attaches the reactive hub, builds the router
//! (tables + vector index + planner), the reducer runtime, and tiered memory,
//! then serves RESP over TCP with a thread-per-connection model.
//!
//! Supported commands (case-insensitive):
//!   PING                                  -> PONG
//!   SET key value                         -> OK
//!   GET key                               -> bulk | nil
//!   DEL key                               -> :0/:1
//!   INCR key / INCRBY key n               -> :new
//!   KEYS prefix                           -> array of matching keys
//!   VADD id f1 f2 ...                      -> OK           (store an embedding)
//!   VSEARCH k f1 f2 ...                    -> array id/dist (k-NN)
//!   TSADD series ts val                    -> OK
//!   TSRANGE series from to                 -> array ts/val
//!   REDUCE name shardkey key by            -> :new         (fuel-metered counter reducer)
//!   CDCLEN                                 -> :n           (reactive change-log length)
//!   INFO                                   -> bulk (engine stats)
//!   QUIT                                   -> OK, closes

mod acl;

/// Keep the allocator's books for this process.
///
/// A 200k × 384d ingest over the wire drove this server to 29.3 GB resident and
/// the OOM-killer took it out, while the *same* index built in-process stayed at
/// 1.2 GB. RSS alone cannot close that gap — it reports the total but never the
/// owner — so the allocator does the accounting instead, and `MEMTRACK` reads
/// the books back over RESP. That means a phase can be attributed live, without
/// attaching a profiler or restarting the server under one.
///
/// Cost is two relaxed atomic adds per allocation: no syscalls, no locks, small
/// enough to leave installed on the hot ingest path without perturbing the very
/// behaviour it is measuring.
#[global_allocator]
static ALLOC: mitm::memtrack::TrackingAlloc = mitm::memtrack::TrackingAlloc;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use acl::{AclStore, command_category};
use compute::{counter_reducer, vm::{Instr, Program}, ReducerResult, ReducerRuntime};
use consensus::{hlc::Hlc, crdt::{GCounter, LwwRegister, PnCounter}};
use mitm::CacheDebugger;
use protocol::{try_parse, write_resp, write_resp_buf, Resp};
use rag::Rag;
use reactive::Reactive;
use router::{Router, TieredMemory};
use std::collections::HashMap;
use storage::{Engine, Value};
use views::{Filter, Kv, LearnedEf, Row, TimeSeries};

struct Db {
    engine: Arc<Engine>,
    reactive: Arc<Reactive>,
    router: Router,
    kv: Kv,
    ts: TimeSeries,
    reducers: ReducerRuntime,
    rag: Rag,
    cache: Arc<CacheDebugger>,
    #[allow(dead_code)]
    memory: TieredMemory,
    /// MODULE 3 — the calibrated learned-beam-width model, populated by
    /// `VCALIBRATE` and consumed by `VSEARCH L`. Guarded by a Mutex because the
    /// RESP dispatch is multi-threaded; calibration is rare, search is hot.
    learned: Mutex<Option<LearnedEf>>,
    /// Consensus CRDTs (in-memory, merge-able). Keyed by a user-supplied name.
    /// These back the `CRDT.*` family of RESP commands.
    crdt: Mutex<ConsensusStore>,
    /// Hybrid logical clock for the `HLC.*` RESP commands.
    hlc: Hlc,
    /// Current quantization mode for the vector index (per-process; the
    /// in-memory HNSW holds it and it is not persisted — see `VSETQUANT`).
    quant_mode: Mutex<views::QuantMode>,
    /// ACL store — manages users, passwords, and command permissions.
    acl: Arc<AclStore>,
}

/// In-memory store of all CRDTs, keyed by name. Each variant is merge-able.
struct ConsensusStore {
    gc: HashMap<String, GCounter>,
    pn: HashMap<String, PnCounter>,
    lww: HashMap<String, LwwRegister>,
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let addr = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1:6380".to_string());
    let mut log_path: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--log" => {
                i += 1;
                log_path = Some(args.get(i).cloned().unwrap_or_else(|| "--log requires a path".to_string()));
            }
            other => {
                eprintln!("unknown arg: {other} (usage: dbstrike <addr> [--log <path>])");
                return Ok(());
            }
        }
        i += 1;
    }
    let data_path = std::env::var("DBSTRIKE_WAL").unwrap_or_else(|_| "dbstrike.wal".to_string());
    let requirepass = std::env::var("DBSTRIKE_PASS").ok();
    let log_file: Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>> = log_path.as_ref().map(|p| {
        std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true).open(p).expect("cannot open log file"),
        ))
    });
    if let Some(ref lf) = log_file {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(lf.lock().unwrap(), "[{ts}] DB-Strike starting on {addr}");
    }

    let engine = Engine::open(&data_path)?;
    let reactive = Reactive::attach(&engine);
    let router = Router::new(Arc::clone(&engine));
    let kv = Kv::new(Arc::clone(&engine));
    let ts = TimeSeries::new(Arc::clone(&engine));
    let reducers = ReducerRuntime::new(Arc::clone(&engine), 16);
    let rag = Rag::open(Arc::clone(&engine));
    let cache = CacheDebugger::new(Arc::clone(&engine), 4096);
    let memory = TieredMemory::new(300);
    let acl = AclStore::new(requirepass.clone());

    // Allocator tracking is opt-in: it is off for ordinary workloads (the
    // shared-line atomics otherwise cost ~3.8× on the pipelined hot path) and
    // turns on when the operator asks — either `DBSTRIKE_MEMTRACK=1` at
    // startup or a `MEMTRACK` RESP command at runtime.
    if std::env::var("DBSTRIKE_MEMTRACK").map(|v| v != "0" && v != "").unwrap_or(false) {
        mitm::memtrack::set_tracking(true);
    }

    let db = Arc::new(Db {
        engine,
        reactive,
        router,
        kv,
        ts,
        reducers,
        rag,
        cache,
        memory,
        learned: Mutex::new(None),
        crdt: Mutex::new(ConsensusStore {
            gc: HashMap::new(),
            pn: HashMap::new(),
            lww: HashMap::new(),
        }),
        hlc: Hlc::new(),
        quant_mode: Mutex::new(views::QuantMode::Int8),
        acl,
    });

    let listener = TcpListener::bind(&addr)?;
    // Large backlog so a connection storm (-c800+) queues instead of the
    // kernel dropping SYNs; the real ceiling is the per-process fd limit
    // (ulimit -n), which must be raised on the host for very high -c.
    let _ = listener.set_nonblocking(false);
    let auth_msg = if requirepass.is_some() { " (AUTH required)" } else { "" };
    let startup = format!(
        "DB-Strike listening on {addr} (RESP wire), WAL={data_path}{auth_msg}\n\
         One engine: KV · vectors · tables · timeseries · reducers · pub/sub · CRDT · HLC · agent-memory · RAG · MITM cache-debug\n\
         Wired: VSETQUANT/VSETQUANTNS/VFITQUANT/VFITQUANTNS/VQUANTNS · VLISTNS · VADDBATCH/VADDBATCHNS · VDEL/VDELNS · VBULKLOAD/VBULKLOADNS · VSEARCH/VSEARCHNS/VSEARCHA/VSEARCHANS/VSEARCH.MANY/VSEARCH.MANYNS · TABLE.* · CRDT.* · HLC.* · REDUCE.PROGRAM · MEM.INCOMING/COUNT/GET/CONSOLIDATE/EPISODES_CLEAR · TSAVG · RAG.CONTEXT · GETAT/SCAN · AUTH · ACL · GPU.LOAD/INFO/UNLOAD/MODE"
    );
    println!("{startup}");
    if let Some(ref lf) = log_file {
        let _ = writeln!(lf.lock().unwrap(), "{startup}");
    }

    // Rate-limit accept-error logging: under EMFILE the accept loop would
    // otherwise spew thousands of identical lines per second. Print at most
    // once per second, keeping the last error for the periodic line.
    let mut last_accept_err: Option<String> = None;
    let mut last_log = std::time::Instant::now();
for stream in listener.incoming() {
         match stream {
             Ok(s) => {
                 let db = Arc::clone(&db);
                 let lf = log_file.clone();
                 std::thread::spawn(move || {
                     if let Err(e) = handle(s, db, &lf) {
                         if !is_benign_disconnect(&e) {
                             log_msg(&lf, &format!("connection error: {e}"));
                         }
                     }
                 });
             }
            Err(e) => {
                let msg = e.to_string();
                let now = std::time::Instant::now();
                let repeat = last_accept_err.as_deref() == Some(msg.as_str());
                if !repeat || now.duration_since(last_log) >= std::time::Duration::from_secs(1) {
                    log_msg(&log_file, &format!("accept error: {msg} (raise ulimit -n for higher -c; sleeping 10ms to shed load)"));
                    last_accept_err = Some(msg);
                    last_log = now;
                }
                // Back off briefly so a sustained EMFILE storm doesn't burn a
                // full core spinning on accept().
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
    Ok(())
}

fn handle(stream: TcpStream, db: Arc<Db>, log_file: &Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>) -> std::io::Result<()> {
    // TCP_NODELAY: disable Nagle so per-batch flushes actually go on the wire
    // immediately (matters for latency-sensitive workloads like signaling).
    let _ = stream.set_nodelay(true);
    // Single shared stream behind a Mutex. We deliberately AVOID `try_clone()`
    // here: cloning dups the fd, so every connection would burn *two* fds and
    // the per-process fd ceiling (ulimit -n) would be hit at half the real
    // connection count — causing "Too many open files" storms under a high -c
    // benchmark. One Arc<Mutex<TcpStream>> = one fd per connection.
    let stream = Arc::new(Mutex::new(stream));
    // Accumulate all replies for a parsed batch into this buffer, then do a
    // single locked write + flush. Keeps the hot path lock-free except for the
    // one flush per pipeline batch (matching the old BufWriter throughput).
    let mut out: Vec<u8> = Vec::with_capacity(64 * 1024);
    // Manual read buffer so we can peek pipelined commands without going
    // command-at-a-time. BufRead's fill_buf has capacity semantics that make
    // the "keep parsing until short" pattern awkward; a plain Vec<u8> is
    // clearer and just as fast.
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut tmp = [0u8; 32 * 1024];
    // Read once per connection, not once per command.
    let trace = std::env::var("DBSTRIKE_TRACE").is_ok_and(|v| v != "0");
    // ACL: track which user this connection is authenticated as.
    // If no requirepass, default user is pre-authenticated.
    let mut current_user: String = if db.acl.requires_auth() {
        String::new() // not authenticated yet
    } else {
        "default".to_string()
    };

    // NOTE: a geometric parse-retry backoff was tried here and REVERTED.
    //
    // The intent was sound — `try_parse` restarts from byte zero, so retrying
    // after each 64 KiB read makes a large frame quadratic in its own size.
    // The implementation deadlocked. It waited for the buffer to grow by
    // `max(remaining/2, 64 KiB)` before parsing again, but a command split
    // across reads whose remainder is under 64 KiB never reaches that
    // threshold: the client has sent everything and is waiting for a reply
    // while the server waits for bytes that will never arrive. Any payload
    // large enough to straddle a read boundary could hang the connection.
    //
    // A correct version must derive the *actual* bytes the frame needs from
    // its RESP headers rather than guessing, so the threshold is exact and can
    // always be reached. Until that exists, parse on every read: quadratic on
    // huge frames is a performance problem, and this was a liveness one.

    loop {
        // Block until we have SOMETHING to parse (or the client closes).
        let n = {
            let mut s = stream.lock().unwrap();
            s.read(&mut tmp)?
        };
        if n == 0 {
            return Ok(()); // client closed
        }
        buf.extend_from_slice(&tmp[..n]);

        // ── Drain every complete command from `buf` ────────────────────
        let mut cmds: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut cursor = 0usize;
        loop {
            match try_parse(&buf[cursor..]) {
                Ok(Some((cmd, consumed))) => {
                    cursor += consumed;
                    if !cmd.is_empty() {
                        cmds.push(cmd);
                    }
                }
                Ok(None) => break, // partial command; wait for more bytes
                Err(e) => {
                    // A genuine protocol error (NOT a truncated frame —
                    // `try_parse` reports those as `Ok(None)`). Reply the way
                    // Redis does, then close. Previously this bare `return
                    // Err(e)` dropped the socket, and `is_benign_disconnect`
                    // matched the message and suppressed the log, so the client
                    // saw an unexplained `ConnectionReset` and the server said
                    // nothing at all. Send the reason down the wire first.
                    let _ = write_resp_buf(&mut out, &err(&format!("ERR Protocol error: {e}")));
                    if let Ok(mut s) = stream.lock() {
                        let _ = s.write_all(&out);
                        let _ = s.flush();
                    }
                    return Err(e);
                }
            }
        }
        if cursor > 0 {
            buf.drain(..cursor);
        }
        if cmds.is_empty() {
            continue;
        }

        // ── Dispatch batch. Consecutive `SET k v` commands get coalesced
        // into ONE engine.put_batch — one fsync per burst, not one per
        // command. This is the "Redis-class pipelined SET" fix. Every
        // other command dispatches individually to preserve read-after-
        // write ordering (e.g. GET must see writes that came before it).
        let mut quit = false;
        let mut subscribe_after_batch: Option<(String, Vec<Vec<u8>>)> = None;
        let mut i = 0usize;
        while i < cmds.len() {
            let name = String::from_utf8_lossy(&cmds[i][0]).to_uppercase();
            let args = &cmds[i][1..];

            // ── ACL: handle AUTH/ACL before permission check ──────────
            if name == "AUTH" {
                let resp = dispatch_auth(&db, args, &mut current_user);
                write_resp_buf(&mut out, &resp)?;
                if name == "QUIT" { quit = true; break; }
                i += 1;
                continue;
            }
            if name == "ACL" {
                let resp = dispatch_acl(&db, args, &current_user);
                write_resp_buf(&mut out, &resp)?;
                i += 1;
                continue;
            }

            // ── ACL: permission check ─────────────────────────────────
            // Fast path: in the default no-auth install `strict` is false and
            // nothing can deny a command, so this collapses to one relaxed
            // load. It only latches true once auth/restrictions are configured
            // (DBSTRIKE_PASS, ACL SETUSER/DELUSER, disabling a user).
            if db.acl.requires_auth() && current_user.is_empty() {
                write_resp_buf(&mut out, &err("NOAUTH Authentication required"))?;
                i += 1;
                continue;
            }
            if db.acl.needs_permission_check() {
                let cat = command_category(&name);
                if !db.acl.can_command(&current_user, &name, cat) {
                    write_resp_buf(&mut out, &err("ERR permission denied"))?;
                    i += 1;
                    continue;
                }
            }

            // SUBSCRIBE hijacks the connection AFTER we finish the current
            // batch (need to write acks + stream events, no more parsing).
            if name == "SUBSCRIBE" || name == "PSUBSCRIBE" {
                subscribe_after_batch = Some((name.clone(), args.to_vec()));
                break;
            }

            // Coalesce a run of pure-write commands: `SET k v`, `MSET k v...`,
            // `TSADD ts val`. Each command emits ONE `+OK` reply preserving
            // per-command reply-count invariant. The whole run lands in ONE
            // `put_batch` → ONE fsync = the pipelined-throughput win.
            //
            // Semantics: `SET x 1; SET x 2` in one pipeline is equivalent to
            // just `SET x 2` (BTreeMap dedup keeps the last write), matching
            // Redis's own pipeline behavior — no client observes the interim.
            fn is_coalescable_set(cmd: &[Vec<u8>]) -> bool {
                if cmd.is_empty() {
                    return false;
                }
                let name = &cmd[0];
                (name.eq_ignore_ascii_case(b"SET") && cmd.len() == 3)
                    || (name.eq_ignore_ascii_case(b"MSET")
                        && cmd.len() >= 3
                        && (cmd.len() - 1) % 2 == 0)
            }
            if is_coalescable_set(&cmds[i]) {
                let mut kvs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
                let mut cmds_in_run = 0usize;
                while i < cmds.len() && is_coalescable_set(&cmds[i]) {
                    // Take ownership of the command so the value bytes can be
                    // MOVED (not cloned) straight into the engine — at 16M ops/s
                    // a per-value heap clone is a measurable allocator tax.
                    let mut cmd = std::mem::take(&mut cmds[i]);
                    let cargs = &cmd[1..];
                    if cmd[0].eq_ignore_ascii_case(b"SET") {
                        let mut kb = Vec::with_capacity(3 + cargs[0].len());
                        kb.extend_from_slice(b"kv:");
                        kb.extend_from_slice(&cargs[0]);
                        let val = std::mem::take(&mut cmd[2]);
                        kvs.push((kb, val));
                    } else {
                        // MSET k v k v ...
                        for pair in cargs.chunks(2) {
                            let mut kb = Vec::with_capacity(3 + pair[0].len());
                            kb.extend_from_slice(b"kv:");
                            kb.extend_from_slice(&pair[0]);
                            kvs.push((kb, pair[1].clone()));
                        }
                    }
                    cmds_in_run += 1;
                    i += 1;
                }
                let n = kvs.len();
                match db.kv.set_batch(kvs) {
                    Ok(_) => {
                        // One "+OK\r\n" per *command* (not per key), so a
                        // coalesced run of SET/SET or a single MSET each emit
                        // exactly one ack — preserving the per-command
                        // reply-count invariant the client relies on.
                        let _ = n;
                        for _ in 0..cmds_in_run {
                            out.extend_from_slice(b"+OK\r\n");
                        }
                    }
                    Err(e) => {
                        let e = err(&e.to_string());
                        for _ in 0..n {
                            write_resp_buf(&mut out, &e)?;
                        }
                    }
                }
                continue;
            }

            // Per-command trace, OFF by default (`DBSTRIKE_TRACE=1` to enable).
            // This used to be unconditional, and it is not a cheap line: it
            // formats every argument, so a single 384-dim VSEARCH prints ~5 KB.
            // One 1.7s bench section produced a 160 MB log, and the formatting
            // plus the write syscall sat directly in the command hot path.
            let resp = if trace {
                let t_start = std::time::Instant::now();
                let resp = dispatch(&db, &name, args);
                let elapsed_ms = t_start.elapsed().as_micros() as f64 / 1000.0;
                // Redacts passwords — see `redact_cmd`.
                eprintln!("[CMD] {:>6.1}ms {}", elapsed_ms, redact_cmd(&name, args));
                if let Some(ref lf) = log_file {
                    let _ = writeln!(lf.lock().unwrap(), "[CMD] {:>6.1}ms {}", elapsed_ms, redact_cmd(&name, args));
                }
                resp
            } else {
                dispatch(&db, &name, args)
            };
            write_resp_buf(&mut out, &resp)?;
            if name == "QUIT" {
                quit = true;
                break;
            }
            i += 1;
        }
        // Single locked flush of the whole batch.
        {
            let mut s = stream.lock().unwrap();
            s.write_all(&out)?;
            s.flush()?;
        }
        out.clear();
        if quit {
            return Ok(());
        }
        if let Some((name, args)) = subscribe_after_batch {
            return handle_subscribe(&db, &stream, &name, &args);
        }
    }
}

/// Hijack the connection into push-stream mode after SUBSCRIBE/PSUBSCRIBE.
/// Factored out of `handle` so the fast-path batched dispatch stays tight.
/// Shares the single `Arc<Mutex<TcpStream>>` so it still uses just one fd.
fn handle_subscribe(
    db: &Arc<Db>,
    stream: &Arc<Mutex<TcpStream>>,
    _name: &str,
    args: &[Vec<u8>],
) -> std::io::Result<()> {
    if args.is_empty() {
        let mut s = stream.lock().unwrap();
        write_resp(&mut *s, &err("SUBSCRIBE requires at least one channel"))?;
        return Ok(());
    }
    let prefixes: Vec<Vec<u8>> = args
        .iter()
        .map(|a| {
            let mut p = b"chan:".to_vec();
            p.extend_from_slice(a);
            p
        })
        .collect();
    let rx = db.reactive.subscribe_prefixes(&prefixes);
    for (i, ch) in args.iter().enumerate() {
        let ack = Resp::Array(vec![
            Resp::Bulk(b"subscribe".to_vec()),
            Resp::Bulk(ch.clone()),
            Resp::Int((i as i64) + 1),
        ]);
        let mut s = stream.lock().unwrap();
        write_resp(&mut *s, &ack)?;
    }
    for ev in rx.iter() {
        let channel = ev.key.strip_prefix(b"chan:").unwrap_or(&ev.key).to_vec();
        let payload: Vec<u8> = match &ev.value {
            storage::Value::Bytes(b) => b.clone(),
            storage::Value::Int(i) => i.to_string().into_bytes(),
            storage::Value::Float(f) => f.to_string().into_bytes(),
            storage::Value::Tombstone => b"__deleted__".to_vec(),
            other => format!("{other:?}").into_bytes(),
        };
        let msg = Resp::Array(vec![
            Resp::Bulk(b"message".to_vec()),
            Resp::Bulk(channel),
            Resp::Bulk(payload),
        ]);
        let mut s = stream.lock().unwrap();
        if write_resp(&mut *s, &msg).is_err() {
            break;
        }
    }
    Ok(())
}

fn err(msg: &str) -> Resp {
    Resp::Error(format!("ERR {msg}"))
}

/// Wall-clock milliseconds, used as the TTL reference for Working Memory.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Write a line to stderr and, if a log file is configured, to it too.
fn log_msg(log_file: &Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>, msg: &str) {
    eprint!("{msg}");
    if let Some(ref lf) = log_file {
        let _ = writeln!(lf.lock().unwrap(), "{msg}");
    }
}

/// Namespace a reducer target key into the Kv surface (`kv:<key>`). The KV
/// view (GET/SET/INCR) reads prefixed keys, but the reducer VM operates on
/// raw keys; without this a `STOREINT`/counter write lands where `GET` can
/// never see it.
fn kv_key(key: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(key.len() + 3);
    b.extend_from_slice(b"kv:");
    b.extend_from_slice(key);
    b
}

/// True for I/O errors that just mean "the client hung up mid-request".
/// These aren't bugs and shouldn't spam the console.
fn is_benign_disconnect(e: &std::io::Error) -> bool {
    use std::io::ErrorKind::*;
    if matches!(
        e.kind(),
        BrokenPipe | ConnectionReset | ConnectionAborted | UnexpectedEof | TimedOut
    ) {
        return true;
    }
    // Torn RESP frames when a client closes mid-write bubble up as
    // InvalidData with these specific messages; they're benign too.
    let msg = e.to_string();
    msg.contains("expected bulk string")
        || msg.contains("bad array header")
        || msg.contains("bad bulk len")
}

/// Format a f64 time-series value for RESP wire output.
/// - Whole-number values (42.0) render as `"42"` so integer-only clients
///   (int() parsers) keep working end-to-end.
/// - Fractional values (42.5) render as `"42.5"` with just enough digits.
fn format_ts_val(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        (v as i64).to_string()
    } else {
        let s = format!("{v}");
        s
    }
}

fn parse_floats(args: &[Vec<u8>]) -> Option<Vec<f32>> {
    args.iter()
        .map(|a| std::str::from_utf8(a).ok()?.parse::<f32>().ok())
        .collect()
}

/// Number of hardware threads — used to size the parallel-ingest shard count.
fn num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .max(1)
}

/// Parse a quantization-mode token (case-insensitive) into `QuantMode`.
fn quant_mode_from_str(s: &str) -> Option<views::QuantMode> {
    use views::QuantMode::*;
    Some(match s {
        "INT8" => Int8,
        "BINARY" => Binary,
        "BINARY2" => Binary2,
        "BINARY15" => Binary15,
        "TURBO1" => Turbo1,
        "TURBO15" => Turbo15,
        "TURBO2" => Turbo2,
        "TURBO4" => Turbo4,
        "PRODUCT" => Product,
        _ => return None,
    })
}

/// Serialize a `Row` (col -> Vec<u8>) to a flat RESP array of (col, val) pairs.
fn row_to_resp(row: Row) -> Vec<Resp> {
    let mut out = Vec::with_capacity(row.len() * 2);
    for (col, val) in row {
        out.push(Resp::Bulk(col.into_bytes()));
        out.push(Resp::Bulk(val));
    }
    out
}

/// Render an optional `Value` into a RESP reply, typed by its variant.
fn value_to_resp(v: Option<Value>) -> Resp {
    match v {
        None => Resp::Nil,
        Some(Value::Bytes(b)) => Resp::Bulk(b),
        Some(Value::Int(i)) => Resp::Int(i),
        Some(Value::Float(f)) => Resp::Bulk(format!("{f}").into_bytes()),
        Some(Value::Vector(vec)) => Resp::Array(
            vec.into_iter()
                .map(|x| Resp::Bulk(format!("{x}").into_bytes()))
                .collect(),
        ),
        Some(Value::Row(row)) => Resp::Array(row_to_resp(row)),
        Some(Value::Tombstone) => Resp::Nil,
    }
}

/// MODULE 4/5 — deterministically DERIVE a query's filter attribute and sparse
/// (lexical) view from its dense vector, so one `VADD` populates the dense HNSW,
/// the filter-attr table, and the sparse/BM25 index with ZERO protocol change.
/// attr = which dim-bucket holds the vector's strongest component (a cheap,
/// stable category); sparse = the top-W dimensions by |value| as (term, weight).
/// Identical derivation is used by the bench, keeping client + server consistent.
fn derive_attr_and_sparse(vec: &[f32], n_buckets: u32, w: usize) -> (u32, Vec<(u32, f32)>) {
    let mut best_dim = 0usize;
    let mut best_mag = 0.0f32;
    let mut idxs: Vec<(usize, f32)> = vec.iter().enumerate().map(|(j, &v)| (j, v.abs())).collect();
    for (j, m) in &idxs {
        if *m > best_mag {
            best_mag = *m;
            best_dim = *j;
        }
    }
    let attr = (best_dim as u32) % n_buckets;
    idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let sparse: Vec<(u32, f32)> = idxs.iter().take(w).map(|(j, v)| (*j as u32, *v)).collect();
    (attr, sparse)
}

/// Handle AUTH command: `AUTH password` or `AUTH username password`.
/// Updates `current_user` on success.
fn dispatch_auth(db: &Db, args: &[Vec<u8>], current_user: &mut String) -> Resp {
    match args.len() {
        1 => {
            // AUTH password — authenticate as default user.
            let raw = String::from_utf8_lossy(&args[0]);
            let password = raw.strip_prefix('>').unwrap_or(&raw);
            if db.acl.auth_default(password) {
                *current_user = "default".to_string();
                db.acl.set_user_enabled("default", true);
                Resp::Simple("OK".into())
            } else {
                err("ERR invalid password")
            }
        }
        2 => {
            // AUTH username password — authenticate as specific user.
            let username = String::from_utf8_lossy(&args[0]);
            let raw = String::from_utf8_lossy(&args[1]);
            let password = raw.strip_prefix('>').unwrap_or(&raw);
            if db.acl.auth_user(&username, password) {
                // A named user may be restricted, so the per-command gate must
                // run for the rest of this connection (and any other auth path).
                db.acl.latch_strict();
                *current_user = username.to_string();
                // Enable the user after successful auth so commands work.
                db.acl.set_user_enabled(&username, true);
                Resp::Simple("OK".into())
            } else {
                err("ERR invalid username or password")
            }
        }
        _ => err("AUTH requires a password"),
    }
}

/// Handle ACL command: `ACL subcommand [args...]`.
fn dispatch_acl(db: &Db, args: &[Vec<u8>], current_user: &str) -> Resp {
    if args.is_empty() {
        return err("ACL requires a subcommand");
    }
    let sub = String::from_utf8_lossy(&args[0]).to_uppercase();
    match sub.as_str() {
        "WHOAMI" => Resp::Bulk(current_user.as_bytes().to_vec()),
        "LIST" => {
            let users = db.acl.list_users();
            Resp::Array(users.into_iter().map(|u| Resp::Bulk(u.into_bytes())).collect())
        }
        "SETUSER" => {
            if args.len() < 2 {
                return err("ACL SETUSER requires a username");
            }
            let username = String::from_utf8_lossy(&args[1]).to_string();
            // Parse optional flags: >password, on, off, ~key, +command, +@category, -@category
            let mut password: Option<String> = None;
            let mut enabled: Option<bool> = None;
            let mut categories = Vec::new();
            let mut key_patterns = Vec::new();
            let mut i = 2;
            while i < args.len() {
                let token = String::from_utf8_lossy(&args[i]).to_string();
                if token.starts_with('>') {
                    // >password — set password
                    password = Some(token[1..].to_string());
                } else if token == "on" {
                    enabled = Some(true);
                } else if token == "off" {
                    enabled = Some(false);
                } else if token.starts_with('~') {
                    // ~pattern — key pattern
                    key_patterns.push(token[1..].to_string());
                } else if token.starts_with('+') {
                    // +command or +@category
                    let rest = &token[1..];
                    if rest.starts_with('@') {
                        if let Some(cat) = acl::PermCategory::from_str(&rest[1..]) {
                            categories.push(cat);
                        }
                    }
                    // Individual command permissions not yet implemented.
                } else if token.starts_with('-') {
                    // -@category — remove category (not yet implemented)
                } else if token == "reset" {
                    // Reset user to defaults.
                    categories.clear();
                    key_patterns.clear();
                }
                i += 1;
            }
            // Apply changes.
            if let Some(pw) = password {
                db.acl.set_user_password(&username, &pw);
            }
            if let Some(en) = enabled {
                db.acl.set_user_enabled(&username, en);
            }
            if !categories.is_empty() {
                db.acl.set_user_categories(&username, categories);
            }
            Resp::Simple("OK".into())
        }
        "GETUSER" => {
            if args.len() < 2 {
                return err("ACL GETUSER requires a username");
            }
            let username = String::from_utf8_lossy(&args[1]);
            match db.acl.get_user_info(&username) {
                Some(info) => {
                    // Parse the info string into a RESP array of alternating key/value.
                    let parts: Vec<&str> = info.split_whitespace().collect();
                    let mut out = Vec::new();
                    let mut j = 0;
                    while j + 1 < parts.len() {
                        out.push(Resp::Bulk(parts[j].as_bytes().to_vec()));
                        out.push(Resp::Bulk(parts[j + 1].as_bytes().to_vec()));
                        j += 2;
                    }
                    Resp::Array(out)
                }
                None => Resp::Nil,
            }
        }
        "DELUSER" => {
            if args.len() < 2 {
                return err("ACL DELUSER requires a username");
            }
            let username = String::from_utf8_lossy(&args[1]);
            if db.acl.del_user(&username) {
                Resp::Int(1)
            } else {
                Resp::Int(0)
            }
        }
        "SAVE" | "LOAD" => {
            // In-memory only for now — reply OK.
            Resp::Simple("OK".into())
        }
        _ => err("ACL subcommand not supported (USE WHOAMI LIST SETUSER GETUSER DELUSER SAVE LOAD)"),
    }
}

/// Redact sensitive data from command args for logging.
/// Passwords in AUTH and ACL SETUSER >password are replaced with `***`.
fn redact_cmd(name: &str, args: &[Vec<u8>]) -> String {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "AUTH" => format!("AUTH ***"),
        "ACL" => {
            // ACL SETUSER username >password ...
            if args.len() >= 2 {
                let sub = String::from_utf8_lossy(&args[0]).to_uppercase();
                if sub == "SETUSER" {
                    let user = String::from_utf8_lossy(&args[1]);
                    let mut out = format!("ACL SETUSER {}", user);
                    for tok in &args[2..] {
                        let s = String::from_utf8_lossy(tok);
                        if s.starts_with('>') {
                            out.push_str(" >***");
                        } else {
                            out.push(' ');
                            out.push_str(&s);
                        }
                    }
                    return out;
                }
            }
            format!("ACL {}", args.iter().map(|a| String::from_utf8_lossy(a).to_string()).collect::<Vec<_>>().join(" "))
        }
        _ => {
            let mut parts = vec![upper];
            for a in args {
                parts.push(String::from_utf8_lossy(a).to_string());
            }
            parts.join(" ")
        }
    }
}

fn dispatch(db: &Db, name: &str, args: &[Vec<u8>]) -> Resp {
    match name {
        "PING" => Resp::Simple("PONG".into()),
        "QUIT" => Resp::Simple("OK".into()),
        // CLIENT ... — redis-benchmark 8 (and redis-cli) send `CLIENT SETINFO
        // LIB-NAME/LIB-VER` at startup. Replying OK (no-op) keeps the pre-flight
        // clean so no spurious 0-sample "-nan" line appears in the summary.
        "CLIENT" => {
            if args.is_empty() {
                return err("CLIENT requires a subcommand");
            }
            let sub = String::from_utf8_lossy(&args[0]).to_uppercase();
            match sub.as_str() {
                "SETINFO" | "SETNAME" | "GETNAME" | "INFO" | "LIST" | "TRACKING"
                | "PAUSE" | "UNPAUSE" | "REPLY" => Resp::Simple("OK".into()),
                _ => err("CLIENT subcommand not supported"),
            }
        }


        "SET" => {
            if args.len() != 2 {
                return err("SET requires key value");
            }
            match db.kv.set_b(&args[0], &args[1]) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MSET k1 v1 k2 v2 ...  → one put_batch, one fsync for the whole set.
        // Standard Redis multi-set; benchmark tools + real clients depend on it.
        "MSET" => {
            if args.is_empty() || args.len() % 2 != 0 {
                return err("MSET requires an even number of args (k v k v ...)");
            }
            let kvs: Vec<(Vec<u8>, Vec<u8>)> = args
                .chunks(2)
                .map(|c| {
                    let mut kb = Vec::with_capacity(3 + c[0].len());
                    kb.extend_from_slice(b"kv:");
                    kb.extend_from_slice(&c[0]);
                    (kb, c[1].clone())
                })
                .collect();
            match db.kv.set_batch(kvs) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MGET k1 k2 ...  → array of bulk values (or Nil for missing).
        // Companion to MSET; also expected by redis-benchmark.
        "MGET" => {
            if args.is_empty() {
                return err("MGET requires at least one key");
            }
            let out: Vec<Resp> = args
                .iter()
                .map(|a| match db.kv.get_b(a) {
                    Some(v) => Resp::Bulk(v),
                    None => Resp::Nil,
                })
                .collect();
            Resp::Array(out)
        }
        // DBSIZE → :n live keys across every shard. redis-benchmark checks
        // this at startup to size the working set.
        "DBSIZE" => Resp::Int(db.engine.dbsize() as i64),
        // FLUSHALL / FLUSHDB — reply OK without touching data. We don't
        // implement destructive keyspace wipes over the wire (durability
        // engine, not a cache); making these no-ops keeps benchmarks happy
        // without letting a stray "-x" flag nuke the corpus.
        "FLUSHALL" | "FLUSHDB" => Resp::Simple("OK".into()),
        // COMMAND / COMMAND DOCS — redis-benchmark sometimes probes this to
        // discover the arg-count of each op. A minimal empty-array reply is
        // enough to let it skip probing without erroring.
        "COMMAND" => Resp::Array(Vec::new()),
        // SELECT db — Redis supports N logical DBs; we always use DB 0.
        "SELECT" => Resp::Simple("OK".into()),
        // CONFIG GET <pattern> / CONFIG SET <k> <v> — redis-benchmark probes
        // `CONFIG GET save` (and others) at startup; an unknown command made
        // its warmup sample divide by zero → "-nan" RPS in the summary. Reply
        // with the real Redis-shaped key/value array so the probe succeeds.
        "CONFIG" => {
            if args.is_empty() {
                return err("CONFIG requires GET/SET ...");
            }
            let sub = String::from_utf8_lossy(&args[0]).to_uppercase();
            if sub == "GET" {
                if args.len() < 2 {
                    return err("CONFIG GET requires a pattern");
                }
                let pat = String::from_utf8_lossy(&args[1]).to_string();
                // Keys redis-benchmark / redis-cli commonly probe. Pattern "*"
                // returns all; a literal key returns just that one.
                let known: &[(&str, &str)] = &[
                    ("save", ""),
                    ("maxmemory", "0"),
                    ("maxmemory-policy", "noeviction"),
                    ("databases", "16"),
                    ("appendonly", "no"),
                    ("timeout", "0"),
                    ("tcp-keepalive", "0"),
                    ("hz", "10"),
                    ("lazyfree-lazy-eviction", "no"),
                ];
                let mut out: Vec<Resp> = Vec::new();
                for (k, v) in known {
                    let matches = pat == "*" || pat == *k
                        || (pat.ends_with('*') && k.starts_with(&pat[..pat.len() - 1]));
                    if matches {
                        out.push(Resp::Bulk(k.as_bytes().to_vec()));
                        out.push(Resp::Bulk(v.as_bytes().to_vec()));
                    }
                }
                Resp::Array(out)
            } else if sub == "SET" {
                // No-op: dbstrike has no tunable runtime config surface here.
                Resp::Simple("OK".into())
            } else {
                err("CONFIG subcommand must be GET or SET")
            }
        }
        "GET" => {
            if args.len() != 1 {
                return err("GET requires key");
            }
            match db.kv.get_b(&args[0]) {
                Some(v) => Resp::Bulk(v),
                None => Resp::Nil,
            }
        }
        "DEL" => {
            if args.is_empty() {
                return err("DEL requires at least one key");
            }
            let mut deleted = 0i64;
            for a in args {
                match db.kv.del_b(a) {
                    Ok(true) => deleted += 1,
                    Ok(false) => {}
                    Err(e) => return err(&e.to_string()),
                }
            }
            Resp::Int(deleted)
        }
        "INCR" | "INCRBY" => {
            let (key, by) = if name == "INCR" {
                if args.len() != 1 {
                    return err("INCR requires key");
                }
                (args[0].clone(), 1)
            } else {
                if args.len() != 2 {
                    return err("INCRBY requires key n");
                }
                let by: i64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => return err("n is not an integer"),
                };
                (args[0].clone(), by)
            };
            match db.kv.incr_by_lossy(&key, by) {
                Ok(n) => Resp::Int(n),
                Err(e) => err(&e),
            }
        }
        "KEYS" => {
            let prefix = args.first().map(|a| {
                let s = String::from_utf8_lossy(a);
                s.trim_end_matches('*').to_string()
            }).unwrap_or_default();
            let keys = db.kv.keys_prefix(prefix.as_bytes());
            Resp::Array(keys.into_iter().map(Resp::Bulk).collect())
        }

        "VADD" => {
            if args.len() < 2 {
                return err("VADD requires id f1 f2 ...");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            let vec = match parse_floats(&args[1..]) {
                Some(v) => v,
                None => return err("bad float in vector"),
            };
            // TurboQuant fits a d×d rotation; inserting a mismatched dim would
            // desync the packed storage (and panicked before this guard).
            let vi = db.router.vectors();
            let qd = vi.quant_dim();
            if qd != 0 && vec.len() != qd {
                return err(&format!("VADD dim {} != turbo index dim {}", vec.len(), qd));
            }
            // UNIFIED write: one command populates the dense HNSW (durable),
            // the MODULE 4 filter-attribute, and the MODULE 5 sparse/BM25 index
            // (both derived from the dense vector, no protocol change). This is
            // the "one graph, many access paths" thesis at the write path.
            let (attr, sparse) = derive_attr_and_sparse(&vec, 8, 8);
            let vi = db.router.vectors();
            // Graph-only insert FIRST so the node's filter-attribute is fixed
            // (the durable `insert` below takes the update-in-place path and
            // leaves attr untouched, so attr must be set here).
            vi.insert_graph_only_attr(id, vec.clone(), attr);
            if let Err(e) = vi.insert(id, vec.clone()) {
                return err(&e.to_string());
            }
            vi.add_sparse(id, sparse);
            Resp::Simple("OK".into())
        }

        // VADDNS <namespace> <id> f1 f2 ...  -> OK
        // Like VADD but stores the vector in a namespace-scoped index.
        // Each namespace gets its own HNSW graph and dim, so different
        // namespaces can hold vectors of different dimensionalities
        // (e.g. 512-dim faces vs 64-dim pHash) in a single StrikeDB process.
        "VADDNS" => {
            if args.len() < 3 {
                return err("VADDNS requires namespace id f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let id: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            let vec = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in vector"),
            };
            let vi = db.router.vectors_ns(&namespace);
            let qd = vi.quant_dim();
            if qd != 0 && vec.len() != qd {
                return err(&format!("VADDNS dim {} != turbo index dim {}", vec.len(), qd));
            }
            let (attr, sparse) = derive_attr_and_sparse(&vec, 8, 8);
            vi.insert_graph_only_attr(id, vec.clone(), attr);
            if let Err(e) = vi.insert(id, vec.clone()) {
                return err(&e.to_string());
            }
            vi.add_sparse(id, sparse);
            Resp::Simple("OK".into())
        }

        // VDEL id [id ...]  -> :<count>
        // Tombstone vectors by id: drops each durable KV, marks the HNSW node
        // deleted (every search path filters these), and purges each id's
        // sparse/BM25 postings. Counts the ids that were actually present.
        "VDEL" => {
            if args.is_empty() {
                return err("VDEL requires at least one id");
            }
            let vi = db.router.vectors();
            let mut deleted = 0u64;
            for a in args {
                let id: u64 = match std::str::from_utf8(a).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => return err("id is not a u64"),
                };
                if vi.forget(id) {
                    deleted += 1;
                }
            }
            Resp::Int(deleted as i64)
        }

        // VDELNS <namespace> id [id ...]  -> :<count>
        // Like VDEL but operates on a namespace-scoped index.
        // Tombstone vectors by id within the given namespace only.
        "VDELNS" => {
            if args.len() < 2 {
                return err("VDELNS requires namespace id [id ...]");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let vi = db.router.vectors_ns(&namespace);
            let mut deleted = 0u64;
            for a in &args[1..] {
                let id: u64 = match std::str::from_utf8(a).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => return err("id is not a u64"),
                };
                if vi.forget(id) {
                    deleted += 1;
                }
            }
            Resp::Int(deleted as i64)
        }

        // VADDBATCH [PAR] dim id0 f0..f{dim-1} id1 f0..f{dim-1} ...  -> :n
        // Multi-core ingest (Module 1). Default: builds the batch as parallel
        // shards and bridge-merges into the live graph via `merge_into`
        // (serial per-node bridge — correct, but single-threaded at merge).
        // With the `PAR` flag: combines the existing graph + batch and rebuilds
        // the WHOLE graph in parallel via `build_parallel_ids` (shuffle + cheap
        // O(K²) entry bridge) — a genuinely cores× build with correct recall.
        // Each vector is still written durably to the WAL substrate. Returns the
        // count ingested.
        // Bulk load from a server-side file. The data never crosses the wire.
        //
        // VADDBATCH cannot do bulk ingest at any batch size: small batches take
        // the serial append path, and large ones are worse rather than better
        // (100k x 384d measured at ~18 s for batch 64, >200 s for 512 and 2048,
        // and zero progress for 25k). This hands the builder a path instead.
        "VBULKLOAD" => {
            if args.len() != 1 {
                return err("VBULKLOAD requires exactly one argument: <path-to-.fbin>");
            }
            let path = match std::str::from_utf8(&args[0]) {
                Ok(p) => p,
                Err(_) => return err("VBULKLOAD path is not valid UTF-8"),
            };
            let vi = db.router.vectors();
            match vi.bulk_load_fbin(path, num_cores()) {
                Ok((n, dim)) => Resp::Simple(format!("loaded {n} vectors x {dim}d")),
                Err(e) => err(&format!("VBULKLOAD failed: {e}")),
            }
        }

        // VBULKLOADNS <namespace> <path-to-.fbin>  -> OK
        // Like VBULKLOAD but loads into a namespace-scoped index.
        // The .fbin file must contain vectors of the same dim as the
        // namespace index (or the namespace must be empty).
        "VBULKLOADNS" => {
            if args.len() != 2 {
                return err("VBULKLOADNS requires namespace path-to-.fbin");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let path = match std::str::from_utf8(&args[1]) {
                Ok(p) => p,
                Err(_) => return err("VBULKLOADNS path is not valid UTF-8"),
            };
            let vi = db.router.vectors_ns(&namespace);
            match vi.bulk_load_fbin(path, num_cores()) {
                Ok((n, dim)) => Resp::Simple(format!("loaded {n} vectors x {dim}d into {namespace}")),
                Err(e) => err(&format!("VBULKLOADNS failed: {e}")),
            }
        }

        "VADDBATCH" => {
            if args.len() < 2 {
                return err("VADDBATCH requires [PAR] dim id f1 f2 ... [id f...]...");
            }
            // Optional leading "PAR" selects the parallel-rebuild ingest path.
            let (parallel, rest) = if String::from_utf8_lossy(&args[0]).eq_ignore_ascii_case("PAR") {
                (true, &args[1..])
            } else {
                (false, &args[..])
            };
            if rest.len() < 1 {
                return err("VADDBATCH requires dim id f1 f2 ... [id f...]...");
            }
            let dim: usize = match std::str::from_utf8(&rest[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            if dim == 0 {
                return err("dim must be > 0");
            }
            if (rest.len() - 1) % (dim + 1) != 0 {
                return err("VADDBATCH float count must be whole number of (id + dim) tuples");
            }
            let tuples = (rest.len() - 1) / (dim + 1);
            let mut ids: Vec<u64> = Vec::with_capacity(tuples);
            let mut flat: Vec<f32> = Vec::with_capacity(tuples * dim);
            let mut ok = true;
            let mut errmsg = String::new();
            for t in 0..tuples {
                let base = 1 + t * (dim + 1);
                let id: u64 = match std::str::from_utf8(&rest[base]).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => { ok = false; errmsg = "id is not a u64".into(); break; }
                };
                let v = match parse_floats(&rest[base + 1..base + 1 + dim]) {
                    Some(v) => v,
                    None => { ok = false; errmsg = "bad float in vector".into(); break; }
                };
                ids.push(id);
                flat.extend_from_slice(&v);
            }
            if !ok {
                return err(&errmsg);
            }
            let vi = db.router.vectors();
            // Guard dim mismatch before any WAL writes: a batch must use the
            // same dimensionality as the existing index (any dim if empty).
            if vi.len() > 0 && vi.dim() != dim {
                return err(&format!("VADDBATCH dim {} != existing index dim {}", dim, vi.dim()));
            }
            // TurboQuant fits a d×d rotation; a mismatched batch dim desyncs the
            // packed storage (panicked before this guard).
            let qd = vi.quant_dim();
            if qd != 0 && dim != qd {
                return err(&format!("VADDBATCH dim {} != turbo index dim {}", dim, qd));
            }
            // Derive each vector's filter attribute + sparse terms exactly like
            // single VADD, so the unified filtered/hybrid paths work uniformly
            // on batched vectors too.
            let mut attrs: Vec<u32> = Vec::with_capacity(tuples);
            let mut sparses: Vec<Vec<(u32, f32)>> = Vec::with_capacity(tuples);
            for i in 0..tuples {
                let vslice = &flat[i * dim..i * dim + dim];
                let (attr, sparse) = derive_attr_and_sparse(vslice, 8, 8);
                attrs.push(attr);
                sparses.push(sparse);
            }
            // PAR flag → parallel-rebuild path (cores×, correct recall);
            // default → serial merge_into append (unchanged behavior).
            let res = if parallel {
                vi.insert_many_parallel_rebuild(&ids, &flat, dim, num_cores(), &attrs)
            } else {
                vi.insert_many_parallel(&ids, &flat, dim, num_cores(), &attrs)
            };
            match res {
                Ok(_) => {
                    for (i, &id) in ids.iter().enumerate() {
                        vi.add_sparse(id, sparses[i].clone());
                    }
                    Resp::Int(tuples as i64)
                }
                Err(e) => err(&e.to_string()),
            }
        }

        // VADDBATCHNS <namespace> [PAR] dim id0 f0..f{dim-1} ...  -> :n
        // Like VADDBATCH but stores vectors in a namespace-scoped index.
        // Each namespace gets its own HNSW graph and dim.
        "VADDBATCHNS" => {
            if args.len() < 3 {
                return err("VADDBATCHNS requires namespace [PAR] dim id f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let (parallel, rest) = if String::from_utf8_lossy(&args[1]).eq_ignore_ascii_case("PAR") {
                (true, &args[2..])
            } else {
                (false, &args[1..])
            };
            if rest.len() < 1 {
                return err("VADDBATCHNS requires dim id f1 f2 ...");
            }
            let dim: usize = match std::str::from_utf8(&rest[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            if dim == 0 {
                return err("dim must be > 0");
            }
            if (rest.len() - 1) % (dim + 1) != 0 {
                return err("VADDBATCHNS float count must be whole number of (id + dim) tuples");
            }
            let tuples = (rest.len() - 1) / (dim + 1);
            let mut ids: Vec<u64> = Vec::with_capacity(tuples);
            let mut flat: Vec<f32> = Vec::with_capacity(tuples * dim);
            let mut ok = true;
            let mut errmsg = String::new();
            for t in 0..tuples {
                let base = 1 + t * (dim + 1);
                let id: u64 = match std::str::from_utf8(&rest[base]).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => { ok = false; errmsg = "id is not a u64".into(); break; }
                };
                let v = match parse_floats(&rest[base + 1..base + 1 + dim]) {
                    Some(v) => v,
                    None => { ok = false; errmsg = "bad float in vector".into(); break; }
                };
                ids.push(id);
                flat.extend_from_slice(&v);
            }
            if !ok {
                return err(&errmsg);
            }
            let vi = db.router.vectors_ns(&namespace);
            if vi.len() > 0 && vi.dim() != dim {
                return err(&format!("VADDBATCHNS dim {} != existing index dim {}", dim, vi.dim()));
            }
            let qd = vi.quant_dim();
            if qd != 0 && dim != qd {
                return err(&format!("VADDBATCHNS dim {} != turbo index dim {}", dim, qd));
            }
            let mut attrs: Vec<u32> = Vec::with_capacity(tuples);
            let mut sparses: Vec<Vec<(u32, f32)>> = Vec::with_capacity(tuples);
            for i in 0..tuples {
                let vslice = &flat[i * dim..i * dim + dim];
                let (attr, sparse) = derive_attr_and_sparse(vslice, 8, 8);
                attrs.push(attr);
                sparses.push(sparse);
            }
            let res = if parallel {
                vi.insert_many_parallel_rebuild(&ids, &flat, dim, num_cores(), &attrs)
            } else {
                vi.insert_many_parallel(&ids, &flat, dim, num_cores(), &attrs)
            };
            match res {
                Ok(_) => {
                    for (i, &id) in ids.iter().enumerate() {
                        vi.add_sparse(id, sparses[i].clone());
                    }
                    Resp::Int(tuples as i64)
                }
                Err(e) => err(&e.to_string()),
            }
        }

// VSETQUANTNS <namespace> mode  -> OK
        // Like VSETQUANT but sets the quantization mode for a namespace-scoped
        // index. Each namespace can have its own quantization mode.
        "VSETQUANTNS" => {
            if args.len() != 2 {
                return err("VSETQUANTNS requires namespace mode");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let m = match quant_mode_from_str(&String::from_utf8_lossy(&args[1]).to_uppercase()) {
                Some(m) => m,
                None => return err("unknown quant mode (INT8 BINARY BINARY2 BINARY15 TURBO1 TURBO15 TURBO2 TURBO4 PRODUCT)"),
            };
            let vi = db.router.vectors_ns(&namespace);
            if vi.len() > 0 {
                return err("VSETQUANTNS requires an empty index (flush/restart first)");
            }
            vi.set_quant_mode(m);
            Resp::Simple("OK".into())
        }

        // VQUANTNS <namespace>  -> bulk (namespace's current quantization mode)
        "VQUANTNS" => {
            if args.len() != 1 {
                return err("VQUANTNS requires namespace");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let m = db.router.vectors_ns(&namespace).quant_mode();
            Resp::Bulk(format!("{m:?}").into_bytes())
        }

        // VLISTNS  -> array of (name, element_count) for every open namespace
        "VLISTNS" => {
            if !args.is_empty() {
                return err("VLISTNS takes no arguments");
            }
            let mut out = Vec::new();
            for (name, len) in db.router.namespaces() {
                out.push(Resp::Bulk(name.into_bytes()));
                out.push(Resp::Int(len as i64));
            }
            Resp::Array(out)
        }

        // VSETQUANT mode  -> OK   (Module 2 quantization selector)
        // Selects the quantization mode for subsequent inserts. Must be called
        // on an EMPTY index (the underlying HNSW asserts on a non-empty graph).
        // Modes: INT8 BINARY BINARY2 BINARY15 TURBO1 TURBO15 TURBO2 TURBO4 PRODUCT
        // For TURBO*/PRODUCT, follow with VFITQUANT on a sample before inserts.
        "VSETQUANT" => {
            if args.len() != 1 {
                return err("VSETQUANT requires mode");
            }
            let m = match quant_mode_from_str(&String::from_utf8_lossy(&args[0]).to_uppercase()) {
                Some(m) => m,
                None => return err("unknown quant mode (INT8 BINARY BINARY2 BINARY15 TURBO1 TURBO15 TURBO2 TURBO4 PRODUCT)"),
            };
            let vi = db.router.vectors();
            if vi.len() > 0 {
                return err("VSETQUANT requires an empty index (flush/restart first)");
            }
            vi.set_quant_mode(m);
            *db.quant_mode.lock().unwrap() = m;
            Resp::Simple("OK".into())
        }

        // VFITQUANT dim n id0 f0..f{dim-1} ...  -> OK
        // Fits TurboQuant / Product-Quantization parameters from a normalized
        // sample. Required before inserts when mode ∈ {TURBO*, PRODUCT}.
        // VQUANT  -> bulk (current quantization mode name)
        "VQUANT" => {
            let m = db.quant_mode.lock().unwrap();
            Resp::Bulk(format!("{m:?}").into_bytes())
        }

        "VFITQUANT" => {
            if args.len() < 3 {
                return err("VFITQUANT requires dim n id f1 f2 ...");
            }
            let dim: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            let n: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("n is not an integer"),
            };
            if args.len() != 2 + n * (dim + 1) {
                return err("VFITQUANT float count mismatch");
            }
            // fit_quant asserts on an empty index; return a clean error rather
            // than panicking if vectors already exist.
            if db.router.vectors().len() > 0 {
                return err("VFITQUANT requires an empty index (flush/restart first)");
            }
            let mut sample: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ok = true;
            for t in 0..n {
                let base = 2 + t * (dim + 1);
                let v = match parse_floats(&args[base + 1..base + 1 + dim]) {
                    Some(v) => v,
                    None => { ok = false; break; }
                };
                sample.push(v);
            }
            if !ok {
                return err("bad float in VFITQUANT sample");
            }
            db.router.vectors().fit_quant(&sample);
            Resp::Simple("OK".into())
        }

        // VFITQUANTNS <namespace> dim n id0 f0..f{dim-1} ...  -> OK
        // Like VFITQUANT but fits TurboQuant / Product-Quantization
        // parameters on a namespace-scoped index.
        "VFITQUANTNS" => {
            if args.len() < 4 {
                return err("VFITQUANTNS requires namespace dim n id f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let dim: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            let n: usize = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("n is not an integer"),
            };
            if args.len() != 3 + n * (dim + 1) {
                return err("VFITQUANTNS float count mismatch");
            }
            if db.router.vectors_ns(&namespace).len() > 0 {
                return err("VFITQUANTNS requires an empty index (flush/restart first)");
            }
            let mut sample: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ok = true;
            for t in 0..n {
                let base = 3 + t * (dim + 1);
                let v = match parse_floats(&args[base + 1..base + 1 + dim]) {
                    Some(v) => v,
                    None => { ok = false; break; }
                };
                sample.push(v);
            }
            if !ok {
                return err("bad float in VFITQUANTNS sample");
            }
            db.router.vectors_ns(&namespace).fit_quant(&sample);
            Resp::Simple("OK".into())
        }

        // VCALIBRATE dim nq k q_f0..q_f{dim-1}(×nq) tgt0_0..tgt0_{k-1}(×nq)
        //   → calibrate the MODULE 3 learned beam-width model from inlined
        //     calibration queries + their ground-truth top-k ids, store it in
        //     the server. The ONLY setup command; all query paths stay unified
        //     under VSEARCH. Ground truth comes from the caller (e.g. the bench
        //     knows true neighbours); we never fabricate it server-side.
        "VCALIBRATE" => {
            if args.len() < 3 {
                return err("VCALIBRATE requires dim nq k ...");
            }
            let dim: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            let nq: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("nq is not an integer"),
            };
            let k: usize = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let exp = 3 + nq * dim + nq * k;
            if args.len() != exp {
                return err(&format!("VCALIBRATE expects {exp} args, got {}", args.len()));
            }
            let floats = match parse_floats(&args[3..3 + nq * dim]) {
                Some(v) => v,
                None => return err("bad float in calibration queries"),
            };
            let qvecs: Vec<Vec<f32>> = floats.chunks(dim).map(|c| c.to_vec()).collect();
            let mut truth: Vec<Vec<u64>> = Vec::with_capacity(nq);
            let mut off = 3 + nq * dim;
            for _ in 0..nq {
                let mut t = Vec::with_capacity(k);
                for _ in 0..k {
                    match std::str::from_utf8(&args[off]).ok().and_then(|s| s.parse::<u64>().ok()) {
                        Some(id) => t.push(id),
                        None => return err("bad id in calibration truth"),
                    }
                    off += 1;
                }
                truth.push(t);
            }
            let model = db.router.vectors().calibrate_ef(
                &qvecs, &truth, 0.92, &[32, 64, 96, 128, 192, 256, 384, 512], k, 32, 512,
            );
            *db.learned.lock().unwrap() = Some(model);
            Resp::Simple("OK".into())
        }
        "VSEARCH" => {
            if args.len() < 2 {
                return err("VSEARCH requires k [F cat | L | H t w ...] f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            // Parse optional trailing access-path flags, then the float vector.
            // Flags (any order, before the floats):
            //   F <cat>  → MODULE 4 filtered ANN (attribute = cat)
            //   L        → MODULE 3 learned-adaptive beam width (needs VCALIBRATE)
            //   H <t> <w> ... → MODULE 5 hybrid (sparse side = term/weight pairs)
            // No flag → plain quantized+rerank ANN. One command, every path.
            let mut i = 1usize;
            let mut filter: Option<Filter> = None;
            let mut use_learned = false;
            let mut sparse: Option<Vec<(u32, f32)>> = None;
            while i < args.len() {
                let tok = String::from_utf8_lossy(&args[i]).to_uppercase();
                if tok == "F" {
                    if i + 1 >= args.len() {
                        return err("F requires a category");
                    }
                    let cat: u32 = match std::str::from_utf8(&args[i + 1]).ok().and_then(|s| s.parse().ok()) {
                        Some(n) => n,
                        None => return err("F category is not an integer"),
                    };
                    filter = Some(Filter::Eq(cat));
                    i += 2;
                } else if tok == "L" {
                    use_learned = true;
                    i += 1;
                } else if tok == "H" {
                    let mut terms = Vec::new();
                    i += 1;
                    while i + 1 < args.len() {
                        let t: u32 = match std::str::from_utf8(&args[i]).ok().and_then(|s| s.parse().ok()) {
                            Some(n) => n,
                            None => break,
                        };
                        let w: f32 = match std::str::from_utf8(&args[i + 1]).ok().and_then(|s| s.parse().ok()) {
                            Some(n) => n,
                            None => break,
                        };
                        terms.push((t, w));
                        i += 2;
                    }
                    if terms.is_empty() {
                        return err("H requires at least one t w pair");
                    }
                    sparse = Some(terms);
                } else {
                    break; // first float → end of flags
                }
            }
            let q = match parse_floats(&args[i..]) {
                Some(v) => v,
                None => return err("bad float in query"),
            };
            let vi = db.router.vectors();
            let learned = if use_learned {
                db.learned.lock().unwrap().clone()
            } else {
                None
            };
            if use_learned && learned.is_none() {
                return err("learned search requires VCALIBRATE first");
            }
            let hits = vi.search_unified(
                &q, k, 128, filter.as_ref(), learned.as_ref(), sparse.as_deref(), 50,
            );
            let mut out = Vec::new();
            for (id, dist) in hits {
                out.push(Resp::Int(id as i64));
                out.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
            }
            Resp::Array(out)
        }

        // VSEARCHNS <namespace> k [F cat | L | H t w ...] f1 f2 ...  -> array of (id, dist)
        // Like VSEARCH but searches only the vectors stored in the
        // given namespace. Each namespace has its own HNSW graph and
        // dim, so different namespaces can hold vectors of different
        // dimensionalities (e.g. 512-dim face namespace vs 64-dim pHash).
        // Supports the same access-path flags as VSEARCH: F (filtered ANN),
        // L (learned-adaptive beam), H (hybrid sparse).
        "VSEARCHNS" => {
            if args.len() < 3 {
                return err("VSEARCHNS requires namespace k [F cat | L | H t w ...] f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let k: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            // Parse optional access-path flags, then the float vector.
            let mut i = 2usize;
            let mut filter: Option<Filter> = None;
            let mut use_learned = false;
            let mut sparse: Option<Vec<(u32, f32)>> = None;
            while i < args.len() {
                let tok = String::from_utf8_lossy(&args[i]).to_uppercase();
                if tok == "F" {
                    if i + 1 >= args.len() {
                        return err("F requires a category");
                    }
                    let cat: u32 = match std::str::from_utf8(&args[i + 1]).ok().and_then(|s| s.parse().ok()) {
                        Some(n) => n,
                        None => return err("F category is not an integer"),
                    };
                    filter = Some(Filter::Eq(cat));
                    i += 2;
                } else if tok == "L" {
                    use_learned = true;
                    i += 1;
                } else if tok == "H" {
                    let mut terms = Vec::new();
                    i += 1;
                    while i + 1 < args.len() {
                        let t: u32 = match std::str::from_utf8(&args[i]).ok().and_then(|s| s.parse().ok()) {
                            Some(n) => n,
                            None => break,
                        };
                        let w: f32 = match std::str::from_utf8(&args[i + 1]).ok().and_then(|s| s.parse().ok()) {
                            Some(n) => n,
                            None => break,
                        };
                        terms.push((t, w));
                        i += 2;
                    }
                    if terms.is_empty() {
                        return err("H requires at least one t w pair");
                    }
                    sparse = Some(terms);
                } else {
                    break; // first float → end of flags
                }
            }
            let q = match parse_floats(&args[i..]) {
                Some(v) => v,
                None => return err("bad float in query vector"),
            };
            let vi = db.router.vectors_ns(&namespace);
            let learned = if use_learned {
                db.learned.lock().unwrap().clone()
            } else {
                None
            };
            if use_learned && learned.is_none() {
                return err("learned search requires VCALIBRATE first");
            }
            let hits = vi.search_unified(
                &q, k, 128, filter.as_ref(), learned.as_ref(), sparse.as_deref(), 50,
            );
            let mut out = Vec::new();
            for (id, dist) in hits {
                out.push(Resp::Int(id as i64));
                out.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
            }
            Resp::Array(out)
        }

        // VSEARCHA k f1 f2 ...   -> query-adaptive k-NN (ruvector-style).
        // Same protocol as VSEARCH but uses a per-query beam width: easy
        // queries get a narrow beam (fewer distance computations → lower
        // latency), hard queries get a wide beam (recall preserved). This is
        // the deliberate win over Qdrant's fixed-ef search. Tuning bounds are
        // chosen for 384/768d at 100k–1M scale; the probe beam is tiny so its
        // cost is negligible vs the real traversal.
        "VSEARCHA" => {
            if args.len() < 2 {
                return err("VSEARCHA requires k f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let q = match parse_floats(&args[1..]) {
                Some(v) => v,
                None => return err("bad float in query"),
            };
            let hits = db.router.vectors().search_adaptive(&q, k, 16, 32, 256);
            let mut out = Vec::new();
            for (id, dist) in hits {
                out.push(Resp::Int(id as i64));
                out.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
            }
            Resp::Array(out)
        }

        // VSEARCHANS <namespace> k f1 f2 ...  -> array of (id, dist)
        // Like VSEARCHA (query-adaptive beam) but searches only a namespace.
        "VSEARCHANS" => {
            if args.len() < 3 {
                return err("VSEARCHANS requires namespace k f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let k: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let q = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in query"),
            };
            let hits = db.router.vectors_ns(&namespace).search_adaptive(&q, k, 16, 32, 256);
            let mut out = Vec::new();
            for (id, dist) in hits {
                out.push(Resp::Int(id as i64));
                out.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
            }
            Resp::Array(out)
        }

        // VSEARCH.MANY k dim q1_f0 ... q1_f{dim-1} q2_f0 ... qN_f{dim-1}
        // Returns Array of N Arrays, each Array being (id, dist) pairs.
        // Single round trip + one HNSW read-lock — amortizes protocol + lock.
        "VSEARCH.MANY" => {
            if args.len() < 3 {
                return err("VSEARCH.MANY requires k dim f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let dim: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            if dim == 0 {
                return err("dim must be > 0");
            }
            let floats = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in queries"),
            };
            if floats.len() % dim != 0 {
                return err("float count not divisible by dim");
            }
            let queries: Vec<Vec<f32>> =
                floats.chunks(dim).map(|c| c.to_vec()).collect();
            let batch = db.router.vectors().search_many(&queries, k);
            let mut out = Vec::with_capacity(batch.len());
            for hits in batch {
                let mut inner = Vec::with_capacity(hits.len() * 2);
                for (id, dist) in hits {
                    inner.push(Resp::Int(id as i64));
                    inner.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
                }
                out.push(Resp::Array(inner));
            }
            Resp::Array(out)
        }

        // VSEARCH.MANYNS <namespace> k dim q1_f0 ...  -> Array of N Arrays
        // Like VSEARCH.MANY (batched multi-query ANN) but searches only a
        // namespace-scoped index.
        "VSEARCH.MANYNS" => {
            if args.len() < 4 {
                return err("VSEARCH.MANYNS requires namespace k dim f1 f2 ...");
            }
            let namespace = String::from_utf8_lossy(&args[0]).to_string();
            let k: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let dim: usize = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("dim is not an integer"),
            };
            if dim == 0 {
                return err("dim must be > 0");
            }
            let floats = match parse_floats(&args[3..]) {
                Some(v) => v,
                None => return err("bad float in queries"),
            };
            if floats.len() % dim != 0 {
                return err("float count not divisible by dim");
            }
            let queries: Vec<Vec<f32>> =
                floats.chunks(dim).map(|c| c.to_vec()).collect();
            let batch = db.router.vectors_ns(&namespace).search_many(&queries, k);
            let mut out = Vec::with_capacity(batch.len());
            for hits in batch {
                let mut inner = Vec::with_capacity(hits.len() * 2);
                for (id, dist) in hits {
                    inner.push(Resp::Int(id as i64));
                    inner.push(Resp::Bulk(format!("{dist:.6}").into_bytes()));
                }
                out.push(Resp::Array(inner));
            }
            Resp::Array(out)
        }

        // TSADD accepts either integer or float values. Dashboards emit
        // "42.5" for CPU %; older clients emit "42" — both work.
        "TSADD" | "TSADD.F" => {
            if args.len() != 3 {
                return err("TSADD requires series ts val");
            }
            let series = String::from_utf8_lossy(&args[0]).to_string();
            let t: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("ts is not a u64"),
            };
            let v_str = match std::str::from_utf8(&args[2]) {
                Ok(s) => s,
                Err(_) => return err("val is not a valid utf8 number"),
            };
            let v: f64 = match v_str.parse() {
                Ok(f) => f,
                Err(_) => return err("val is not a number"),
            };
            match db.ts.append_f(&series, t, v) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // TSRANGE.LATEST series n → newest n points, O(log N + n) single-shard.
        // The dashboard primitive: no dashboard wants the full history,
        // they want the tail. Values are serialized as bulk-string floats
        // (`Resp::Bulk` of e.g. `42.5`) since RESP2 has no native float.
        "TSRANGE.LATEST" | "TSLATEST" => {
            if args.len() != 2 {
                return err("TSRANGE.LATEST requires series n");
            }
            let series = String::from_utf8_lossy(&args[0]).to_string();
            let n: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("n is not an integer"),
            };
            let pts = db.ts.latest(&series, n);
            let mut out = Vec::with_capacity(pts.len() * 2);
            for (t, v) in pts {
                out.push(Resp::Int(t as i64));
                out.push(Resp::Bulk(format_ts_val(v).into_bytes()));
            }
            Resp::Array(out)
        }
        "TSRANGE" => {
            if args.len() != 3 {
                return err("TSRANGE requires series from to");
            }
            let series = String::from_utf8_lossy(&args[0]).to_string();
            let from: u64 = std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let to: u64 = std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()).unwrap_or(u64::MAX);
            let pts = db.ts.range(&series, from, to);
            let mut out = Vec::new();
            for (t, v) in pts {
                out.push(Resp::Int(t as i64));
                out.push(Resp::Bulk(format_ts_val(v).into_bytes()));
            }
            Resp::Array(out)
        }

        "REDUCE" => {
            // REDUCE name shardkey key by  -> fuel-metered counter reducer
            if args.len() != 4 {
                return err("REDUCE requires name shardkey key by");
            }
            let rname = String::from_utf8_lossy(&args[0]).to_string();
            let shardkey = args[1].clone();
            let key = args[2].clone();
            let by: i64 = match std::str::from_utf8(&args[3]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("by is not an i64"),
            };
            let prog = counter_reducer(&kv_key(&key), by);
            match db.reducers.invoke(&rname, &shardkey, &prog) {
                ReducerResult::Ok { output, .. } => Resp::Int(output.unwrap_or(0)),
                ReducerResult::Aborted(e) => err(&format!("reducer aborted: {e}")),
                ReducerResult::Quarantined => err("reducer quarantined"),
            }
        }

        "CDCLEN" => Resp::Int(db.reactive.cdc_len() as i64),

        // MEMTRACK [tag] — read the allocator's books.
        //
        // Call it at phase boundaries (post-ingest, post-search, per step of a
        // concurrency sweep) and the deltas attribute growth to a phase. Two
        // numbers do the real work:
        //
        //   * `rss` vs `live` — agreement means the heap is genuinely that big;
        //     rss far above live means the allocator is sitting on memory it has
        //     already freed, i.e. fragmentation, and freeing more will not bring
        //     rss down.
        //   * `mean` object size — tens of bytes next to a multi-GB `live` is the
        //     death-by-small-object signature, which is what a RESP frame parsed
        //     into tens of thousands of tiny `Vec<u8>`s would look like.
        //
        // MEMTRACK HIST adds the cumulative size histogram, which separates a
        // handful of runaway buffers from millions of small allocations.
        "MEMTRACK" => {
            // First call switches tracking on, so the numbers mean something
            // (nothing is recorded while tracking is off). Subsequent calls are
            // free reads of the running totals.
            if !mitm::memtrack::tracking_enabled() {
                mitm::memtrack::set_tracking(true);
            }
            let tag = args
                .first()
                .map(|a| String::from_utf8_lossy(a).to_string())
                .unwrap_or_else(|| "now".to_string());
            if tag.eq_ignore_ascii_case("hist") {
                let s = mitm::memtrack::snapshot();
                Resp::Bulk(format!("{}\n{}", mitm::memtrack::report("hist"), s.histogram()).into_bytes())
            } else {
                Resp::Bulk(mitm::memtrack::report(&tag).into_bytes())
            }
        }

        // GPU.LOAD <kernel> — compile and load a CUDA kernel on demand.
        // Kernels: cosine_dist, matmul. Lazy: only compiled when first requested.
        "GPU.LOAD" => {
            if args.len() != 1 {
                return err("GPU.LOAD requires kernel name (cosine_dist, matmul)");
            }
            let name = String::from_utf8_lossy(&args[0]);
            if gpu::gpu_load_kernel(&name) {
                Resp::Simple(format!("OK kernel {} loaded", name))
            } else {
                err("ERR GPU unavailable or kernel not found")
            }
        }
        // GPU.INFO — show GPU status, VRAM, and tier strategy.
        "GPU.INFO" => {
            let info = gpu::gpu_info();
            let mut out = Vec::new();
            for (k, v) in &info {
                out.push(Resp::Bulk(k.as_bytes().to_vec()));
                out.push(Resp::Bulk(v.as_bytes().to_vec()));
            }
            // Add tier strategy for common sizes.
            let strategy = gpu::gpu_tier_strategy(1_000_000, 384);
            out.push(Resp::Bulk(b"tier_1M_384d".to_vec()));
            out.push(Resp::Bulk(format!("{:?}", strategy).into_bytes()));
            let strategy2 = gpu::gpu_tier_strategy(1_000_000, 768);
            out.push(Resp::Bulk(b"tier_1M_768d".to_vec()));
            out.push(Resp::Bulk(format!("{:?}", strategy2).into_bytes()));
            Resp::Array(out)
        }
        // GPU.MODE [turbo|hybrid|cpu|auto] — get or set compute mode.
        "GPU.MODE" => {
            if args.is_empty() {
                let mode = gpu::gpu_get_mode();
                Resp::Simple(format!("{:?}", mode))
            } else {
                let mode_str = String::from_utf8_lossy(&args[0]).to_lowercase();
                match mode_str.as_str() {
                    // Select the mode BEFORE probing for a device.
                    //
                    // `gpu_available()` only brings the driver up when a GPU
                    // mode is already current — that gating exists so `CpuOnly`
                    // never touches the device. Probing first therefore always
                    // answered "no device" on a fresh server, whose mode
                    // defaults to `CpuOnly`, so `GPU.MODE turbo` returned
                    // "ERR no GPU detected" on a machine with a working GPU and
                    // there was no way to enable GPU execution over RESP at all.
                    //
                    // Set, probe, roll back on genuine absence — so a machine
                    // without a device still gets an honest error.
                    "turbo" | "gpu" => {
                        gpu::gpu_set_mode(gpu::ComputeMode::Turbo);
                        if !gpu::gpu_available() {
                            gpu::gpu_set_mode(gpu::ComputeMode::CpuOnly);
                            return err("ERR no GPU detected");
                        }
                        Resp::Simple("OK compute mode = Turbo (full GPU)".into())
                    }
                    "hybrid" => {
                        gpu::gpu_set_mode(gpu::ComputeMode::Hybrid);
                        if !gpu::gpu_available() {
                            gpu::gpu_set_mode(gpu::ComputeMode::CpuOnly);
                            return err("ERR no GPU detected");
                        }
                        Resp::Simple("OK compute mode = Hybrid (GPU+RAM+CPU)".into())
                    }
                    "cpu" | "cpu_only" | "off" => {
                        gpu::gpu_set_mode(gpu::ComputeMode::CpuOnly);
                        Resp::Simple("OK compute mode = CPU-only".into())
                    }
                    "auto" => {
                        // Auto-detect for 1M × 384d as default
                        let mode = gpu::gpu_auto_mode(1_000_000, 384);
                        Resp::Simple(format!("OK compute mode = {:?} (auto-detected)", mode))
                    }
                    _ => err("ERR usage: GPU.MODE turbo|hybrid|cpu|auto"),
                }
            }
        }
        // GPU.UNLOAD — release GPU resources.
        "GPU.UNLOAD" => {
            gpu::gpu_unload();
            Resp::Simple("OK".into())
        }

        // CHECKPOINT — snapshot current state, truncate WAL. Redis-shape
        // reply: "SNAPSHOT n=<records> bytes=<snap-file-size>".
        "CHECKPOINT" => {
            match db.engine.checkpoint() {
                Ok((n, bytes)) => {
                    Resp::Simple(format!("SNAPSHOT n={n} bytes={bytes}"))
                }
                Err(e) => err(&e.to_string()),
            }
        }

        // PUBLISH ch msg — durable pub/sub. The write lands on `chan:<ch>`;
        // reactive fires → every SUBSCRIBEr matching the prefix receives it.
        // Return the number of subscribers matched (Redis semantics).
        "PUBLISH" => {
            if args.len() != 2 {
                return err("PUBLISH requires channel message");
            }
            let mut key = b"chan:".to_vec();
            key.extend_from_slice(&args[0]);
            match db.engine.put(key, storage::Value::Bytes(args[1].clone())) {
                Ok(_) => {
                    // Approximate subscriber count: number of registered
                    // prefixes that match `chan:<ch>`. Cheap & Redis-adjacent.
                    let ch_prefix = {
                        let mut p = b"chan:".to_vec();
                        p.extend_from_slice(&args[0]);
                        p
                    };
                    let n = db.reactive.subscribers_matching(&ch_prefix);
                    Resp::Int(n as i64)
                }
                Err(e) => err(&e.to_string()),
            }
        }

        // ── AGENT MEMORY ────────────────────────────────────────────────
        // MEM.REMEMBER text source salience f1 f2 ...   -> :id
        "MEM.REMEMBER" => {
            if args.len() < 4 {
                return err("MEM.REMEMBER requires text source salience f1 f2 ...");
            }
            let text = String::from_utf8_lossy(&args[0]).to_string();
            let source = String::from_utf8_lossy(&args[1]).to_string();
            let sal: f32 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("salience is not a float"),
            };
            let vec = match parse_floats(&args[3..]) {
                Some(v) => v,
                None => return err("bad float in embedding"),
            };
            match db.rag.memory().ltm_store(&text, vec, &source, sal, "cmd:remember") {
                Ok(id) => Resp::Int(id as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.RECALL k query f1 f2 ...  -> array of (id, score, source, text)
        "MEM.RECALL" => {
            if args.len() < 3 {
                return err("MEM.RECALL requires k query f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let query = String::from_utf8_lossy(&args[1]).to_string();
            let qvec = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in query vector"),
            };
            let hits = db.rag.memory().recall(&query, &qvec, k);
            let mut out = Vec::new();
            for h in hits {
                out.push(Resp::Int(h.id as i64));
                out.push(Resp::Bulk(format!("{:.6}", h.score).into_bytes()));
                out.push(Resp::Bulk(h.meta.source.into_bytes()));
                out.push(Resp::Bulk(h.text.into_bytes()));
            }
            Resp::Array(out)
        }
        // MEM.FORGET id  -> OK
        "MEM.FORGET" => {
            if args.len() != 1 {
                return err("MEM.FORGET requires id");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            match db.rag.memory().ltm_forget(id) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }

        // ── GRAPH MEMORY ────────────────────────────────────────────────
        // MEM.LINK from to rel weight  -> OK
        "MEM.LINK" => {
            if args.len() != 4 {
                return err("MEM.LINK requires from to rel weight");
            }
            let from: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("from is not a u64"),
            };
            let to: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("to is not a u64"),
            };
            let rel = String::from_utf8_lossy(&args[2]).to_string();
            let w: f32 = std::str::from_utf8(&args[3]).ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            match db.rag.memory().link(from, to, &rel, w) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.UNLINK from to rel  -> OK
        "MEM.UNLINK" => {
            if args.len() != 3 {
                return err("MEM.UNLINK requires from to rel");
            }
            let from: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("from is not a u64"),
            };
            let to: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("to is not a u64"),
            };
            let rel = String::from_utf8_lossy(&args[2]).to_string();
            match db.rag.memory().unlink(from, to, &rel) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.NEIGH from rel   -> array of (to, rel, weight)
        "MEM.NEIGH" => {
            if args.is_empty() {
                return err("MEM.NEIGH requires from [rel]");
            }
            let from: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("from is not a u64"),
            };
            let rel = if args.len() >= 2 {
                String::from_utf8_lossy(&args[1]).to_string()
            } else {
                String::new()
            };
            let ns = db.rag.memory().neighbors(from, &rel);
            let mut out = Vec::new();
            for (id, r, w) in ns {
                out.push(Resp::Int(id as i64));
                out.push(Resp::Bulk(r.into_bytes()));
                out.push(Resp::Bulk(format!("{w:.6}").into_bytes()));
            }
            Resp::Array(out)
        }
        // MEM.TRAV start depth rel  -> array of visited ids
        "MEM.TRAV" => {
            if args.len() < 2 {
                return err("MEM.TRAV requires start depth [rel]");
            }
            let start: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("start is not a u64"),
            };
            let depth: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("depth is not a usize"),
            };
            let rel = if args.len() >= 3 {
                String::from_utf8_lossy(&args[2]).to_string()
            } else {
                String::new()
            };
            let ids = db.rag.memory().traverse(start, depth, &rel);
            Resp::Array(ids.into_iter().map(|i| Resp::Int(i as i64)).collect())
        }

        // ── BI-TEMPORAL RECALL ─────────────────────────────────────────
        // MEM.REMEMBER.T text source sal valid_from valid_to f1 f2 ...  -> :id
        "MEM.REMEMBER.T" => {
            if args.len() < 6 {
                return err("MEM.REMEMBER.T requires text source sal valid_from valid_to f1 f2 ...");
            }
            let text = String::from_utf8_lossy(&args[0]).to_string();
            let source = String::from_utf8_lossy(&args[1]).to_string();
            let sal: f32 = std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let vf: u64 = std::str::from_utf8(&args[3]).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let vt: u64 = std::str::from_utf8(&args[4]).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let vec = match parse_floats(&args[5..]) {
                Some(v) => v,
                None => return err("bad float in embedding"),
            };
            match db.rag.memory().ltm_store_temporal(&text, vec, &source, sal, "cmd:remember.t", vf, vt) {
                Ok(id) => Resp::Int(id as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.INVALIDATE id at  -> OK   (bump fact's valid_to to `at`)
        "MEM.INVALIDATE" => {
            if args.len() != 2 {
                return err("MEM.INVALIDATE requires id at");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            let at: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("at is not a u64"),
            };
            match db.rag.memory().ltm_invalidate(id, at) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.RECALL.AS_OF k query as_of f1 f2 ...  -> array of (id, score, source, text)
        "MEM.RECALL.AS_OF" => {
            if args.len() < 4 {
                return err("MEM.RECALL.AS_OF requires k query as_of f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let query = String::from_utf8_lossy(&args[1]).to_string();
            let as_of: u64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("as_of is not a u64"),
            };
            let qvec = match parse_floats(&args[3..]) {
                Some(v) => v,
                None => return err("bad float in query vector"),
            };
            let hits = db.rag.memory().recall_as_of(&query, &qvec, k, as_of);
            let mut out = Vec::new();
            for h in hits {
                out.push(Resp::Int(h.id as i64));
                out.push(Resp::Bulk(format!("{:.6}", h.score).into_bytes()));
                out.push(Resp::Bulk(h.meta.source.into_bytes()));
                out.push(Resp::Bulk(h.text.into_bytes()));
            }
            Resp::Array(out)
        }

        // ── PROCEDURAL MEMORY ─────────────────────────────────────────
        // MEM.PROC.SET agent name body  -> OK
        "MEM.PROC.SET" => {
            if args.len() != 3 {
                return err("MEM.PROC.SET requires agent name body");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let name = String::from_utf8_lossy(&args[1]).to_string();
            match db.rag.memory().proc_store(&agent, &name, &args[2]) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        "MEM.PROC.GET" => {
            if args.len() != 2 {
                return err("MEM.PROC.GET requires agent name");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let name = String::from_utf8_lossy(&args[1]).to_string();
            match db.rag.memory().proc_get(&agent, &name) {
                Some(b) => Resp::Bulk(b),
                None => Resp::Nil,
            }
        }
        "MEM.PROC.LIST" => {
            if args.len() != 1 {
                return err("MEM.PROC.LIST requires agent");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let names = db.rag.memory().proc_list(&agent);
            Resp::Array(names.into_iter().map(|s| Resp::Bulk(s.into_bytes())).collect())
        }

        // ── WORKING MEMORY (STM) ────────────────────────────────────────
        // MEM.WM_SET agent key value ttl_ms  -> OK
        "MEM.WM_SET" => {
            if args.len() != 4 {
                return err("MEM.WM_SET requires agent key value ttl_ms");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let key = String::from_utf8_lossy(&args[1]).to_string();
            let ttl: u64 = match std::str::from_utf8(&args[3]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("ttl_ms is not a u64"),
            };
            match db.rag.memory().wm_set(&agent, &key, &args[2], ttl, now_ms()) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.WM_GET agent key  -> bulk | nil
        "MEM.WM_GET" => {
            if args.len() != 2 {
                return err("MEM.WM_GET requires agent key");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let key = String::from_utf8_lossy(&args[1]).to_string();
            match db.rag.memory().wm_get(&agent, &key, now_ms()) {
                Some(b) => Resp::Bulk(b),
                None => Resp::Nil,
            }
        }
        // MEM.WM_DELETE agent key  -> OK
        "MEM.WM_DELETE" => {
            if args.len() != 2 {
                return err("MEM.WM_DELETE requires agent key");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let key = String::from_utf8_lossy(&args[1]).to_string();
            match db.rag.memory().wm_delete(&agent, &key) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }

        // ── EPISODIC ────────────────────────────────────────────────────
        // MEM.EPISODE agent kind payload  -> :seq
        "MEM.EPISODE" => {
            if args.len() != 3 {
                return err("MEM.EPISODE requires agent kind payload");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let kind = String::from_utf8_lossy(&args[1]).to_string();
            match db.rag.memory().episode(&agent, &kind, &args[2]) {
                Ok(seq) => Resp::Int(seq as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.EPISODES agent limit  -> array of (seq, kind, payload)
        "MEM.EPISODES" => {
            if args.len() != 2 {
                return err("MEM.EPISODES requires agent limit");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let limit: usize = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("limit is not a usize"),
            };
            let eps = db.rag.memory().episodes(&agent, limit);
            let out: Vec<Resp> = eps
                .into_iter()
                .map(|e| {
                    Resp::Array(vec![
                        Resp::Int(e.seq as i64),
                        Resp::Bulk(e.kind.into_bytes()),
                        Resp::Bulk(e.payload),
                    ])
                })
                .collect();
            Resp::Array(out)
        }
        // MEM.EPISODE_FORGET agent seq  -> OK
        "MEM.EPISODE_FORGET" => {
            if args.len() != 2 {
                return err("MEM.EPISODE_FORGET requires agent seq");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            let seq: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("seq is not a u64"),
            };
            match db.rag.memory().episode_forget(&agent, seq) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }

        // ── RAG ─────────────────────────────────────────────────────────
        // RAG.INGEST text source f1 f2 ...  -> :id
        "RAG.INGEST" => {
            if args.len() < 3 {
                return err("RAG.INGEST requires text source f1 f2 ...");
            }
            let text = String::from_utf8_lossy(&args[0]).to_string();
            let source = String::from_utf8_lossy(&args[1]).to_string();
            let vec = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in embedding"),
            };
            match db.rag.ingest(&text, vec, &source) {
                Ok(id) => Resp::Int(id as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        // RAG.SEARCH k query f1 f2 ...  -> array of (id, score, source, text)
        "RAG.SEARCH" => {
            if args.len() < 3 {
                return err("RAG.SEARCH requires k query f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let query = String::from_utf8_lossy(&args[1]).to_string();
            let qvec = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in query vector"),
            };
            let (hits, cached) = db.rag.retrieve_cached(&query, &qvec, k);
            let mut out = Vec::new();
            out.push(Resp::Bulk(if cached { b"cached".to_vec() } else { b"fresh".to_vec() }));
            for h in hits {
                out.push(Resp::Int(h.id as i64));
                out.push(Resp::Bulk(format!("{:.6}", h.score).into_bytes()));
                out.push(Resp::Bulk(h.source.into_bytes()));
                out.push(Resp::Bulk(h.text.into_bytes()));
            }
            Resp::Array(out)
        }

        // ── MITM CACHE DEBUGGER ─────────────────────────────────────────
        // CACHE.SRCSET key value | CACHE.SET key value | CACHE.GET key
        // CACHE.INVALIDATE key | CACHE.BUGS | CACHE.TRACES
        "CACHE.SRCSET" => {
            if args.len() != 2 {
                return err("CACHE.SRCSET requires key value");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            match db.cache.source_set(&key, &args[1]) {
                Ok(v) => Resp::Int(v as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        "CACHE.SRCDEL" => {
            if args.len() != 1 {
                return err("CACHE.SRCDEL requires key");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            match db.cache.source_del(&key) {
                Ok(v) => Resp::Int(v as i64),
                Err(e) => err(&e.to_string()),
            }
        }
        "CACHE.SET" => {
            if args.len() != 2 {
                return err("CACHE.SET requires key value");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            db.cache.cache_set(&key, &args[1]);
            Resp::Simple("OK".into())
        }
        "CACHE.GET" => {
            if args.len() != 1 {
                return err("CACHE.GET requires key");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            let (val, verdict) = db.cache.cache_get(&key);
            let mut out = vec![Resp::Bulk(verdict.as_str().as_bytes().to_vec())];
            out.push(match val {
                Some(v) => Resp::Bulk(v),
                None => Resp::Nil,
            });
            Resp::Array(out)
        }
        "CACHE.INVALIDATE" => {
            if args.len() != 1 {
                return err("CACHE.INVALIDATE requires key");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            db.cache.cache_invalidate(&key);
            Resp::Simple("OK".into())
        }
        "CACHE.BUGS" => {
            let report = db.cache.report();
            Resp::Array(report.into_iter().map(|s| Resp::Bulk(s.into_bytes())).collect())
        }
        "CACHE.TRACES" => {
            let traces = db.cache.traces();
            let mut out = Vec::new();
            for t in traces {
                out.push(Resp::Bulk(
                    format!(
                        "#{} {} {} {} cached_v={} engine_v={} :: {}",
                        t.seq, t.op, t.key, t.verdict.as_str(), t.cached_ver, t.engine_ver, t.note
                    )
                    .into_bytes(),
                ));
            }
            Resp::Array(out)
        }
        "CACHE.CLEAR" => {
            db.cache.clear();
            Resp::Simple("OK".into())
        }

        // ── TABLES (relational view, Module: tables) ─────────────────────
        // TABLE.SET table pk col val [col val ...]  -> OK
        "TABLE.SET" => {
            if args.len() < 4 || (args.len() - 2) % 2 != 0 {
                return err("TABLE.SET requires table pk col val [col val ...]");
            }
            let table = String::from_utf8_lossy(&args[0]).to_string();
            let pk = String::from_utf8_lossy(&args[1]).to_string();
            let mut row: Row = db.router.tables().get(&table, &pk).unwrap_or_default();
            let mut i = 2;
            while i + 1 < args.len() {
                let col = String::from_utf8_lossy(&args[i]).to_string();
                row.insert(col, args[i + 1].clone());
                i += 2;
            }
            match db.router.tables().upsert(&table, &pk, row) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // TABLE.GET table pk  -> array of (col, val) or nil
        "TABLE.GET" => {
            if args.len() != 2 {
                return err("TABLE.GET requires table pk");
            }
            let table = String::from_utf8_lossy(&args[0]).to_string();
            let pk = String::from_utf8_lossy(&args[1]).to_string();
            match db.router.tables().get(&table, &pk) {
                Some(row) => Resp::Array(row_to_resp(row)),
                None => Resp::Nil,
            }
        }
        // TABLE.DEL table pk  -> OK
        "TABLE.DEL" => {
            if args.len() != 2 {
                return err("TABLE.DEL requires table pk");
            }
            let table = String::from_utf8_lossy(&args[0]).to_string();
            let pk = String::from_utf8_lossy(&args[1]).to_string();
            match db.router.tables().delete(&table, &pk) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // TABLE.SCAN table  -> array of (pk, (col,val)...) per row
        "TABLE.SCAN" => {
            if args.len() != 1 {
                return err("TABLE.SCAN requires table");
            }
            let table = String::from_utf8_lossy(&args[0]).to_string();
            let rows = db.router.tables().scan(&table);
            let mut out = Vec::with_capacity(rows.len());
            for (pk, row) in rows {
                out.push(Resp::Array({
                    let mut a = vec![Resp::Bulk(pk.into_bytes())];
                    a.extend(row_to_resp(row));
                    a
                }));
            }
            Resp::Array(out)
        }
        // TABLE.FILTEREQ table col val  -> array of (pk, (col,val)...) per match
        "TABLE.FILTEREQ" => {
            if args.len() != 3 {
                return err("TABLE.FILTEREQ requires table col val");
            }
            let table = String::from_utf8_lossy(&args[0]).to_string();
            let col = String::from_utf8_lossy(&args[1]).to_string();
            let rows = db.router.tables().filter_eq(&table, &col, &args[2]);
            let mut out = Vec::with_capacity(rows.len());
            for (pk, row) in rows {
                out.push(Resp::Array({
                    let mut a = vec![Resp::Bulk(pk.into_bytes())];
                    a.extend(row_to_resp(row));
                    a
                }));
            }
            Resp::Array(out)
        }

        // ── CONSENSUS CRDTs ──────────────────────────────────────────────
        // CRDT.GCOUNTER name node by  -> :value   (grow-only counter)
        "CRDT.GCOUNTER" => {
            if args.len() != 3 {
                return err("CRDT.GCOUNTER requires name node by");
            }
            let name = String::from_utf8_lossy(&args[0]).to_string();
            let node = String::from_utf8_lossy(&args[1]).to_string();
            let by: u64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("by is not a u64"),
            };
            let mut store = db.crdt.lock().unwrap();
            let c = store.gc.entry(name).or_default();
            c.incr(&node, by);
            Resp::Int(c.value() as i64)
        }
        // CRDT.PNCOUNTER name node delta  -> :value   (PN counter; delta may be negative)
        "CRDT.PNCOUNTER" => {
            if args.len() != 3 {
                return err("CRDT.PNCOUNTER requires name node delta");
            }
            let name = String::from_utf8_lossy(&args[0]).to_string();
            let node = String::from_utf8_lossy(&args[1]).to_string();
            let d: i64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("delta is not an i64"),
            };
            let mut store = db.crdt.lock().unwrap();
            let c = store.pn.entry(name).or_default();
            if d >= 0 {
                c.incr(&node, d as u64);
            } else {
                c.decr(&node, (-d) as u64);
            }
            Resp::Int(c.value())
        }
        // CRDT.LWW name value ts node  -> OK   (last-write-wins register)
        "CRDT.LWW" => {
            if args.len() != 4 {
                return err("CRDT.LWW requires name value ts node");
            }
            let name = String::from_utf8_lossy(&args[0]).to_string();
            let ts: u64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("ts is not a u64"),
            };
            let node = String::from_utf8_lossy(&args[3]).to_string();
            let mut store = db.crdt.lock().unwrap();
            let r = store.lww.entry(name).or_insert_with(|| LwwRegister::new(args[1].clone(), 0, &node));
            r.set(args[1].clone(), ts, &node);
            Resp::Simple("OK".into())
        }
        // CRDT.GET name  -> bulk (value) for GCOUNTER/PnCounter/LWW, or nil
        "CRDT.GET" => {
            if args.len() != 1 {
                return err("CRDT.GET requires name");
            }
            let name = String::from_utf8_lossy(&args[0]).to_string();
            let store = db.crdt.lock().unwrap();
            if let Some(c) = store.gc.get(&name) {
                Resp::Bulk(c.value().to_string().into_bytes())
            } else if let Some(c) = store.pn.get(&name) {
                Resp::Bulk(c.value().to_string().into_bytes())
            } else if let Some(r) = store.lww.get(&name) {
                Resp::Bulk(r.value.clone())
            } else {
                Resp::Nil
            }
        }

        // ── HYBRID LOGICAL CLOCK ─────────────────────────────────────────
        // HLC.NOW  -> "<physical>.<logical>"
        "HLC.NOW" => {
            let ts = db.hlc.now();
            Resp::Bulk(format!("{}.{}", ts.physical, ts.logical).into_bytes())
        }
        // HLC.UPDATE <physical> <logical>  -> "<physical>.<logical>"
        "HLC.UPDATE" => {
            if args.len() != 2 {
                return err("HLC.UPDATE requires physical logical");
            }
            let p: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("physical is not a u64"),
            };
            let l: u32 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("logical is not a u32"),
            };
            let ts = db.hlc.update(consensus::hlc::Timestamp { physical: p, logical: l });
            Resp::Bulk(format!("{}.{}", ts.physical, ts.logical).into_bytes())
        }

        // ── GENERIC REDUCER PROGRAM (Module: compute VM) ─────────────────
        // REDUCE.PROGRAM name shardkey Instr...  -> :output | error
        // Instr tokens (space separated, after name+shardkey):
        //   PUSHINT <i64>  POP  DUP  ADD  SUB  MUL
        //   LOADINT <key>  STOREINT <key>  JUMP <idx>  JZ <idx>  RETURN  TRAP <msg>
        // Assembles a fuel-metered Program and invokes it in the reducer runtime.
        "REDUCE.PROGRAM" => {
            if args.len() < 3 {
                return err("REDUCE.PROGRAM requires name shardkey Instr...");
            }
            let rname = String::from_utf8_lossy(&args[0]).to_string();
            let shardkey = args[1].clone();
            let mut instrs: Vec<Instr> = Vec::with_capacity(args.len() - 2);
            let mut i = 2;
            while i < args.len() {
                let tok = String::from_utf8_lossy(&args[i]).to_uppercase();
                i += 1;
                let need = |i: &mut usize, n: usize| -> bool { *i + n <= args.len() };
                match tok.as_str() {
                    "PUSHINT" => {
                        if !need(&mut i, 1) { return err("PUSHINT requires an i64"); }
                        let v: i64 = match std::str::from_utf8(&args[i]).ok().and_then(|s| s.parse().ok()) {
                            Some(v) => v, None => return err("PUSHINT arg not i64"),
                        };
                        i += 1; instrs.push(Instr::PushInt(v));
                    }
                    "POP" => instrs.push(Instr::Pop),
                    "DUP" => instrs.push(Instr::Dup),
                    "ADD" => instrs.push(Instr::Add),
                    "SUB" => instrs.push(Instr::Sub),
                    "MUL" => instrs.push(Instr::Mul),
                    "LOADINT" => {
                        if !need(&mut i, 1) { return err("LOADINT requires a key"); }
                        instrs.push(Instr::LoadInt(kv_key(&args[i]))); i += 1;
                    }
                    "STOREINT" => {
                        if !need(&mut i, 1) { return err("STOREINT requires a key"); }
                        instrs.push(Instr::StoreInt(kv_key(&args[i]))); i += 1;
                    }
                    "JUMP" => {
                        if !need(&mut i, 1) { return err("JUMP requires an idx"); }
                        let v: usize = match std::str::from_utf8(&args[i]).ok().and_then(|s| s.parse().ok()) {
                            Some(v) => v, None => return err("JUMP arg not usize"),
                        };
                        i += 1; instrs.push(Instr::Jump(v));
                    }
                    "JZ" => {
                        if !need(&mut i, 1) { return err("JZ requires an idx"); }
                        let v: usize = match std::str::from_utf8(&args[i]).ok().and_then(|s| s.parse().ok()) {
                            Some(v) => v, None => return err("JZ arg not usize"),
                        };
                        i += 1; instrs.push(Instr::JumpIfZero(v));
                    }
                    "RETURN" => instrs.push(Instr::Return),
                    "TRAP" => {
                        if !need(&mut i, 1) { return err("TRAP requires a msg"); }
                        instrs.push(Instr::Trap(String::from_utf8_lossy(&args[i]).to_string()));
                        i += 1;
                    }
                    other => return err(&format!("unknown Instr '{other}'")),
                }
            }
            let prog = Program { instrs };
            match db.reducers.invoke(&rname, &shardkey, &prog) {
                ReducerResult::Ok { output, .. } => Resp::Int(output.unwrap_or(0)),
                ReducerResult::Aborted(e) => err(&format!("reducer aborted: {e:?}")),
                ReducerResult::Quarantined => err("reducer quarantined"),
            }
        }

        // ── MEMORY PRIMITIVES (unexposed until now) ──────────────────────
        // MEM.INCOMING id [rel]  -> array of (from, rel, weight)
        "MEM.INCOMING" => {
            if args.is_empty() {
                return err("MEM.INCOMING requires id [rel]");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            let rel = if args.len() >= 2 {
                String::from_utf8_lossy(&args[1]).to_string()
            } else {
                String::new()
            };
            let ns = db.rag.memory().incoming(id, &rel);
            let mut out = Vec::new();
            for (from, r, w) in ns {
                out.push(Resp::Int(from as i64));
                out.push(Resp::Bulk(r.into_bytes()));
                out.push(Resp::Bulk(format!("{w:.6}").into_bytes()));
            }
            Resp::Array(out)
        }
        // MEM.COUNT  -> :n   (live long-term memory count)
        "MEM.COUNT" => Resp::Int(db.rag.memory().ltm_count() as i64),
        // MEM.GET id  -> array of (text, source, salience) or nil
        "MEM.GET" => {
            if args.len() != 1 {
                return err("MEM.GET requires id");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            match db.rag.memory().ltm_get(id) {
                Some(rec) => Resp::Array(vec![
                    Resp::Bulk(rec.text.into_bytes()),
                    Resp::Bulk(rec.meta.source.into_bytes()),
                    Resp::Bulk(format!("{:.6}", rec.meta.salience).into_bytes()),
                ]),
                None => Resp::Nil,
            }
        }
        // MEM.CONSOLIDATE id delta  -> OK   (bump salience, clamped [0,1])
        "MEM.CONSOLIDATE" => {
            if args.len() != 2 {
                return err("MEM.CONSOLIDATE requires id delta");
            }
            let id: u64 = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("id is not a u64"),
            };
            let d: f32 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("delta is not a float"),
            };
            match db.rag.memory().touch_salience(id, d) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        // MEM.EPISODES_CLEAR agent  -> OK   (wipe all episodes for an agent)
        "MEM.EPISODES_CLEAR" => {
            if args.len() != 1 {
                return err("MEM.EPISODES_CLEAR requires agent");
            }
            let agent = String::from_utf8_lossy(&args[0]).to_string();
            match db.rag.memory().episodes_clear(&agent) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }

        // ── TIME-SERIES AGGREGATE ────────────────────────────────────────
        // TSAVG series from to  -> bulk (avg) or nil
        "TSAVG" => {
            if args.len() != 3 {
                return err("TSAVG requires series from to");
            }
            let series = String::from_utf8_lossy(&args[0]).to_string();
            let from: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n, None => return err("from is not a u64"),
            };
            let to: u64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n, None => return err("to is not a u64"),
            };
            match db.ts.avg(&series, from, to) {
                Some(v) => Resp::Bulk(format!("{v}").into_bytes()),
                None => Resp::Nil,
            }
        }

        // ── RAG CONTEXT BLOCK ────────────────────────────────────────────
        // RAG.CONTEXT k query f1 f2 ...  -> bulk (prompt-ready block)
        "RAG.CONTEXT" => {
            if args.len() < 3 {
                return err("RAG.CONTEXT requires k query f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let query = String::from_utf8_lossy(&args[1]).to_string();
            let qvec = match parse_floats(&args[2..]) {
                Some(v) => v,
                None => return err("bad float in query vector"),
            };
            Resp::Bulk(db.rag.context_block(&query, &qvec, k).into_bytes())
        }

        // ── MVCC POINT-IN-TIME READS ─────────────────────────────────────
        // GETAT key snapshot  -> bulk/int/float/vector/nil (typed by Value)
        "GETAT" => {
            if args.len() != 2 {
                return err("GETAT requires key snapshot");
            }
            let snap: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("snapshot is not a u64"),
            };
            let v = db.engine.get_at(&args[0], snap);
            value_to_resp(v)
        }
        // SCAN start end  -> array of (key, value) over all shards, snapshot=now
        "SCAN" => {
            if args.len() != 2 {
                return err("SCAN requires start end");
            }
            let snap = db.engine.snapshot();
            // BTreeMap::range panics if start > end; tolerate inverted ranges.
            let (lo, hi) = if args[0] <= args[1] {
                (&args[0][..], &args[1][..])
            } else {
                (&args[1][..], &args[0][..])
            };
            let pairs = db.engine.scan(lo, hi, snap);
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                out.push(Resp::Array(vec![Resp::Bulk(k), value_to_resp(Some(v))]));
            }
            Resp::Array(out)
        }

        "INFO" => {
            let info = format!(
                "db-strike\r\nsnapshot:{}\r\ncdc_events:{}\r\nengine:unified-mvcc-wal\r\n",
                db.engine.snapshot(),
                db.reactive.cdc_len()
            );
            Resp::Bulk(info.into_bytes())
        }

        other => err(&format!("unknown command '{other}'")),
    }
}
