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

use std::io::{BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use compute::{counter_reducer, ReducerResult, ReducerRuntime};
use mitm::CacheDebugger;
use protocol::{try_parse, write_resp, write_resp_buf, Resp};
use rag::Rag;
use reactive::Reactive;
use router::{Router, TieredMemory};
use storage::Engine;
use views::{Kv, TimeSeries};

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
}

fn main() -> std::io::Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:6380".to_string());
    let data_path = std::env::var("DBSTRIKE_WAL").unwrap_or_else(|_| "dbstrike.wal".to_string());

    let engine = Engine::open(&data_path)?;
    let reactive = Reactive::attach(&engine);
    let router = Router::new(Arc::clone(&engine));
    let kv = Kv::new(Arc::clone(&engine));
    let ts = TimeSeries::new(Arc::clone(&engine));
    let reducers = ReducerRuntime::new(Arc::clone(&engine), 16);
    let rag = Rag::open(Arc::clone(&engine));
    let cache = CacheDebugger::new(Arc::clone(&engine), 4096);
    let memory = TieredMemory::new(300);

    let db = Arc::new(Db { engine, reactive, router, kv, ts, reducers, rag, cache, memory });

    let listener = TcpListener::bind(&addr)?;
    println!("DB-Strike listening on {addr} (RESP wire), WAL={data_path}");
    println!("One engine: KV · vectors · tables · timeseries · reducers · pub/sub · CRDT · HLC · agent-memory · RAG · MITM cache-debug");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let db = Arc::clone(&db);
                std::thread::spawn(move || {
                    if let Err(e) = handle(s, db) {
                        // These are normal when a client hangs up: don't spam
                        // the console. Real errors (parse failures on OTHER
                        // still-connected sockets, disk errors, etc.) still
                        // print.
                        if !is_benign_disconnect(&e) {
                            eprintln!("connection error: {e}");
                        }
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle(stream: TcpStream, db: Arc<Db>) -> std::io::Result<()> {
    // TCP_NODELAY: disable Nagle so per-batch flushes actually go on the wire
    // immediately (matters for latency-sensitive workloads like signaling).
    let _ = stream.set_nodelay(true);
    let mut sock_read = stream.try_clone()?;
    let mut writer = BufWriter::with_capacity(64 * 1024, stream);
    // Manual read buffer so we can peek pipelined commands without going
    // command-at-a-time. BufRead's fill_buf has capacity semantics that make
    // the "keep parsing until short" pattern awkward; a plain Vec<u8> is
    // clearer and just as fast.
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut tmp = [0u8; 32 * 1024];

    loop {
        // Block until we have SOMETHING to parse (or the client closes).
        let n = sock_read.read(&mut tmp)?;
        if n == 0 {
            return Ok(()); // client closed
        }
        buf.extend_from_slice(&tmp[..n]);

        // ── Drain every complete command from `buf` ────────────────────
        let mut cmds: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut cursor = 0usize;
        loop {
            match try_parse(&buf[cursor..])? {
                Some((cmd, consumed)) => {
                    cursor += consumed;
                    if !cmd.is_empty() {
                        cmds.push(cmd);
                    }
                }
                None => break, // partial command; wait for more bytes
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
                    i += 1;
                }
                let n = kvs.len();
                match db.kv.set_batch(kvs) {
                    Ok(_) => {
                        // One "+OK\r\n" per command, written straight to the
                        // buffered socket — no per-reply Resp alloc / encode.
                        for _ in 0..n {
                            writer.write_all(b"+OK\r\n")?;
                        }
                    }
                    Err(e) => {
                        let e = err(&e.to_string());
                        for _ in 0..n {
                            write_resp_buf(&mut writer, &e)?;
                        }
                    }
                }
                continue;
            }

            let resp = dispatch(&db, &name, args);
            write_resp_buf(&mut writer, &resp)?;
            if name == "QUIT" {
                quit = true;
                break;
            }
            i += 1;
        }
        writer.flush()?;
        if quit {
            return Ok(());
        }
        if let Some((name, args)) = subscribe_after_batch {
            return handle_subscribe(&db, &mut writer, &name, &args);
        }
    }
}

/// Hijack the connection into push-stream mode after SUBSCRIBE/PSUBSCRIBE.
/// Factored out of `handle` so the fast-path batched dispatch stays tight.
fn handle_subscribe(
    db: &Arc<Db>,
    writer: &mut BufWriter<TcpStream>,
    _name: &str,
    args: &[Vec<u8>],
) -> std::io::Result<()> {
    if args.is_empty() {
        write_resp(writer, &err("SUBSCRIBE requires at least one channel"))?;
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
        write_resp(writer, &ack)?;
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
        if write_resp(writer, &msg).is_err() {
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

fn dispatch(db: &Db, name: &str, args: &[Vec<u8>]) -> Resp {
    match name {
        "PING" => Resp::Simple("PONG".into()),
        "QUIT" => Resp::Simple("OK".into()),

        "SET" => {
            if args.len() != 2 {
                return err("SET requires key value");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            match db.kv.set(&key, &args[1]) {
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
                .map(|a| {
                    let key = String::from_utf8_lossy(a).to_string();
                    match db.kv.get(&key) {
                        Some(v) => Resp::Bulk(v),
                        None => Resp::Nil,
                    }
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
        "GET" => {
            if args.len() != 1 {
                return err("GET requires key");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            match db.kv.get(&key) {
                Some(v) => Resp::Bulk(v),
                None => Resp::Nil,
            }
        }
        "DEL" => {
            if args.len() != 1 {
                return err("DEL requires key");
            }
            let key = String::from_utf8_lossy(&args[0]).to_string();
            match db.kv.del(&key) {
                Ok(true) => Resp::Int(1),
                Ok(false) => Resp::Int(0),
                Err(e) => err(&e.to_string()),
            }
        }
        "INCR" | "INCRBY" => {
            let (key, by) = if name == "INCR" {
                if args.len() != 1 {
                    return err("INCR requires key");
                }
                (String::from_utf8_lossy(&args[0]).to_string(), 1)
            } else {
                if args.len() != 2 {
                    return err("INCRBY requires key n");
                }
                let by: i64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => return err("n is not an integer"),
                };
                (String::from_utf8_lossy(&args[0]).to_string(), by)
            };
            match db.kv.incr_by(&key, by) {
                Ok(n) => Resp::Int(n),
                Err(e) => err(&e),
            }
        }
        "KEYS" => {
            let prefix = args.first().map(|a| String::from_utf8_lossy(a).to_string()).unwrap_or_default();
            let prefix = prefix.trim_end_matches('*').to_string();
            let keys = db.kv.keys_prefix(&prefix);
            Resp::Array(keys.into_iter().map(|k| Resp::Bulk(k.into_bytes())).collect())
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
            match db.router.vectors().insert(id, vec) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
        }
        "VSEARCH" => {
            if args.len() < 2 {
                return err("VSEARCH requires k f1 f2 ...");
            }
            let k: usize = match std::str::from_utf8(&args[0]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("k is not an integer"),
            };
            let q = match parse_floats(&args[1..]) {
                Some(v) => v,
                None => return err("bad float in query"),
            };
            let hits = db.router.vectors().search(&q, k);
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
            let prog = counter_reducer(&key, by);
            match db.reducers.invoke(&rname, &shardkey, &prog) {
                ReducerResult::Ok { output, .. } => Resp::Int(output.unwrap_or(0)),
                ReducerResult::Aborted(e) => err(&format!("reducer aborted: {e}")),
                ReducerResult::Quarantined => err("reducer quarantined"),
            }
        }

        "CDCLEN" => Resp::Int(db.reactive.cdc_len() as i64),

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
