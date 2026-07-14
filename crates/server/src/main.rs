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

use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use compute::{counter_reducer, ReducerResult, ReducerRuntime};
use mitm::CacheDebugger;
use protocol::{read_command, write_resp, Resp};
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
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle(stream: TcpStream, db: Arc<Db>) -> std::io::Result<()> {
    let read_half = stream.try_clone()?;
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(stream);

    loop {
        let cmd = match read_command(&mut reader)? {
            Some(c) if c.is_empty() => continue,
            Some(c) => c,
            None => break, // client closed
        };
        let name = String::from_utf8_lossy(&cmd[0]).to_uppercase();
        let args = &cmd[1..];

        // ── SUBSCRIBE hijacks the connection into push-stream mode ─────────
        // (Redis-compatible pattern: after SUBSCRIBE, the connection is
        // dedicated to receiving pushed messages until closed.)
        if name == "SUBSCRIBE" || name == "PSUBSCRIBE" {
            if args.is_empty() {
                write_resp(&mut writer, &err("SUBSCRIBE requires at least one channel"))?;
                continue;
            }
            // Every subscription channel is stored under the `chan:<name>`
            // key-prefix. Multiple channels merge into ONE receiver so we can
            // block on a single mpsc from this thread.
            let prefixes: Vec<Vec<u8>> = args
                .iter()
                .map(|a| {
                    let mut p = b"chan:".to_vec();
                    p.extend_from_slice(a);
                    p
                })
                .collect();
            let rx = db.reactive.subscribe_prefixes(&prefixes);
            // ack each subscription in Redis format: *3\r\n +subscribe channel count
            for (i, ch) in args.iter().enumerate() {
                let ack = Resp::Array(vec![
                    Resp::Bulk(b"subscribe".to_vec()),
                    Resp::Bulk(ch.clone()),
                    Resp::Int((i as i64) + 1),
                ]);
                write_resp(&mut writer, &ack)?;
            }
            // Stream events until the client disconnects.
            for ev in rx.iter() {
                // Extract channel name from key (`chan:<name>`).
                let channel = ev
                    .key
                    .strip_prefix(b"chan:")
                    .unwrap_or(&ev.key)
                    .to_vec();
                let payload: Vec<u8> = match &ev.value {
                    storage::Value::Bytes(b) => b.clone(),
                    storage::Value::Int(i) => i.to_string().into_bytes(),
                    storage::Value::Tombstone => b"__deleted__".to_vec(),
                    other => format!("{other:?}").into_bytes(),
                };
                let msg = Resp::Array(vec![
                    Resp::Bulk(b"message".to_vec()),
                    Resp::Bulk(channel),
                    Resp::Bulk(payload),
                ]);
                // On write error the client is gone — drop the subscription.
                if write_resp(&mut writer, &msg).is_err() {
                    break;
                }
            }
            break; // connection is done after subscribe stream ends
        }

        let resp = dispatch(&db, &name, args);
        let quit = name == "QUIT";
        write_resp(&mut writer, &resp)?;
        if quit {
            break;
        }
    }
    Ok(())
}

fn err(msg: &str) -> Resp {
    Resp::Error(format!("ERR {msg}"))
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

        "TSADD" => {
            if args.len() != 3 {
                return err("TSADD requires series ts val");
            }
            let series = String::from_utf8_lossy(&args[0]).to_string();
            let t: u64 = match std::str::from_utf8(&args[1]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("ts is not a u64"),
            };
            let v: i64 = match std::str::from_utf8(&args[2]).ok().and_then(|s| s.parse().ok()) {
                Some(n) => n,
                None => return err("val is not an i64"),
            };
            match db.ts.append(&series, t, v) {
                Ok(_) => Resp::Simple("OK".into()),
                Err(e) => err(&e.to_string()),
            }
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
                out.push(Resp::Int(v));
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
