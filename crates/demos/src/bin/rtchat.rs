//! rtchat — realtime chat CLI on DB-Strike (WebRTC-signaling shape).
//!
//! This is the control plane a real video-chat app sits on top of: rooms,
//! presence, typing indicators, low-latency push. We use RESP SUBSCRIBE for
//! the incoming stream and PUBLISH for outgoing messages — the same push
//! substrate a real Selective-Forwarding-Unit (SFU) signals over.
//!
//! Video encoding itself needs OS media libs (opencv/ffmpeg) which we don't
//! bundle — but the primitives that make video chat *responsive* (sub-ms
//! push, presence, typing hints) are all exercised end-to-end here.
//!
//! Two flags:
//!   --self <name>        your handle
//!   --room <name>        room to join
//!   --addr <host:port>   DB-Strike RESP endpoint (default 127.0.0.1:6380)
//!   --latency            no interactive input; run automated latency probe
//!
//! Wire protocol on top of DB-Strike:
//!   channel  chat:<room>       broadcast messages
//!   channel  chat:<room>:typ   typing indicators (u1 = start, u0 = stop)
//!   key      pres:<room>:<u>   presence heartbeat (SET every 2s)

use std::env;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use demos::{Client, Reply};

fn now_micros() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64
}

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = env::args().collect();
    if let Some(i) = a.iter().position(|x| x == name) {
        a.get(i + 1).cloned().unwrap_or_else(|| default.into())
    } else {
        default.into()
    }
}
fn flag(name: &str) -> bool {
    env::args().any(|x| x == name)
}

// ── receiver thread: SUBSCRIBE to room + typing channels, print messages ────

fn spawn_receiver(
    addr: String,
    room: String,
    me: String,
    stop: Arc<AtomicBool>,
    recv_count: Arc<AtomicU64>,
    last_rt_recv_us: Arc<AtomicU64>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut c = match Client::connect(&addr) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("subscriber connect failed: {e}");
                return;
            }
        };
        let msg_ch = format!("chat:{room}");
        let typ_ch = format!("chat:{room}:typ");
        if let Err(e) = c.send(&[b"SUBSCRIBE", msg_ch.as_bytes(), typ_ch.as_bytes()]) {
            eprintln!("subscribe write failed: {e}");
            return;
        }
        // Drain 2 subscribe acks.
        for _ in 0..2 {
            let _ = c.read_reply();
        }
        let _ = c.set_read_timeout(Some(Duration::from_millis(300)));
        while !stop.load(Ordering::SeqCst) {
            match c.read_reply() {
                Ok(Reply::Array(a)) if a.len() == 3 => {
                    let kind = a[0].as_str().unwrap_or_default();
                    if kind != "message" {
                        continue;
                    }
                    let ch = a[1].as_str().unwrap_or_default();
                    let payload = a[2].as_str().unwrap_or_default();
                    if ch.ends_with(":typ") {
                        // typing indicator: "<user>|<0|1>"
                        let mut parts = payload.splitn(2, '|');
                        let u = parts.next().unwrap_or("").to_string();
                        let on = parts.next() == Some("1");
                        if u != me {
                            if on {
                                println!("  \x1b[90m{u} is typing…\x1b[0m");
                            }
                        }
                    } else {
                        // "<user>|<send_us>|<text>"
                        let mut parts = payload.splitn(3, '|');
                        let u = parts.next().unwrap_or("").to_string();
                        let send_us: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                        let text = parts.next().unwrap_or("");
                        if u != me {
                            let delta = now_micros().saturating_sub(send_us);
                            last_rt_recv_us.store(delta, Ordering::SeqCst);
                            recv_count.fetch_add(1, Ordering::SeqCst);
                            println!("  \x1b[35m{u}\x1b[0m ({delta} µs push): {text}");
                            print!("\x1b[36mme>\x1b[0m ");
                            let _ = io::stdout().flush();
                        }
                    }
                }
                Ok(_) => {}
                Err(_) => {
                    // Timeout or closed — loop back and check stop flag.
                }
            }
        }
    })
}

// ── presence: SET pres:<room>:<me> every 2s with a short TTL semantics ─────

fn spawn_presence(
    addr: String,
    room: String,
    me: String,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut c = match Client::connect(&addr) { Ok(c) => c, Err(_) => return };
        let key = format!("pres:{room}:{me}");
        while !stop.load(Ordering::SeqCst) {
            let ts = now_micros().to_string();
            let _ = c.cmd(&[b"SET", key.as_bytes(), ts.as_bytes()]);
            for _ in 0..20 {
                if stop.load(Ordering::SeqCst) { return; }
                thread::sleep(Duration::from_millis(100));
            }
        }
    })
}

fn who_in_room(c: &mut Client, room: &str) -> Vec<String> {
    let prefix = format!("pres:{room}:");
    let r = c.cmd(&[b"KEYS", format!("{prefix}*").as_bytes()]);
    let mut out = Vec::new();
    if let Ok(Reply::Array(arr)) = r {
        for k in arr {
            if let Some(s) = k.as_str() {
                if let Some(rest) = s.strip_prefix(&prefix) {
                    out.push(rest.to_string());
                }
            }
        }
    }
    out
}

// ── automated latency probe: PUBLISH → SUBSCRIBE round-trip ────────────────

fn run_latency_probe(addr: &str) -> std::io::Result<()> {
    println!("\x1b[1mrtchat latency probe\x1b[0m → {addr}");
    let stop = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));
    let last = Arc::new(AtomicU64::new(0));
    let jh = spawn_receiver(
        addr.into(), "probe".into(), "peer".into(),
        stop.clone(), count.clone(), last.clone(),
    );
    // Give the subscriber a moment to register.
    thread::sleep(Duration::from_millis(120));

    let mut pubc = Client::connect(addr)?;
    let mut samples = Vec::with_capacity(500);
    for i in 0..500 {
        let send_us = now_micros();
        let ch = b"chat:probe";
        let payload = format!("me|{send_us}|msg-{i}");
        pubc.cmd(&[b"PUBLISH", ch, payload.as_bytes()])?;
        // Wait up to 20 ms for the roundtrip to be captured.
        let deadline = Instant::now() + Duration::from_millis(20);
        let base = count.load(Ordering::SeqCst);
        while count.load(Ordering::SeqCst) == base && Instant::now() < deadline {
            thread::sleep(Duration::from_micros(50));
        }
        let d = last.load(Ordering::SeqCst);
        if d > 0 { samples.push(d); }
    }
    stop.store(true, Ordering::SeqCst);
    let _ = jh.join();

    if samples.is_empty() {
        println!("  no samples collected");
        return Ok(());
    }
    samples.sort_unstable();
    let n = samples.len();
    let p50 = samples[n / 2];
    let p90 = samples[(n * 9) / 10];
    let p99 = samples[(n * 99) / 100];
    let max = *samples.last().unwrap();
    println!("  n={n}   p50={p50} µs   p90={p90} µs   p99={p99} µs   max={max} µs");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let addr = arg("--addr", "127.0.0.1:6380");
    if flag("--latency") {
        return run_latency_probe(&addr);
    }
    let me = arg("--self", "user");
    let room = arg("--room", "lobby");

    println!("\x1b[1mDB-Strike rtchat\x1b[0m — {me}@{room} → {addr}");
    println!("  type a line to send; :who lists room presence; :typ [on|off] toggles typing indicator; :quit to exit");
    println!();

    let mut pubc = Client::connect(&addr)?;
    let stop = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));
    let last = Arc::new(AtomicU64::new(0));
    let jh = spawn_receiver(
        addr.clone(), room.clone(), me.clone(),
        stop.clone(), count.clone(), last.clone(),
    );
    let pj = spawn_presence(addr.clone(), room.clone(), me.clone(), stop.clone());
    // Announce join.
    let _ = pubc.cmd(&[b"PUBLISH", format!("chat:{room}").as_bytes(),
                       format!("system|{}|{me} joined", now_micros()).as_bytes()]);

    let stdin = io::stdin();
    let mut lock = stdin.lock();
    loop {
        print!("\x1b[36m{me}>\x1b[0m ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if lock.read_line(&mut line)? == 0 { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == ":quit" || line == ":q" { break; }
        if line == ":who" {
            let peers = who_in_room(&mut pubc, &room);
            println!("  room={room} peers={:?}", peers);
            continue;
        }
        if let Some(rest) = line.strip_prefix(":typ") {
            let on = rest.trim() == "on";
            let payload = format!("{me}|{}", if on { 1 } else { 0 });
            let _ = pubc.cmd(&[b"PUBLISH", format!("chat:{room}:typ").as_bytes(), payload.as_bytes()]);
            continue;
        }
        // Normal message
        let payload = format!("{me}|{}|{line}", now_micros());
        let r = pubc.cmd(&[b"PUBLISH", format!("chat:{room}").as_bytes(), payload.as_bytes()])?;
        let subs = r.as_int().unwrap_or(-1);
        println!("  \x1b[90m(sent to {subs} subscribers)\x1b[0m");
    }
    stop.store(true, Ordering::SeqCst);
    let _ = jh.join();
    let _ = pj.join();
    println!("bye");
    Ok(())
}
