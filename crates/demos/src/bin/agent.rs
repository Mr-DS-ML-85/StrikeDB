//! agent-cli — interactive AI agent demo backed by DB-Strike.
//!
//! Uses every memory primitive we ship:
//!   * WM   — hot session context (last N turns)
//!   * LTM  — durable facts with lineage + salience
//!   * Graph — typed edges between remembered entities
//!   * Bi-temporal — facts with valid_from/valid_to windows
//!   * RAG  — hybrid dense+sparse recall for relevant context
//!
//! Commands you can type at the prompt:
//!   :remember <text>         → store as LTM (with a tiny hash-based embedding)
//!   :recall <query>          → hybrid recall (top-5)
//!   :link <a> <rel> <b>      → typed edge between two LTM ids
//!   :neigh <a>               → list a's outgoing edges
//!   :fact <text> at <t> for <name>  → temporal fact (valid_from=t)
//!   :asof <t> <query>        → recall as-of world-time t
//!   :history                 → show recent working-memory turns
//!   :quit                    → exit
//!
//! Anything else is treated as a "user turn" — the agent auto-stores it in
//! WM (short-term) and does a background recall to show 3 relevant memories.
//!
//! No embedding model — we hash tokens into a 32-d bag-of-features vector,
//! good enough for the demo to show ranking behavior. Real usage would plug
//! in an actual embedder.

use std::env;
use std::io::{self, BufRead, Write};

use demos::{Client, Reply};

fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for tok in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if tok.len() < 3 {
            continue;
        }
        // FNV-1a → bucket
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in tok.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        v[(h as usize) % dim] += 1.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn fvec_args(v: &[f32]) -> Vec<Vec<u8>> {
    v.iter().map(|f| format!("{f:.6}").into_bytes()).collect()
}

fn as_bytes_all(argv: &[Vec<u8>]) -> Vec<&[u8]> {
    argv.iter().map(|v| v.as_slice()).collect()
}

fn banner(agent: &str, addr: &str) {
    println!("\x1b[1mDB-Strike agent-cli\x1b[0m ({agent}) → {addr}");
    println!("  type your message, or use :remember :recall :link :neigh :fact :asof :history :quit");
    println!();
}

fn main() -> std::io::Result<()> {
    let addr = env::args().nth(1).unwrap_or_else(|| "127.0.0.1:6380".to_string());
    let agent = env::args().nth(2).unwrap_or_else(|| "user".to_string());
    let mut c = Client::connect(&addr)?;
    banner(&agent, &addr);

    let stdin = io::stdin();
    let mut turn = 0u64;
    let mut lock = stdin.lock();
    loop {
        print!("\x1b[36m{agent}>\x1b[0m ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if lock.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == ":quit" || line == ":q" {
            break;
        }

        // ── commands ────────────────────────────────────────────────────
        if let Some(rest) = line.strip_prefix(":remember ") {
            let v = embed(rest, 32);
            let sal = "0.7";
            let mut args: Vec<Vec<u8>> = vec![
                b"MEM.REMEMBER".to_vec(),
                rest.as_bytes().to_vec(),
                agent.as_bytes().to_vec(),
                sal.as_bytes().to_vec(),
            ];
            for f in fvec_args(&v) {
                args.push(f);
            }
            let r = c.cmd(&as_bytes_all(&args))?;
            println!("  stored LTM id={}", r.as_int().unwrap_or(-1));
            continue;
        }
        if let Some(rest) = line.strip_prefix(":recall ") {
            let v = embed(rest, 32);
            let mut args: Vec<Vec<u8>> = vec![
                b"MEM.RECALL".to_vec(),
                b"5".to_vec(),
                rest.as_bytes().to_vec(),
            ];
            for f in fvec_args(&v) {
                args.push(f);
            }
            print_hits("recall", &c.cmd(&as_bytes_all(&args))?);
            continue;
        }
        if let Some(rest) = line.strip_prefix(":link ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() != 3 {
                println!("  usage: :link <from-id> <rel> <to-id>");
                continue;
            }
            let r = c.cmd(&[b"MEM.LINK", parts[0].as_bytes(), parts[2].as_bytes(),
                            parts[1].as_bytes(), b"1.0"])?;
            println!("  {}", r.as_str().unwrap_or_default());
            continue;
        }
        if let Some(rest) = line.strip_prefix(":neigh ") {
            let id = rest.trim();
            let r = c.cmd(&[b"MEM.NEIGH", id.as_bytes()])?;
            print_neighbors(&r);
            continue;
        }
        if let Some(rest) = line.strip_prefix(":fact ") {
            // :fact <text> at <t> for <name>   — very tiny parser
            if let (Some(a), Some(b_idx)) = (rest.find(" at "), rest.find(" for ")) {
                let text = &rest[..a];
                let t = &rest[a + 4..b_idx];
                let _name = &rest[b_idx + 5..];
                let v = embed(text, 32);
                let mut args: Vec<Vec<u8>> = vec![
                    b"MEM.REMEMBER.T".to_vec(),
                    text.as_bytes().to_vec(),
                    b"user-fact".to_vec(),
                    b"0.9".to_vec(),
                    t.as_bytes().to_vec(),
                    b"0".to_vec(),
                ];
                for f in fvec_args(&v) {
                    args.push(f);
                }
                let r = c.cmd(&as_bytes_all(&args))?;
                println!("  temporal fact id={} valid_from={t}", r.as_int().unwrap_or(-1));
            } else {
                println!("  usage: :fact <text> at <world-ts> for <who>");
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix(":asof ") {
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() != 2 {
                println!("  usage: :asof <world-ts> <query>");
                continue;
            }
            let t = parts[0];
            let q = parts[1];
            let v = embed(q, 32);
            let mut args: Vec<Vec<u8>> = vec![
                b"MEM.RECALL.AS_OF".to_vec(),
                b"5".to_vec(),
                q.as_bytes().to_vec(),
                t.as_bytes().to_vec(),
            ];
            for f in fvec_args(&v) {
                args.push(f);
            }
            print_hits(&format!("as-of {t}"), &c.cmd(&as_bytes_all(&args))?);
            continue;
        }
        if line == ":history" {
            for i in turn.saturating_sub(10)..turn {
                let k = format!("wm-agent:{agent}:turn:{i}");
                if let Reply::Bulk(Some(b)) = c.cmd(&[b"GET", k.as_bytes()])? {
                    println!("  [{i}] {}", String::from_utf8_lossy(&b));
                }
            }
            continue;
        }

        // ── plain user turn ────────────────────────────────────────────
        // Store in working-memory-shaped KV (TTL is enforced by MEM.WM in the
        // full memory API; here we just use SET with agent:turn key).
        let wmk = format!("wm-agent:{agent}:turn:{turn}");
        c.cmd(&[b"SET", wmk.as_bytes(), line.as_bytes()])?;
        turn += 1;
        // background recall for context (top-3)
        let v = embed(&line, 32);
        let mut args: Vec<Vec<u8>> = vec![
            b"MEM.RECALL".to_vec(),
            b"3".to_vec(),
            line.as_bytes().to_vec(),
        ];
        for f in fvec_args(&v) {
            args.push(f);
        }
        let r = c.cmd(&as_bytes_all(&args))?;
        let hits = r.as_array().map(|a| a.len()).unwrap_or(0);
        if hits == 0 {
            println!("  (no relevant memories)");
        } else {
            print_hits("context", &r);
        }
    }
    println!("bye");
    Ok(())
}

fn print_hits(label: &str, r: &Reply) {
    let arr = match r.as_array() {
        Some(a) => a,
        None => {
            println!("  ({label}: no results)");
            return;
        }
    };
    // Each hit is 4 flat entries: id, score, source, text
    if arr.len() % 4 != 0 {
        println!("  ({label}: unexpected reply shape)");
        return;
    }
    println!("  \x1b[90m{label}:\x1b[0m");
    for chunk in arr.chunks(4) {
        let id = chunk[0].as_int().unwrap_or(-1);
        let score = chunk[1].as_str().unwrap_or_default();
        let src = chunk[2].as_str().unwrap_or_default();
        let txt = chunk[3].as_str().unwrap_or_default();
        println!("    #{id:<4} [{src}] score={score}  {txt}");
    }
}

fn print_neighbors(r: &Reply) {
    let arr = match r.as_array() {
        Some(a) => a,
        None => {
            println!("  (no neighbors)");
            return;
        }
    };
    if arr.is_empty() {
        println!("  (no outgoing edges)");
        return;
    }
    for chunk in arr.chunks(3) {
        let to = chunk[0].as_int().unwrap_or(-1);
        let rel = chunk[1].as_str().unwrap_or_default();
        let w = chunk[2].as_str().unwrap_or_default();
        println!("  --{rel}({w})--> #{to}");
    }
}
