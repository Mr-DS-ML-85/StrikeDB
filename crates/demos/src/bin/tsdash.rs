//! tsdash — live time-series dashboard against DB-Strike.
//!
//! Simulates a system metrics collector: three synthetic series (cpu, mem,
//! rps) get one sample per tick, written via TSADD. Every 500 ms the
//! dashboard reads the last-60-sample window for each series with TSRANGE
//! and renders:
//!
//!   * current value + delta
//!   * min / max / avg over window
//!   * 60-char sparkline
//!
//! This is the "time-series dashboard" from the ask, exercising TSADD +
//! TSRANGE at realistic dashboard cadence (2 Hz refresh, ~1000 writes/sec).
//!
//! Usage:
//!   tsdash [addr=127.0.0.1:6380] [duration_secs=30]

use std::env;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use demos::{pctl, rss_mb, Client, Reply};

const SERIES: &[&str] = &["cpu", "mem", "rps"];

// Synthetic signal generator — sine + noise so it looks alive.
fn synth(name: &str, t: f64) -> i64 {
    let (amp, freq, base): (f64, f64, f64) = match name {
        "cpu" => (25.0, 0.15, 55.0),
        "mem" => (10.0, 0.05, 70.0),
        "rps" => (400.0, 0.3, 800.0),
        _ => (10.0, 0.1, 50.0),
    };
    let x = base + amp * (t * freq).sin();
    let noise = ((t * 3.7).cos() * amp * 0.15) + ((t * 11.1).sin() * amp * 0.05);
    (x + noise).round() as i64
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

fn sparkline(values: &[i64]) -> String {
    if values.is_empty() {
        return "(no data)".into();
    }
    let ticks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    let span = (max - min).max(1);
    values
        .iter()
        .map(|v| {
            let bucket =
                ((v - min) as f64 / span as f64 * (ticks.len() as f64 - 1.0)) as usize;
            ticks[bucket.min(ticks.len() - 1)]
        })
        .collect()
}

fn render(rows: &[(String, Vec<i64>)]) {
    // Clear screen + home cursor.
    print!("\x1b[2J\x1b[H");
    println!(
        "\x1b[1mDB-Strike tsdash\x1b[0m — live @ {}",
        now_ms()
    );
    println!("{:-<80}", "");
    for (name, series) in rows {
        let cur = *series.last().unwrap_or(&0);
        let prev = *series
            .get(series.len().saturating_sub(2))
            .unwrap_or(&cur);
        let delta = cur - prev;
        let min = series.iter().min().copied().unwrap_or(0);
        let max = series.iter().max().copied().unwrap_or(0);
        let sum: i64 = series.iter().sum();
        let avg = if series.is_empty() { 0 } else { sum / series.len() as i64 };
        let arrow = if delta > 0 { "\x1b[32m▲\x1b[0m" }
                    else if delta < 0 { "\x1b[31m▼\x1b[0m" }
                    else { " " };
        println!(
            "  {name:<4}  cur={cur:>6}  {arrow}{delta:>+4}   min={min:>5}  avg={avg:>5}  max={max:>5}    {}",
            sparkline(series)
        );
    }
    println!();
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn read_window(client: &mut Client, name: &str, from: u64, to: u64) -> std::io::Result<Vec<i64>> {
    let r = client.cmd(&[b"TSRANGE", name.as_bytes(),
                         from.to_string().as_bytes(),
                         to.to_string().as_bytes()])?;
    let arr = match r {
        Reply::Array(a) => a,
        _ => return Ok(Vec::new()),
    };
    // Server flattens as (ts, val, ts, val, ...) — take every second one.
    Ok(arr.chunks(2).filter_map(|c| c.get(1).and_then(|v| v.as_int())).collect())
}

fn run_bench(addr: &str, n_points: usize, n_queries: usize) -> std::io::Result<()> {
    let rss0 = rss_mb().unwrap_or(0);
    println!("\x1b[1mtsdash --bench\x1b[0m → {addr}   n_points={n_points}  n_queries={n_queries}");
    println!("  starting RSS = {rss0} MB");
    let mut c = Client::connect(addr)?;

    // Ingest phase — one series, N points
    let mut ins = Vec::with_capacity(n_points);
    let t0 = Instant::now();
    for i in 0..n_points {
        let start = Instant::now();
        c.cmd(&[
            b"TSADD",
            b"bench-series",
            i.to_string().as_bytes(),
            (i as i64 * 3).to_string().as_bytes(),
        ])?;
        ins.push(start.elapsed().as_micros() as u64);
    }
    let dt = t0.elapsed().as_secs_f64();
    let p50 = pctl(&mut ins.clone(), 50.0);
    let p99 = pctl(&mut ins.clone(), 99.0);
    let rate = n_points as f64 / dt;
    println!("  TSADD   × {n_points:>6}  p50={p50:>4}µs  p99={p99:>5}µs  → {rate:>7.0} ops/s");

    // Query phase — TSRANGE over 1000-sample windows
    let mut qs = Vec::with_capacity(n_queries);
    let t0 = Instant::now();
    for i in 0..n_queries {
        let from = (i * 100) as u64 % (n_points as u64 - 1000).max(1);
        let to = from + 1000;
        let start = Instant::now();
        c.cmd(&[
            b"TSRANGE",
            b"bench-series",
            from.to_string().as_bytes(),
            to.to_string().as_bytes(),
        ])?;
        qs.push(start.elapsed().as_micros() as u64);
    }
    let dt = t0.elapsed().as_secs_f64();
    let p50 = pctl(&mut qs.clone(), 50.0);
    let p99 = pctl(&mut qs.clone(), 99.0);
    let rate = n_queries as f64 / dt;
    println!("  TSRANGE × {n_queries:>6}  p50={p50:>4}µs  p99={p99:>5}µs  → {rate:>7.0} ops/s");

    let rss1 = rss_mb().unwrap_or(0);
    println!("  ending RSS = {rss1} MB  (Δ = {} MB)", rss1 as i64 - rss0 as i64);
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        let addr = args.iter().position(|a| a == "--addr")
            .and_then(|i| args.get(i + 1)).cloned()
            .unwrap_or_else(|| "127.0.0.1:6380".to_string());
        let n_p: usize = args.iter().position(|a| a == "--n")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
            .unwrap_or(20_000);
        return run_bench(&addr, n_p, 500);
    }
    let addr = env::args().nth(1).unwrap_or_else(|| "127.0.0.1:6380".to_string());
    let duration_s: u64 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut writer = Client::connect(&addr)?;
    let mut reader = Client::connect(&addr)?;
    println!("tsdash → {addr}, running {duration_s}s ...");
    thread::sleep(Duration::from_millis(300));

    let start = Instant::now();
    let mut sample_seq: u64 = 0;
    let mut last_render = Instant::now() - Duration::from_secs(1);
    while start.elapsed() < Duration::from_secs(duration_s) {
        // Write phase — one sample per series per tick.
        let t = start.elapsed().as_secs_f64();
        let ts = now_ms();
        for name in SERIES {
            let val = synth(name, t);
            writer.cmd(&[b"TSADD", name.as_bytes(),
                         ts.to_string().as_bytes(),
                         val.to_string().as_bytes()])?;
        }
        sample_seq += 1;

        // Render every 500 ms.
        if last_render.elapsed() >= Duration::from_millis(500) {
            let mut rows = Vec::new();
            let from = ts.saturating_sub(30_000);  // 30-second window
            for name in SERIES {
                rows.push((name.to_string(), read_window(&mut reader, name, from, ts)?));
            }
            render(&rows);
            last_render = Instant::now();
        }
        thread::sleep(Duration::from_millis(30));
    }
    println!();
    println!("✓ wrote {sample_seq} samples per series ({} total) over {duration_s}s",
             sample_seq * SERIES.len() as u64);
    Ok(())
}
