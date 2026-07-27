use std::sync::{Arc, Barrier, Mutex};
use storage::{Engine, Value};

/// Probe: do concurrent writers to the SAME key preserve last-writer-wins by
/// commit timestamp? `Engine::put` returns the ts it committed under, so the
/// highest returned ts is by definition the newest version and must be what a
/// subsequent read observes.
///
/// Threads are pre-spawned and synchronised on a `Barrier` each round, because
/// the suspected race window (between reserving `ts` and enqueueing) is on the
/// order of 100ns — far narrower than the ~10-30us cost of `thread::spawn`, so
/// spawning per round never lets the writers actually collide.
#[test]
fn concurrent_same_key_last_writer_wins() {
    const THREADS: usize = 32;
    const ROUNDS: usize = 2000;

    let dir = std::env::temp_dir().join(format!("dbstrike_orderprobe_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("o.wal");
    let _ = std::fs::remove_file(&p);
    let e = Engine::open(&p).unwrap();

    // Per-round results, filled by every thread, drained by the checker.
    let results: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    // +1 for the checker thread.
    let start = Arc::new(Barrier::new(THREADS + 1));
    let done = Arc::new(Barrier::new(THREADS + 1));

    let mut hs = Vec::new();
    for t in 0..THREADS as u64 {
        let e = Arc::clone(&e);
        let results = Arc::clone(&results);
        let start = Arc::clone(&start);
        let done = Arc::clone(&done);
        hs.push(std::thread::spawn(move || {
            for round in 0..ROUNDS as u64 {
                start.wait();
                let v = round * 1000 + t;
                let ts = e.put(b"hot".to_vec(), Value::Int(v as i64)).unwrap();
                results.lock().unwrap().push((ts, v));
                done.wait();
            }
        }));
    }

    let mut violations = 0usize;
    let mut inversions_seen = 0usize;
    for round in 0..ROUNDS {
        start.wait();
        done.wait();
        let mut res = std::mem::take(&mut *results.lock().unwrap());
        res.sort_by_key(|r| r.0);
        let (max_ts, winner) = *res.last().unwrap();
        match e.get(b"hot") {
            Some(Value::Int(got)) if got as u64 == winner => {}
            other => {
                violations += 1;
                if violations <= 3 {
                    eprintln!(
                        "round {round}: highest ts {max_ts} wrote Int({winner}), but read {other:?}"
                    );
                }
            }
        }
        // Track how often timestamps were actually contiguous+interleaved,
        // i.e. whether the round even had the opportunity to invert.
        if res.windows(2).any(|w| w[1].0 != w[0].0 + 1) {
            inversions_seen += 1;
        }
    }
    for h in hs {
        h.join().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);

    eprintln!("VIOLATIONS: {violations} / {ROUNDS} rounds ({THREADS} writers/round)");
    eprintln!("rounds with non-contiguous ts blocks: {inversions_seen} / {ROUNDS}");
    assert_eq!(violations, 0, "an older timestamp overwrote a newer one — lost update");
}
