//! Compute layer — sandboxed reducers with the three SpacetimeDB-stall fixes:
//!   1. Fuel metering: every instruction costs fuel; exceeding budget aborts and
//!      rolls back cleanly (no trusting the author to be well-behaved).
//!   2. Shard bulkhead: a reducer is dispatched against the key-range it declares,
//!      so a stall blocks only its partition, not the whole engine.
//!   3. Circuit breaker: a reducer erroring above a threshold gets auto-quarantined.
//!
//! The reducer body is a tiny stack VM (pure Rust, zero deps) standing in for a
//! wasm module — same guarantee, minimal surface. Programs read/write the
//! substrate transactionally through the host `Ctx`.

pub mod vm;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use storage::Engine;
use vm::{Instr, Program, VmError};

/// Per-shard bulkhead: serializes reducers touching the same shard, isolates stalls.
struct Shard {
    lock: Mutex<()>,
}

pub struct ReducerRuntime {
    engine: Arc<Engine>,
    shards: Vec<Shard>,
    breakers: Mutex<HashMap<String, Breaker>>,
    default_fuel: u64,
    error_threshold: u32,
}

struct Breaker {
    consecutive_errors: AtomicU32,
    quarantined: AtomicU64, // 0 = open, 1 = quarantined
}

/// Outcome of a reducer invocation.
#[derive(Debug)]
pub enum ReducerResult {
    Ok { fuel_used: u64, output: Option<i64> },
    Aborted(VmError),
    Quarantined,
}

fn shard_of(key: &[u8], n: usize) -> usize {
    // FNV-1a hash of the key, mod shard count — deterministic bulkhead routing.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % n as u64) as usize
}

impl ReducerRuntime {
    pub fn new(engine: Arc<Engine>, shard_count: usize) -> Self {
        let shards = (0..shard_count.max(1)).map(|_| Shard { lock: Mutex::new(()) }).collect();
        Self {
            engine,
            shards,
            breakers: Mutex::new(HashMap::new()),
            default_fuel: 100_000,
            error_threshold: 3,
        }
    }

    /// Invoke a named reducer program bound to a shard key.
    /// Runs inside the shard bulkhead, metered, with circuit-breaker protection.
    pub fn invoke(&self, name: &str, shard_key: &[u8], prog: &Program) -> ReducerResult {
        // Circuit breaker check
        {
            let mut b = self.breakers.lock().unwrap();
            let br = b.entry(name.to_string()).or_insert_with(|| Breaker {
                consecutive_errors: AtomicU32::new(0),
                quarantined: AtomicU64::new(0),
            });
            if br.quarantined.load(Ordering::SeqCst) == 1 {
                return ReducerResult::Quarantined;
            }
        }

        // Bulkhead: lock only this reducer's shard.
        let sh = shard_of(shard_key, self.shards.len());
        let _guard = self.shards[sh].lock.lock().unwrap();

        // Execute metered against a fresh txn.
        let mut txn = self.engine.begin();
        let mut vm = vm::Vm::new(self.default_fuel);
        let result = vm.run(prog, &mut txn);

        match result {
            Ok(output) => {
                // commit; treat conflict as an error for the breaker
                match txn.commit() {
                    Ok(_) => {
                        self.record_success(name);
                        ReducerResult::Ok { fuel_used: vm.fuel_used(), output }
                    }
                    Err(_) => {
                        self.record_error(name);
                        ReducerResult::Aborted(VmError::Trap("commit conflict".into()))
                    }
                }
            }
            Err(e) => {
                // txn dropped without commit == clean rollback
                self.record_error(name);
                ReducerResult::Aborted(e)
            }
        }
    }

    fn record_success(&self, name: &str) {
        if let Some(br) = self.breakers.lock().unwrap().get(name) {
            br.consecutive_errors.store(0, Ordering::SeqCst);
        }
    }

    fn record_error(&self, name: &str) {
        if let Some(br) = self.breakers.lock().unwrap().get(name) {
            let n = br.consecutive_errors.fetch_add(1, Ordering::SeqCst) + 1;
            if n >= self.error_threshold {
                br.quarantined.store(1, Ordering::SeqCst);
            }
        }
    }

    /// Manually re-enable a quarantined reducer.
    pub fn reenable(&self, name: &str) {
        if let Some(br) = self.breakers.lock().unwrap().get(name) {
            br.quarantined.store(0, Ordering::SeqCst);
            br.consecutive_errors.store(0, Ordering::SeqCst);
        }
    }

    pub fn is_quarantined(&self, name: &str) -> bool {
        self.breakers
            .lock()
            .unwrap()
            .get(name)
            .map(|b| b.quarantined.load(Ordering::SeqCst) == 1)
            .unwrap_or(false)
    }
}

/// Convenience: build the classic "increment a counter key" reducer program.
pub fn counter_reducer(key: &[u8], by: i64) -> Program {
    Program {
        instrs: vec![
            Instr::LoadInt(key.to_vec()), // push current int value of key (0 if absent)
            Instr::PushInt(by),
            Instr::Add,
            Instr::Dup,                    // keep a copy of the new value for the return
            Instr::StoreInt(key.to_vec()), // write back (consumes one copy)
            Instr::Return,                 // return the new value
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use storage::Value;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_comp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Engine::open(dir.join(format!("c_{n}.wal"))).unwrap()
    }

    #[test]
    fn reducer_increments_and_commits() {
        let e = eng();
        let rt = ReducerRuntime::new(Arc::clone(&e), 8);
        let prog = counter_reducer(b"kv:hits", 5);
        let r = rt.invoke("hit", b"kv:hits", &prog);
        assert!(matches!(r, ReducerResult::Ok { output: Some(5), .. }));
        assert_eq!(e.get(b"kv:hits"), Some(Value::Int(5)));
        let r2 = rt.invoke("hit", b"kv:hits", &prog);
        assert!(matches!(r2, ReducerResult::Ok { output: Some(10), .. }));
        assert_eq!(e.get(b"kv:hits"), Some(Value::Int(10)));
    }

    #[test]
    fn fuel_exhaustion_aborts_and_rolls_back() {
        let e = eng();
        let rt = ReducerRuntime::new(Arc::clone(&e), 4);
        // infinite loop program: Jump back to 0 forever
        let prog = Program {
            instrs: vec![Instr::PushInt(1), Instr::Pop, Instr::Jump(0)],
        };
        let r = rt.invoke("bad", b"shardA", &prog);
        assert!(matches!(r, ReducerResult::Aborted(VmError::OutOfFuel)));
    }

    #[test]
    fn circuit_breaker_quarantines_after_threshold() {
        let e = eng();
        let rt = ReducerRuntime::new(Arc::clone(&e), 4);
        let bad = Program {
            instrs: vec![Instr::Trap("boom".into())],
        };
        for _ in 0..3 {
            let _ = rt.invoke("flaky", b"k", &bad);
        }
        assert!(rt.is_quarantined("flaky"));
        assert!(matches!(rt.invoke("flaky", b"k", &bad), ReducerResult::Quarantined));
        rt.reenable("flaky");
        assert!(!rt.is_quarantined("flaky"));
    }
}
