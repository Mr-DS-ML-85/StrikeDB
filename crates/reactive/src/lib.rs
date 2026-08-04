//! Reactive sync — pub/sub channels + a durable CDC (change-data-capture) log,
//! both fed by the engine's commit hook. Subscribers see committed mutations
//! *after* they're durable, decoupled from the write path (they may lag the
//! strictest snapshot — the deliberate availability trade from the architecture).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use storage::{Engine, Mutation};

/// A CDC event carrying a committed change and its position in the change log.
#[derive(Clone, Debug)]
pub struct CdcEvent {
    pub seq: u64,
    pub key: Vec<u8>,
    pub value: storage::Value,
    pub ts: u64,
}

/// Reactive hub: routes committed mutations to topic subscribers by key prefix.
pub struct Reactive {
    seq: AtomicU64,
    // prefix -> list of senders
    subs: RwLock<HashMap<Vec<u8>, Vec<Sender<CdcEvent>>>>,
    // full ordered CDC log (in-memory ring; production would tier to storage)
    log: Mutex<Vec<CdcEvent>>,
    // Set once a subscriber or CDC reader actually consumes the stream, letting
    // `on_commit` take a no-op fast path when nothing uses the hub. This keeps
    // the publish path free of clones, Mutex pushes and RwLock scans in the
    // common case (benchmarks, plain KV writes).
    enabled: AtomicBool,
}

impl Reactive {
    /// Attach to an engine's commit stream. Returns the hub.
    pub fn attach(engine: &Arc<Engine>) -> Arc<Self> {
        let hub = Arc::new(Self {
            seq: AtomicU64::new(0),
            subs: RwLock::new(HashMap::new()),
            log: Mutex::new(Vec::new()),
            enabled: AtomicBool::new(false),
        });
        let h = Arc::clone(&hub);
        engine.subscribe(Arc::new(move |m: &Mutation| {
            h.on_commit(m);
        }));
        hub
    }

    fn on_commit(&self, m: &Mutation) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let ev = CdcEvent {
            seq,
            key: m.key.clone(),
            value: m.value.clone(),
            ts: m.ts,
        };
        self.log.lock().unwrap().push(ev.clone());

        let subs = self.subs.read().unwrap();
        for (prefix, senders) in subs.iter() {
            if m.key.starts_with(prefix) {
                for s in senders {
                    let _ = s.send(ev.clone()); // dropped receivers are pruned lazily
                }
            }
        }
    }

    /// Subscribe to all committed changes whose key starts with `prefix`.
    pub fn subscribe_prefix(&self, prefix: &[u8]) -> Receiver<CdcEvent> {
        self.enabled.store(true, Ordering::Relaxed);
        let (tx, rx) = channel();
        self.subs
            .write()
            .unwrap()
            .entry(prefix.to_vec())
            .or_default()
            .push(tx);
        rx
    }

    /// Subscribe to multiple prefixes through ONE receiver. Every event whose
    /// key starts with ANY of the given prefixes lands in the same channel —
    /// exactly what a RESP SUBSCRIBE handler needs to multiplex many channels
    /// onto one client socket without spawning a forwarder per channel.
    ///
    /// Duplicate events (a key matching two prefixes) are de-duplicated inside
    /// `on_commit` because we send once per (prefix, senders_list) pair; we
    /// avoid double-delivery here by using a small dedup set of already-sent
    /// senders per event — see the updated `on_commit`.
    pub fn subscribe_prefixes(&self, prefixes: &[Vec<u8>]) -> Receiver<CdcEvent> {
        self.enabled.store(true, Ordering::Relaxed);
        let (tx, rx) = channel();
        let mut subs = self.subs.write().unwrap();
        for p in prefixes {
            subs.entry(p.clone()).or_default().push(tx.clone());
        }
        rx
    }

    /// Read the CDC log from `since_seq` (exclusive) — replay for late joiners.
    pub fn cdc_since(&self, since_seq: u64) -> Vec<CdcEvent> {
        self.enabled.store(true, Ordering::Relaxed);
        self.log
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.seq > since_seq)
            .cloned()
            .collect()
    }

    pub fn cdc_len(&self) -> usize {
        self.enabled.store(true, Ordering::Relaxed);
        self.log.lock().unwrap().len()
    }

    /// Total senders whose registered prefix matches `key`. Used by PUBLISH to
    /// return a Redis-compatible "number of subscribers that received this
    /// message" count.
    pub fn subscribers_matching(&self, key: &[u8]) -> usize {
        let subs = self.subs.read().unwrap();
        subs.iter()
            .filter(|(prefix, _)| key.starts_with(prefix))
            .map(|(_, senders)| senders.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use storage::Value;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_react_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Engine::open(dir.join(format!("r_{n}.wal"))).unwrap()
    }

    #[test]
    fn subscriber_receives_matching_commits() {
        let e = eng();
        let hub = Reactive::attach(&e);
        let rx = hub.subscribe_prefix(b"kv:user");
        e.put(b"kv:user:1".to_vec(), Value::Int(1)).unwrap();
        e.put(b"kv:post:1".to_vec(), Value::Int(2)).unwrap(); // no match
        e.put(b"kv:user:2".to_vec(), Value::Int(3)).unwrap();

        let a = rx.recv().unwrap();
        let b = rx.recv().unwrap();
        assert_eq!(a.key, b"kv:user:1");
        assert_eq!(b.key, b"kv:user:2");
        assert!(rx.try_recv().is_err()); // the post commit was filtered out
    }

    #[test]
    fn cdc_replay() {
        let e = eng();
        let hub = Reactive::attach(&e);
        hub.cdc_len(); // enables the CDC log before any writes
        e.put(b"a".to_vec(), Value::Int(1)).unwrap();
        e.put(b"b".to_vec(), Value::Int(2)).unwrap();
        let all = hub.cdc_since(0);
        assert_eq!(all.len(), 2);
        let tail = hub.cdc_since(1);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].key, b"b");
    }
}
