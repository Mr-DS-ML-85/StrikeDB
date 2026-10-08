//! Redis pub/sub broker.
//!
//! Replaces the old design that routed PUBLISH through the storage engine as a
//! `chan:<channel>` key. That (a) wrote every message to the WAL and kept it in
//! the keyspace forever, (b) matched subscribers by key PREFIX, so a client
//! subscribed to `news` also received `newsletter`, and (c) treated PSUBSCRIBE
//! patterns as literal channel names. Pub/sub is fire-and-forget in Redis; this
//! broker is purely in-memory, matches channels exactly, matches patterns with
//! Redis glob semantics, and prunes a subscriber the moment its send fails.

use crate::acl::glob_match;
use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::RwLock;

/// Messages a subscriber may have queued before it is considered too slow
/// and disconnected (Redis `client-output-buffer-limit pubsub`). Without a
/// bound, a subscriber that stops reading made the server buffer every
/// published message in RAM forever. Override with DBSTRIKE_PUBSUB_QUEUE.
pub fn queue_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("DBSTRIKE_PUBSUB_QUEUE").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(100_000)
    })
}

pub type Sender<T> = SyncSender<T>;

#[derive(Clone, Debug)]
pub enum Msg {
    Message { channel: Vec<u8>, payload: Vec<u8> },
    PMessage { pattern: Vec<u8>, channel: Vec<u8>, payload: Vec<u8> },
}

#[derive(Default)]
pub struct Broker {
    /// channel -> (subscriber id -> sender)
    channels: RwLock<HashMap<Vec<u8>, HashMap<u64, Sender<Msg>>>>,
    /// pattern -> (subscriber id -> sender)
    patterns: RwLock<HashMap<Vec<u8>, HashMap<u64, Sender<Msg>>>>,
    /// subscriber id -> a handle on its socket, used to disconnect it when
    /// its queue overflows (its reader then sees EOF and cleans up).
    sockets: RwLock<HashMap<u64, TcpStream>>,
}

impl Broker {
    pub fn register(&self, id: u64, sock: TcpStream) {
        self.sockets.write().unwrap().insert(id, sock);
    }

    pub fn deregister(&self, id: u64) {
        self.sockets.write().unwrap().remove(&id);
    }

    /// Drop every subscription of `id` and shut its socket down.
    fn evict(&self, id: u64) {
        for map in [&self.channels, &self.patterns] {
            let mut m = map.write().unwrap();
            for subs in m.values_mut() {
                subs.remove(&id);
            }
            m.retain(|_, v| !v.is_empty());
        }
        if let Some(sock) = self.sockets.write().unwrap().remove(&id) {
            let _ = sock.shutdown(std::net::Shutdown::Both);
        }
    }

    pub fn subscribe(&self, id: u64, channel: &[u8], tx: &Sender<Msg>) {
        self.channels.write().unwrap().entry(channel.to_vec()).or_default().insert(id, tx.clone());
    }

    pub fn psubscribe(&self, id: u64, pattern: &[u8], tx: &Sender<Msg>) {
        self.patterns.write().unwrap().entry(pattern.to_vec()).or_default().insert(id, tx.clone());
    }

    pub fn unsubscribe(&self, id: u64, channel: &[u8]) {
        let mut ch = self.channels.write().unwrap();
        if let Some(m) = ch.get_mut(channel) {
            m.remove(&id);
            if m.is_empty() {
                ch.remove(channel);
            }
        }
    }

    pub fn punsubscribe(&self, id: u64, pattern: &[u8]) {
        let mut p = self.patterns.write().unwrap();
        if let Some(m) = p.get_mut(pattern) {
            m.remove(&id);
            if m.is_empty() {
                p.remove(pattern);
            }
        }
    }

    /// Deliver to every exact and pattern subscriber; returns how many
    /// received it (the Redis PUBLISH reply). Dead receivers are removed.
    pub fn publish(&self, channel: &[u8], payload: &[u8]) -> usize {
        let mut delivered = 0;
        // Subscribers whose receiver is gone, or whose queue is full.
        let mut evict: Vec<u64> = Vec::new();
        let mut deliver = |id: u64, tx: &Sender<Msg>, msg: Msg| match tx.try_send(msg) {
            Ok(()) => delivered += 1,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => evict.push(id),
        };
        if let Some(subs) = self.channels.read().unwrap().get(channel) {
            for (id, tx) in subs {
                deliver(*id, tx, Msg::Message { channel: channel.to_vec(), payload: payload.to_vec() });
            }
        }
        for (pat, subs) in self.patterns.read().unwrap().iter() {
            if !glob_match(pat, channel) {
                continue;
            }
            for (id, tx) in subs {
                deliver(*id, tx, Msg::PMessage { pattern: pat.clone(), channel: channel.to_vec(), payload: payload.to_vec() });
            }
        }
        evict.sort_unstable();
        evict.dedup();
        for id in evict {
            self.evict(id);
        }
        delivered
    }

    /// `PUBSUB CHANNELS [pattern]`
    pub fn channels(&self, pattern: Option<&[u8]>) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = self
            .channels
            .read()
            .unwrap()
            .keys()
            .filter(|c| pattern.is_none_or(|p| glob_match(p, c)))
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// `PUBSUB NUMSUB ch` count.
    pub fn numsub(&self, channel: &[u8]) -> usize {
        self.channels.read().unwrap().get(channel).map_or(0, |m| m.len())
    }

    /// `PUBSUB NUMPAT`
    pub fn numpat(&self) -> usize {
        self.patterns.read().unwrap().values().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;
    fn channel() -> (Sender<Msg>, std::sync::mpsc::Receiver<Msg>) {
        sync_channel(4)
    }

    #[test]
    fn exact_channels_and_patterns() {
        let b = Broker::default();
        let (tx, rx) = channel();
        b.subscribe(1, b"news", &tx);
        b.psubscribe(1, b"n*s", &tx);
        assert_eq!(b.publish(b"newsletter", b"x"), 0, "prefix must not match");
        assert_eq!(b.publish(b"news", b"y"), 2, "exact + pattern");
        assert!(matches!(rx.try_recv().unwrap(), Msg::Message { .. }));
        assert!(matches!(rx.try_recv().unwrap(), Msg::PMessage { .. }));
        b.unsubscribe(1, b"news");
        assert_eq!(b.publish(b"news", b"z"), 1);
    }

    #[test]
    fn slow_subscriber_is_evicted_when_its_queue_fills() {
        let b = Broker::default();
        let (tx, _rx) = channel(); // never drained, capacity 4
        b.subscribe(9, b"c", &tx);
        for _ in 0..4 {
            assert_eq!(b.publish(b"c", b"m"), 1);
        }
        assert_eq!(b.publish(b"c", b"m"), 0, "5th message overflows");
        assert_eq!(b.numsub(b"c"), 0, "evicted");
    }

    #[test]
    fn dead_subscribers_are_pruned() {
        let b = Broker::default();
        let (tx, rx) = channel();
        b.subscribe(7, b"c", &tx);
        drop(rx);
        assert_eq!(b.publish(b"c", b"m"), 0);
        assert_eq!(b.numsub(b"c"), 0);
    }
}
