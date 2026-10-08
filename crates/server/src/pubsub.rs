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
use std::sync::mpsc::Sender;
use std::sync::RwLock;

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
}

impl Broker {
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
        let mut dead_exact: Vec<u64> = Vec::new();
        if let Some(subs) = self.channels.read().unwrap().get(channel) {
            for (id, tx) in subs {
                let msg = Msg::Message { channel: channel.to_vec(), payload: payload.to_vec() };
                if tx.send(msg).is_ok() {
                    delivered += 1;
                } else {
                    dead_exact.push(*id);
                }
            }
        }
        let mut dead_pat: Vec<(Vec<u8>, u64)> = Vec::new();
        for (pat, subs) in self.patterns.read().unwrap().iter() {
            if !glob_match(pat, channel) {
                continue;
            }
            for (id, tx) in subs {
                let msg = Msg::PMessage { pattern: pat.clone(), channel: channel.to_vec(), payload: payload.to_vec() };
                if tx.send(msg).is_ok() {
                    delivered += 1;
                } else {
                    dead_pat.push((pat.clone(), *id));
                }
            }
        }
        for id in dead_exact {
            self.unsubscribe(id, channel);
        }
        for (p, id) in dead_pat {
            self.punsubscribe(id, &p);
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
    use std::sync::mpsc::channel;

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
    fn dead_subscribers_are_pruned() {
        let b = Broker::default();
        let (tx, rx) = channel();
        b.subscribe(7, b"c", &tx);
        drop(rx);
        assert_eq!(b.publish(b"c", b"m"), 0);
        assert_eq!(b.numsub(b"c"), 0);
    }
}
