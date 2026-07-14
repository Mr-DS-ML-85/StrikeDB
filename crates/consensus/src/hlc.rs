//! Hybrid Logical Clock — monotonic timestamps that respect causality across
//! nodes. Each timestamp packs (physical_millis, logical_counter). Merging a
//! remote timestamp on receive keeps the whole cluster causally ordered.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub physical: u64, // millis since epoch
    pub logical: u32,
}

pub struct Hlc {
    state: Mutex<Timestamp>,
}

fn phys_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

impl Hlc {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Timestamp { physical: 0, logical: 0 }),
        }
    }

    /// Generate a timestamp for a local event.
    pub fn now(&self) -> Timestamp {
        let mut s = self.state.lock().unwrap();
        let pt = phys_now();
        if pt > s.physical {
            s.physical = pt;
            s.logical = 0;
        } else {
            s.logical += 1;
        }
        *s
    }

    /// Merge a timestamp received from another node, advancing our clock.
    pub fn update(&self, remote: Timestamp) -> Timestamp {
        let mut s = self.state.lock().unwrap();
        let pt = phys_now();
        let max_p = pt.max(s.physical).max(remote.physical);
        if max_p == s.physical && max_p == remote.physical {
            s.logical = s.logical.max(remote.logical) + 1;
        } else if max_p == s.physical {
            s.logical += 1;
        } else if max_p == remote.physical {
            s.logical = remote.logical + 1;
        } else {
            s.logical = 0;
        }
        s.physical = max_p;
        *s
    }
}

impl Default for Hlc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic() {
        let c = Hlc::new();
        let a = c.now();
        let b = c.now();
        assert!(b > a);
    }

    #[test]
    fn causal_merge_advances() {
        let node1 = Hlc::new();
        let node2 = Hlc::new();
        let t1 = node1.now();
        // node2 receives t1 and must produce something strictly greater
        let t2 = node2.update(t1);
        assert!(t2 > t1);
    }
}
