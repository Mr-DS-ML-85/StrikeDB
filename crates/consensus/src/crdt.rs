//! CRDTs — conflict-free replicated data types for coordination-free convergence.
//! GCounter (grow-only), PnCounter (inc/dec), LwwRegister (last-writer-wins).
//! Every type has a commutative, idempotent `merge` so replicas converge
//! regardless of message order or duplication.

use std::collections::HashMap;

/// Grow-only counter: per-node increments, value = sum across nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GCounter {
    counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }
    pub fn incr(&mut self, node: &str, by: u64) {
        *self.counts.entry(node.to_string()).or_insert(0) += by;
    }
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }
    /// Merge = per-node max (idempotent, commutative, associative).
    pub fn merge(&mut self, other: &GCounter) {
        for (node, &v) in &other.counts {
            let e = self.counts.entry(node.clone()).or_insert(0);
            *e = (*e).max(v);
        }
    }
}

/// Positive-negative counter: two GCounters, value = P - N.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PnCounter {
    p: GCounter,
    n: GCounter,
}

impl PnCounter {
    pub fn new() -> Self {
        Self { p: GCounter::new(), n: GCounter::new() }
    }
    pub fn incr(&mut self, node: &str, by: u64) {
        self.p.incr(node, by);
    }
    pub fn decr(&mut self, node: &str, by: u64) {
        self.n.incr(node, by);
    }
    pub fn value(&self) -> i64 {
        self.p.value() as i64 - self.n.value() as i64
    }
    pub fn merge(&mut self, other: &PnCounter) {
        self.p.merge(&other.p);
        self.n.merge(&other.n);
    }
}

/// Last-writer-wins register, ordered by (timestamp, node) for deterministic ties.
#[derive(Clone, Debug, PartialEq)]
pub struct LwwRegister {
    pub value: Vec<u8>,
    pub ts: u64,
    pub node: String,
}

impl LwwRegister {
    pub fn new(value: Vec<u8>, ts: u64, node: &str) -> Self {
        Self { value, ts, node: node.to_string() }
    }
    pub fn set(&mut self, value: Vec<u8>, ts: u64, node: &str) {
        if (ts, node) > (self.ts, self.node.as_str()) {
            self.value = value;
            self.ts = ts;
            self.node = node.to_string();
        }
    }
    pub fn merge(&mut self, other: &LwwRegister) {
        if (other.ts, other.node.as_str()) > (self.ts, self.node.as_str()) {
            self.value = other.value.clone();
            self.ts = other.ts;
            self.node = other.node.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcounter_converges_regardless_of_order() {
        let mut a = GCounter::new();
        let mut b = GCounter::new();
        a.incr("n1", 3);
        b.incr("n2", 5);
        a.incr("n1", 1); // a: n1=4
        let mut a2 = a.clone();
        a.merge(&b);
        b.merge(&a2);
        a2.merge(&b);
        assert_eq!(a.value(), 9);
        assert_eq!(b.value(), 9);
    }

    #[test]
    fn pncounter_inc_dec() {
        let mut c = PnCounter::new();
        c.incr("n1", 10);
        c.decr("n1", 3);
        assert_eq!(c.value(), 7);
    }

    #[test]
    fn lww_resolves_deterministically() {
        let mut r1 = LwwRegister::new(b"a".to_vec(), 1, "n1");
        let r2 = LwwRegister::new(b"b".to_vec(), 2, "n2");
        r1.merge(&r2);
        assert_eq!(r1.value, b"b");
        // idempotent
        let r3 = r1.clone();
        r1.merge(&r3);
        assert_eq!(r1.value, b"b");
    }
}
