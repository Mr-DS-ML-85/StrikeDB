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
        let e = self.counts.entry(node.to_string()).or_insert(0);
        *e = e.saturating_add(by);
    }
    pub fn value(&self) -> u64 {
        self.counts.values().fold(0u64, |a, &b| a.saturating_add(b))
    }

    /// Compact binary encoding (durable storage): `[n:u32]` then per node
    /// `[len:u32][name][count:u64]`, little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut nodes: Vec<_> = self.counts.iter().collect();
        nodes.sort();
        out.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        for (n, c) in nodes {
            out.extend_from_slice(&(n.len() as u32).to_le_bytes());
            out.extend_from_slice(n.as_bytes());
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        let g = Self::read(&mut r)?;
        r.0.is_empty().then_some(g)
    }

    fn read(r: &mut Reader) -> Option<Self> {
        let n = r.u32()?;
        let mut counts = HashMap::new();
        for _ in 0..n {
            let name = String::from_utf8(r.bytes()?.to_vec()).ok()?;
            counts.insert(name, r.u64()?);
        }
        Some(Self { counts })
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
        (self.p.value() as i128 - self.n.value() as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.p.to_bytes();
        out.extend_from_slice(&self.n.to_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        let p = GCounter::read(&mut r)?;
        let n = GCounter::read(&mut r)?;
        r.0.is_empty().then_some(Self { p, n })
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
    /// `[ts:u64][len:u32][node][value...]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.ts.to_le_bytes().to_vec();
        out.extend_from_slice(&(self.node.len() as u32).to_le_bytes());
        out.extend_from_slice(self.node.as_bytes());
        out.extend_from_slice(&self.value);
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        let ts = r.u64()?;
        let node = String::from_utf8(r.bytes()?.to_vec()).ok()?;
        Some(Self { value: r.0.to_vec(), ts, node })
    }

    pub fn merge(&mut self, other: &LwwRegister) {
        if (other.ts, other.node.as_str()) > (self.ts, self.node.as_str()) {
            self.value = other.value.clone();
            self.ts = other.ts;
            self.node = other.node.clone();
        }
    }
}

/// Little-endian cursor for the encodings above.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Some(a)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_roundtrip_and_saturate() {
        let mut g = GCounter::new();
        g.incr("a", u64::MAX);
        g.incr("a", 5);
        g.incr("b", 7);
        assert_eq!(g.value(), u64::MAX);
        assert_eq!(GCounter::from_bytes(&g.to_bytes()), Some(g));
        let mut p = PnCounter::new();
        p.incr("x", 3);
        p.decr("y", 10);
        assert_eq!(PnCounter::from_bytes(&p.to_bytes()).unwrap().value(), -7);
        let r = LwwRegister::new(b"val".to_vec(), 9, "n1");
        assert_eq!(LwwRegister::from_bytes(&r.to_bytes()), Some(r));
        assert!(GCounter::from_bytes(b"\xff").is_none());
    }

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
