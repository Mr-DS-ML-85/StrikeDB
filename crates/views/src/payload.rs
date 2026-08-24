//! Arbitrary JSON payloads per point + Qdrant-style filtering — GAP-4's
//! flagship closure. Zero external crates: includes a small recursive-descent
//! JSON parser sufficient for metadata documents.
//!
//! Wire contract (opt-in extensions only, frozen strides untouched):
//!   VSETPAYLOAD ns id <json>     — durable under `vp:` keys
//!   VGETPAYLOAD ns id            — raw JSON back
//!   VSEARCHNS ... PFILTER <json> — Qdrant-shaped filter:
//!     {"must":[{"key":"cat","match":{"value":"news"}}],
//!      "should":[{"key":"n","range":{"gte":5}}],
//!      "must_not":[{"key":"tag","match":{"any":["x","y"]}}]}
//!   should needs at least one match when present (Qdrant min_should=1).

use storage::Engine;
use std::collections::HashMap;
use std::sync::Arc;

// ── Minimal JSON value ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(HashMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub fn parse_json(s: &[u8]) -> Result<Json, String> {
    let mut p = P { b: s, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes at {}", p.i));
    }
    Ok(v)
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn lit(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(format!("expected {s} at {}", self.i))
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek().ok_or("unexpected end")? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true").map(|_| Json::Bool(true)),
            b'f' => self.lit("false").map(|_| Json::Bool(false)),
            b'n' => self.lit("null").map(|_| Json::Null),
            _ => self.number(),
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.lit("{")?;
        let mut m = HashMap::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.lit(":")?;
            let v = self.value()?;
            m.insert(k, v);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected , or }} at {}", self.i)),
            }
        }
        Ok(Json::Obj(m))
    }
    fn array(&mut self) -> Result<Json, String> {
        self.lit("[")?;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected , or ] at {}", self.i)),
            }
        }
        Ok(Json::Arr(out))
    }
    fn string(&mut self) -> Result<String, String> {
        self.lit("\"")?;
        // Byte-accurate accumulation: pushing `byte as char` would corrupt
        // every multibyte UTF-8 sequence.
        let mut out: Vec<u8> = Vec::new();
        while let Some(c) = self.peek() {
            self.i += 1;
            match c {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|_| "invalid utf-8 in string".into())
                }
                b'\\' => {
                    let e = self.peek().ok_or("bad escape")?;
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'u' => {
                            if self.i + 4 > self.b.len() {
                                return Err("bad \\u".into());
                            }
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                .map_err(|_| "bad \\u")?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u")?;
                            self.i += 4;
                            let ch = char::from_u32(cp)
                                .filter(|_| !(0xD800..0xDC00).contains(&cp));
                            match ch {
                                Some(ch) => {
                                    let mut buf = [0u8; 4];
                                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                                }
                                None => return Err("bad \\u escape".into()),
                            }
                        }
                        other => out.push(other),
                    }
                }
                other => out.push(other),
            }
        }
        Err("unterminated string".into())
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') || self.peek() == Some(b'+') {
            self.i += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' {
                self.i += 1;
            } else {
                break;
            }
        }
        let txt = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad num")?;
        txt.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number {txt:?}"))
    }
}

// ── Filter AST + evaluation ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MatchVal {
    Str(String),
    Num(f64),
    Bool(bool),
    Any(Vec<MatchVal>),
}

#[derive(Debug, Clone)]
pub struct RangeCond {
    pub lt: Option<f64>,
    pub lte: Option<f64>,
    pub gt: Option<f64>,
    pub gte: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum Cond {
    Field {
        key: String,
        kind: FieldCond,
    },
    HasId(Vec<u64>),
}

#[derive(Debug, Clone)]
pub enum FieldCond {
    Match(MatchVal),
    Range(RangeCond),
    IsEmpty,
}

#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub must: Vec<Cond>,
    /// At least one should must match when the list is non-empty.
    pub should: Vec<Cond>,
    pub must_not: Vec<Cond>,
}

fn values_at<'a>(root: &'a Json, key: &str) -> Vec<&'a Json> {
    // Supports dotted paths ("a.b") and top-level arrays of objects.
    let mut cur = vec![root];
    for seg in key.split('.') {
        let mut next = Vec::new();
        for c in &cur {
            match c.get(seg) {
                Some(v) => next.push(v),
                None => {
                    if let Json::Arr(items) = c {
                        for it in items {
                            if let Some(v) = it.get(seg) {
                                next.push(v);
                            }
                        }
                    }
                }
            }
        }
        cur = next;
        if cur.is_empty() {
            return vec![];
        }
    }
    cur
}

fn match_one(val: &Json, mv: &MatchVal) -> bool {
    match (val, mv) {
        (Json::Str(s), MatchVal::Str(t)) => s == t,
        (Json::Num(a), MatchVal::Num(b)) => (a - b).abs() < 1e-9,
        (Json::Bool(a), MatchVal::Bool(b)) => a == b,
        _ => false,
    }
}

fn cond_holds(cond: &Cond, root: &Json, id: u64) -> bool {
    match cond {
        Cond::HasId(ids) => ids.contains(&id),
        Cond::Field { key, kind } => {
            let vals = values_at(root, key);
            match kind {
                FieldCond::IsEmpty => vals.is_empty()
                    || vals.iter().all(|v| match v {
                        Json::Null => true,
                        Json::Arr(a) => a.is_empty(),
                        _ => false,
                    }),
                FieldCond::Match(mv) => match mv {
                    MatchVal::Any(list) => vals.iter().any(|v| list.iter().any(|m| match_one(v, m))),
                    one => vals.iter().any(|v| match_one(v, one)),
                },
                FieldCond::Range(r) => vals.iter().any(|v| {
                    v.as_f64().map_or(false, |n| {
                        r.lt.map_or(true, |b| n < b)
                            && r.lte.map_or(true, |b| n <= b)
                            && r.gt.map_or(true, |b| n > b)
                            && r.gte.map_or(true, |b| n >= b)
                    })
                }),
            }
        }
    }
}

impl Filter {
    pub fn eval(&self, id: u64, root: &Json) -> bool {
        self.must.iter().all(|c| cond_holds(c, root, id))
            && !self.must_not.iter().any(|c| cond_holds(c, root, id))
            && (self.should.is_empty() || self.should.iter().any(|c| cond_holds(c, root, id)))
    }

    /// Parse a Qdrant-shaped filter JSON document.
    pub fn parse(doc: &Json) -> Result<Filter, String> {
        fn conds(v: &Json) -> Result<Vec<Cond>, String> {
            let arr = match v {
                Json::Arr(a) => a,
                _ => return Err("conditions must be arrays".into()),
            };
            arr.iter().map(one_cond).collect()
        }
        fn one_cond(v: &Json) -> Result<Cond, String> {
            if let Some(ids) = v.get("has_id") {
                if let Json::Arr(a) = ids {
                    let mut out = Vec::new();
                    for x in a {
                        out.push(x.as_f64().ok_or("has_id must be numeric")? as u64);
                    }
                    return Ok(Cond::HasId(out));
                }
            }
            let key = v
                .get("key")
                .and_then(|k| k.as_str())
                .ok_or("condition missing key")?
                .to_string();
            if let Some(m) = v.get("match") {
                let mv = if let Some(x) = m.get("value") {
                    scal(x)?
                } else if let Some(Json::Arr(a)) = m.get("keyword") {
                    MatchVal::Any(a.iter().map(scal).collect::<Result<_, _>>()?)
                } else if let Some(Json::Arr(a)) = m.get("text") {
                    MatchVal::Any(a.iter().map(scal).collect::<Result<_, _>>()?)
                } else if let Some(Json::Arr(a)) = m.get("integer") {
                    MatchVal::Any(a.iter().map(scal).collect::<Result<_, _>>()?)
                } else if let Some(Json::Arr(a)) = m.get("any") {
                    MatchVal::Any(a.iter().map(scal).collect::<Result<_, _>>()?)
                } else {
                    return Err("match missing value/any/keyword/integer/text".into());
                };
                return Ok(Cond::Field { key, kind: FieldCond::Match(mv) });
            }
            if let Some(r) = v.get("range") {
                let g = |k: &str| r.get(k).and_then(|x| x.as_f64());
                return Ok(Cond::Field {
                    key,
                    kind: FieldCond::Range(RangeCond {
                        lt: g("lt"),
                        lte: g("lte"),
                        gt: g("gt"),
                        gte: g("gte"),
                    }),
                });
            }
            if v.get("is_empty").is_some() {
                return Ok(Cond::Field { key, kind: FieldCond::IsEmpty });
            }
            Err("unsupported condition shape".into())
        }
        fn scal(v: &Json) -> Result<MatchVal, String> {
            Ok(match v {
                Json::Str(x) => MatchVal::Str(x.clone()),
                Json::Num(x) => MatchVal::Num(*x),
                Json::Bool(x) => MatchVal::Bool(*x),
                _ => return Err("match value must be scalar".into()),
            })
        }
        let mut f = Filter::default();
        if let Some(m) = doc.get("must") {
            f.must = conds(m)?;
        }
        if let Some(m) = doc.get("should") {
            f.should = conds(m)?;
        }
        if let Some(m) = doc.get("must_not") {
            f.must_not = conds(m)?;
        }
        if f.must.is_empty() && f.should.is_empty() && f.must_not.is_empty() {
            return Err("filter has no conditions".into());
        }
        Ok(f)
    }
}

/// Public accessor for dotted-path value lookup (used by facet counting).
pub fn values_at_pub<'a>(root: &'a Json, key: &str) -> Vec<&'a Json> {
    values_at(root, key)
}

// ── Durable payload store ────────────────────────────────────────────────────

pub fn payload_key(ns: &str, id: u64) -> Vec<u8> {
    let mut k = format!("vp:{ns}:").into_bytes();
    k.extend_from_slice(&id.to_be_bytes());
    k
}

/// Durable JSON payload storage over the shared MVCC+WAL engine.
pub struct PayloadStore {
    engine: Arc<Engine>,
}

impl PayloadStore {
    pub fn new(engine: Arc<Engine>) -> Self {
        PayloadStore { engine }
    }
    pub fn set(&self, ns: &str, id: u64, json: &[u8]) -> std::io::Result<Json> {
        let doc = parse_json(json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad JSON: {e}"))
        })?;
        self.engine.put(payload_key(ns, id), storage::Value::Bytes(json.to_vec()))?;
        Ok(doc)
    }
    pub fn get_raw(&self, ns: &str, id: u64) -> Option<Vec<u8>> {
        match self.engine.get(&payload_key(ns, id)) {
            Some(storage::Value::Bytes(b)) => Some(b),
            _ => None,
        }
    }
    pub fn del(&self, ns: &str, id: u64) -> std::io::Result<bool> {
        let existed = self.get_raw(ns, id).is_some();
        self.engine.delete(payload_key(ns, id))?;
        Ok(existed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metadata_documents() {
        let doc = br#"{"cat":"news","price":12.5,"tags":["a","b"],"ok":true,"nested":{"deep":7}}"#;
        let j = parse_json(doc).unwrap();
        assert_eq!(j.get("cat").unwrap().as_str(), Some("news"));
        assert_eq!(j.get("price").unwrap().as_f64(), Some(12.5));
        assert_eq!(j.get("nested").unwrap().get("deep").unwrap().as_f64(), Some(7.0));
        // Multibyte survives round-trip.
        let uni = parse_json("{\"k\":\"héllo→世界\"}".as_bytes()).unwrap();
        assert_eq!(uni.get("k").unwrap().as_str(), Some("héllo→世界"));
        // Rejects garbage.
        assert!(parse_json(b"{bad}").is_err());
    }

    #[test]
    fn qdrant_shaped_filter_evaluates() {
        let fdoc = r#"{
            "must":[{"key":"cat","match":{"value":"news"}}],
            "should":[{"key":"price","range":{"gte":10,"lt":100}}],
            "must_not":[{"key":"tag","match":{"any":["spam","junk"]}}]
        }"#;
        let f = Filter::parse(&parse_json(fdoc.as_bytes()).unwrap()).unwrap();

        let hit = parse_json(br#"{"cat":"news","price":42,"tag":"clean"}"#).unwrap();
        assert!(f.eval(1, &hit));

        // Wrong category → must fails.
        let miss = parse_json(br#"{"cat":"blog","price":42}"#).unwrap();
        assert!(!f.eval(2, &miss));

        // Price below should-window → should fails.
        let low = parse_json(br#"{"cat":"news","price":3,"tag":"clean"}"#).unwrap();
        assert!(!f.eval(3, &low));

        // must_not tag hits even when must+should pass.
        let junk = parse_json(br#"{"cat":"news","price":50,"tag":"spam"}"#).unwrap();
        assert!(!f.eval(4, &junk));

        // Dotted path + has_id.
        let f2doc = r#"{"must":[{"key":"a.b","range":{"gte":2}},
                                 {"has_id":[7,9]}]}"#;
        let f2 = Filter::parse(&parse_json(f2doc.as_bytes()).unwrap()).unwrap();
        let d = parse_json(br#"{"a":{"b":3}}"#).unwrap();
        assert!(f2.eval(7, &d));
        assert!(!f2.eval(8, &d));
    }
}
