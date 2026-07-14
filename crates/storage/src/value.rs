//! Value types stored in the substrate + a tiny self-describing binary codec.
//! No serde, no external crates — everything is length-prefixed little-endian.

use std::collections::BTreeMap;

/// A value in the unified substrate. Every "view" (KV, table, vector, TS, log)
/// ultimately stores one of these under a key.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Opaque bytes (KV strings, blobs).
    Bytes(Vec<u8>),
    /// 64-bit signed integer (counters, TS values).
    Int(i64),
    /// Dense f32 vector (embeddings).
    Vector(Vec<f32>),
    /// A row: ordered map of column -> value.
    Row(BTreeMap<String, Vec<u8>>),
    /// Tombstone marker (a delete at some version).
    Tombstone,
}

const T_BYTES: u8 = 1;
const T_INT: u8 = 2;
const T_VECTOR: u8 = 3;
const T_ROW: u8 = 4;
const T_TOMB: u8 = 5;

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn get_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let end = *pos + 4;
    let b = buf.get(*pos..end)?;
    *pos = end;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

impl Value {
    /// Serialize to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Value::Bytes(b) => {
                out.push(T_BYTES);
                put_u32(&mut out, b.len() as u32);
                out.extend_from_slice(b);
            }
            Value::Int(i) => {
                out.push(T_INT);
                out.extend_from_slice(&i.to_le_bytes());
            }
            Value::Vector(v) => {
                out.push(T_VECTOR);
                put_u32(&mut out, v.len() as u32);
                for f in v {
                    out.extend_from_slice(&f.to_le_bytes());
                }
            }
            Value::Row(map) => {
                out.push(T_ROW);
                put_u32(&mut out, map.len() as u32);
                for (k, v) in map {
                    put_u32(&mut out, k.len() as u32);
                    out.extend_from_slice(k.as_bytes());
                    put_u32(&mut out, v.len() as u32);
                    out.extend_from_slice(v);
                }
            }
            Value::Tombstone => out.push(T_TOMB),
        }
        out
    }

    /// Deserialize from bytes.
    pub fn decode(buf: &[u8]) -> Option<Value> {
        let tag = *buf.first()?;
        let mut pos = 1usize;
        match tag {
            T_BYTES => {
                let n = get_u32(buf, &mut pos)? as usize;
                let b = buf.get(pos..pos + n)?.to_vec();
                Some(Value::Bytes(b))
            }
            T_INT => {
                let b = buf.get(pos..pos + 8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Some(Value::Int(i64::from_le_bytes(a)))
            }
            T_VECTOR => {
                let n = get_u32(buf, &mut pos)? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let b = buf.get(pos..pos + 4)?;
                    v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    pos += 4;
                }
                Some(Value::Vector(v))
            }
            T_ROW => {
                let n = get_u32(buf, &mut pos)? as usize;
                let mut map = BTreeMap::new();
                for _ in 0..n {
                    let kl = get_u32(buf, &mut pos)? as usize;
                    let k = String::from_utf8(buf.get(pos..pos + kl)?.to_vec()).ok()?;
                    pos += kl;
                    let vl = get_u32(buf, &mut pos)? as usize;
                    let v = buf.get(pos..pos + vl)?.to_vec();
                    pos += vl;
                    map.insert(k, v);
                }
                Some(Value::Row(map))
            }
            T_TOMB => Some(Value::Tombstone),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let vals = vec![
            Value::Bytes(b"hi".to_vec()),
            Value::Int(-42),
            Value::Vector(vec![1.0, 2.5, -3.0]),
            Value::Tombstone,
        ];
        for v in vals {
            assert_eq!(Value::decode(&v.encode()).unwrap(), v);
        }
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), b"1".to_vec());
        m.insert("b".to_string(), b"22".to_vec());
        let r = Value::Row(m);
        assert_eq!(Value::decode(&r.encode()).unwrap(), r);
    }
}
