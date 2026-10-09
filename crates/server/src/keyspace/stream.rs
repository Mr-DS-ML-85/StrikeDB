//! Redis Streams on the keyspace executor.
//!
//! Layout (all under the stream's `{hash tag}`, so one stream is one shard):
//!   * `kv:<key>`            header (`Meta`, type stream; `len` = entry
//!                           count, `head`/`tail` = last generated ID)
//!   * `c:{k}:x<ms><seq>`    one entry: field/value list (IDs big-endian,
//!                           so the engine's key order IS stream order)
//!                           and, in `Meta::ext`, entries-added and the
//!                           max deleted ID
//!   * `c:{k}:g<group>`      consumer group: last-delivered ID, entries-read
//!   * `c:{k}:p<g><id>`      pending entry (PEL): owner, delivery time/count
//!   * `c:{k}:q<g><name>`    consumer: seen / active times
//!
//! Unlike other collections a stream may be empty (after XDEL, or created by
//! XGROUP CREATE MKSTREAM): it only disappears on DEL / expiry.
//!
//! Blocking (`XREAD`/`XREADGROUP ... BLOCK`) is driven by the server, which
//! re-runs the command without BLOCK until it returns data or times out.

use super::*;

pub(super) const T_STREAM: u8 = 5;
pub(super) const SUB_XENT: u8 = b'x';
pub(super) const SUB_XGRP: u8 = b'g';
pub(super) const SUB_XPEL: u8 = b'p';
pub(super) const SUB_XCON: u8 = b'q';

/// Stream entry ID `<ms>-<seq>`.
type Id = (u64, u64);

const ID_MIN: Id = (0, 0);
const ID_MAX: Id = (u64::MAX, u64::MAX);
/// Approximate trimming (`~`) removes whole nodes of this many entries,
/// as Redis does with its default `stream-node-max-entries`.
const NODE: u64 = 100;

fn invalid_id() -> Resp {
    err("Invalid stream ID specified as stream command argument")
}

fn id_bytes(id: Id) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&id.0.to_be_bytes());
    b[8..].copy_from_slice(&id.1.to_be_bytes());
    b
}

fn id_from(b: &[u8]) -> Id {
    (u64::from_be_bytes(b[..8].try_into().unwrap()), u64::from_be_bytes(b[8..16].try_into().unwrap()))
}

fn fmt_id(id: Id) -> Vec<u8> {
    format!("{}-{}", id.0, id.1).into_bytes()
}

fn id_resp(id: Id) -> Resp {
    bulk(fmt_id(id))
}

fn next_id(id: Id) -> Option<Id> {
    if id.1 < u64::MAX {
        Some((id.0, id.1 + 1))
    } else if id.0 < u64::MAX {
        Some((id.0 + 1, 0))
    } else {
        None
    }
}

fn prev_id(id: Id) -> Option<Id> {
    if id.1 > 0 {
        Some((id.0, id.1 - 1))
    } else if id.0 > 0 {
        Some((id.0 - 1, u64::MAX))
    } else {
        None
    }
}

fn parse_u64(s: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(s).ok()?;
    if s.is_empty() || s.starts_with('+') {
        return None;
    }
    s.parse().ok()
}

/// `<ms>` or `<ms>-<seq>`; a missing seq becomes `missing_seq`.
fn parse_id(a: &[u8], missing_seq: u64) -> Option<Id> {
    match a.iter().position(|&b| b == b'-') {
        Some(p) => Some((parse_u64(&a[..p])?, parse_u64(&a[p + 1..])?)),
        None => Some((parse_u64(a)?, missing_seq)),
    }
}

fn strict_id(a: &[u8]) -> R<Id> {
    parse_id(a, 0).ok_or_else(invalid_id)
}

/// Range bound: `-`, `+`, `(`-exclusive, incomplete IDs (`ms`).
fn range_start(a: &[u8]) -> R<Option<Id>> {
    if a == b"-" {
        return Ok(Some(ID_MIN));
    }
    if a == b"+" {
        return Ok(Some(ID_MAX));
    }
    if let Some(rest) = a.strip_prefix(b"(") {
        let id = parse_id(rest, 0).ok_or_else(invalid_id)?;
        return match next_id(id) {
            Some(n) => Ok(Some(n)),
            None => Err(err("invalid start ID for the interval")),
        };
    }
    Ok(Some(parse_id(a, 0).ok_or_else(invalid_id)?))
}

fn range_end(a: &[u8]) -> R<Option<Id>> {
    if a == b"-" {
        return Ok(Some(ID_MIN));
    }
    if a == b"+" {
        return Ok(Some(ID_MAX));
    }
    if let Some(rest) = a.strip_prefix(b"(") {
        let id = parse_id(rest, u64::MAX).ok_or_else(invalid_id)?;
        return match prev_id(id) {
            Some(p) => Ok(Some(p)),
            None => Err(err("invalid end ID for the interval")),
        };
    }
    Ok(Some(parse_id(a, u64::MAX).ok_or_else(invalid_id)?))
}

fn enc_fields(f: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + f.iter().map(|x| x.len() + 4).sum::<usize>());
    v.extend_from_slice(&(f.len() as u32).to_le_bytes());
    for x in f {
        v.extend_from_slice(&(x.len() as u32).to_le_bytes());
        v.extend_from_slice(x);
    }
    v
}

fn dec_fields(b: &[u8]) -> Vec<Vec<u8>> {
    let rd = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    if b.len() < 4 {
        return Vec::new();
    }
    let n = rd(0);
    let mut out = Vec::with_capacity(n);
    let mut i = 4;
    for _ in 0..n {
        if i + 4 > b.len() {
            break;
        }
        let l = rd(i);
        i += 4;
        out.push(b[i..(i + l).min(b.len())].to_vec());
        i += l;
    }
    out
}

fn entry_resp(id: Id, fields: Vec<Vec<u8>>) -> Resp {
    arr(vec![id_resp(id), arr(fields.into_iter().map(bulk).collect())])
}

fn as_bytes(v: &Value) -> &[u8] {
    match v {
        Value::Bytes(b) => b,
        _ => &[],
    }
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    b.get(i..i + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

fn group_part(group: &[u8]) -> Vec<u8> {
    let mut v = (group.len() as u32).to_be_bytes().to_vec();
    v.extend_from_slice(group);
    v
}

// ── State ──────────────────────────────────────────────────────────────────

struct Stream {
    m: Meta,
    last: Id,
    added: u64,
    maxdel: Id,
}

#[derive(Clone, Copy)]
struct Group {
    last: Id,
    /// -1 = unknown ("invalid" in Redis terms).
    read: i64,
}

struct Pel {
    time: u64,
    count: u64,
    owner: Vec<u8>,
}

impl Pel {
    fn enc(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(16 + self.owner.len());
        v.extend_from_slice(&self.time.to_le_bytes());
        v.extend_from_slice(&self.count.to_le_bytes());
        v.extend_from_slice(&self.owner);
        v
    }

    fn dec(b: &[u8]) -> Pel {
        Pel { time: u64_at(b, 0), count: u64_at(b, 8), owner: b.get(16..).unwrap_or(&[]).to_vec() }
    }
}

fn load(cx: &mut Ctx, key: &[u8]) -> R<Option<Stream>> {
    let Some((m, _)) = cx.coll(key, T_STREAM)? else { return Ok(None) };
    Ok(Some(Stream { m, last: (m.head as u64, m.tail as u64), added: m.ext[0], maxdel: (m.ext[1], m.ext[2]) }))
}

fn new_stream() -> Stream {
    Stream { m: Meta::new(T_STREAM), last: ID_MIN, added: 0, maxdel: ID_MIN }
}

/// Persist the header and counters. Unlike `put_meta`, an empty stream
/// stays (Redis keeps a stream until DEL).
fn save(cx: &mut Ctx, key: &[u8], s: &mut Stream) {
    if s.m.ver == 0 {
        cx.created = true;
    }
    s.m.ver += 1;
    s.m.head = s.last.0 as i64;
    s.m.tail = s.last.1 as i64;
    s.m.ext = [s.added, s.maxdel.0, s.maxdel.1];
    cx.tx.put(kv(key), Value::Meta(s.m.encode()));
}

fn ent_key(key: &[u8], id: Id) -> Vec<u8> {
    ekey(key, SUB_XENT, &id_bytes(id))
}

/// Entries with `lo <= id <= hi`, at most `count` (0 = all).
fn range(cx: &mut Ctx, key: &[u8], lo: Id, hi: Id, rev: bool, count: usize) -> Vec<(Id, Vec<Vec<u8>>)> {
    let base = ebase(key, SUB_XENT);
    let plen = base.len();
    let start = ent_key(key, lo);
    let mut end = ent_key(key, hi);
    end.push(0); // exclusive end just above `hi`
    let mut out = Vec::new();
    if lo > hi {
        return out;
    }
    cx.each(&base, start, end, rev, |k, v| {
        out.push((id_from(&k[plen..]), dec_fields(as_bytes(v))));
        count == 0 || out.len() < count
    });
    out
}

fn first_id(cx: &mut Ctx, key: &[u8]) -> Option<Id> {
    range(cx, key, ID_MIN, ID_MAX, false, 1).first().map(|e| e.0)
}

fn entry(cx: &mut Ctx, key: &[u8], id: Id) -> Option<Vec<Vec<u8>>> {
    cx.tx.get(&ent_key(key, id)).map(|v| dec_fields(as_bytes(&v)))
}

fn group_key(key: &[u8], g: &[u8]) -> Vec<u8> {
    ekey(key, SUB_XGRP, g)
}

fn load_group(cx: &mut Ctx, key: &[u8], g: &[u8]) -> Option<Group> {
    let v = cx.tx.get(&group_key(key, g))?;
    let b = as_bytes(&v);
    Some(Group { last: (u64_at(b, 0), u64_at(b, 8)), read: u64_at(b, 16) as i64 })
}

fn save_group(cx: &mut Ctx, key: &[u8], g: &[u8], gr: Group) {
    let mut b = Vec::with_capacity(24);
    b.extend_from_slice(&gr.last.0.to_le_bytes());
    b.extend_from_slice(&gr.last.1.to_le_bytes());
    b.extend_from_slice(&(gr.read as u64).to_le_bytes());
    cx.tx.put(group_key(key, g), Value::Bytes(b));
}

fn groups(cx: &mut Ctx, key: &[u8]) -> Vec<(Vec<u8>, Group)> {
    cx.elements(key, SUB_XGRP)
        .into_iter()
        .map(|(name, v)| {
            let b = as_bytes(&v);
            (name, Group { last: (u64_at(b, 0), u64_at(b, 8)), read: u64_at(b, 16) as i64 })
        })
        .collect()
}

fn pel_key(key: &[u8], g: &[u8], id: Id) -> Vec<u8> {
    let mut rest = group_part(g);
    rest.extend_from_slice(&id_bytes(id));
    ekey(key, SUB_XPEL, &rest)
}

fn load_pel(cx: &mut Ctx, key: &[u8], g: &[u8], id: Id) -> Option<Pel> {
    cx.tx.get(&pel_key(key, g, id)).map(|v| Pel::dec(as_bytes(&v)))
}

/// Pending entries of group `g` with `lo <= id <= hi` (count 0 = all).
fn pel_range(cx: &mut Ctx, key: &[u8], g: &[u8], lo: Id, hi: Id, count: usize, mut keep: impl FnMut(&Pel) -> bool) -> Vec<(Id, Pel)> {
    let base = ekey(key, SUB_XPEL, &group_part(g));
    let plen = base.len();
    let start = pel_key(key, g, lo);
    let mut end = pel_key(key, g, hi);
    end.push(0);
    let mut out = Vec::new();
    if lo > hi {
        return out;
    }
    cx.each(&base, start, end, false, |k, v| {
        let p = Pel::dec(as_bytes(v));
        if keep(&p) {
            out.push((id_from(&k[plen..]), p));
        }
        count == 0 || out.len() < count
    });
    out
}

fn con_key(key: &[u8], g: &[u8], name: &[u8]) -> Vec<u8> {
    let mut rest = group_part(g);
    rest.extend_from_slice(name);
    ekey(key, SUB_XCON, &rest)
}

/// (seen, active) times; active -1 = never.
fn load_con(cx: &mut Ctx, key: &[u8], g: &[u8], name: &[u8]) -> Option<(u64, i64)> {
    cx.tx.get(&con_key(key, g, name)).map(|v| {
        let b = as_bytes(&v);
        (u64_at(b, 0), u64_at(b, 8) as i64)
    })
}

fn save_con(cx: &mut Ctx, key: &[u8], g: &[u8], name: &[u8], seen: u64, active: i64) {
    let mut b = Vec::with_capacity(16);
    b.extend_from_slice(&seen.to_le_bytes());
    b.extend_from_slice(&(active as u64).to_le_bytes());
    cx.tx.put(con_key(key, g, name), Value::Bytes(b));
}

/// Touch (creating if needed) a consumer; `active` = it got entries.
fn touch_con(cx: &mut Ctx, key: &[u8], g: &[u8], name: &[u8], active: bool) -> bool {
    let cur = load_con(cx, key, g, name);
    let act = if active { cx.now as i64 } else { cur.map_or(-1, |c| c.1) };
    save_con(cx, key, g, name, cx.now, act);
    cur.is_none()
}

fn consumers(cx: &mut Ctx, key: &[u8], g: &[u8]) -> Vec<(Vec<u8>, u64, i64)> {
    let base = ekey(key, SUB_XCON, &group_part(g));
    let plen = base.len();
    let mut end = base.clone();
    end.push(0xff);
    end.push(0xff);
    let mut out = Vec::new();
    cx.each(&base.clone(), base, end, false, |k, v| {
        let b = as_bytes(v);
        out.push((k[plen..].to_vec(), u64_at(b, 0), u64_at(b, 8) as i64));
        true
    });
    out
}

fn delete_prefix(cx: &mut Ctx, start: Vec<u8>) {
    let mut end = start.clone();
    end.push(0xff);
    end.push(0xff);
    for (k, _) in cx.tx.scan_range(&start, &start, &end, usize::MAX, false) {
        cx.tx.delete(k);
    }
}

fn nogroup(key: &[u8], g: &[u8]) -> Resp {
    err(&format!("NOGROUP No such key '{}' or consumer group '{}'", String::from_utf8_lossy(key), String::from_utf8_lossy(g)))
}

fn need_key() -> Resp {
    err("The XGROUP subcommand requires the key to exist. Note that for CREATE you may want to use the MKSTREAM option to create an empty stream automatically.")
}

// ── entries-read / lag (Redis 7 consumer-group lag tracking) ───────────────

fn has_tombstones(cx: &mut Ctx, key: &[u8], s: &Stream, start: Id) -> bool {
    if s.m.len == 0 || s.maxdel == ID_MIN {
        return false;
    }
    match first_id(cx, key) {
        Some(f) if f > s.maxdel => false,
        _ => start <= s.maxdel,
    }
}

fn estimate_read(cx: &mut Ctx, key: &[u8], s: &Stream, id: Id) -> i64 {
    if s.added == 0 {
        return 0;
    }
    if s.m.len == 0 && id <= s.last {
        return s.added as i64;
    }
    if id == s.last {
        return s.added as i64;
    }
    if id > s.last {
        return -1;
    }
    let first = first_id(cx, key).unwrap_or(ID_MAX);
    if s.maxdel == ID_MIN || s.maxdel < first {
        if id < first {
            return (s.added - s.m.len) as i64;
        }
        if id == first {
            return (s.added - s.m.len + 1) as i64;
        }
    }
    -1
}

// ── Trimming ───────────────────────────────────────────────────────────────

enum Trim {
    MaxLen(u64),
    MinId(Id),
}

struct TrimSpec {
    how: Trim,
    approx: bool,
    limit: u64,
}

/// Parse `MAXLEN|MINID [=|~] threshold [LIMIT n]` starting at `a[*i]`.
/// Returns None if `a[*i]` is not a trim keyword.
fn parse_trim(a: &[Vec<u8>], i: &mut usize, spec: &mut Option<TrimSpec>, limit: &mut Option<u64>) -> R<bool> {
    let opt = String::from_utf8_lossy(&a[*i]).to_ascii_uppercase();
    match opt.as_str() {
        "MAXLEN" | "MINID" => {
            *i += 1;
            let mut approx = false;
            if let Some(x) = a.get(*i) {
                if x == b"~" || x == b"=" {
                    approx = x == b"~";
                    *i += 1;
                }
            }
            let t = a.get(*i).ok_or_else(|| err("syntax error"))?;
            let how = if opt == "MAXLEN" {
                let n = parse_i64(t).ok_or_else(not_int)?;
                if n < 0 {
                    return Err(err("The MAXLEN argument must be >= 0."));
                }
                Trim::MaxLen(n as u64)
            } else {
                Trim::MinId(strict_id(t)?)
            };
            *i += 1;
            *spec = Some(TrimSpec { how, approx, limit: 0 });
            Ok(true)
        }
        "LIMIT" => {
            let n = a.get(*i + 1).and_then(|v| parse_i64(v)).ok_or_else(not_int)?;
            if n < 0 {
                return Err(err("The LIMIT argument must be >= 0."));
            }
            *limit = Some(n as u64);
            *i += 2;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn finish_trim(spec: &mut Option<TrimSpec>, limit: Option<u64>) -> R<()> {
    match (spec.as_mut(), limit) {
        (None, Some(_)) => Err(err("syntax error")),
        (Some(s), Some(_)) if !s.approx => Err(err("syntax error, LIMIT cannot be used without the special ~ option")),
        (Some(s), l) => {
            s.limit = if s.approx { l.unwrap_or(NODE * 100) } else { 0 };
            Ok(())
        }
        (None, None) => Ok(()),
    }
}

/// Apply a trim; returns entries removed.
fn trim(cx: &mut Ctx, key: &[u8], s: &mut Stream, t: &TrimSpec) -> u64 {
    let mut n = match t.how {
        Trim::MaxLen(max) => s.m.len.saturating_sub(max),
        Trim::MinId(min) => match prev_id(min) {
            Some(below) => range(cx, key, ID_MIN, below, false, 0).len() as u64,
            None => 0,
        },
    };
    if t.approx {
        if t.limit > 0 {
            n = n.min(t.limit);
        }
        n = n / NODE * NODE; // whole nodes only
    }
    if n == 0 {
        return 0;
    }
    let victims = range(cx, key, ID_MIN, ID_MAX, false, n as usize);
    for (id, _) in &victims {
        cx.tx.delete(ent_key(key, *id));
    }
    s.m.len -= victims.len() as u64;
    victims.len() as u64
}

// ── Commands ───────────────────────────────────────────────────────────────

pub(super) fn is_stream_cmd(name: &str) -> bool {
    matches!(
        name,
        "XADD" | "XLEN" | "XRANGE" | "XREVRANGE" | "XDEL" | "XTRIM" | "XREAD" | "XREADGROUP" | "XGROUP" | "XACK"
            | "XPENDING" | "XCLAIM" | "XAUTOCLAIM" | "XINFO" | "XSETID"
    )
}

pub(super) fn is_stream_read(name: &str) -> bool {
    matches!(name, "XLEN" | "XRANGE" | "XREVRANGE" | "XREAD" | "XPENDING" | "XINFO")
}

pub(super) fn exec(cx: &mut Ctx, name: &str, a: &[Vec<u8>]) -> R<Resp> {
    let n = a.len();
    let need = |ok: bool| if ok { Ok(()) } else { Err(wrong_args(name)) };
    match name {
        "XADD" => {
            need(n >= 2)?;
            let key = &a[0];
            let (mut i, mut nomk, mut spec, mut limit) = (1, false, None, None);
            while i < n {
                if a[i].eq_ignore_ascii_case(b"NOMKSTREAM") {
                    nomk = true;
                    i += 1;
                } else if !parse_trim(a, &mut i, &mut spec, &mut limit)? {
                    break;
                }
            }
            finish_trim(&mut spec, limit)?;
            let id_arg = a.get(i).ok_or_else(|| wrong_args(name))?;
            // `*`, `<ms>-*`, or a full/partial explicit ID.
            let req: Option<(u64, Option<u64>)> = if id_arg == b"*" {
                None
            } else if let Some(ms) = id_arg.strip_suffix(b"-*") {
                Some((parse_u64(ms).ok_or_else(invalid_id)?, None))
            } else {
                let id = strict_id(id_arg)?;
                Some((id.0, Some(id.1)))
            };
            let fields = &a[i + 1..];
            need(!fields.is_empty() && fields.len() % 2 == 0)?;
            if let Some((0, Some(0))) = req {
                return Err(err("The ID specified in XADD must be greater than 0-0"));
            }
            let mut s = match load(cx, key)? {
                Some(s) => s,
                None if nomk => return Ok(Resp::Nil),
                None => new_stream(),
            };
            if s.last == ID_MAX {
                return Err(err("The stream has exhausted the last possible ID, unable to add more items"));
            }
            let too_small = || err("The ID specified in XADD is equal or smaller than the target stream top item");
            let id = match req {
                None => {
                    if cx.now > s.last.0 {
                        (cx.now, 0)
                    } else {
                        next_id(s.last).ok_or_else(too_small)?
                    }
                }
                Some((ms, None)) => {
                    if ms > s.last.0 {
                        (ms, 0)
                    } else if ms == s.last.0 && s.last.1 < u64::MAX {
                        (ms, s.last.1 + 1)
                    } else {
                        return Err(too_small());
                    }
                }
                Some((ms, Some(seq))) => {
                    if (ms, seq) <= s.last {
                        return Err(too_small());
                    }
                    (ms, seq)
                }
            };
            cx.tx.put(ent_key(key, id), Value::Bytes(enc_fields(fields)));
            s.m.len += 1;
            s.added += 1;
            s.last = id;
            if let Some(t) = &spec {
                trim(cx, key, &mut s, t);
            }
            save(cx, key, &mut s);
            Ok(id_resp(id))
        }
        "XLEN" => {
            need(n == 1)?;
            Ok(int(load(cx, &a[0])?.map_or(0, |s| s.m.len as i64)))
        }
        "XRANGE" | "XREVRANGE" => {
            need(n == 3 || n == 5)?;
            let rev = name == "XREVRANGE";
            let (sa, ea) = if rev { (&a[2], &a[1]) } else { (&a[1], &a[2]) };
            let (lo, hi) = (range_start(sa)?, range_end(ea)?);
            let mut count = 0usize;
            if n == 5 {
                if !a[3].eq_ignore_ascii_case(b"COUNT") {
                    return Err(err("syntax error"));
                }
                let c = parse_i64(&a[4]).ok_or_else(not_int)?;
                count = c.max(0) as usize;
                if load(cx, &a[0])?.is_none() {
                    return Ok(arr(vec![]));
                }
                if count == 0 {
                    return Ok(Resp::NilArray); // Redis: COUNT <= 0 on a stream is a null reply
                }
            }
            if load(cx, &a[0])?.is_none() {
                return Ok(arr(vec![]));
            }
            let (Some(lo), Some(hi)) = (lo, hi) else { return Ok(arr(vec![])) };
            Ok(arr(range(cx, &a[0], lo, hi, rev, count).into_iter().map(|(id, f)| entry_resp(id, f)).collect()))
        }
        "XDEL" => {
            need(n >= 2)?;
            // Redis 7.0: type check and missing key before ID validation.
            let Some(mut s) = load(cx, &a[0])? else { return Ok(int(0)) };
            let ids: Vec<Id> = a[1..].iter().map(|x| strict_id(x)).collect::<R<_>>()?;
            let mut deleted = 0;
            for id in ids {
                let k = ent_key(&a[0], id);
                if cx.tx.get(&k).is_some() {
                    cx.tx.delete(k);
                    s.m.len -= 1;
                    deleted += 1;
                    if id > s.maxdel {
                        s.maxdel = id;
                    }
                }
            }
            if deleted > 0 {
                save(cx, &a[0], &mut s);
            }
            Ok(int(deleted))
        }
        "XTRIM" => {
            need(n >= 3)?;
            let (mut i, mut spec, mut limit) = (1, None, None);
            while i < n {
                if !parse_trim(a, &mut i, &mut spec, &mut limit)? {
                    return Err(err("syntax error"));
                }
            }
            finish_trim(&mut spec, limit)?;
            let Some(spec) = spec else { return Err(err("syntax error")) };
            let Some(mut s) = load(cx, &a[0])? else { return Ok(int(0)) };
            let removed = trim(cx, &a[0], &mut s, &spec);
            if removed > 0 {
                save(cx, &a[0], &mut s);
            }
            Ok(int(removed as i64))
        }
        "XREAD" => {
            need(n >= 3)?;
            xread(cx, None, a)
        }
        "XREADGROUP" => {
            need(n >= 6)?;
            if !a[0].eq_ignore_ascii_case(b"GROUP") {
                return Err(err("syntax error"));
            }
            xread(cx, Some((a[1].clone(), a[2].clone())), &a[3..])
        }
        "XACK" => {
            need(n >= 3)?;
            // Redis 7.0: no key/group answers 0 before IDs are validated.
            if load(cx, &a[0])?.is_none() || load_group(cx, &a[0], &a[1]).is_none() {
                return Ok(int(0));
            }
            let ids: Vec<Id> = a[2..].iter().map(|x| strict_id(x)).collect::<R<_>>()?;
            let mut acked = 0;
            for id in ids {
                let k = pel_key(&a[0], &a[1], id);
                if cx.tx.get(&k).is_some() {
                    cx.tx.delete(k);
                    acked += 1;
                }
            }
            Ok(int(acked))
        }
        "XPENDING" => xpending(cx, a),
        "XCLAIM" => xclaim(cx, a),
        "XAUTOCLAIM" => xautoclaim(cx, a),
        "XGROUP" => xgroup(cx, a),
        "XINFO" => xinfo(cx, a),
        "XSETID" => {
            need(n >= 2)?;
            let id = strict_id(&a[1])?;
            let (mut added, mut maxdel) = (None, None);
            let mut i = 2;
            while i < n {
                let v = a.get(i + 1).ok_or_else(|| err("syntax error"))?;
                match String::from_utf8_lossy(&a[i]).to_ascii_uppercase().as_str() {
                    "ENTRIESADDED" => {
                        let e = parse_i64(v).ok_or_else(not_int)?;
                        if e < 0 {
                            return Err(err("entries_added must be positive"));
                        }
                        added = Some(e as u64);
                    }
                    "MAXDELETEDID" => maxdel = Some(strict_id(v)?),
                    _ => return Err(err("syntax error")),
                }
                i += 2;
            }
            let Some(mut s) = load(cx, &a[0])? else { return Err(err("no such key")) };
            if let Some(md) = maxdel {
                if id < md {
                    return Err(err("The ID specified in XSETID is smaller than the provided max_deleted_entry_id"));
                }
            }
            if let Some(e) = added {
                if e < s.m.len {
                    return Err(err("The entries_added specified in XSETID is smaller than the target stream length"));
                }
            }
            if s.m.len > 0 && id < s.last {
                return Err(err("The ID specified in XSETID is smaller than the target stream top item"));
            }
            s.last = id;
            if let Some(e) = added {
                s.added = e;
            }
            if let Some(md) = maxdel {
                s.maxdel = md;
            }
            save(cx, &a[0], &mut s);
            Ok(ok())
        }
        _ => Err(err(&format!("unknown command '{name}'"))),
    }
}

/// XREAD / XREADGROUP (BLOCK is parsed and ignored here; the server loops).
fn xread(cx: &mut Ctx, group: Option<(Vec<u8>, Vec<u8>)>, a: &[Vec<u8>]) -> R<Resp> {
    let cmd = if group.is_some() { "xreadgroup" } else { "xread" };
    let (mut count, mut noack, mut i) = (0usize, false, 0);
    let streams_at = loop {
        let Some(opt) = a.get(i) else { return Err(err("syntax error")) };
        match String::from_utf8_lossy(opt).to_ascii_uppercase().as_str() {
            "COUNT" => {
                let c = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(not_int)?;
                count = c.max(0) as usize;
                i += 2;
            }
            "BLOCK" => {
                let b = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(|| err("timeout is not an integer or out of range"))?;
                if b < 0 {
                    return Err(err("timeout is negative"));
                }
                i += 2;
            }
            "NOACK" if group.is_some() => {
                noack = true;
                i += 1;
            }
            "STREAMS" => break i + 1,
            _ => return Err(err("syntax error")),
        }
    };
    let rest = &a[streams_at..];
    if rest.is_empty() || rest.len() % 2 != 0 {
        return Err(err(&format!(
            "Unbalanced '{cmd}' list of streams: for each stream key an ID{} must be specified.",
            if group.is_some() { " or '>'" } else { " or '$'" }
        )));
    }
    let half = rest.len() / 2;
    let (keys, ids) = (&rest[..half], &rest[half..]);
    // Parse every ID (and check every group) before reading anything.
    enum Start {
        After(Id),
        New, // '>'
    }
    let mut starts = Vec::with_capacity(half);
    for (k, idv) in keys.iter().zip(ids) {
        // Redis order per key: type, then group, then the ID.
        if let Some((g, _)) = &group {
            if load(cx, k)?.is_none() || load_group(cx, k, g).is_none() {
                return Err(err(&format!(
                    "NOGROUP No such key '{}' or consumer group '{}' in XREADGROUP with GROUP option",
                    String::from_utf8_lossy(k),
                    String::from_utf8_lossy(g)
                )));
            }
        } else {
            load(cx, k)?;
        }
        let st = if idv == b">" {
            if group.is_none() {
                return Err(err("The > ID can be specified only when calling XREADGROUP using the GROUP <group> <consumer> option."));
            }
            Start::New
        } else if idv == b"$" {
            if group.is_some() {
                return Err(err("The $ ID is meaningless in the context of XREADGROUP: you want to read the history of this consumer by specifying a proper ID, or use the > ID to get new messages. The $ ID would just return an empty result set."));
            }
            Start::After(load(cx, k)?.map_or(ID_MIN, |s| s.last))
        } else if idv == b"+" && group.is_none() {
            // Redis 7.4: the last entry itself.
            load(cx, k)?;
            match range(cx, k, ID_MIN, ID_MAX, true, 1).first() {
                Some((last, _)) => prev_id(*last).map_or(Start::After(ID_MIN), Start::After),
                None => Start::After(ID_MAX),
            }
        } else {
            Start::After(strict_id(idv)?)
        };
        starts.push(st);
    }
    let mut out = Vec::new();
    for (k, st) in keys.iter().zip(starts) {
        let Some(mut s) = load(cx, k)? else { continue };
        match (&group, st) {
            (None, Start::After(after)) => {
                let Some(lo) = next_id(after) else { continue };
                let es = range(cx, k, lo, ID_MAX, false, count);
                if !es.is_empty() {
                    out.push(arr(vec![bulk(k.clone()), arr(es.into_iter().map(|(id, f)| entry_resp(id, f)).collect())]));
                }
            }
            (Some((g, c)), Start::New) => {
                let mut gr = load_group(cx, k, g).unwrap();
                let es = match next_id(gr.last) {
                    Some(lo) => range(cx, k, lo, ID_MAX, false, count),
                    None => Vec::new(),
                };
                touch_con(cx, k, g, c, !es.is_empty());
                if es.is_empty() {
                    continue;
                }
                for (id, _) in &es {
                    gr.last = *id;
                    if gr.read != -1 && !has_tombstones(cx, k, &s, *id) {
                        gr.read += 1;
                    } else if s.added > 0 {
                        gr.read = estimate_read(cx, k, &s, *id);
                    }
                    if !noack {
                        let p = Pel { time: cx.now, count: 1, owner: c.clone() };
                        cx.tx.put(pel_key(k, g, *id), Value::Bytes(p.enc()));
                    }
                }
                save_group(cx, k, g, gr);
                save(cx, k, &mut s);
                out.push(arr(vec![bulk(k.clone()), arr(es.into_iter().map(|(id, f)| entry_resp(id, f)).collect())]));
            }
            (Some((g, c)), Start::After(after)) => {
                // History: this consumer's pending entries after `after`.
                touch_con(cx, k, g, c, false);
                let Some(lo) = next_id(after) else {
                    out.push(arr(vec![bulk(k.clone()), arr(vec![])]));
                    continue;
                };
                let pend = pel_range(cx, k, g, lo, ID_MAX, count, |p| p.owner == *c);
                let mut es = Vec::new();
                for (id, mut p) in pend {
                    p.time = cx.now;
                    p.count += 1;
                    cx.tx.put(pel_key(k, g, id), Value::Bytes(p.enc()));
                    es.push(match entry(cx, k, id) {
                        Some(f) => entry_resp(id, f),
                        None => arr(vec![id_resp(id), Resp::NilArray]),
                    });
                }
                out.push(arr(vec![bulk(k.clone()), arr(es)]));
            }
            (None, Start::New) => unreachable!(),
        }
    }
    Ok(if out.is_empty() { Resp::NilArray } else { arr(out) })
}

fn xpending(cx: &mut Ctx, a: &[Vec<u8>]) -> R<Resp> {
    let n = a.len();
    if n < 2 {
        return Err(wrong_args("XPENDING"));
    }
    let (key, g) = (&a[0], &a[1]);
    // Extended form: [IDLE ms] start end count [consumer]
    let mut ext = None;
    if n > 2 {
        let mut i = 2;
        let mut idle = 0u64;
        if a[i].eq_ignore_ascii_case(b"IDLE") {
            let v = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(not_int)?;
            idle = v.max(0) as u64;
            i += 2;
        }
        if n - i != 3 && n - i != 4 {
            return Err(err("syntax error"));
        }
        let lo = range_start(&a[i])?;
        let hi = range_end(&a[i + 1])?;
        let cnt = parse_i64(&a[i + 2]).ok_or_else(not_int)?;
        let who = a.get(i + 3).cloned();
        ext = Some((idle, lo, hi, cnt, who));
    }
    if load(cx, key)?.is_none() || load_group(cx, key, g).is_none() {
        return Err(nogroup(key, g));
    }
    match ext {
        None => {
            let all = pel_range(cx, key, g, ID_MIN, ID_MAX, 0, |_| true);
            if all.is_empty() {
                return Ok(arr(vec![int(0), Resp::Nil, Resp::Nil, Resp::NilArray]));
            }
            let mut per: Vec<(Vec<u8>, u64)> = Vec::new();
            for (_, p) in &all {
                match per.iter_mut().find(|(o, _)| *o == p.owner) {
                    Some(e) => e.1 += 1,
                    None => per.push((p.owner.clone(), 1)),
                }
            }
            per.sort();
            Ok(arr(vec![
                int(all.len() as i64),
                id_resp(all[0].0),
                id_resp(all[all.len() - 1].0),
                arr(per.into_iter().map(|(o, c)| arr(vec![bulk(o), bulk(c.to_string().into_bytes())])).collect()),
            ]))
        }
        Some((idle, lo, hi, cnt, who)) => {
            let (Some(lo), Some(hi)) = (lo, hi) else { return Ok(arr(vec![])) };
            if cnt <= 0 {
                return Ok(arr(vec![]));
            }
            let now = cx.now;
            let got = pel_range(cx, key, g, lo, hi, cnt as usize, |p| {
                who.as_ref().is_none_or(|w| *w == p.owner) && now.saturating_sub(p.time) >= idle
            });
            Ok(arr(got
                .into_iter()
                .map(|(id, p)| arr(vec![id_resp(id), bulk(p.owner), int(now.saturating_sub(p.time) as i64), int(p.count as i64)]))
                .collect()))
        }
    }
}

fn xclaim(cx: &mut Ctx, a: &[Vec<u8>]) -> R<Resp> {
    let n = a.len();
    if n < 5 {
        return Err(wrong_args("XCLAIM"));
    }
    let (key, g, c) = (&a[0], &a[1], &a[2]);
    if load(cx, key)?.is_none() || load_group(cx, key, g).is_none() {
        return Err(nogroup(key, g));
    }
    let min_idle = parse_i64(&a[3]).ok_or_else(|| err("Invalid min-idle-time argument for XCLAIM"))?.max(0) as u64;
    let mut i = 4;
    let mut ids = Vec::new();
    while i < n {
        match parse_id(&a[i], 0) {
            Some(id) => ids.push(id),
            None => break,
        }
        i += 1;
    }
    let (mut time, mut retry, mut force, mut justid, mut lastid) = (None::<u64>, None::<u64>, false, false, None);
    while i < n {
        let opt = String::from_utf8_lossy(&a[i]).to_ascii_uppercase();
        match opt.as_str() {
            "FORCE" => force = true,
            "JUSTID" => justid = true,
            "IDLE" | "TIME" | "RETRYCOUNT" | "LASTID" => {
                let v = a.get(i + 1).ok_or_else(|| err("syntax error"))?;
                i += 1;
                match opt.as_str() {
                    "IDLE" => time = Some(cx.now.saturating_sub(parse_i64(v).ok_or_else(|| err("Invalid IDLE option argument for XCLAIM"))?.max(0) as u64)),
                    "TIME" => time = Some(parse_i64(v).ok_or_else(|| err("Invalid TIME option argument for XCLAIM"))?.max(0) as u64),
                    "RETRYCOUNT" => retry = Some(parse_i64(v).ok_or_else(|| err("Invalid RETRYCOUNT option argument for XCLAIM"))?.max(0) as u64),
                    _ => lastid = Some(strict_id(v)?),
                }
            }
            _ => return Err(err(&format!("Unrecognized XCLAIM option '{}'", String::from_utf8_lossy(&a[i])))),
        }
        i += 1;
    }
    let mut gr = load_group(cx, key, g).unwrap(); // checked above
    if let Some(l) = lastid {
        if l > gr.last {
            gr.last = l;
            save_group(cx, key, g, gr);
        }
    }
    let delivery = time.unwrap_or(cx.now);
    let mut out = Vec::new();
    let mut claimed_any = false;
    for id in ids {
        let pk = pel_key(key, g, id);
        let fields = entry(cx, key, id);
        let mut p = match load_pel(cx, key, g, id) {
            Some(p) => p,
            None if force && fields.is_some() => Pel { time: cx.now, count: 0, owner: c.clone() },
            None => continue,
        };
        let Some(fields) = fields else {
            cx.tx.delete(pk); // entry deleted from the stream: drop it
            continue;
        };
        if min_idle > 0 && cx.now.saturating_sub(p.time) < min_idle {
            continue;
        }
        p.owner = c.clone();
        p.time = delivery;
        if let Some(r) = retry {
            p.count = r;
        } else if !justid {
            p.count += 1;
        }
        cx.tx.put(pk, Value::Bytes(p.enc()));
        claimed_any = true;
        out.push(if justid { id_resp(id) } else { entry_resp(id, fields) });
    }
    touch_con(cx, key, g, c, claimed_any);
    Ok(arr(out))
}

fn xautoclaim(cx: &mut Ctx, a: &[Vec<u8>]) -> R<Resp> {
    let n = a.len();
    if n < 5 {
        return Err(wrong_args("XAUTOCLAIM"));
    }
    let (key, g, c) = (&a[0], &a[1], &a[2]);
    load(cx, key)?; // WRONGTYPE before argument errors
    let min_idle = parse_i64(&a[3]).ok_or_else(|| err("Invalid min-idle-time argument for XAUTOCLAIM"))?.max(0) as u64;
    let start = range_start(&a[4])?.unwrap_or(ID_MIN);
    let (mut count, mut justid, mut i) = (100usize, false, 5);
    while i < n {
        match String::from_utf8_lossy(&a[i]).to_ascii_uppercase().as_str() {
            "COUNT" => {
                let v = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(not_int)?;
                if v < 1 || v > (i64::MAX / 10) {
                    return Err(err("COUNT must be > 0"));
                }
                count = v as usize;
                i += 2;
            }
            "JUSTID" => {
                justid = true;
                i += 1;
            }
            _ => return Err(err("syntax error")),
        }
    }
    if load(cx, key)?.is_none() || load_group(cx, key, g).is_none() {
        return Err(nogroup(key, g));
    }
    // Scan at most count*10 pending entries (Redis's attempts bound).
    let scan = pel_range(cx, key, g, start, ID_MAX, count * 10 + 1, |_| true);
    let (mut claimed, mut deleted, mut next) = (Vec::new(), Vec::new(), ID_MIN);
    let mut attempts = count * 10;
    for (id, mut p) in scan {
        if claimed.len() >= count || attempts == 0 {
            next = id; // resume here next call; 0-0 = the scan reached the end
            break;
        }
        attempts -= 1;
        let pk = pel_key(key, g, id);
        let Some(fields) = entry(cx, key, id) else {
            cx.tx.delete(pk);
            deleted.push(id_resp(id));
            continue;
        };
        if cx.now.saturating_sub(p.time) < min_idle {
            continue;
        }
        p.owner = c.clone();
        p.time = cx.now;
        if !justid {
            p.count += 1;
        }
        cx.tx.put(pk, Value::Bytes(p.enc()));
        claimed.push(if justid { id_resp(id) } else { entry_resp(id, fields) });
    }
    touch_con(cx, key, g, c, !claimed.is_empty());
    Ok(arr(vec![id_resp(next), arr(claimed), arr(deleted)]))
}

fn parse_entries_read(a: &[Vec<u8>], i: usize) -> R<Option<i64>> {
    match a.get(i) {
        None => Ok(None),
        Some(o) if o.eq_ignore_ascii_case(b"ENTRIESREAD") => {
            let v = a.get(i + 1).and_then(|v| parse_i64(v)).ok_or_else(not_int)?;
            if v < -1 {
                return Err(err("value for ENTRIESREAD must be positive or -1"));
            }
            Ok(Some(v))
        }
        Some(_) => Err(err("syntax error")),
    }
}

fn xgroup(cx: &mut Ctx, a: &[Vec<u8>]) -> R<Resp> {
    let Some(sub) = a.first() else { return Err(wrong_args("XGROUP")) };
    let sub = String::from_utf8_lossy(sub).to_ascii_uppercase();
    let n = a.len();
    let bad = || err(&format!("unknown subcommand or wrong number of arguments for '{}'. Try XGROUP HELP.", sub.to_lowercase()));
    match sub.as_str() {
        "CREATE" => {
            if n < 4 {
                return Err(bad());
            }
            let (key, g) = (&a[1], &a[2]);
            let (mut mk, mut read, mut i) = (false, None, 4);
            while i < n {
                if a[i].eq_ignore_ascii_case(b"MKSTREAM") {
                    mk = true;
                    i += 1;
                } else {
                    read = parse_entries_read(a, i)?;
                    i += 2;
                }
            }
            let id_arg = &a[3];
            let explicit = if id_arg == b"$" { None } else { Some(strict_id(id_arg)?) };
            let mut s = match load(cx, key)? {
                Some(s) => s,
                None if mk => {
                    let mut s = new_stream();
                    save(cx, key, &mut s);
                    s
                }
                None => return Err(need_key()),
            };
            if load_group(cx, key, g).is_some() {
                return Err(err("BUSYGROUP Consumer Group name already exists"));
            }
            let id = explicit.unwrap_or(s.last);
            let read = read.unwrap_or_else(|| estimate_read(cx, key, &s, id));
            save_group(cx, key, g, Group { last: id, read });
            save(cx, key, &mut s);
            Ok(ok())
        }
        "SETID" => {
            if n < 4 {
                return Err(bad());
            }
            let (key, g) = (&a[1], &a[2]);
            let explicit = if a[3] == b"$" { None } else { Some(strict_id(&a[3])?) };
            let read = parse_entries_read(a, 4)?;
            let Some(s) = load(cx, key)? else { return Err(need_key()) };
            if load_group(cx, key, g).is_none() {
                return Err(err(&format!(
                    "NOGROUP No such consumer group '{}' for key name '{}'",
                    String::from_utf8_lossy(g),
                    String::from_utf8_lossy(key)
                )));
            }
            let id = explicit.unwrap_or(s.last);
            let read = read.unwrap_or_else(|| estimate_read(cx, key, &s, id));
            save_group(cx, key, g, Group { last: id, read });
            Ok(ok())
        }
        "DESTROY" => {
            if n != 3 {
                return Err(bad());
            }
            let (key, g) = (&a[1], &a[2]);
            if load(cx, key)?.is_none() {
                return Err(need_key());
            }
            if load_group(cx, key, g).is_none() {
                return Ok(int(0));
            }
            cx.tx.delete(group_key(key, g));
            delete_prefix(cx, ekey(key, SUB_XPEL, &group_part(g)));
            delete_prefix(cx, ekey(key, SUB_XCON, &group_part(g)));
            Ok(int(1))
        }
        "CREATECONSUMER" | "DELCONSUMER" => {
            if n != 4 {
                return Err(bad());
            }
            let (key, g, c) = (&a[1], &a[2], &a[3]);
            if load(cx, key)?.is_none() {
                return Err(need_key());
            }
            if load_group(cx, key, g).is_none() {
                return Err(err(&format!(
                    "NOGROUP No such consumer group '{}' for key name '{}'",
                    String::from_utf8_lossy(g),
                    String::from_utf8_lossy(key)
                )));
            }
            let exists = load_con(cx, key, g, c).is_some();
            if sub == "CREATECONSUMER" {
                if exists {
                    return Ok(int(0));
                }
                save_con(cx, key, g, c, cx.now, -1);
                return Ok(int(1));
            }
            if !exists {
                return Ok(int(0));
            }
            let pend = pel_range(cx, key, g, ID_MIN, ID_MAX, 0, |p| p.owner == *c);
            for (id, _) in &pend {
                cx.tx.delete(pel_key(key, g, *id));
            }
            cx.tx.delete(con_key(key, g, c));
            Ok(int(pend.len() as i64))
        }
        "HELP" => Ok(arr(
            [
                "XGROUP <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
                "CREATE <key> <groupname> <id|$> [option]",
                "CREATECONSUMER <key> <groupname> <consumer>",
                "DELCONSUMER <key> <groupname> <consumer>",
                "DESTROY <key> <groupname>",
                "SETID <key> <groupname> <id|$> [ENTRIESREAD entries_read]",
            ]
            .iter()
            .map(|l| Resp::Simple(l.to_string()))
            .collect(),
        )),
        _ => Err(err(&format!("unknown subcommand '{}'. Try XGROUP HELP.", String::from_utf8_lossy(&a[0])))),
    }
}

fn lag(cx: &mut Ctx, key: &[u8], s: &Stream, gr: &Group) -> Resp {
    if s.added == 0 {
        return int(0);
    }
    if gr.read != -1 && !has_tombstones(cx, key, s, gr.last) {
        return int(s.added as i64 - gr.read);
    }
    match estimate_read(cx, key, s, gr.last) {
        -1 => Resp::Nil,
        r => int(s.added as i64 - r),
    }
}

fn xinfo(cx: &mut Ctx, a: &[Vec<u8>]) -> R<Resp> {
    let Some(sub) = a.first() else { return Err(wrong_args("XINFO")) };
    let sub = String::from_utf8_lossy(sub).to_ascii_uppercase();
    let bad = || err(&format!("unknown subcommand or wrong number of arguments for '{}'. Try XINFO HELP.", sub.to_lowercase()));
    let key = a.get(1).ok_or_else(bad)?;
    let s = |cx: &mut Ctx| -> R<Stream> { load(cx, key)?.ok_or_else(|| err("no such key")) };
    let sb = |x: &str| bulk(x.as_bytes().to_vec());
    match sub.as_str() {
        "STREAM" => {
            if a.len() > 2 && !(a[2].eq_ignore_ascii_case(b"FULL")) {
                return Err(err("syntax error"));
            }
            let st = s(cx)?;
            let first = range(cx, key, ID_MIN, ID_MAX, false, 1).into_iter().next();
            let last = range(cx, key, ID_MIN, ID_MAX, true, 1).into_iter().next();
            let ngroups = groups(cx, key).len();
            Ok(arr(vec![
                sb("length"),
                int(st.m.len as i64),
                sb("radix-tree-keys"),
                int(st.m.len.div_ceil(NODE) as i64),
                sb("radix-tree-nodes"),
                int(st.m.len.div_ceil(NODE) as i64 + 1),
                sb("last-generated-id"),
                id_resp(st.last),
                sb("max-deleted-entry-id"),
                id_resp(st.maxdel),
                sb("entries-added"),
                int(st.added as i64),
                sb("recorded-first-entry-id"),
                id_resp(first.as_ref().map_or(ID_MIN, |e| e.0)),
                sb("groups"),
                int(ngroups as i64),
                sb("first-entry"),
                first.map_or(Resp::Nil, |(id, f)| entry_resp(id, f)),
                sb("last-entry"),
                last.map_or(Resp::Nil, |(id, f)| entry_resp(id, f)),
            ]))
        }
        "GROUPS" => {
            if a.len() != 2 {
                return Err(bad());
            }
            let st = s(cx)?;
            let mut out = Vec::new();
            for (name, gr) in groups(cx, key) {
                let ncons = consumers(cx, key, &name).len();
                let pending = pel_range(cx, key, &name, ID_MIN, ID_MAX, 0, |_| true).len();
                let lag = lag(cx, key, &st, &gr);
                out.push(arr(vec![
                    sb("name"),
                    bulk(name),
                    sb("consumers"),
                    int(ncons as i64),
                    sb("pending"),
                    int(pending as i64),
                    sb("last-delivered-id"),
                    id_resp(gr.last),
                    sb("entries-read"),
                    if gr.read == -1 { Resp::Nil } else { int(gr.read) },
                    sb("lag"),
                    lag,
                ]));
            }
            Ok(arr(out))
        }
        "CONSUMERS" => {
            if a.len() != 3 {
                return Err(bad());
            }
            let g = &a[2];
            s(cx)?;
            if load_group(cx, key, g).is_none() {
                return Err(err(&format!(
                    "NOGROUP No such consumer group '{}' for key name '{}'",
                    String::from_utf8_lossy(g),
                    String::from_utf8_lossy(key)
                )));
            }
            let pend = pel_range(cx, key, g, ID_MIN, ID_MAX, 0, |_| true);
            let now = cx.now;
            Ok(arr(consumers(cx, key, g)
                .into_iter()
                .map(|(name, seen, active)| {
                    let p = pend.iter().filter(|(_, p)| p.owner == name).count();
                    arr(vec![
                        sb("name"),
                        bulk(name),
                        sb("pending"),
                        int(p as i64),
                        sb("idle"),
                        int(now.saturating_sub(seen) as i64),
                        sb("inactive"),
                        int(if active < 0 { -1 } else { now.saturating_sub(active as u64) as i64 }),
                    ])
                })
                .collect()))
        }
        _ => Err(err(&format!("unknown subcommand '{}'. Try XINFO HELP.", String::from_utf8_lossy(&a[0])))),
    }
}

/// Current last-generated ID of `key` (0-0 if absent): lets the server
/// resolve `XREAD ... BLOCK ... $` once, before it starts waiting.
pub(super) fn last_id(cx: &mut Ctx, key: &[u8]) -> Vec<u8> {
    match load(cx, key) {
        Ok(Some(s)) => fmt_id(s.last),
        _ => b"0-0".to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_order_and_parse() {
        assert!(id_bytes((1, 5)) < id_bytes((1, 6)));
        assert!(id_bytes((1, u64::MAX)) < id_bytes((2, 0)));
        assert_eq!(id_from(&id_bytes((7, 9))), (7, 9));
        assert_eq!(parse_id(b"5", 0), Some((5, 0)));
        assert_eq!(parse_id(b"5", u64::MAX), Some((5, u64::MAX)));
        assert_eq!(parse_id(b"5-3", 0), Some((5, 3)));
        assert_eq!(parse_id(b"5-", 0), None);
        assert_eq!(parse_id(b"x", 0), None);
        assert_eq!(next_id((1, u64::MAX)), Some((2, 0)));
        assert_eq!(prev_id((2, 0)), Some((1, u64::MAX)));
        assert_eq!(prev_id(ID_MIN), None);
    }

    #[test]
    fn fields_roundtrip() {
        let f = vec![b"a".to_vec(), b"".to_vec(), b"long value".to_vec(), vec![0, 255]];
        assert_eq!(dec_fields(&enc_fields(&f)), f);
    }
}
