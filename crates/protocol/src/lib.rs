//! Protocol layer — RESP (REdis Serialization Protocol) v2 parser + encoder.
//! Pure Rust, zero deps. This is the Redis wire shim from the architecture; it
//! lets any Redis client talk to DB-Strike's KV/counter/vector commands.

use std::io::{self, BufRead, Write};

/// A parsed RESP value.
#[derive(Clone, Debug, PartialEq)]
pub enum Resp {
    Simple(String),
    Error(String),
    Int(i64),
    Bulk(Vec<u8>),
    Nil,
    Array(Vec<Resp>),
    /// RESP3 map (`%`) — flat k1 v1 k2 v2 ... Used only for HELLO when the
    /// client negotiates protocol 3; everything else stays RESP2.
    Map(Vec<Resp>),
    /// Out-of-band push (pub/sub messages, subscribe acks): RESP3 `>` on a
    /// connection that negotiated proto 3, a plain `*` array on RESP2.
    Push(Vec<Resp>),
}

impl Resp {
    /// Encode to wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out, false);
        out
    }

    /// Encode for a connection that negotiated `resp3`: nulls at ANY depth
    /// (e.g. the misses inside an MGET array) become `_`, not `$-1`.
    pub fn encode_as(&self, resp3: bool) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out, resp3);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>, resp3: bool) {
        match self {
            Resp::Simple(s) => {
                out.push(b'+');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Resp::Error(s) => {
                out.push(b'-');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Resp::Int(i) => {
                out.push(b':');
                out.extend_from_slice(i.to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            Resp::Bulk(b) => {
                out.push(b'$');
                out.extend_from_slice(b.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                out.extend_from_slice(b);
                out.extend_from_slice(b"\r\n");
            }
            Resp::Nil if resp3 => out.extend_from_slice(b"_\r\n"),
            Resp::Nil => {
                out.extend_from_slice(b"$-1\r\n");
            }
            Resp::Array(items) => {
                out.push(b'*');
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for it in items {
                    it.encode_into(out, resp3);
                }
            }
            // RESP3 map type (`%`). Used ONLY by the HELLO handshake when the
            // client negotiates protocol 3 (redis-py >= 8 defaults to 3 and
            // requires a real map reply). The rest of the wire stays RESP2 —
            // which is legal because RESP2 frames (+ - : $ *) are a strict
            // subset of RESP3, so a client that switched parsers still reads
            // every other reply we emit. `items` is flat: k1 v1 k2 v2 ...
            Resp::Push(items) => {
                out.push(if resp3 { b'>' } else { b'*' });
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for it in items {
                    it.encode_into(out, resp3);
                }
            }
            Resp::Map(items) => {
                let pairs = items.len() / 2;
                out.push(b'%');
                out.extend_from_slice(pairs.to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for it in items {
                    it.encode_into(out, resp3);
                }
            }
        }
    }
}

/// Largest bulk string accepted from a client (Redis `proto-max-bulk-len`).
pub const MAX_BULK_LEN: usize = 512 * 1024 * 1024;
/// Largest multibulk argument count accepted from a client.
pub const MAX_ARGS: usize = 1024 * 1024;
/// Longest header / inline line accepted before the terminating `\n`.
const MAX_LINE: usize = 64 * 1024;

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Index of the `\n` ending the line that starts at `from`, or `None` if it
/// has not arrived yet. A line that grows past `MAX_LINE` without a newline is
/// a protocol error, so a client cannot make us buffer an unbounded header.
fn find_line_end(buf: &[u8], from: usize) -> io::Result<Option<usize>> {
    match buf[from..].iter().position(|&b| b == b'\n') {
        Some(i) => Ok(Some(from + i)),
        None if buf.len() - from > MAX_LINE => Err(invalid("too big inline request")),
        None => Ok(None),
    }
}

/// Parse a `*`/`$` length header (sans prefix). `-1` means null → `None`.
fn parse_len(hdr: &[u8], what: &str) -> io::Result<Option<usize>> {
    let hdr = if hdr.last() == Some(&b'\r') { &hdr[..hdr.len() - 1] } else { hdr };
    let n: i64 = std::str::from_utf8(hdr)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| invalid(what))?;
    match n {
        -1 => Ok(None),
        n if n < 0 => Err(invalid(what)),
        n => usize::try_from(n).map(Some).map_err(|_| invalid(what)),
    }
}

/// Try to parse one command from a byte slice WITHOUT blocking. Returns
/// `Ok(Some((cmd, consumed)))` if a full command is present,
/// `Ok(None)` if the buffer is incomplete (need more bytes), or
/// `Err` on malformed input.
///
/// Used by the server's pipelined dispatch loop to drain every complete
/// command already sitting in the socket buffer in one go — the "batched
/// command I/O" trick that unlocks Redis-scale throughput on pipelined
/// connections (`redis-benchmark -P N`).
pub fn try_parse(buf: &[u8]) -> io::Result<Option<(Vec<Vec<u8>>, usize)>> {
    if buf.is_empty() {
        return Ok(None);
    }
    // RESP array form: `*N\r\n$L1\r\nBULK1\r\n$L2\r\nBULK2\r\n...`
    //
    // Every count and length below comes straight off the wire from an
    // unauthenticated client, and the release profile is `panic = "abort"`,
    // so a single overflow or over-allocation here would take down the whole
    // process. All sizes are bounded and all arithmetic is checked.
    if buf[0] == b'*' {
        let nl = match find_line_end(buf, 0)? {
            Some(i) => i,
            None => return Ok(None), // need more
        };
        let count = parse_len(&buf[1..nl], "bad array header")?;
        let mut pos = nl + 1;
        // `*-1` / `*0` are legal null/empty arrays: consume and ignore.
        let count = match count {
            None => return Ok(Some((Vec::new(), pos))),
            Some(c) if c > MAX_ARGS => {
                return Err(invalid("invalid multibulk length"));
            }
            Some(c) => c,
        };
        // Never trust `count` for the allocation: each arg needs at least
        // 4 bytes (`$0\r\n` + CRLF), so the buffer bounds the useful capacity.
        let mut args = Vec::with_capacity(count.min(buf.len() / 4 + 1));
        for _ in 0..count {
            if pos >= buf.len() {
                return Ok(None);
            }
            if buf[pos] != b'$' {
                return Err(invalid("expected bulk string"));
            }
            let bnl = match find_line_end(buf, pos)? {
                Some(i) => i,
                None => return Ok(None),
            };
            let len = match parse_len(&buf[pos + 1..bnl], "bad bulk len")? {
                Some(l) if l <= MAX_BULK_LEN => l,
                _ => return Err(invalid("invalid bulk length")),
            };
            pos = bnl + 1;
            // Need `len` bytes + trailing \r\n. `len` is bounded above, so
            // this cannot overflow, but stay checked anyway.
            let end = pos.checked_add(len).and_then(|e| e.checked_add(2))
                .ok_or_else(|| invalid("invalid bulk length"))?;
            if end > buf.len() {
                return Ok(None);
            }
            if &buf[pos + len..end] != b"\r\n" {
                return Err(invalid("bulk string not terminated by CRLF"));
            }
            args.push(buf[pos..pos + len].to_vec());
            pos = end;
        }
        Ok(Some((args, pos)))
    } else {
        // Inline: read up to `\n`, split on spaces.
        let nl = match find_line_end(buf, 0)? {
            Some(i) => i,
            None => return Ok(None),
        };
        let mut line = &buf[..nl];
        if line.last() == Some(&b'\r') { line = &line[..line.len() - 1]; }
        let args: Vec<Vec<u8>> = line
            .split(|&b| b == b' ')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_vec())
            .collect();
        Ok(Some((args, nl + 1)))
    }
}

/// Write a RESP reply into a buffered writer WITHOUT flushing. Callers batch
/// many replies then call `flush` once — this alone gives a 5-10× throughput
/// bump on pipelined workloads by cutting the per-reply syscall.
pub fn write_resp_buf<W: Write>(w: &mut W, resp: &Resp) -> io::Result<()> {
    w.write_all(&resp.encode())
}

/// RESP3-aware variant. Once a client negotiates protocol 3 via HELLO, null
/// replies MUST be the RESP3 null (`_`), not the legacy RESP2 `$-1`: strict
/// RESP3 parsers (e.g. redis-py >= 8) have no `$-1` case and block forever
/// trying to read `-1` bulk bytes. Everything else encodes identically.
pub fn write_resp_buf_as<W: Write>(w: &mut W, resp: &Resp, resp3: bool) -> io::Result<()> {
    w.write_all(&resp.encode_as(resp3))
}

/// Parse one client command from a buffered reader.
/// Supports the RESP array-of-bulk-strings form used by real clients, plus the
/// inline command form (space-separated, newline-terminated) for `nc`/telnet.
pub fn read_command<R: BufRead>(reader: &mut R) -> io::Result<Option<Vec<Vec<u8>>>> {
    let mut first = Vec::new();
    let n = reader.read_until(b'\n', &mut first)?;
    if n == 0 {
        return Ok(None); // connection closed
    }
    // strip trailing \r\n
    while matches!(first.last(), Some(b'\n') | Some(b'\r')) {
        first.pop();
    }
    if first.is_empty() {
        return Ok(Some(Vec::new()));
    }

    if first[0] == b'*' {
        // RESP array of bulk strings
        let count = match parse_len(&first[1..], "bad array header")? {
            None => return Ok(Some(Vec::new())),
            Some(c) if c > MAX_ARGS => return Err(invalid("invalid multibulk length")),
            Some(c) => c,
        };
        let mut args = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let mut hdr = Vec::new();
            reader.read_until(b'\n', &mut hdr)?;
            while matches!(hdr.last(), Some(b'\n') | Some(b'\r')) {
                hdr.pop();
            }
            if hdr.first() != Some(&b'$') {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "expected bulk string"));
            }
            let len = match parse_len(&hdr[1..], "bad bulk len")? {
                Some(l) if l <= MAX_BULK_LEN => l,
                _ => return Err(invalid("invalid bulk length")),
            };
            let mut buf = vec![0u8; len + 2]; // include trailing \r\n
            reader.read_exact(&mut buf)?;
            buf.truncate(len);
            args.push(buf);
        }
        Ok(Some(args))
    } else {
        // Inline command
        let args = first
            .split(|&b| b == b' ')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_vec())
            .collect();
        Ok(Some(args))
    }
}

/// Write a RESP value to a stream.
pub fn write_resp<W: Write>(w: &mut W, resp: &Resp) -> io::Result<()> {
    w.write_all(&resp.encode())?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    /// Hostile headers that used to abort the server (capacity overflow /
    /// usize wraparound with `panic = "abort"`) must be protocol errors.
    #[test]
    fn hostile_lengths_are_errors_not_panics() {
        for frame in [
            &b"*99999999999999\r\n"[..],
            b"*1\r\n$18446744073709551615\r\n",
            b"*1\r\n$18446744073709551610\r\nab",
            b"*1\r\n$-5\r\n",
            b"*1\r\n$999999999999\r\n",
        ] {
            assert!(try_parse(frame).is_err(), "{:?}", String::from_utf8_lossy(frame));
        }
    }

    #[test]
    fn null_array_and_crlf_check() {
        assert_eq!(try_parse(b"*-1\r\n").unwrap(), Some((Vec::new(), 5)));
        assert!(try_parse(b"*1\r\n$2\r\nabXY").is_err());
        assert_eq!(try_parse(b"*1\r\n$2\r\nab").unwrap(), None);
    }

    #[test]
    fn unterminated_header_is_bounded() {
        let mut big = b"*1\r\n$".to_vec();
        big.extend(std::iter::repeat(b'1').take(MAX_LINE + 10));
        assert!(try_parse(&big).is_err());
    }

    #[test]
    fn resp3_nested_nulls() {
        let r = Resp::Array(vec![Resp::Bulk(b"a".to_vec()), Resp::Nil]);
        assert_eq!(r.encode_as(true), b"*2\r\n$1\r\na\r\n_\r\n".to_vec());
        assert_eq!(r.encode_as(false), b"*2\r\n$1\r\na\r\n$-1\r\n".to_vec());
    }

    /// A VADDBATCH-sized frame (64 vectors × 384 dims ≈ 447 KB, 24642 bulk
    /// args) delivered the way a real socket delivers it: in 32 KB chunks, the
    /// same size as the server's `tmp` read buffer. Every prefix must parse as
    /// `Ok(None)` (incomplete) and the whole thing must parse exactly once.
    ///
    /// This is the regression test for the s13 ingest failure: the bench got
    /// `-ERR Protocol error: expected bulk string` on every VADDBATCH, i.e. the
    /// parser mistook a truncated frame for a malformed one. VSEARCH never hit
    /// it because a single query fits in one read.
    #[test]
    fn chunked_large_frame_parses_once() {
        const DIM: usize = 384;
        const NVEC: usize = 64;
        let n_args = 2 + NVEC * (1 + DIM);
        let mut frame: Vec<u8> = format!("*{n_args}\r\n").into_bytes();
        let mut push_bulk = |f: &mut Vec<u8>, s: &str| {
            f.extend_from_slice(format!("${}\r\n{s}\r\n", s.len()).as_bytes());
        };
        push_bulk(&mut frame, "VADDBATCH");
        push_bulk(&mut frame, &DIM.to_string());
        for i in 0..NVEC {
            push_bulk(&mut frame, &i.to_string());
            for j in 0..DIM {
                // Vary the float text length (1..12 bytes) so bulk headers land
                // at irregular offsets and chunk boundaries fall mid-header,
                // mid-payload and mid-CRLF across the run.
                let v = (i * DIM + j) as f32 * 0.000_123_45;
                push_bulk(&mut frame, &format!("{v}"));
            }
        }
        assert!(frame.len() > 300_000, "frame should be a few hundred KB, got {}", frame.len());

        let mut buf: Vec<u8> = Vec::new();
        let mut parsed = 0;
        for chunk in frame.chunks(32 * 1024) {
            buf.extend_from_slice(chunk);
            loop {
                match try_parse(&buf) {
                    Ok(Some((cmd, consumed))) => {
                        assert_eq!(cmd.len(), n_args);
                        assert_eq!(cmd[0], b"VADDBATCH");
                        buf.drain(..consumed);
                        parsed += 1;
                    }
                    Ok(None) => break,
                    Err(e) => panic!(
                        "truncated frame misreported as malformed at {} / {} bytes: {e}",
                        buf.len(),
                        frame.len()
                    ),
                }
            }
        }
        assert_eq!(parsed, 1, "expected exactly one complete command");
        assert!(buf.is_empty(), "{} bytes left over", buf.len());
    }

    #[test]
    fn encode_types() {
        assert_eq!(Resp::Simple("OK".into()).encode(), b"+OK\r\n");
        assert_eq!(Resp::Int(42).encode(), b":42\r\n");
        assert_eq!(Resp::Bulk(b"hi".to_vec()).encode(), b"$2\r\nhi\r\n");
        assert_eq!(Resp::Nil.encode(), b"$-1\r\n");
        let arr = Resp::Array(vec![Resp::Int(1), Resp::Bulk(b"x".to_vec())]);
        assert_eq!(arr.encode(), b"*2\r\n:1\r\n$1\r\nx\r\n");
    }

    #[test]
    fn parse_resp_array() {
        let input = b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$3\r\nbar\r\n";
        let mut r = BufReader::new(&input[..]);
        let cmd = read_command(&mut r).unwrap().unwrap();
        assert_eq!(cmd, vec![b"SET".to_vec(), b"foo".to_vec(), b"bar".to_vec()]);
    }

    #[test]
    fn parse_inline() {
        let input = b"PING\r\n";
        let mut r = BufReader::new(&input[..]);
        let cmd = read_command(&mut r).unwrap().unwrap();
        assert_eq!(cmd, vec![b"PING".to_vec()]);
    }
}

