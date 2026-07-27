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
}

impl Resp {
    /// Encode to wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
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
            Resp::Nil => {
                out.extend_from_slice(b"$-1\r\n");
            }
            Resp::Array(items) => {
                out.push(b'*');
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for it in items {
                    it.encode_into(out);
                }
            }
        }
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
    if buf[0] == b'*' {
        // find '\n' of the array header
        let nl = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => i,
            None => return Ok(None), // need more
        };
        let hdr = &buf[1..nl];
        // strip trailing \r if present
        let hdr = if hdr.last() == Some(&b'\r') { &hdr[..hdr.len() - 1] } else { hdr };
        let count: usize = std::str::from_utf8(hdr)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad array header"))?;
        let mut pos = nl + 1;
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            if pos >= buf.len() {
                return Ok(None);
            }
            if buf[pos] != b'$' {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "expected bulk string"));
            }
            let bnl = match buf[pos..].iter().position(|&b| b == b'\n') {
                Some(i) => pos + i,
                None => return Ok(None),
            };
            let bhdr = &buf[pos + 1..bnl];
            let bhdr = if bhdr.last() == Some(&b'\r') { &bhdr[..bhdr.len() - 1] } else { bhdr };
            let len: usize = std::str::from_utf8(bhdr)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad bulk len"))?;
            pos = bnl + 1;
            // Need `len` bytes + trailing \r\n
            if pos + len + 2 > buf.len() {
                return Ok(None);
            }
            args.push(buf[pos..pos + len].to_vec());
            pos += len + 2;
        }
        Ok(Some((args, pos)))
    } else {
        // Inline: read up to `\n`, split on spaces.
        let nl = match buf.iter().position(|&b| b == b'\n') {
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
        let count: usize = std::str::from_utf8(&first[1..])
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad array header"))?;
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            let mut hdr = Vec::new();
            reader.read_until(b'\n', &mut hdr)?;
            while matches!(hdr.last(), Some(b'\n') | Some(b'\r')) {
                hdr.pop();
            }
            if hdr.first() != Some(&b'$') {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "expected bulk string"));
            }
            let len: usize = std::str::from_utf8(&hdr[1..])
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad bulk len"))?;
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

