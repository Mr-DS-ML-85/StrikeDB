//! Tiny shared RESP client for the DB-Strike demo apps. No external crates.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub enum Reply {
    Simple(Vec<u8>),
    Error(Vec<u8>),
    Int(i64),
    Bulk(Option<Vec<u8>>),
    Array(Vec<Reply>),
}

impl Reply {
    pub fn as_bulk(&self) -> Option<&[u8]> {
        if let Reply::Bulk(Some(b)) = self { Some(b) } else { None }
    }
    pub fn as_int(&self) -> Option<i64> {
        if let Reply::Int(i) = self { Some(*i) } else { None }
    }
    pub fn as_array(&self) -> Option<&[Reply]> {
        if let Reply::Array(v) = self { Some(v) } else { None }
    }
    pub fn as_str(&self) -> Option<String> {
        match self {
            Reply::Simple(b) | Reply::Bulk(Some(b)) => {
                Some(String::from_utf8_lossy(b).to_string())
            }
            Reply::Int(i) => Some(i.to_string()),
            _ => None,
        }
    }
}

pub struct Client {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Client {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        Ok(Self { stream: s, buf: Vec::with_capacity(4096) })
    }
    pub fn set_read_timeout(&mut self, d: Option<Duration>) -> std::io::Result<()> {
        self.stream.set_read_timeout(d)
    }
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self { stream: self.stream.try_clone()?, buf: Vec::with_capacity(4096) })
    }
    pub fn send(&mut self, args: &[&[u8]]) -> std::io::Result<()> {
        let mut out = format!("*{}\r\n", args.len()).into_bytes();
        for a in args {
            out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
            out.extend_from_slice(a);
            out.extend_from_slice(b"\r\n");
        }
        self.stream.write_all(&out)
    }
    fn read_line(&mut self) -> std::io::Result<Vec<u8>> {
        loop {
            if let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
                while matches!(line.last(), Some(b'\r') | Some(b'\n')) {
                    line.pop();
                }
                return Ok(line);
            }
            let mut tmp = [0u8; 4096];
            let n = self.stream.read(&mut tmp)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof, "server closed"));
            }
            self.buf.extend_from_slice(&tmp[..n]);
        }
    }
    fn read_n(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        while self.buf.len() < n {
            let mut tmp = [0u8; 4096];
            let g = self.stream.read(&mut tmp)?;
            if g == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof, "server closed"));
            }
            self.buf.extend_from_slice(&tmp[..g]);
        }
        Ok(self.buf.drain(..n).collect())
    }
    pub fn read_reply(&mut self) -> std::io::Result<Reply> {
        let line = self.read_line()?;
        let (t, rest) = (line.first().copied().unwrap_or(0), &line[1..]);
        Ok(match t {
            b'+' => Reply::Simple(rest.to_vec()),
            b'-' => Reply::Error(rest.to_vec()),
            b':' => {
                let s = std::str::from_utf8(rest).unwrap_or("0");
                Reply::Int(s.trim().parse().unwrap_or(0))
            }
            b'$' => {
                let n: i64 = std::str::from_utf8(rest).ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(-1);
                if n < 0 {
                    Reply::Bulk(None)
                } else {
                    let mut buf = self.read_n(n as usize + 2)?;
                    buf.truncate(n as usize);
                    Reply::Bulk(Some(buf))
                }
            }
            b'*' => {
                let n: i64 = std::str::from_utf8(rest).ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let mut v = Vec::with_capacity(n.max(0) as usize);
                for _ in 0..n.max(0) {
                    v.push(self.read_reply()?);
                }
                Reply::Array(v)
            }
            _ => Reply::Simple(Vec::new()),
        })
    }
    pub fn cmd(&mut self, args: &[&[u8]]) -> std::io::Result<Reply> {
        self.send(args)?;
        self.read_reply()
    }
}
