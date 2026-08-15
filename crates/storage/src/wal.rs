//! Write-Ahead Log — append-only, crash-safe. Pure std.
//!
//! Record framing on disk:
//!   [u32 len][u32 crc(payload)][payload bytes]
//! len/crc are little-endian. On recovery we stop at the first record whose
//! length runs past EOF or whose CRC fails (a torn tail), and truncate there.

use crate::crc::crc32;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct Wal {
    file: File,
    path: PathBuf,
}

impl Wal {
    /// Open (or create) the WAL at `path`, seeking to the end for appends.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        Ok(Self { file, path })
    }

    /// Append one record and fsync so it survives a crash.
    pub fn append(&mut self, payload: &[u8]) -> io::Result<()> {
        let len = payload.len() as u32;
        let crc = crc32(payload);
        let mut hdr = [0u8; 8];
        hdr[0..4].copy_from_slice(&len.to_le_bytes());
        hdr[4..8].copy_from_slice(&crc.to_le_bytes());
        self.file.write_all(&hdr)?;
        self.file.write_all(payload)?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Append one record WITHOUT fsync. The caller is responsible for an
    /// eventual `sync()` — the group-commit flusher appends a whole batch
    /// then fsyncs once. Per-record fsyncs here turn a 400k-record bulk load
    /// into 400k fsyncs (minutes), which is why batch writers must use this.
    pub fn append_unsynced(&mut self, payload: &[u8]) -> io::Result<()> {
        let len = payload.len() as u32;
        let crc = crc32(payload);
        let mut hdr = [0u8; 8];
        hdr[0..4].copy_from_slice(&len.to_le_bytes());
        hdr[4..8].copy_from_slice(&crc.to_le_bytes());
        self.file.write_all(&hdr)?;
        self.file.write_all(payload)?;
        Ok(())
    }

    /// Flush all buffered data to durable storage (used by the group-commit
    /// flusher to amortise ONE fsync across a whole batch of records).
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// Replay every intact record from the start of the log.
    /// Stops cleanly at a torn tail and truncates the file to the last good byte.
    pub fn replay(&mut self) -> io::Result<Vec<Vec<u8>>> {
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut out = Vec::new();
        let mut good_offset: u64 = 0;

        loop {
            let mut hdr = [0u8; 8];
            match reader.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
            let want_crc = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);

            let mut payload = vec![0u8; len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // torn tail
                Err(e) => return Err(e),
            }
            if crc32(&payload) != want_crc {
                break; // corrupt tail — stop here
            }
            good_offset += (8 + len) as u64;
            out.push(payload);
        }

        // Truncate any torn/corrupt tail so future appends stay aligned.
        let f = OpenOptions::new().write(true).open(&self.path)?;
        f.set_len(good_offset)?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(out)
    }

    /// Truncate the whole log (used after a checkpoint/snapshot).
    pub fn truncate(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_data()
    }

    /// Current log size in bytes (used to decide whether a checkpoint is worth it).
    pub fn len(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_replay() {
        let dir = std::env::temp_dir().join(format!("dbstrike_wal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.wal");
        let _ = std::fs::remove_file(&path);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(b"hello").unwrap();
            wal.append(b"world").unwrap();
        }
        let mut wal = Wal::open(&path).unwrap();
        let recs = wal.replay().unwrap();
        assert_eq!(recs, vec![b"hello".to_vec(), b"world".to_vec()]);
    }

    #[test]
    fn torn_tail_is_dropped() {
        let dir = std::env::temp_dir().join(format!("dbstrike_wal2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("torn.wal");
        let _ = std::fs::remove_file(&path);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(b"good").unwrap();
        }
        // Append a bogus header claiming a huge payload that isn't there.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&999u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(b"xx").unwrap();
        }
        let mut wal = Wal::open(&path).unwrap();
        let recs = wal.replay().unwrap();
        assert_eq!(recs, vec![b"good".to_vec()]);
    }
}
