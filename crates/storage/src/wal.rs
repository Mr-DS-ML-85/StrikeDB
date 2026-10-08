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
    /// Byte length of the intact, fully-written prefix. Every append either
    /// extends it or truncates the file back to it, so a failed write can
    /// never leave a partial frame IN FRONT of later records. (Previously a
    /// short write — e.g. ENOSPC — left garbage mid-log; later acked records
    /// were appended after it, and the next replay stopped at the garbage and
    /// truncated every one of them away.)
    good_len: u64,
    /// Set when an fsync fails. On Linux a failed fsync may already have
    /// dropped the dirty pages and a retry can falsely report success
    /// ("fsyncgate"), so durability can no longer be promised: every later
    /// write is refused until the process restarts and replays from disk.
    poisoned: bool,
}

/// Largest payload one frame can carry (the length field is a u32).
pub const MAX_FRAME: usize = u32::MAX as usize;

fn frame_header(payload: &[u8]) -> io::Result<[u8; 8]> {
    if payload.len() > MAX_FRAME {
        // `len as u32` used to wrap silently, writing a frame whose header
        // disagreed with its body; replay then failed its CRC and truncated
        // everything after it.
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("WAL frame of {} bytes exceeds the 4 GiB frame limit", payload.len()),
        ));
    }
    let mut hdr = [0u8; 8];
    hdr[0..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    hdr[4..8].copy_from_slice(&crc32(payload).to_le_bytes());
    Ok(hdr)
}

/// fsync the directory containing `path`, making a create/rename durable.
pub(crate) fn sync_parent_dir(path: &Path) {
    if let Some(dir) = path.parent() {
        let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
}

impl Wal {
    /// Open (or create) the WAL at `path`, seeking to the end for appends.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let existed = path.exists();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        if !existed {
            // A freshly created log's directory entry must be durable too, or
            // a power cut can lose the whole file along with its acked writes.
            sync_parent_dir(&path);
        }
        let good_len = file.metadata()?.len();
        Ok(Self { file, path, good_len, poisoned: false })
    }

    fn check_poisoned(&self) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "WAL disabled after an fsync failure; restart the server to recover",
            ));
        }
        Ok(())
    }

    /// Append one record and fsync so it survives a crash.
    pub fn append(&mut self, payload: &[u8]) -> io::Result<()> {
        self.append_frames(std::slice::from_ref(&payload.to_vec()))?;
        self.sync()
    }

    /// Append one record WITHOUT fsync (caller syncs).
    pub fn append_unsynced(&mut self, payload: &[u8]) -> io::Result<()> {
        self.append_frames(std::slice::from_ref(&payload.to_vec()))
    }

    /// Append several frames as ONE contiguous write, WITHOUT fsync. All or
    /// nothing: on any error the file is truncated back to the last intact
    /// byte, so the log stays a clean sequence of whole frames. The
    /// group-commit flusher writes an entire drained batch through this.
    pub fn append_frames(&mut self, payloads: &[Vec<u8>]) -> io::Result<()> {
        self.check_poisoned()?;
        let total: usize = payloads.iter().map(|p| p.len() + 8).sum();
        let mut buf = Vec::with_capacity(total);
        for p in payloads {
            buf.extend_from_slice(&frame_header(p)?);
            buf.extend_from_slice(p);
        }
        match self.file.write_all(&buf) {
            Ok(()) => {
                self.good_len += buf.len() as u64;
                Ok(())
            }
            Err(e) => {
                self.rollback();
                Err(e)
            }
        }
    }

    /// Discard anything past `good_len` (a partially written batch).
    fn rollback(&mut self) {
        if self.file.set_len(self.good_len).is_err() {
            // Can't even truncate: refuse further writes rather than risk
            // appending after a torn frame.
            self.poisoned = true;
        }
        let _ = self.file.seek(SeekFrom::End(0));
    }

    /// Flush all buffered data to durable storage (used by the group-commit
    /// flusher to amortise ONE fsync across a whole batch of records).
    pub fn sync(&mut self) -> io::Result<()> {
        self.check_poisoned()?;
        self.file.sync_data().map_err(|e| {
            self.poisoned = true;
            e
        })
    }

    /// Replay every intact record from the start of the log.
    pub fn replay(&mut self) -> io::Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        self.replay_with(|rec| out.push(rec))?;
        Ok(out)
    }

    /// Stream every intact record to `f`, in order, without holding the whole
    /// log in memory. Stops at the first frame that is torn (runs past EOF) or
    /// fails its CRC. Bytes past that point are copied to
    /// `<wal>.corrupt-<unix ms>` BEFORE the log is truncated, with a warning:
    /// a single flipped bit mid-log used to silently discard every later
    /// (valid, acked) record with no way to recover them.
    pub fn replay_with<F: FnMut(Vec<u8>)>(&mut self, mut f: F) -> io::Result<()> {
        let file_len = std::fs::metadata(&self.path)?.len();
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut good_offset: u64 = 0;
        let mut reason = "";

        loop {
            let mut hdr = [0u8; 8];
            match reader.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if good_offset < file_len {
                        reason = "torn frame header";
                    }
                    break;
                }
                Err(e) => return Err(e),
            }
            let len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as u64;
            let want_crc = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
            // Never allocate from an unverified length: a corrupt header could
            // claim up to 4 GiB. A frame can't extend past the file anyway.
            if good_offset + 8 + len > file_len {
                reason = "frame runs past end of file";
                break;
            }
            let mut payload = vec![0u8; len as usize];
            reader.read_exact(&mut payload)?;
            if crc32(&payload) != want_crc {
                reason = "CRC mismatch";
                break;
            }
            good_offset += 8 + len;
            f(payload);
        }

        if good_offset < file_len {
            let lost = file_len - good_offset;
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let mut quarantine = self.path.as_os_str().to_owned();
            quarantine.push(format!(".corrupt-{ms}"));
            let saved = (|| -> io::Result<()> {
                let mut src = File::open(&self.path)?;
                src.seek(SeekFrom::Start(good_offset))?;
                let mut dst = File::create(&quarantine)?;
                io::copy(&mut src, &mut dst)?;
                dst.sync_all()
            })();
            match saved {
                Ok(()) => eprintln!(
                    "[WAL] {reason} at byte {good_offset} of {}: {lost} trailing bytes moved to {} before truncation",
                    self.path.display(),
                    std::path::Path::new(&quarantine).display()
                ),
                Err(e) => {
                    // Refuse to destroy data we could not preserve.
                    return Err(io::Error::new(
                        e.kind(),
                        format!("WAL {reason} at byte {good_offset}; could not save the {lost}-byte tail ({e}); refusing to truncate"),
                    ));
                }
            }
            let f = OpenOptions::new().write(true).open(&self.path)?;
            f.set_len(good_offset)?;
            f.sync_all()?;
        }
        self.good_len = good_offset;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    /// Truncate the whole log (used after a checkpoint/snapshot).
    pub fn truncate(&mut self) -> io::Result<()> {
        self.check_poisoned()?;
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.good_len = 0;
        self.sync()
    }

    /// Current log size in bytes (used to decide whether a checkpoint is worth it).
    pub fn len(&self) -> u64 {
        self.good_len
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

    /// One flipped byte in the MIDDLE of the log: earlier records replay, and
    /// the tail (including later valid records) is preserved in a
    /// `.corrupt-*` file instead of being silently destroyed.
    #[test]
    fn mid_log_corruption_preserves_tail() {
        let dir = std::env::temp_dir().join(format!("dbstrike_wal3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mid.wal");
        {
            let mut wal = Wal::open(&path).unwrap();
            for r in [&b"one"[..], b"two", b"three"] {
                wal.append(r).unwrap();
            }
        }
        // corrupt the payload of record #2 (offset 8+3 header+"one", +8 hdr)
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[11 + 8] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let mut wal = Wal::open(&path).unwrap();
        assert_eq!(wal.replay().unwrap(), vec![b"one".to_vec()]);
        let saved: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert_eq!(saved.len(), 1);
        assert_eq!(std::fs::metadata(saved[0].path()).unwrap().len(), (bytes.len() - 11) as u64);
        // appends continue cleanly after the truncation point
        wal.append(b"four").unwrap();
        drop(wal);
        let mut wal = Wal::open(&path).unwrap();
        assert_eq!(wal.replay().unwrap(), vec![b"one".to_vec(), b"four".to_vec()]);
    }

    #[test]
    fn huge_length_header_does_not_allocate() {
        let dir = std::env::temp_dir().join(format!("dbstrike_wal4_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.wal");
        std::fs::write(&path, [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 1, 2, 3]).unwrap();
        let mut wal = Wal::open(&path).unwrap();
        assert!(wal.replay().unwrap().is_empty());
    }
}
