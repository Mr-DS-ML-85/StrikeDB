//! T2: the NVMe spill tier.
//!
//! The paper's cold tier (§4, 5–7 GB/s). A page that has been idle long enough
//! to be evicted from DRAM is written to a backing file and its host memory
//! released; loading it back reads straight into the destination pool, which is
//! page-locked, so the subsequent DRAM→VRAM promotion is still a real DMA.
//!
//! ## Why plain `pread`/`pwrite`
//!
//! GPUDirect Storage (cuFile) would let the NVMe controller DMA into VRAM
//! without a host bounce. It is also a separate closed library (`libcufile`),
//! which this crate cannot take: nothing here links anything but libc and the
//! CUDA driver, and the driver itself is `dlopen`'d. Positional file I/O is in
//! the standard library, needs no seek/read pair (so no shared file cursor and
//! no lock around it), and lands the bytes directly in the pinned pool — which
//! keeps the *second* hop, the one that actually crosses PCIe, on the DMA path.
//!
//! The honest cost of that choice: a spill round trip is one extra copy
//! compared to GPUDirect. It is recorded here rather than papered over.

use crate::{Result, VugvaError};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::range_alloc::RangeAllocator;

/// A file-backed cold tier with its own offset allocator.
#[derive(Debug)]
pub struct SpillFile {
    file: File,
    path: PathBuf,
    alloc: RangeAllocator,
    /// Remove the file on drop.
    unlink_on_drop: bool,
}

impl SpillFile {
    /// Open (creating, truncating) a spill file of at most `capacity` bytes.
    ///
    /// The path is the caller's to choose. Nothing here invents a location:
    /// the file can be as large as the model, and picking a directory on the
    /// user's behalf is how a tiering layer ends up filling a tmpfs — which is
    /// RAM, so the cold tier would silently consume the very resource it
    /// exists to relieve.
    ///
    /// `unlink_on_drop` deletes the file when the pool goes away. Spilled data
    /// is a cache of pages the process still owns in its VMT, so it has no
    /// meaning once that VMT is gone; keeping it would leave model-sized
    /// garbage behind on every run.
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize, unlink_on_drop: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Size the file up front. Writing into a hole is legal but makes the
        // first write to each block a metadata update, so the cost of growing
        // the file would land inside the spill latency it is meant to hide.
        file.set_len(capacity as u64)?;

        Ok(SpillFile {
            file,
            path,
            alloc: RangeAllocator::new(capacity),
            unlink_on_drop,
        })
    }

    /// Path of the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.alloc.capacity()
    }

    /// Bytes currently occupied by spilled pages.
    pub fn used(&self) -> usize {
        self.alloc.used()
    }

    /// Reserve `bytes` without writing anything, returning the file offset.
    ///
    /// For a page created *cold* — allocated straight into T2 rather than
    /// demoted into it — there is nothing to write yet. The range still has to
    /// be claimed at allocation time so two cold pages cannot be handed the
    /// same offset, and so the caller learns the tier is full then rather than
    /// at first writeback.
    ///
    /// The file is set to its full length at creation, so a reserved range is
    /// already readable and reads as zeros until something writes it.
    pub fn reserve(&mut self, bytes: usize) -> Result<u64> {
        Ok(self.alloc.allocate(bytes)? as u64)
    }

    /// Write `bytes` from `src` at an offset obtained from [`SpillFile::reserve`].
    ///
    /// # Safety
    ///
    /// `src` must point to at least `bytes` of initialised, readable memory,
    /// and `offset` must name a range previously reserved for at least `bytes`.
    pub unsafe fn write_at(&self, src: *const u8, bytes: usize, offset: u64) -> Result<()> {
        // SAFETY: the caller guarantees `src` covers `bytes`. The slice is used
        // only for the duration of the write and never escapes.
        let buf = unsafe { std::slice::from_raw_parts(src, bytes) };
        // `write_all_at`, as in `write_at_new_offset`: a short write is legal
        // and would silently truncate the page.
        self.file.write_all_at(buf, offset)?;
        Ok(())
    }

    /// Reserve `bytes` and write `data` there, returning the file offset.
    ///
    /// # Safety
    ///
    /// `src` must point to at least `bytes` of initialised, readable memory.
    pub unsafe fn write_at_new_offset(&mut self, src: *const u8, bytes: usize) -> Result<u64> {
        let offset = self.alloc.allocate(bytes)?;
        // SAFETY: the caller guarantees `src` covers `bytes`. The slice is used
        // only for the duration of the write and never escapes.
        let buf = unsafe { std::slice::from_raw_parts(src, bytes) };
        // `write_all_at` rather than `write_at`: a short write is legal and
        // would otherwise silently truncate the page, which surfaces much later
        // as a corrupt tensor rather than as an I/O error.
        if let Err(e) = self.file.write_all_at(buf, offset as u64) {
            // Do not strand the reservation on a failed write.
            self.alloc.free(offset, bytes);
            return Err(VugvaError::Io(e));
        }
        Ok(offset as u64)
    }

    /// Read `bytes` from `offset` into `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least `bytes` of writable memory.
    pub unsafe fn read_into(&self, offset: u64, dst: *mut u8, bytes: usize) -> Result<()> {
        // SAFETY: the caller guarantees `dst` covers `bytes`. `read_exact_at`
        // writes every byte before the slice is observed.
        let buf = unsafe { std::slice::from_raw_parts_mut(dst, bytes) };
        self.file.read_exact_at(buf, offset)?;
        Ok(())
    }

    /// Release a spilled range.
    pub fn free(&mut self, offset: u64, bytes: usize) {
        self.alloc.free(offset as usize, bytes);
    }

    /// Flush written pages to the device.
    ///
    /// Not called on the spill path: a spilled page is a *cache* of something
    /// the VMT still describes, so it does not need to survive a crash, and
    /// paying an fsync per eviction would dominate the tier's latency. Exposed
    /// for callers that do want the durability.
    pub fn sync(&self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        if self.unlink_on_drop {
            // Errors are unactionable here; a leftover file is not worth a
            // panic in a destructor.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spill files are created next to the test binary, under `target/`.
    ///
    /// Deliberately not `/tmp`: on most desktop installs that is a tmpfs, i.e.
    /// RAM, so a cold tier placed there consumes the exact resource it exists
    /// to relieve, and a large spill can drive the machine into swap.
    fn scratch(name: &str) -> PathBuf {
        let mut p = std::env::current_exe().expect("current_exe");
        p.pop();
        p.push(format!("vugva_spill_test_{name}_{}", std::process::id()));
        p
    }

    #[test]
    fn round_trips_bytes_at_the_offset_it_reports() {
        let path = scratch("roundtrip");
        let mut s = SpillFile::create(&path, 1 << 20, true).expect("create");

        let a: Vec<u8> = (0..4096u32).map(|i| (i ^ 0x5A) as u8).collect();
        let b: Vec<u8> = (0..4096u32).map(|i| (i ^ 0xC3) as u8).collect();

        let off_a = unsafe { s.write_at_new_offset(a.as_ptr(), a.len()) }.expect("write a");
        let off_b = unsafe { s.write_at_new_offset(b.as_ptr(), b.len()) }.expect("write b");
        assert_ne!(off_a, off_b, "two live pages must not share an offset");

        // Read back in the opposite order, so a stale-cursor bug cannot pass.
        let mut got_b = vec![0u8; b.len()];
        unsafe { s.read_into(off_b, got_b.as_mut_ptr(), b.len()) }.expect("read b");
        assert_eq!(got_b, b);

        let mut got_a = vec![0u8; a.len()];
        unsafe { s.read_into(off_a, got_a.as_mut_ptr(), a.len()) }.expect("read a");
        assert_eq!(got_a, a);
    }

    #[test]
    fn freed_offsets_are_reused() {
        let path = scratch("reuse");
        let mut s = SpillFile::create(&path, 1 << 20, true).expect("create");
        let data = vec![7u8; 8192];

        let first = unsafe { s.write_at_new_offset(data.as_ptr(), data.len()) }.expect("write");
        assert_eq!(s.used(), 8192);
        s.free(first, data.len());
        assert_eq!(s.used(), 0);

        let second = unsafe { s.write_at_new_offset(data.as_ptr(), data.len()) }.expect("write");
        assert_eq!(first, second, "a released offset must be handed back out");
    }

    #[test]
    fn a_full_file_reports_an_error_rather_than_overwriting() {
        let path = scratch("full");
        let mut s = SpillFile::create(&path, 4096, true).expect("create");
        let data = vec![0u8; 4096];
        assert!(unsafe { s.write_at_new_offset(data.as_ptr(), data.len()) }.is_ok());
        // The decisive part: not merely an error, but no aliasing of the range
        // already in use.
        assert!(unsafe { s.write_at_new_offset(data.as_ptr(), 64) }.is_err());
    }

    #[test]
    fn the_backing_file_is_removed_on_drop() {
        let path = scratch("unlink");
        {
            let _s = SpillFile::create(&path, 4096, true).expect("create");
            assert!(path.exists());
        }
        assert!(
            !path.exists(),
            "spilled pages are a cache of live VMT entries and are meaningless \
             once the pool is gone — leaving them behind drops a model-sized \
             file on disk every run"
        );
    }
}
