//! A coalescing best-fit allocator over a flat range of offsets.
//!
//! Two tiers need the same thing: hand out sub-ranges of a fixed-size region
//! and take them back. The DRAM tier's region is an `mmap`'d, page-locked host
//! mapping; the SSD tier's is a file. Neither cares about the other's
//! addressing, and both need reuse — a tier that can only ever hand out space
//! turns its capacity into a *lifetime* budget rather than a working-set one,
//! which is the failure that made `demote` and `spill` pointless (BUG #19).
//!
//! Offsets, not pointers, so the same allocator serves a host mapping and a
//! file without the two being able to confuse each other's units.

use crate::{Result, VugvaError};

/// Alignment for every allocation.
///
/// 64 bytes: the DMA descriptor's granularity, one cache line, and enough for
/// any vector load a kernel will issue against a promoted chunk.
pub const ALIGN: usize = 64;

/// Best-fit allocator over `[0, capacity)`.
#[derive(Debug)]
pub struct RangeAllocator {
    /// Total size of the region.
    capacity: usize,
    /// High-water mark. Everything below has been handed out at least once; a
    /// sub-range of it may be back on `free_blocks`.
    offset: usize,
    /// Returned ranges, sorted by offset and coalesced on insert.
    free_blocks: Vec<(usize, usize)>,
}

impl RangeAllocator {
    /// Create an allocator over a region of `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        RangeAllocator {
            capacity,
            offset: 0,
            free_blocks: Vec::new(),
        }
    }

    /// Round up to [`ALIGN`].
    pub fn align(bytes: usize) -> usize {
        (bytes + ALIGN - 1) & !(ALIGN - 1)
    }

    /// Total size of the region.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently handed out.
    pub fn used(&self) -> usize {
        self.offset - self.free_blocks.iter().map(|&(_, l)| l).sum::<usize>()
    }

    /// Number of distinct free ranges. A large count against a large
    /// [`RangeAllocator::used`] gap is fragmentation.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Carve out `bytes`, reusing a freed range when one fits.
    ///
    /// Best-fit rather than first-fit: these regions see a handful of distinct
    /// tensor shapes allocated repeatedly, so best-fit tends to land on an
    /// exact-size hole left by the previous instance of the same shape and
    /// leaves the larger holes intact. First-fit would shred a big block to
    /// satisfy a small request.
    pub fn allocate(&mut self, bytes: usize) -> Result<usize> {
        let aligned = Self::align(bytes);

        // Reuse before extending.
        let mut best: Option<usize> = None;
        for (i, &(_, len)) in self.free_blocks.iter().enumerate() {
            if len >= aligned && best.is_none_or(|b| len < self.free_blocks[b].1) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let (off, len) = self.free_blocks[i];
            if len == aligned {
                self.free_blocks.remove(i);
            } else {
                // Keep the tail. The list stays sorted: this block's offset
                // only moves up, and it still precedes the next one.
                self.free_blocks[i] = (off + aligned, len - aligned);
            }
            return Ok(off);
        }

        if self.offset + aligned > self.capacity {
            return Err(VugvaError::DramOom {
                requested: aligned,
                available: self.capacity - self.used(),
                capacity: self.capacity,
            });
        }
        let off = self.offset;
        self.offset += aligned;
        Ok(off)
    }

    /// Return a previously allocated range.
    ///
    /// Coalesces with the neighbours on both sides, and if the result reaches
    /// the high-water mark it retracts `offset` rather than parking a block at
    /// the top. Without that retraction, a free/alloc cycle of decreasing
    /// sizes would ratchet `offset` up forever even though the region never
    /// holds more than one allocation at a time.
    ///
    /// Ranges that are out of bounds, or that overlap something already free,
    /// are dropped. That leaks the range, which is strictly better than the
    /// alternative: a corrupted free list hands the same bytes to two live
    /// allocations, and the resulting data corruption is silent and remote
    /// from its cause.
    pub fn free(&mut self, off: usize, bytes: usize) {
        let len = Self::align(bytes);
        if len == 0 || off + len > self.offset {
            return;
        }

        let pos = self.free_blocks.partition_point(|&(o, _)| o < off);
        if pos < self.free_blocks.len() && off + len > self.free_blocks[pos].0 {
            return;
        }
        if pos > 0 {
            let (po, pl) = self.free_blocks[pos - 1];
            if po + pl > off {
                return;
            }
        }
        self.free_blocks.insert(pos, (off, len));

        // Coalesce right, then left.
        if pos + 1 < self.free_blocks.len() {
            let (no, nl) = self.free_blocks[pos + 1];
            if off + len == no {
                self.free_blocks[pos].1 += nl;
                self.free_blocks.remove(pos + 1);
            }
        }
        let mut idx = pos;
        if pos > 0 {
            let (po, pl) = self.free_blocks[pos - 1];
            if po + pl == off {
                self.free_blocks[pos - 1].1 += self.free_blocks[pos].1;
                self.free_blocks.remove(pos);
                idx = pos - 1;
            }
        }

        // Retract the high-water mark if the top block now touches it.
        let (o, l) = self.free_blocks[idx];
        if o + l == self.offset {
            self.offset = o;
            self.free_blocks.remove(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_is_aligned_and_sequential() {
        let mut r = RangeAllocator::new(4096);
        assert_eq!(r.allocate(1).unwrap(), 0);
        assert_eq!(
            r.allocate(1).unwrap(),
            64,
            "sub-alignment sizes must still consume a line"
        );
        assert_eq!(r.used(), 128);
    }

    #[test]
    fn free_then_allocate_reuses_the_same_range() {
        let mut r = RangeAllocator::new(4096);
        let keep = r.allocate(128).unwrap();
        let a = r.allocate(256).unwrap();
        r.free(a, 256);
        let b = r.allocate(256).unwrap();
        assert_eq!(a, b, "the freed range must be handed back out");
        assert_eq!(r.used(), 384);
        assert_ne!(keep, b);
    }

    #[test]
    fn survives_more_churn_than_it_has_capacity_for() {
        // The exact failure the bump-only allocator had: a region that never
        // holds more than one allocation still ran out after `capacity/size`
        // cycles.
        let mut r = RangeAllocator::new(4096);
        for _ in 0..1000 {
            let off = r.allocate(1024).unwrap();
            r.free(off, 1024);
        }
        assert_eq!(r.used(), 0);
    }

    #[test]
    fn free_coalesces_adjacent_blocks() {
        let mut r = RangeAllocator::new(4096);
        let a = r.allocate(256).unwrap();
        let b = r.allocate(256).unwrap();
        let c = r.allocate(256).unwrap();
        let _tail = r.allocate(256).unwrap(); // keeps the top from retracting

        // Free out of order so the middle block has to merge both ways.
        r.free(a, 256);
        r.free(c, 256);
        assert_eq!(r.free_block_count(), 2);
        r.free(b, 256);
        assert_eq!(r.free_block_count(), 1, "a+b+c must merge into one block");

        // The merged block must be usable as a single large allocation, which
        // is the whole point of coalescing.
        assert_eq!(r.allocate(768).unwrap(), a);
    }

    #[test]
    fn free_retracts_the_high_water_mark() {
        let mut r = RangeAllocator::new(4096);
        let a = r.allocate(1024).unwrap();
        r.free(a, 1024);
        assert_eq!(r.used(), 0);
        assert_eq!(
            r.free_block_count(),
            0,
            "a freed top block must not park on the list"
        );
    }

    #[test]
    fn free_ignores_out_of_range_and_double_frees() {
        let mut r = RangeAllocator::new(4096);
        let a = r.allocate(256).unwrap();
        let _b = r.allocate(256).unwrap();
        let before = r.used();

        r.free(1 << 20, 256); // past the high-water mark
        assert_eq!(r.used(), before, "out-of-range frees must be ignored");

        r.free(a, 256);
        let after_one = r.used();
        r.free(a, 256); // double free
        assert_eq!(
            r.used(),
            after_one,
            "a double free must not put the range on the list twice"
        );

        // The decisive check: the range must be handed out exactly once.
        let x = r.allocate(256).unwrap();
        let y = r.allocate(256).unwrap();
        assert_ne!(x, y, "double free must never alias two live allocations");
    }

    #[test]
    fn best_fit_prefers_the_tightest_hole() {
        let mut r = RangeAllocator::new(8192);
        let small = r.allocate(128).unwrap();
        let _g1 = r.allocate(64).unwrap();
        let large = r.allocate(1024).unwrap();
        let _g2 = r.allocate(64).unwrap();
        r.free(small, 128);
        r.free(large, 1024);

        assert_eq!(
            r.allocate(128).unwrap(),
            small,
            "must not shred the 1024-byte hole for 128 bytes"
        );
        // The large hole is therefore still intact.
        assert_eq!(r.allocate(1024).unwrap(), large);
    }

    #[test]
    fn exhaustion_is_a_host_error_not_a_device_one() {
        let mut r = RangeAllocator::new(256);
        assert!(r.allocate(256).is_ok());
        // Must not masquerade as a *device* OOM — no GPU is involved, and the
        // fixes that error suggests all target the wrong resource.
        match r.allocate(64) {
            Err(VugvaError::DramOom {
                requested,
                available,
                capacity,
            }) => {
                assert_eq!(requested, 64);
                assert_eq!(available, 0);
                assert_eq!(capacity, 256);
            }
            other => panic!("expected DramOom, got {other:?}"),
        }
    }

    #[test]
    fn oom_distinguishes_fragmentation_from_exhaustion() {
        let mut r = RangeAllocator::new(1024);
        let a = r.allocate(256).unwrap();
        let _b = r.allocate(256).unwrap();
        let c = r.allocate(256).unwrap();
        let _d = r.allocate(256).unwrap();
        r.free(a, 256);
        r.free(c, 256);

        // 512 bytes free, but in two non-adjacent holes.
        match r.allocate(512) {
            Err(VugvaError::DramOom { available, .. }) => assert_eq!(
                available, 512,
                "a fragmentation failure must report the free bytes, so it is \
                 distinguishable from a genuinely full region"
            ),
            other => panic!("expected DramOom, got {other:?}"),
        }
    }
}
