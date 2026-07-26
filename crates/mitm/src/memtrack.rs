//! MITM memory tracker — "catch memory bugs instantly".
//!
//! The sibling of the cache debugger, built for the same reason: when the
//! process is doing something inexplicable, measure it rather than reason about
//! it. The cache debugger answers "is this value stale?"; this answers "who is
//! holding the memory?".
//!
//! It exists because a 200k × 384d ingest drove the server to 29.3 GB resident
//! (61.7 GB virtual) and the kernel OOM-killer took it out, while the *same*
//! index built in-process stayed flat at 1.2 GB. Ordinary tools could not close
//! that gap: RSS tells you the total but not the owner, and a heap profiler is
//! an external crate this project does not take. So the allocator itself keeps
//! the books.
//!
//! Two things are recorded on every allocation:
//!
//!   * **Live bytes and peak**, so a leak is visible as a live count that never
//!     comes back down between phases.
//!   * **A size histogram**, because the shape distinguishes the two failure
//!     modes that look identical in RSS. A few enormous allocations mean a
//!     runaway buffer; millions of small ones mean allocator fragmentation,
//!     where RSS balloons far past live bytes and never returns to the OS.
//!
//! Cost when idle is two relaxed atomic adds per allocation — a couple of
//! nanoseconds, no syscalls, no locks. That matters: this has to be able to run
//! on the hot ingest path without changing the behaviour it is measuring.
//!
//! Pure Rust. Zero external crates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live (allocated minus freed) bytes.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of `LIVE`.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Total allocation calls, ever.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// Live allocation count (allocs minus frees). Divided into `LIVE` this gives
/// the mean live object size, which is the fragmentation tell.
static LIVE_OBJS: AtomicUsize = AtomicUsize::new(0);

/// Power-of-two size buckets: <=32, <=256, <=2K, <=16K, <=128K, <=1M, <=8M, >8M.
const NBUCKETS: usize = 8;
static BUCKETS: [AtomicUsize; NBUCKETS] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

const BUCKET_LABELS: [&str; NBUCKETS] = [
    "<=32B", "<=256B", "<=2K", "<=16K", "<=128K", "<=1M", "<=8M", ">8M",
];

#[inline]
fn bucket_of(size: usize) -> usize {
    match size {
        0..=32 => 0,
        33..=256 => 1,
        257..=2048 => 2,
        2049..=16384 => 3,
        16385..=131_072 => 4,
        131_073..=1_048_576 => 5,
        1_048_577..=8_388_608 => 6,
        _ => 7,
    }
}

/// A `GlobalAlloc` that forwards to the system allocator and keeps the books.
///
/// Install in a binary with:
/// ```ignore
/// #[global_allocator]
/// static ALLOC: mitm::memtrack::TrackingAlloc = mitm::memtrack::TrackingAlloc;
/// ```
pub struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_free(layout.size());
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            // A realloc is a free of the old size and an alloc of the new one.
            // Tracking it as such is what makes Vec growth visible: a Vec that
            // doubles repeatedly shows up as a climbing bucket, not as silence.
            record_free(layout.size());
            record_alloc(new_size);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }
}

#[inline]
fn record_alloc(size: usize) {
    // Relaxed everywhere: these are statistics, not synchronisation. We never
    // make a correctness decision from them, so ordering costs would buy
    // nothing and this sits on the allocation hot path.
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    LIVE_OBJS.fetch_add(1, Ordering::Relaxed);
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BUCKETS[bucket_of(size)].fetch_add(1, Ordering::Relaxed);
    // Compare-and-swap the high-water mark. Contended only while the peak is
    // actually moving, which is rare after warmup.
    let mut peak = PEAK.load(Ordering::Relaxed);
    while live > peak {
        match PEAK.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(p) => peak = p,
        }
    }
}

#[inline]
fn record_free(size: usize) {
    LIVE.fetch_sub(size, Ordering::Relaxed);
    LIVE_OBJS.fetch_sub(1, Ordering::Relaxed);
}

/// A point-in-time reading of the allocator's books.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemSnapshot {
    pub live_bytes: usize,
    pub peak_bytes: usize,
    pub live_objects: usize,
    pub total_allocs: usize,
    pub buckets: [usize; NBUCKETS],
}

impl MemSnapshot {
    /// Mean size of a currently-live allocation. A value in the tens of bytes
    /// alongside a large `live_bytes` is the signature of death-by-small-object:
    /// the heap is dominated by millions of tiny allocations, and glibc will
    /// hold the arenas long after they are freed.
    pub fn mean_live_object(&self) -> f64 {
        if self.live_objects == 0 {
            0.0
        } else {
            self.live_bytes as f64 / self.live_objects as f64
        }
    }

    /// One-line summary for logs.
    pub fn line(&self, tag: &str) -> String {
        format!(
            "[MEM] {tag:<18} live={:>8.1} MB  peak={:>8.1} MB  objs={:>10}  mean={:>7.0} B",
            self.live_bytes as f64 / 1_048_576.0,
            self.peak_bytes as f64 / 1_048_576.0,
            self.live_objects,
            self.mean_live_object(),
        )
    }

    /// Allocation-count histogram by size class, cumulative since start.
    pub fn histogram(&self) -> String {
        let mut s = String::from("[MEM] size histogram (cumulative allocs):\n");
        for i in 0..NBUCKETS {
            s.push_str(&format!(
                "        {:>7}  {:>14}\n",
                BUCKET_LABELS[i], self.buckets[i]
            ));
        }
        s
    }

    /// Difference between two readings, for attributing growth to a phase.
    pub fn since(&self, base: &MemSnapshot) -> i64 {
        self.live_bytes as i64 - base.live_bytes as i64
    }
}

/// Read the allocator's books.
pub fn snapshot() -> MemSnapshot {
    let mut buckets = [0usize; NBUCKETS];
    for i in 0..NBUCKETS {
        buckets[i] = BUCKETS[i].load(Ordering::Relaxed);
    }
    MemSnapshot {
        live_bytes: LIVE.load(Ordering::Relaxed),
        peak_bytes: PEAK.load(Ordering::Relaxed),
        live_objects: LIVE_OBJS.load(Ordering::Relaxed),
        total_allocs: ALLOCS.load(Ordering::Relaxed),
        buckets,
    }
}

/// This process's VmRSS in bytes, or 0 where /proc is unavailable.
///
/// Reported next to `live_bytes` on purpose. The two agreeing means the heap is
/// simply large; RSS far exceeding live bytes means the allocator is holding
/// freed memory — fragmentation — and no amount of freeing will bring RSS down.
/// Separating those two cases is the whole point of this module.
pub fn rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmRSS:").map(|r| r.to_string()))
        })
        .and_then(|r| r.split_whitespace().next()?.parse::<usize>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// `live`, `peak`, `rss` in MB — the three numbers that together localise a leak.
pub fn report(tag: &str) -> String {
    let s = snapshot();
    format!(
        "{}  rss={:>8.1} MB  overhead={:>6.2}x",
        s.line(tag),
        rss_bytes() as f64 / 1_048_576.0,
        if s.live_bytes > 0 {
            rss_bytes() as f64 / s.live_bytes as f64
        } else {
            0.0
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `#[global_allocator]` only takes effect in the final artifact, so the
    // library's own tests would otherwise measure an allocator that is never
    // installed and see every counter sit at zero. Installing it here means the
    // tests exercise the real thing rather than a hopeful assertion about it.
    #[global_allocator]
    static ALLOC: TrackingAlloc = TrackingAlloc;

    #[test]
    fn tracks_live_and_peak() {
        let before = snapshot();
        let big: Vec<u8> = vec![7u8; 4 * 1024 * 1024];
        let during = snapshot();
        assert!(
            during.live_bytes > before.live_bytes,
            "a 4 MB allocation must move live bytes"
        );
        drop(big);
        let after = snapshot();
        assert!(
            after.live_bytes < during.live_bytes,
            "freeing must bring live bytes back down"
        );
        assert!(after.peak_bytes >= during.live_bytes, "peak is a high-water mark");
    }

    #[test]
    fn buckets_classify_by_size() {
        assert_eq!(bucket_of(1), 0);
        assert_eq!(bucket_of(32), 0);
        assert_eq!(bucket_of(33), 1);
        assert_eq!(bucket_of(2048), 2);
        assert_eq!(bucket_of(9_000_000), 7);
    }
}
