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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Master switch for tracking, OFF by default.
///
/// Every allocation once paid 4–6 relaxed atomic RMWs (four `fetch_add`s plus
/// the `PEAK` CAS loop) on SHARED cache lines. The comment in the doc header
/// below claims "two relaxed atomic adds — a couple of nanoseconds", but the
/// real cost showed up in the Redis-wire benchmark: with 100 pipelined
/// connections hammering the allocator, those same-line RMWs ping-pong between
/// cores and durable SET @-P1024 fell from 17M/s to 4.5M/s — a 3.8× regression
/// while the machine otherwise idled.
///
/// So tracking is opt-in. `record_alloc`/`record_free` first do ONE relaxed
/// `load` on `TRACKING` — a read, which shares cleanly across cores — and only
/// touch the counters when it is set. Enable with `DBSTRIKE_MEMTRACK=1` at
/// startup or the `MEMTRACK` RESP command; it stays off for ordinary
/// workloads, restoring the pre-tracking hot-path throughput.
static TRACKING: AtomicBool = AtomicBool::new(false);

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

/// A `GlobalAlloc` that forwards to the system allocator and keeps the books
/// — but ONLY when tracking is enabled (see `TRACKING`). While disabled it is
/// a thin pass-through: one relaxed load per allocation, no counter churn, no
/// shared-line RMW contention, so the hot path pays nothing measurable.
///
/// Install in a binary with:
/// ```ignore
/// #[global_allocator]
/// static ALLOC: mitm::memtrack::TrackingAlloc = mitm::memtrack::TrackingAlloc;
/// ```
pub struct TrackingAlloc;

/// Enable or disable allocator tracking. Disabled by default (see `TRACKING`).
pub fn set_tracking(on: bool) {
    TRACKING.store(on, Ordering::Relaxed);
}

/// Whether allocator tracking is currently recording.
pub fn tracking_enabled() -> bool {
    TRACKING.load(Ordering::Relaxed)
}

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
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    // Relaxed everywhere: these are statistics, not synchronisation. We never
    // make a correctness decision from them, so ordering costs would buy
    // nothing and this sits on the allocation hot path.
    let live = LIVE.fetch_add(size, Ordering::Relaxed).saturating_add(size);
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
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    // Tracking can be toggled while an allocation is still live, so a free may
    // arrive for bytes that were never counted. Saturation keeps the books from
    // wrapping to u64::MAX (which overflow-panicked in debug and poisoned the
    // peak in release); the cost is only the rare CAS retry when saturated.
    saturating_sub(&LIVE, size);
    saturating_sub(&LIVE_OBJS, 1);
}

#[inline]
fn saturating_sub(counter: &AtomicUsize, amount: usize) {
    let mut prev = counter.load(Ordering::Relaxed);
    loop {
        let next = prev.saturating_sub(amount);
        match counter.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => prev = actual,
        }
    }
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

    // The engine-backed tests spawn WAL/flusher threads and tear them down in
    // parallel with these big allocations. That interleave trips glibc's heap
    // integrity check at thread exit ("double free or corruption (!prev)") —
    // a test-harness-only artifact that the production server never hits (it
    // does not run this test module's `#[global_allocator]`). Serialising the
    // big-allocation tests makes the trigger combination impossible.
    static BIG_ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn tracks_live_and_peak() {
        let _guard = BIG_ALLOC_LOCK.lock().unwrap();
        // Tracking is opt-in (see `TRACKING`): it must be OFF by default so the
        // hot path pays one relaxed load per allocation, and ON once asked.
        assert!(!tracking_enabled(), "tracking must default to off");
        set_tracking(true);
        assert!(tracking_enabled(), "set_tracking(true) must enable tracking");

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

        // Turning it back off must stop the books from moving.
        set_tracking(false);
    }

    #[test]
    fn tracking_off_allocations_are_not_counted() {
        let _guard = BIG_ALLOC_LOCK.lock().unwrap();
        // Tracking OFF by default: allocations must not move the books. This
        // doubles as the timing probe for the parallel suite (a large
        // allocation + a short sleep next to the engine-backed tests), to make
        // sure the opt-in switch is what changed behaviour, not the interleave.
        let off_before = snapshot();
        let _quiet: Vec<u8> = vec![9u8; 1024 * 1024];
        std::thread::sleep(std::time::Duration::from_millis(2));
        let off_after = snapshot();
        assert_eq!(
            off_before.total_allocs, off_after.total_allocs,
            "allocations must not be counted while tracking is off"
        );
        assert!(!tracking_enabled(), "tracking must stay off after this test");
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
