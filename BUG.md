# Known Bugs & Regressions

## 1. `TrackingAlloc` global allocator collapsed SET/GET throughput 4×

**Status: FIXED** — opt-in tracking gate; benchmark restored to README numbers.

**Severity:** High (durable SET @ -P1024 -c100 dropped from ~17M/s to ~4.5M/s, GET to ~2.5M/s).

**Root cause:** `crates/mitm/src/memtrack.rs` installed a `#[global_allocator]` (added in `141d44e`) that did 4–6 shared-cache-line atomic RMWs (4× `fetch_add` + a PEAK CAS loop) **per allocation**, on every thread. At 100 threads the contention on those counters thrashed cache lines and collapsed throughput.

**Fix:** Tracking is now opt-in. `TRACKING: AtomicBool` defaults off; the hot path pays one relaxed load per allocation. Enabled via `DBSTRIKE_MEMTRACK=1` at startup or the `MEMTRACK` RESP command.

**Measured after fix:** SET 16.96M/s, GET 15.88M/s @ -P1024; SET 5.88M/s @ -P64 — README numbers restored.

## 2. Reactive CDC hub ran on every write even with zero subscribers

**Status: FIXED** — lazy enable.

**Root cause:** `crates/reactive/src/lib.rs` registered a subscriber on `engine.subscribe()` that fired in the group-commit flusher's hot path for every mutation (SeqCst `fetch_add` on the CDC seq, two heap clones, a `Mutex` lock + `Vec::push` on the CDC log, an always-empty `RwLock` read).

**Fix:** `enabled: AtomicBool` is set only by the first `subscribe_prefix`/`subscribe_prefixes`/`cdc_since`/`cdc_len`; `on_commit` early-returns when disabled; seq counter is now `Relaxed`.

## 3. Memtrack counter desync when tracking toggles mid-allocation-lifecycle

**Status: FIXED** — saturating counters.

**Root cause:** `record_free` did a raw `fetch_sub` on `LIVE` while `record_alloc` did `prev + size`. A global tracking flag means an allocation can be created while tracking is OFF and freed while ON: `LIVE` wrapped to `u64::MAX`, then `prev + size` overflow-panicked inside the allocator (debug) or silently poisoned `peak` (release). Under the parallel test harness this surfaced as a sporadic `double free or corruption (!prev)` SIGABRT at thread exit.

**Fix:** `record_free` uses a saturating compare-and-swap helper for `LIVE`/`LIVE_OBJS`; `record_alloc` uses `saturating_add`. Big-allocation tests are serialized so they cannot interleave with engine/WAL thread teardown.
