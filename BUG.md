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

## 4. Wire / storage audit (2026-10): remote crashes, ACL bypass, WAL loss, lost updates

**Status: FIXED** — found by redis-cli / redis-benchmark probing, a per-command junk-arg fuzzer, and source review. Each fix has a regression test.

| Area | Bug | Fix |
|---|---|---|
| Protocol | `*99999999999999\r\n` or an overflowing `$len` aborted the whole server pre-auth (`panic = "abort"`) | `try_parse` bounds counts/lengths (Redis limits), checked arithmetic, CRLF check, 1 GiB query-buffer cap |
| Vectors | `VADD 1 1 nan`, `VCALIBRATE 0 1 0`, `VADDBATCH <huge dim>` aborted the server | NaN/inf rejected at parse; all `partial_cmp().unwrap()` → `total_cmp`; dims bounded, size math checked |
| Vectors | mismatched-dim `VADD`/`VADDNS`/RAG ingest acked `+OK`, then was invisible to search | `VectorIndex::check_dim` before any durable write; queries checked too |
| ACL | `+GET` rules ignored, `ACL` open to every user (self-escalation), AUTH re-enabled `off` users, coalesced SETs skipped checks, `ACL SAVE` was a no-op | Redis rule semantics (last match wins, `~key` patterns enforced), deny-by-default for unmapped commands, admin-only ACL, salted iterated hashes + constant-time compare, real SAVE/LOAD |
| WAL | a failed write left a partial frame mid-log; next replay truncated every later acked record. Mid-log corruption silently discarded the tail. >4 GiB frames wrapped their length | all-or-nothing group append (truncate back on error), poison after fsync failure, corrupt tail preserved as `<wal>.corrupt-<ms>`, streamed replay, frame-size check |
| MVCC | OCC validation ran before enqueue (two txns could both commit); snapshots could later gain older versions | validation inside the flusher; ts reserved under the queue lock; `snapshot()` = published `visible_ts` |
| INCR | global lock held across fsync (~3k ops/s), silent i64 wrap | engine-level atomic INCRBY resolved by the flusher, pipelined runs share one fsync (~58× faster pipelined), overflow errors |
| CRDT | counters lived only in RAM (prove-it never checked) | persisted under `crdt:*`, reloaded at boot; prove-it now asserts it |
| Pub/sub | prefix matching (`news` got `newsletter`), PSUBSCRIBE literal, connection deaf after SUBSCRIBE, every message written to the WAL | in-memory broker, exact + glob patterns, real subscribe mode (PING/UNSUBSCRIBE/RESET/QUIT), RESP3 pushes |
| Wire | fake `SELECT n`, prefix-only `KEYS`, O(N) `DBSIZE` counting internal keys, `GETAT` nil for pruned history, `ERR ERR …`, MSET error reply desync, `HELLO 4` accepted, u64 ids replied negative, reads created namespaces | see `crates/server/src/main.rs` comments at each site |
| RAG | no relevance floor; score was an RRF rank constant; cache key ignored the query vector | cosine > 0 floor; evidence score reported; vector hashed into the cache key |
| Build | `crates/gpu` hard-linked libcuda/libnvrtc (no build without the toolkit, no start without a driver) | runtime `dlsym` via vugva-core |

Not addressed here (feature work, not bugs): TTL/expiry, Redis collection types, MULTI/EXEC, replication, event-loop networking.
