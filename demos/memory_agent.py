#!/usr/bin/env python3
"""
A real AI agent with persistent memory, built on DB-Strike.

It uses all SEVEN memory primitives the engine exposes over RESP — the same
set Mem0 (LTM + graph + keyword), Zep (bi-temporal) and Mem0's "third
pillar" (procedural) ship, PLUS the two we just wired: Working Memory (STM)
and Episodic memory. No external crates, one process, one WAL.

  * Working memory (STM)  — hot per-turn scratchpad, TTL-cleared
  * Long-term memory (LTM) — semantic facts, vector + keyword searchable
  * Episodic memory         — append-only event log ("what happened")
  * Graph memory            — typed edges between entities (multi-hop)
  * Bi-temporal facts      — "what did we know as of time T"
  * Keyword memory (BM25)  — exact-term recall, fused into recall
  * Procedural memory      — learned per-agent playbooks

Run a DB-Strike server first (see run_demos.sh), then:
    python3 demos/memory_agent.py
"""

import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from dbstrike_client import DBStrike, embed


def main():
    db = DBStrike()
    print("Connected to DB-Strike. Booting agent 'Atlas'...\n")
    AGENT = "atlas"
    now = int(time.time())

    # --- 1. WORKING MEMORY (STM): hot per-turn context, TTL-cleared ----
    db.wm_set(AGENT, "goal", "ship the v0.2 release", 60_000)
    goal = db.wm_get(AGENT, "goal")
    print(f"[1 WM]  goal = {goal!r}  (hot, TTL-backed, Redis-speed)")

    # --- 2. LONG-TERM MEMORY (LTM): semantic facts -------------------
    facts = [
        ("Atlas is a release-engineering agent", "user", 0.9),
        ("The v0.2 release is blocked on the WAL fsync bug", "tool:ci", 0.95),
        ("Irfan prefers Rust over Python for the hot path", "user", 0.8),
        ("Benchmarks run on loopback, not over the wire", "agent:atlas", 0.6),
    ]
    for text, src, sal in facts:
        fid = db.remember(text, src, sal, embed(text, dim=128))
        print(f"[2 LTM] stored fact #{fid}: {text!r}")

    # Blended semantic + keyword recall
    q = "what is blocking the release?"
    hits = db.recall(3, q, embed(q, dim=128))
    print(f"\n[2 LTM recall] query: {q!r}")
    for h in hits:
        print(f"   score={h['score']:.3f} src={h['source']:<14} {h['text']}")

    # --- 3. EPISODIC MEMORY: append-only event log ------------------
    db.episode(AGENT, "observation", b"user mentioned a deadline next Friday")
    db.episode(AGENT, "action", b"ran the regression suite (3 fails)")
    db.episode(AGENT, "observation", b"user switched context to the caching layer")
    eps = db.episodes(AGENT, 50)
    print(f"\n[3 EP]  {len(eps)} events logged (kind+time retained):")
    for e in eps:
        print(f"   seq={e['seq']}  kind={e['kind']:<12} {e['payload']!r}")

    # --- 4. GRAPH MEMORY: entities + typed edges --------------------
    alice = db.remember("Alice is the release manager", "hr", 0.9,
                        embed("alice release manager", 128))
    acme = db.remember("Acme Corp builds DB-Strike", "org", 0.9,
                        embed("acme db-strike", 128))
    nyc = db.remember("Acme is headquartered in New York", "org", 0.7,
                        embed("acme new york hq", 128))
    db.link(alice, acme, "works_at", 0.9)
    db.link(acme, nyc, "located_in", 1.0)
    visited = db.traverse(alice, 2)
    print(f"\n[4 GRAPH] 2-hop from Alice: {visited}  (reaches HQ, no graph DB)")

    # --- 5. BI-TEMPORAL FACTS: state that evolves over time ----------
    t0, t1 = now - 1000, now
    cto = db.remember_temporal("Alice is CTO at Acme", "hr", 0.9, t0, t1,
                              embed("alice cto acme", 128))
    ceo = db.remember_temporal("Alice is CEO at Acme", "hr", 0.9, t1, 0,
                              embed("alice ceo acme", 128))
    mid = db.recall_as_of(1, "alice title acme", (t0 + t1) // 2,
                          embed("alice title acme", 128))
    after = db.recall_as_of(1, "alice title acme", now + 500,
                            embed("alice title acme", 128))
    print(f"\n[5 TEMP] as-of mid:   {mid[0]['text'] if mid else '∅'}")
    print(f"[5 TEMP] as-of now+500: {after[0]['text'] if after else '∅'}")

    # --- 6. KEYWORD (BM25): fused inside recall (no separate call) ---
    q2 = "rust python hot path"
    kw = db.recall(2, q2, embed(q2, dim=128))
    print(f"\n[6 KW] keyword query {q2!r} -> "
          f"{[h['text'][:32] for h in kw]}  (exact-term recall)")

    # --- 7. PROCEDURAL MEMORY: learned playbook ----------------------
    db.proc_set(AGENT, "deploy",
                "1. run tests\n2. tag release\n3. push image\n4. bump helm chart")
    db.proc_set(AGENT, "commit-style", "imperative, <60 chars, no trailing period")
    print(f"\n[7 PROC] {AGENT} playbooks: {db.proc_list(AGENT)}")
    print(f"[7 PROC] deploy playbook:\n{db.proc_get(AGENT, 'deploy').decode()}")

    # --- Tie it together: one agent "turn" (WM + LTM recall + EP) ---
    t0 = time.perf_counter()
    db.wm_set(AGENT, "last_turn", "user asked about the release blocker", 30_000)
    db.recall(3, "release blocker status", embed("release blocker status", 128))
    db.episode(AGENT, "turn", b"answered the release-blocker question")
    dt = (time.perf_counter() - t0) * 1e6
    print(f"\n[agent turn] WM write + LTM recall + EP append in {dt:.0f} µs")

    print("\nAgent 'Atlas' state is durable: kill + restart the server, the "
          "memory replays from the WAL. Done.")


if __name__ == "__main__":
    main()
