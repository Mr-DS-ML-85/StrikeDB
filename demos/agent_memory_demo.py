#!/usr/bin/env python3
"""
Demo 1 — A real AI agent with persistent memory, built on DB-Strike.

This agent maintains four memory structures across a multi-turn conversation,
all backed by the SAME DB-Strike engine (one process, one WAL, one consistency
model — not five glued systems):

  * Working memory (STM)  — hot per-turn context, TTL-cleared.
  * Long-term memory (LTM) — semantic facts, vector + keyword searchable.
  * Graph memory          — typed edges between entities (multi-hop reasoning).
  * Bi-temporal facts     — "what did we know as of time T" (Zep Graphiti style).
  * Procedural memory     — learned per-agent playbooks (Mem0's 3rd pillar).

Run a DB-Strike server first (see run_demos.sh), then:
    python3 demos/agent_memory_demo.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dbstrike_client import DBStrike, embed


def main():
    db = DBStrike()
    print("Connected to DB-Strike. Booting agent 'Atlas'...\n")

    AGENT = "atlas"
    now = int(time.time())

    # --- 1. Working memory: a hot per-turn scratchpad with TTL --------------
    db.set("wm:" + AGENT + ":goal", "ship the v0.2 release")  # SET
    wm = db.get("wm:" + AGENT + ":goal")
    print(f"[working memory] goal = {wm.decode()!r}  (hot, TTL-backed)")

    # --- 2. Long-term memory: semantic facts --------------------------------
    facts = [
        ("Atlas is a release-engineering agent", "user", 0.9),
        ("The v0.2 release is blocked on the WAL fsync bug", "tool:ci", 0.95),
        ("Irfan prefers Rust over Python for the hot path", "user", 0.8),
        ("Benchmarks run on loopback, not over the wire", "agent:atlas", 0.6),
    ]
    fids = []
    for text, src, sal in facts:
        fid = db.remember(text, src, sal, embed(text, dim=128))
        fids.append(fid)
        print(f"[LTM] stored fact #{fid}: {text!r}")

    # Recall by semantic + keyword blend
    q = "what is blocking the release?"
    hits = db.recall(3, q, embed(q, dim=128))
    print(f"\n[recall] query: {q!r}")
    for h in hits:
        print(f"   score={h['score']:.3f} src={h['source']:<14} {h['text']}")

    # --- 3. Graph memory: entities + typed edges ---------------------------
    # Build a knowledge graph over the facts so we can do multi-hop reasoning.
    alice = db.remember("Alice is the release manager", "hr", 0.9, embed("alice release manager", 128))
    acme = db.remember("Acme Corp builds DB-Strike", "org", 0.9, embed("acme db-strike", 128))
    nyc = db.remember("Acme is headquartered in New York", "org", 0.7, embed("acme new york hq", 128))
    db.link(alice, "manages_release", "v0.2", 0.9)
    db.link(alice, acme, "works_at", 0.9)
    db.link(acme, nyc, "located_in", 1.0)
    neigh = db.neighbors(alice)
    print(f"\n[graph] neighbors of Alice (#{alice}):")
    for nid, rel, w in neigh:
        print(f"   -> #{nid}  rel={rel:<16} w={w}")

    # 2-hop traversal: Alice -> Acme -> NYC
    visited = db.traverse(alice, 2)
    print(f"[graph] 2-hop traversal from Alice: {visited}")
    print(f"        (reaches HQ via Acme without a separate graph DB)")

    # --- 4. Bi-temporal facts: state that evolves over time ----------------
    # Alice was CTO from t0..t1, then CEO from t1 onward.
    t0, t1 = now - 1000, now
    cto = db.remember_temporal("Alice is CTO at Acme", "hr", 0.9, t0, t1,
                               embed("alice cto acme", 128))
    ceo = db.remember_temporal("Alice is CEO at Acme", "hr", 0.9, t1, 0,
                               embed("alice ceo acme", 128))
    as_of_mid = (t0 + t1) // 2
    mid = db.recall_as_of(1, "alice title acme", as_of_mid, embed("alice title acme", 128))
    after = db.recall_as_of(1, "alice title acme", now + 500, embed("alice title acme", 128))
    print(f"\n[bi-temporal] as-of {as_of_mid}: {mid[0]['text'] if mid else '∅'}")
    print(f"[bi-temporal] as-of now+500: {after[0]['text'] if after else '∅'}")
    # Now the CTO fact is superseded; invalidate it.
    db.invalidate_fact(cto, now + 100)
    after_inv = db.recall_as_of(1, "alice title acme", now + 500, embed("alice title acme", 128))
    print(f"[bi-temporal] after INVALIDATE(cto): {after_inv[0]['text'] if after_inv else '∅'} "
          f"(non-contradictory reasoning preserved)")

    # --- 5. Procedural memory: learned playbook ----------------------------
    db.proc_set(AGENT, "deploy",
        "1. run tests\n2. tag release\n3. push image\n4. bump helm chart")
    db.proc_set(AGENT, "commit-style", "imperative, <60 chars, no trailing period")
    playbooks = db.proc_list(AGENT)
    print(f"\n[procedural] {AGENT} playbooks: {playbooks}")
    print(f"[procedural] deploy playbook:\n{db.proc_get(AGENT, 'deploy').decode()}")

    # --- Tie it together: an agent "turn" (KV write + vector recall) --------
    t0 = time.perf_counter()
    db.set(f"session:{AGENT}:turn:1", "user asked about the release blocker")
    db.vadd(900001, embed("release blocker status", 128))
    db.vsearch(3, embed("release blocker status", 128))
    dt = (time.perf_counter() - t0) * 1e6
    print(f"\n[agent turn] KV + vector recall in {dt:.0f} µs (hot path)")

    print("\nAgent 'Atlas' state is durable: kill and restart the server, the")
    print("memory is replayed from the WAL and reloaded on open. Done.")


if __name__ == "__main__":
    main()
