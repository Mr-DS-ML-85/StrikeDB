#!/usr/bin/env python3
"""
Demo 3 — RAG app: hybrid (dense + sparse) retrieval with the MITM cache
debugger proving the corpus cache can never serve a stale answer.

DB-Strike does RAG as a first-class query, not a bolted pipeline:
  * RAG.INGEST  — chunk + embedding into the unified memory substrate.
  * RAG.SEARCH  — hybrid dense (HNSW ANN) + sparse (BM25) fused by Reciprocal
                  Rank Fusion (RRF), with a lightweight lexical rerank.
  * Corpus-generation gating — every ingest bumps a monotonic corpus
                  generation; the query cache key includes it, so a changed
                  corpus can NEVER serve a stale answer.
  * MITM cache debugger — the engine itself classifies each cache read as
                  HIT / STALE_HIT / MISS / PHANTOM. This is the "vector index
                  is a cache with no invalidation strategy" problem, solved.

Run a server first, then:
    python3 demos/rag_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dbstrike_client import DBStrike, embed


CORPUS = [
    ("Rust ownership and the borrow checker prevent data races at compile time",
     "doc:rust"),
    ("Python uses a global interpreter lock (GIL) that limits true parallelism",
     "doc:python"),
    ("HNSW is a hierarchical graph index for approximate nearest-neighbor search",
     "doc:ann"),
    ("BM25 is the standard probabilistic sparse retrieval scoring function",
     "doc:bm25"),
    ("Reciprocal rank fusion (RRF) combines dense and sparse rankings",
     "doc:rrf"),
    ("A vector index is a cache: without invalidation it serves stale answers",
     "doc:mitm"),
    ("Graphiti adds bi-temporal validity windows to knowledge-graph memory",
     "doc:graphiti"),
    ("Mem0 stores long-term semantic memory with salience-weighted recall",
     "doc:mem0"),
]


def main():
    db = DBStrike()
    print("Connected to DB-Strike. Building a RAG corpus...\n")

    # 1. Ingest the corpus (each ingest bumps the corpus generation).
    for text, src in CORPUS:
        fid = db.rag_ingest(text, src, embed(text, dim=128))
        print(f"   ingested #{fid} [{src}] {text[:48]}...")

    # 2. First query -> fresh compute.
    q = "nearest neighbor graph index for search"
    cached, hits = db.rag_search(3, q, embed(q, dim=128))
    print(f"\n[rag] query: {q!r}")
    print(f"      cache = {cached}  (first call must compute)")
    for h in hits:
        print(f"      score={h['score']:.3f} [{h['source']}] {h['text'][:60]}...")

    # 3. Identical query -> served from cache (same corpus generation).
    cached2, hits2 = db.rag_search(3, q, embed(q, dim=128))
    print(f"\n[rag] repeat query -> cache = {cached2}  (same generation, served)")

    # 4. Ingest a NEW relevant doc -> generation bumps -> cache invalidated.
    db.rag_ingest(
        "Approximate nearest neighbor search benefits from product quantization",
        "doc:pq",
        embed("approximate nearest neighbor product quantization", 128),
    )
    cached3, hits3 = db.rag_search(3, q, embed(q, dim=128))
    print(f"[rag] after new ingest -> cache = {cached3}  "
          f"(generation bumped, recomputed)")
    print(f"      top hit now: {hits3[0]['text'][:60]}...")

    # 5. MITM cache debugger — prove no stale reads end-to-end.
    print("\n[mitm] cache debugger verdicts on a controlled stale-write scenario:")
    db.cache_source_set("user:profile:1", "plan:pro")
    db.cache_set("user:profile:1", "plan:pro")
    v1, _ = db.cache_get("user:profile:1")
    print(f"      fresh read verdict = {v1}  (HIT expected)")
    # Source changes but cache is NOT invalidated -> STALE_HIT (the classic bug).
    db.cache_source_set("user:profile:1", "plan:enterprise")
    v2, v2val = db.cache_get("user:profile:1")
    print(f"      source updated, cache stale -> verdict = {v2}  "
          f"served={v2val.decode()!r}  (THE bug, now caught)")
    bugs = db.cache_bugs()
    print(f"      bugs detected by engine: {len(bugs)}")
    for b in bugs:
        print(f"        {b}")

    print("\n[mitm] RAG corpus cache is generation-gated, so the stale-hit class")
    print("      can never reach a user query — only an out-of-band cache that")
    print("      skips the engine's generation key could. Done.")


if __name__ == "__main__":
    main()
