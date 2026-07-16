#!/usr/bin/env python3
"""
Lightweight RESP client for DB-Strike.

Talks the Redis wire protocol (RESP2) so it works against the release binary
with *any* Redis client too. This module is a thin wrapper that gives the demo
apps a clean, typed surface (kv / vector / memory / rag / cache) over the raw
commands the server dispatches in crates/server/src/main.rs.

    from dbstrike_client import DBStrike
    db = DBStrike("127.0.0.1", 6380)
    db.set("user:1", "ada")
    db.vadd(42, [0.1, 0.2, ...])
"""

import socket
import math
import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("DBSTRIKE_PORT", "6380"))


class ProtocolError(Exception):
    pass


class DBStrike:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""

    # ---- low-level RESP -------------------------------------------------
    def cmd(self, *args):
        out = [f"*{len(args)}\r\n".encode()]
        for a in args:
            if isinstance(a, (int, float)):
                a = str(a)
            if isinstance(a, str):
                a = a.encode()
            out.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
        self.sock.sendall(b"".join(out))
        return self._parse()

    def _read_line(self):
        while b"\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\r\n", 1)
        return line

    def _read_n(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("server closed")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _parse(self):
        line = self._read_line()
        t, rest = line[:1], line[1:]
        if t == b"+":
            return rest.decode()
        if t == b"-":
            return ProtocolError(rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n == -1:
                return None
            return self._read_n(n + 2)[:-2]
        if t == b"*":
            n = int(rest)
            if n == -1:
                return None
            return [self._parse() for _ in range(n)]
        raise ProtocolError(f"bad RESP type {t!r}")

    def pipeline(self, *cmds):
        """Send a batch of argument-tuples; returns the list of replies."""
        for c in cmds:
            out = [f"*{len(c)}\r\n".encode()]
            for a in c:
                if isinstance(a, (int, float)):
                    a = str(a)
                if isinstance(a, str):
                    a = a.encode()
                out.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
            self.sock.sendall(b"".join(out))
        return [self._parse() for _ in cmds]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # ---- KV --------------------------------------------------------------
    def ping(self):
        return self.cmd("PING")

    def set(self, key, value):
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("SET", key, value)

    def get(self, key):
        return self.cmd("GET", key)

    def delete(self, key):
        return self.cmd("DEL", key)

    def incr(self, key, by=1):
        return self.cmd("INCRBY", key, by) if by != 1 else self.cmd("INCR", key)

    def keys(self, prefix):
        res = self.cmd("KEYS", prefix)
        return [k.decode() for k in res] if res else []

    # ---- VECTOR ----------------------------------------------------------
    def vadd(self, vid, vec):
        return self.cmd("VADD", vid, *[round(x, 6) for x in vec])

    def vsearch(self, k, vec):
        raw = self.cmd("VSEARCH", k, *[round(x, 6) for x in vec])
        if not raw:
            return []
        out = []
        for i in range(0, len(raw), 2):
            out.append((int(raw[i]), float(raw[i + 1])))
        return out

    # ---- TIME SERIES -----------------------------------------------------
    def tsadd(self, series, ts, val):
        return self.cmd("TSADD", series, ts, val)

    def tsrange(self, series, frm, to):
        raw = self.cmd("TSRANGE", series, frm, to)
        if not raw:
            return []
        return [(int(raw[i]), int(raw[i + 1])) for i in range(0, len(raw), 2)]

    # ---- REDUCERS (fuel-metered) ----------------------------------------
    def reduce(self, name, shardkey, key, by):
        return self.cmd("REDUCE", name, shardkey, key, by)

    # ---- AGENT MEMORY ----------------------------------------------------
    def remember(self, text, source, salience, vec):
        return self.cmd("MEM.REMEMBER", text, source, salience, *[round(x, 6) for x in vec])

    def recall(self, k, query, vec):
        raw = self.cmd("MEM.RECALL", k, query, *[round(x, 6) for x in vec])
        return self._recall_hits(raw)

    def link(self, frm, to, rel, weight=1.0):
        return self.cmd("MEM.LINK", frm, to, rel, weight)

    def unlink(self, frm, to, rel):
        return self.cmd("MEM.UNLINK", frm, to, rel)

    def neighbors(self, frm, rel=""):
        raw = self.cmd("MEM.NEIGH", frm, rel) if rel else self.cmd("MEM.NEIGH", frm)
        if not raw:
            return []
        return [(int(raw[i]), raw[i + 1].decode(), float(raw[i + 2]))
                for i in range(0, len(raw), 3)]

    def traverse(self, start, depth, rel=""):
        raw = self.cmd("MEM.TRAV", start, depth, rel) if rel else self.cmd("MEM.TRAV", start, depth)
        return [int(x) for x in raw] if raw else []

    def remember_temporal(self, text, source, salience, vf, vt, vec):
        return self.cmd("MEM.REMEMBER.T", text, source, salience, vf, vt,
                        *[round(x, 6) for x in vec])

    def invalidate_fact(self, fid, at):
        return self.cmd("MEM.INVALIDATE", fid, at)

    def recall_as_of(self, k, query, as_of, vec):
        raw = self.cmd("MEM.RECALL.AS_OF", k, query, as_of, *[round(x, 6) for x in vec])
        return self._recall_hits(raw)

    def forget(self, fid):
        return self.cmd("MEM.FORGET", fid)

    def proc_set(self, agent, name, body):
        if isinstance(body, str):
            body = body.encode()
        return self.cmd("MEM.PROC.SET", agent, name, body)

    def proc_get(self, agent, name):
        return self.cmd("MEM.PROC.GET", agent, name)

    def proc_list(self, agent):
        res = self.cmd("MEM.PROC.LIST", agent)
        return [n.decode() for n in res] if res else []

    # ---- WORKING MEMORY (STM) ----------------------------------------
    def wm_set(self, agent, key, value, ttl_ms):
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("MEM.WM_SET", agent, key, value, ttl_ms)

    def wm_get(self, agent, key):
        return self.cmd("MEM.WM_GET", agent, key)

    def wm_delete(self, agent, key):
        return self.cmd("MEM.WM_DELETE", agent, key)

    # ---- EPISODIC MEMORY --------------------------------------------
    def episode(self, agent, kind, payload):
        if isinstance(payload, str):
            payload = payload.encode()
        return self.cmd("MEM.EPISODE", agent, kind, payload)

    def episodes(self, agent, limit):
        raw = self.cmd("MEM.EPISODES", agent, limit)
        if not raw:
            return []
        out = []
        for e in raw:
            out.append({
                "seq": int(e[0]),
                "kind": e[1].decode(),
                "payload": e[2],
            })
        return out

    def episode_forget(self, agent, seq):
        return self.cmd("MEM.EPISODE_FORGET", agent, seq)

    def _recall_hits(self, raw):
        if not raw:
            return []
        out = []
        for i in range(0, len(raw), 4):
            out.append({
                "id": int(raw[i]),
                "score": float(raw[i + 1]),
                "source": raw[i + 2].decode(),
                "text": raw[i + 3].decode(),
            })
        return out

    # ---- RAG -------------------------------------------------------------
    def rag_ingest(self, text, source, vec):
        return self.cmd("RAG.INGEST", text, source, *[round(x, 6) for x in vec])

    def rag_search(self, k, query, vec):
        raw = self.cmd("RAG.SEARCH", k, query, *[round(x, 6) for x in vec])
        if not raw:
            cached = raw[0].decode() if raw else None
            return cached, []
        cached = raw[0].decode()
        hits = []
        for i in range(1, len(raw), 4):
            hits.append({
                "id": int(raw[i]),
                "score": float(raw[i + 1]),
                "source": raw[i + 2].decode(),
                "text": raw[i + 3].decode(),
            })
        return cached, hits

    # ---- MITM CACHE DEBUGGER --------------------------------------------
    def cache_set(self, key, value):
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("CACHE.SET", key, value)

    def cache_source_set(self, key, value):
        if isinstance(value, str):
            value = value.encode()
        return self.cmd("CACHE.SRCSET", key, value)

    def cache_get(self, key):
        raw = self.cmd("CACHE.GET", key)
        if not raw:
            return None, None
        verdict = raw[0].decode()
        val = raw[1]
        return verdict, val

    def cache_invalidate(self, key):
        return self.cmd("CACHE.INVALIDATE", key)

    def cache_bugs(self):
        res = self.cmd("CACHE.BUGS")
        return [b.decode() for b in res] if res else []

    def cache_traces(self):
        res = self.cmd("CACHE.TRACES")
        return [t.decode() for t in res] if res else []


# ---------------------------------------------------------------------------
# Deterministic pseudo-embedding helpers (no external model needed for demos).
# In production you'd swap these for a real embedding endpoint (OpenAI,
# sentence-transformers, a local llama.cpp embedder, etc.).
# ---------------------------------------------------------------------------
def embed(text, dim=128, seed=None):
    """Hashed bag-of-tokens pseudo-embedding in [-1,1], L2-normalized."""
    import random
    rnd = random.Random(seed if seed is not None else text.__hash__() & 0xFFFFFFFF)
    vec = [0.0] * dim
    toks = [t for t in text.lower().split() if len(t) > 2]
    if not toks:
        toks = ["__empty__"]
    for t in toks:
        h = rnd.randint(0, dim - 1)
        vec[h] += 1.0
    # light noise so identical-token-set sentences aren't identical vectors
    for i in range(dim):
        vec[i] += rnd.uniform(-0.05, 0.05)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
