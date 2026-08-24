#!/usr/bin/env python3
"""locomo_diag.py — StrikeDB-side LoCoMo retrieval diagnostic.

Measures PURE RETRIEVAL quality of MEM.RECALL / MEM.RECALL.AS_OF on the
LoCoMo benchmark, independent of any extraction pipeline (memgent) and of
LLM-as-Judge:

  * ingest every turn with REAL neural embeddings (fastembed bge-small),
    session timestamp grounded in the text ("[D1 2023-05-07] Speaker: ..."),
    bi-temporal valid_from set from the session date, dia_id as `source`
  * per QA: recall k hits, check gold evidence dia_ids against top-k sources
  * report hit@k per category (1 single-hop, 2 temporal, 3 multi-hop,
    4 open-domain, 5 adversarial)

Usage:
    uv run --project /tmp/opencode/locomo-env python scripts/locomo_diag.py \
        [--data /run/media/irfan/models/OmniMemEval/data/locomo/locomo10.json]
        [--convs 1] [--k 10] [--port 6597]
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "target", "release", "dbstrike")
SCRATCH = os.path.join(REPO, ".prove-it")
HOST = "127.0.0.1"


class Resp:
    def __init__(self, port):
        self.s = socket.create_connection((HOST, port), timeout=30)
        self.buf = b""

    def _line(self):
        while b"\r\n" not in self.buf:
            d = self.s.recv(1 << 16)
            if not d:
                raise ConnectionError("closed")
            self.buf += d
        line, _, self.buf = self.buf.partition(b"\r\n")
        return line

    def _n(self, n):
        while len(self.buf) < n:
            d = self.s.recv(1 << 16)
            if not d:
                raise ConnectionError("closed")
            self.buf += d
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read(self):
        line = self._line()
        t, arg = line[:1], line[1:]
        if t == b"+":
            return arg.decode()
        if t == b"-":
            return Exception(arg.decode())
        if t == b":":
            return int(arg)
        if t == b"$":
            n = int(arg)
            return None if n == -1 else self._n(n + 2)[:-2]
        if t == b"*":
            return [self._read() for _ in range(int(arg))]
        raise ValueError(t)

    def cmd(self, *args):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            b = a if isinstance(a, (bytes, bytearray)) else str(a).encode()
            out += f"${len(b)}\r\n".encode() + bytes(b) + b"\r\n"
        self.s.sendall(out)
        return self._read()


def start(port, wal):
    p = subprocess.Popen(
        [BIN, f"{HOST}:{port}"], cwd=REPO,
        env={**os.environ, "DBSTRIKE_WAL": wal, "DBSTRIKE_SYNC": "1"},
        stdout=open(os.path.join(SCRATCH, "locomodiag.log"), "ab"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    for _ in range(200):
        try:
            Resp(port).cmd("PING")
            return p
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("no server")


def to_epoch(date_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(date_str.strip(), fmt).timestamp())
        except ValueError:
            continue
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/run/media/irfan/models/OmniMemEval/data/locomo/locomo10.json")
    ap.add_argument("--convs", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--port", type=int, default=6597)
    ap.add_argument("--qprefix", action="store_true",
                    help="bge query instruction prefix on queries only")
    ap.add_argument("--embed-url", default=None,
                    help="OpenAI-compatible /v1/embeddings endpoint (e.g. nomic)")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--granularity", choices=["turn", "sentence"], default="turn")
    args = ap.parse_args()

    import urllib.request

    class HttpEmbedder:
        """OpenAI-compatible embeddings with nomic task prefixes."""

        DOC_PREFIX = "search_document: "
        QUERY_PREFIX = "search_query: "

        def __init__(self, url, model):
            self.url = url
            self.model = model

        def _post(self, texts, batch=32):
            import concurrent.futures as cf
            chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]

            def one(chunk):
                body = json.dumps({"input": chunk, "model": self.model}).encode()
                req = urllib.request.Request(self.url, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    obj = json.load(r)
                return [d["embedding"] for d in obj["data"]]

            t0 = time.time()
            done = [None] * len(chunks)
            with cf.ThreadPoolExecutor(max_workers=6) as ex:
                futs = {ex.submit(one, ch): i for i, ch in enumerate(chunks)}
                for n, fut in enumerate(cf.as_completed(futs), 1):
                    done[futs[fut]] = fut.result()
                    print(f"    embedded chunk {n}/{len(chunks)} "
                          f"({time.time()-t0:.1f}s)", flush=True)
            return [v for ch in done for v in ch]

    if args.embed_url:
        E = HttpEmbedder(args.embed_url, args.embed_model)

        def emb_batch(texts, is_query=False):
            pre = E.QUERY_PREFIX if is_query else E.DOC_PREFIX
            return E._post([pre + t for t in texts])
    else:
        from fastembed import TextEmbedding
        print("loading bge-small-en-v1.5 ...", flush=True)
        model = TextEmbedding("BAAI/bge-small-en-v1.5")

        def emb_batch(texts, is_query=False):
            pre = ("Represent this sentence for searching relevant passages: "
                   if is_query and args.qprefix else "")
            return [list(v) for v in model.embed([pre + t for t in texts])]

    data = json.load(open(args.data))[: args.convs]

    # pre-flight: how fast is the embedder? prints expected wall time so the
    # run never looks stuck.
    _t = time.time()
    emb_batch(["warmup probe sentence"] * 8, is_query=False)
    per8 = time.time() - _t
    n_turns = sum(
        len(v) for s in data for k, v in s["conversation"].items()
        if k.startswith("session_") and not k.endswith("_date_time")
    )
    n_q = sum(len(s["qa"]) for s in data)
    est = (n_turns + n_q) / 8 * per8
    print(f"embedder warmup: 8 texts in {per8:.2f}s -> "
          f"estimated embedding time ~{est:.0f}s for {n_turns} turns + {n_q} queries",
          flush=True)

    wal = os.path.join(SCRATCH, f"locomo_k{args.k}.wal")
    for f in [wal] + [wal + s for s in (".snap", ".bak-*")]:
        pass
    import glob
    for f in [wal] + glob.glob(wal + "*"):
        if os.path.exists(f):
            os.remove(f)
    p = start(args.port, wal)
    c = Resp(args.port)

    CATS = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain", 5: "adversarial"}

    total_ingest = 0
    for si, sample in enumerate(data):
        agent = f"conv{sample.get('sample_id', si)}"
        conv = sample["conversation"]
        sessions = sorted(
            [(int(k.split("_")[1]), v) for k, v in conv.items()
             if k.startswith("session_") and not k.endswith("_date_time")],
        )
        # Pre-collect all turns for this conversation, then ONE batch embed.
        import re
        _sent = re.compile(r"(?<=[.!?])\s+")
        jobs = []  # (text, src, vf)
        for sess_num, turns in sessions:
            dt_raw = conv.get(f"session_{sess_num}_date_time", "")
            vf = to_epoch(dt_raw)
            for turn in turns:
                text = f"[{dt_raw}] {turn['speaker']}: {turn['text']}"
                if args.granularity == "sentence":
                    for frag in _sent.split(turn["text"]):
                        if len(frag.strip()) < 12:
                            continue
                        jobs.append((f"[{dt_raw}] {turn['speaker']}: {frag.strip()}",
                                     turn["dia_id"], vf))
                else:
                    jobs.append((text, turn["dia_id"], vf))
        print(f"conv{si}: embedding {len(jobs)} turns (batched) ...", flush=True)
        vecs = emb_batch([j[0] for j in jobs], is_query=False)
        t0 = time.time()
        for (text, src, vf), vec in zip(jobs, vecs):
            r = c.cmd("MEM.REMEMBER.T", "AGENT", agent, text, src, 0.8,
                      vf, 0, *[round(float(x), 4) for x in vec])
            if not isinstance(r, int):
                print("INGEST ERR:", r)
                break
            total_ingest += 1
        print(f"conv{si}: ingested {len(jobs)} in {time.time()-t0:.1f}s", flush=True)
    print(f"ingested {total_ingest} memories across {len(data)} conversation(s)")

    # ── retrieval ────────────────────────────────────────────────────────
    stats = {}  # cat -> [hit@1, hit@5, hit@10, count]
    lat_sum = lat_n = 0
    for si, sample in enumerate(data):
        agent = f"conv{sample.get('sample_id', si)}"
        qas = sample["qa"]
        print(f"conv{si}: embedding {len(qas)} queries (batched) ...", flush=True)
        qvecs = emb_batch([qa["question"] for qa in qas], is_query=True)
        for qa, qv in zip(qas, qvecs):
            q = qa["question"]
            gold = set(qa.get("evidence") or [])
            cat = qa["category"]
            t0 = time.time()
            # query text feeds the BM25 leg; the vector feeds the dense leg
            hits = c.cmd("MEM.RECALL", "AGENT", agent, args.k, q,
                         *[round(float(x), 4) for x in qv])
            lat_sum += time.time() - t0
            lat_n += 1
            srcs = []
            if isinstance(hits, list):
                # flat: id score source text created vf vt (7 per hit)
                for i in range(0, len(hits) - 6, 7):
                    s = hits[i + 2]
                    srcs.append(s.decode() if isinstance(s, bytes) else str(s))
            st = stats.setdefault(cat, [0, 0, 0, 0])
            st[3] += 1
            for rank, s in enumerate(srcs[: args.k], 1):
                if s in gold:
                    if rank <= 1: st[0] += 1
                    if rank <= 5: st[1] += 1
                    st[2] += 1
                    break

    print(f"\nretrieval-only hit rates (k={args.k}, {lat_n} queries, "
          f"avg {1000*lat_sum/max(lat_n,1):.2f} ms):\n")
    print(f"{'category':<14}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'n':>6}")
    agg = [0, 0, 0, 0]
    for cat in sorted(stats):
        h1, h5, h10, n = stats[cat]
        name = CATS.get(cat, str(cat))
        print(f"{name:<14}{h1/n:>8.1%}{h5/n:>8.1%}{h10/n:>8.1%}{n:>6}")
        for i in range(4):
            agg[i] += stats[cat][i]
    print(f"{'OVERALL':<14}{agg[0]/agg[3]:>8.1%}{agg[1]/agg[3]:>8.1%}"
          f"{agg[2]/agg[3]:>8.1%}{agg[3]:>6}")


if __name__ == "__main__":
    import atexit
    import signal as _signal
    _proc = {"p": None}

    def _cleanup():
        p = _proc["p"]
        if p is not None:
            try:
                os.killpg(os.getpgid(p.pid), _signal.SIGKILL)
            except Exception:
                pass

    atexit.register(_cleanup)

    orig_start = start

    def start_tracked(port, wal):
        p = orig_start(port, wal)
        _proc["p"] = p
        return p

    start = start_tracked
    sys.exit(main())
