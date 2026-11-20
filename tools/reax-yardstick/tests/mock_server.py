#!/usr/bin/env python3
"""Gold-aware CPU mock of the vLLM endpoints the readout uses (no model, no GPU, no network).

  /health, /tokenize (chat -> handle tokens; label -> label tokens), /v1/completions with logprob_token_ids

A mock "model" is a skill s in [0, 1]: for each (item, question) it knows the gold answer with probability s
(decided by a hash of seed|item|question), otherwise it prefers a hash-chosen wrong label. Labels longer than 12
characters become 2 tokens, so the label-prefix trie is exercised.
"""
import argparse
import hashlib
import json
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reax_yardstick.common import iter_jsonl  # noqa: E402
from reax_yardstick.items import FAMILIES  # noqa: E402,F401
from reax_yardstick.readout import hf_wire, render_prompt, render_prompt_aa  # noqa: E402

HANDLE_BASE = 1_000_000
LABEL_BASE = 2_000_000


def h01(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def label_tokens(label):
    label = str(label)
    if len(label) <= 12:
        return [LABEL_BASE + zlib.crc32(label.encode()) % 900_000]
    return [LABEL_BASE + zlib.crc32(label[:12].encode()) % 900_000, LABEL_BASE + 1_000_000 + zlib.crc32(label.encode()) % 900_000]


def row_id(row):
    return (row.get("_evaluation") or {}).get("run_id") or row["id"]


class Model:
    def __init__(self, rows, golds, skill, seed):
        self.skill, self.seed = skill, seed
        self.entries, self.by_content = [], {}
        for row in rows:
            rid = row_id(row)
            g = golds.get(rid, {}).get("answers", {})
            wired = hf_wire(row)["questions"]
            for qid, q in wired.items():
                labels = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
                contents = [render_prompt(row["state"], q)]
                if isinstance(row["questions"][qid].get("criteria", {}), dict):
                    contents.append(render_prompt_aa(row["state"], row["questions"][qid]))
                for content in contents:
                    self.by_content.setdefault(content, len(self.entries))
                target = g.get(qid)
                if isinstance(target, bool):
                    target = "true" if target else "false"
                self.entries.append({"rid": rid, "qid": qid, "labels": labels, "target": None if target is None else str(target)})

    def scores(self, entry):
        labels, target = entry["labels"], entry["target"]
        knows = target in labels and h01(self.seed, entry["rid"], entry["qid"], "k") < self.skill
        if knows:
            fav = target
        else:
            wrong = [l for l in labels if l != target] or labels
            fav = wrong[int(h01(self.seed, entry["rid"], entry["qid"], "w") * len(wrong)) % len(wrong)]
        return {l: (-0.1 if l == fav else -2.5 - h01(self.seed, entry["rid"], entry["qid"], l)) for l in labels}


def make_handler(model, delay):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"ok": True}) if self.path == "/health" else self._send({}, 404)

        def do_POST(self):
            r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(delay)
            if self.path == "/tokenize":
                if "messages" in r:
                    idx = model.by_content.get(r["messages"][-1]["content"])
                    if idx is None:
                        return self._send({"error": "unknown prompt"}, 400)
                    return self._send({"tokens": [HANDLE_BASE + idx, 5, 6, 7]})
                return self._send({"tokens": label_tokens(r["prompt"])})
            if self.path == "/v1/completions" and "logprob_token_ids" in r:
                prompt = r["prompt"]
                entry = model.entries[prompt[0] - HANDLE_BASE]
                sc = model.scores(entry)
                seqs = {l: label_tokens(l) for l in entry["labels"]}
                prefix = prompt[4:]
                out = {}
                for tok in r["logprob_token_ids"]:
                    v = None
                    for l, seq in seqs.items():
                        if seq[:len(prefix)] == prefix and len(seq) > len(prefix) and seq[len(prefix)] == tok:
                            v = sc[l] if len(prefix) == 0 else 0.0
                    out["token_id:" + str(tok)] = v if v is not None else -20.0
                return self._send({"choices": [{"logprobs": {"top_logprobs": [out]}}]})
            self._send({}, 404)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--skill", type=float, default=0.5)
    p.add_argument("--seed", default="mock")
    p.add_argument("--rows", action="append", default=[])
    p.add_argument("--gold", action="append", default=[])
    p.add_argument("--latency-ms", type=float, default=0.0)
    a = p.parse_args()
    rows = [r for f in a.rows for r in iter_jsonl(f)]
    golds = {g["id"]: g for f in a.gold for g in iter_jsonl(f)}
    model = Model(rows, golds, a.skill, a.seed)
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(model, a.latency_ms / 1000.0)).serve_forever()


if __name__ == "__main__":
    main()
