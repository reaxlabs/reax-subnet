"""Small stdlib helpers: hashes, canonical JSON, JSONL io, atomic writes."""
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    """Stable serialisation used for the manifest hash (sorted keys, no spaces, ASCII)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def content_sha256(state, questions, answers):
    """The item hash every loader checks: what the model sees (state, questions) plus the gold answers.
    Key order is KEPT (not sorted): the order of the options is part of the prompt, so a reordered item must not match."""
    return sha256_text(json.dumps({"state": state, "questions": questions, "answers": answers}, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False))


def open_text(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def iter_jsonl(path):
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl_tolerant(path):
    """Read a journal, ignoring a torn last line (a run killed during an append)."""
    out = []
    path = Path(path)
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i >= len(lines) - 2:
                continue
            raise
    return out


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
