"""The four public families, their weights, and the item loaders.

Every item is pinned by `content_sha256` = sha256(canonical({"state", "questions", "answers"})) in
`items/<family>/index.jsonl`. A loader only hands an item to the runner when its hash matches; an item that cannot be
loaded or does not match is reported (never silently dropped, never scored with different text).

  PV-HF   300 items from the Hugging Face Decision Index public suite   -> ids + hashes; you rebuild the suite locally
  PV-HLE  509 Humanity's Last Exam multiple-choice items                -> ids + hashes; loaded from cais/hle (gated)
  PV-W2C  3,423 When2Call multiple-choice items (CC BY 4.0)             -> text shipped in items/when2call/
  PV-ONET 1,000 EvalSafe O*NET questions (Apache-2.0)                   -> text shipped in items/onet/
Why two families ship only ids: see items/README.md and THIRD-PARTY.md.
"""
import hashlib
import re
from pathlib import Path

from .common import content_sha256, iter_jsonl, read_json

ITEMS_DIR = Path(__file__).resolve().parent.parent / "items"

FAMILIES = {
    "PV-HF": {"style": "hf", "dir": "di", "mode": "di-suite"},
    "PV-HLE": {"style": "aa", "dir": "hle", "mode": "hle"},
    "PV-W2C": {"style": "aa", "dir": "when2call", "mode": "text"},
    "PV-ONET": {"style": "aa", "dir": "onet", "mode": "text"},
}
# Declared preview weights (the same numbers the preview uses): HF public tasks 0.4, the three AA-style families 0.2 each.
WEIGHTS = {"PV-HF": 0.4, "PV-HLE": 0.2, "PV-W2C": 0.2, "PV-ONET": 0.2}

HLE_SUFFIX = "\nWhich option is the correct answer?"
HLE_INSTRUCTIONS = "Which option is the correct answer?"


def manifest(items_dir=ITEMS_DIR):
    return read_json(Path(items_dir) / "MANIFEST.json")


def load_index(fam, items_dir=ITEMS_DIR):
    return list(iter_jsonl(Path(items_dir) / FAMILIES[fam]["dir"] / "index.jsonl"))


def subsample(ids, n, seed="reax-yardstick"):
    """First n ids in a seeded hash order (deterministic, machine independent)."""
    if not n or n >= len(ids):
        return list(ids)
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).hexdigest())[:n]


def _accept(index_row, state, questions, answers, out, report):
    rid = index_row["id"]
    if content_sha256(state, questions, answers) != index_row["content_sha256"]:
        report["hash_mismatch"].append(rid)
        return False
    out.append(({"state": state, "questions": questions, "_evaluation": {"run_id": rid}}, {"id": rid, "answers": answers}))
    return True


def _new_report(fam, n_index):
    return {"family": fam, "indexed": n_index, "loaded": 0, "missing": [], "hash_mismatch": [], "note": None}


# --------------------------------------------------------------------------------------------------- text families
def load_text(fam, wanted, items_dir=ITEMS_DIR):
    d = Path(items_dir) / FAMILIES[fam]["dir"]
    index = {r["id"]: r for r in iter_jsonl(d / "index.jsonl") if r["id"] in wanted}
    reqs = {r["id"]: r for r in iter_jsonl(d / "requests.jsonl.gz") if r["id"] in index}
    golds = {g["id"]: g for g in iter_jsonl(d / "gold.jsonl.gz") if g["id"] in index}
    out, report = [], _new_report(fam, len(index))
    for rid in sorted(index):
        if rid not in reqs or rid not in golds:
            report["missing"].append(rid)
            continue
        _accept(index[rid], reqs[rid]["state"], reqs[rid]["questions"], golds[rid]["answers"], out, report)
    report["loaded"] = len(out)
    return out, report


# ------------------------------------------------------------------------------- PV-HF: Decision Index public suite
def _suite_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(x for x in p.rglob("*") if x.name.endswith((".jsonl", ".jsonl.gz")))
        else:
            files.append(p)
    return files


def load_di(wanted, suite_paths, items_dir=ITEMS_DIR):
    """Read rows of a locally rebuilt Decision Index public suite (`python -m decision_index suite rebuild`, see
    README) and pick the PV-HF items by run id. Suite rows carry `state`, `questions`, `expected` and `_evaluation.run_id`."""
    index = {r["id"]: r for r in load_index("PV-HF", items_dir) if r["id"] in wanted}
    report = _new_report("PV-HF", len(index))
    if not suite_paths:
        report["missing"] = sorted(index)
        report["note"] = "no --di-suite given: PV-HF not loaded (rebuild the Decision Index public suite first, see README)"
        return [], report
    by_key = {}
    for r in index.values():
        by_key[r["id"]] = r["id"]
        if r.get("source_run_id"):
            by_key.setdefault(r["source_run_id"], r["id"])
    found = {}
    for f in _suite_files(suite_paths):
        for row in iter_jsonl(f):
            rid = (row.get("_evaluation") or {}).get("run_id")
            key = by_key.get(rid)
            if key is None or "state" not in row or "questions" not in row:
                continue
            cand = (row["state"], row["questions"], row.get("expected"))
            # keep the first candidate whose hash matches; a later file (e.g. the rebuilt GSM8K rows) may hold the match
            if key not in found or content_sha256(*found[key]) != index[key]["content_sha256"]:
                found[key] = cand
    out = []
    for rid in sorted(index):
        if rid not in found:
            report["missing"].append(rid)
            continue
        state, questions, expected = found[rid]
        _accept(index[rid], state, questions, expected, out, report)
    report["loaded"] = len(out)
    return out, report


# ----------------------------------------------------------------------------------------- PV-HLE: cais/hle (gated)
_OPT = re.compile(r"^\s*([A-Z])[\.\)]\s?(.*)$")


def hle_candidates(question, decline_key, decline_text):
    """Possible (state, criteria) renderings of one HLE multiple-choice row. The item hash decides which one is ours;
    several are tried because the published question text varies in whitespace around the answer choices."""
    parts = re.split(r"\n[ \t]*Answer Choices:[ \t]*\n", question, maxsplit=1)
    if len(parts) != 2:
        return []
    stem, block = parts
    opts, cur = [], None
    for line in block.split("\n"):
        m = _OPT.match(line)
        if m and (not opts or ord(m.group(1)) == ord(opts[-1][0]) + 1):
            opts.append([m.group(1), m.group(2)])
        elif opts:
            opts[-1][1] += "\n" + line
    if not opts or opts[0][0] != "A":
        return []
    out = []
    for strip_opt in (str.strip, str.rstrip, lambda s: s):
        crit = {k: strip_opt(v) for k, v in opts}
        crit[decline_key] = decline_text
        for s in {stem.strip(), stem.rstrip()}:
            out.append((s + HLE_SUFFIX, crit))
    return out


def _iter_hle(source):
    """Rows of cais/hle from a local .parquet (needs pyarrow), a .jsonl(.gz) export, or 'hub' (needs `datasets` and an
    account that accepted the dataset terms)."""
    if source == "hub":
        from datasets import load_dataset  # optional dependency
        yield from load_dataset("cais/hle", split="test")
        return
    if str(source).endswith(".parquet"):
        import pyarrow.parquet as pq  # optional dependency
        table = pq.read_table(source, columns=["id", "question", "answer", "answer_type", "image"])
        yield from table.to_pylist()
        return
    yield from iter_jsonl(source)


def load_hle(wanted, source, items_dir=ITEMS_DIR):
    index = {r["id"]: r for r in load_index("PV-HLE", items_dir) if r["id"] in wanted}
    report = _new_report("PV-HLE", len(index))
    if not source:
        report["missing"] = sorted(index)
        report["note"] = "no --hle given: PV-HLE not loaded (accept the cais/hle terms on Hugging Face, see README)"
        return [], report
    by_src = {r["source_id"]: r for r in index.values()}
    out, seen = [], set()
    for row in _iter_hle(source):
        ix = by_src.get(str(row.get("id")))
        if ix is None or ix["id"] in seen:
            continue
        seen.add(ix["id"])
        answers = {"answer": str(row.get("answer", "")).strip()}
        ok = False
        for state, crit in hle_candidates(row["question"], ix["decline_key"], ix["decline_text"]):
            questions = {"answer": {"type": "choice", "instructions": HLE_INSTRUCTIONS, "criteria": crit}}
            if content_sha256(state, questions, answers) == ix["content_sha256"]:
                out.append(({"state": state, "questions": questions, "_evaluation": {"run_id": ix["id"]}}, {"id": ix["id"], "answers": answers}))
                ok = True
                break
        if not ok:
            report["hash_mismatch"].append(ix["id"])
    report["missing"] = sorted(set(index) - seen)
    report["loaded"] = len(out)
    return out, report


# --------------------------------------------------------------------------------------------------------- public
def load_family(fam, wanted=None, di_suite=None, hle_source=None, items_dir=ITEMS_DIR):
    """Returns ([(request_row, gold_row)], loader report). `wanted` = set of ids (default: all indexed ids)."""
    if wanted is None:
        wanted = {r["id"] for r in load_index(fam, items_dir)}
    mode = FAMILIES[fam]["mode"]
    if mode == "text":
        return load_text(fam, wanted, items_dir)
    if mode == "di-suite":
        return load_di(wanted, di_suite, items_dir)
    return load_hle(wanted, hle_source, items_dir)
