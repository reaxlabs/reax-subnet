#!/usr/bin/env python3
"""Offline GPU parity check. Download the pinned public revision separately, then run with network disabled.

python tools/joint_readout_gpu_check.py MODEL_DIR items.jsonl --limit 300
Only the exact pinned snapshot is accepted (HF cache snapshot basename or --revision receipt).
No author code, native joint prompt, saved temperature, or benchmark scoring changes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reax_subnet.spec.joint import JointRunner, PINNED_REPO, PINNED_REVISION
from reax_subnet.spec import prompt as P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("items", type=Path)
    parser.add_argument("--revision", default="", help="Pinned download receipt for a non-cache directory")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.revision or args.model_dir.resolve().name) != PINNED_REVISION:
        parser.error(f"requires {PINNED_REPO}@{PINNED_REVISION}; supply the pinned download receipt")
    if not 1 <= args.limit <= 1000:
        parser.error("--limit must be 1..1000")
    runner = JointRunner.load(str(args.model_dir), device=args.device)
    n = correct = 0
    brier = elapsed = 0.0
    with args.items.open() as fh:
        for line in fh:
            item = json.loads(line)
            if "request" in item:
                request = item["request"]
                if request.get("images") or len(request["questions"]) != 1:
                    continue
                qid, question = next(iter(request["questions"].items()))
                row = {"state": request["state"], "question": question}
                gold = item["gold"][qid]
            else:
                row, gold = item, item["gold"]
            try:
                keys, _ = P.options(row["question"])
            except P.PromptError:
                continue
            if row["question"]["type"] == "noul":
                gold = "yes" if gold is True or str(gold).lower() in ("true", "yes", "1") else "no"
                keys = ["no", "yes"]
            gold = str(gold)
            if gold not in keys:
                raise ValueError("gold outside option keys")
            start = time.perf_counter()
            probs = runner.decide([row])[0]
            seconds = time.perf_counter() - start  # runner copies logits to CPU, synchronizing CUDA
            if probs is None:
                raise ValueError("item failed (no silent exclusion from accuracy)")
            correct += max(probs, key=probs.get) == gold
            brier += sum((probs[k] - (k == gold)) ** 2 for k in keys)
            elapsed += seconds
            n += 1
            print(json.dumps({"item": n, "seconds": seconds}), flush=True)
            if n >= args.limit:
                break
    print(json.dumps({"repo": PINNED_REPO, "revision": PINNED_REVISION, "n": n,
                      "accuracy": correct / n if n else None, "brier": brier / n if n else None,
                      "seconds_per_item": elapsed / n if n else None}))


if __name__ == "__main__":
    main()
