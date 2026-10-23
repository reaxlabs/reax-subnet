"""Synthetic v5.1 fixtures (no real data): typed T / N / pub rows, v4-format public and old-tier rows, base / cand / flat answers.
Run from this directory: python gen_synthetic.py"""
import json, random
random.seed(1)
IK = "idk"
AREAS = {  # area -> [(task_type, gold, option keys, idk)]
    "knowledge": [("omniscience", False, list("abcd"), True), ("hard_mc", True, list("abcd"), False)],
    "language": [("wildguard", False, ["yes", "no"], False), ("nli", True, ["entail", "neutral", "contra"], False)],
    "retrieval": [("intent", True, list("abcde"), False), ("product_match", False, ["yes", "no"], False)],
    "tools": [("when2call", False, list("abcd"), False), ("tool_call", True, list("abc"), False)],
    "arts": [("preference", False, ["a", "b"], False)],
}
N_TYPES = [("triage", list("abc")), ("claims", ["yes", "no"]), ("clauses", list("abcd"))]
PUB_TYPES = [("pub_mc", "knowledge", list("abcd")), ("pub_nli", "language", ["entail", "neutral", "contra"])]
PER_TYPE = 15


def q(keys, exp):
    return {"instructions": "p", "criteria": {k: k.upper() for k in keys}, "expected": exp}


def gold(keys):
    return random.choice(keys)


items = []
for area, types in AREAS.items():
    for tt, g, keys, idk in types:
        for i in range(PER_TYPE):
            qs = [q(keys, gold(keys)) for _ in range(3 if tt == "wildguard" else 1)]
            items.append({"id": f"T-{tt}-{i}", "tier": "mirror", "part": "T", "area": area, "task_type": tt, "gold": g, "idk": idk,
                          "questions": qs})
for tt, keys in N_TYPES:
    for i in range(PER_TYPE):
        items.append({"id": f"N-{tt}-{i}", "tier": "mirror", "part": "N", "area": "tools", "task_type": tt, "gold": False,
                      "idk": False, "questions": [q(keys, gold(keys))]})
for tt, area, keys in PUB_TYPES:
    for i in range(PER_TYPE):
        items.append({"id": f"P-{tt}-{i}", "tier": "mirror", "part": "pub", "area": area, "task_type": tt, "gold": False,
                      "idk": False, "questions": [q(keys, gold(keys))]})
legacy = []
for tier in ("same_family", "proc_sealed", "new_family", "new_domain"):
    for i in range(30):
        legacy.append({"id": f"L-{tier}-{i}", "tier": tier, "category": "math_quant", "question": {"type": "choice"},
                       "labels": ["a", "b", "c"], "expected": random.choice("abc")})


def ans(it, skill, flat=False):
    out = []
    for qq in it["questions"]:
        keys = list(qq["criteria"]) + ([IK] if it.get("idk") else [])
        if flat:
            p = {k: 1 / len(keys) for k in keys}
        else:
            gd = qq["expected"]; r = random.random()
            pick = gd if r < skill else (IK if it.get("idk") and r < skill + .1 else random.choice([k for k in keys if k != gd]))
            p = {k: float(k == pick) for k in keys}
        out.append(p)
    return out if len(out) > 1 else out[0]


def hans(r, skill):
    gd = r["expected"]; pick = gd if random.random() < skill else random.choice([k for k in r["labels"] if k != gd])
    return {k: float(k == pick) for k in r["labels"]}


def w(path, rows):
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))


w("items_typed.jsonl", items); w("items_legacy.jsonl", legacy)
json.dump({"pool_version": "synthetic-v0", "T": {"mean": 35.0, "sd": 20.0}, "N": {"mean": 30.0, "sd": 15.0}, "Pub": {"mean": 40.0, "sd": 18.0}},
          open("panel.json", "w"), indent=1)
json.dump(["hard_mc"], open("exclude.json", "w"))
for name, skill, flat in (("base", .4, False), ("cand", .7, False), ("flat", 0, True)):
    rows = [{"id": it["id"], "probs": ans(it, skill, flat)} for it in items]
    rows += [{"id": r["id"], "probs": ({k: 1 / 3 for k in r["labels"]} if flat else hans(r, skill))} for r in legacy]
    w(f"{name}.jsonl", rows)
