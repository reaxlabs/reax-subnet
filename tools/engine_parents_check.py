#!/usr/bin/env python3
"""Operator tool for the engine copy line (docs/ENGINE-COMPETITION.md, "Reusing other entries (copy line)").

Compares the extracted file tree of a candidate engine image with the trees of other entries and with the parents the
candidate declares in its image label ``ai.reax.engine.parents``, and prints for each parent which files are
byte-identical, which of those are binaries, and the verdict: "derived (declared)", "copy (undeclared)", "same owner" or
"independent". No docker is used: extract the trees first (``docker create`` + ``docker cp <id>:/app dir``) and pass the
directories. On a mark of "copy (undeclared)" the operator creates ``<review_dir>/COPY/<child digest>`` by hand.

  engine_parents_check.py CHILD_DIR --parent DIGEST=DIR [--parent ...] [--declared LABEL_JSON_OR_FILE]
        [--same-owner DIGEST ...] [--ignore-tree DIR ...] [--json]

``--ignore-tree``: trees whose files are never counted (the reference image, a common base): shared boilerplate is not reuse.
``--same-owner``: parent digests held by the same coldkey (snapshot) or proven by a signed ownership proof.
Pass EVERY digest of the parent's hotkey as --parent (the validator's bar is the best score over all of them, and the same
files may sit in an older digest). A declared digest without a --parent tree cannot be verified: verdict "unverified
declaration", exit 3. A file entry of the label that ends in "/" is a directory prefix. Closed components (binary code without
source, weights excluded) are reported: withhold the award approval while any is present. Detected reuse that the label
does not declare is a copy (create COPY/<child digest>; COPY alone is enough, REVIEWED is not needed).
Exit code: 0 independent / derived / same owner, 2 copy (undeclared), 3 declared digest not verifiable, 1 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Mapping, Optional

from reax_subnet import engcopyline

MIN_SHARED_BYTES = 1  # an empty file (e.g. __init__.py) is no evidence of reuse


def tree_hashes(root: str) -> dict[str, str]:
    """``{"/relative/path": sha256}`` of every regular, non-empty file under ``root`` (symlinks are not followed)."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) < MIN_SHARED_BYTES:
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out["/" + os.path.relpath(path, root).replace(os.sep, "/")] = h.hexdigest()
    return out


def shared_files(child: Mapping[str, str], other: Mapping[str, str], ignore_hashes: set[str]) -> list[str]:
    """Child paths whose content is byte-identical with any file of ``other`` (any path: a rename is still reuse)."""
    theirs = set(other.values()) - ignore_hashes
    return sorted(p for p, h in child.items() if h in theirs)


def check(child: Mapping[str, str], parents: Mapping[str, Mapping[str, str]], declared_label: Optional[str],
          same_owner: set[str], ignore: Mapping[str, str]) -> dict:
    """The report (a dict, JSON-serialisable). Pure on the hash maps."""
    declared, problems = engcopyline.parse_parents(declared_label)
    ignore_hashes = set(ignore.values())
    shared = {d: shared_files(child, tree, ignore_hashes) for d, tree in parents.items()}
    declared_files = {p.digest: p.files for p in declared}
    unmatched = sorted(d for d in declared_files if d not in parents)  # declared, but no tree was given for it
    overall, per = engcopyline.verdict([p.digest for p in declared], shared, same_owner, declared_files)
    if unmatched and overall != engcopyline.VERDICT_COPY:
        overall = "unverified declaration"
    return {
        "verdict": overall,
        "parents": {d: {"verdict": per.get(d, engcopyline.VERDICT_INDEPENDENT), "shared_files": shared.get(d, []),
                        "shared_binaries": [f for f in shared.get(d, []) if engcopyline.is_binary_name(f)],
                        "declared": d in declared_files, "same_owner": d in same_owner,
                        "undeclared_files": [f for f in shared.get(d, []) if not engcopyline.file_declared(f, declared_files.get(d, ()))]}
                    for d in sorted(set(parents) | set(declared_files))},
        "must_beat": engcopyline.constraint_parents((p.digest for p in declared), (), same_owner),
        "closed_components": engcopyline.closed_components(child),  # withhold the award approval while any is present
        "unverified_declared": unmatched,
        "label_problems": problems,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("child", help="extracted file tree of the candidate image")
    ap.add_argument("--parent", action="append", default=[], metavar="DIGEST=DIR")
    ap.add_argument("--declared", default=None, help="the parents label value (JSON) or a file holding it")
    ap.add_argument("--same-owner", action="append", default=[], metavar="DIGEST")
    ap.add_argument("--ignore-tree", action="append", default=[], metavar="DIR")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    declared = a.declared
    if declared and os.path.isfile(declared):
        with open(declared, encoding="utf-8") as fh:
            declared = fh.read()
    parents = {}
    for item in a.parent:
        digest, sep, d = item.partition("=")
        if not sep or not os.path.isdir(d):
            print(f"bad --parent {item!r} (DIGEST=DIR, DIR must exist)", file=sys.stderr)
            return 1
        parents[digest] = tree_hashes(d)
    if not os.path.isdir(a.child):
        print("child directory does not exist", file=sys.stderr)
        return 1
    ignore: dict[str, str] = {}
    for d in a.ignore_tree:
        ignore.update(tree_hashes(d))
    child = tree_hashes(a.child)
    report = check(child, parents, declared, set(a.same_owner), ignore)
    if a.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print("verdict:", report["verdict"])
        for d, r in report["parents"].items():
            print(f"  parent {d[:12]}: {r['verdict']}; {len(r['shared_files'])} identical file(s), "
                  f"{len(r['shared_binaries'])} binary; declared={r['declared']} same_owner={r['same_owner']}")
            for f in r["undeclared_files"]:
                print("    undeclared:", f)
        if report["must_beat"]:
            print("must beat by the full margin:", ", ".join(x[:12] for x in report["must_beat"]))
        if report["closed_components"]:
            print("closed components (WITHHOLD the award approval until their source is published):",
                  ", ".join(c + (" (weights? (check))" if engcopyline.needs_weights_check(c) else "")
                            for c in report["closed_components"]))
        for d in report["unverified_declared"]:
            print(f"declared parent {d[:12]} has no --parent tree: cannot verify (pass every digest of that hotkey)")
        for p in report["label_problems"]:
            print("label problem:", p)
    if report["verdict"] == engcopyline.VERDICT_COPY:
        return 2
    return 3 if report["unverified_declared"] else 0


if __name__ == "__main__":
    sys.exit(main())
