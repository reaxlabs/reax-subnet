"""Comparison database: index.json + fingerprint files (LINEAGE-PROCEDURE.md section 7).

{"schema": "reax-lineage-db/1",
 "families": {"gemma-4-31b": {"root": "repo@rev"}},
 "models": {"repo@rev": {"repo", "revision", "family", "kind": official|list|public|entry|round1|reax|flagged,
                          "start": "repo@rev" | null, "public_at": ISO | null, "block": int | null, "extrinsic": int | null,
                          "uid": int | null, "holder_ids": [..], "flagged": bool, "result": "PASS"|..., "official": bool,
                          "fp": {"A": "fp/x.npz", "B": "fp/y.npz"},
                          # checker 1.6.0 (all optional):
                          "parents": ["repo@rev", ..], "parent_shares": {"repo@rev": share},   # an entry with several declared parents; `start` stays the largest-share one
                          "private": bool, "void": bool}}}

CONFIDENTIAL (RULES 2.7 / 3.2): the row and the fingerprint files of an entry that is still `private` are sampled weights of a private model. The database that holds them
is as confidential as the private weights themselves: it is never published or exported (`export_public` writes only what is public), it is not part of the dry run, and its
rows are deleted with the entry when it becomes void. `public_at` (the time we first saw the model / entry public) is written by the caller.
"""
import json
import os
from typing import Dict, List, Optional

from .fp import Fingerprint, layer_view


def mkey(repo, rev):
    return "%s@%s" % (repo, rev)


class ComparisonDB:
    def __init__(self, path: str):
        self.path = path
        ip = os.path.join(path, "index.json")
        self.index = json.load(open(ip)) if os.path.exists(ip) else {"schema": "reax-lineage-db/1", "models": {}, "families": {}}
        self.index.setdefault("models", {})
        self.index.setdefault("families", {})
        self._fp_cache: Dict[str, Fingerprint] = {}
        self.view: Optional[dict] = None         # see set_layer_view

    def set_layer_view(self, rho: Dict[int, int], n_layers: int, direct: Optional[Dict[int, int]], start_key: Optional[str]):
        """Entries with a declared layer_map (or a start that has one): every fingerprint this DB hands out is returned in the entry's
        layer numbering (fp.layer_view); the entry's start by its declared map, every other model by the root layer it samples."""
        self.view = {"rho": dict(rho), "n_layers": int(n_layers), "direct": dict(direct) if direct else None, "start": start_key}

    @property
    def models(self) -> Dict[str, dict]:
        return self.index["models"]

    @property
    def families(self) -> Dict[str, dict]:
        return self.index["families"]

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "index.json"), "w") as f:
            json.dump(self.index, f, indent=1, sort_keys=True)

    def add(self, meta: dict, fps: Dict[str, str]):
        meta = dict(meta)
        meta["fp"] = {**(self.models.get(mkey(meta["repo"], meta["revision"]), {}).get("fp", {})), **fps}
        self.models[mkey(meta["repo"], meta["revision"])] = meta

    def get(self, key: str) -> Optional[dict]:
        return self.models.get(key)

    def fingerprint(self, key: str, seed_id: str, keep=True) -> Optional[Fingerprint]:
        fp = self._raw_fingerprint(key, seed_id, keep)
        return self.seen(key, fp)

    def seen(self, key: Optional[str], fp: Optional[Fingerprint]) -> Optional[Fingerprint]:
        """`fp` of model `key` in the entry's layer numbering (unchanged when the entry declares no layer change)."""
        v = self.view
        if v is None or fp is None:
            return fp
        is_start = key is not None and key == v["start"]
        return layer_view(fp, v["rho"], v["n_layers"], (v["direct"] or {}) if is_start else None)

    def _raw_fingerprint(self, key: str, seed_id: str, keep=True) -> Optional[Fingerprint]:
        m = self.models.get(key)
        if not m or seed_id not in (m.get("fp") or {}):
            return None
        ck = key + "#" + seed_id
        if ck in self._fp_cache:
            return self._fp_cache[ck]
        p = m["fp"][seed_id]
        fp = Fingerprint.load(p if os.path.isabs(p) else os.path.join(self.path, p))
        if keep:
            if len(self._fp_cache) > 12:
                self._fp_cache.pop(next(iter(self._fp_cache)))
            self._fp_cache[ck] = fp
        return fp

    def parent_list(self, key: str):
        """The declared parents of a model as [(key, share)]: `parents` (+ `parent_shares`, equal shares when missing) of an entry with several parents, else its `start`
        (share 1), else []."""
        m = self.models.get(key) or {}
        ps = [p for p in (m.get("parents") or []) if p]
        if ps:
            sh = m.get("parent_shares") or {}
            return [(p, float(sh.get(p, 1.0 / len(ps)))) for p in ps]
        return [(m["start"], 1.0)] if m.get("start") else []

    def ancestors(self, key: str) -> List[str]:
        """Every ancestor of a model, nearest first (breadth first over all declared parents; a chain gives the start, its start, ...), each once."""
        out, seen = [], {key}
        frontier = [key]
        while frontier:
            nxt = []
            for k in frontier:
                for p, _ in self.parent_list(k):
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                        nxt.append(p)
            frontier = nxt
        return out

    @staticmethod
    def is_private(meta: Optional[dict]) -> bool:
        return bool(meta and meta.get("private"))

    def export_public(self, dest: str) -> dict:
        """Write a copy of this database that holds only public rows (rows not flagged `private`) and their fingerprint files into `dest` (a new or empty directory).
        A private entry's row and fingerprints are never written (confidential, see the module text); a public row's `start`, `parents` and `parent_shares` that
        point at a private row are dropped. -> {"rows": n, "skipped_private": n}"""
        import shutil
        if os.path.exists(dest) and os.listdir(dest):
            raise ValueError("export_public needs a new or empty directory")
        os.makedirs(os.path.join(dest, "fp"), exist_ok=True)
        keep = {k: m for k, m in self.models.items() if not m.get("private")}
        out = {}
        for k, m in keep.items():
            m = json.loads(json.dumps(m))
            for fld in ("start",):
                if m.get(fld) and m[fld] not in keep:
                    m[fld] = None
            if m.get("parents"):
                m["parents"] = [p for p in m["parents"] if p in keep]
                m["parent_shares"] = {p: v for p, v in (m.get("parent_shares") or {}).items() if p in keep}
            fps = {}
            for sid, p in (m.get("fp") or {}).items():
                src = p if os.path.isabs(p) else os.path.join(self.path, p)
                rel = os.path.join("fp", os.path.basename(src))
                shutil.copyfile(src, os.path.join(dest, rel))
                fps[sid] = rel
            m["fp"] = fps
            out[k] = m
        idx = {k: v for k, v in self.index.items() if k != "models"}
        idx["models"] = out
        with open(os.path.join(dest, "index.json"), "w") as f:
            json.dump(idx, f, indent=1, sort_keys=True)
        return {"rows": len(out), "skipped_private": len(self.models) - len(keep)}

    def root_of(self, family: str) -> Optional[str]:
        return (self.families.get(family) or {}).get("root")

    def candidates(self, entry_key: str, family: Optional[str], block: Optional[int], extrinsic: int,
                   commit_time: Optional[str], exclude: set, seed_id: str, include_private: bool = True) -> List[str]:
        """Models older than the commitment (earlier block/extrinsic for entries, earlier HF commit time for public models),
        same family, with a fingerprint of this sample, excluding the start and its ancestors. `include_private=False` (the dry run, RULES 3.2): only public models and
        revealed entries (rows flagged `private` are skipped)."""
        out = []
        for k, m in sorted(self.models.items()):
            if k == entry_key or k in exclude or seed_id not in (m.get("fp") or {}):
                continue
            if m.get("private") and not include_private:
                continue
            if family and m.get("family") and m["family"] != family:
                continue
            if m.get("block") is not None:
                if block is None or (m["block"], m.get("extrinsic") or 0) >= (block, extrinsic):
                    continue
            elif m.get("public_at"):
                if not commit_time or m["public_at"][:19] > commit_time[:19]:
                    continue
            out.append(k)
        return out
