"""The stage-0 check (LINEAGE-PROCEDURE.md steps 0.1-0.9) and its dry-run."""
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .db import ComparisonDB, mkey
from .evidence import (Problem, config_diff, declared_ids, merge_parent_problems, normalise_parents, parents_rule_problems, scan_evidence, template_differences,
                       text_only_export, tokenizer_diff, undeclared_ids, validate_record, walk_own_parents, TEXT_EXT, MAX_EVIDENCE_BYTES, canonical_json, SKIP_FILES)
from .fp import Fingerprint, SampleSpec, SEED_A_DEFAULT, effective_map, get_fingerprint, seed_sha256
from .hf import HFError
from .listing import Listing, check_data_and_licence, check_parents, check_teachers
from .measure import (Thresholds, aligned_keys, copy_stats, delta_vs, frame_check, grid_shares, head_source, joint_fit,
                      joint_summary, measure_candidate, own_delta, summarise, _per_tensor)
from .sources import Hub

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class CheckError(RuntimeError):
    pass


def script_sha256() -> str:
    """sha256 over every source file of the checker + schema + thresholds (published before the round opens)."""
    h = hashlib.sha256()
    files = sorted(f for f in os.listdir(PKG_DIR) if f.endswith(".py") or f == "thresholds.json")
    for f in files:
        h.update(f.encode())
        h.update(open(os.path.join(PKG_DIR, f), "rb").read())
    for name in ("reax-lineage.schema.json", "reax-lineage-1.schema.json"):          # v2 (published) and the legacy v1 schema (v1.6.0)
        sp = os.path.join(PKG_DIR, "..", "schema", name)
        if os.path.exists(sp):
            h.update(open(sp, "rb").read())
    return h.hexdigest()


def blas_gflops(n: int = 1024) -> float:
    """Single-precision matrix-product speed of this numpy (best of 3 after a warm-up). A distribution's reference BLAS measured 4.5 GFLOPS,
    the pip wheel (OpenBLAS) about 1,000."""
    a = np.random.default_rng(0).random((n, n), dtype=np.float32)
    a @ a
    best = 1e9
    for _ in range(3):
        t = time.time()
        a @ a
        best = min(best, time.time() - t)
    return 2 * n ** 3 / max(best, 1e-6) / 1e9


def utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_layer_map(lm) -> Optional[Dict[int, int]]:
    if lm is None:
        return None
    if isinstance(lm, list):
        return {i: int(v) for i, v in enumerate(lm)}
    return {int(k): int(v) for k, v in lm.items()}


# --------------------------------------------------------------- scanning ----

_SCAN = {}


def _candidate_parents(db, key, seed_id, extra):
    """The fingerprints of a candidate's own declared parents with their shares, [(fp, share)]: its `parents` (an entry with several parents, v1.6.0) or its `start` (share 1).
    A parent without a fingerprint is left out (the candidate's own change is then measured against the others, or against nothing: the full change from the root)."""
    out = []
    for pk, w in db.parent_list(key):
        f = db.fingerprint(pk, seed_id) if pk not in extra else extra[pk]
        if f is not None:
            out.append((f, w))
    return out


def _scan_one(key):
    S = _SCAN
    thr, db, seed_id = S["thr"], S["db"], S["seed_id"]
    try:
        P = db.fingerprint(key, seed_id) if key not in S["extra"] else S["extra"][key]
        if P is None:
            return key, None
        meta = db.get(key) or S["extra_meta"].get(key, {})
        PS = _candidate_parents(db, key, seed_id, S["extra"]) if db.get(key) else []
        keys = aligned_keys(S["C"], P, S["R"], *[f for f, _ in PS], keys=S["keys"])
        if not keys:
            return key, {"n_tensors": 0, "detected": False, "beta": 0.0, "cos": 0.0, "share": 0.0, "se": 0.0, "null": 0.0,
                         "splice_layers": 0, "beta_lb": 0.0, "layer_beta": [], "layer_cos": [], "layers_ge_05": 0, "beta_q10": 0.0,
                         "n_layers": 0, "copy": None, "head_source": False, "coverage": 0.0}
        op = own_delta(P, PS, S["R"], keys)
        oc = {k: S["oc"][k] for k in keys}
        res = measure_candidate(oc, op, keys, S["n_layers"], thr)
        res["coverage"] = round(len(keys) / max(1, len(S["C"].body)), 4)
        res["copy"] = copy_stats(S["C"], P, thr, keys=S["keys"])
        res["head_source"] = bool(head_source(S["C"], S["S"], P, thr))
        res["layer_beta"], res["layer_cos"] = res["layer_beta"][:0], res["layer_cos"][:0]     # keep the JSON small
        return key, res
    except Exception as e:          # one broken fingerprint must not kill the check
        return key, {"error": "%s: %s" % (type(e).__name__, e), "detected": False}


def scan_candidates(C, S, R, cand_keys, db, seed_id, thr, extra, extra_meta, jobs=1, oc=None, keys=None, head_refs=None):
    """`oc` / `keys` (v1.6.0, an entry with several declared parents): the entry's own change as the caller computed it (the residual after the declared parents) and the tensor
    keys it is defined on; `head_refs`: the declared parents' fingerprints (the head test: a head equal to a candidate's, which differs from every declared parent's)."""
    if oc is None:
        keys = aligned_keys(C, R, keys=sorted(C.body)) if S is None else aligned_keys(C, S, R)
        oc = own_delta(C, S if (S is not None and S.name != R.name) else None, R, keys)
    _SCAN.clear()
    _SCAN.update(thr=thr, db=db, seed_id=seed_id, C=C, S=(head_refs if head_refs is not None else S), R=R, keys=keys, oc=oc, n_layers=C.n_layers,
                 extra=extra, extra_meta=extra_meta)
    if jobs > 1 and len(cand_keys) > 4:
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs) as pool:
            out = pool.map(_scan_one, cand_keys, chunksize=2)
    else:
        out = [_scan_one(k) for k in cand_keys]
    return {k: v for k, v in out if v is not None}


def start_beta(C, S, R, db, seed_id, thr):
    """beta of the declared start's own change in the entry, measuring the entry from start(S)."""
    keys = aligned_keys(C, S, R)
    if not keys:
        return None
    pairs = []
    sk_key = mkey(S.meta["repo"], S.meta["rev"])
    if db and db.get(sk_key):
        for pk, w in db.parent_list(sk_key):
            f = db.fingerprint(pk, seed_id)
            if f is not None and f.name != R.name:
                pairs.append((f, w))
    dC = delta_vs(C, R, keys)
    dS = delta_vs(S, R, keys)
    for f, w in pairs:                       # the start's own parents (one: its start, share 1: exactly the v1.5.2 computation)
        dB = delta_vs(f, R, keys)
        dC = {k: dC[k] - w * dB[k] for k in keys}
        dS = {k: dS[k] - w * dB[k] for k in keys}
    s = summarise(_per_tensor(dC, dS, keys), thr.splice_min_tensors)
    return s


# ------------------------------------------------------------ the check ----

def found_ids(res) -> set:
    return set(res.get("repo_ids_found", []))


def is_transient(e) -> bool:
    """v1.5.1: a Hugging Face hiccup (an `HFError` of kind `transient`: 429, 5xx, timeout, connection error, after the client's own retries) must never end as a note,
    a missing measurement or a NEEDS_CORRECTION for the miner. The steps of the check that catch `Exception` to carry on (evidence scan, start / config reads, Tier B)
    re-raise it, so the CLI prints `error_kind: transient` (exit 2) and the caller retries the whole check later. A `repo_unreadable` HFError stays what it was."""
    return isinstance(e, HFError) and e.kind == "transient"


def hf_holder_id(repo):
    return "hf:" + repo.split("/")[0].lower()


def holder_set(repo, extra=None):
    """The OLD behaviour (v1.4.1 and before), kept only behind `auto_hf_holder` / --auto-hf-holder: the Hugging Face account in a repo id counted as a holder id
    of that repo's committer. RULES 2.9 says otherwise (see Check.own_holder_ids)."""
    return set([hf_holder_id(repo)] + list(extra or []))


class Check:
    def __init__(self, repo, rev, hotkey=None, block=None, extrinsic=0, commit_time=None, record=None, list_path=None,
                 db_path=None, cache_dir=None, hub: Optional[Hub] = None, thr: Optional[Thresholds] = None,
                 dry_run=False, seed_a=None, seed_b=None, jobs=None, tier_b=True, holder_ids=None, spec=None,
                 local_dir=None, auto_hf_holder=False):
        self.repo, self.rev, self.hotkey, self.block, self.extrinsic = repo, rev, hotkey, block, extrinsic or 0
        self.commit_time, self.record_path, self.list_path, self.db_path = commit_time, record, list_path, db_path
        self.cache_dir, self.hub, self.thr = cache_dir, hub or Hub(), thr or Thresholds.load()
        self.dry_run, self.jobs, self.tier_b, self.holder_ids = dry_run, jobs or min(8, os.cpu_count() or 1), tier_b, holder_ids
        self.seed_a = seed_a or os.environ.get("REAX_LINEAGE_SEED_A") or SEED_A_DEFAULT
        self.seed_b = None if dry_run else (seed_b or os.environ.get("REAX_LINEAGE_SEED_B"))
        self.spec = spec or SampleSpec()
        self.local_dir = local_dir
        self.auto_hf_holder = bool(auto_hf_holder)
        self._view = None            # (effective map, entry layers, declared map, start key) when the entry declares a layer change
        self.parents: List[dict] = []         # the declared parents (v1.6.0, normalised: evidence.normalise_parents); [] = from scratch
        self.parents_known = False            # True once the record was read (the result then carries `parents`)
        self.copy_of_declared_parent = False  # an entry with several parents that is a copy of one of them (RULES 3.5): REFUSED, even though its other parents are not found
        self.legacy = False                   # a reax-lineage/1 record (one start)
        self._jcache: Dict[tuple, object] = {}
        self.reasons: List[dict] = []
        self.notes: List[str] = []
        self.t0 = time.time()
        self.timing: Dict[str, float] = {}

    # -- holders (RULES 2.9, v1.5.0; last sentence v1.5.1) ---
    def record_names_hotkey(self, rec) -> bool:
        """True exactly when the committing hotkey is known and the record names it (`record.hotkey == hotkey`, the same exact comparison as `validate_record`).
        No committing hotkey (a manual run without --hotkey) means it cannot be established, so False."""
        return bool(self.hotkey) and isinstance(rec, dict) and rec.get("hotkey") == self.hotkey

    def own_holder_ids(self, rec=None) -> set:
        """Holder ids of the entry. What the caller passes (`holder_ids`, CLI `--holder`): the coldkey plus the Hugging Face accounts linked through entries that PASSED
        intake and whose lineage record names the committing hotkey, also through a chain of such links (RULES 2.9). A commitment of someone else's repo creates no
        link, so the checker must not conclude "this entry's holder owns hf:<owner of the repo>" from the repo id alone (v1.4.1 did: a copy of that account's earlier
        model was then a `duplicate` instead of `copy_undeclared`).
        v1.5.1, RULES 2.9 last sentence: for the entry's OWN check only, `hf:<account of the entry's repo id>` is added exactly when `rec` names the committing hotkey
        (`record_names_hotkey`), so a first entry counts its own account's earlier public models as own work. Never for other models' ids (`model_holder_ids`),
        never when the record does not name the committing hotkey (or there is no committing hotkey). `auto_hf_holder` (CLI --auto-hf-holder, default off) restores
        the v1.4.1 derivation, unconditionally, for manual use."""
        ids = set(self.holder_ids or [])
        if self.auto_hf_holder:
            ids |= holder_set(self.repo)
        pinned_ok = getattr(self, "pinned_names_hotkey", None)
        if "/" in self.repo and self.record_names_hotkey(rec) and pinned_ok is not False:
            ids.add(hf_holder_id(self.repo))
        return {i.lower() for i in ids}

    def model_holder_ids(self, meta) -> set:
        """Holder ids of a comparison-DB model: the `holder_ids` of its index record (written by whoever builds the DB: the intake service for entries, the baseline builder
        for public models), nothing derived from its repo id (see own_holder_ids). With `auto_hf_holder` the old derivation is added again."""
        ids = {i.lower() for i in (meta.get("holder_ids") or [])}
        if self.auto_hf_holder and meta.get("repo"):
            ids.add(hf_holder_id(meta["repo"]))
        return ids

    # -- parents (v1.6.0) ---
    @property
    def start_decl(self) -> dict:
        """The ONE declared parent of an entry with one parent (what v1.5.2 called the start), or {"kind": "none"} for an entry from scratch. Entries with several parents have no start."""
        return self.parents[0] if len(self.parents) == 1 else ({"kind": "none"} if not self.parents else {"kind": "multi"})

    @staticmethod
    def pkey(p) -> str:
        return mkey(p["repo"], p["revision"])

    def declared_chain(self, db) -> set:
        """Every declared parent (and, for an own parent, the parents the record names for it) and every ancestor of each: nothing in it is a weight overlap, everything in it is on the
        bar list (RULES 3.6: every parent and every ancestor of every parent, no cap)."""
        keys = [self.pkey(p) for p in self.parents] + [self.pkey(q) for q, _ in walk_own_parents(self.parents)]
        out = set(keys)
        for k in keys:
            out |= set(db.ancestors(k))
        return out

    def display(self, db, key) -> str:
        """How a model is named in a message: its key, except a still-private entry of this round, which is "a private entry of this round (uid N)" until it is revealed (RULES 2.7)."""
        m = db.get(key) if db else None
        if m and m.get("private"):
            return "a private entry of this round" + (" (uid %s)" % m["uid"] if m.get("uid") is not None else "")
        return key

    # -- helpers ---
    def reason(self, cls, code, message, field=""):
        """cls: N needs correction, R refused, H hold (misdeclaration candidate), I info."""
        self.reasons.append({"class": cls, "code": code, "message": message, "field": field})

    def tick(self, name):
        self.timing[name] = round(time.time() - self.t0, 2)

    def evidence_files(self, source) -> Dict[str, str]:
        out, total = {}, 0
        for p, fi in sorted(source.list_files().items()):
            base = os.path.basename(p)
            if fi.size > MAX_EVIDENCE_BYTES or base in SKIP_FILES or not p.lower().endswith(TEXT_EXT) or p.startswith(".git"):
                continue
            if base == "model.safetensors.index.json" or p.count("/") > 2:
                continue
            try:
                txt = source.read_file(p).decode("utf-8", "replace")
            except Exception as e:
                if is_transient(e):
                    raise
                self.notes.append("could not read %s: %s" % (p, type(e).__name__))
                continue
            out[p] = txt
            total += len(txt)
            if len(out) >= 80 or total > 8_000_000:
                self.notes.append("evidence scan truncated at %d files" % len(out))
                break
        return out

    def resolver(self):
        hub, cache = self.hub, {}
        hub_dir = hub.hub_dir

        def res(i, kind_hint):
            if i in cache:
                return cache[i]
            kind = None
            if hub_dir:
                def ci_dir(base, ident):
                    org, _, name = ident.partition("/")
                    try:
                        o = next((d for d in os.listdir(base) if d.lower() == org), None)
                        n = next((d for d in os.listdir(os.path.join(base, o)) if d.lower() == name), None) if o else None
                    except OSError:
                        return False
                    return bool(o and n)
                if ci_dir(hub_dir, i):
                    kind = "model"
                elif os.path.isdir(os.path.join(hub_dir, "datasets")) and ci_dir(os.path.join(hub_dir, "datasets"), i):
                    kind = "dataset"
            else:
                c = hub._client()
                order = ["datasets", "models"] if kind_hint == "dataset" else ["models", "datasets"]
                for k in order:
                    try:
                        c.api("%s/%s" % (k, i))
                        kind = "model" if k == "models" else "dataset"
                        break
                    except HFError as e:
                        if is_transient(e):            # not "this id does not exist": the answer is unknown
                            raise
                        continue
            cache[i] = kind
            return kind
        return res

    # -- main ---
    def run(self) -> dict:
        thr = self.thr
        gf = blas_gflops()
        if gf < 50:
            self.notes.append("numpy runs on a slow BLAS (%.1f GFLOPS): the weight scan is fine, but Tier B (matrix products of 5,000-dim layers) would take hours; "
                              "use a numpy wheel with OpenBLAS (pip install numpy), not the distribution's reference BLAS" % gf)
        res = {"checker_version": __version__, "check_script_sha256": script_sha256(), "dry_run": self.dry_run, "blas_gflops": round(gf, 1),
               "entry": {"repo": self.repo, "revision": None}, "hotkey": self.hotkey, "commitment_block": self.block,
               "extrinsic_index": self.extrinsic}
        if self.local_dir:
            from .hf import LocalSource
            src = LocalSource(self.local_dir)
            src.repo, src.rev = self.repo, self.rev
        else:
            self.rev = self.hub.resolve_rev(self.repo, self.rev)
            src = self.hub.source(self.repo, self.rev)
        res["entry"]["revision"] = self.rev
        db = ComparisonDB(self.db_path) if self.db_path else ComparisonDB("/nonexistent")
        listing = Listing.load(self.list_path) if self.list_path else Listing({"models": []})
        res["list_version"] = listing.version
        if not self.commit_time and self.block is not None:
            a = db.index.get("anchor")
            if a:
                t = dt.datetime.strptime(a["time"][:19], "%Y-%m-%dT%H:%M:%S") + dt.timedelta(seconds=12 * (self.block - a["block"]))
                self.commit_time = t.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.notes.append("commit time estimated from the chain anchor (12 s blocks)")
        res["commit_time"] = self.commit_time

        # 0.2 record
        files = src.list_files()
        raw = None
        if self.record_path:
            raw = open(self.record_path, "rb").read()
        elif "reax-lineage.json" in files:
            raw = src.read_file("reax-lineage.json")
        # RULES 2.9 last sentence (v1.5.2): the own-repo holder link needs the file AT THE PINNED REVISION to name the committing hotkey;
        # a signed correction (--record) alone cannot create it (otherwise a stranger committing someone else's repo could correct the
        # record to name themselves and turn a copy of the owner's earlier model into an own duplicate).
        self.pinned_names_hotkey = None
        if self.record_path:
            try:
                pinned = json.loads(src.read_file("reax-lineage.json")) if "reax-lineage.json" in files else None
            except Exception:
                pinned = None
            self.pinned_names_hotkey = self.record_names_hotkey(pinned)
        if raw is None:
            self.reason("N", "record_missing", "reax-lineage.json is not in the repo at this revision")
            return self.finish(res, None, {})
        res["record_sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            rec = json.loads(raw)
        except Exception as e:
            self.reason("N", "schema", "reax-lineage.json is not valid JSON: %s" % e)
            return self.finish(res, None, {})
        probs = validate_record(rec, self.hotkey, self.repo, self.rev)
        for p in probs:
            self.reason("N", p.code, p.message, p.field)
        if probs:
            self.tick("record")
            return self.finish(res, rec if isinstance(rec, dict) else None, {})
        self.tick("record")
        parents, scratch, legacy = normalise_parents(rec)
        self.parents, self.legacy, self.parents_known = parents, legacy, True
        res["record_schema"] = rec["schema"]
        # RULES 2.2 / 3.3 (d8b M6): more than 4 parents, or a parent with a share below 0.10, is REFUSED (no penalty). No correction can help (re-commit): nothing else is measured.
        pr = parents_rule_problems(parents, legacy)
        if pr:
            for p in pr:
                self.reason("R", p.code, p.message, p.field)
            return self.finish(res, rec, {})

        # 0.3 allowed-ness
        own_ids = self.own_holder_ids(rec)
        parent_probs, parent_metas = check_parents(parents, listing, db.index, self.commit_time, own_ids, legacy=legacy)
        start_meta = parent_metas[0] if len(parents) == 1 else None
        listing_probs = parent_probs + check_teachers(rec, listing, db.index, self.commit_time, own_ids) + check_data_and_licence(rec)
        for p in listing_probs:
            self.reason("N" if p.code == "parent_not_own" else "R", p.code, p.message, p.field)

        # 0.4 repo evidence
        try:
            texts = self.evidence_files(src)
            found = scan_evidence(texts, self.resolver(), skip_ids={self.repo.lower()})
        except Exception as e:
            if is_transient(e):
                raise
            texts, found = {}, {}
            self.notes.append("repo evidence scan failed: %s" % e)
        for u in undeclared_ids(found, rec):
            self.reason("N", "undeclared_id", "%s %s is named in %s but not in the record" % (u["kind"], u["id"], ", ".join(u["files"][:3])), "mentioned_not_used")
        res["repo_ids_found"] = sorted(found)
        self.tick("evidence")

        # 0.5 config + tokenizer vs the parents
        P_srcs = []
        for p in parents:
            S_ = None
            try:
                S_ = self.hub.source(p["repo"], p["revision"])
            except Exception as e:
                if is_transient(e):
                    raise
                self.notes.append("parent %s@%s not readable: %s" % (p["repo"], p["revision"][:8], e))
            P_srcs.append(S_)
        S_src = P_srcs[0] if len(parents) == 1 else None
        arch = rec.get("architecture_changes") or {}
        cfg_e = self._json(src, "config.json")
        if len(parents) <= 1:
            cfg_s = self._json(S_src, "config.json") if S_src else None
            if cfg_e is None:
                self.reason("N", "config", "config.json is missing or not valid JSON", "config.json")
            elif cfg_s is not None:
                for p in config_diff(cfg_e, cfg_s, arch):
                    self.reason("N", p.code, p.message, p.field)
                te = text_only_export(cfg_e, cfg_s)
                if te:
                    res["config_text_only_export"] = te
                    self.notes.append("config: text-only export of a multimodal start (%s -> %s): allowed without a declaration (rule 2.4)" % (
                        te["start_architecture"], te["entry_architecture"]))
            if S_src is not None:
                tprobs, tnote = self.tokenizer_vs(src, S_src, arch)
                if tnote:
                    self.notes.append(tnote)
                for p in tprobs:
                    self.reason("N", p.code, p.message, p.field)
        else:
            labels = ["%s@%s" % (p["repo"], p["revision"][:8]) for p in parents]
            cfgs = {labels[i]: self._json(P_srcs[i], "config.json") for i in range(len(parents)) if P_srcs[i] is not None}
            cfgs = {k: v for k, v in cfgs.items() if v is not None}
            diffs = template_differences(cfgs)
            for d in diffs:
                self.reason("R", "parents_template_mismatch", "the parents %s and %s are not served by the same engine template (RULES 2.2: all parents must be): %s" % (
                    d["against"], d["parent"], "; ".join("%s %r vs %r" % (k, v[0], v[1]) for k, v in sorted(d["fields"].items()))), "parents")
            if cfg_e is None:
                self.reason("N", "config", "config.json is missing or not valid JSON", "config.json")
            else:
                per = {lab: config_diff(cfg_e, c, arch) for lab, c in cfgs.items()}
                for p in merge_parent_problems(per):
                    self.reason("N", p.code, p.message, p.field)
                for lab, c in cfgs.items():
                    te = text_only_export(cfg_e, c)
                    if te:
                        res["config_text_only_export"] = te
                        self.notes.append("config: text-only export of a multimodal parent %s (%s -> %s): allowed without a declaration (rule 2.4)" % (
                            lab, te["start_architecture"], te["entry_architecture"]))
                        break
            tper, tnotes = {}, []
            for i, S_ in enumerate(P_srcs):
                if S_ is None:
                    continue
                tprobs, tnote = self.tokenizer_vs(src, S_, arch, label=labels[i])
                tper[labels[i]] = tprobs
                if tnote:
                    tnotes.append(tnote)
            self.notes.extend(dict.fromkeys(tnotes))
            for p in merge_parent_problems(tper):
                self.reason("N", p.code, p.message, p.field)
        self.tick("config")
        if self.has("R", ("parents_template_mismatch",)):
            return self.finish(res, rec, {})             # parents that no single engine template serves: refused, nothing to measure

        # 0.6-0.7 weights
        if len(parents) >= 2:
            weights = self.weights_multi(res, rec, src, P_srcs, parent_metas, db, listing, found, cfg_e)
        else:
            weights = self.weights(res, rec, src, S_src, start_meta, db, listing, found, cfg_e)
        return self.finish(res, rec, weights)

    def tokenizer_vs(self, src, S_src, arch, label=None):
        """Tokenizer of the entry against one parent -> (problems, note or None): equal file hash = fine; otherwise every token of the parent keeps its id, added tokens are
        declared, normaliser and pre-tokeniser are unchanged (evidence.tokenizer_diff)."""
        same = False
        try:
            fe, fs = src.list_files().get("tokenizer.json"), S_src.list_files().get("tokenizer.json")
            same = bool(fe and fs and fe.sha256 and fe.sha256 == fs.sha256)
        except Exception:
            pass
        te = self._json(src, "tokenizer.json") if not same else {}
        ts = self._json(S_src, "tokenizer.json") if not same else {}
        if not same and (te is None or ts is None):
            if "tokenizer.model" in src.list_files() or "vocab.json" in src.list_files():
                return [], "tokenizer compared by file hash only (no tokenizer.json)"
            return [Problem("tokenizer", "tokenizer.json is missing in the entry or the %s" % ("parent " + label if label else "start"), "tokenizer")], None
        return tokenizer_diff(te, ts, arch, same_sha=same), None

    def _json(self, src, name):
        ck = (id(src), name)
        if ck in self._jcache:
            return self._jcache[ck]
        try:
            self._jcache[ck] = v = json.loads(src.read_file(name))
            return v
        except Exception as e:
            if is_transient(e):             # v1.5.1: a Hugging Face hiccup is not "config.json is missing" (that would be a NEEDS_CORRECTION for the miner)
                raise
            return None

    # -- weights ---
    def weights(self, res, rec, src, S_src, start_meta, db, listing, found, cfg_e):
        thr = self.thr
        out = {"samples": {}, "sources": {}, "start": {}}
        st = self.start_decl
        lm = normalise_layer_map((rec.get("architecture_changes") or {}).get("layer_map"))
        skey = mkey(st["repo"], st["revision"]) if st["kind"] != "none" else None
        family = (start_meta or {}).get("family") or (db.get(skey) or {}).get("family") if skey else None
        if not family and cfg_e:
            family = listing.detect_family(cfg_e)
        root_key = db.root_of(family) or listing.family_root(family) if family else None
        if not root_key:
            self.reason("N", "family_unknown", "cannot determine the model family / root for the weight check (manual)")
            self.notes.append("no family root: weight check skipped")
            return out
        res_family = family
        out["family"] = family
        out["root"] = root_key
        samples = [("A", self.seed_a)] + ([("B", self.seed_b)] if self.seed_b else [])
        if not self.dry_run and not self.seed_b:
            self.notes.append("sealed Sample B seed (REAX_LINEAGE_SEED_B) not set: only Sample A was checked")
        own_ids = self.own_holder_ids(rec)
        # extra models named in the repo and not in the DB
        for sid, seed in samples:
            S_fp = R_fp = None
            db.view = None                                   # raw fingerprints first (the view of sample A must not leak into B)
            R_raw = self._model_fp(db, root_key, sid, seed)
            S_raw = self._model_fp(db, skey, sid, seed, S_src) if skey else None
            # a declared layer_map (physically duplicated / dropped own layers, RULES 2.5), also inherited from a start that is an entry
            # with one: the entry's rows sit at the positions of the root layer each layer descends from, and every comparison model is
            # read in the entry's numbering (fp.layer_view), so a duplicated block does not shift the later layers against the start
            rho = effective_map(lm, S_raw)
            C = get_fingerprint(src, seed, sid, self.cache_dir, self.spec, rho, publish_seed=(sid == "A"))
            C.meta["rev"] = src.rev
            if rho:
                self._view = (rho, C.n_layers, lm, skey)
                db.set_layer_view(*self._view)
                out["layer_map"] = {"declared": {str(k): v for k, v in (lm or {}).items()} or None,
                                    "effective": {str(k): v for k, v in rho.items()}, "entry_layers": C.n_layers}
                R_fp, S_fp = db.seen(root_key, R_raw), db.seen(skey, S_raw)
                if sid == "A":
                    self.notes.append("layer_map in effect: the comparison models are read in the entry's layer numbering (%d entry layers)" % C.n_layers)
            else:
                R_fp, S_fp = R_raw, S_raw
            samp = {"n_tensors": C.meta["n_tensors"], "values": C.meta["values"], "seconds": C.meta["seconds"],
                    "seed_sha256": seed_sha256(seed), "warnings": C.meta["warnings"]}
            if sid == "A":
                samp["seed"] = seed
            if C.meta["non_float_body"]:
                self.reason("R", "quantised_dtype", "weights are stored in a non-float dtype (e.g. %s %s): only bf16/fp16/fp32 safetensors are allowed" % C.meta["non_float_body"][0])
            if C.meta["warnings"]:
                self.notes.extend(C.meta["warnings"])
            # frame check vs start
            fr = None
            if S_fp is not None:
                fr = frame_check(C, S_fp, thr)
                samp["frame"] = fr
                sb = start_beta(C, S_fp, R_fp, db, sid, thr) if S_fp.name != R_fp.name else None
                if sb is not None:
                    samp["start_beta"] = {"beta": round(sb["beta"], 4), "cos": round(sb["cos"], 4), "share": round(sb["share"], 4)}
                    fr["start_beta_ok"] = bool(sb["beta"] >= thr.start_beta)
                else:
                    fr["start_beta_ok"] = True
                fr["confirmed"] = bool(fr["ok"] and fr["start_beta_ok"])
            # candidates
            exclude = set()
            if skey:
                exclude = self.declared_chain(db)
            exclude.add(root_key)
            ekey = mkey(self.repo, src.rev)
            cands = db.candidates(ekey, family, self.block, self.extrinsic, self.commit_time, exclude, sid, include_private=not self.dry_run)
            extra, extra_meta = self._named_models(found, db, sid, seed, family, exclude, cands, C)
            cands = cands + [k for k in extra if k not in cands]
            out.setdefault("_extra_fp", {})[sid] = (extra, extra_meta)
            use_S = S_fp if (fr is None or fr["confirmed"] or not self.tier_b) else S_fp
            t = time.time()
            scan = scan_candidates(C, S_fp, R_fp, cands, db, sid, thr, extra, extra_meta, self.jobs)
            samp["scan_seconds"] = round(time.time() - t, 1)
            samp["candidates_scanned"] = len(scan)
            # copy vs the start itself, grid
            if S_fp is not None:
                cs = copy_stats(C, S_fp, thr)
                samp["copy_vs_start"] = cs
            gs = grid_shares(C)
            gsS = grid_shares(S_fp) if S_fp is not None else (grid_shares(R_fp) if R_fp is not None else {"max": 0.0})
            samp["grid"] = {"entry": {k: round(v, 4) for k, v in gs.items()}, "start_max": round(gsS["max"], 4)}
            out["samples"][sid] = samp
            out.setdefault("_scan", {})[sid] = scan
            out.setdefault("_fp", {})[sid] = (C, S_fp, R_fp)
        self.tick("weights")
        return out

    # -- several declared parents (v1.6.0) ---
    def fit_parents(self, C, R_fp, P_fp):
        """Joint non-negative fit of the entry's change from the family root on the declared parents' changes from the root (RULES 3.2: "with the published procedure we measure which
        models' weights are in yours, and how much"; the joint_fit / joint_summary machinery of v1.4.0 on the parents only). Per layer: delta_entry ~ sum_i w_i delta_parent_i.
        Per parent: the median over layers of its weight (`weight`), the standard error across layers (`se`), `lower_bound` = weight - 2 se. A parent that IS the family root has no
        change from the root: its weight is what the others leave, 1 - sum of their weights, per layer. Also returns the entry's own change after removing the parents at their
        fitted (median) weights: `oc`, the quantity every undeclared candidate is measured against (for ONE parent at weight 1 this is delta_entry - delta_parent, exactly the start's
        own change of v1.5.2). Raises ValueError / ArithmeticError when nothing can be fitted (the caller turns that into a not-confirmed parent with the reason)."""
        keys = aligned_keys(C, R_fp, *P_fp)
        if not keys:
            raise ValueError("no tensor of the entry has the same shape and sampled rows as the family root and every parent")
        dC = delta_vs(C, R_fp, keys)
        ops, idx = {}, {}
        for i, f in enumerate(P_fp):
            if f.name != R_fp.name:
                ops["p%d" % i] = delta_vs(f, R_fp, keys)
                idx[i] = "p%d" % i
        if not ops:
            raise ValueError("every declared parent is the family root: nothing to fit")
        fit = joint_fit(dC, ops, keys)
        W = fit["layer_weights"]
        if W.shape[0] == 0:
            raise ValueError("the fit has no layers")
        col = {n: j for j, n in enumerate(fit["names"])}
        other = W.sum(1)
        per, wts = [], {}
        for i in range(len(P_fp)):
            lw = W[:, col[idx[i]]] if i in idx else 1.0 - other
            med = float(np.median(lw))
            se = float(lw.std(ddof=1) / np.sqrt(len(lw))) if len(lw) > 1 else 0.0
            if not all(np.isfinite(v) for v in (med, se)):
                raise ArithmeticError("non-finite weight in the parent fit")
            per.append({"weight": med, "se": se, "lower_bound": med - 2 * se, "is_root": i not in idx})
            if i in idx:
                wts[idx[i]] = med
        oc = {k: dC[k] - sum(wts[n] * ops[n][k] for n in wts) for k in keys}
        return {"keys": keys, "per": per, "oc": oc, "layers": int(W.shape[0])}

    @staticmethod
    def mixture_fp(R_fp, P_fp, fitr):
        """The sampled weights of the mixture of the declared parents as the fit found their proportions: root + sum(u_i * (parent_i - root)) on the fitted tensors, u_i = the fitted weight of
        parent i divided by the sum of the fitted weights of all parents (a parent that is the root counts with what the others leave, at least 0), so the weights add up to 1. A fit of an
        entry with a large change of its own can be biased in its SIZE (the fitted weights then add up to less than 1: 0.56 for the round-1 entry v5h, CALIBRATION.md 20.3) while the
        proportions stay informative; the reference of the frame check should not shrink towards the root because of that. The entry is compared with this mixture by the frame check: an honest merge is the mixture plus the entry's own change, so rows keep their direction and scale."""
        from .fp import Fingerprint
        w = [max(d["weight"], 0.0) for d in fitr["per"]]
        tot = sum(w) or 1.0
        ws = [(f, wi / tot) for f, wi, d in zip(P_fp, w, fitr["per"]) if not d["is_root"]]
        body = {}
        for k in fitr["keys"]:
            r = R_fp.body[k]
            m = r.astype(np.float32, copy=True)
            for f, u in ws:
                m += np.float32(u) * (f.body[k] - r)
            body[k] = m
        meta = dict(R_fp.meta)
        meta.update(repo="parents-mixture", rev="fit", row_starts={k: R_fp.meta["row_starts"][k] for k in fitr["keys"]})
        return Fingerprint(meta, body, {})

    def weights_multi(self, res, rec, src, P_srcs, parent_metas, db, listing, found, cfg_e):
        """The weight step of an entry with 2-4 declared parents. Per sample: the frame of the entry against every parent, the joint fit on the parents (confirmation: every parent's
        lower bound > 0 on every sample, see `classify_parents`), the entry's own change after the parents, the candidates measured against it (as for one parent)."""
        thr = self.thr
        parents = self.parents
        out = {"samples": {}, "sources": {}, "start": {}, "multi": True}
        lm = normalise_layer_map((rec.get("architecture_changes") or {}).get("layer_map"))
        pkeys = [self.pkey(p) for p in parents]
        order = sorted(range(len(parents)), key=lambda i: (-float(parents[i]["share"]), i))
        fams = [((parent_metas[i] or db.get(pkeys[i]) or {}).get("family")) for i in order]
        family = next((f for f in fams if f), None)
        if len({f for f in fams if f}) > 1:
            self.notes.append("the parents are of different families in the comparison set (%s): the family of the parent with the largest share (%s) is used" % (
                ", ".join(sorted({f for f in fams if f})), family))
        if not family and cfg_e:
            family = listing.detect_family(cfg_e)
        root_key = db.root_of(family) or listing.family_root(family) if family else None
        if not root_key:
            self.reason("N", "family_unknown", "cannot determine the model family / root for the weight check (manual)")
            self.notes.append("no family root: weight check skipped")
            return out
        out["family"], out["root"] = family, root_key
        samples = [("A", self.seed_a)] + ([("B", self.seed_b)] if self.seed_b else [])
        if not self.dry_run and not self.seed_b:
            self.notes.append("sealed Sample B seed (REAX_LINEAGE_SEED_B) not set: only Sample A was checked")
        for sid, seed in samples:
            db.view = None
            R_raw = self._model_fp(db, root_key, sid, seed)
            P_raw = [self._model_fp(db, pkeys[i], sid, seed, P_srcs[i]) for i in range(len(parents))]
            rho = effective_map(lm, P_raw[order[0]])
            C = get_fingerprint(src, seed, sid, self.cache_dir, self.spec, rho, publish_seed=(sid == "A"))
            C.meta["rev"] = src.rev
            if rho:
                # a declared layer change: every comparison model is read in the entry's layer numbering, the layout of the parent with the largest share is inherited
                self._view = (rho, C.n_layers, None, None)
                db.set_layer_view(*self._view)
                out["layer_map"] = {"declared": {str(k): v for k, v in (lm or {}).items()} or None,
                                    "effective": {str(k): v for k, v in rho.items()}, "entry_layers": C.n_layers}
                R_fp = db.seen(root_key, R_raw)
                P_fp = [db.seen(pkeys[i], P_raw[i]) for i in range(len(parents))]
                if sid == "A":
                    self.notes.append("layer_map in effect: the comparison models are read in the entry's layer numbering (%d entry layers)" % C.n_layers)
            else:
                R_fp, P_fp = R_raw, P_raw
            samp = {"n_tensors": C.meta["n_tensors"], "values": C.meta["values"], "seconds": C.meta["seconds"],
                    "seed_sha256": seed_sha256(seed), "warnings": C.meta["warnings"]}
            if sid == "A":
                samp["seed"] = seed
            if C.meta["non_float_body"]:
                self.reason("R", "quantised_dtype", "weights are stored in a non-float dtype (e.g. %s %s): only bf16/fp16/fp32 safetensors are allowed" % C.meta["non_float_body"][0])
            if C.meta["warnings"]:
                self.notes.extend(C.meta["warnings"])
            # `frame` (information per parent): the row-cosine share of the entry against each parent alone. A merge is NOT in any single parent's frame (the row scale of a 50/50 mix
            # differs from each parent's when the parents moved far from the root), so the condition is the entry-level frame against the fitted mixture of the parents (below).
            plist = [{"index": i, "key": pkeys[i], "declared_share": float(parents[i]["share"]),
                      "frame": {k: v for k, v in frame_check(C, f, thr).items() if k in ("frame_share", "scale_share", "min_cos", "coverage")}} for i, f in enumerate(P_fp)]
            fitr = None
            samp["frame"] = None
            try:
                fitr = self.fit_parents(C, R_fp, P_fp)
                for i, d in enumerate(fitr["per"]):
                    plist[i].update(weight=round(d["weight"], 4), se=round(d["se"], 4), lower_bound=round(d["lower_bound"], 4), is_root=d["is_root"])
                samp["parents_fit"] = {"layers": fitr["layers"], "tensors": len(fitr["keys"])}
                samp["frame"] = frame_check(C, self.mixture_fp(R_fp, P_fp, fitr), thr)             # the entry against the fitted mixture of its parents
            except Exception as e:           # never lose the check: the parents are then not confirmed on this sample (said in the reason)
                samp["parents_fit"] = {"error": "%s: %s" % (type(e).__name__, e)}
            samp["parents"] = plist
            exclude = self.declared_chain(db) | {root_key}
            ekey = mkey(self.repo, src.rev)
            cands = db.candidates(ekey, family, self.block, self.extrinsic, self.commit_time, exclude, sid, include_private=not self.dry_run)
            extra, extra_meta = self._named_models(found, db, sid, seed, family, exclude, cands, C)
            cands = cands + [k for k in extra if k not in cands]
            out.setdefault("_extra_fp", {})[sid] = (extra, extra_meta)
            t = time.time()
            if fitr is not None:
                scan = scan_candidates(C, None, R_fp, cands, db, sid, thr, extra, extra_meta, self.jobs, oc=fitr["oc"], keys=fitr["keys"], head_refs=list(P_fp))
            else:
                scan = {}
                self.notes.append("sample %s: no candidate was measured, the parents could not be fitted (%s)" % (sid, samp["parents_fit"]["error"]))
            samp["scan_seconds"] = round(time.time() - t, 1)
            samp["candidates_scanned"] = len(scan)
            samp["copy_vs_parents"] = [copy_stats(C, f, thr) for f in P_fp]
            gs = grid_shares(C)
            gsS = max((grid_shares(f)["max"] for f in P_fp), default=0.0)
            samp["grid"] = {"entry": {k: round(v, 4) for k, v in gs.items()}, "start_max": round(gsS, 4)}
            out["samples"][sid] = samp
            out.setdefault("_scan", {})[sid] = scan
            out.setdefault("_fp", {})[sid] = (C, None, R_fp)
            out.setdefault("_multi", {})[sid] = {"P_fp": P_fp, "fit": fitr}
        self.tick("weights")
        return out

    def _model_fp(self, db, key, sid, seed, src=None):
        fp = db.fingerprint(key, sid) if key else None
        if fp is not None:
            return fp
        repo, _, rev = key.partition("@")
        s = src or self.hub.source(repo, rev)
        fp = get_fingerprint(s, seed, sid, self.cache_dir, self.spec, None, publish_seed=(sid == "A"))
        return fp

    def _named_models(self, found, db, sid, seed, family, exclude, cands, C):
        """Models named in the repo that are not in the comparison DB: fingerprint them on demand (best effort)."""
        extra, meta = {}, {}
        declared_repos = {p["repo"].lower() for p in self.parents} | {q["repo"].lower() for q, _ in walk_own_parents(self.parents)}
        for i, d in found.items():
            if d["kind"] != "model" or any(k.lower().startswith(i + "@") for k in db.models) or i in declared_repos:     # (a declared parent is never its own overlap)
                continue
            try:
                rev = self.hub.resolve_rev(i, "main")
                s = self.hub.source(i, rev)
                fp = get_fingerprint(s, seed, sid, self.cache_dir, self.spec, None, publish_seed=(sid == "A"))
                fp = db.seen(mkey(i, rev), fp)
                if aligned_keys(C, fp):
                    k = mkey(i, rev)
                    extra[k] = fp
                    meta[k] = {"repo": i, "revision": rev, "kind": "named-in-repo", "start": None}
            except Exception as e:
                self.notes.append("named model %s not fingerprinted: %s" % (i, type(e).__name__))
            if len(extra) >= 12:
                break
        return extra, meta

    # -- classification ---
    def finish(self, res, rec, weights) -> dict:
        thr = self.thr
        res["parents"] = ([{"index": i, "kind": p["kind"], "repo": p["repo"], "revision": p["revision"], "uid": p.get("uid"), "key": self.pkey(p), "declared_share": float(p["share"])}
                           for i, p in enumerate(self.parents)] if self.parents_known else None)
        res["parent_keys"] = [self.pkey(p) for p in self.parents] if self.parents_known else None
        res["from_scratch"] = (not self.parents) if self.parents_known else None
        res.update({"result": None, "start_confirmed": None, "sources_detected": [], "weight_overlaps": [], "joint_share": None, "bar_list_order": None, "top5_nearest": [], "bar_list": [],
                    "copy_share": None, "grid_share": None, "tier_b_used": False, "transform_found": False,
                    "sample_A_seed": self.seed_a, "sample_B_seed_sha256": seed_sha256(self.seed_b) if self.seed_b else None})
        if weights.get("samples"):
            self.classify_weights(res, rec, weights)
        self.decide(res, rec)
        res["reasons"] = self.reasons
        res["notes"] = self.notes
        res["decided_at"] = utcnow()
        res["timing_seconds"] = self.timing
        res["total_seconds"] = round(time.time() - self.t0, 1)
        if res["result"] == "NEEDS_CORRECTION":
            res["correction_deadline"] = (dt.datetime.utcnow() + dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            res["correction_deadline"] = None
        for k in ("_scan", "_fp", "_extra_fp"):
            weights.pop(k, None)
        res["weights"] = weights
        return res

    def classify_weights(self, res, rec, weights):
        thr = self.thr
        db = ComparisonDB(self.db_path) if self.db_path else ComparisonDB("/nonexistent")
        if self._view:
            db.set_layer_view(*self._view)
        scan_by_sample = weights.get("_scan", {})
        fps = weights.get("_fp", {})
        st = self.start_decl
        multi = st["kind"] == "multi"
        skey = mkey(st["repo"], st["revision"]) if st["kind"] not in ("none", "multi") else None
        dkeys = {self.pkey(p) for p in self.parents}
        self.check_own_parents(weights, db)
        if multi:
            self.classify_parents(res, rec, weights, db, fps)
        else:
            # start confirmation: both samples must hold (the sealed sample catches edits of the published rows)
            confirmations = {sid: s.get("frame") for sid, s in weights["samples"].items() if s.get("frame")}
            confirmed = bool(confirmations) and all(f["confirmed"] for f in confirmations.values())
            likely = None
            if skey and confirmations and not confirmed:
                likely = self.find_likely_start(weights, db, rec)
            tier_b = None
            if skey and confirmations and not confirmed and likely is None and self.tier_b:
                tier_b = self.run_tier_b(res, rec, fps)
                if tier_b and tier_b.get("start_confirmed"):
                    confirmed = True
            res["start_confirmed"] = None if not skey else {
                "confirmed": confirmed, "beta": {sid: (s.get("start_beta") or {}).get("beta") for sid, s in weights["samples"].items()},
                "frame_share": {sid: f["frame_share"] for sid, f in confirmations.items()},
                "scale_share": {sid: f["scale_share"] for sid, f in confirmations.items()},
                "likely_start": likely, "tier_b": ({k: v for k, v in tier_b.items() if k != "layers"} if tier_b else None)}
            if skey and not confirmed:
                if likely:
                    self.reason("N", "start_not_confirmed", "the weights do not match the declared start; they match %s (%s)" % (likely["model"], likely["how"]))
                elif tier_b is None and self.tier_b is False:
                    self.reason("N", "start_not_confirmed", "weights do not match the declared start (Tier B disabled): manual review")
                elif tier_b is None:
                    self.reason("N", "tier_b_unavailable", "tier B not available -> NEEDS_CORRECTION, manual: the weights do not match the declared start and the reparametrisation check could not run")
                else:
                    self.reason("N", "start_not_confirmed", "the weights do not match the declared start, also not after undoing rotation, permutation and scaling (Tier B)")
            if tier_b:
                res["tier_b_used"] = True
                res["transform_found"] = bool(tier_b.get("transform_found"))
                if tier_b.get("transform_found"):
                    # v1.3.0: NEEDS_CORRECTION (as v1.1.0), not a MISDECLARATION_CANDIDATE. Tier B only aligns the entry to the DECLARED START,
                    # so it can say "this is the declared start under a disguise", never "this is another holder's model under a disguise": rule 3.8(b) is
                    # NOT implemented for other holders' models (CALIBRATION.md section 16.3). No penalty follows from it.
                    # v1.5.1 (RULES 3.3 / CRITIQUE-4 S4): the miner is told what the check found and how to contest it (a dispute pauses the 24 hours; Florian decides at the
                    # Thursday slot), instead of "manual review". The machine field (code `transform_found`, class N, result NEEDS_CORRECTION) is unchanged.
                    self.reason("N", "transform_found", "our check finds your weights are a rotated, permuted or rescaled form of your start; "
                                "if you think the check is wrong, dispute it (the dispute pauses the 24 hours). This result carries no penalty.")
            if not skey:        # start none: the weights must not be those of a listed / official model of the family
                for sid in ("A",):
                    C, S_, R_ = fps[sid]
                    pool = [weights.get("root")] + [k for k, m in db.models.items() if m.get("family") == weights.get("family") and (m.get("kind") in ("official", "list") or m.get("official"))]
                    for k in dict.fromkeys(pool):
                        fp = db.fingerprint(k, sid) if k else None
                        if fp is not None and frame_check(C, fp, thr)["ok"]:
                            self.reason("N", "start_not_declared", "the record says start none, but the weights are those of %s (plus own change): declare it as the start" % k)
                            res["start_confirmed"] = {"confirmed": False, "likely_start": {"model": k, "how": "start none but frame matches"}}
                            break
        # merge samples per candidate
        merged: Dict[str, dict] = {}
        for sid, scan in scan_by_sample.items():
            for k, r in scan.items():
                if "error" in r:
                    self.notes.append("candidate %s (%s): %s" % (k, sid, r["error"]))
                    continue
                m = merged.setdefault(k, {"model": k})
                m[sid] = r
        own_ids = self.own_holder_ids(rec)
        chain = self.declared_chain(db)
        detections, copies, heads = [], [], []
        for k, m in merged.items():
            meta = db.get(k) or {"repo": k.split("@")[0], "kind": "named-in-repo"}
            samples_hit = [sid for sid in ("A", "B") if sid in m and m[sid].get("detected")]
            best = max((m[sid] for sid in ("A", "B") if sid in m), key=lambda r: r.get("beta", 0.0))
            m["detected_on"] = samples_hit
            hs = [sid for sid in ("A", "B") if sid in m and m[sid].get("head_source")]
            if samples_hit:
                owner = self.model_holder_ids(meta)
                own = bool(owner & own_ids)
                named = meta["repo"].lower() in found_ids(res) or meta["repo"].lower() in declared_ids(rec)
                allowed = self.is_allowed(meta, db)
                detections.append({"model": k, "beta": round(best["beta"], 4), "cos": round(best["cos"], 4),
                                   "share": round(best["share"], 4), "null": round(best["null"], 4), "se": round(best["se"], 4),
                                   "beta_lb": round(best["beta_lb"], 4), "splice_layers": best["splice_layers"],
                                   "detected_on": samples_hit, "sample_disagreement": len(samples_hit) != len([s for s in m if s in ("A", "B")]),
                                   "own_holder": own, "named": named, "allowed": allowed, "flagged": bool(meta.get("flagged")), "kind": meta.get("kind"),
                                   "copy": bool(any(m[sid]["copy"] and m[sid]["copy"]["is_copy"] for sid in ("A", "B") if sid in m)),
                                   # RULES 3.6 (v1.2.0): every detected model of the same base family is a weight overlap unless it is a copy, an own-holder
                                   # model or in the start chain. `main_test` / `same_start_entry` / `relation` only describe it (information, no consequence).
                                   "main_test": bool(any(m[sid].get("main") for sid in samples_hit)),
                                   "same_start_entry": self.same_start_entry(meta, dkeys),
                                   "relation": self.relation(meta, dkeys),
                                   "family": meta.get("family") or weights.get("family"),
                                   "uid": meta.get("uid")})
                if meta.get("private"):
                    detections[-1].update(private=True, public_name=self.display(db, k))
            if hs:
                heads.append(k)
        # copy of the start itself or of anything in the DB
        cp = []
        pk_list = [self.pkey(p) for p in self.parents]
        for sid, s in weights["samples"].items():
            c = s.get("copy_vs_start")
            if c and c["is_copy"]:
                cp.append((skey, sid, c))
            for i, c in enumerate(s.get("copy_vs_parents") or []):         # "A merge that is a copy of one of its parents is a copy" (RULES 3.5)
                if c and c["is_copy"]:
                    cp.append((pk_list[i], sid, c))
        res["copy_share"] = {sid: {"vs_start": (s.get("copy_vs_start") or {}).get("body")} for sid, s in weights["samples"].items()}
        for sid, s in weights["samples"].items():
            if s.get("copy_vs_parents"):
                res["copy_share"][sid]["vs_parents"] = [c["body"] for c in s["copy_vs_parents"]]
        for d in detections:
            if d["copy"]:
                cp.append((d["model"], None, None))
        # quantisation round trip
        grid = {sid: s["grid"] for sid, s in weights["samples"].items()}
        res["grid_share"] = {sid: g["entry"]["max"] for sid, g in grid.items()}
        quant = any(g["entry"]["max"] >= thr.grid_share and g["start_max"] < thr.grid_share for g in grid.values())
        if quant:
            what = "at least %.0f%% of 128-value groups lie on a quantisation grid (entry %s, start %s)" % (
                100 * thr.grid_share, {s: g["entry"]["max"] for s, g in grid.items()}, {s: g["start_max"] for s, g in grid.items()})
            if self.declares_qat(rec):
                # v1.5.1 (RULES 2.4, CRITIQUE-4 M4): a declared quantisation-aware training is the exception to the round-trip refusal; the grid share stays in the
                # result (`grid_share`, shown on the uid page) and is repeated as information. An undeclared round trip is still REFUSED. `quantised_dtype` (non-float
                # storage) is a different reason and is not affected.
                self.reason("I", "quantisation_aware_declared", what + ". You declared quantisation-aware training (architecture_changes.quantisation_aware_training), "
                            "so this is not a refusal; the grid share is shown for information. No correction, no penalty.", "architecture_changes.quantisation_aware_training")
            else:
                self.reason("R", "quantisation_round_trip", what)
        # copies
        copy_models = sorted({c[0] for c in cp if c[0]})
        for cm in copy_models:
            meta = db.get(cm) or {"repo": cm.split("@")[0]}
            owner = self.model_holder_ids(meta)
            named = meta["repo"].lower() in found_ids(res) or meta["repo"].lower() in declared_ids(rec) or cm == skey
            nm = self.display(db, cm)
            if multi and cm in dkeys:
                self.copy_of_declared_parent = True
            if owner & own_ids:
                self.reason("R", "duplicate", "copy of your own earlier model %s (a duplicate, not a new entry; no penalty)" % nm)
            elif named or cm == skey or cm in dkeys:
                self.reason("R", "copy", "at least %.1f%% of the sampled values equal %s within bf16 rounding (a declared copy: refused, no penalty)" % (100 * thr.copy_share, nm))
            else:
                self.reason("H", "copy_undeclared", "copy of %s (%s), which neither the record nor the repo files name: rule 3.8" % (nm, meta.get("kind")))
        # head source
        for k in heads:
            if k not in chain:
                self.reason("N", "head_source", "the output head / embeddings equal those of %s (not %s): heads taken from another model are not allowed in an entry (rule 2.3)" % (
                    self.display(db, k), "the start's" if len(self.parents) <= 1 else "a declared parent's"))
        # sources (RULES 3.3 / 3.6 / 3.8, final text of 13 Nov). A detected model of the SAME BASE FAMILY (every candidate is, see
        # ComparisonDB.candidates) that is not a copy (3.5), not of your own holder and not in your start chain is a WEIGHT OVERLAP: it joins
        # the bar list (if among the largest, see below) and nothing else follows (no correction, no refusal, no misdeclaration). Entry or public
        # model, same start or another, whole-model or splice hit: the check cannot tell honest same-start, same-data siblings from a merge (C4: 96 of
        # 96 flagged), so a statistical share is never a finding. Only copies (above) have a penalty path (3.8(a)); an exact head copy (below,
        # `head_source`, rule 2.3) and a disguise of the declared start undone by Tier B (`transform_found`) are NEEDS_CORRECTION. The joint share is
        # shown on the UID page as information; the weights in the joint fit also order the bar list (below).
        # v1.4.0 (RULES 3.6 final): the joint non-negative fit of the entry's own change is computed BEFORE the overlaps are ranked; the at most
        # `bar_list_max_overlaps` (3) overlaps with the largest weight in it join the bar list (ties: larger cos, then earlier commitment). If the fit
        # fails for this entry the order falls back to beta (noted). All overlaps stay listed on the UID page as information, in the order that decides.
        overlaps = [d for d in detections if self.is_weight_overlap(d, chain)]
        fit, why = self.joint_fit_all(weights, db, detections, overlaps, chain) if overlaps else (None, None)
        by = "joint_weight" if fit else "beta"
        wmap = fit["per"][fit["best"]]["by_model"] if fit else {}
        for d in overlaps:
            d["joint_weight"] = wmap.get(d["model"])                      # None: not in the fit (fit failed, or beyond the 16 strongest by beta): counts as 0.0 in the order
            d["in_joint_fit"] = d["joint_weight"] is not None
        overlaps = self.rank_overlaps(overlaps, db, by)
        for i, d in enumerate(overlaps):
            d["weight_overlap"] = True
            d["bar_list_rank"] = i + 1
            # rule 3.6: only a positive weight takes a place (a zero weight explains nothing of the entry's change); in the beta fallback a positive beta
            d["on_bar_list"] = i < thr.bar_list_max_overlaps and ((d.get("joint_weight") or 0.0) > 0.0 if by == "joint_weight" else d.get("beta", 0.0) > 0.0)
        res["weight_overlaps"] = overlaps
        if overlaps:
            res["bar_list_order"] = {"key": by, "sample": fit["best"] if fit else None, "fallback": fit is None, "why": why,
                                     "ties": "larger cos, then earlier commitment"}
            if fit is None:
                self.notes.append("joint fit of the entry's own change failed (%s): the bar list is ordered by beta (rule 3.6 fallback)" % why)
                self.reason("I", "joint_fit_failed", "The joint fit of your own change could not be computed for this entry (%s), so your weight overlaps are ordered by beta "
                            "instead of by their weight in the joint fit (rule 3.6, fallback). No correction, no penalty." % why, "bar_list")
        own_other = [d for d in detections if d["own_holder"] and d["model"] not in chain]
        for d in own_other:
            self.notes.append("detected own-holder model %s (beta %.3f): own work" % (self.display(db, d["model"]), d["beta"]))
        for d in overlaps:
            if d["flagged"]:
                self.notes.append("weight overlap with %s, which is flagged on the Hugging Face Decision Index: %s; the weights alone have no consequence (a human may look)" % (
                    self.display(db, d["model"]), "it joins the bar list" if d["on_bar_list"] else "it is listed for information only (not among the largest overlaps)"))
        res["sources_detected"] = detections
        res["joint_share"] = self.joint_share_info(fit, overlaps)
        # top 5 nearest
        near = []
        for k, m in merged.items():
            r = max((m[s] for s in ("A", "B") if s in m), key=lambda r: r.get("beta", 0.0))
            near.append({"model": k, "beta": round(r["beta"], 4), "cos": round(r["cos"], 4), "share": round(r["share"], 4),
                         "null": round(r["null"], 4)})
            if db.is_private(db.get(k)):
                near[-1].update(private=True, public_name=self.display(db, k))
        res["top5_nearest"] = sorted(near, key=lambda x: -x["beta"])[:5]
        crit = ("the %d overlaps with the largest weight in the joint fit of your own change" if fit else
                "the %d overlaps with the largest beta, because the joint fit of your change failed for this entry") % thr.bar_list_max_overlaps
        for d in overlaps:
            what = "weights of %s%s are measured in yours (beta %.2f, cos %.2f, share of layers %.2f%s): %s. " % (
                self.display(db, d["model"]), (" (uid %s)" % d["uid"]) if (d.get("uid") is not None and not d.get("private")) else "", d["beta"], d["cos"], d["share"],
                (", weight in the joint fit %.2f" % d["joint_weight"]) if d["in_joint_fit"] else (", not in the joint fit" if fit else ""), self.RELATION_TEXT[d["relation"]])
            if d["on_bar_list"]:
                self.reason("I", "weight_overlap", what + "It joins your bar list (rule 3.6: %s; this is number %d); no correction, no penalty. "
                            "The check cannot tell this from honest training of the same start on the same data." % (crit, d["bar_list_rank"]), "bar_list")
            else:
                self.reason("I", "weight_overlap_not_listed", what + "Shown for information only: your bar list takes %s (rule 3.6), and this is number %d of %d. "
                            "No correction, no penalty." % (crit, d["bar_list_rank"], len(overlaps)), "bar_list")
        # bar list: the start chain, teachers, and the overlaps that made the cut (in rank order)
        res["bar_list"] = self.bar_list(rec, db, own_ids, [d for d in overlaps if d["on_bar_list"]], own_other)

    def check_own_parents(self, weights, db):
        """RULES 2.2 / 3.1: your own public model may be a parent "as long as its weights come only from allowed parents"; the record names ITS parents so that this can be checked. Allowed-ness
        of those parents is checked with the other parents (listing.check_parents); here the weights: when the own model is not an entry of this round that the check already measured (no
        row in the comparison DB), its change from the family root is fitted on the changes of the parents the record names for it, and each of them must be found (lower bound > 0 on every
        sample), else NEEDS_CORRECTION `own_parent_not_confirmed` (field `parents/<i>/parents/<j>`). Best effort: if the models cannot be fitted the reason is a note, never a correction."""
        root_key = weights.get("root")
        todo = [(i, p) for i, p in enumerate(self.parents) if p["kind"] == "own" and p.get("parents") and not db.get(self.pkey(p))]
        if not root_key or not todo:
            return
        samples = [("A", self.seed_a)] + ([("B", self.seed_b)] if self.seed_b else [])
        saved, db.view = db.view, None
        try:
            for i, p in todo:
                nested = p["parents"]
                per = {}
                try:
                    for sid, seed in samples:
                        R = self._model_fp(db, root_key, sid, seed)
                        O = self._model_fp(db, self.pkey(p), sid, seed)
                        Ns = [self._model_fp(db, self.pkey(q), sid, seed) for q in nested]
                        per[sid] = self.fit_parents(O, R, Ns)["per"]
                except Exception as e:
                    if is_transient(e):
                        raise
                    self.notes.append("own parent %s: its declared parents could not be checked against its weights (%s: %s)" % (self.pkey(p), type(e).__name__, e))
                    continue
                for j, q in enumerate(nested):
                    lbs = {sid: per[sid][j]["lower_bound"] for sid in per}
                    if not all(v > 0.0 for v in lbs.values()):
                        w = {sid: round(per[sid][j]["weight"], 4) for sid in per}
                        self.reason("N", "own_parent_not_confirmed", "the weights of your own model %s do not contain its declared parent %s: weight in a joint fit %s, lower bound %s. "
                                    "Its weights must come only from allowed parents (RULES 2.2): correct the record (the parents of your own model) or commit a new revision." % (
                                        p["repo"], self.display(db, self.pkey(q)), w, {sid: round(v, 4) for sid, v in lbs.items()}), "parents/%d/parents/%d" % (i, j))
        finally:
            db.view = saved

    def classify_parents(self, res, rec, weights, db, fps):
        """RULES 3.2 / 3.3 (d8): "every declared parent must be found". A parent is CONFIRMED when its weight in the joint non-negative fit of the entry's change on the parents' changes
        (fit_parents) has a lower bound above 0 on every sample that was measured (median over layers minus two standard errors across layers), AND the entry as a whole is in the frame
        of the fitted mixture of its parents on every sample (frame check against the mixture; a merge is not in any one parent's frame). A parent that is not found -> NEEDS_CORRECTION
        `parents_not_confirmed` (one reason per parent, field `parents/<i>`, the measured weight in the text; no penalty). A frame that fails is tried with Tier B against the parent with the
        largest weight (a rotated / permuted / rescaled merge is `transform_found`, as for a start; else `parents_not_confirmed`, field `parents`). The per-parent numbers are in `res["parents"]`."""
        parents = self.parents
        samples = weights["samples"]
        errs = {sid: s["parents_fit"]["error"] for sid, s in samples.items() if "error" in (s.get("parents_fit") or {})}
        frames = {sid: s.get("frame") for sid, s in samples.items()}
        frame_ok = not errs and all(f and f["ok"] for f in frames.values())
        report = []
        for i, p in enumerate(parents):
            key = self.pkey(p)
            per = {sid: (s.get("parents") or [None] * len(parents))[i] for sid, s in samples.items()}
            fit = {sid: d for sid, d in per.items() if d and d.get("lower_bound") is not None}
            fit_ok = bool(per) and len(fit) == len(per) and all(d["lower_bound"] > 0.0 for d in fit.values())
            best = max(fit, key=lambda sid: fit[sid]["lower_bound"]) if fit else None
            row = {"index": i, "key": key, "kind": p["kind"], "repo": p["repo"], "revision": p["revision"], "uid": p.get("uid"), "declared_share": float(p["share"]),
                   "fit_ok": fit_ok, "frame_share": {sid: d["frame"].get("frame_share") for sid, d in per.items() if d},
                   "weight": {sid: d.get("weight") for sid, d in per.items() if d}, "lower_bound": {sid: d.get("lower_bound") for sid, d in per.items() if d},
                   "se": {sid: d.get("se") for sid, d in per.items() if d}, "measured_weight": fit[best]["weight"] if best else None,
                   "measured_lower_bound": fit[best]["lower_bound"] if best else None, "weight_sample": best, "is_root": bool(per.get("A") and per["A"].get("is_root"))}
            if db.is_private(db.get(key)):
                row.update(private=True, public_name=self.display(db, key))
            report.append(row)
        tier_b, transform = None, False
        if not frame_ok and not errs and self.tier_b:
            target = max(range(len(parents)), key=lambda i: (report[i]["measured_weight"] if report[i]["measured_weight"] is not None else -1.0, -i))
            tier_b = self.run_tier_b(res, rec, fps, parent=parents[target])
            if tier_b and tier_b.get("start_confirmed"):
                frame_ok = True
                transform = bool(tier_b.get("transform_found"))
        for r in report:
            r["confirmed"] = bool(r["fit_ok"] and frame_ok and not transform)
            r["tier_b"] = ({k: v for k, v in tier_b.items() if k != "layers"} if tier_b else None)
        if transform:
            self.reason("N", "transform_found", "our check finds that your weights are a rotated, permuted or rescaled form of your parents; if you think the check is wrong, dispute it "
                        "(the dispute pauses the 24 hours). This result carries no penalty.", "parents")
        elif not frame_ok and not errs:
            fs = {sid: f["frame_share"] for sid, f in frames.items() if f}
            why = ("also not after undoing rotation, permutation and scaling (Tier B)" if tier_b else
                   ("Tier B is not available: manual review" if not self.tier_b else "the reparametrisation check (Tier B) could not run"))
            self.reason("N", "parents_not_confirmed", "the measurement does not confirm your declared parents: your weights are not in the frame of the mixture of your parents (frame share %s), %s" % (fs, why), "parents")
        for r in report:
            if r["confirmed"] or transform or (not frame_ok and not errs):
                continue
            name = self.display(db, r["key"])
            if errs:
                why = "the joint fit of your weights on your parents failed (%s)" % "; ".join("sample %s: %s" % kv for kv in sorted(errs.items()))
            else:
                why = ("its weight in the joint fit of your weights on your parents is %s (lower bound %s; sample %s): it is not found" %
                       (r["measured_weight"], r["measured_lower_bound"], r["weight_sample"])) if r["weight_sample"] else "it could not be measured"
                fit = {sid: {"weight": r["weight"].get(sid), "lower_bound": r["lower_bound"].get(sid)} for sid in r["weight"] if r["lower_bound"].get(sid) is not None}
                worst = min(fit, key=lambda sid: fit[sid]["lower_bound"]) if fit else None
                if worst and worst != r["weight_sample"]:
                    why += "; on sample %s: weight %s, lower bound %s" % (worst, fit[worst]["weight"], fit[worst]["lower_bound"])
            self.reason("N", "parents_not_confirmed", "the measurement does not confirm your declared parent %s (declared share %.2f): %s. Every declared parent must be found in your "
                        "weights (RULES 3.2); correct the record (a parent that is not in your weights, or a share that is wrong) or commit a new revision." % (name, r["declared_share"], why),
                        "parents/%d" % r["index"])
        res["parents"] = report
        res["start_confirmed"] = {"confirmed": all(r["confirmed"] for r in report), "multi_parent": True,
                                  "parents": [{k: r[k] for k in ("key", "declared_share", "measured_weight", "confirmed")} for r in report],
                                  "frame_vs_mixture": {sid: (f or {}).get("frame_share") for sid, f in frames.items()}}
        res["tier_b_used"] = bool(tier_b)
        res["transform_found"] = bool(transform)
        res["parents_fit"] = {sid: {"layers": (s.get("parents_fit") or {}).get("layers"), "tensors": (s.get("parents_fit") or {}).get("tensors"),
                                    "error": (s.get("parents_fit") or {}).get("error")} for sid, s in samples.items()}

    JOINT_MAX = 16          # the joint fit has at most this many regressors (the strongest by beta) plus the overlaps shown

    def joint_fit_all(self, weights, db, detections, overlaps, chain):
        """The joint non-negative fit of the entry's own change on every detected non-start source at once (so a composite and its parents, or
        same-data siblings, do not each claim the same part). Computed BEFORE the overlaps are ranked: the weight of each overlap in it (median over
        layers) is the key that orders the bar list (RULES 3.6 final), and the whole fit is the information on the UID page (`joint_share`).
        A plain sum of betas over overlapping candidates read 1.0-14.0 on the C1 recipe entries for stated totals of 0.75-2.35 (CALIBRATION.md
        section 6). The 16 strongest overlaps by beta are the ones whose weight is read (`act`); the other detections stay regressors.
        Every sample is fitted on its own; a sample that fails (an exception, no overlap alignable, a non-finite weight, no layers) is noted and skipped.
        Returns (fit, None) with fit = {"act", "per": {sample: summary}, "best": the sample with the highest lower bound, the one published and used
        for the order (like the detection: a source seen on either sample counts; the sealed sample exists to catch edits of the public rows)},
        or (None, why) if no sample gave a fit: then the order falls back to beta."""
        act = [d["model"] for d in sorted(overlaps, key=lambda d: (-d["beta"], -d["cos"], d["model"]))][:self.JOINT_MAX]
        if not act:
            return None, "no weight overlap"
        regress = sorted((d for d in detections if d["model"] not in chain and not d["copy"]), key=lambda d: (-d["beta"], -d["cos"], d["model"]))
        names = [d["model"] for d in regress][:self.JOINT_MAX]
        for m in act:                       # the overlaps shown are always in the fit
            if m not in names:
                names.append(m)
        per, failed = {}, []
        for sid, (C, S_fp, R_fp) in sorted(weights.get("_fp", {}).items()):
            try:
                extra, extra_meta = weights.get("_extra_fp", {}).get(sid, ({}, {}))
                mu = weights.get("_multi", {}).get(sid)
                if mu is not None:              # an entry with several declared parents: its own change is what is left after the parents (fit_parents)
                    if mu["fit"] is None:
                        failed.append("sample %s: the declared parents could not be fitted" % sid)
                        continue
                    keys, oc = list(mu["fit"]["keys"]), mu["fit"]["oc"]
                else:
                    keys = aligned_keys(C, R_fp) if S_fp is None else aligned_keys(C, S_fp, R_fp)
                    oc = own_delta(C, S_fp if (S_fp is not None and S_fp.name != R_fp.name) else None, R_fp, keys)
                ops = {}
                for k in names:
                    P = extra[k] if k in extra else db.fingerprint(k, sid, keep=False)
                    if P is None:
                        continue
                    PS = _candidate_parents(db, k, sid, extra) if db.get(k) else []
                    kk = aligned_keys(C, P, R_fp, *[f for f, _ in PS], keys=keys)
                    if not kk:
                        continue
                    ops[k] = own_delta(P, PS, R_fp, kk)
                    keys = kk
                if not any(m in ops for m in act):
                    failed.append("sample %s: no overlap could be aligned with the entry's tensors" % sid)
                    continue
                fit = joint_fit({k: oc[k] for k in keys}, {n: {kk: v[kk] for kk in keys} for n, v in ops.items()}, keys)
                summ = joint_summary(fit, act)
                if not summ["by_model"]:
                    raise ValueError("the fit has no layers or no overlap among its regressors")
                bad = [k for k in ("share", "se", "lower_bound") if not np.isfinite(summ[k])] + [m for m, w in summ["by_model"].items() if not np.isfinite(w)]
                if bad:
                    raise ArithmeticError("non-finite value in the joint fit (%s)" % ", ".join(bad[:3]))
                summ["fitted_models"] = len(ops)
                per[sid] = summ
            except Exception as e:          # never lose the check because the fit failed: the order falls back to beta (rule 3.6)
                failed.append("sample %s: %s: %s" % (sid, type(e).__name__, e))
        for f in failed:
            self.notes.append("joint fit %s" % f)
        if not per:
            return None, "; ".join(failed) or "no fingerprint of the entry"
        return {"act": act, "per": per, "best": max(sorted(per), key=lambda sid: per[sid]["lower_bound"])}, None

    def joint_share_info(self, fit, overlaps) -> Optional[dict]:
        """The joint fit as it is shown on the UID page: the share of the weight overlaps together (information, no verdict: RULES 3.8 final text, a
        statistical share alone is no misdeclaration) and the weight of each overlap (`by_model`), which is the key of the bar-list order (3.6).
        None if there was nothing to fit or the fit failed."""
        if not fit:
            return None
        per = fit["per"]
        best = per[fit["best"]]
        return {"rule": "no verdict (no correction, refusal or penalty follows from it): joint non-negative fit of the entry's own change on the detected sources; "
                        "share = joint weight of the weight overlaps; by_model = the weight of each overlap, the key that orders the bar list (rule 3.6)",
                "share": round(best["share"], 4), "se": round(best["se"], 4), "lower_bound": round(best["lower_bound"], 4),
                "sum_beta_overlaps": round(sum(max(0.0, d["beta"]) for d in overlaps), 4), "old_limit_for_reference": self.thr.deliberate_share,
                "by_model": best["by_model"], "order_sample": fit["best"],
                "samples": {sid: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "by_model"} for sid, r in per.items()},
                "sources": fit["act"]}

    def is_allowed(self, meta, db):
        if meta.get("flagged"):
            return False
        if meta.get("kind") in ("official", "list"):
            return True
        if meta.get("kind") == "entry":
            return meta.get("result") == "PASS"
        return False

    def find_likely_start(self, weights, db, rec):
        """Frame check against every official release / list model of the family and the 5 highest-beta candidates."""
        sid = "A"
        C, S, R = weights["_fp"][sid]
        scan = weights["_scan"][sid]
        pool = [k for k, m in db.models.items() if m.get("family") == weights.get("family") and
                (m.get("kind") in ("official", "list") or m.get("official"))]
        top = sorted(scan, key=lambda k: -scan[k].get("beta", 0))[:5]
        for k in list(dict.fromkeys(pool + top)):
            fp = db.fingerprint(k, sid)
            if fp is None or (S is not None and fp.name == S.name):
                continue
            f = frame_check(C, fp, self.thr)
            if f["ok"]:
                return {"model": k, "how": "official release / list model" if k in pool else "top-beta candidate", "frame_share": f["frame_share"]}
        return None

    def run_tier_b(self, res, rec, fps, parent=None):
        """Tier B against ONE declared parent (the start of an entry with one parent; for several parents the caller asks per parent that failed its frame check)."""
        from .tierb import canonicalise, fetch_tierb
        try:
            sid = "B" if "B" in fps else "A"
            seed = self.seed_b if sid == "B" else self.seed_a
            st = parent or self.start_decl
            S_src = self.hub.source(st["repo"], st["revision"])
            if self.local_dir:
                from .hf import LocalSource
                E_src = LocalSource(self.local_dir)
            else:
                E_src = self.hub.source(self.repo, self.rev)
            t = time.time()
            E = fetch_tierb(E_src, seed)
            S = fetch_tierb(S_src, seed, layers=E.meta["layers"])
            cfg = self._json(S_src, "config.json") or {}
            from .evidence import text_config
            hd = text_config(cfg).get("head_dim")
            rep = canonicalise(E, S, self.thr, head_dim=hd, consume=True)        # E.layers is released as the transformed copy is built (memory: CALIBRATION.md 18.4)
            rep["seconds"] = round(time.time() - t, 1)
            rep["layers_used"] = E.meta["layers"]
            rep.pop("layers", None)
            return rep
        except Exception as e:
            if is_transient(e):             # v1.5.1: not `tier_b_unavailable` (NEEDS_CORRECTION): the whole check is retried later
                raise
            self.notes.append("tier B failed: %s: %s" % (type(e).__name__, e))
            return None

    @staticmethod
    def declares_qat(rec) -> bool:
        """The record declares quantisation-aware training: `architecture_changes.quantisation_aware_training` is `true` (the schema's field; a boolean)."""
        ac = rec.get("architecture_changes") if isinstance(rec, dict) else None
        return isinstance(ac, dict) and ac.get("quantisation_aware_training") is True

    @staticmethod
    def same_start_entry(meta, skey) -> bool:
        """The candidate is an entry of this round whose declared start is exactly the entry's start (repo@revision). Entries from scratch
        (start none) have no common start and are never 'same start'. v1.6.0: `skey` may be a set of keys (the entry's declared parents) and the candidate may have several parents:
        true when they share a declared parent."""
        mine = {skey.lower()} if isinstance(skey, str) else {k.lower() for k in (skey or ())}
        theirs = {k.lower() for k in (meta.get("parents") or [meta.get("start")]) if k}
        return bool(mine and meta.get("kind") == "entry" and (mine & theirs))

    RELATION_TEXT = {"same_start_entry": "an earlier entry of your base family with your start", "other_start_entry": "an earlier entry of your base family with another start",
                     "public": "an earlier public model of your base family", "other": "an earlier model of your base family"}

    @staticmethod
    def relation(meta, skey) -> str:
        if meta.get("kind") == "entry":
            return "same_start_entry" if Check.same_start_entry(meta, skey) else "other_start_entry"
        return "public" if meta.get("kind") in ("public", "round1", "reax", "official", "list", "flagged") else "other"

    @staticmethod
    def is_weight_overlap(d, chain) -> bool:
        """RULES 3.6 (final): a detected model of the same base family that is not a copy (3.5), not of the entry's own holder and not in the
        start chain. Entry or public model, any start, whole-model or splice hit. (Candidates are filtered by family in ComparisonDB.candidates;
        a model named in the repo is only measured if its tensors align with the entry's.)"""
        return bool(not d["copy"] and not d["own_holder"] and d["model"] not in chain)

    @staticmethod
    def commitment_key(meta, anchor=None):
        """Order of commitment, earlier first. An entry: block, then extrinsic (RULES 3.4). A public model counts from its Hugging Face time
        (`public_at`). With a chain anchor (block -> time at 12 s per block, as the check does for the entry's own commit time) an entry and a
        public model compare by time; without one a public model counts as earlier than any entry. A model with neither counts as the latest."""
        last = "9999-12-31T23:59:59"
        if meta.get("block") is not None:
            t = None
            if anchor:
                t = (dt.datetime.strptime(anchor["time"][:19], "%Y-%m-%dT%H:%M:%S")
                     + dt.timedelta(seconds=12 * (meta["block"] - anchor["block"]))).strftime("%Y-%m-%dT%H:%M:%S")
            return (t or last, meta["block"], meta.get("extrinsic") or 0)
        return ((meta.get("public_at") or last)[:19], -1, 0)

    @staticmethod
    def rank_overlaps(overlaps, db, by="joint_weight"):
        """RULES 3.6 (final): the order that decides which weight overlaps join the bar list: the largest weight in the joint non-negative fit of the
        entry's own change first (`joint_weight`; an overlap that is not in the fit counts as 0.0); if the fit failed for the entry, the largest `beta`
        (`by="beta"`). Ties: the larger cos, then the earlier commitment. The values compared are the ones published on the UID page (rounded to
        4 places), so a reader can redo the order; the model key is the last resort (a total order)."""
        anchor = db.index.get("anchor")
        return sorted(overlaps, key=lambda d: (-(d.get(by) or 0.0), -d["cos"], Check.commitment_key(db.get(d["model"]) or {}, anchor), d["model"]))

    def bar_list(self, rec, db, own_ids, overlaps=(), own_detected=()):
        """`overlaps`: the weight overlaps that join the list (already ranked and cut to `bar_list_max_overlaps` by `classify_weights`).
        RULES 3.6 (d8): every declared parent and every ancestor of every parent, back to the list: no cap. One parent: role `start` (as before); several: role `parent`. The parents
        the record names for an own parent are on the list too (role `ancestor`, `declared_by`)."""
        out = []
        single = len(self.parents) == 1
        for i, p in enumerate(self.parents):
            k = self.pkey(p)
            if all(b["model"] != k for b in out):
                item = {"model": k, "role": "start" if single else "parent", "declared": True, "kind": p["kind"]}
                if not single:
                    item["declared_share"] = float(p["share"])
                out.append(item)
        for p in self.parents:
            k = self.pkey(p)
            nested = [self.pkey(q) for q, _ in walk_own_parents([p])]
            for a in db.ancestors(k):
                if all(b["model"] != a for b in out):
                    out.append({"model": a, "role": "ancestor"})
            for nk in nested:
                if all(b["model"] != nk for b in out):
                    out.append({"model": nk, "role": "ancestor", "declared": True, "declared_by": k})
                for a in db.ancestors(nk):
                    if all(b["model"] != a for b in out):
                        out.append({"model": a, "role": "ancestor"})
        for t in rec.get("teachers", []):
            rid = t["repo_or_api"].split("@")[0].lower()
            for k, m in db.models.items():
                if m.get("kind") == "entry" and m["repo"].lower() == rid and all(b["model"] != k for b in out):
                    out.append({"model": k, "role": "teacher"})
        for d in overlaps:
            m = {"beta": d["beta"], "cos": d["cos"], "share": d["share"], "joint_weight": d.get("joint_weight"), "rank": d.get("bar_list_rank"), "detected_on": d["detected_on"]}
            have = next((b for b in out if b["model"] == d["model"]), None)
            if have is not None:                       # already there (a teacher): keep its role, add what was measured
                have["weight_overlap"] = m
            else:
                out.append({"model": d["model"], "role": "weight_overlap", "reason": "weight_overlap", "kind": d.get("kind"), "relation": d.get("relation"),
                            "uid": d.get("uid"), **m})
        keep = []
        for b in out:
            m = db.get(b["model"]) or {}
            owner = {i.lower() for i in (m.get("holder_ids") or [])}
            if b["role"] in ("teacher",) and owner & own_ids:
                continue             # own-holder entries leave the list, their ancestors stay (added below)
            keep.append(b)
        for b in out:
            if b not in keep:
                for a in db.ancestors(b["model"]):
                    if all(x["model"] != a for x in keep):
                        keep.append({"model": a, "role": "ancestor of own entry"})
        for d in own_detected:                 # 3.6: an own-holder entry measured in the weights is left out, its ancestors stay
            for a in db.ancestors(d["model"]):
                if all(x["model"] != a for x in keep):
                    keep.append({"model": a, "role": "ancestor of own entry"})
        for b in keep:                         # RULES 2.7: a still-private entry is shown as "a private entry of this round (uid N)" until it is revealed
            if db.is_private(db.get(b["model"])):
                b.update(private=True, public_name=self.display(db, b["model"]))
        return keep

    RECORD_N = ("schema", "record_missing", "hotkey", "entry", "signature", "undeclared_id", "config", "tokenizer", "family_unknown",
                "parent_shares", "parents_duplicate", "parent_self", "parent_not_own")
    START_N = ("start_not_confirmed", "tier_b_unavailable", "transform_found", "start_not_declared", "parents_not_confirmed", "own_parent_not_confirmed")
    HARD_R = ("parents_rule", "parents_template_mismatch")          # a refusal no correction can undo: it beats a record correction (the miner commits again)
    WEIGHT_R = ("copy", "duplicate", "quantisation_round_trip", "quantised_dtype")
    MISDECL_H = ("copy_undeclared",)

    def has(self, cls, codes):
        return any(r["class"] == cls and r["code"] in codes for r in self.reasons)

    def decide(self, res, rec):
        """Decision core of LINEAGE-PROCEDURE.md section 14, in its order; every reason found is still listed.
        v1.3.0 (RULES 3.3 / 3.8 final): MISDECLARATION_CANDIDATE only from an undeclared copy (3.8(a)). A disguise of the declared start that Tier B
        undoes (`transform_found`) is NEEDS_CORRECTION (as v1.1.0; the miner may dispute it, v1.5.1); 3.8(b) for other holders' models is not implemented. A weight overlap
        (any detected source of the base family that is no copy) is never a correction, a refusal or a candidate."""
        result = None
        if self.has("R", self.HARD_R):
            result = "REFUSED"                                # more than 4 parents / a share below 0.10 / parents of different engine templates (RULES 3.3)
        elif self.has("N", self.RECORD_N):
            result = "NEEDS_CORRECTION"                       # record / repo evidence / config / tokenizer first
        elif any(r["class"] == "R" and r["code"] not in self.WEIGHT_R for r in self.reasons):
            result = "REFUSED"                                # start, teacher, data, licence not allowed
        elif self.copy_of_declared_parent and self.has("R", ("copy", "duplicate")):
            result = "REFUSED"                                # a merge that is a copy of one of its parents is a copy (3.5); its other parents are then not found, which is no correction
        elif self.has("N", self.START_N):
            result = "NEEDS_CORRECTION"                       # start not confirmed by the measurement
        elif self.has("R", self.WEIGHT_R):
            result = "REFUSED"                                # copy, duplicate, quantisation round trip, quantised dtype
        elif self.has("H", self.MISDECL_H):
            result = "MISDECLARATION_CANDIDATE"               # undeclared copy (3.8a) only
        if result is None and self.has("N", ("head_source",)):
            result = "NEEDS_CORRECTION"                       # heads taken from another model (2.3): an exact equality, not a share
        res["result"] = result or "PASS"
