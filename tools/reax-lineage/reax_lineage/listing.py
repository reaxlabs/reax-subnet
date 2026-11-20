"""Step 0.3: allowed-ness against the List at commitment time (RULES-v2 section 4)."""
import json
from typing import Dict, List, Optional, Tuple

from .evidence import MAX_OWN_DEPTH, Problem, normalise_parents

def key(repo, rev):
    return "%s@%s" % (repo, rev)


class Listing:
    """list.json: {"version", "models": [{repo, revision, family, licence, use: start|teacher|both, added (ISO), flagged,
    official, root}], "families": {family: {"root": "repo@rev", "config": {...}}}}"""

    def __init__(self, data: dict):
        self.data = data
        self.version = data.get("version", "unversioned")
        self.models = {key(m["repo"], m["revision"]): m for m in data.get("models", [])}
        self.families = data.get("families", {})

    @staticmethod
    def load(path):
        return Listing(json.load(open(path)))

    def get(self, repo, rev) -> Optional[dict]:
        return self.models.get(key(repo, rev))

    def at(self, m: dict, commit_time: Optional[str]) -> bool:
        """On the List at the commitment time (`added` is an ISO date/time)."""
        if not commit_time or not m.get("added"):
            return True
        return m["added"] <= commit_time

    def family_root(self, family: str) -> Optional[str]:
        return (self.families.get(family) or {}).get("root")

    def detect_family(self, cfg: dict) -> Optional[str]:
        from .evidence import text_config
        c = text_config(cfg)
        for fam, d in self.families.items():
            sig = d.get("config") or {}
            if sig and all(c.get(k) == v for k, v in sig.items()):
                return fam
        return None


def row_public_at(row: dict, commit_time: Optional[str]) -> bool:
    """RULES 2.2 / 4.7 (d8b M2): was this comparison-DB row PUBLIC at the commitment? The caller writes `public_at` (the time we first saw the model / entry public: its
    Hugging Face time, or the reveal time of an entry) and `private` (true while an entry is still private). A row flagged `private`, or without a `public_at`, was not
    public. With a commitment time, `public_at` must not be later; without one (a dry run, a manual run) any `public_at` of a row that is not private counts."""
    if row.get("private"):
        return False
    pa = row.get("public_at")
    if not pa:
        return False
    return not commit_time or str(pa)[:19] <= commit_time[:19]


def row_published_before(row: dict, commit_time: Optional[str]) -> bool:
    """RULES 2.2 / 4.7 (d11, Florian 15 Nov 2026): was this entry's scored publication BEFORE the commitment? The caller (the intake service) writes `scored_published_at` = the time the results of the
    window in which the entry was first officially scored were FIRST published (a window re-published after an objection keeps its first time); a row without it was never officially scored in a
    published window. With a commitment time, `scored_published_at` must be STRICTLY earlier (a commitment at the publication second is refused); without one (a dry run, a manual run) any
    value counts: the dry run asks "is it published now?"."""
    pa = row.get("scored_published_at")
    if not pa:
        return False
    return not commit_time or str(pa)[:19] < commit_time[:19]


def _holders(row: dict) -> set:
    return {i.lower() for i in (row.get("holder_ids") or [])}


def check_start(rec: dict, listing: Listing, db_index: dict, commit_time: Optional[str], own_ids: Optional[set] = None) -> Tuple[List[Problem], Optional[dict]]:
    """-> (problems that make it REFUSED, the start's list/db record). The ONE declared parent of a v1 record or of a /2 record with one parent (legacy codes `start_*`)."""
    parents, _, legacy = normalise_parents(rec)
    if not parents:
        return [], None
    probs, metas = check_parents(parents, listing, db_index, commit_time, own_ids, legacy=legacy)
    return probs, metas[0]


def check_parents(parents: List[dict], listing: Listing, db_index: dict, commit_time: Optional[str], own_ids: Optional[set] = None,
                  legacy: bool = False) -> Tuple[List[Problem], List[Optional[dict]]]:
    """RULES 2.2 / 3.3: is every declared parent allowed? -> (problems that make it REFUSED, one list/db record per parent (None: not found)).
    list  = on the List with use start / both, added before the commitment, not flagged.
    entry = an earlier entry of this round: known, PASS, not void, not flagged, the uid matches, and (another holder's) PUBLIC at the commitment (`parent_not_public`;
            a row flagged `private` or without `public_at` was not public). An entry of your own holder needs none of that.
    own   = your own earlier entry or public model: must be provable as yours (a DB row with your holder id, or a Hugging Face account that is one of your holder ids, RULES 2.9),
            else NEEDS-CORRECTION-like `parent_not_own` (returned with code `parent_not_own`; the caller makes it class N); a flagged / void / not-PASS own entry is refused.
    `own_ids` = the entry's holder ids (lower case). The parents of an own model (`parents` inside the record's own parent) must be allowed in the same way (their codes carry the
    path, e.g. parents/0/parents/1). Legacy (a v1 record): codes `start_*`, field `start`; v2: `parent_*`, field `parents/<i>`."""
    own_ids = {i.lower() for i in (own_ids or [])}
    models = db_index.get("models") or {}
    out: List[Problem] = []
    metas: List[Optional[dict]] = []

    def code(base):
        return ("start_" if legacy else "parent_") + base

    def one(p, path):
        field = "start" if legacy else path
        label = "start" if legacy else "parent"
        kind, repo, rev = p["kind"], p["repo"], p["revision"]
        k = key(repo, rev)
        if kind == "list":
            m = listing.get(repo, rev)
            if not m:
                out.append(Problem(code("not_listed"), "%s %s@%s is not on the List" % (label, repo, rev[:8]), field))
            else:
                if m.get("use") not in ("start", "both"):
                    out.append(Problem(code("not_allowed"), "%s is on the List as a teacher only" % repo, field))
                if m.get("flagged"):
                    out.append(Problem(code("flagged"), "%s is flagged on the Hugging Face Decision Index" % repo, field))
                if not listing.at(m, commit_time):
                    out.append(Problem(code("not_listed"), "%s was added to the List after the commitment" % repo, field))
            return m
        e = models.get(k)
        if kind == "entry":
            if not e or e.get("kind") != "entry":
                out.append(Problem(code("not_listed"), "%s entry %s@%s is not a known entry of this round" % (label, repo, rev[:8]), field))
                return e
            if e.get("result") != "PASS":
                out.append(Problem(code("not_pass"), "%s entry has result %s, only PASS entries can be a %s" % (label, e.get("result"), label), field))
            if e.get("void"):
                out.append(Problem(code("void"), "%s entry %s is void and cannot be a parent for new commitments (RULES 2.2)" % (label, repo), field))
            if e.get("uid") is not None and p.get("uid") is not None and e.get("uid") != p.get("uid"):
                out.append(Problem(code("uid"), "%s.uid %s does not match the entry's uid %s" % (label, p.get("uid"), e.get("uid")), field + ".uid"))
            if e.get("flagged"):
                out.append(Problem(code("flagged"), "%s entry has a flagged ancestor" % label, field))
            if not (_holders(e) & own_ids) and not row_published_before(e, commit_time):
                out.append(Problem(code("not_yet_published"), "%s entry %s of another holder was not officially scored in a window whose results were published before your commitment "
                                   "(RULES 2.2). No penalty: commit a new revision after its window result is published" % (label, repo), field))
            if not (_holders(e) & own_ids) and not row_public_at(e, commit_time):
                out.append(Problem(code("not_public"), "%s entry %s of another holder was not public at your commitment (RULES 2.2: another holder's entry is an allowed parent "
                                   "only if it was public then)%s" % (label, repo, "; it is still private" if e.get("private") else ""), field))
            return e
        # kind == own
        if e is not None:
            if e.get("kind") == "entry" and e.get("result") != "PASS":
                out.append(Problem(code("not_pass"), "your own entry %s has result %s" % (repo, e.get("result")), field))
            if e.get("void"):
                out.append(Problem(code("void"), "your own entry %s is void and cannot be a parent for new commitments (RULES 2.2)" % repo, field))
            if e.get("flagged"):
                out.append(Problem(code("flagged"), "%s has a flagged ancestor" % repo, field))
            if not (_holders(e) & own_ids):
                out.append(Problem("parent_not_own", "%s is declared as your own model, but the check does not find it among your holder ids (RULES 2.9): declare it as kind "
                                   "entry (with its uid), or ask for the account link" % repo, field))
        else:
            if "hf:" + repo.split("/")[0].lower() not in own_ids:
                out.append(Problem("parent_not_own", "%s is declared as your own model, but its Hugging Face account is not one of your holder ids (RULES 2.9): "
                                   "a public model of somebody else that is not on the List is not an allowed parent" % repo, field))
        return e

    for i, p in enumerate(parents):
        metas.append(one(p, "parents/%d" % i))

    def walk(plist, base, depth):                      # the parents of an own model (RULES 3.1): allowed in the same way
        for i, p in enumerate(plist):
            if p.get("kind") == "own" and p.get("parents") and depth < MAX_OWN_DEPTH:
                for j, q in enumerate(p["parents"]):
                    one(q, "%s/%d/parents/%d" % (base, i, j))
                walk(p["parents"], "%s/%d/parents" % (base, i), depth + 1)
    walk(parents, "parents", 0)
    return out, metas
