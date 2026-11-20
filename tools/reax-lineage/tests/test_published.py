"""Another holder's entry used as a weight parent must have been officially scored in a window whose results were published before the
commitment (`scored_published_at` on the DB row, strictly earlier than the commitment time); the entry's own holder is exempt; without a commitment time (dry run) any publication counts."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import REV, SPEC, World, combine, delta, finetune, make_base                      # noqa: E402
from test_merge import SA, prec                                                         # noqa: E402
from test_pipeline import COMMIT, codes, run                                            # noqa: E402
from reax_lineage.listing import row_published_before                                   # noqa: E402

PUB_OK = "2026-05-31T23:59:59Z"          # one second before COMMIT 2026-06-01T00:00:00Z


def build(tmp_path, spub, holder_repo="bob/ent", **kw):
    w = World(tmp_path)
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    a, b = finetune(root, 31, 0.3), finetune(root, 32, 0.3)
    w.add("org/startA", a, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    w.add(holder_repo, b, kind="entry", start="org/root@" + REV(1), block=500, uid=22, result="PASS", rev=REV(12), public_at="2026-01-02T00:00:00Z", scored_published_at=spub, **kw)
    w.list_path = w.finish()
    c = combine(root, (0.5, delta(a, root)), (0.5, delta(b, root)))
    return w, c


def go(w, c, repo="dina/m", parent="bob/ent", dry=False, teacher=None):
    rec = prec(repo, [(SA, "list", 0.5), ("%s@%s" % (parent, REV(12)), "entry", 0.5, 22)])
    if teacher:
        rec["teachers"] = [{"name": "T", "repo_or_api": teacher, "use": "distillation"}]
    return run(w, repo, c, rec, dry=dry)


def test_published_before_the_commitment_is_allowed(tmp_path):
    w, c = build(tmp_path, PUB_OK)
    res = go(w, c)
    assert res["result"] == "PASS", res["reasons"]


def test_an_entry_never_published_is_not_yet_published(tmp_path):
    w, c = build(tmp_path, None)
    res = go(w, c)
    assert res["result"] == "REFUSED" and codes(res, "R") == ["parent_not_yet_published"], res["reasons"]
    assert res["reasons"][0]["field"] == "parents/1" and "No penalty" in res["reasons"][0]["message"]


def test_published_after_the_commitment_is_refused(tmp_path):
    w, c = build(tmp_path, "2026-06-01T00:00:01Z")
    assert codes(go(w, c), "R") == ["parent_not_yet_published"]


def test_commitment_at_the_publication_second_is_refused(tmp_path):
    w, c = build(tmp_path, COMMIT)
    res = go(w, c)
    assert res["result"] == "REFUSED" and codes(res, "R") == ["parent_not_yet_published"], res["reasons"]


def test_the_entrys_own_holder_is_exempt(tmp_path):
    w, c = build(tmp_path, None, holder_repo="dina/ent")
    res = go(w, c, parent="dina/ent")
    assert "parent_not_yet_published" not in codes(res), res["reasons"]
    assert res["result"] == "PASS", res["reasons"]


def test_dry_run_without_a_commitment_time_accepts_any_publication(tmp_path):
    w, c = build(tmp_path, "2030-01-01T00:00:00Z")
    from reax_lineage.check import Check
    from reax_lineage.sources import Hub
    from synth import write_repo
    rec = prec("dina/m", [(SA, "list", 0.5), ("bob/ent@" + REV(12), "entry", 0.5, 22)])
    rec["entry"] = {"repo": "dina/m", "revision": REV(2000)}
    write_repo(w.hub, "dina/m", REV(2000), c, readme="# model\n", record=rec)
    ck = Check("dina/m", REV(2000), hotkey=None, block=1000, commit_time=None, list_path=w.list_path, db_path=w.dbdir, hub=Hub(hub_dir=w.hub), spec=SPEC,
               cache_dir=None, dry_run=True, seed_b=None, jobs=1, tier_b=False)
    assert "parent_not_yet_published" not in codes(ck.run()), "dry run"


@pytest.mark.parametrize("pa,ct,ok", [
    (None, COMMIT, False), ("", COMMIT, False), (PUB_OK, COMMIT, True), (COMMIT, COMMIT, False), ("2026-06-01T00:00:01Z", COMMIT, False),
    ("2030-01-01T00:00:00Z", None, True), (None, None, False),
])
def test_row_published_before(pa, ct, ok):
    assert row_published_before({"scored_published_at": pa}, ct) is ok


def test_teacher_note_does_not_depend_on_window_publication(tmp_path):
    w, c = build(tmp_path, None)
    rec = prec("dina/t", [(SA, "list", 1.0)])
    res = run(w, "dina/t", c, dict(rec, teachers=[{"name": "T", "repo_or_api": "bob/ent", "use": "distillation"}]))
    assert codes(res, "N") == ["start_not_confirmed"], res["reasons"]
    assert codes(res, "R") == [] and all(b["role"] != "teacher" for b in res["bar_list"])
    w2, c2 = build(tmp_path / "ok", PUB_OK)
    res2 = run(w2, "dina/t", c2, dict(rec, teachers=[{"name": "T", "repo_or_api": "bob/ent", "use": "distillation"}]))
    assert not [x for x in codes(res2, "R") if x.startswith("teacher_")], res2["reasons"]


def test_a_teacher_of_the_own_holder_is_exempt(tmp_path):
    w, c = build(tmp_path, None, holder_repo="dina/ent")
    rec = prec("dina/t", [(SA, "list", 1.0)])
    res = run(w, "dina/t", c, dict(rec, teachers=[{"name": "T", "repo_or_api": "dina/ent", "use": "distillation"}]))
    assert not [x for x in codes(res, "R") if x.startswith("teacher_")], res["reasons"]
