"""v1.5.1 (checker-v6 open issue 2): a transient Hugging Face failure (HFError kind `transient`: 429, 5xx, timeout, connection error, after the client's own retries) in a step
of the check that used to carry on (`except Exception` -> note / `None`) must never become a NEEDS_CORRECTION for the miner or a silently weaker check. It is raised, so the CLI
prints `error_kind: transient` (exit 2) and the caller retries the whole check later. Steps: the entry's `config.json` / tokenizer read (`_json`), the start repo and its
config, the evidence scan (listing, file reads, id resolution) and Tier B (the Tier B tests are in test_disguise.py). A failure that is NOT transient (404, a missing file) keeps
the old behaviour."""
import copy

import pytest

from reax_lineage.check import Check, is_transient
from reax_lineage.hf import HFError
from reax_lineage.sources import Hub
from synth import HOTKEY, REV, SEED_B, SPEC, finetune, write_repo
from test_pipeline import COMMIT, codes, rec_for, world  # noqa: F401  (the module-scoped `world` fixture)

T503 = HFError("HTTP 503 for https://hf.example/api/x", 503)
T429 = HFError("HTTP 429 for https://hf.example/api/x", 429)
TIMEOUT = HFError("timed out", None, "transient")
E404 = HFError("HTTP 404 for https://hf.example/api/x", 404)


class FlakySource:
    """Wraps an offline source; `fail` maps ("read_file", path) to the exception raised on every read of that file, and ("list_files", n) to the exception raised on the
    n-th call of list_files (the check lists the files once at the record step, then again in the evidence scan)."""

    def __init__(self, src, fail):
        self._src, self._fail, self._n = src, fail, 0

    def __getattr__(self, name):
        return getattr(self._src, name)

    def list_files(self):
        self._n += 1
        if ("list_files", self._n) in self._fail:
            raise self._fail[("list_files", self._n)]
        return self._src.list_files()

    def read_file(self, path, max_bytes=None):
        if ("read_file", path) in self._fail:
            raise self._fail[("read_file", path)]
        return self._src.read_file(path, max_bytes)


class FlakyHub(Hub):
    def __init__(self, hub_dir, fail):
        super().__init__(hub_dir=hub_dir)
        self.fail = fail                        # {repo: {method / (method, path): exception}}

    def source(self, repo, rev):
        s = super().source(repo, rev)
        return FlakySource(s, self.fail[repo]) if repo in self.fail else s


def run_flaky(world, repo, tensors, rec, fail, **kw):
    rev = REV(5000 + abs(hash(repo)) % 10**6)
    rec = copy.deepcopy(rec)
    rec["entry"] = {"repo": repo, "revision": rev}
    write_repo(world.hub, repo, rev, tensors, record=rec, readme=kw.pop("readme", "# model\n"))
    ck = Check(repo, rev, hotkey=HOTKEY, block=1000, commit_time=COMMIT, list_path=world.list_path, db_path=world.dbdir, hub=FlakyHub(world.hub, fail),
               spec=SPEC, cache_dir=None, dry_run=False, seed_b=SEED_B, jobs=1, tier_b=False)
    return ck.run()


def test_is_transient():
    assert is_transient(T503) and is_transient(T429) and is_transient(TIMEOUT)
    assert not is_transient(E404) and not is_transient(ValueError("x")) and not is_transient(FileNotFoundError("x"))


@pytest.mark.parametrize("exc", [T503, T429, TIMEOUT])
def test_a_transient_failure_reading_the_entrys_config_is_raised_not_a_missing_config(world, exc):
    t = finetune(world.tensors["start"], 80, 0.15)
    with pytest.raises(HFError) as ei:
        run_flaky(world, "tr/cfg", t, rec_for(world, "tr/cfg"), {"tr/cfg": {("read_file", "config.json"): exc}})
    assert ei.value.kind == "transient"


def test_a_failure_that_is_not_transient_keeps_the_old_behaviour(world):
    t = finetune(world.tensors["start"], 81, 0.15)
    res = run_flaky(world, "tr/cfg404", t, rec_for(world, "tr/cfg404"), {"tr/cfg404": {("read_file", "config.json"): E404}})
    assert res["result"] == "NEEDS_CORRECTION" and "config" in codes(res, "N")                      # a 404 on config.json = "config.json is missing"
    res2 = run_flaky(world, "tr/cfgmissing", t, rec_for(world, "tr/cfgmissing"), {"tr/cfgmissing": {("read_file", "config.json"): FileNotFoundError("gone")}})
    assert res2["result"] == "NEEDS_CORRECTION" and "config" in codes(res2, "N")


@pytest.mark.parametrize("path", ["config.json", "tokenizer.json"])
def test_a_transient_failure_reading_the_starts_config_or_tokenizer_is_raised(world, path):
    """Before: the start's config (or tokenizer) read failed quietly and the comparison with the start was skipped (a weaker check, no note)."""
    t = finetune(world.tensors["start"], 82, 0.15)
    with pytest.raises(HFError) as ei:
        run_flaky(world, "tr/st-" + path[:3], t, rec_for(world, "tr/st-" + path[:3]), {"org/start": {("read_file", path): T503}})
    assert ei.value.kind == "transient"


def test_a_start_repo_that_cannot_be_opened_transiently_is_raised(world):
    t = finetune(world.tensors["start"], 83, 0.15)

    class OpenFails(FlakyHub):
        def source(self, repo, rev):
            if repo == "org/start":
                raise T503
            return super().source(repo, rev)
    rev = REV(5555)
    rec = copy.deepcopy(rec_for(world, "tr/open"))
    rec["entry"] = {"repo": "tr/open", "revision": rev}
    write_repo(world.hub, "tr/open", rev, t, record=rec)
    ck = Check("tr/open", rev, hotkey=HOTKEY, block=1000, commit_time=COMMIT, list_path=world.list_path, db_path=world.dbdir, hub=OpenFails(world.hub, {}),
               spec=SPEC, cache_dir=None, dry_run=True, jobs=1, tier_b=False)
    with pytest.raises(HFError):
        ck.run()


@pytest.mark.parametrize("which, fail", [("list", {("list_files", 2): T503}), ("read", {("read_file", "README.md"): T429})])
def test_a_transient_failure_in_the_evidence_scan_is_raised_not_a_note(world, which, fail):
    t = finetune(world.tensors["start"], 84, 0.15)
    name = "tr/ev-" + which
    with pytest.raises(HFError) as ei:
        run_flaky(world, name, t, rec_for(world, name), {name: fail})
    assert ei.value.kind == "transient"


def test_an_unreadable_evidence_file_is_still_only_a_note(world):
    t = finetune(world.tensors["start"], 85, 0.15)
    res = run_flaky(world, "tr/ev404", t, rec_for(world, "tr/ev404"), {"tr/ev404": {("read_file", "README.md"): E404}})
    assert res["result"] == "PASS" and any("could not read README.md" in n for n in res["notes"])


class StubClient:
    def __init__(self, exc_for):
        self.exc_for = exc_for

    def api(self, path):
        e = self.exc_for(path)
        if e:
            raise e
        return {}


class StubHub:
    hub_dir = None

    def __init__(self, client):
        self.c = client

    def _client(self):
        return self.c


def test_the_id_resolver_raises_on_a_transient_failure_and_skips_a_missing_id():
    """Before: any HFError while looking up an id meant "this id does not exist", so an undeclared model named in the files could be missed on a hiccup."""
    ck = Check("a/b", "main", hub=StubHub(StubClient(lambda p: T503)))
    with pytest.raises(HFError) as ei:
        ck.resolver()("x/y", "model")
    assert ei.value.kind == "transient"
    ck2 = Check("a/b", "main", hub=StubHub(StubClient(lambda p: E404)))
    assert ck2.resolver()("x/y", "model") is None                                         # 404 everywhere: not a model, not a dataset
    ck3 = Check("a/b", "main", hub=StubHub(StubClient(lambda p: E404 if p.startswith("datasets/") else None)))
    assert ck3.resolver()("x/y", "dataset") == "model"                                   # not a dataset (404), then it is a model
    ck4 = Check("a/b", "main", hub=StubHub(StubClient(lambda p: E404 if p.startswith("datasets/") else T503)))
    with pytest.raises(HFError):                                                         # 404 on the dataset lookup, a hiccup on the model lookup: unknown, not "does not exist"
        ck4.resolver()("x/y", "dataset")
