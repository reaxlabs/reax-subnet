"""F1 (intake replay, 13 Nov): tokenizer_diff was O(V^2) -- the merged vocab dicts were rebuilt for every token inside the comprehensions. With a
248k-262k vocabulary and a tokenizer.json that differs byte-wise from the start's, one check ran 280 s of CPU without finishing. These tests pin
(1) the speed (250k vocab, < 10 s; the diff runs in a child process that is killed at the limit, so a regression fails fast instead of hanging the suite)
and (2) that the output did not change (differential test against a verbatim copy of the v1.4.1 function on random small tokenizers)."""
import multiprocessing as mp
import random
import time

from reax_lineage.evidence import Problem, _tok_parts, tokenizer_diff

V = 250_000
LIMIT_S = 10.0


def tokenizer_diff_v141(entry_tok, start_tok, arch, same_sha=False):
    """Verbatim copy of v1.4.1 (the quadratic version), the reference for the differential test. Do not 'fix' it."""
    if same_sha:
        return []
    if entry_tok is None or start_tok is None:
        return [Problem("tokenizer", "tokenizer.json missing in the entry or in the start: cannot compare", "tokenizer")]
    out = []
    ev, ea, en, ep, em = _tok_parts(entry_tok)
    sv, sa, sn, sp, sm = _tok_parts(start_tok)
    declared = set((arch or {}).get("added_tokens") or [])
    moved = [t for t, i in {**sv, **sa}.items() if {**ev, **ea}.get(t) != i]
    if moved:
        out.append(Problem("tokenizer", "%d start tokens changed id or vanished (e.g. %r)" % (len(moved), moved[0]), "tokenizer"))
    new = [t for t in {**ev, **ea} if t not in {**sv, **sa}]
    undeclared = [t for t in new if t not in declared]
    if undeclared:
        out.append(Problem("tokenizer", "%d added tokens are not in architecture_changes.added_tokens (e.g. %r)" % (len(undeclared), undeclared[0]), "tokenizer"))
    if en != sn:
        out.append(Problem("tokenizer", "normalizer changed", "tokenizer.normalizer"))
    if ep != sp:
        out.append(Problem("tokenizer", "pre_tokenizer changed", "tokenizer.pre_tokenizer"))
    if em != sm and not new:
        out.append(Problem("tokenizer", "BPE merges changed", "tokenizer.merges"))
    return out


def big_tokenizer(n, shift=0, normalizer=None):
    vocab = {"tok%d" % i: (i + shift) for i in range(n)}
    added = [{"id": n + shift + i, "content": "<special%d>" % i} for i in range(8)]
    return {"model": {"type": "BPE", "vocab": vocab, "merges": ["a b"] * 10}, "added_tokens": added, "normalizer": normalizer,
            "pre_tokenizer": {"type": "ByteLevel"}}


def _child(q, entry, start, arch):
    t = time.time()
    out = tokenizer_diff(entry, start, arch)
    q.put(([(p.code, p.message, p.field) for p in out], time.time() - t))


def run_limited(entry, start, arch, limit=LIMIT_S):
    """tokenizer_diff in a forked child; (problems, seconds) or None if it did not finish within `limit` seconds (the child is killed)."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, entry, start, arch))
    p.start()
    try:
        return q.get(timeout=limit)
    except Exception:
        return None
    finally:
        p.kill()
        p.join()


def test_250k_vocab_that_differs_from_the_first_token_is_diffed_in_under_10_seconds():
    """Every start token has another id in the entry (the vocabulary differs from the very first token on), plus a changed normaliser: the worst case
    for the quadratic code (every token is `moved`, every token is looked up)."""
    start = big_tokenizer(V)
    entry = big_tokenizer(V, shift=1, normalizer={"type": "NFC"})
    got = run_limited(entry, start, {})
    assert got is not None, "tokenizer_diff of a %d-token vocabulary did not finish within %.0f s (O(V^2) again?)" % (V, LIMIT_S)
    problems, secs = got
    assert secs < LIMIT_S
    msgs = [m for _, m, _ in problems]
    assert any(m.startswith("%d start tokens changed id or vanished" % (V + 8)) for m in msgs), msgs
    assert any(f == "tokenizer.normalizer" for _, _, f in problems)


def test_250k_vocab_identical_vocabulary_but_other_bytes_is_fast_and_clean():
    """The other way into the slow path: same_sha is false (the file differs, here only in the normaliser) but the vocabulary is identical."""
    start = big_tokenizer(V)
    entry = big_tokenizer(V, normalizer={"type": "NFC"})
    got = run_limited(entry, start, {})
    assert got is not None, "tokenizer_diff of a %d-token vocabulary did not finish within %.0f s (O(V^2) again?)" % (V, LIMIT_S)
    problems, secs = got
    assert [f for _, _, f in problems] == ["tokenizer.normalizer"]


def test_250k_vocab_with_undeclared_new_tokens_is_fast():
    start = big_tokenizer(V)
    entry = big_tokenizer(V)
    entry["model"]["vocab"].update({"new%d" % i: V + 100 + i for i in range(5000)})
    got = run_limited(entry, start, {"added_tokens": ["new0"]})
    assert got is not None, "tokenizer_diff did not finish within %.0f s (O(V^2) again?)" % LIMIT_S
    problems, _ = got
    assert any(m.startswith("4999 added tokens are not in architecture_changes.added_tokens") for _, m, _ in problems), problems


def random_tokenizer(rng, base=None):
    """A small random tokenizer (BPE dict vocab or unigram list vocab), possibly derived from `base` by moving, dropping and adding tokens;
    added tokens may shadow vocabulary entries of the same string (the merge order matters there)."""
    if base is None:
        n = rng.randint(5, 40)
        names = ["t%d" % i for i in range(n)]
        vocab = {t: i for i, t in enumerate(names)}
        added = [{"id": n + i, "content": "<a%d>" % i} for i in range(rng.randint(0, 3))]
        tok = {"model": {"type": "BPE", "vocab": vocab, "merges": ["m%d" % rng.randint(0, 3)]}, "added_tokens": added,
               "normalizer": rng.choice([None, {"type": "NFC"}]), "pre_tokenizer": rng.choice([None, {"type": "ByteLevel"}])}
    else:
        tok = {"model": dict(base["model"]), "added_tokens": [dict(a) for a in base["added_tokens"]], "normalizer": base["normalizer"],
               "pre_tokenizer": base["pre_tokenizer"]}
        tok["model"]["vocab"] = dict(base["model"]["vocab"])
        v = tok["model"]["vocab"]
        for t in list(v):
            r = rng.random()
            if r < 0.08:
                v[t] += rng.randint(1, 3)                 # moved
            elif r < 0.12:
                del v[t]                                  # vanished
        for i in range(rng.randint(0, 4)):
            v["new%d" % i] = 100 + i                      # added in the vocabulary
        if rng.random() < 0.4 and v:
            tok["added_tokens"].append({"id": 7, "content": rng.choice(sorted(v))})       # an added token that shadows a vocabulary entry
        if rng.random() < 0.2:
            tok["normalizer"] = {"type": "NFD"}
        if rng.random() < 0.2:
            tok["pre_tokenizer"] = {"type": "Whitespace"}
        if rng.random() < 0.3:
            tok["model"]["merges"] = ["x y"]
    if rng.random() < 0.25:                               # unigram style: a list of [token, score]
        v = tok["model"]["vocab"]
        tok["model"] = dict(tok["model"], vocab=[[t, 0.0] for t, _ in sorted(v.items(), key=lambda kv: kv[1])])
    return tok


def test_output_is_identical_to_v1_4_1_on_random_tokenizers():
    rng = random.Random(1234)
    n_problem_lists = set()
    for _ in range(400):
        start = random_tokenizer(rng)
        entry = random_tokenizer(rng, base=start) if rng.random() < 0.85 else random_tokenizer(rng)
        arch = rng.choice([{}, None, {"added_tokens": ["new0", "new1", "<a0>"]}])
        old = tokenizer_diff_v141(entry, start, arch)
        new = tokenizer_diff(entry, start, arch)
        assert [(p.code, p.message, p.field) for p in new] == [(p.code, p.message, p.field) for p in old]
        n_problem_lists.add(tuple(p.field for p in new))
    # the random inputs reach every kind of finding (so "identical" is not "identical and empty")
    kinds = {f for t in n_problem_lists for f in t}
    assert {"tokenizer", "tokenizer.normalizer", "tokenizer.pre_tokenizer", "tokenizer.merges"} <= kinds
    assert () in n_problem_lists


def test_edge_cases_are_identical_to_v1_4_1():
    tok = big_tokenizer(50)
    for e, s, same in [(None, tok, False), (tok, None, False), (None, None, False), (tok, tok, True), (tok, tok, False)]:
        a = [(p.code, p.message, p.field) for p in tokenizer_diff_v141(e, s, {}, same)]
        b = [(p.code, p.message, p.field) for p in tokenizer_diff(e, s, {}, same)]
        assert a == b
