"""critic22 M4 / m7 / m1: the vision generator must not leak its answer through the question text alone, and the rules hash binds
the generator by its output. The truth of an item is read back from what the family draws (no answer key exists in production)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from reax_subnet import engcomposite as ecmp
from reax_subnet import s1vision as V

N = 3000
TOL = 0.045  # ~4 sigma of a hit rate over ~1400 multiple-choice items


class _Rec:
    def __init__(self, monkeypatch):
        self.digits: list[int] = []
        self.shapes: list[tuple[str, int, tuple[int, int, int]]] = []
        monkeypatch.setattr(V, "_digit", lambda canvas, d, *a, **k: self.digits.append(d))
        monkeypatch.setattr(V, "_shape", lambda canvas, kind, cx, cy, r, colour: self.shapes.append((kind, r, colour)))
        monkeypatch.setattr(V, "_jitter", lambda rng, colour, amount=10: colour)  # the base colour identifies the name
        self.names = {v: k for k, v in V._COLOURS.items()}

    def run(self, family, seed):
        self.digits, self.shapes = [], []
        rng = random.Random(seed)
        canvas = V.Canvas((255, 255, 255))
        return family(rng, canvas)


def _hit_rate(picks, truths):
    return sum(p == t for p, t in zip(picks, truths)) / len(truths)


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def test_number_options_do_not_reveal_the_written_number(monkeypatch):
    rec = _Rec(monkeypatch)
    rows = []
    for seed in range(N):
        kind, text, criteria = rec.run(V._number, seed)
        if kind != "choice":
            continue
        rows.append(([k for k in criteria], "".join(map(str, rec.digits))))
    assert len(rows) > 2000
    expected = sum(1 / len(o) for o, _ in rows) / len(rows)

    def central_hamming(opts):
        return min(opts, key=lambda o: sum(_hamming(o, p) for p in opts))

    def central_numeric(opts):
        return min(opts, key=lambda o: sum(abs(int(o) - int(p)) for p in opts))

    def median(opts):
        return sorted(opts, key=int)[len(opts) // 2]

    heuristics = {"hamming-centre": central_hamming, "numeric-centre": central_numeric, "median": median,
                  "largest": lambda o: max(o, key=int), "smallest": lambda o: min(o, key=int), "first": lambda o: o[0],
                  "last": lambda o: o[-1],
                  "most digits shared with the others": lambda o: max(o, key=lambda x: sum(a == b for p in o for a, b in zip(x, p)))}
    for name, h in heuristics.items():
        rate = _hit_rate([h(o) for o, _ in rows], [t for _, t in rows])
        assert abs(rate - expected) < TOL, (name, rate, expected)
    # the options are not tied to the truth by distance either: the nearest-neighbour count of the truth is as for any option
    near = Counter()
    for opts, truth in rows:
        for o in opts:
            near[o == truth] += sum(1 for p in opts if p != o and _hamming(o, p) == 1)
    assert near[True] / len(rows) < 0.35  # old generator: every distractor was one digit away from the truth (>= 3 per item)


def test_the_test_has_power_the_old_digit_swap_distractors_are_caught():
    rng = random.Random(1)
    hits = n = 0
    for _ in range(1500):
        value = rng.randint(100, 999)
        options = {value}
        want = rng.randint(4, 6)
        while len(options) < want:
            digits = list(str(value))
            i = rng.randrange(3)
            digits[i] = str((int(digits[i]) + rng.randint(1, 9)) % 10)
            if digits[0] != "0":
                options.add(int("".join(digits)))
        opts = [str(v) for v in options]
        rng.shuffle(opts)
        pick = min(opts, key=lambda o: sum(_hamming(o, p) for p in opts))
        hits += pick == str(value)
        n += 1
    assert hits / n > 0.8  # the previous rule: the Hamming centre was the truth ~96 % of the time


def test_largest_lists_every_colour_and_the_option_text_alone_is_at_chance(monkeypatch):
    rec = _Rec(monkeypatch)
    picks_first, picks_alpha, truths = [], [], []
    kinds_per_picture = set()
    for seed in range(N):
        kind, text, criteria = rec.run(V._largest, seed)
        assert list(sorted(criteria)) == sorted(V._COLOURS)  # all six colours, whatever the picture shows
        kinds_per_picture.add(len({k for k, _, _ in rec.shapes}))
        big = max(rec.shapes, key=lambda s: s[1])
        truths.append(rec.names[big[2]])
        opts = list(criteria)
        picks_first.append(opts[0])
        picks_alpha.append(min(opts))
    assert kinds_per_picture == {1}  # m7: one shape kind per picture
    for picks in (picks_first, picks_alpha):
        assert abs(_hit_rate(picks, truths) - 1 / len(V._COLOURS)) < 0.03
    assert max(Counter(truths).values()) / N < 0.2  # no colour is the answer much more often than 1/6


def test_present_is_balanced_whatever_the_shape_count_and_palette(monkeypatch):
    yes = []
    for seed in range(N):
        rng = random.Random(seed)
        canvas = V.Canvas((255, 255, 255))
        shapes = []
        real = V._shape
        monkeypatch.setattr(V, "_shape", lambda c, kind, cx, cy, r, colour, shapes=shapes: shapes.append(kind))
        kind, text, _ = V._present(rng, canvas)
        monkeypatch.setattr(V, "_shape", real)
        target = text.split()[3]
        yes.append((target in shapes, len(shapes)))
    rate = sum(a for a, _ in yes) / N
    assert abs(rate - 0.5) < 0.04
    for n in range(3, 10):  # the picture's shape count says nothing about the answer
        sub = [a for a, k in yes if k == n]
        assert abs(sum(sub) / len(sub) - 0.5) < 0.12, n


def test_the_rules_hash_binds_the_generator_by_its_output(monkeypatch):
    assert V.GENERATOR_RULE.endswith(V.generator_sha256()) and V.GENERATOR_RULE.startswith("reax-vision/2 ")
    base = V.generator_sha256()
    assert V.generator_sha256() == base  # pure
    original = V._COLOURS["red"]
    monkeypatch.setitem(V._COLOURS, "red", (215, 40, 40))  # one colour channel moved by one: same label, other generator
    assert V.generator_sha256() != base
    monkeypatch.setitem(V._COLOURS, "red", original)
    assert V.generator_sha256() == base
    monkeypatch.setattr(V, "_TASKS", V._TASKS[:-1] + ("Other card",))
    assert V.generator_sha256() != base


def test_the_rules_carry_the_pool_option_caps_and_the_image_caps(monkeypatch):
    from reax_subnet import s1contract as C

    track = ecmp.default_track()
    rules = ecmp.composite_rules(track)
    vision = next(s for s in rules["slots"] if s["pool"] == "s1-vision")
    assert vision["max_options"] == C.MAX_OPTIONS["s1-vision"] and vision["max_image_bytes"] == C.MAX_IMAGE_BYTES
    assert vision["max_image_pixels"] == C.MAX_IMAGE_PIXELS and vision["image_items"] == V.GENERATOR_RULE
    base = ecmp.composite_rules_sha256(track)
    monkeypatch.setitem(C.MAX_OPTIONS, "s1-vision", C.MAX_OPTIONS["s1-vision"] + 1)
    assert ecmp.composite_rules_sha256(track) != base
