"""Procedural image items for the ``s1-vision`` pool of the engine competition (docs/ENGINE-COMPETITION-V2.md, section 9).

Every item is one small picture drawn here from a seed, plus a typed question whose answer depends on what the picture
shows (how many red circles, which bar is the tallest, which number is written). Nothing is looked up or shipped:

* no image files, no fonts, no third-party image licence (shapes, bars and 7-segment digits are rectangles, ellipses
  and triangles; pure Python + zlib, no Pillow);
* the seed of a session is secret and fresh, so there is no fixed image set a miner could precompute or cache;
* ``vision_item(seed, index)`` is a pure function: the same (seed, index) gives the same item on every host.

As in the text items there is no answer key. The validator scores a candidate against the reference engine's answers
for the same picture, so the picture alone has to make a model's distribution move: every item depends on its image.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import random
import secrets
import struct
import zlib
from typing import Any

from .s1items import S1Item, _levels

# GENERATOR_RULE (below) is part of composite_rules_sha256 of every track with an image slot: it binds the generator by its
# output, so a change of any family, palette or text is a new rule set even if the label is not bumped.
WIDTH, HEIGHT = 320, 240
CELL_W, CELL_H = 80, 80  # a 4 x 3 grid of cells: one shape per cell, so shapes never overlap

_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (214, 40, 40), "green": (40, 150, 66), "blue": (40, 88, 214), "yellow": (236, 206, 36),
    "orange": (238, 132, 26), "purple": (136, 58, 176),
}
_BACKGROUNDS = ((255, 255, 255), (245, 240, 226), (234, 238, 246), (248, 244, 250))
_SHAPES = ("circle", "square", "triangle")
_DIGITS = {  # 7-segment digits: a top, b top right, c bottom right, d bottom, e bottom left, f top left, g middle
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}
_TASKS = ("Picture exercise", "Image quiz", "Visual check", "Worksheet", "Look-and-answer card")
_LEVELS_CROWD = ["Empty", "Sparse", "Moderate", "Crowded", "Packed"]
_LEVELS_BARS = ["Almost even", "A little uneven", "Uneven", "Very uneven"]


# ------------------------------------------------------------------------------------------------ raster + PNG
class Canvas:
    """An RGB canvas of ``bytearray`` rows. Only spans are written (a row slice per rectangle row), so a picture takes a
    few milliseconds."""

    def __init__(self, background: tuple[int, int, int]):
        self.rows = [bytearray(bytes(background) * WIDTH) for _ in range(HEIGHT)]

    def rect(self, x0: int, y0: int, x1: int, y1: int, colour: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x0), min(WIDTH, x1)
        if x1 <= x0:
            return
        span = bytes(colour) * (x1 - x0)
        for y in range(max(0, y0), min(HEIGHT, y1)):
            self.rows[y][x0 * 3:x1 * 3] = span

    def ellipse(self, cx: int, cy: int, rx: int, ry: int, colour: tuple[int, int, int]) -> None:
        for y in range(max(0, cy - ry), min(HEIGHT, cy + ry + 1)):
            half = int(rx * math.sqrt(max(0.0, 1.0 - ((y - cy) / ry) ** 2)))
            self.rect(cx - half, y, cx + half + 1, y + 1, colour)

    def triangle(self, cx: int, cy: int, r: int, colour: tuple[int, int, int]) -> None:
        """Isosceles, apex up, inside the box of half size ``r`` around (cx, cy)."""
        for y in range(max(0, cy - r), min(HEIGHT, cy + r + 1)):
            half = (y - (cy - r)) * r // (2 * r)
            self.rect(cx - half, y, cx + half + 1, y + 1, colour)

    def raw(self) -> bytes:
        return b"".join(b"\x00" + bytes(row) for row in self.rows)  # filter type 0 on every row

    def png(self) -> bytes:
        raw = self.raw()

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _jitter(rng: random.Random, colour: tuple[int, int, int], amount: int = 10) -> tuple[int, int, int]:
    return tuple(min(255, max(0, c + rng.randint(-amount, amount))) for c in colour)  # type: ignore[return-value]


def _shape(canvas: Canvas, kind: str, cx: int, cy: int, r: int, colour: tuple[int, int, int]) -> None:
    if kind == "circle":
        canvas.ellipse(cx, cy, r, r, colour)
    elif kind == "square":
        canvas.rect(cx - r, cy - r, cx + r + 1, cy + r + 1, colour)
    else:
        canvas.triangle(cx, cy, r, colour)


def _cells(rng: random.Random, n: int) -> list[tuple[int, int]]:
    """Centres of ``n`` distinct grid cells, with a little jitter that keeps a shape of radius <= 30 inside its cell."""
    grid = [(c, r) for c in range(WIDTH // CELL_W) for r in range(HEIGHT // CELL_H)]
    rng.shuffle(grid)
    return [(c * CELL_W + CELL_W // 2 + rng.randint(-5, 5), r * CELL_H + CELL_H // 2 + rng.randint(-5, 5))
            for c, r in grid[:n]]


# ------------------------------------------------------------------------------------------------ item families
def _count(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    colour, shape = rng.choice(list(_COLOURS)), rng.choice(_SHAPES)
    k = rng.choice((0, 1, 2, 2, 3, 3, 4, 5, 6))
    others = [(c, s) for c in _COLOURS for s in _SHAPES if (c, s) != (colour, shape)]
    n = min(12, k + rng.randint(2, 6))
    for i, (cx, cy) in enumerate(_cells(rng, n)):
        c, s = (colour, shape) if i < k else rng.choice(others)
        _shape(canvas, s, cx, cy, rng.randint(18, 29), _jitter(rng, _COLOURS[c]))
    criteria = {str(i): (None if rng.random() < 0.5 else f"{i} of them") for i in range(7)}
    return "choice", f"How many {colour} {shape}s are in the picture?", criteria


def _largest(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    m = rng.randint(3, 6)
    names = rng.sample(list(_COLOURS), m)
    sizes = rng.sample(range(12, 31, 3), m)  # distinct radii: one clear winner
    kind = rng.choice(_SHAPES)  # one shape kind per picture: the radius then orders the areas as well
    for (cx, cy), name, r in zip(_cells(rng, m), names, sizes):
        _shape(canvas, kind, cx, cy, r, _jitter(rng, _COLOURS[name]))
    # every colour is an option, in random order, whatever the picture shows: the option list says nothing about the answer
    criteria = {n: (None if rng.random() < 0.6 else n) for n in rng.sample(list(_COLOURS), len(_COLOURS))}
    return "choice", rng.choice(("Which colour is the largest shape?", "What colour is the biggest shape in the picture?")), criteria


def _present(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    target = rng.choice(_SHAPES)
    answer = rng.random() < 0.5  # decided first, then drawn: yes and no are equally likely whatever the shape count or palette
    pool = list(_SHAPES) if answer else [s for s in _SHAPES if s != target]
    cells = _cells(rng, rng.randint(3, 9))
    forced = rng.randrange(len(cells)) if answer else -1  # a "yes" picture holds at least one target
    for i, (cx, cy) in enumerate(cells):
        _shape(canvas, target if i == forced else rng.choice(pool), cx, cy, rng.randint(16, 29),
               _jitter(rng, _COLOURS[rng.choice(list(_COLOURS))]))
    return "noul", f"Is there a {target} in the picture?", (None if rng.random() < 0.5 else {"true": "Yes", "false": "No"})


def _bars(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    n = rng.randint(3, 6)
    names = rng.sample(list(_COLOURS), n)
    heights = rng.sample(range(24, 181, 6), n)  # distinct heights
    base, slot = HEIGHT - 22, (WIDTH - 40) // n
    width = int(slot * 0.62)
    for i, (name, h) in enumerate(zip(names, heights)):
        x = 24 + i * slot + (slot - width) // 2
        canvas.rect(x, base - h, x + width, base, _jitter(rng, _COLOURS[name]))
    canvas.rect(18, base, WIDTH - 12, base + 3, (30, 30, 30))  # axes
    canvas.rect(18, 10, 21, base + 3, (30, 30, 30))
    way = rng.randrange(4)
    if way == 0:
        return "choice", "Which bar is the tallest?", {c: None for c in rng.sample(names, n)}
    if way == 1:
        return "choice", "Which bar is the shortest?", {c: None for c in rng.sample(names, n)}
    if way == 2:
        a, b = rng.sample(names, 2)
        return "noul", f"Is the {a} bar taller than the {b} bar?", None
    return "score", "How uneven are the bars?", _levels(rng, _LEVELS_BARS)


def _digit(canvas: Canvas, d: int, x: int, y: int, colour: tuple[int, int, int], w: int = 44, h: int = 78, t: int = 9) -> None:
    mid = y + h // 2
    seg = {"a": (x + t, y, x + w - t, y + t), "b": (x + w - t, y, x + w, mid), "c": (x + w - t, mid, x + w, y + h),
           "d": (x + t, y + h - t, x + w - t, y + h), "e": (x, mid, x + t, y + h), "f": (x, y, x + t, mid),
           "g": (x + t, mid - t // 2, x + w - t, mid - t // 2 + t)}
    for s in _DIGITS[d]:
        canvas.rect(*seg[s], colour)


def _number(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    value = rng.randint(100, 999)
    colour = _jitter(rng, _COLOURS[rng.choice(list(_COLOURS))])
    x0, y0 = 40 + rng.randint(-8, 8), 70 + rng.randint(-14, 14)
    for i, ch in enumerate(str(value)):
        _digit(canvas, int(ch), x0 + i * 82, y0, colour)
    if rng.random() < 0.3:
        return "noul", "Is the number in the picture even?", None
    # distractors are drawn like the truth (uniform 3-digit numbers), independent of it: no option is the odd one out,
    # nearest to the others, or otherwise special without the picture
    options = {value}
    want = rng.randint(4, 6)
    while len(options) < want:
        options.add(rng.randint(100, 999))
    keys = [str(v) for v in options]
    rng.shuffle(keys)
    return "choice", "Which number is written in the picture?", {k: None for k in keys}


def _crowd(rng: random.Random, canvas: Canvas) -> tuple[str, Any, Any]:
    k = rng.choice((0, 2, 5, 9, 14, 22, 35, 50, 70))
    colour = _jitter(rng, _COLOURS[rng.choice(list(_COLOURS))])
    for _ in range(k):
        canvas.ellipse(rng.randint(8, WIDTH - 9), rng.randint(8, HEIGHT - 9), 6, 6, colour)
    return "score", rng.choice(("How crowded is the picture?", "How many dots does the picture have, roughly?")), list(_LEVELS_CROWD)


_FAMILIES = (_count, _count, _count, _largest, _largest, _present, _present, _bars, _bars, _bars, _number, _number, _crowd)


# ------------------------------------------------------------------------------------------------ public API
def _rng(seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"reax-vision/1|{seed}|{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _make(seed: int, index: int) -> tuple[S1Item, Canvas]:
    rng = _rng(seed, index)
    canvas = Canvas(_jitter(rng, rng.choice(_BACKGROUNDS), 3))
    kind, instructions, criteria = rng.choice(_FAMILIES)(rng, canvas)
    task = f"{rng.choice(_TASKS)} {rng.randint(1, 99)}: look at the picture and answer the question."
    state: Any = task if rng.random() < 0.6 else {"task": task, "level": rng.choice(["easy", "normal", "hard"])}
    return S1Item("%032x" % rng.getrandbits(128), state, kind, instructions, criteria, ()), canvas


def vision_item(seed: int, index: int) -> S1Item:
    """Item ``index`` of the session with secret ``seed``: one picture and one question about it. Pure and deterministic."""
    item, canvas = _make(seed, index)
    return dataclasses.replace(item, images=(data_url(canvas.png()),))


def generator_sha256(samples: int = 16) -> str:
    """sha256 over a fixed-seed sample of the generator (seed 0, items 0..samples-1): the raw pixels of each picture (not the
    PNG file, whose zlib stream may differ between zlib builds) and its item id, state and question. Any change of a family,
    palette, radius, text or draw order changes it."""
    h = hashlib.sha256()
    for i in range(samples):
        item, canvas = _make(0, i)
        h.update(hashlib.sha256(canvas.raw()).digest())
        h.update(json.dumps([item.item_id, item.state, item.question()], sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


GENERATOR_RULE = "reax-vision/2 png 320x240 sha256-seeded families count/largest/present/bars/number/crowd content:" + generator_sha256()


def generate_vision_items(n: int, seed: int | None = None) -> list[S1Item]:
    """``n`` fresh image items. ``seed`` is for tests and calibration only (None: a secret 128-bit seed)."""
    base = secrets.randbits(128) if seed is None else seed
    return [vision_item(base, i) for i in range(n)]
