"""Round 2 Generalist podium runtime (feature flag REAX_GR2_PODIUM): fragment reader, weight integration, set-bound record.

Plain ``unittest`` (no pytest needed), local only, network mocked. Run from the repo root with

    python3.10 -I -c "import sys,unittest; sys.path[:0]=['src','tests']; unittest.main(module=None, argv=['x','test_gr2_podium_runtime'])"
"""
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

from reax_subnet import gr2_podium_reader as P  # noqa: E402
from reax_subnet import gr2_podium_record as REC  # noqa: E402
from reax_subnet.chain import Neuron, Snapshot  # noqa: E402
from reax_subnet.competition_split import RankedLeaders, _need_weights  # noqa: E402
from reax_subnet.engvalidator import EngineCompetition  # noqa: E402
from reax_subnet.gr2_podium import (  # noqa: E402
    OPENING_SHARES, STANDARD_SHARES, payout_after_keepalive, payout_fractions,
)
import gr2_podium_flag_off_scenarios as SC  # noqa: E402

FIXTURES = HERE / "fixtures" / "gr2-podium"
OPENING, SECOND, THIRD = "2026-11-26", "2026-12-03", "2026-12-10"
PUBLISHED = "2026-11-26T15:00:00Z"
AFTER = datetime(2026, 11, 27, 9, 0, tzinfo=timezone.utc)
OWNER_UID, SERVING_UID, KA_UID = 0, 251, 60
BASE = {OWNER_UID: .1, SERVING_UID: .9}
BURN_FRACTION = .1
MINER_PART = .9
POOL, KEEPALIVE = .30, .075          # the generalist slot and the keep-alive cap (25 %), fractions of the miner part
REST = POOL - KEEPALIVE


def ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def hk(uid):
    return f"5Hot{uid:03d}" + "x" * 40


def commit(uid):
    return f"reax-spec/1:generalist:lab{uid}/model@{uid:040x}"


def rec(uid, start=PUBLISHED, **over):
    out = dict(uid=uid, hotkey=hk(uid), holder=f"5Cold{uid:03d}", holder_ids=[f"5Cold{uid:03d}"], holder_source="intake",
               repo=f"lab{uid}/model", revision=f"{uid:040x}", entry_id=f"E{uid}", private=False, revealed_at=None,
               held=False, effective_from=start, effective_until=None)
    out.update(over)
    return out


def tenure(uid, start=PUBLISHED, end=None):
    return dict(uid=uid, hotkey=hk(uid), holder=f"5Cold{uid:03d}", effective_from=start, effective_until=end,
                state="current" if end is None else "ended", entry_id=f"E{uid}")


def fragment(window, champion=None, former=(None, None), place2=None, place3=None, tenures="auto", problems=(),
             shares=None, **podium):
    shares = dict(shares or (OPENING_SHARES if window == OPENING else STANDARD_SHARES))
    slots = {"champion": champion, "former[0]": former[0], "former[1]": former[1], "place2": place2, "place3": place3}
    paid = {s: shares[s] for s, r in slots.items() if r and not r["held"] and shares[s]}
    if tenures == "auto":
        tenures = [] if champion is None else [tenure(champion["uid"], champion["effective_from"])]
    paid_total = math.fsum(paid.values())
    body = dict(schema="reax-round2-podium/1", window=window, shares=shares, champion=champion, pending_champion=None,
                former=list(former), place2=place2, place3=place3, title_tenures=tenures, paid_shares=paid,
                paid_total=round(paid_total, 12), burned_total=round(1.0 - paid_total, 12), burned_places=[])
    body.update(podium)
    return dict(schema="reax-round2-award-fragment/1", window=window, track_id="generalist", decision="crown",
                competitions_json=None, podium=body, burned_places=[], extra={}, problems=list(problems))


def blob(doc):
    return json.dumps(doc, indent=1, sort_keys=True).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _filled(d):
    return {k: v for k, v in d.items() if v is not None}


class Net:
    """The mocked transport: a dict url -> bytes (or an exception to raise)."""

    def __init__(self):
        self.files, self.calls = {}, []

    def row(self, window, digest):
        entry = {"id": "generalist"}
        if window is not None:
            entry.update(podium_window=window, podium_sha256=digest)
        self.files[P.COMPETITIONS_URL] = json.dumps({"schema": 2, "tracks": [{"id": "engine"}, entry]}).encode()

    def publish(self, window, doc, row_sha=None):
        raw = blob(doc) if isinstance(doc, dict) else doc
        self.files[P.FRAGMENT_URL.format(window=window)] = raw
        self.row(window, row_sha or sha(raw))
        return sha(raw)

    def __call__(self, url, limit):
        self.calls.append(url)
        value = self.files.get(url)
        if value is None:
            raise P.FetchError("not_found")
        if isinstance(value, Exception):
            raise value
        if len(value) > limit:
            raise P.FetchError("too_large")
        return value

    def fragment_calls(self):
        return [c for c in self.calls if "/podium/" in c]


def neurons(uids=(15, 25, 35, 45, 99), override=None):
    override = override or {}
    out = [Neuron(OWNER_UID, "owner", "oc", False, 0., None, None), Neuron(SERVING_UID, "serving", "sc", False, 0., None, None)]
    for uid in uids:
        out.append(Neuron(uid, override.get(uid, {}).get("hotkey", hk(uid)), f"ck{uid}", False, 0., None, None,
                          commitment=override.get(uid, {}).get("commitment", commit(uid))))
    return out


def snapshot(override=None):
    return Snapshot(1000, tuple(neurons(override=override)), "owner")


class Harness:
    """A minimal real ``EngineCompetition`` (generalist track only) with a mocked clock, relay and award gate."""

    def __init__(self, net=None, approved=(15, 25, 35, 45, 60, 99), ranks=None, alive=((KA_UID, 100.0),), now=AFTER):
        self.clock = {"now": now}
        self.net = net
        self.engine = EngineCompetition(
            SC.Runner(), {}, "/nonexistent/engine.json", tracks={}, start_utc=None,
            specialist=SC.Specialist(RankedLeaders(ranks) if ranks else None, list(alive)),
            award_gate=SC.Gate([hk(u) for u in approved] + ["serving"]), utcnow=lambda: self.clock["now"])
        if net is not None:
            self.engine.podium_reader = P.PodiumReader(net)

    def weights(self, snap=None):
        return self.engine.weights(dict(BASE), snap or snapshot(), set(), BURN_FRACTION)

    def record(self):
        return self.engine.podium_record()


def expected_vector(pay_fractions, keepalive=KEEPALIVE):
    """The staged calculation (replay.py): podium after keep-alive, through the existing _need_weights."""
    podium = payout_after_keepalive(POOL, keepalive, pay_fractions)
    return _need_weights(BASE, {**podium, KA_UID: keepalive}, OWNER_UID, BURN_FRACTION, .4, 0), podium


class FragmentRules(unittest.TestCase):
    def parse(self, doc, window=None):
        raw = blob(doc) if isinstance(doc, dict) else doc
        return P.parse_fragment(raw, window or doc["window"], sha(raw))

    def refused(self, doc, reason, window=None):
        with self.assertRaises(P.PodiumError) as caught:
            self.parse(doc, window)
        self.assertEqual(caught.exception.reason, reason)

    def test_opening_full_podium_vacant_third_and_standard_window(self):
        full = self.parse(fragment(OPENING, rec(15), place2=rec(25), place3=rec(35)))
        self.assertEqual({s: p.uid for s, p in full.eligible_at(ts("2026-11-26T15:00:01Z")).items()},
                         {"champion": 15, "place2": 25, "place3": 35})
        vacant = self.parse(fragment(OPENING, rec(15), place2=rec(25)))
        self.assertEqual(set(vacant.eligible_at(ts("2026-11-27T00:00:00Z"))), {"champion", "place2"})
        standard = self.parse(fragment(SECOND, rec(45), former=(rec(15), None), place2=rec(25), place3=rec(35),
                                       tenures=[tenure(15, "2026-11-26T14:00:00Z", PUBLISHED), tenure(45)]))
        self.assertEqual({s: p.uid for s, p in standard.eligible_at(ts("2026-12-03T15:00:00Z")).items()},
                         {"champion": 45, "former[0]": 15, "place2": 25, "place3": 35})

    def test_old_table_refused_for_the_opening_window_and_the_new_one_for_later(self):
        self.refused(fragment(OPENING, rec(15), shares=STANDARD_SHARES), "wrong_shares")
        self.refused(fragment(SECOND, rec(15), shares=OPENING_SHARES), "wrong_shares")
        self.refused(fragment(THIRD, rec(15), shares=OPENING_SHARES), "wrong_shares")
        self.refused(fragment("2026-12-17", rec(15)), "unknown_window")

    def test_duplicate_uid_and_duplicate_holder_refused(self):
        self.refused(fragment(OPENING, rec(15), place2=rec(15)), "duplicate_uid")
        self.refused(fragment(OPENING, rec(15), place2=rec(25, holder_ids=["5Cold015"])), "duplicate_holder")

    def test_missing_or_mistyped_fields_refused_with_the_field_named(self):
        doc = fragment(OPENING, rec(15), place2=rec(25))
        bad = copy.deepcopy(doc)
        del bad["podium"]["champion"]["hotkey"]
        self.refused(bad, "missing_field.podium.champion.hotkey")
        bad = copy.deepcopy(doc)
        bad["podium"]["place2"]["uid"] = None          # the redacted-record fallback shape is NOT accepted
        self.refused(bad, "bad_type.podium.place2.uid")
        bad = copy.deepcopy(doc)
        bad["podium"]["place2"]["uid"] = True
        self.refused(bad, "bad_type.podium.place2.uid")
        bad = copy.deepcopy(doc)
        del bad["podium"]["title_tenures"]
        self.refused(bad, "missing_field.podium.title_tenures")
        bad = copy.deepcopy(doc)
        bad["podium"]["place2"]["effective_from"] = "tomorrow"
        self.refused(bad, "bad_time.podium.place2.effective_from")
        bad = copy.deepcopy(doc)
        del bad["problems"]
        self.refused(bad, "missing_field.fragment.problems")

    def test_window_is_pinned_and_schema_checked(self):
        doc = fragment(OPENING, rec(15))
        self.refused(doc, "window_mismatch", window=SECOND)
        bad = copy.deepcopy(doc)
        bad["podium"]["window"] = SECOND
        self.refused(bad, "window_mismatch")
        bad = copy.deepcopy(doc)
        bad["track_id"] = "engine"
        self.refused(bad, "fragment_schema")

    def test_problems_only_the_coldkey_note_is_tolerated(self):
        note = ("holder of podium.champion, podium.former[0] is the coldkey: the intake service exposes no holder id "
                "(RULES 2.9), so one place per holder cannot merge holders that share only a Hugging Face account")
        self.parse(fragment(OPENING, rec(15), problems=[note]))
        self.refused(fragment(OPENING, rec(15), problems=[note, "podium.place3 (uid 3): hotkey missing"]), "fragment_problems")
        self.refused(fragment(OPENING, rec(15), problems=["title_tenures: E1 ends x but E2 begins y"]), "fragment_problems")

    def test_title_tenure_and_pending_invariants(self):
        gap = fragment(SECOND, rec(45), former=(rec(15), None),
                       tenures=[tenure(15, "2026-11-26T14:00:00Z", "2026-12-03T14:00:00Z"), tenure(45)])
        self.refused(gap, "tenure_gap")
        late = fragment(SECOND, rec(45), former=(rec(15, start="2026-12-03T16:00:00Z"), None),
                        tenures=[tenure(15, "2026-11-26T14:00:00Z", "2026-12-03T15:00:00Z"),
                                 tenure(45, "2026-12-03T15:00:00Z")])
        self.refused(late, "tenure_former_start")
        other_champion = fragment(SECOND, rec(45), tenures=[tenure(15)])
        self.refused(other_champion, "tenure_champion")
        hotkey = fragment(SECOND, rec(45), tenures=[dict(tenure(45), hotkey="someone-else")])
        self.refused(hotkey, "tenure_hotkey")
        pending = fragment(SECOND, rec(15), pending_champion=rec(45, start="2026-12-04T09:00:00Z"))
        self.refused(pending, "pending_paid")
        closed = fragment(SECOND, rec(15, effective_until="2026-12-04T09:00:00Z"),
                          pending_champion=rec(45, start=None))
        self.refused(closed, "pending_tenure_closed")
        held_paid = fragment(SECOND, rec(15, held=True))
        held_paid["podium"]["paid_shares"] = {"champion": .7}
        held_paid["podium"].update(paid_total=.7, burned_total=.3)
        self.refused(held_paid, "held_but_paid")
        totals = fragment(SECOND, rec(15))
        totals["podium"]["burned_total"] = .5
        self.refused(totals, "totals_mismatch")

    def test_sha_is_checked_before_anything_is_parsed(self):
        garbage = b"\xff not json {"
        with self.assertRaises(P.PodiumError) as caught:
            P.parse_fragment(garbage, OPENING, "0" * 64)
        self.assertEqual(caught.exception.reason, "sha_mismatch")
        with self.assertRaises(P.PodiumError) as caught:
            P.parse_fragment(garbage, OPENING, sha(garbage))      # right sha, still not JSON
        self.assertEqual(caught.exception.reason, "fragment_not_json")
        for text in (b'{"a":1,"a":2}', b'{"a": NaN}', b"[1, 2", b'{"a": Infinity}'):
            with self.assertRaises(P.PodiumError) as caught:
                P.parse_fragment(text, OPENING, sha(text))
            self.assertEqual(caught.exception.reason, "fragment_not_json", text)
        big = b" " * (P.MAX_FRAGMENT_BYTES + 1)
        with self.assertRaises(P.PodiumError) as caught:
            P.parse_fragment(big, OPENING, sha(big))
        self.assertEqual(caught.exception.reason, "fragment_too_large")

    def test_effective_instants_follow_tenures_reveal_and_hold(self):
        raw = (FIXTURES / "3-after-reveal.json").read_bytes()      # kit case 3: private winner revealed 09:30Z, place 3 at 12:00Z
        fragment_ = P.parse_fragment(raw, SECOND, sha(raw))

        def at(text):
            return {s: p.uid for s, p in fragment_.eligible_at(ts(text)).items()}
        self.assertEqual(at("2026-12-03T14:59:59Z"), {"champion": 1})
        self.assertEqual(at("2026-12-03T15:00:00Z"), {"champion": 1, "place2": 12})
        self.assertEqual(at("2026-12-04T09:29:59Z"), {"champion": 1, "place2": 12})
        self.assertEqual(at("2026-12-04T09:30:00Z"), {"champion": 11, "former[0]": 1, "place2": 12})   # exact instant
        self.assertEqual(at("2026-12-04T12:00:00Z"), {"champion": 11, "former[0]": 1, "place2": 12, "place3": 13})
        held = P.parse_fragment((FIXTURES / "5-held-champion.json").read_bytes(), SECOND,
                                sha((FIXTURES / "5-held-champion.json").read_bytes()))
        self.assertEqual(held.eligible_at(ts("2026-12-04T00:00:00Z")), {})        # a held champion's share burns
        ended = fragment(OPENING, rec(15), place2=rec(25, effective_until="2026-11-27T00:00:00Z"))
        parsed = self.parse(ended)
        self.assertIn("place2", parsed.eligible_at(ts("2026-11-26T23:59:59Z")))
        self.assertNotIn("place2", parsed.eligible_at(ts("2026-11-27T00:00:00Z")))

    def test_the_kit_examples_are_accepted_and_pay_as_documented(self):
        names = sorted(p.name for p in FIXTURES.glob("[1-9]-*.json"))
        self.assertEqual(len(names), 9)
        for name in names:
            raw = (FIXTURES / name).read_bytes()
            fragment_ = P.parse_fragment(raw, SECOND, sha(raw))          # every public kit example passes the validator rules
            picks = {s: p.uid for s, p in fragment_.eligible_at(ts("2026-12-05T00:00:00Z")).items()}
            with self.subTest(name):
                if name.startswith("1-"):
                    self.assertEqual(picks, {"champion": 11, "former[0]": 1, "place2": 12, "place3": 13})
                if name.startswith("2-"):        # private crown pending: the beaten champion keeps the title, place 3 burns
                    self.assertEqual(picks, {"champion": 1, "place2": 12})
                if name.startswith("5-"):
                    self.assertEqual(picks, {})
                if name.startswith("9-"):        # private place 2: its share burns, nothing is promoted
                    self.assertEqual(picks, {"champion": 11, "former[0]": 1, "place3": 13})


class Reader(unittest.TestCase):
    def reader(self, net, **kw):
        clock = {"t": 0.0}
        self.clock = clock
        return P.PodiumReader(net, clock=lambda: clock["t"], **kw)

    def test_reads_row_then_file_and_verifies_the_row_sha(self):
        net = Net()
        digest = net.publish(OPENING, fragment(OPENING, rec(15)))
        got = self.reader(net).prefetch()
        self.assertEqual((got.kind, got.window, got.sha256), ("ok", OPENING, digest))
        self.assertEqual(net.calls, [P.COMPETITIONS_URL, P.FRAGMENT_URL.format(window=OPENING)])

    def test_each_failure_is_a_typed_burn_with_a_reason(self):
        for label, prepare, reason in (
            ("404", lambda n: n.files.pop(P.FRAGMENT_URL.format(window=OPENING)), "fragment_not_found"),
            ("sha mismatch", lambda n: n.row(OPENING, "0" * 64), "sha_mismatch"),
            ("unreachable", lambda n: n.files.update({P.FRAGMENT_URL.format(window=OPENING): P.FetchError("unavailable")}),
             "fragment_unavailable"),
            ("row 404", lambda n: n.files.pop(P.COMPETITIONS_URL), "row_not_found"),
            ("row unreachable", lambda n: n.files.update({P.COMPETITIONS_URL: P.FetchError("unavailable")}), "row_unavailable"),
            ("row not json", lambda n: n.files.update({P.COMPETITIONS_URL: b"<html>"}), "row_invalid"),
            ("no generalist row", lambda n: n.files.update({P.COMPETITIONS_URL: b'{"tracks":[{"id":"engine"}]}'}), "row_missing"),
            ("half a row", lambda n: n.files.update({P.COMPETITIONS_URL: json.dumps(
                {"tracks": [{"id": "generalist", "podium_window": OPENING}]}).encode()}), "row_incomplete"),
            ("bad sha in row", lambda n: n.files.update({P.COMPETITIONS_URL: json.dumps(
                {"tracks": [{"id": "generalist", "podium_window": OPENING, "podium_sha256": "ABC"}]}).encode()}), "sha_invalid"),
            ("path-like window", lambda n: n.files.update({P.COMPETITIONS_URL: json.dumps(
                {"tracks": [{"id": "generalist", "podium_window": "../x", "podium_sha256": "0" * 64}]}).encode()}),
             "window_invalid"),
        ):
            net = Net()
            net.publish(OPENING, fragment(OPENING, rec(15)))
            prepare(net)
            got = self.reader(net).prefetch()
            with self.subTest(label):
                self.assertEqual((got.kind, got.reason), ("burn", reason))
                self.assertIsNone(got.fragment)

    def test_a_rerun_with_a_new_sha_is_a_new_fragment_under_the_same_window(self):
        net = Net()
        first = net.publish(OPENING, fragment(OPENING, rec(15), place2=rec(25)))
        reader = self.reader(net)
        self.assertEqual(reader.prefetch().sha256, first)
        reader.prefetch()
        self.assertEqual(len(net.fragment_calls()), 1)                   # same sha: the verified copy is reused
        second = net.publish(OPENING, fragment(OPENING, rec(15), place2=rec(25), place3=rec(35, start="2026-11-27T12:00:00Z")))
        self.assertNotEqual(first, second)
        got = reader.prefetch()
        self.assertEqual((got.window, got.sha256), (OPENING, second))
        self.assertEqual(len(net.fragment_calls()), 2)
        self.assertEqual(set(got.fragment.eligible_at(ts("2026-11-28T00:00:00Z"))), {"champion", "place2", "place3"})
        net.row(OPENING, first)       # the publisher points the row back at the first file: its verified bytes are the row's sha
        self.assertEqual(reader.prefetch().sha256, first)
        self.assertEqual(len(net.fragment_calls()), 2)

    def test_the_row_sha_must_match_the_served_bytes_not_a_cached_other_file(self):
        net = Net()
        net.publish(OPENING, fragment(OPENING, rec(15)))
        reader = self.reader(net)
        self.assertEqual(reader.prefetch().kind, "ok")
        net.files[P.FRAGMENT_URL.format(window=OPENING)] = blob(fragment(OPENING, rec(25)))   # a CDN serves other bytes
        net.row(OPENING, "1" * 64)                                                           # and the row moved on
        got = reader.prefetch()
        self.assertEqual((got.kind, got.reason), ("burn", "sha_mismatch"))

    def test_the_window_cannot_move_back(self):
        net = Net()
        net.publish(SECOND, fragment(SECOND, rec(15)))
        reader = self.reader(net)
        self.assertEqual(reader.prefetch().kind, "ok")
        net.publish(OPENING, fragment(OPENING, rec(15)))                  # a stale row naming an older (valid) window
        got = reader.prefetch()
        self.assertEqual((got.kind, got.reason, got.window), ("burn", "window_regressed", OPENING))
        net.publish(THIRD, fragment(THIRD, rec(15)))
        self.assertEqual(reader.prefetch().kind, "ok")                    # moving forward is fine

    def test_a_row_without_a_window_is_no_window_not_a_burn(self):
        net = Net()
        net.row(None, None)
        got = self.reader(net).prefetch()
        self.assertEqual((got.kind, got.reason), ("no-window", None))

    def test_snapshot_age_and_never_raises(self):
        net = Net()
        net.publish(OPENING, fragment(OPENING, rec(15)))
        reader = self.reader(net)
        self.assertEqual(reader.read().kind, "ok")
        calls = len(net.calls)
        self.clock["t"] = 10.0
        reader.read()
        self.assertEqual(len(net.calls), calls)                           # fresh snapshot: no network
        self.clock["t"] = P.MAX_AGE_S + 1
        reader.read()
        self.assertGreater(len(net.calls), calls)                         # stale snapshot: re-read before use

        def boom(url, limit):
            raise ZeroDivisionError("transport bug")
        got = P.PodiumReader(boom).read()
        self.assertEqual((got.kind, got.reason), ("burn", "internal_error"))

    def test_http_fetch_maps_status_size_and_redirects(self):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/ok":
                    body = b'{"x":1}'
                elif self.path == "/big":
                    body = b"0" * 100
                elif self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/ok")
                    self.end_headers()
                    return
                else:
                    self.send_response(404 if self.path == "/missing" else 500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            os.environ.pop(name, None)
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            self.assertEqual(P.http_fetch(base + "/ok", 1000, allowed_prefix=base), b'{"x":1}')
            for path, kind in (("/missing", "not_found"), ("/boom", "unavailable"), ("/big", "too_large"),
                               ("/redirect", "unavailable")):
                with self.assertRaises(P.FetchError) as caught:
                    P.http_fetch(base + path, 50, allowed_prefix=base)
                self.assertEqual(caught.exception.kind, kind, path)
            with self.assertRaises(P.FetchError):      # only below the allowed prefix (default https://reax.co/)
                P.http_fetch(base + "/ok", 1000)
        finally:
            server.shutdown()
            server.server_close()


class WeightIntegration(unittest.TestCase):
    def setUp(self):
        os.environ.pop("REAX_KA_FEED_PATH", None)
        from reax_subnet import competition_split as C
        patcher = mock.patch.object(C, "NEED_RULE", True)     # the release rule (the test suite's conftest switches it off)
        patcher.start()
        self.addCleanup(patcher.stop)

    def setup_net(self, window, doc):
        net = Net()
        self.digest = net.publish(window, doc)
        return net

    def assertSums(self, vector):
        self.assertAlmostEqual(math.fsum(vector.values()), 1.0, places=12)

    def check_against_staged_calculation(self, vector, eligible_fractions):
        expected, podium = expected_vector(eligible_fractions)
        self.assertEqual(sorted(vector), sorted(expected))
        for uid, value in expected.items():
            self.assertAlmostEqual(vector[uid], value, places=12, msg=f"uid {uid}")
        return podium

    def test_full_opening_podium_70_20_10_of_pool_minus_keepalive(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25), place3=rec(35)))
        h = Harness(net)
        vector = h.weights()
        self.assertSums(vector)
        podium = self.check_against_staged_calculation(vector, payout_fractions(
            OPENING, OPENING_SHARES, {"champion": 15, "place2": 25, "place3": 35}))
        self.assertAlmostEqual(vector[15], MINER_PART * REST * .70)
        self.assertAlmostEqual(vector[25], MINER_PART * REST * .20)
        self.assertAlmostEqual(vector[35], MINER_PART * REST * .10)
        self.assertAlmostEqual(vector[KA_UID], MINER_PART * KEEPALIVE)
        part = h.record()
        self.assertEqual(part["status"], "paid")
        self.assertEqual((part["window"], part["fragment_sha256"], part["source_block"]), (OPENING, self.digest, 1000))
        self.assertEqual(_filled(part["eligible"]), {"champion": 15, "place2": 25, "place3": 35})
        self.assertAlmostEqual(part["pool"], MINER_PART * POOL)
        self.assertAlmostEqual(part["keepalive"], MINER_PART * KEEPALIVE)
        self.assertAlmostEqual(part["burn"], 0.0, places=15)
        for uid, amount in podium.items():
            self.assertAlmostEqual(part["podium"][str(uid)], MINER_PART * amount)

    def test_third_place_vacant_burns_its_own_share(self):
        full = Harness(self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25), place3=rec(35)))).weights()
        h = Harness(self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25))))
        vector = h.weights()
        self.assertSums(vector)
        self.assertNotIn(35, vector)
        self.assertAlmostEqual(vector[15], full[15])                       # no promotion
        self.assertAlmostEqual(vector[25], full[25])
        self.assertAlmostEqual(vector[OWNER_UID] - full[OWNER_UID], MINER_PART * REST * .10)   # the burn
        part = h.record()
        self.assertEqual(_filled(part["eligible"]), {"champion": 15, "place2": 25})
        self.assertAlmostEqual(part["burn"], MINER_PART * REST * .10)
        self.check_against_staged_calculation(vector, payout_fractions(OPENING, OPENING_SHARES, {"champion": 15, "place2": 25}))

    def test_private_unrevealed_second_place_is_absent_and_burns(self):
        doc = fragment(OPENING, rec(15), place3=rec(35))
        doc["podium"]["burned_places"] = doc["burned_places"] = [
            dict(slot="place2", share=.2, reason="not_yet_public")]
        h = Harness(self.setup_net(OPENING, doc))
        vector = h.weights()
        self.assertEqual(sorted(u for u in vector if u not in (OWNER_UID, SERVING_UID, KA_UID)), [15, 35])
        self.assertAlmostEqual(vector[35], MINER_PART * REST * .10)        # place 3 is NOT promoted to place 2
        self.assertAlmostEqual(h.record()["burn"], MINER_PART * REST * .20)
        self.check_against_staged_calculation(vector, payout_fractions(OPENING, OPENING_SHARES, {"champion": 15, "place3": 35}))

    def test_standard_second_window_70_10_5_10_5(self):
        doc = fragment(SECOND, rec(45, start="2026-12-03T15:00:00Z"), former=(rec(15), None), place2=rec(25, start="2026-12-03T15:00:00Z"),
                       place3=rec(35, start="2026-12-03T15:00:00Z"),
                       tenures=[tenure(15, "2026-11-26T15:00:00Z", "2026-12-03T15:00:00Z"), tenure(45, "2026-12-03T15:00:00Z")])
        doc["podium"]["former"][0]["effective_from"] = "2026-12-03T15:00:00Z"
        h = Harness(self.setup_net(SECOND, doc), now=datetime(2026, 12, 4, 9, 0, tzinfo=timezone.utc))
        vector = h.weights()
        self.assertSums(vector)
        slots = {"champion": 45, "former[0]": 15, "place2": 25, "place3": 35}
        self.check_against_staged_calculation(vector, payout_fractions(SECOND, STANDARD_SHARES, slots))
        self.assertAlmostEqual(vector[45], MINER_PART * REST * .70)
        self.assertAlmostEqual(vector[15], MINER_PART * REST * .10)
        self.assertAlmostEqual(vector[25], MINER_PART * REST * .10)
        self.assertAlmostEqual(vector[35], MINER_PART * REST * .05)
        self.assertAlmostEqual(h.record()["burn"], MINER_PART * REST * .05)   # former[1] is empty: its share burns
        self.assertEqual(_filled(h.record()["eligible"]), slots)

    def test_old_table_and_duplicate_uid_burn_the_whole_podium(self):
        for label, doc, reason in (
            ("old table in the opening window", fragment(OPENING, rec(15), place2=rec(25), shares=STANDARD_SHARES), "wrong_shares"),
            ("duplicate uid", fragment(OPENING, rec(15), place2=rec(15)), "duplicate_uid"),
        ):
            h = Harness(self.setup_net(OPENING, doc))
            vector = h.weights()
            with self.subTest(label):
                self.assertSums(vector)
                self.assertTrue(all(u in (OWNER_UID, SERVING_UID, KA_UID) for u in vector), vector)
                part = h.record()
                self.assertEqual(part["status"], f"burned:{reason}")
                self.assertAlmostEqual(part["burn"], MINER_PART * REST)
                self.assertEqual((part["podium"], _filled(part["eligible"]), part["fragment_sha256"]), ({}, {}, None))

    def test_404_sha_mismatch_and_missing_field_burn_and_weights_are_still_set(self):
        good = fragment(OPENING, rec(15), place2=rec(25), place3=rec(35))
        missing = copy.deepcopy(good)
        del missing["podium"]["place2"]["revision"]
        for label, build, reason in (
            ("404", lambda n: n.files.pop(P.FRAGMENT_URL.format(window=OPENING)), "fragment_not_found"),
            ("sha mismatch", lambda n: n.row(OPENING, "f" * 64), "sha_mismatch"),
            ("missing field", lambda n: n.publish(OPENING, missing), "missing_field.podium.place2.revision"),
        ):
            net = self.setup_net(OPENING, good)
            build(net)
            h = Harness(net)
            vector = h.weights()                                              # does not raise
            with self.subTest(label):
                self.assertSums(vector)
                self.assertGreater(vector[SERVING_UID], 0)                    # serving and keep-alive still paid
                self.assertGreater(vector[KA_UID], 0)
                self.assertTrue(all(u in (OWNER_UID, SERVING_UID, KA_UID) for u in vector), vector)
                part = h.record()
                self.assertEqual(part["status"], f"burned:{reason}")
                self.assertEqual(part["window"], OPENING)
                self.assertAlmostEqual(part["burn"], MINER_PART * REST)

    def test_no_stale_podium_when_the_transport_fails_after_a_good_read(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25)))
        h = Harness(net)
        self.assertIn(15, h.weights())
        h.engine.podium_reader._snap = None                                  # force a re-read
        net.files[P.COMPETITIONS_URL] = P.FetchError("unavailable")
        vector = h.weights()
        self.assertNotIn(15, vector)
        self.assertEqual(h.record()["status"], "burned:row_unavailable")

    def test_row_without_window_keeps_the_ordinary_rank_path_exactly(self):
        ranks = [(15, 80.0), (25, 70.0), (35, 60.0)]
        net = Net()
        net.row(None, None)
        on = Harness(net, ranks=ranks)
        off = Harness(None, ranks=ranks)
        self.assertEqual(on.weights(), off.weights())
        self.assertEqual(on.record()["status"], "no-window")
        self.assertEqual((on.record()["window"], on.record()["fragment_sha256"]), (None, None))
        self.assertIsNone(off.record())

    def test_the_podium_replaces_the_ordinary_generalist_ranks(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15)))
        h = Harness(net, ranks=[(99, 90.0), (45, 80.0), (35, 70.0)])
        vector = h.weights()
        self.assertNotIn(99, vector)
        self.assertNotIn(45, vector)
        self.assertNotIn(35, vector)
        self.assertAlmostEqual(vector[15], MINER_PART * REST * .70)

    def test_award_gate_stays_in_force_and_an_unapproved_slot_burns_unpromoted(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25), place3=rec(35)))
        full = Harness(net).weights()
        h = Harness(net, approved=(15, 35, 60))                              # place 2's hotkey/revision is not approved
        vector = h.weights()
        self.assertNotIn(25, vector)
        self.assertAlmostEqual(vector[35], full[35])                         # place 3 stays place 3
        self.assertEqual(_filled(h.record()["eligible"]), {"champion": 15, "place3": 35})
        self.assertAlmostEqual(h.record()["burn"], MINER_PART * REST * .20)
        none = Harness(net, approved=(60,)).weights()
        self.assertTrue(all(u in (OWNER_UID, SERVING_UID, KA_UID) for u in none))

    def test_chain_binding_a_reused_uid_or_a_new_revision_is_not_paid(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15), place2=rec(25), place3=rec(35)))
        moved = snapshot({25: dict(hotkey="another-hotkey"), 35: dict(commitment=commit(35).replace("@0", "@1"))})
        h = Harness(net, approved=(15, 25, 35, 60))
        h.engine.award_gate.ok.add("another-hotkey")
        vector = h.weights(moved)
        self.assertEqual(sorted(u for u in vector if u not in (OWNER_UID, SERVING_UID, KA_UID)), [15])
        self.assertEqual(_filled(h.record()["eligible"]), {"champion": 15})
        gone = h.weights(Snapshot(1000, tuple(n for n in neurons() if n.uid != 15), "owner"))
        self.assertNotIn(15, gone)

    def test_the_owner_uid_is_never_paid_and_its_slot_burns(self):
        doc = fragment(OPENING, rec(OWNER_UID, hotkey="owner"), place2=rec(25),
                       tenures=[dict(tenure(OWNER_UID), hotkey="owner")])
        h = Harness(self.setup_net(OPENING, doc), approved=(25, 60))
        h.engine.award_gate.ok.add("owner")
        snap = Snapshot(1000, tuple([Neuron(OWNER_UID, "owner", "oc", False, 0., None, None, commitment=commit(0))]
                                    + [n for n in neurons() if n.uid != OWNER_UID]), "owner")
        vector = h.weights(snap)
        self.assertEqual(_filled(h.record()["eligible"]), {"place2": 25})
        self.assertAlmostEqual(h.record()["burn"], MINER_PART * REST * .70 + MINER_PART * REST * .10)
        self.assertSums(vector)

    def test_a_paused_generalist_track_runs_the_ordinary_paused_path_and_claims_no_record(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15)))
        h = Harness(net, ranks=[(99, 90.0)])
        from reax_subnet import engvalidator
        real = engvalidator._need.decide_with_starts
        engvalidator._need.decide_with_starts = lambda *a: (.4, {"generalist": AFTER.timestamp() - 86400})
        try:
            vector = h.weights()
        finally:
            engvalidator._need.decide_with_starts = real
        self.assertSums(vector)
        self.assertIsNone(h.record())
        self.assertNotIn(15, vector)

    def test_flag_off_is_byte_identical_to_the_staged_base_and_touches_nothing(self):
        golden = json.loads((FIXTURES / "flag-off-golden.json").read_text())
        self.assertEqual(SC.run_all(), golden)                               # repr of every float, every scenario
        net = Net()
        h = Harness(None, ranks=[(15, 80.0)])
        h.weights()
        self.assertIsNone(h.record())
        self.assertIsNone(h.engine.podium_prefetcher())
        self.assertEqual(net.calls, [])
        self.assertFalse(P.enabled({}))
        self.assertFalse(P.enabled({"REAX_GR2_PODIUM": "0"}))
        self.assertTrue(P.enabled({"REAX_GR2_PODIUM": "1"}))

    def test_flag_on_without_any_podium_input_equals_flag_off_when_there_is_no_window(self):
        net = Net()
        net.row(None, None)
        for ranks in (None, [(15, 80.0), (25, 70.0)]):
            self.assertEqual(Harness(net, ranks=ranks).weights(), Harness(None, ranks=ranks).weights())

    def test_prefetcher_exposes_the_blocking_read(self):
        net = self.setup_net(OPENING, fragment(OPENING, rec(15)))
        h = Harness(net)
        fetch = h.engine.podium_prefetcher()
        self.assertEqual(fetch().kind, "ok")
        self.assertEqual(len(net.fragment_calls()), 1)
        h.weights()
        self.assertEqual(len(net.fragment_calls()), 1)                       # weights() used the prefetched snapshot


class SetBoundRecord(unittest.TestCase):
    PART = dict(window=OPENING, fragment_sha256="a" * 64, source_block=1000, pool=.27, keepalive=.0675,
                eligible={"champion": 15, "place2": 25}, podium={"15": .14175, "25": .0405}, burn=.02025, status="paid")
    VECTOR = {0: .52, 15: .14175, 25: .0405, 251: .24, 60: .0675}

    def test_line_has_exactly_the_agreed_fields_and_types(self):
        line = REC.build(self.PART, 9555702, self.VECTOR)
        self.assertEqual(list(line), ["v", "set_block", "window", "fragment_sha256", "source_block", "pool", "keepalive",
                                      "eligible", "podium", "burn", "status", "vector_sha256", "submitted"])
        self.assertEqual(line["v"], 1)
        self.assertIs(type(line["set_block"]), int)
        self.assertIsInstance(line["window"], str)
        self.assertRegex(line["fragment_sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(type(line["source_block"]), int)
        for key in ("pool", "keepalive", "burn"):
            self.assertIs(type(line[key]), float)
        self.assertEqual(_filled(line["eligible"]), {"champion": 15, "place2": 25})
        self.assertTrue(all(isinstance(k, str) and type(v) is float for k, v in line["podium"].items()))
        self.assertRegex(line["vector_sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(line["submitted"], True)                                   # explicit on every line

    def test_vector_sha256_is_the_canonical_json_of_the_submitted_vector(self):
        canonical = json.dumps({str(u): w for u, w in self.VECTOR.items()}, sort_keys=True, separators=(",", ":"))
        self.assertEqual(canonical, '{"0":0.52,"15":0.14175,"25":0.0405,"251":0.24,"60":0.0675}')
        self.assertEqual(REC.build(self.PART, 1, self.VECTOR)["vector_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        shuffled = dict(reversed(list(self.VECTOR.items())))
        self.assertEqual(REC.vector_sha256(shuffled), REC.vector_sha256(self.VECTOR))
        self.assertNotEqual(REC.vector_sha256({**self.VECTOR, 60: .0676}), REC.vector_sha256(self.VECTOR))

    def test_failed_set_is_marked_and_statuses_follow_the_read(self):
        self.assertIs(REC.build(self.PART, 1, self.VECTOR, submitted=False)["submitted"], False)
        used = dict(window=OPENING, pool=.27, keepalive=.0675, eligible={}, podium={}, burn=.2025)
        self.assertEqual(REC.partial("burn", "sha_mismatch", OPENING, None, 1000, used)["status"], "burned:sha_mismatch")
        self.assertEqual(REC.partial("burn", "row_not_found", None, None, 1000, used)["window"], None)
        none = REC.partial("no-window", None, None, None, 1000, {})
        self.assertEqual((none["status"], none["window"], none["pool"], none["podium"]), ("no-window", None, 0.0, {}))
        self.assertEqual(REC.partial("ok", None, OPENING, "b" * 64, 1000, used)["status"], "burned:no_eligible_slot")
        self.assertEqual(REC.partial("ok", None, OPENING, "b" * 64, 1000, {})["status"], "burned:no_generalist_slot")

    def test_path_is_the_validators_own_state_directory_and_append_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = REC.path_for(os.path.join(tmp, "validator-state.json"))
            self.assertEqual(path, os.path.join(tmp, "gr2-podium-sets.jsonl"))
            self.assertTrue(REC.append(path, REC.build(self.PART, 10, self.VECTOR)))
            self.assertTrue(REC.append(path, REC.build(self.PART, 11, self.VECTOR, submitted=False)))
            lines = Path(path).read_text().splitlines()
            self.assertEqual([json.loads(x)["set_block"] for x in lines], [10, 11])
            self.assertTrue(REC.append(os.path.join(tmp, "missing-dir", "x.jsonl"), {"v": 1}) is False)   # never raises


if __name__ == "__main__":
    unittest.main()


class RecordAllSlots(unittest.TestCase):
    def test_every_slot_of_the_window_table_is_listed_null_when_vacant(self):
        from reax_subnet import gr2_podium, gr2_podium_record as R
        used = {"window": "2026-12-03", "pool": 0.27, "keepalive": 0.0, "eligible": {"champion": 15}, "podium": {15: 0.1}, "burn": 0.0}
        part = R.partial("ok", None, "2026-12-03", "ab" * 32, 1, used)
        self.assertEqual(set(part["eligible"]), set(gr2_podium.STANDARD_SHARES))
        self.assertIsNone(part["eligible"]["former[0]"]); self.assertIsNone(part["eligible"]["former[1]"])
        self.assertEqual(part["eligible"]["champion"], 15)
        opening = R.partial("ok", None, gr2_podium.OPENING_WINDOW, "ab" * 32, 1, dict(used, window=gr2_podium.OPENING_WINDOW))
        self.assertEqual(set(opening["eligible"]), set(gr2_podium.OPENING_SHARES))
