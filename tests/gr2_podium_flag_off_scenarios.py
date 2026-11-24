"""Flag-OFF scenarios for the Round 2 podium: the weight vectors of today's code paths.

Only the pre-podium API is used, so ``python tests/gr2_podium_flag_off_scenarios.py`` prints the same JSON on the staged base
(e99e2b29, post8.16 + the fraction table) and on the podium runtime branch. ``tests/fixtures/gr2-podium/flag-off-golden.json``
was written by the base; ``test_gr2_podium_runtime`` compares every float of the branch with it (``repr``, byte-identical).
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]

from reax_subnet import competition_split as C  # noqa: E402
from reax_subnet.chain import Neuron, Snapshot  # noqa: E402
from reax_subnet.competition_split import RankedLeaders, combine_weights  # noqa: E402
from reax_subnet.engvalidator import EngineCompetition  # noqa: E402

NOW = datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc)
TS = NOW.timestamp()
OWNER = 0
BASE = {0: .1, 251: .6, 252: .2, 253: .1}


def commit(uid):
    return f"reax-spec/1:generalist:lab{uid}/model@{uid:040x}"


def vector(w):
    return {str(u): repr(w[u]) for u in sorted(w)}


class Runner:
    binds_hotkey = True


class Gate:
    def __init__(self, ok):
        self.ok, self.path = set(ok), ""

    def approved(self, track, hotkey, sid):
        return hotkey in self.ok

    def approved_at(self, *args):
        return 0.0


class Specialist:
    """Generalist results / keep-alive entries as the remote relay hands them out."""

    def __init__(self, ranks, alive):
        self.ranks, self.alive = ranks, alive

    def track_results(self, ids, now):
        return {t: self.ranks for t in ids} if self.ranks else {}

    def keepalive(self, ids, now):
        return {t: self.alive for t in ids} if self.alive else {}


def direct():
    out = {}
    ranks = RankedLeaders([(15, 80.0), (25, 70.0), (35, 60.0), (45, 50.0)])
    alive = {"generalist": [(60, 100.0), (61, 100.0)]}
    kw = dict(owner_uid=OWNER, burn_fraction=.1, now=TS, inference=.4)
    out["ranks_gate_all"] = combine_weights(BASE, None, 0., {"generalist": ranks}, award_ok=lambda t, u: True, **kw)
    out["ranks_gate_no_second"] = combine_weights(BASE, None, 0., {"generalist": ranks},
                                                  award_ok=lambda t, u: u != 25, **kw)
    out["ranks_no_gate"] = combine_weights(BASE, None, 0., {"generalist": ranks}, **kw)
    out["keepalive_and_ranks"] = combine_weights(BASE, None, 0., {"generalist": ranks}, keepalive=alive,
                                                 award_ok=lambda t, u: True, **kw)
    out["keepalive_only"] = combine_weights(BASE, None, 0., {}, keepalive=alive, award_ok=lambda t, u: True, **kw)
    out["nothing"] = combine_weights(BASE, None, 0., {}, **kw)
    out["engine_champion"] = combine_weights(BASE, 70, 1.0, {"generalist": ranks}, keepalive=alive,
                                             award_ok=lambda t, u: True, **kw)
    out["need_off_legacy"] = combine_weights(BASE, None, 0., {"generalist": ranks}, owner_uid=OWNER, burn_fraction=.1,
                                             award_ok=lambda t, u: True)
    out["paused_generalist"] = combine_weights(BASE, None, 0., {"generalist": ranks}, keepalive=alive,
                                               award_ok=lambda t, u: True, paused=["generalist"],
                                               pause_starts={"generalist": TS - 5 * 86400}, **kw)
    co = RankedLeaders([(15, 80.0), (25, 70.0), (35, 60.0)])
    co.cochamps = {15: .5, 25: .5}
    out["cochampions"] = combine_weights(BASE, None, 0., {"generalist": co}, award_ok=lambda t, u: True, **kw)
    return {k: vector(v) for k, v in out.items()}


def engine():
    out = {}
    neurons = [Neuron(0, "owner", "oc", False, 0., None, None), Neuron(251, "serving", "sc", False, 0., None, None),
               Neuron(252, "serving2", "sc2", False, 0., None, None)]
    neurons += [Neuron(u, f"hk{u}", f"ck{u}", False, 0., None, None, commitment=commit(u)) for u in (15, 25, 35, 60)]
    snap = Snapshot(1000, tuple(neurons), "owner")
    cases = {
        "ranks_all_approved": (Specialist(RankedLeaders([(15, 80.0), (25, 70.0), (35, 60.0)]), []), ["hk15", "hk25", "hk35"]),
        "ranks_second_unapproved": (Specialist(RankedLeaders([(15, 80.0), (25, 70.0), (35, 60.0)]), []), ["hk15", "hk35"]),
        "ranks_and_keepalive": (Specialist(RankedLeaders([(15, 80.0), (25, 70.0)]), [(60, 100.0)]), ["hk15", "hk25", "hk60"]),
        "keepalive_only": (Specialist(None, [(60, 100.0)]), ["hk60"]),
        "empty": (Specialist(None, []), []),
    }
    for name, (spec, approved) in cases.items():
        eng = EngineCompetition(Runner(), {}, "/nonexistent/engine.json", tracks={}, start_utc=None, specialist=spec,
                                award_gate=Gate(approved), utcnow=lambda: NOW)
        out[name] = vector(eng.weights({0: .1, 251: .6, 252: .3}, snap, set(), .1))
    return out


def run_all():
    os.environ.pop("REAX_KA_FEED_PATH", None)
    return {"direct": direct(), "engine": engine()}


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1, sort_keys=True))
