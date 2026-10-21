"""Multi-element commitments (elements.py, docs/DESIGN-MULTIROLE.md)."""

from __future__ import annotations

import pytest

from reax_subnet import elements as E
from reax_subnet.chain import INVALID_FIELD, Neuron, Snapshot, _commitment_fields

M = {m: f"{i:x}" * 16 for i, m in enumerate(E.MODEL_ORDER, start=1)}  # fake manifest ids
FAST = f"reax-s1/1:s1-fast:{M['s1-fast']}"
PRO = f"reax-s1/1:s1-pro:{M['s1-pro']}"
SPEC = "reax-spec/1:demand_forecasting:alice/df-model@" + "a" * 40
SPEC2 = "reax-spec/1:demand_forecasting:alice/df-model@" + "b" * 40
ENG = "reax-eng/1:s1-fast:ghcr.io/alice/engine@sha256:" + "c" * 64


@pytest.fixture
def started(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 100)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 100)


def neuron(uid, fields, block=50, hotkey=None):
    text = "".join(fields)
    return Neuron(uid=uid, hotkey=hotkey or f"hk{uid}", coldkey=f"ck{uid}", validator_permit=False, stake=0.0,
                  ip="1.2.3.4", port=8000 + uid, commitment=text, commitment_block=block,
                  commitment_fields=tuple(fields))


# ------------------------------------------------------------------ inference elements
def test_inference_text_canonical_and_roundtrip():
    one = E.inference_text({"s1-pro": M["s1-pro"]})
    assert one == PRO
    multi = E.inference_text({"s1-lite": M["s1-lite"], "s1-fast": M["s1-fast"], "s1-pro": M["s1-pro"]})
    assert multi == f"reax-s1m/1:s1-fast:{M['s1-fast']},s1-pro:{M['s1-pro']},s1-lite:{M['s1-lite']}"
    assert E.parse_inference(multi) == {"s1-fast": M["s1-fast"], "s1-pro": M["s1-pro"], "s1-lite": M["s1-lite"]}
    four = E.inference_text(M)
    assert len(four.encode()) <= E.MAX_FIELD_BYTES and E.parse_inference(four) == M


@pytest.mark.parametrize("text", [
    f"reax-s1m/1:s1-fast:{M['s1-fast']}",                                  # one model must use reax-s1/1
    f"reax-s1m/1:s1-fast:{M['s1-fast']},s1-fast:{M['s1-fast']}",           # duplicate
    f"reax-s1m/1:s1-fast:{M['s1-fast']},s1-huge:{M['s1-pro']}",            # unknown model
    f"reax-s1m/1:s1-fast:{M['s1-fast']},s1-pro:XYZ",                       # bad manifest id
    f"reax-s1m/1:s1-fast:{M['s1-fast']},,s1-pro:{M['s1-pro']}",            # empty part
    f"reax-s1/1:s1-fast:{M['s1-fast']}:x",
    "reax-s1/1:s1-fast",
    "",
    None,
])
def test_parse_inference_rejects(text):
    assert E.parse_inference(text) is None


def test_inference_text_refuses_unknown_model():
    with pytest.raises(ValueError):
        E.inference_text({"s1-huge": M["s1-fast"]})
    with pytest.raises(ValueError):
        E.inference_text({})


# ------------------------------------------------------------------ commitments
def test_single_field_is_todays_format_even_when_held(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)  # held (the committed default; the release sets a block)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", None)
    assert E.split_elements([FAST], 10**9) == E.Elements(inference=FAST)
    assert E.split_elements([SPEC], 10**9) == E.Elements(competition=SPEC)
    assert E.split_elements([ENG], 10**9) == E.Elements(competition=ENG)
    assert E.split_elements([FAST + "\n"], 1) == E.Elements(inference=FAST)  # stripped, as the old parsers did
    assert E.split_elements(["hello"], 1) is None


def test_multi_element_held_or_before_start(started):
    multi = E.inference_text(M)
    assert E.split_elements([multi], 99) is None
    assert E.split_elements([FAST, SPEC], 99) is None
    assert E.split_elements([multi], 100) == E.Elements(inference=multi)
    assert E.split_elements([FAST, SPEC], 100) == E.Elements(inference=FAST, competition=SPEC)
    assert E.split_elements([SPEC, multi], 100) == E.Elements(inference=multi, competition=SPEC)  # any field order


@pytest.mark.parametrize("fields", [
    [FAST, PRO],            # two inference elements
    [SPEC, ENG],            # two competition elements
    [FAST, SPEC, ENG],      # a third field is never valid today
    [FAST, SPEC, "x"],
    [FAST, ""],
    [FAST, "hello"],
    [FAST, INVALID_FIELD],
    [FAST, "reax-spec/1:demand_forecasting:bad"],  # malformed competition element voids the whole commitment
    [],
])
def test_invalid_combinations(started, fields):
    assert E.split_elements(fields, 200) is None


def test_merge_fields_keeps_the_other_kind():
    assert E.merge_fields([SPEC], FAST) == [FAST, SPEC]
    assert E.merge_fields([FAST, SPEC], PRO) == [PRO, SPEC]
    assert E.merge_fields([FAST, SPEC], SPEC2) == [FAST, SPEC2]
    assert E.merge_fields([FAST], ENG) == [FAST, ENG]
    assert E.merge_fields([FAST, SPEC], SPEC2, drop=E.INFERENCE) == [SPEC2]
    assert E.merge_fields(["junk", SPEC], FAST) == [FAST, SPEC]
    assert E.merge_fields([INVALID_FIELD, SPEC], FAST) == [FAST, SPEC]
    with pytest.raises(ValueError):
        E.merge_fields([FAST], "junk")


def test_merge_fields_never_deletes_what_validators_read_or_a_later_release_wrote():
    """Critic round 1 F1/F2: a whitespace-padded element (validators strip it) and an element of a LATER release
    (unknown reax- prefix, e.g. the generalist track) are kept; two elements of the kind to keep are refused."""
    assert E.merge_fields([" " + SPEC + "\n"], FAST) == [FAST, SPEC]
    gen = "reax-gen/1:generalist:alice/model@" + "d" * 40
    assert E.merge_fields([FAST, gen], PRO) == [PRO, gen]
    assert E.merge_fields([gen], FAST) == [FAST, gen]
    assert E.merge_fields(["reax-s1x/9:future", SPEC], FAST) == [FAST, SPEC]  # a later inference format is replaced
    assert E.merge_fields([FAST, gen], SPEC) == [FAST, SPEC]  # a new competition element replaces the unknown one
    with pytest.raises(ValueError):
        E.merge_fields([SPEC, gen], FAST)


# ------------------------------------------------------------------ slots
def test_slot_keys():
    assert E.slot_key(7, "s1-fast") == 7
    for m in E.MODEL_ORDER:
        s = E.slot_key(255, m)
        assert E.slot_uid(s) == 255 and E.slot_model(s) == m
    assert len({E.slot_key(u, m) for u in range(256) for m in E.MODEL_ORDER}) == 256 * 4


# ------------------------------------------------------------------ chain fields
def test_commitment_fields_from_sdk_record():
    class C:
        fields = [{"Raw34": "0x" + FAST.encode().hex()}, {"Raw81": "0x" + SPEC.encode().hex()}]

    assert _commitment_fields(C()) == (FAST, SPEC)

    class Bad:
        fields = [{"Sha256": "0x" + "00" * 32}, {"Raw2": "0xfffe"}, "x"]

    assert _commitment_fields(Bad()) == (INVALID_FIELD,) * 3
    assert _commitment_fields(None) == ()


# ------------------------------------------------------------------ views
def test_views_of_old_commitments_equal_todays_snapshot():
    snap = Snapshot(500, (neuron(1, [FAST], 10), neuron(2, [SPEC], 20), neuron(3, [], 0), neuron(4, ["junk"], 30)), None)
    v = E.role_views(snap)
    assert [n.commitment for n in v.inference.neurons] == [FAST, "", "", ""]
    assert [n.commitment for n in v.competition.neurons] == ["", SPEC, "", ""]
    assert v.inference.neurons[0].commitment_block == 10 and v.competition.neurons[1].commitment_block == 20
    assert v.elements[4] is None


def test_views_split_a_dual_role_commitment(started):
    multi = E.inference_text(M)
    v = E.role_views(Snapshot(400, (neuron(1, [multi, SPEC], 300),), None))
    assert v.inference.neurons[0].commitment == multi and v.inference.neurons[0].commitment_block == 300
    # the competition view carries the chain block too (deterministic for every validator)
    assert v.competition.neurons[0].commitment == SPEC and v.competition.neurons[0].commitment_block == 300
    # held: the same commitment has no role at all
    v2 = E.role_views(Snapshot(50, (neuron(1, [multi, SPEC], 30),), None))
    assert v2.inference.neurons[0].commitment == "" and v2.competition.neurons[0].commitment == ""


def test_held_multimodel_never_voids_the_competition_element(monkeypatch):
    """Critic round 1 F5: holding or rolling back multimodel must not withdraw the entries next to it."""
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 100)
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)
    multi = E.inference_text(M)
    assert E.split_elements([multi, SPEC], 200) == E.Elements(inference="", competition=SPEC)
    assert E.split_elements([multi], 200) is None  # alone it still has no role
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 300)
    assert E.split_elements([SPEC, multi], 200) == E.Elements(inference="", competition=SPEC)
    assert E.split_elements([SPEC, multi], 300) == E.Elements(inference=multi, competition=SPEC)


def test_views_without_fields_fall_back_to_the_text():
    n = Neuron(uid=1, hotkey="hk", coldkey="ck", validator_permit=False, stake=0.0, ip=None, port=None,
               commitment=FAST, commitment_block=7)
    v = E.role_views(Snapshot(10, (n,), None))
    assert v.inference.neurons[0].commitment == FAST and v.inference.neurons[0].commitment_block == 7
