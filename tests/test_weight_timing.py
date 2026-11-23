"""Submission timing changes, never payment eligibility or drand encryption."""
from dataclasses import replace
from types import SimpleNamespace
import asyncio
import threading
import pytest
from reax_subnet import elements, weight_timing as T, validator as V
from reax_subnet.chain import Neuron, Snapshot, SdkChain
from reax_subnet.config import load_validator
from reax_subnet.s1validator import S1Validator
from reax_subnet.s1backends import S1HashBackend

LAST, COMMIT = 9385779, 9385856
TEXT = 'reax-spec/1:generalist:jacksonchen1103/sn92-gen-mg@96e90931cb461156e8d2d3984655a750d3bf396e'


def snapshot(block=COMMIT, **changes):
    me = Neuron(3, 'validator', 'vc', True, 100., None, None, last_update=LAST)
    owner = Neuron(0, 'owner', 'oc', False, 0., None, None)
    n = Neuron(109, 'entrant', 'ec', False, 0., None, None, commitment=TEXT,
               commitment_block=COMMIT, registered_block=9378063, incentive=0.)
    return Snapshot(block, (owner, me, replace(n, **changes)), 'owner')


def candidates(snap=None, weights=None, submitted_block=0, submitted_payees=None):
    return T.new_exposed_payees(snap or snapshot(), weights or {0: .999, 109: .001},
                                'validator', submitted_block, submitted_payees or set())


def test_new_exposed_positive_u16_only(monkeypatch):
    assert candidates() == [109]
    for kw in [dict(incentive=None), dict(incentive=.0001), dict(incentive=float('nan')),
               dict(registered_block=0), dict(registered_block=COMMIT-5200),
               dict(commitment_block=LAST), dict(commitment_block=COMMIT+1),
               dict(commitment='broken'), dict(commitment_fields=(TEXT, 'broken'))]:
        assert candidates(snapshot(**kw)) == [], kw
    assert candidates(weights={0: 1., 109: 1e-9}) == []
    assert candidates(weights={0: 1.}) == []
    assert candidates(submitted_block=COMMIT) == []
    assert candidates(submitted_payees={(109, 'entrant')}) == []
    assert candidates(submitted_payees={(109, 'old-hotkey')}) == [109]
    assert candidates(replace(snapshot(), neurons=tuple(replace(n, last_update=0) for n in snapshot().neurons))) == []
    monkeypatch.setattr(T, 'KA_NEWCOMMIT_SET', False)
    assert candidates() == []


@pytest.mark.parametrize('remaining, expected', [
    (600, [109]), (2000, []), (1500, []), (1499, [109]), (0, [109]), (-1, [109]),
])
def test_immunity_lead_window(remaining, expected):
    registered = COMMIT + remaining - T.CHAIN_IMMUNITY_BLOCKS
    assert candidates(snapshot(registered_block=registered)) == expected


def test_dual_role_keeps_competition_and_rejects_invalid_fields():
    inf = 'reax-s1/1:s1-fast:' + 'a'*16
    assert candidates(elements.role_views(snapshot(commitment_fields=(inf, TEXT))).competition) == [109]
    assert candidates(elements.role_views(snapshot(commitment_fields=(TEXT, TEXT))).competition) == []


@pytest.fixture
def arena(tmp_path, fake_chain_cls):
    chain = fake_chain_cls(list(snapshot().neurons), owner_hotkey='owner', rate_limit=100, since_update=100)
    cfg = load_validator({'REAX_NETWORK': 'ws://127.0.0.1:9951', 'REAX_NETUID': '92',
                          'REAX_ALLOW_DEV_MODELS': '1', 'REAX_STATE_PATH': str(tmp_path/'state'),
                          'REAX_WARMUP_ROUNDS': '0'})
    val = S1Validator(cfg, SimpleNamespace(hotkey=SimpleNamespace(ss58_address='validator')), chain,
                      {'s1-fast': S1HashBackend('s1-fast')}, shares={'s1-fast': 1.})
    val.compute_weights = lambda snap: {0: .999, 109: .001}
    return chain, val


@pytest.mark.parametrize('remaining, expected', [(600, True), (2000, False)])
async def test_immune_entry_early_set(arena, remaining, expected):
    chain, val = arena
    block = LAST + 100
    snap = snapshot(block, registered_block=block + remaining - T.CHAIN_IMMUNITY_BLOCKS)
    chain.since_update = 100
    assert await val.maybe_set_weights(snap) is expected
    assert len(chain.set_calls) == int(expected)
    if expected:
        assert 109 in chain.set_calls[0][0]


async def test_uid109_replay(arena, monkeypatch):
    chain, val = arena
    from reax_subnet import competition_split as CS
    from reax_subnet.spec import evaluator as E, state as ST
    # Replay the actual unseen-entry eligibility and share allocation, too.
    def real_weights(snap):
        provisional = E.compute_provisional(ST.SpecState(), ['generalist'], snap, now=1793234222.)
        assert provisional == {'generalist': [109]}
        return CS.combine_weights({10: .6, 11: .4}, None, 0., {}, owner_uid=0, provisional=provisional)
    val.compute_weights = real_weights
    chain.since_update = 99
    assert not await val.maybe_set_weights(snapshot(LAST+99))
    chain.since_update = 100
    monkeypatch.setattr(T, 'KA_NEWCOMMIT_SET', False)
    assert not await val.maybe_set_weights(snapshot(LAST+100))
    monkeypatch.setattr(T, 'KA_NEWCOMMIT_SET', True)
    assert await val.maybe_set_weights(snapshot(LAST+100))
    assert 109 in chain.set_calls[0][0]
    assert chain.set_calls[0][2] == V.WEIGHTS_VERSION_KEY
    # Stable measured 360-block stateful schedule (phase 48); standard SDK's
    # reveal-boundary +3-block pulse misses that consensus run, counts at +360.
    def counts(inclusion):
        return 9385608 + ((inclusion-9385608)//360+2)*360
    old_inclusion, new_inclusion = LAST+360+1, LAST+100+1
    assert counts(old_inclusion) == 9386688
    assert counts(new_inclusion) == 9386328
    print(f'uid109: inclusion {new_inclusion}, first count {counts(new_inclusion)}; '
          f'old inclusion {old_inclusion}, first count {counts(old_inclusion)}; saved 360 blocks (72 min)')
    assert not await val.maybe_set_weights(snapshot(LAST+101))
    assert not await val.maybe_set_weights(snapshot(COMMIT+200, commitment_block=COMMIT+10))
    assert len(chain.set_calls) == 1


async def test_retry_normal_cadence_and_restart(arena, monkeypatch):
    chain, val = arena
    chain.fail_set_weights = True
    assert not await val.maybe_set_weights(snapshot(LAST+100))
    chain.fail_set_weights = False
    assert await val.maybe_set_weights(snapshot(LAST+101))
    val._submitted_block = 0
    val._submitted_payees.clear()
    snap = snapshot(LAST+200)
    snap = replace(snap, neurons=tuple(replace(n, last_update=LAST+102) if n.hotkey=='validator' else n
                                      for n in snap.neurons))
    assert not await val.maybe_set_weights(snap)
    chain.since_update = 360
    assert await val.maybe_set_weights(snap)
    monkeypatch.setattr(T, 'KA_NEWCOMMIT_SET', False)
    assert await val.maybe_set_weights(snap)


async def test_pending_and_late_success_consume_trigger(arena, monkeypatch):
    chain, val = arena
    gate = threading.Event()
    real = chain.set_weights
    def delayed(*args):
        gate.wait(2)
        return real(*args)
    monkeypatch.setattr(chain, 'set_weights', delayed)
    monkeypatch.setattr(V, 'SET_WEIGHTS_TIMEOUT_S', .01)
    try:
        assert not await val.maybe_set_weights(snapshot(LAST+100))
        assert not await val.maybe_set_weights(snapshot(LAST+101))
    finally:
        gate.set()
    await val._weights_task
    await asyncio.sleep(0)
    assert not await val.maybe_set_weights(snapshot(LAST+102))
    assert len(chain.set_calls) == 1


async def test_no_early_set_without_new_payee(arena):
    chain, val = arena
    val.compute_weights = lambda snap: {0: 1.}
    assert not await val.maybe_set_weights(snapshot(LAST+100))
    chain.since_update = 360
    assert await val.maybe_set_weights(snapshot(LAST+360))


def test_sdk_snapshot_carries_pinned_incentive_and_last_update():
    n = SimpleNamespace(uid=109, hotkey='entrant', coldkey='ec', validator_permit=False,
                        total_stake=SimpleNamespace(amount=1.), axon=None, incentive=0.,
                        last_update=LAST, block_at_registration=9378063)
    c = SdkChain.__new__(SdkChain)
    c.netuid, c.read_commitments = 92, True
    c.subtensor = SimpleNamespace(subnets=SimpleNamespace(metagraph=lambda *a, **kw:
                                  SimpleNamespace(neurons=[n], block=COMMIT, owner_hotkey='owner')))
    snap = c.snapshot()
    assert snap.neurons[0].incentive == 0.
    assert snap.neurons[0].last_update == LAST
