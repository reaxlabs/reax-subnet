"""Run once with each PYTHONPATH: actual 107cbf5 source versus merged source."""
import json
import pytest
from unittest.mock import patch
from test_spec_evaluator import Env
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

async def scripted_states(tmp_path):
    env = Env(tmp_path, {})
    assert not getattr(env.cfg, 'v3_decisions', False)
    assert getattr(env.cfg, 'x_tier_mode',
        'off') == 'off'
    states = []
    def checkpoint(name):
        with patch.object(ST.time, "time", env.clock):
            raw = ST._to_json(env.ev.state)
        # New optional defaults are the only permissible schema difference.
        for ts in raw['tracks'].values():
            # Generalist diagnostics are add-only defaults, empty on the legacy specialist path.
            assert ts.pop('item_categories', {}) == {}
            # 1.2.0rc11.post3: the shared slot group fields are add-only defaults, empty without a crown slot
            for field, default in (('cochamps', []), ('cochamps_since', None), ('former_sets', []), ('slot_results', [])):
                assert ts.pop(field, default) == default
            for draw in ts['draws'].values():
                assert draw.pop('item_categories', {}) == {}
                assert draw.pop('x_shortfall', {}) == {}
            for sub in ts['submissions'].values():
                # 1.2.0rc9.post6: the parameter count recorded at the fetch (open-class keep-alive) is not in the 107cbf5 schema
                assert sub.pop('params', None) in (None, 4 * 64 * 64)
                for field, default in [('decision_rule',
        ''), ('close_draws', []), ('confirm_draws', []),
                    ('category_draws', {}), ('category_pooled', {}), ('generalist_budget_checked', False), ('hash_backbone', ''), ('joint_infra_outcomes', 0), ('first_pass_ts', None)]:
                    if field in sub:
                        assert sub[field] == default
                        del sub[field]
                # Joint-layout canonicalization renews stored sketches (rule 4); this cache-version
                # bookkeeping must not change the legacy decisions compared with rule-3 source.
                from reax_subnet.spec import tensors as T
                assert sub['sketch_rule'] == T.SKETCH_RULE == 4
                sub['sketch_rule'] = 3
                # Python 3.12's more accurate float sum changes scores/margins by a
                # few ulps (also with the actual 107cbf5 source). Canonicalize only
                # these numeric fields for hashing, retaining 12 decimal places;
                # all other state, including history structure, draw/accounting
                # data, events, serve logs and timestamps, stays exact.
                for record in [sub, *sub['history']]:
                    for field in ('score', 'delta'):
                        if isinstance(record.get(field), float):
                            record[field] = round(record[field], 12)
        states.append([name, raw])
    env.add(1, 'a/champ', 0.65)
    await env.cycle({1: 'a/champ'})
    checkpoint('first draw vacant')
    env.clock.advance(13 * 3600)
    await env.cycle({1: 'a/champ'})
    checkpoint('crown vacant')
    env.add(2, 'a/weak', 0.3)
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak'})
    checkpoint('reject weak')
    env.add(3, 'a/better', 0.95)
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('pass better')
    env.clock.advance(13 * 3600)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('replace incumbent')
    env.ev = env.make()
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('persist restart idle')
    return states

# Golden states produced by scripted_states with the actual 107cbf5 source (not the merged evaluator).
# Absent v3 defaults and score/delta float precision are canonicalized above;
# the serialization clock is the scenario clock. Regenerate with both source trees.
async def test_flags_off_matches_107cbf5_state_at_every_checkpoint(tmp_path):
    import hashlib
    states = await scripted_states(tmp_path)
    got = [hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest() for _, raw in states]
    assert got == [
        '9b97d83c03872b54d74800350f92cb2d6527210c27c0d64e27fef11d1873a354',
        'b8db12e78a474440c3529c2cb0d549c939479730c27b8033b0a5a207e0e3a2d5',
        '9f63ef6e95bbc974863ba23cf78c894a94ed3ed4c2db059e70304d99b704090a',
        'cf1b5929453f1b375a1f0970a4919ee240203a2b73a7dd78e61bfa986dad7861',
        '0e04c6e49b63d9824f02da400c272064749b8be4dedb7a9a4446c87ae6658032',
        '45f90673928c2abef5076567b70e1750b9c32da234b8bdf464b5dec91420f997',
    ]
