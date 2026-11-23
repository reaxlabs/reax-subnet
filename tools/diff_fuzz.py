"""Differential fuzz of combine_weights: run under PYTHONPATH=<src> with args MODE N SEED; prints JSON of results.
MODE: unset | env_prestart | env_empty | env_set (env_set only meaningful for feed-on runs)."""
import json, math, os, random, sys, logging
logging.disable(logging.CRITICAL)
from datetime import datetime, timezone
import reax_subnet.competition_split as C
assert C.__file__.startswith(sys.argv[4]), C.__file__
mode, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
START = datetime(2026, 11, 21, 15, tzinfo=timezone.utc).timestamp()
D = 86400
if mode == 'unset':
    os.environ.pop('REAX_KA_FEED_PATH', None)
elif mode == 'env_empty':
    os.environ['REAX_KA_FEED_PATH'] = ''
else:
    os.environ['REAX_KA_FEED_PATH'] = '/nonexistent/must-not-read'
NEED = sys.argv[5] == '1'
C.NEED_RULE = NEED
out = []
for i in range(N):
    r = random.Random(seed * 1000003 + i)
    if mode == 'env_prestart':
        now = r.choice([START - 1, START - 0.001, START - 3 * D, START - 30 * D, START - 5 * D + 7])
    else:
        now = r.choice([START - 1, START, START + 1, START + 3 * D, START + 20 * D, START - 10 * D, START + 0.5])
    uids = list(range(0, 60))
    owner = r.choice([0, None, 0, 0])
    serving = {u: r.choice([0.0, r.random(), r.random() * 0.01, 1e-9]) for u in r.sample(uids, r.randint(1, 40))}
    if owner is not None and r.random() < 0.8:
        serving[owner] = r.random()
    if sum(w for w in serving.values() if w > 0) <= 0:
        serving[59] = 1.0
    def rl(n):
        L = C.RankedLeaders([(u, r.choice([r.random() * 100, 0.0, -1.0, 50.0])) for u in r.sample(uids, n)])
        if r.random() < 0.7:
            L.champions = {L[0][0]} if L and r.random() < 0.8 else set()
        for u, _ in L:
            if r.random() < 0.6:
                L.crowned[u] = now - r.random() * 20 * D
        if L and r.random() < 0.2:
            L.cochamps = {L[0][0]: 0.5, r.choice(uids): 0.5}
        if len(L) > 1 and r.random() < 0.2:
            L.formset = {L[1][0]: 0.5, r.choice(uids): 0.5}
        if r.random() < 0.1:
            L.formsets = [{r.choice(uids): 1.0}]
        L.cochamps_dynamic = r.random() < 0.3
        return L
    tids = ['generalist'] + (['df'] if r.random() < 0.3 else []) + (['zz'] if r.random() < 0.2 else [])
    tracks = {}
    for t in tids:
        if r.random() < 0.8:
            tracks[t] = rl(r.randint(0, 5)) if r.random() < 0.8 else {u: r.random() * 100 for u in r.sample(uids, r.randint(0, 4))}
    keepalive = {t: [(u, r.choice([r.random() * 100, 0.0, 1e9])) for u in r.sample(uids, r.randint(0, 15))] for t in tids if r.random() < 0.7}
    provisional = {t: r.sample(uids, r.randint(0, 6)) for t in tids if r.random() < 0.5}
    protected = {t: r.sample(uids, r.randint(0, 4)) for t in tids if r.random() < 0.3} or None
    paused = r.sample(tids + ['engine'], r.randint(0, 2)) if r.random() < 0.5 else None
    pause_starts = {k: now - r.random() * 15 * D for k in (paused or [])} if r.random() < 0.8 else None
    engine_champion = r.choice([None, r.choice(uids)])
    engine_cochampions = {r.choice(uids): r.random() for _ in range(r.randint(0, 3))} if r.random() < 0.3 else None
    engine_runout = {r.choice(uids): (r.random(), r.choice([None, now - r.random() * 10 * D])) for _ in range(r.randint(0, 3))} if r.random() < 0.4 else None
    owner_allocation = {r.choice(uids): r.random() * 0.01 for _ in range(r.randint(0, 3))} if r.random() < 0.3 else None
    approved_set = set(r.sample(uids, 30))
    award_ok = (lambda t, u, s=approved_set: u in s) if r.random() < 0.7 else None
    award_at = (lambda t, u, n=now: n - 2 * D) if r.random() < 0.3 else None
    kwargs = dict(owner_uid=owner, burn_fraction=r.choice([0.0, 0.1, 0.5]), award_ok=award_ok,
                  keepalive=keepalive or None, provisional=provisional or None,
                  engine_waiting=r.sample(uids, r.randint(0, 5)) or None, keep_share=r.random() < 0.3, now=now,
                  engine_cochampions=engine_cochampions, legacy_keepalive=r.sample(uids, r.randint(0, 3)) or None,
                  keepalive_protected=protected, paused=paused, inference=r.choice([None, 0.4, 0.6]),
                  pause_starts=pause_starts, engine_runout=engine_runout, owner_allocation=owner_allocation,
                  award_approved_at=award_at)
    if r.random() < 0.3:
        kwargs['keepalive_share'] = r.choice([0.0, 0.1, 0.5])
    if r.random() < 0.5:
        kwargs['floors'] = {t: r.choice([None, 10.0, 60.0]) for t in tids}
    feed = r.sample(range(100), r.randint(0, 70 if r.random() < 0.2 else 10)) if r.random() < 0.7 else None
    if 'generalist_feed_uids' in C.combine_weights.__code__.co_varnames and mode != 'none':
        kwargs['generalist_feed_uids'] = feed
    try:
        w = C.combine_weights(serving, engine_champion, r.random(), tracks, C.SPLIT_PROFILE, **kwargs)
        out.append([[u, float(x).hex()] for u, x in w.items()])
    except Exception as e:
        out.append(['EXC', type(e).__name__, str(e)[:80]])
print(json.dumps(out))
