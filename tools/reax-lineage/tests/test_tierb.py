"""Tier B: residual rotation, norm folding, MLP-neuron and attention-head permutation are undone on full layers."""
import numpy as np
import pytest

from reax_lineage.measure import Thresholds
from reax_lineage.tierb import TierBData, canonicalise, fetch_tierb, frame_of, procrustes
from synth import D, FF, HD, HEADS, MemSource, finetune, make_base

THR = Thresholds()


def layers_of(t, ids=(1, 4)):
    out = {}
    for k, v in t.items():
        p = k.split(".")
        if p[1] == "layers" and int(p[2]) in ids and v.ndim == 2:
            out.setdefault(int(p[2]), {})[".".join(p[3:])] = v.astype(np.float32)
    return out


def rand_orth(d, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def attack(t, Q=None, g=None, mlp_perm=None, head_perm=None, rotate_embed=True):
    """Re-express model t in a rotated / folded / permuted parametrisation that computes the same function."""
    out = {}
    rng = np.random.default_rng(5)
    for k, v in t.items():
        n = k.split(".")
        out[k] = v.copy()
        if k == "model.embed_tokens.weight" and Q is not None:
            out[k] = v @ Q
    for l in sorted({int(k.split(".")[2]) for k in t if k.startswith("model.layers.")}):
        p = "model.layers.%d." % l
        gl = (g(l) if g else np.ones(D, np.float32))
        for nme in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            w = t[p + nme + ".weight"] * gl[None, :]
            out[p + nme + ".weight"] = w @ Q if Q is not None else w
        gate, up, down = (t[p + "mlp." + x + ".weight"] for x in ("gate_proj", "up_proj", "down_proj"))
        gate, up = gate * gl[None, :], up * gl[None, :]
        if Q is not None:
            gate, up = gate @ Q, up @ Q
        o, dn = t[p + "self_attn.o_proj.weight"], down
        if Q is not None:
            o, dn = Q.T @ o, Q.T @ dn
        if mlp_perm is not None:
            pm = mlp_perm(l)
            gate, up, dn = gate[pm], up[pm], dn[:, pm]
        if head_perm is not None:
            hp = head_perm(l)
            q, k_, v_ = (out[p + "self_attn.%s_proj.weight" % x] for x in "qkv")
            out[p + "self_attn.q_proj.weight"] = q.reshape(HEADS, HD, -1)[hp].reshape(q.shape)
            out[p + "self_attn.k_proj.weight"] = k_.reshape(HEADS, HD, -1)[hp].reshape(k_.shape)
            out[p + "self_attn.v_proj.weight"] = v_.reshape(HEADS, HD, -1)[hp].reshape(v_.shape)
            o = o.reshape(o.shape[0], HEADS, HD)[:, hp, :].reshape(o.shape)
        out[p + "mlp.gate_proj.weight"], out[p + "mlp.up_proj.weight"], out[p + "mlp.down_proj.weight"] = gate, up, dn
        out[p + "self_attn.o_proj.weight"] = o
    return out


@pytest.fixture(scope="module")
def models():
    start = make_base(seed=1, layers=6)
    entry = finetune(start, 3, 0.1)
    return start, entry


def tb(t, ids=(1, 4)):
    return TierBData(layers_of(t, ids), t["model.embed_tokens.weight"][:900].astype(np.float32), 6)


def test_honest_entry_needs_no_transform(models):
    start, entry = models
    rep = canonicalise(tb(entry), tb(start), THR, head_dim=HD)
    assert rep["frame_before"]["ok"] and rep["frame_after"]["ok"]
    assert not rep["transform_found"] and not rep["rotation"]["found"]
    assert rep["mlp_perm"]["identity_share_mean"] > 0.99


def test_rotation_and_norm_folding_undone(models):
    start, entry = models
    Q = rand_orth(D, 11)
    g = lambda l: (1.0 + 0.5 * np.random.default_rng(100 + l).standard_normal(D)).astype(np.float32)
    att = attack(entry, Q=Q, g=g)
    before = frame_of(layers_of(att), layers_of(start), THR)
    assert not before["ok"] and before["frame_share"] < 0.2
    rep = canonicalise(tb(att), tb(start), THR, head_dim=HD)
    assert rep["rotation"]["found"] and rep["rotation"]["residual_after"] < 0.3 * rep["rotation"]["residual_before"]
    assert rep["frame_after"]["ok"] and rep["transform_found"] and rep["start_confirmed"]


def test_mlp_permutation_undone(models):
    start, entry = models
    perms = {l: np.random.default_rng(200 + l).permutation(FF) for l in range(6)}
    att = attack(entry, mlp_perm=lambda l: perms[l])
    assert not frame_of(layers_of(att), layers_of(start), THR)["ok"]
    rep = canonicalise(tb(att), tb(start), THR, head_dim=HD)
    assert rep["mlp_perm"]["one_to_one_min"] > 0.99 and rep["mlp_perm"]["identity_share_mean"] < 0.1
    assert rep["frame_after"]["ok"] and rep["transform_found"]


def test_head_permutation_undone(models):
    start, entry = models
    hp = {l: np.random.default_rng(300 + l).permutation(HEADS) for l in range(6)}
    att = attack(entry, head_perm=lambda l: hp[l])
    assert not frame_of(layers_of(att), layers_of(start), THR)["ok"]
    rep = canonicalise(tb(att), tb(start), THR, head_dim=HD)
    assert rep["frame_after"]["ok"] and rep["transform_found"], rep["heads_perm"]


def test_everything_at_once(models):
    start, entry = models
    Q = rand_orth(D, 12)
    perms = {l: np.random.default_rng(400 + l).permutation(FF) for l in range(6)}
    hp = {l: np.random.default_rng(500 + l).permutation(HEADS) for l in range(6)}
    g = lambda l: (1.0 + 0.3 * np.random.default_rng(600 + l).standard_normal(D)).astype(np.float32)
    att = attack(entry, Q=Q, g=g, mlp_perm=lambda l: perms[l], head_perm=lambda l: hp[l])
    rep = canonicalise(tb(att), tb(start), THR, head_dim=HD)
    assert rep["frame_after"]["ok"] and rep["transform_found"]


def test_unrelated_model_is_not_confirmed(models):
    start, _ = models
    other = make_base(seed=77, layers=6)
    rep = canonicalise(tb(other), tb(start), THR, head_dim=HD)
    assert not rep["frame_after"]["ok"] and not rep["start_confirmed"] and not rep["transform_found"]


def test_procrustes_recovers_rotation():
    E = np.random.default_rng(1).standard_normal((500, 64)).astype(np.float32)
    Q = rand_orth(64, 2)
    assert np.allclose(procrustes(E, E @ Q), Q, atol=1e-4)


def test_fetch_tierb_from_a_source(models):
    start, _ = models
    src = MemSource(start, "org/m", "r1")
    d = fetch_tierb(src, "seed", embed_rows=2048)
    assert len(d.meta["layers"]) == 4 and d.embed.shape == (2048, D)
    assert set(d.layers[d.meta["layers"][0]]) >= {"mlp.gate_proj.weight", "self_attn.q_proj.weight"}


# ---------------------------------------------------------------- v1.5.0: canonicalise(consume=True) (F4, memory) ----

def _everything_attack(entry):
    Q = rand_orth(D, 12)
    perms = {l: np.random.default_rng(400 + l).permutation(FF) for l in range(6)}
    hp = {l: np.random.default_rng(500 + l).permutation(HEADS) for l in range(6)}
    g = lambda l: (1.0 + 0.3 * np.random.default_rng(600 + l).standard_normal(D)).astype(np.float32)
    return attack(entry, Q=Q, g=g, mlp_perm=lambda l: perms[l], head_perm=lambda l: hp[l])


@pytest.mark.parametrize("kind", ["honest", "everything"])
def test_consume_gives_the_identical_report_and_transformed_layers(models, kind):
    """consume=True only frees the entry's original tensors; every number of the report and every transformed tensor is bit-identical."""
    start, entry = models
    att = entry if kind == "honest" else _everything_attack(entry)
    plain_in, cons_in = tb(att), tb(att)
    plain = canonicalise(plain_in, tb(start), THR, head_dim=HD)
    cons = canonicalise(cons_in, tb(start), THR, head_dim=HD, consume=True)
    assert plain_in.layers and sorted(plain_in.layers) == [1, 4], "without consume the entry's layers are left alone"
    assert cons_in.layers == {}, "with consume the entry object no longer holds its layers"
    assert {k: v for k, v in plain.items() if k != "layers"} == {k: v for k, v in cons.items() if k != "layers"}
    assert sorted(plain["layers"]) == sorted(cons["layers"])
    for l in plain["layers"]:
        assert sorted(plain["layers"][l]) == sorted(cons["layers"][l])
        for suf in plain["layers"][l]:
            assert np.array_equal(plain["layers"][l][suf], cons["layers"][l][suf]), (l, suf)
    if kind == "everything":
        assert cons["transform_found"] and cons["frame_after"]["ok"]


def test_consume_lowers_the_peak_memory_of_canonicalise():
    """The transformed layers used to be a second copy of the entry's layers (column scaling / permutation replace tensors while `entry.layers` kept the originals):
    on 4 synthetic layers (entry 55 MB) the traced peak of canonicalise drops by at least a third of the entry's size with consume=True."""
    import tracemalloc
    d, ff, hd, nh, NL = 512, 2048, 64, 8, 4
    rng = np.random.default_rng(3)

    def layer():
        r = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
        return {"self_attn.q_proj.weight": r(nh * hd, d), "self_attn.k_proj.weight": r(2 * hd, d), "self_attn.v_proj.weight": r(2 * hd, d), "self_attn.o_proj.weight": r(d, nh * hd),
                "mlp.gate_proj.weight": r(ff, d), "mlp.up_proj.weight": r(ff, d), "mlp.down_proj.weight": r(d, ff)}
    S = {l: layer() for l in range(NL)}
    E = {l: {k: v + (rng.standard_normal(v.shape) * 0.002).astype(np.float32) for k, v in S[l].items()} for l in range(NL)}
    embed = rng.standard_normal((1024, d)).astype(np.float32)
    entry_bytes = sum(v.nbytes for t in E.values() for v in t.values())

    def peak(consume):
        tracemalloc.start()                  # BEFORE the copies: tracemalloc ignores the freeing of blocks that were allocated before it started
        e = TierBData({l: {k: v.copy() for k, v in t.items()} for l, t in E.items()}, embed + 0.001, NL)        # own copies: the originals above must not keep the arrays alive
        s = TierBData({l: {k: v.copy() for k, v in t.items()} for l, t in S.items()}, embed, NL)
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        rep = canonicalise(e, s, THR, head_dim=hd, consume=consume)
        top = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert rep["start_confirmed"]
        return top - base
    plain, cons = peak(False), peak(True)
    assert cons < plain - entry_bytes / 3, (plain, cons, entry_bytes)
