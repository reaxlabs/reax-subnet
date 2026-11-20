#!/usr/bin/env python3
"""Tier B on 4 REAL full layers (+16,384 embedding rows) of gemma-4-31B-it (start) and a real fine-tune (entry), honest and attacked.
Usage: tierb_real.py honest|attack <out.json>"""
import json, os, resource, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from reax_lineage.hf import HFClient, HFSource
from reax_lineage.measure import Thresholds
from reax_lineage.tierb import TierBData, canonicalise, fetch_tierb, frame_of
mode, outf = sys.argv[1], sys.argv[2]
thr = Thresholds.load()
man = {m["repo"]: m for m in json.load(open(os.path.join(HERE, "c1_manifest.json"))) if m.get("ok")}
c = HFClient()
S_repo, E_repo = "google/gemma-4-31B-it", "tattrongvu/Jeffan-31b-1710"
t0 = time.time()
E = fetch_tierb(HFSource(E_repo, man[E_repo]["rev"], c), "A0")
S = fetch_tierb(HFSource(S_repo, man[S_repo]["rev"], c), "A0", layers=E.meta["layers"])
fetch_s = time.time() - t0
cfg = json.loads(HFSource(S_repo, man[S_repo]["rev"], c).read_file("config.json"))
tc = cfg.get("text_config", cfg)
hd, nh = tc.get("head_dim"), tc.get("num_attention_heads")
d = E.embed.shape[1]
print("layers", E.meta["layers"], "hidden", d, "head_dim", hd, "heads", nh, "fetched in %.0fs (%d requests, %.1f GB)" % (fetch_s, c.stats["requests"], c.stats["bytes"] / 1e9), flush=True)
rep_extra = {}
if mode == "attack":
    t1 = time.time()
    rng = np.random.default_rng(11)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, Rr = np.linalg.qr(G)
    Q = (Q * np.sign(np.diag(Rr))).astype(np.float32)
    del G
    E.embed = E.embed @ Q
    for l, tens in E.layers.items():
        gl = {"attn": (1.0 + 0.3 * rng.standard_normal(d)).astype(np.float32), "mlp": (1.0 + 0.3 * rng.standard_normal(d)).astype(np.float32)}
        for suf in list(tens):
            w = tens[suf]
            if suf.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")) and w.shape[1] == d:
                w = w * gl["attn" if "attn" in suf else "mlp"][None, :]
                tens[suf] = (w @ Q).astype(np.float32)
            elif suf.endswith(("o_proj.weight", "down_proj.weight")) and w.shape[0] == d:
                tens[suf] = (Q.T @ w).astype(np.float32)
        gate, up, dn = (tens.get("mlp." + x + ".weight") for x in ("gate_proj", "up_proj", "down_proj"))
        pm = rng.permutation(gate.shape[0])
        tens["mlp.gate_proj.weight"], tens["mlp.up_proj.weight"], tens["mlp.down_proj.weight"] = gate[pm], up[pm], dn[:, pm]
        q, o = tens.get("self_attn.q_proj.weight"), tens.get("self_attn.o_proj.weight")
        if q is not None and hd and nh and q.shape[0] == nh * hd:
            hp = rng.permutation(nh)
            tens["self_attn.q_proj.weight"] = q.reshape(nh, hd, -1)[hp].reshape(q.shape)
            tens["self_attn.o_proj.weight"] = o.reshape(o.shape[0], nh, hd)[:, hp, :].reshape(o.shape)
            for nm in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                w = tens[nm]
                nk = w.shape[0] // hd
                kp = rng.permutation(nk)
                tens[nm] = w.reshape(nk, hd, -1)[kp].reshape(w.shape)
    rep_extra["attack_seconds"] = round(time.time() - t1, 1)
    print("attack built in %.0fs" % (time.time() - t1), flush=True)
t2 = time.time()
rep = canonicalise(E, S, thr, head_dim=hd)
rep.pop("layers", None)
rep["canonicalise_seconds"] = round(time.time() - t2, 1)
rep["fetch_seconds"] = round(fetch_s, 1)
rep["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
rep["mode"], rep["layers_used"], rep["entry"], rep["start"] = mode, E.meta["layers"], E_repo, S_repo
rep.update(rep_extra)
json.dump(rep, open(outf, "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("mode", "frame_before", "frame_after", "rotation", "col_scale", "mlp_perm", "heads_perm", "transform_found", "start_confirmed", "canonicalise_seconds", "peak_rss_gb")}, indent=1))
