#!/usr/bin/env python3
"""Multi-role end-to-end on a local subtensor (docs/DESIGN-MULTIROLE.md). ON-CHAIN incentive. LOCAL TEST ONLY.

Prerequisite: a local chain is running (REAX_NETWORK=ws://127.0.0.1:<port>). Hash test model everywhere; every machine is
a throughput-limited fake sidecar of one model (fake_capacity_backend.py, 80 decisions/s). The start constants
(elements.MULTIMODEL_START_BLOCK / DUALROLE_START_BLOCK, held in the release) are set to 0 inside every process this
driver starts, as a release that has started them would.

    mMulti: ONE hotkey serving s1-fast + s1-pro + s1-lite (three machines)
    mFast / mPro / mLite: one model each (one machine each) = the same work on three hotkeys
    mDual: an entrant: its hotkey first commits a use-case element alone, then its miner (s1-fast, one machine) adds
           the inference element next to it; mid-run the driver replaces the use-case element (new revision)

PASS when, for --streak readings in a row after the dual-role rewrite: share(mMulti) = share(mFast) + share(mPro) +
share(mLite) and share(mDual) = share(mFast) (within tolerance), the chain still holds both of mDual's elements, and no
validator logged a role change (score reset) for mDual's or mMulti's uid after the start.

Usage: REAX_NETWORK=ws://127.0.0.1:<port> e2e_multirole.py <work_dir> --host <private ip> [--minutes 40]
The code under test is whatever ``reax_subnet`` the interpreter (PYTHONPATH) imports.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["v1", "v2"]
MINERS = ["mMulti", "mFast", "mPro", "mLite", "mDual"]
CAPACITY = 80.0
TOL_REL, TOL_ABS = 0.12, 0.012
STARTED = ("import reax_subnet.elements as E; E.MULTIMODEL_START_BLOCK = 0; E.DUALROLE_START_BLOCK = 0; "
           "from reax_subnet.cli import {fn}; {fn}()")
SPEC = "reax-spec/1:demand_forecasting:e2e-entrant/df-model@{rev}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--host", required=True, help="a private interface address (the chain refuses loopback axons)")
    ap.add_argument("--minutes", type=float, default=40, help="maximum run time")
    ap.add_argument("--min-minutes", type=float, default=10, help="earliest dual-role rewrite")
    ap.add_argument("--streak", type=int, default=6, help="consecutive passing readings (30 s apart)")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    endpoint = os.environ.get("REAX_NETWORK", "")
    if not endpoint.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("e2e_multirole.py only runs against a local chain")
    import reax_subnet.elements as E
    from reax_subnet.chain import SdkChain

    E.MULTIMODEL_START_BLOCK = E.DUALROLE_START_BLOCK = 0
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *VALIDATORS, *MINERS], check=True,
                   env={**os.environ, "REAX_NETWORK": endpoint})
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}

    def secret(name: str, value: str) -> str:
        p = os.path.join(work, name)
        with open(p, "w") as fh:
            fh.write(value + "\n")
        os.chmod(p, 0o600)
        return p

    token = secret("runtime-token", "e2e-multirole-token")
    base_env = {**os.environ, "REAX_NETWORK": subnet["endpoint"], "REAX_NETUID": str(netuid),
                "REAX_WALLET_PATH": os.path.join(work, "wallets-server"), "REAX_MECHANISM": "s1",
                "REAX_ALLOW_DEV_MODELS": "1", "PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "",
                "REAX_RUNTIME_TOKEN_FILE": token}
    procs: dict[str, subprocess.Popen] = {}

    def spawn(name, cmd, env):
        procs[name] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(work, f"{name}.log"), "a"),
                                       stderr=subprocess.STDOUT)

    def stop(name):
        p = procs.pop(name)
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()

    machines = {"multi-fast": "s1-fast", "multi-pro": "s1-pro", "multi-lite": "s1-lite",
                "fast": "s1-fast", "pro": "s1-pro", "lite": "s1-lite", "dual": "s1-fast"}
    port = {m: 18800 + i for i, m in enumerate(machines)}
    url = {m: f"http://127.0.0.1:{port[m]}" for m in machines}

    def url_env(model: str, machine: str) -> dict[str, str]:
        return {f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL": url[machine]}

    st = bt.Subtensor(network=subnet["endpoint"])
    dual_wallet = bt.Wallet(name="mDual", hotkey="default", path=os.path.join(work, "wallets-full"))
    dual_chain = SdkChain(subnet["endpoint"], netuid, dual_wallet, commitments=True)

    def dual_fields() -> tuple[str, ...]:
        snap = dual_chain.snapshot()
        return next(n.commitment_fields for n in snap.neurons if n.hotkey == dual_wallet.hotkey.ss58_address)

    history: list[dict] = []
    events: list[dict] = []
    verdict, reasons, started = "FAIL", [], time.time()

    def event(what, **kw):
        events.append({"t": round(time.time()), "minutes": round((time.time() - started) / 60, 1), "event": what, **kw})
        print(f"{events[-1]['minutes']:5.1f} min EVENT {what} {kw}", flush=True)

    def validator_logs() -> str:
        return "".join(open(os.path.join(work, f"{v}.log"), encoding="utf-8", errors="replace").read()
                       for v in VALIDATORS)

    try:
        # the entrant's hotkey holds a use-case element alone before its miner starts
        dual_chain.set_commitment_fields([SPEC.format(rev="a" * 40)])
        time.sleep(3)
        event("mDual entrant committed", fields=list(dual_fields()))
        for m, model in machines.items():
            spawn(f"sidecar_{m}", [py, os.path.join(HERE, "fake_capacity_backend.py"), "--port", str(port[m]),
                                   "--capacity", str(CAPACITY), "--token-file", token, "--name", m, "--model", model],
                  base_env)
        time.sleep(3)
        common = {**base_env, "REAX_MINER_HOST": args.host, "REAX_EXTERNAL_IP": args.host,
                  "REAX_MIN_VALIDATOR_STAKE": "10", "REAX_METAGRAPH_REFRESH_S": "30"}
        miner_cmd = [py, "-c", STARTED.format(fn="miner_main")]
        spawn("mMulti", miner_cmd, {**common, "REAX_WALLET_NAME": "mMulti", "REAX_MINER_PORT": "18591",
                                    "REAX_MODELS": "s1-fast,s1-pro,s1-lite", **url_env("s1-fast", "multi-fast"),
                                    **url_env("s1-pro", "multi-pro"), **url_env("s1-lite", "multi-lite")})
        for i, (role, model, machine) in enumerate((("mFast", "s1-fast", "fast"), ("mPro", "s1-pro", "pro"),
                                                    ("mLite", "s1-lite", "lite"), ("mDual", "s1-fast", "dual"))):
            spawn(role, miner_cmd, {**common, "REAX_WALLET_NAME": role, "REAX_MINER_PORT": str(18592 + i),
                                    "REAX_MODEL": model, **url_env(model, machine)})
        time.sleep(45)
        fields = dual_fields()
        event("mDual after its miner started", fields=list(fields))
        if len(fields) != 2 or not any(f.startswith("reax-spec/1:") for f in fields):
            reasons.append(f"mDual's miner did not keep the use-case element: {list(fields)}")
        for i, v in enumerate(VALIDATORS):
            spawn(v, [py, "-c", STARTED.format(fn="validator_main")],
                  {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_BURN_FRACTION": "0.0",
                   "REAX_ROUND_INTERVAL_S": "30", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
                   "REAX_WARMUP_ROUNDS": "2", "REAX_DEV_PROBE_EVERY_ROUNDS": "2", "REAX_LOG_JSON": "1",
                   "REAX_S1_FAST_RUNTIME_URL": "hash", "REAX_S1_PRO_RUNTIME_URL": "hash",
                   "REAX_S1_LITE_RUNTIME_URL": "hash", "REAX_S1_CPU_RUNTIME_URL": "hash",
                   "REAX_FEATURE_ENGINE_COMPETITION": "0",
                   "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json")})
            if i == 0:
                time.sleep(15)
        event("validators started")

        def shares() -> dict[str, float]:
            mg = st.subnets.metagraph(netuid)
            by_uid = {n.uid: float(n.incentive) for n in mg.neurons}
            inc = {r: by_uid.get(uid_of[r], 0.0) for r in MINERS}
            total = sum(inc.values())
            return {r: v / total for r, v in inc.items()} if total > 0 else {}

        def close(a: float, b: float) -> bool:
            return abs(a - b) <= max(TOL_ABS, TOL_REL * max(a, b))

        phase, streak, log_mark = "settle", 0, 0
        while time.time() - started < args.minutes * 60:
            time.sleep(30)
            for name, p in list(procs.items()):
                if p.poll() is not None:
                    print(f"process {name} exited with {p.returncode}", flush=True)
            try:
                s = shares()
            except Exception as exc:  # noqa: BLE001
                print("metagraph read failed:", exc, flush=True)
                continue
            if not s:
                continue
            singles = s["mFast"] + s["mPro"] + s["mLite"]
            ok = close(s["mMulti"], singles) and close(s["mDual"], s["mFast"]) and min(s.values()) > 0
            minutes = (time.time() - started) / 60
            history.append({"t": round(time.time()), "minutes": round(minutes, 1), "phase": phase, "ok": ok,
                            "share": {r: round(v, 4) for r, v in s.items()}, "singles_sum": round(singles, 4)})
            print(f"{minutes:5.1f} min [{phase}] ok={ok} multi={s['mMulti']:.3f} vs singles={singles:.3f} "
                  f"dual={s['mDual']:.3f} vs fast={s['mFast']:.3f} " + " ".join(f"{r}={v:.3f}" for r, v in s.items()),
                  flush=True)
            streak = streak + 1 if ok else 0
            if phase == "settle" and streak >= 3 and minutes >= args.min_minutes:
                phase, streak = "rewrite", 0
                from reax_subnet.elements import merge_fields

                log_mark = len(validator_logs())
                for attempt in range(5):  # the miner signs with the same hotkey: a fresh client gets a fresh nonce
                    try:
                        SdkChain(subnet["endpoint"], netuid, dual_wallet, commitments=True).set_commitment_fields(
                            merge_fields(list(dual_fields()), SPEC.format(rev="b" * 40)))
                        break
                    except RuntimeError as exc:
                        print("rewrite attempt failed:", exc, flush=True)
                        time.sleep(6)
                time.sleep(3)
                event("mDual replaced its use-case element", fields=list(dual_fields()))
                phase = "after"
                continue
            if phase == "after" and streak >= args.streak:
                verdict = "PASS"
                break
        logs = validator_logs()
        tail = logs[log_mark:]
        resets = [line for line in tail.splitlines() if "changed its committed role" in line
                  and any(f"uid {uid_of[r]} " in line for r in ("mDual", "mMulti"))]
        fields = dual_fields()
        if verdict == "PASS":
            if resets:
                verdict, reasons = "FAIL", reasons + [f"score reset after the rewrite: {resets[:3]}"]
            if len(fields) != 2 or SPEC.format(rev="b" * 40) not in fields:
                verdict, reasons = "FAIL", reasons + [f"mDual's elements after the run: {list(fields)}"]
        elif phase != "after":
            reasons.append(f"run ended in phase {phase}")
        else:
            reasons.append("the shares did not hold after the rewrite")
    finally:
        for name in list(procs):
            stop(name)
    ema = {}
    for v in VALIDATORS:
        try:
            state = json.load(open(os.path.join(work, f"{v}-state.json")))
            ema[v] = {k: {"role": m.get("role"), "ema": round(m.get("ema", 0.0), 4), "capacity": m.get("capacity")}
                      for k, m in state.get("miners", {}).items() if m.get("role")}
        except (OSError, ValueError) as exc:
            ema[v] = str(exc)
    json.dump({"verdict": verdict, "reasons": reasons, "tolerance": {"relative": TOL_REL, "absolute": TOL_ABS},
               "machine_dps": CAPACITY, "uids": uid_of, "events": events, "validator_slots": ema,
               "history": history}, open(os.path.join(work, "E2E-MULTIROLE-RESULT.json"), "w"), indent=1)
    print("VERDICT", verdict, reasons)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
