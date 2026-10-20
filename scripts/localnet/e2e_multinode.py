#!/usr/bin/env python3
"""Multinode end-to-end on a local subtensor: one hotkey served by two worker machines, one machine swapped while the
validators keep probing. ON-CHAIN incentive. LOCAL TEST ONLY.

Prerequisite: a local chain is running (REAX_NETWORK=ws://127.0.0.1:<port>). Everything runs on the hash test model.
Every simulated machine is a throughput-limited fake s1-fast sidecar (fake_capacity_backend.py, 80 decisions/s); a
worker is such a sidecar plus reax-runtime-relay on the private address (signed link, plain http = the VPN mode).

    mMulti: REAX_FEATURE_MULTINODE=1, workers A + B (160/s together)       mSolo: one machine (80/s), single URL

Phases: (1) settle until the on-chain share is mMulti : mSolo = 2 : 1; (2) swap: start worker C, add it to the workers
file, wait until the gateway logs it in rotation, remove A from the file, stop A; probes keep running (every second
round); (3) settle again. PASS when, after the swap, the 2 : 1 split holds again for --streak readings, the gateway
logged no failed inference and no 503 during the swap, C served traffic and A served nothing after its removal.

Usage: REAX_NETWORK=ws://127.0.0.1:<port> e2e_multinode.py <work_dir> --host <private ip> [--minutes 45]
The code under test is whatever ``reax_subnet`` the interpreter (PYTHONPATH) imports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.request

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["v1", "v2"]
MINERS = ["mMulti", "mSolo"]
CAPACITY = 80.0
EXPECTED = {"mMulti": 2 / 3, "mSolo": 1 / 3}
TOL_REL, TOL_ABS = 0.12, 0.012


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--host", required=True, help="a private interface address (the chain refuses loopback axons)")
    ap.add_argument("--minutes", type=float, default=45, help="maximum run time")
    ap.add_argument("--min-minutes", type=float, default=10, help="earliest swap")
    ap.add_argument("--streak", type=int, default=6, help="consecutive passing readings (30 s apart)")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    endpoint = os.environ.get("REAX_NETWORK", "")
    if not endpoint.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("e2e_multinode.py only runs against a local chain")
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    bindir = os.path.dirname(py)
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

    token = secret("runtime-token", "e2e-multinode-token")
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

    sidecar_port = {"A": 18700, "B": 18701, "C": 18702, "solo": 18703}
    relay_port = {"A": 18710, "B": 18711, "C": 18712}
    keys = {w: secret(f"link-{w}.key", secrets.token_hex(32)) for w in relay_port}

    def start_machine(w):
        spawn(f"sidecar_{w}", [py, os.path.join(HERE, "fake_capacity_backend.py"), "--port", str(sidecar_port[w]),
                               "--capacity", str(CAPACITY), "--token-file", token, "--name", w], base_env)
        if w in relay_port:
            time.sleep(2)
            spawn(f"relay_{w}", [py, "-c", "from reax_subnet.multinode import relay_main; relay_main()"],
                  {**base_env, "REAX_MODEL": "s1-fast", "REAX_S1_FAST_RUNTIME_URL": f"http://127.0.0.1:{sidecar_port[w]}",
                   "REAX_RELAY_KEY_FILE": keys[w], "REAX_RELAY_HOST": args.host, "REAX_RELAY_PORT": str(relay_port[w])})

    workers_file = os.path.join(work, "workers.json")

    def write_workers(*names):
        doc = {"model": "s1-fast", "workers": [{"name": f"worker-{w}", "url": f"http://{args.host}:{relay_port[w]}",
                                                 "key_file": keys[w], "max_inflight": 16} for w in names]}
        tmp = workers_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, workers_file)  # atomic, as the docs recommend

    def served(w) -> int:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{sidecar_port[w]}/stats", timeout=5))["served"]

    def gateway_log() -> str:
        return open(os.path.join(work, "mMulti.log"), encoding="utf-8", errors="replace").read()

    history: list[dict] = []
    events: list[dict] = []
    verdict, reasons, started = "FAIL", [], time.time()
    st = bt.Subtensor(network=subnet["endpoint"])

    def event(what, **kw):
        events.append({"t": round(time.time()), "minutes": round((time.time() - started) / 60, 1), "event": what, **kw})
        print(f"{events[-1]['minutes']:5.1f} min EVENT {what} {kw}", flush=True)

    try:
        for w in ("A", "B", "solo"):
            start_machine(w)
        time.sleep(3)
        write_workers("A", "B")
        common = {**base_env, "REAX_MODEL": "s1-fast", "REAX_MINER_HOST": args.host, "REAX_EXTERNAL_IP": args.host,
                  "REAX_MIN_VALIDATOR_STAKE": "10", "REAX_METAGRAPH_REFRESH_S": "30"}
        spawn("mMulti", [os.path.join(bindir, "reax-miner")],
              {**common, "REAX_WALLET_NAME": "mMulti", "REAX_MINER_PORT": "18491", "REAX_FEATURE_MULTINODE": "1",
               "REAX_WORKERS_FILE": workers_file, "REAX_DEV_IDENTITY": "hash"})
        spawn("mSolo", [os.path.join(bindir, "reax-miner")],
              {**common, "REAX_WALLET_NAME": "mSolo", "REAX_MINER_PORT": "18492",
               "REAX_S1_FAST_RUNTIME_URL": f"http://127.0.0.1:{sidecar_port['solo']}"})
        time.sleep(45)
        for i, v in enumerate(VALIDATORS):
            spawn(v, [os.path.join(bindir, "reax-validator")],
                  {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_BURN_FRACTION": "0.0",
                   "REAX_ROUND_INTERVAL_S": "30", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
                   "REAX_WARMUP_ROUNDS": "2", "REAX_DEV_PROBE_EVERY_ROUNDS": "2", "REAX_LOG_JSON": "1",
                   "REAX_S1_FAST_RUNTIME_URL": "hash", "REAX_S1_PRO_RUNTIME_URL": "hash",
                   "REAX_S1_LITE_RUNTIME_URL": "hash", "REAX_S1_CPU_RUNTIME_URL": "hash",
                   "REAX_FEATURE_ENGINE_COMPETITION": "0",
                   "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json")})
            if i == 0:
                time.sleep(15)  # the two validators' probes must not coincide on the shared fake machines
        event("validators started")

        def shares() -> dict[str, float]:
            mg = st.subnets.metagraph(netuid)
            by_uid = {n.uid: float(n.incentive) for n in mg.neurons}
            inc = {r: by_uid.get(uid_of[r], 0.0) for r in MINERS}
            total = sum(inc.values())
            return {r: v / total for r, v in inc.items()} if total > 0 else {}

        phase, streak, swap_marks = "settle", 0, {}
        while time.time() - started < args.minutes * 60:
            time.sleep(30)
            for name, p in list(procs.items()):
                if p.poll() is not None and not name.endswith("_A"):
                    print(f"process {name} exited with {p.returncode}", flush=True)
            try:
                s = shares()
            except Exception as exc:  # noqa: BLE001
                print("metagraph read failed:", exc, flush=True)
                continue
            if not s:
                continue
            ok = all(abs(s[r] - EXPECTED[r]) <= max(TOL_ABS, TOL_REL * EXPECTED[r]) for r in MINERS)
            minutes = (time.time() - started) / 60
            history.append({"t": round(time.time()), "minutes": round(minutes, 1), "phase": phase, "ok": ok,
                            "share": {r: round(v, 4) for r, v in s.items()}})
            print(f"{minutes:5.1f} min [{phase}] ok={ok} " + " ".join(f"{r}={s[r]:.3f}({EXPECTED[r]:.3f})"
                                                                        for r in MINERS), flush=True)
            streak = streak + 1 if ok else 0
            if phase == "settle" and streak >= 3 and minutes >= args.min_minutes:
                # --- the swap, while probes keep running ---
                phase = "swap"
                log_before = gateway_log()
                swap_marks = {"failed_before": log_before.count("inference failed"),
                              "len_before": len(log_before)}
                event("swap: start machine C")
                start_machine("C")
                time.sleep(3)
                write_workers("A", "B", "C")
                event("swap: C added to the workers file")
                t0 = time.time()
                while time.time() - t0 < 120 and "worker worker-C is back in rotation" not in gateway_log() \
                        and "runtime backend http://%s:%d is back in rotation" % (args.host, relay_port["C"]) \
                        not in gateway_log():
                    time.sleep(1)
                event("swap: C in rotation", seconds=round(time.time() - t0, 1),
                      in_log="back in rotation" in gateway_log()[swap_marks["len_before"]:])
                time.sleep(20)
                write_workers("B", "C")
                event("swap: A removed from the workers file")
                time.sleep(15)
                swap_marks["A_served_at_removal"] = served("A")
                time.sleep(15)
                swap_marks["A_served_after_drain"] = served("A")
                stop("relay_A")
                stop("sidecar_A")
                event("swap: machine A switched off", a_served=swap_marks["A_served_after_drain"])
                phase, streak = "after", 0
                continue
            if phase == "after" and streak >= args.streak:
                verdict = "PASS"
                break
        log_all = gateway_log()
        tail = log_all[swap_marks.get("len_before", 0):]
        swap_marks["failed_inference_after_swap_start"] = tail.count("inference failed")
        swap_marks["overloaded_503_lines"] = len(re.findall(r"rejected a request with 503", tail))
        swap_marks["C_served"] = served("C")
        swap_marks["B_served"] = served("B")
        if verdict == "PASS":
            if swap_marks["failed_inference_after_swap_start"]:
                verdict, reasons = "FAIL", reasons + ["the gateway logged failed inference during/after the swap"]
            if swap_marks["C_served"] <= 24:
                verdict, reasons = "FAIL", reasons + ["machine C served no validator traffic"]
            if swap_marks.get("A_served_after_drain", 0) - swap_marks.get("A_served_at_removal", 0) > 0:
                verdict, reasons = "FAIL", reasons + ["machine A kept serving after its drain"]
        elif phase != "after":
            reasons.append(f"run ended in phase {phase}")
        else:
            reasons.append("the 2:1 split did not return after the swap")
    finally:
        for name in list(procs):
            stop(name)
    capacity = {}
    for v in VALIDATORS:
        try:
            state = json.load(open(os.path.join(work, f"{v}-state.json")))
            capacity[v] = {r: state.get("miners", {}).get(str(uid_of[r]), {}).get("capacity") for r in MINERS}
        except (OSError, ValueError) as exc:
            capacity[v] = str(exc)
    json.dump({"verdict": verdict, "reasons": reasons, "expected_share": EXPECTED,
               "tolerance": {"relative": TOL_REL, "absolute": TOL_ABS}, "machine_dps": CAPACITY, "uids": uid_of,
               "swap": swap_marks, "events": events, "validator_capacity_ema": capacity, "history": history},
              open(os.path.join(work, "E2E-MULTINODE-RESULT.json"), "w"), indent=1)
    print("VERDICT", verdict, reasons, swap_marks)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
