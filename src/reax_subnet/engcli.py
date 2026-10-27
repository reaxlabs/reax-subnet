"""Engine competition tools (unreleased; docs/ENGINE-COMPETITION.md).

``reax-engine-eval``: measure one engine image against the reference on this machine, exactly as a validator does,
without any chain access. Competitors use it before submitting; validators use it to calibrate.

    reax-engine-eval --reference <image>@sha256:<d> --candidate <image>@sha256:<d> --weights /path/to/model [--gpu 0]

Engine rules v2.0 (``--track dm-all``, docs/ENGINE-COMPETITION-V2.md): every model the image declares, against the
published reference image of that model (config.ENGINE_COMPOSITE_REFERENCE_IMAGES), with the release weights and
calibration; prints the composite score and the per-model breakdown:

    reax-engine-eval --track dm-all --candidate <image>@sha256:<d> \
        --weights mammoth=/w/mammoth,blink=/w/blink,reax1=/w/reax1,jeffan=/w/jeffan [--models mammoth,blink]

``reax-engine-submit``: publish ``reax-eng/1:<track>:<image>@sha256:<digest>`` as the hotkey's engine element, next to
its inference element if it has one (docs/DESIGN-MULTIROLE.md; see ``reax-commit`` for the rules). While the
competition runs on finney (from COMPETITIONS_START_UTC, docs/COMPETITIONS.md); on other networks it needs
REAX_FEATURE_ENGINE_COMPETITION=1. While the engine track is held on finney (competition_split.ENGINE_TRACK_START_UTC is
None) it publishes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

from .competition_split import engine_track_open
from .engcomp import commitment_text, parse_engine_commitment
from .engcomposite import COMPOSITE_PROTOCOL_TRACK
from .engvalidator import TRACKS


def _pairs(text: str, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        name, sep, value = part.strip().partition("=")
        if not sep or not name.strip() or not value.strip():
            sys.exit(f"{what}: expected slot=value,... (got {part.strip()!r})")
        out[name.strip().lower()] = value.strip()
    return out


def composite_eval(args) -> None:
    """``reax-engine-eval --track dm-all``: one image on every model it declares, as a v2.0 validator measures it (same
    track constants, reference digests, gates and per-model budgets; no chain, no champion rule)."""
    from .config import ENGINE_COMPOSITE_REFERENCE_IMAGES
    from .engcomposite import MODEL_ENV, SLOT_WEIGHTS_FROM, CompositeRunner, default_track, evaluate_composite, parse_declared
    from .engrunner import DevRunner, DockerRunner, EngineError

    if args.params:
        sys.exit("--params is not supported for dm-all (the per-model parameters are the release constants)")
    track = default_track()
    names = [s.name for s in track.slots]
    refs = {k: v for k, v in ENGINE_COMPOSITE_REFERENCE_IMAGES.items() if v}
    over = _pairs(args.reference, "--reference")
    unknown = sorted(set(over) - set(names))
    if unknown:
        sys.exit(f"--reference names unknown models: {', '.join(unknown)} (models: {', '.join(names)})")
    bad = sorted(n for n, v in over.items() if "@sha256:" not in v)
    if bad:
        sys.exit("--reference must pin every image by digest (image@sha256:...): " + ", ".join(bad))
    refs.update(over)
    dirs: dict[str, str] = {}
    if args.dev_map:
        runners = {s.params.name: DevRunner.from_file(os.path.join(args.workdir, s.name), args.dev_map)
                   for s in track.slots}
    else:
        dirs = _pairs(args.weights, "--weights")
        unknown = sorted(set(dirs) - set(names))
        if unknown:
            sys.exit(f"--weights names unknown models: {', '.join(unknown)} (models: {', '.join(names)})")
        for slot, source in SLOT_WEIGHTS_FROM.items():
            if source in dirs and not dirs.get(slot):
                dirs[slot] = dirs[source]
        runners = {s.params.name: DockerRunner(args.workdir, dirs.get(s.name) or os.devnull, gpu=args.gpu,
                                               docker_runtime=args.docker_runtime, pull=not args.no_pull,
                                               memory=args.memory, cpus=args.cpus,
                                               registries=[r for r in args.registries.split(",") if r],
                                               model_env={MODEL_ENV: s.model})
                   for s in track.slots}
    forced = None
    if args.models:
        try:
            forced = parse_declared(args.models, track)
        except ValueError as exc:
            sys.exit(f"--models: {exc}")
        try:
            _check_inputs(forced, dirs, refs, bool(args.dev_map))
        except _MissingInput as exc:
            sys.exit(str(exc))
    runner = CompositeRunner(runners)
    header = {"track": track.name, "composite_rules_sha256": _rules_sha(track),
              "weights": {s.name: s.weight for s in track.slots}, "not_release_ready": track.release_ready(),
              "overridden_references": sorted(over), "declared_from": "--models" if forced else "image label"}
    print(json.dumps(header), file=sys.stderr, flush=True)
    results = []
    for cand in args.candidate:
        row: dict = {"candidate": cand}
        try:
            declared = forced or parse_declared(asyncio.run(runner.models_label(cand)), track)
            if not forced:
                _check_inputs(declared, dirs, refs, bool(args.dev_map))
            row["declared"] = sorted(declared)
            budget = track.budget_s(declared)
            res = asyncio.run(asyncio.wait_for(
                evaluate_composite(runner, refs[sorted(declared)[0]], cand, track, declared, references=refs), budget))
            row.update(res.to_json())
        except ValueError as exc:
            row.update(ok=False, reason=f"models label: {exc}")
        except _MissingInput as exc:  # this image's label needs a model we have no weights for: say so, go on
            row["inconclusive"] = str(exc)
        except EngineError as exc:  # the image itself (size, registry, start): a failure, as on a validator
            row.update(ok=False, reason=str(exc))
        except asyncio.TimeoutError:
            row.update(ok=False, reason=f"evaluation exceeded {track.budget_s(declared):.0f} s")
        except Exception as exc:  # noqa: BLE001  ReferenceError_, InfraError, anything else: not a verdict
            row["inconclusive"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(row), flush=True)
        results.append(row)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({**header, "references": refs, "results": results}, fh, indent=1)


class _MissingInput(Exception):
    pass


def _check_inputs(declared, dirs: dict[str, str], refs: dict[str, str], dev: bool) -> None:
    missing = sorted(n for n in declared if not dev and not os.path.isdir(dirs.get(n, "")))
    missing += sorted(n for n in declared if not refs.get(n))
    if missing:
        raise _MissingInput("no weights directory / reference image for declared model(s): " + ", ".join(missing)
                            + " (--weights slot=/dir,...; vision uses the blink directory)")


def _rules_sha(track) -> str:
    from .engcomposite import composite_rules_sha256

    return composite_rules_sha256(track)


def eval_main(argv=None) -> None:
    from .engrunner import DevRunner, DockerRunner, InfraError, ReferenceError_, evaluate

    ap = argparse.ArgumentParser(prog="reax-engine-eval", description=__doc__.split("\n\n")[1])
    ap.add_argument("--reference", default="",
                    help="v1 tracks: required; dm-all: optional slot=<image@sha256:...>,... overrides of the published references")
    ap.add_argument("--candidate", required=True, action="append", help="repeatable")
    ap.add_argument("--track", default="s1-fast", choices=sorted({*TRACKS, COMPOSITE_PROTOCOL_TRACK}))
    ap.add_argument("--weights", default="",
                    help="pinned model weights directory (docker runner); dm-all: slot=/dir,... (vision uses blink's)")
    ap.add_argument("--models", default="",
                    help="dm-all: models to measure (comma-separated slot names or 'all'; default: the image's "
                         "ai.reax.engine.models label, as a validator reads it)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--docker-runtime", default="")
    ap.add_argument("--memory", default="64g", help="container memory limit")
    ap.add_argument("--cpus", default="8", help="container CPU limit")
    ap.add_argument("--registries", default="ghcr.io,docker.io,quay.io", help="allowed registries (comma-separated)")
    ap.add_argument("--no-pull", action="store_true", help="use local images (still addressed by digest)")
    ap.add_argument("--dev-map", default="", help="JSON map image -> command (tests only)")
    ap.add_argument("--workdir", default=os.path.expanduser("~/.reax/engines"))
    ap.add_argument("--params", default="", help="JSON overrides of the track parameters (calibration only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.track == COMPOSITE_PROTOCOL_TRACK:
        composite_eval(args)
        return
    if not args.reference:
        sys.exit("--reference is required for track " + args.track)
    params = TRACKS[args.track]
    if args.params:
        params = dataclasses.replace(params, **json.loads(args.params))
    if args.dev_map:
        runner = DevRunner.from_file(args.workdir, args.dev_map)
    else:
        if not os.path.isdir(args.weights):
            sys.exit("--weights must be the pinned model weights directory")
        runner = DockerRunner(args.workdir, args.weights, gpu=args.gpu, docker_runtime=args.docker_runtime,
                              pull=not args.no_pull, memory=args.memory, cpus=args.cpus,
                              registries=[r for r in args.registries.split(",") if r])
    results = []
    for cand in args.candidate:
        try:
            res = asyncio.run(asyncio.wait_for(evaluate(runner, args.reference, cand, args.track, params),
                                               params.eval_budget_s))
            row = {"candidate": cand, **res.to_json()}
        except (ReferenceError_, InfraError) as exc:
            row = {"candidate": cand, "inconclusive": str(exc)}
        except asyncio.TimeoutError:
            row = {"candidate": cand, "ok": False, "reason": f"evaluation exceeded {params.eval_budget_s:.0f} s"}
        print(json.dumps(row), flush=True)
        results.append(row)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"track": args.track, "params": dataclasses.asdict(params), "reference": args.reference,
                       "results": results}, fh, indent=1)


def submit_main(argv=None) -> None:
    from .cli import _wallet
    from .config import ConfigError, MAINNET_NAMES, load_common

    ap = argparse.ArgumentParser(prog="reax-engine-submit")
    ap.add_argument("image", help="<image>@sha256:<digest>")
    ap.add_argument("--track", default="s1-fast", choices=sorted(TRACKS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace-all", action="store_true",
                    help="write the engine element alone (drops an inference element on the same hotkey)")
    args = ap.parse_args(argv)
    network = os.environ.get("REAX_NETWORK", "finney") or "finney"
    if network not in MAINNET_NAMES and os.environ.get("REAX_FEATURE_ENGINE_COMPETITION") != "1":
        sys.exit("on a development network or the testnet the engine competition needs REAX_FEATURE_ENGINE_COMPETITION=1")
    if network in MAINNET_NAMES and not engine_track_open(datetime.now(timezone.utc)):
        sys.exit("the engine track is not open on finney (held by the subnet owner); no engine submission is accepted")
    image, sep, digest = args.image.partition("@sha256:")
    if not sep:
        sys.exit("pin the image by digest: <image>@sha256:<64 hex>")
    try:
        text = commitment_text(args.track, image, digest)
    except ValueError as exc:
        sys.exit(str(exc))
    assert parse_engine_commitment(text) is not None
    if args.dry_run:
        print(text)
        return
    try:
        common = load_common()
        wallet, hotkey = _wallet(common)
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    from .chain import SdkChain
    from .commitcli import Refused, force_multirole, same_elements, write_plan
    from .elements import fields_of

    # Every write replaces all fields: keep the hotkey's inference element next to the new engine element.
    chain = SdkChain(common.network, common.netuid, wallet, commitments=True)
    snap = chain.snapshot()
    mine = next((n for n in snap.neurons if n.hotkey == hotkey), None)
    if mine is None:
        sys.exit(f"hotkey {hotkey} is not registered on subnet {common.netuid}")
    current = fields_of(mine)
    try:
        fields = write_plan(current, text, snap.block, replace_all=args.replace_all, force=force_multirole())
    except Refused as exc:
        sys.exit(f"reax-engine-submit: {exc}")
    if same_elements(current, fields):
        print(f"{hotkey} already has exactly these elements on chain; nothing written")
        return
    chain.set_commitment_fields(fields)
    print(f"published {' + '.join(fields)} for {hotkey}")
