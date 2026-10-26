"""Pre-run submission check of an engine image (docs/ENGINE-COMPETITION.md, "Pre-run submission check").

Before the FIRST run of a new digest the validator reads the image configuration (``docker image inspect``; nothing of
the image is executed) and the tokenizer / model-config files the runtime might load, and writes a review record per
digest to ``<review_dir>/reviews/<digest>.json``. Obvious violations reject the submission with a clear reason. In
release mode a digest also needs a human / isolated-LLM review: it only runs once the operator has created the file
``<review_dir>/REVIEWED/<digest>`` (64 hex characters, no ``sha256:`` prefix) after that review.

This is a screen for obvious violations, not a proof of safety: the runtime sandbox (runsc, no network, read-only
root, no capabilities) is the security boundary and the review gate is the backstop for what a static screen cannot see.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from typing import Any, Mapping, Optional, Sequence

REVIEW_RECORD_VERSION = 1
MAX_IMAGE_SIZE_BYTES = 40 * 10 ** 9  # uncompressed, as reported by docker; the compressed pull is capped separately
MAX_CHECKED_FILE_BYTES = 64 * 2 ** 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Files a runtime may load as tokenizer or model configuration, and the directories they are looked for in. The
# weights directory is mounted at /model over whatever the image has there, so files in the image's own copy of /model
# are never read by the runtime itself, but an engine can be pointed at any path.
CHECKED_FILE_NAMES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                      "vocab.json", "merges.txt", "generation_config.json", "config.json", "chat_template.jinja")
MODEL_LIKE_DIRS = ("/model", "/models", "/opt/model")  # where a file of the same name is the model's own copy
CHECKED_DIRS = ("/", "/app", "/model", "/models", "/opt/model", "/opt/reax", "/srv", "/workspace")

_PRIVILEGED = re.compile(r"privileged|cap[_-]add|security[_-]opt|docker\.sock|nsenter|--pid[= ]host|--net(work)?[= ]host|"
                         r"host[_-]?network|/proc/sysrq|SYS_ADMIN|SYS_PTRACE", re.I)
_PROXY_ENV = {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy"}
_ROOT_USERS = ("root", "0")


def valid_digest(digest: str) -> bool:
    return bool(_HEX64.match(digest))


def digest_of(ref: str) -> str:
    """The 64 hex characters of ``<image>@sha256:<hex>`` ("" when the reference is not pinned by digest)."""
    _, _, d = ref.partition("@sha256:")
    return d if valid_digest(d) else ""


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _strs(value: Any) -> list[str]:
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def check_image_config(info: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """``(violations, warnings, summary)`` from the JSON of ``docker image inspect`` (one image). Pure."""
    cfg = info.get("Config") if isinstance(info.get("Config"), dict) else {}
    violations: list[str] = []
    warnings: list[str] = []
    user = _text(cfg.get("User")).strip()
    entry, cmd = _strs(cfg.get("Entrypoint")), _strs(cfg.get("Cmd"))
    env = _strs(cfg.get("Env"))
    ports = sorted(cfg.get("ExposedPorts") or {}) if isinstance(cfg.get("ExposedPorts"), dict) else []
    labels = {str(k): str(v) for k, v in (cfg.get("Labels") or {}).items()} if isinstance(cfg.get("Labels"), dict) else {}
    volumes = sorted(cfg.get("Volumes") or {}) if isinstance(cfg.get("Volumes"), dict) else []
    size = info.get("Size") if isinstance(info.get("Size"), int) else 0
    os_name, arch = _text(info.get("Os")), _text(info.get("Architecture"))

    name, _, group = user.partition(":")
    if name in _ROOT_USERS:
        violations.append(f"image runs as root (User={user!r})")
    elif not user:
        warnings.append("image sets no User (the validator forces --user 10002:10002)")
    if group in _ROOT_USERS:
        violations.append(f"image runs with the root group (User={user!r})")
    if ports:
        violations.append("image exposes network ports: " + ", ".join(ports[:8]))
    if volumes:
        violations.append("image declares VOLUME paths (anonymous writable volumes escape --read-only): "
                          + ", ".join(volumes[:8]))
    if not entry and not cmd:
        violations.append("image has neither Entrypoint nor Cmd")
    for item in env:
        key, _, value = item.partition("=")
        if key.lower() in _PROXY_ENV:
            violations.append(f"network configuration in Env: {key}")
        elif key in ("LD_PRELOAD", "LD_AUDIT"):
            violations.append(f"library injection in Env: {key}")
        elif _PRIVILEGED.search(item):
            violations.append(f"privileged hint in Env {key}")
    for text in (" ".join(entry), " ".join(cmd)):
        if _PRIVILEGED.search(text):
            violations.append("privileged hint in Entrypoint/Cmd")
            break
    for k, v in labels.items():
        if _PRIVILEGED.search(k) or _PRIVILEGED.search(v):
            violations.append(f"privileged hint in label {k[:60]!r}")
    if size > MAX_IMAGE_SIZE_BYTES:
        violations.append(f"image is {size / 1e9:.1f} GB unpacked, the limit is {MAX_IMAGE_SIZE_BYTES / 1e9:.0f} GB")
    if (os_name and os_name != "linux") or (arch and arch not in ("amd64", "arm64")):
        violations.append(f"unsupported platform {os_name}/{arch}")
    summary = {"user": user, "entrypoint": entry, "cmd": cmd, "env_keys": sorted(i.partition("=")[0] for i in env),
               "exposed_ports": ports, "volumes": volumes, "labels": {k[:80]: v[:200] for k, v in sorted(labels.items())},
               "workdir": _text(cfg.get("WorkingDir")), "size_bytes": size, "os": os_name, "architecture": arch}
    return list(dict.fromkeys(violations)), warnings, summary


# Names an engine also uses for its own, unrelated settings: matched by bare name only inside a model directory. The
# tokenizer and vocabulary files keep the bare-name comparison everywhere (a tampered /app/tokenizer.json is what we look for).
AMBIGUOUS_NAMES = ("config.json", "generation_config.json")


def model_like(path: str) -> bool:
    """True when ``path`` is a file directly in one of the model directories (the only place a bare file name is matched)."""
    return os.path.dirname(path) in MODEL_LIKE_DIRS


def compare_files(candidate: Mapping[str, str], expected: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Tokenizer / config files of the candidate image (path -> sha256) against the expected ones (path or bare file
    name -> sha256, taken from the pinned weights directory, else from the reference image; a bare name only applies to
    files in a model directory, except for the tokenizer files, which are compared wherever they are). A file that differs from
    an expected copy is a violation; a checked file with nothing to compare with is only a warning."""
    violations, warnings = [], []
    for path, sha in sorted(candidate.items()):
        # same path first; by bare file name everywhere for tokenizer files, but for config.json only in a model
        # directory: an engine's own /app/config.json (server settings) is not the model's and must not be rejected
        base = os.path.basename(path)
        want = expected.get(path) or (expected.get(base) if model_like(path) or base not in AMBIGUOUS_NAMES else None)
        if sha.startswith("!"):
            violations.append(f"{path}: {sha[1:]}")
        elif want is None:
            warnings.append(f"{path}: no reference copy to compare with")
        elif want != sha:
            violations.append(f"{path} differs from the reference copy (tokenizer/config mismatch)")
    return violations, warnings


def hash_regular_file(path: str) -> str:
    """sha256 of a regular file; ``!reason`` for a symlink, another file type or a file above the size cap (the host side
    of a ``docker cp`` must never follow a link the image planted)."""
    try:
        st = os.lstat(path)
    except OSError:
        return "!unreadable"
    if not stat.S_ISREG(st.st_mode):
        return "!not a regular file (link or special file)"
    if st.st_size > MAX_CHECKED_FILE_BYTES:
        return "!file larger than the check limit"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:  # e.g. mode 000 in the image (docker cp keeps the mode bits): the image's doing, not a host fault
        return "!unreadable"
    return h.hexdigest()


def weights_file_hashes(weights_dir: str, names: Sequence[str] = CHECKED_FILE_NAMES) -> dict[str, str]:
    """Expected copies: the checked file names that exist at the top of the pinned weights directory."""
    out = {}
    for n in names:
        p = os.path.join(weights_dir, n)
        if os.path.isfile(p) and not os.path.islink(p):
            out[n] = hash_regular_file(p)
    return out


# ------------------------------------------------------------------------------------------------ records
def reviews_dir(review_dir: str) -> str:
    return os.path.join(review_dir, "reviews")


def record_path(review_dir: str, digest: str) -> str:
    if not valid_digest(digest):
        raise ValueError("bad digest")
    return os.path.join(reviews_dir(review_dir), digest + ".json")


def load_record(review_dir: str, digest: str) -> Optional[dict[str, Any]]:
    try:
        with open(record_path(review_dir, digest), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("version") == REVIEW_RECORD_VERSION else None


def write_record(review_dir: str, digest: str, record: dict[str, Any]) -> None:
    os.makedirs(reviews_dir(review_dir), mode=0o700, exist_ok=True)
    path = record_path(review_dir, digest)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def is_reviewed(review_dir: str, digest: str) -> bool:
    """The operator's marker ``REVIEWED/<digest>``: a regular file (a symlink does not count)."""
    if not valid_digest(digest):
        return False
    try:
        return stat.S_ISREG(os.lstat(os.path.join(review_dir, "REVIEWED", digest)).st_mode)
    except OSError:
        return False


def new_record(ref: str, digest: str, violations: list[str], warnings: list[str], summary: dict[str, Any],
               files: Mapping[str, str]) -> dict[str, Any]:
    return {"version": REVIEW_RECORD_VERSION, "ref": ref[:300], "digest": digest, "checked_at": int(time.time()),
            "status": "rejected" if violations else "static_ok", "violations": violations, "warnings": warnings,
            "image": summary, "files": dict(files)}


# ------------------------------------------------------------------------------------------------ copy line
def copyline_fields(info: Mapping[str, Any]) -> dict[str, str]:
    """The copy-line labels of the image (engcopyline.PARENTS_LABEL / OWNERSHIP_LABEL), complete, for the review record
    (the summary's labels are cut at 200 characters). Only the labels that exist are returned: an image without them
    leaves the record exactly as before."""
    from .engcopyline import MAX_LABEL_BYTES, OWNERSHIP_LABEL, PARENTS_LABEL
    cfg = info.get("Config") if isinstance(info.get("Config"), dict) else {}
    labels = cfg.get("Labels") if isinstance(cfg.get("Labels"), dict) else {}
    out = {}
    for key, name in (("parents_label", PARENTS_LABEL), ("ownership_label", OWNERSHIP_LABEL)):
        v = labels.get(name)
        if isinstance(v, str) and v:
            out[key] = v[:MAX_LABEL_BYTES + 1]  # one byte over the cap stays visible, so the parser rejects it
    return out


def copy_marker(review_dir: str, digest: str) -> bool:
    """The operator's marker ``COPY/<digest>``: review found undeclared reuse of another entry's code (source or binaries)
    in this image. A regular file (a symlink does not count), like ``REVIEWED/<digest>``."""
    if not valid_digest(digest):
        return False
    try:
        return stat.S_ISREG(os.lstat(os.path.join(review_dir, "COPY", digest)).st_mode)
    except OSError:
        return False


# Option A (25 Oct 2026): inventory ONLY layers not shared with the pinned reference.
# Exact tokenizer names come from manifests/s1-fast-v1.json weights.files. There are
# no .bin/.pt/etc tokenizer exceptions in that reference. Do not allow arbitrary
# 'tokenizer.bin' or 'vocab.pt' names. compare_files still checks tokenizer identity.
REFERENCE_TOKENIZER_FILES = frozenset({"tokenizer.json"})
# s1-fast-v1.json, weights.files + read-only pinned cache inventory:
# plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186/tokenizer.json.
# Reference runtime mounts this directory at /model (runtimes/s1-fast/Dockerfile).
REFERENCE_TOKENIZER_PATH = "/model/tokenizer.json"
REFERENCE_TOKENIZER_SIZE = 19989325
REFERENCE_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
WEIGHT_FILE_SUFFIXES = frozenset({".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz", ".gguf", ".ckpt", ".pkl", ".onnx", ".h5", ".tflite", ".msgpack", ".pte"})
_CODE_SUFFIXES = frozenset({".py", ".pyc", ".so", ".cu", ".cubin", ".ptx", ".c", ".cc", ".cpp", ".cxx",
                            ".h", ".hpp", ".json", ".txt", ".md", ".toml", ".yaml", ".yml", ".sh",
                            ".ini", ".cfg", ".conf", ".jinja", ".jinja2"})


def inventory_files(inventory: Any) -> list[dict[str, Any]]:
    """Normalize reviewer JSON: path->metadata, layer->path->metadata, or rows.

    Layer inventories preserve overwritten files too; a merged filesystem alone
    cannot establish absence of weight files in earlier image layers.
    """
    rows: list[dict[str, Any]] = []
    if isinstance(inventory, list):
        for row in inventory:
            if not isinstance(row, dict) or "path" not in row:
                raise ValueError("inventory rows must be objects with path and size")
            rows.append(dict(row))
    elif isinstance(inventory, dict):
        for key, value in inventory.items():
            if not isinstance(value, dict):
                raise ValueError("inventory metadata must be an object")
            if "size" in value:
                rows.append({**value, "path": key})
            else:
                for row in inventory_files(value):
                    rows.append({"layer": key, **row})
    else:
        raise ValueError("inventory must be a JSON object or list")
    for row in rows:
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("inventory path must be a nonempty string")
        if type(row.get("size")) is not int or row["size"] < 0:
            raise ValueError("inventory size must be a nonnegative integer")
    return rows


def weight_file_screen(file_list: Any) -> list[dict[str, Any]]:
    """Find weight-like files in NON-reference layer inventories, without I/O.

    Extension checks apply at any path, regardless of text metadata or size.
    Unknown text files do not become code merely by being marked is_text.
    Source/native-code exemptions are filename classifications, not proof that
    code contains no embedded tensors; source/rebuild review remains mandatory.
    """
    from pathlib import PurePosixPath
    findings = []
    rows = inventory_files(file_list)
    parts = {}
    for row in rows:
        path = PurePosixPath(row["path"])
        suffix = path.suffix.lower()
        reasons = []
        if suffix in WEIGHT_FILE_SUFFIXES or re.search(r"\.(?:safetensors|pt)\.", path.name.lower()):
            reasons.append("weight-like file extension " + suffix)
        code = suffix in _CODE_SUFFIXES or bool(re.search(r"\.so(?:\.\d+)+$", path.name))
        tokenizer = (row["path"] == REFERENCE_TOKENIZER_PATH and
                     row["size"] * 10 <= REFERENCE_TOKENIZER_SIZE * 11 and
                     row.get("sha256") == REFERENCE_TOKENIZER_SHA256)
        if path.name.lower() == "tokenizer.json" and not tokenizer:
            reasons.append("tokenizer differs from pinned reference path/size/sha256")
        if suffix == ".json":
            code = False
        if row["size"] > MAX_CHECKED_FILE_BYTES and not code and not tokenizer:
            reasons.append("non-code file exceeds 64 MiB")
        part = re.match(r"^(.*?)(?:[._-](?:part[._-]?)?)(\d+)(?:[._-]of[._-]?\d+)?$", str(path), re.I)
        if part:
            parts.setdefault(part[1], []).append(row)
        if reasons:
            findings.append({**row, "reason": "; ".join(reasons)})
    for group in parts.values():
        if len({r["path"] for r in group}) >= 2 and sum(r["size"] for r in group) > MAX_CHECKED_FILE_BYTES:
            for row in group:
                findings.append({**row, "reason": "numeric multipart group exceeds 64 MiB"})
    return sorted(findings, key=lambda r: (r["path"], str(r.get("layer", "")), r.get("sha256", "")))


def load_inventory(path: str) -> list[dict[str, Any]]:
    """Read reviewer JSON or inv-N.tsv (sha256, size, path; symlink rows skipped)."""
    with open(path, encoding="utf-8") as fh:
        data = fh.read()
    if data.lstrip().startswith(("{", "[")):
        return inventory_files(json.loads(data))
    # Early reviewer exports have a header (layer, path, size, sha256, type, ...).
    if data.startswith("layer\tpath\tsize\tsha256\t"):
        import csv
        rows = []
        for item in csv.DictReader(data.splitlines(), delimiter="\t"):
            if item["sha256"] == "SYMLINK" or item["size"] in ("LINK", "WHITEOUT"):
                continue
            if not valid_digest(item["sha256"]):
                raise ValueError("invalid inventory sha256")
            rows.append({"path": item["path"], "size": int(item["size"]),
                         "sha256": item["sha256"], "layer": item["layer"]})
        return inventory_files(rows)
    rows = []
    for line in data.splitlines():
        if not line.strip():
            continue
        sha, size, name = line.split("\t", 2)
        if sha == "SYMLINK":
            continue
        if not valid_digest(sha):
            raise ValueError("invalid inventory sha256")
        rows.append({"path": name, "size": int(size), "sha256": sha})
    return inventory_files(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Static engine review tools (no image execution)")
    sub = parser.add_subparsers(dest="command", required=True)
    weights = sub.add_parser("weights", help="screen NON-reference layers for shipped weights")
    weights.add_argument("--inventory", required=True, help="reviewer files-*.json or inv-N.tsv")
    args = parser.parse_args(argv)
    try:
        findings = weight_file_screen(load_inventory(args.inventory))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"compliant": not findings, "findings": findings}, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
