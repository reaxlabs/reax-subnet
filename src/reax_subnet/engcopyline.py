"""Engine track copy line (docs/ENGINE-COMPETITION.md, "Reusing other entries (copy line)"; gate
``engcomp.ENGINE_COPYLINE_START_UTC``, None = off).

Ideas are free. Reusing another entry's code (source OR binaries) is allowed only when declared: the image carries the label
``ai.reax.engine.parents`` (JSON ``[{"uid": 4, "digest": "<64 hex>", "files": ["/app/engine.py", ...]}]``). The entry then
counts as derived and must beat every parent by the full challenger margin (engcomp.update_champion, ``state.derived``).
Undeclared reuse found by review is a copy (``COPY/<digest>`` marker, see ``engreview.copy_marker``). The same owner (same
coldkey, or a signed ownership proof) may reuse its own code. A component without published source stays award-gated
(unpaid) until the source is published: that is the existing award gate and is not changed here.

Why a label and not a commitment field: the chain's raw field is 128 bytes and an engine commitment already uses most of it
(``reax-eng/1:<track>:<image>@sha256:<digest>``); the label sits in the image configuration, which is part of the digest, so
the declaration is bound to exactly the image that is run and cannot be edited afterwards; the pre-run check
(engreview.check_image_config) already reads the labels from ``docker image inspect``. This module is pure (no docker, no
chain, no network).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

PARENTS_LABEL = "ai.reax.engine.parents"
OWNERSHIP_LABEL = "ai.reax.engine.ownership"  # JSON [{"parent_digest", "parent_hotkey", "signature"}]
MAX_PARENTS = 8
MAX_FILES_PER_PARENT = 2000
MAX_LABEL_BYTES = 128 * 1024  # room for MAX_FILES_PER_PARENT paths of several parents
MAX_LABEL_DEPTH = 4  # parents: list > object > files list is 3 deep; anything deeper is not ours
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

VERDICT_DERIVED = "derived (declared)"
VERDICT_COPY = "copy (undeclared)"
VERDICT_SAME_OWNER = "same owner"
VERDICT_INDEPENDENT = "independent"

# File name suffixes of binaries without readable source (a shared object, a CUDA binary, ...).
BINARY_SUFFIXES = (".so", ".cubin", ".fatbin", ".ptx", ".a", ".o", ".pyd", ".dll", ".dylib", ".bin", ".pyc")
SOURCE_SUFFIXES = (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".rs", ".go", ".triton", ".pyx")


@dataclass(frozen=True)
class Parent:
    uid: int
    digest: str
    files: tuple[str, ...]


def _loads(value: str) -> Any:
    """``json.loads`` for a label: nesting deeper than MAX_LABEL_DEPTH is rejected before the parser sees it (a long run
    of ``[`` would otherwise raise RecursionError). Raises ValueError for anything that is not a usable document."""
    depth, in_str, esc = 0, False, False
    for ch in value:
        if in_str:
            in_str, esc = (not (ch == '"' and not esc)), (ch == "\\" and not esc)
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_LABEL_DEPTH:
                raise ValueError("label nested too deeply")
        elif ch in "]}":
            depth -= 1
    try:
        return json.loads(value)
    except RecursionError:
        raise ValueError("label nested too deeply") from None


def file_declared(path: str, declared: Iterable[str]) -> bool:
    """``path`` is named by a declaration: listed exactly, or under a listed directory prefix (an entry ending in "/")."""
    decl = declared if isinstance(declared, (set, frozenset)) else set(declared)
    return path in decl or any(d.endswith("/") and path.startswith(d) for d in decl)


def parse_parents(value: Any) -> tuple[list[Parent], list[str]]:
    """``(parents, problems)`` from the label value (a JSON string). A malformed entry is reported and skipped, never
    guessed; no label (None / empty) = no declaration. A file entry ending in "/" is a directory prefix. Pure."""
    if value is None or value == "":
        return [], []
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_LABEL_BYTES:
        return [], ["parents label is not a string or is larger than 128 KiB"]
    try:
        raw = _loads(value)
    except ValueError:
        return [], ["parents label is not valid JSON"]
    if not isinstance(raw, list):
        return [], ["parents label must be a JSON list"]
    parents: list[Parent] = []
    problems: list[str] = []
    for i, item in enumerate(raw[:MAX_PARENTS]):
        uid, digest, files = (item.get("uid"), item.get("digest"), item.get("files")) if isinstance(item, dict) else (None,) * 3
        if not (isinstance(uid, int) and not isinstance(uid, bool) and uid >= 0 and isinstance(digest, str)
                and _HEX64.match(digest) and isinstance(files, list) and files
                and all(isinstance(f, str) and f.startswith("/") and f != "/" and ".." not in f.split("/") for f in files)):
            problems.append(f"parents[{i}] needs uid, a 64-hex digest and a non-empty list of absolute file paths "
                            "(a directory prefix ends in \"/\")")
            continue
        if len(set(files)) > MAX_FILES_PER_PARENT:
            problems.append(f"parents[{i}] lists more than {MAX_FILES_PER_PARENT} files; list directory prefixes instead")
            continue
        parents.append(Parent(uid, digest, tuple(sorted(set(files)))))
    if len(raw) > MAX_PARENTS:
        problems.append(f"more than {MAX_PARENTS} parents declared; the rest is ignored")
    return parents, problems


# ------------------------------------------------------------------------------------------------ same owner
def ownership_message(parent_hotkey: str, child_hotkey: str, parent_digest: str) -> str:
    """The text the PARENT's hotkey signs (sr25519) to hand its code to ``child_hotkey``. The child's image carries the
    signature in the label ``ai.reax.engine.ownership``; the parent digest is the one named in the parents label."""
    return f"reax-eng/1 same-owner:{parent_hotkey}>{child_hotkey}:{parent_digest}"


Verifier = Callable[[str, str, str], bool]  # (ss58 address, message, hex signature) -> valid


def sr25519_verifier() -> Optional[Verifier]:
    """A verifier built on ``bittensor.sp_core.verify`` (the primitive behind ``bittensor.http_auth``, which auth.py uses for
    the miner exchange), or None when the chain libraries are not installed (every proof is then rejected, never guessed)."""
    try:
        from bittensor import sp_core  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return None

    def verify(address: str, message: str, signature_hex: str) -> bool:
        try:
            sig = bytes.fromhex(signature_hex.removeprefix("0x"))
            return len(sig) == 64 and bool(sp_core.verify(message.encode("utf-8"), sig, address, sp_core.CRYPTO_SR25519))
        except Exception:  # noqa: BLE001 - any malformed input is simply "not valid"
            return False
    return verify


def valid_ownership_proof(value: Any, parent_digest: str, parent_hotkey: str, child_hotkey: str,
                          verify: Optional[Verifier]) -> bool:
    """True when the ownership label holds a valid signature of ``parent_hotkey`` for exactly this parent digest and child
    hotkey. No verifier (chain libraries missing) = False."""
    if verify is None or not isinstance(value, str) or len(value.encode("utf-8")) > MAX_LABEL_BYTES:
        return False
    try:
        raw = _loads(value)
    except ValueError:
        return False
    for item in raw if isinstance(raw, list) else ():
        if (isinstance(item, dict) and item.get("parent_digest") == parent_digest
                and item.get("parent_hotkey") == parent_hotkey and isinstance(item.get("signature"), str)):
            return verify(parent_hotkey, ownership_message(parent_hotkey, child_hotkey, parent_digest), item["signature"])
    return False


def same_coldkey(hotkey_a: str, hotkey_b: str, coldkeys: Mapping[str, str]) -> bool:
    """Both hotkeys are in the snapshot with the same coldkey (``coldkeys``: hotkey -> coldkey)."""
    a, b = coldkeys.get(hotkey_a), coldkeys.get(hotkey_b)
    return a is not None and a == b


# ------------------------------------------------------------------------------------------------ the rule
def constraint_parents(declared: Iterable[str], detected: Iterable[str], same_owner: Iterable[str]) -> list[str]:
    """Parent digests a derived entry must beat by the full margin: every declared or detected parent except the ones the
    same owner holds. Sorted, no duplicates. Pure."""
    return sorted((set(declared) | set(detected)) - set(same_owner))


def verdict(declared: Iterable[str], shared: Mapping[str, Sequence[str]], same_owner: Iterable[str] = (),
            declared_files: Optional[Mapping[str, Iterable[str]]] = None) -> tuple[str, dict[str, str]]:
    """``(overall verdict, per-parent verdict)`` for a candidate. ``shared``: parent digest -> files byte-identical with
    that parent (what review detected; an empty list = nothing shared). ``declared``: parent digests the label names;
    ``declared_files``: digest -> files the label lists (when given, shared files outside that list, or outside its
    directory prefixes, are undeclared reuse).
    Per parent: same owner -> "same owner"; nothing shared -> "independent"; shared and declared (all shared files listed)
    -> "derived (declared)"; otherwise "copy (undeclared)". Overall: copy beats derived beats same owner beats independent.
    """
    decl, own = set(declared), set(same_owner)
    out: dict[str, str] = {}
    for digest in sorted(set(shared) | decl):
        files = set(shared.get(digest, ()))
        if digest in own and files:
            out[digest] = VERDICT_SAME_OWNER
        elif not files:
            out[digest] = VERDICT_INDEPENDENT
        elif digest in decl and (declared_files is None or all(file_declared(f, declared_files.get(digest, ())) for f in files)):
            out[digest] = VERDICT_DERIVED
        else:
            out[digest] = VERDICT_COPY
    for v in (VERDICT_COPY, VERDICT_DERIVED, VERDICT_SAME_OWNER):
        if v in out.values():
            return v, out
    return VERDICT_INDEPENDENT, out


def is_binary_name(path: str) -> bool:
    return path.lower().endswith(BINARY_SUFFIXES)


MODEL_DIRS = frozenset(("model", "models", "weights", "checkpoint", "checkpoints", "tokenizer", "tokenizers"))


def is_weight_file(path: str) -> bool:
    """Model or tokenizer weights, not a code component: ``.safetensors`` anywhere, ``.bin`` inside a directory named exactly
    model, models, weights, checkpoint(s) or tokenizer(s) (whole names, not substrings: ``/app/mymodelkernels/`` is not one).
    Any other ``.bin`` stays listed as a closed component, flagged "weights? (check)" by the review tool."""
    low = path.lower()
    if low.endswith(".safetensors"):
        return True
    return low.endswith(".bin") and any(part in MODEL_DIRS for part in low.split("/")[:-1])


def needs_weights_check(path: str) -> bool:
    """A closed-component candidate the reviewer should look at because it may be weights (a ``.bin`` outside a model directory)."""
    return path.lower().endswith(".bin")


def closed_components(files: Iterable[str]) -> list[str]:
    """Binary code files of ``files`` that have no source file of the same stem in the same list: components without
    published source (weight and tokenizer files are not counted). The entry's award stays unapproved while any is present:
    the operator withholds the approval (award.py); no code gates it. Report only, by name only. Pure."""
    names = sorted(set(files))
    stems = {n.rsplit(".", 1)[0] for n in names if n.lower().endswith(SOURCE_SUFFIXES)}
    return [n for n in names if is_binary_name(n) and not is_weight_file(n) and n.rsplit(".", 1)[0] not in stems]
