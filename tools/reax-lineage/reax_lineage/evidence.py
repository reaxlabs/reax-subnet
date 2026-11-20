"""Steps 0.2-0.5: record schema, repo evidence (model ids), config and tokenizer equality."""
import copy
import fnmatch
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import jsonschema

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schema", "reax-lineage.schema.json")           # reax-lineage/2 (published)
SCHEMA_V1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schema", "reax-lineage-1.schema.json")      # reax-lineage/1 (legacy: one start)
SCHEMA_V1, SCHEMA_V2 = "reax-lineage/1", "reax-lineage/2"
MAX_PARENTS = 4                 # RULES 2.2 / 3.1 (d8b M6): at most 4 parents ...
MIN_PARENT_SHARE = 0.10         # ... each with a share of at least 0.10
SHARE_SUM_TOL = 0.02            # shares add up to 1 +/- 0.02
MAX_OWN_DEPTH = 4               # an own parent's own parents ... (nesting depth that is followed)
MAX_EVIDENCE_BYTES = 1_000_000
HEX40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Problem:
    code: str
    message: str
    field: str = ""

    def to_dict(self):
        return {"code": self.code, "message": self.message, "field": self.field}


def load_schema(version=SCHEMA_V2):
    with open(SCHEMA_V1_PATH if version == SCHEMA_V1 else SCHEMA_PATH) as f:
        return json.load(f)


def canonical_json(rec: dict) -> bytes:
    """Sorted keys, UTF-8, no whitespace, without the signature field (what the hotkey signs for a correction)."""
    r = {k: v for k, v in rec.items() if k != "signature"}
    return json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_signature(rec: dict, hotkey: str):
    """sr25519 check of a correction. True/False when a verifier is installed, None when it is not (the MCP verifies)."""
    sig = rec.get("signature")
    if not sig:
        return False
    try:
        import sr25519    # type: ignore
        from scalecodec.utils.ss58 import ss58_decode  # type: ignore
    except Exception:
        return None
    try:
        pub = bytes.fromhex(ss58_decode(hotkey))
        return bool(sr25519.verify(bytes.fromhex(sig.replace("0x", "")), canonical_json(rec), pub))
    except Exception:
        return False


def validate_record(rec, hotkey: Optional[str] = None, entry_repo: Optional[str] = None,
                    entry_rev: Optional[str] = None) -> List[Problem]:
    """Step 0.2: JSON schema + the semantic checks the schema cannot express. `reax-lineage/1` records are validated against the legacy schema (one `start`);
    everything else against the v2 schema (a wrong `schema` value is then a schema problem)."""
    out: List[Problem] = []
    if not isinstance(rec, dict):
        return [Problem("schema", "reax-lineage.json is not a JSON object")]
    v = jsonschema.Draft7Validator(load_schema(SCHEMA_V1 if rec.get("schema") == SCHEMA_V1 else SCHEMA_V2))
    for e in sorted(v.iter_errors(rec), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in e.absolute_path)
        out.append(Problem("schema", e.message[:300], path))
    if out:
        return out
    if hotkey and rec["hotkey"] != hotkey:
        out.append(Problem("hotkey", "record names hotkey %s, the commitment is from %s" % (rec["hotkey"], hotkey), "hotkey"))
    if entry_repo and rec["entry"]["repo"].lower() != entry_repo.lower():
        out.append(Problem("entry", "record entry.repo %s differs from the committed repo %s" % (rec["entry"]["repo"], entry_repo), "entry.repo"))
    if entry_rev and rec["entry"]["revision"] != entry_rev and not rec.get("corrects"):
        out.append(Problem("entry", "record entry.revision differs from the committed revision", "entry.revision"))
    if rec.get("schema") == SCHEMA_V2:
        out.extend(parent_problems(rec))
    if rec.get("corrects"):
        sv = verify_signature(rec, rec["hotkey"])
        if sv is False:
            out.append(Problem("signature", "a correction must carry a valid hotkey signature", "signature"))
    return out


# ------------------------------------------------------------ parents (v2) ----

def normalise_parents(rec: dict):
    """-> (parents, from_scratch, legacy). A `reax-lineage/1` record is ONE parent with share 1 (its `start`; kind none = from scratch). A /2 record carries `parents`
    (each {kind, repo, revision, uid?, share, parents? (own models only)}) and `from_scratch`. The parents are copied; the record is never changed."""
    if not isinstance(rec, dict):
        return [], True, False
    if rec.get("schema") == SCHEMA_V1 or ("parents" not in rec and "start" in rec):
        st = rec.get("start") or {"kind": "none"}
        if st.get("kind") in (None, "none"):
            return [], True, True
        p = {"kind": st["kind"], "repo": st["repo"], "revision": st["revision"], "share": 1.0}
        if st.get("uid") is not None:
            p["uid"] = st["uid"]
        return [p], False, True
    parents = [dict(p) for p in (rec.get("parents") or [])]
    return parents, bool(rec.get("from_scratch")), False


def parents_rule_problems(parents: List[dict], legacy: bool = False) -> List[Problem]:
    """RULES 2.2 / 3.1 / 3.3 (d8b M6): more than 4 parents, or a parent with a share below 0.10, is REFUSED `parents_rule` (no penalty; the caller makes these class R).
    A v1 record has one parent with share 1: nothing to refuse."""
    out: List[Problem] = []
    if legacy:
        return out
    if len(parents) > MAX_PARENTS:
        out.append(Problem("parents_rule", "the record names %d parents; an entry has at most %d (RULES 2.2)" % (len(parents), MAX_PARENTS), "parents"))
    for i, p in enumerate(parents):
        sh = p.get("share")
        if isinstance(sh, (int, float)) and not isinstance(sh, bool) and sh < MIN_PARENT_SHARE - 1e-9:
            out.append(Problem("parents_rule", "parent %s@%s has a share of %s, below %.2f (RULES 2.2: each parent needs at least %.2f)" % (
                p.get("repo"), str(p.get("revision"))[:8], sh, MIN_PARENT_SHARE, MIN_PARENT_SHARE), "parents/%d/share" % i))
    return out


def parent_problems(rec: dict) -> List[Problem]:
    """The semantic checks of a /2 record's parents that are record fields (NEEDS_CORRECTION): the same model twice, the entry as its own parent, shares that do not add up
    to 1 +/- 0.02. Only when `parents_rule_problems` finds nothing (a refusal beats a correction: it cannot be corrected)."""
    parents, scratch, legacy = normalise_parents(rec)
    out: List[Problem] = []
    if scratch or not parents or parents_rule_problems(parents, legacy):
        return out
    seen = {}
    me = (rec["entry"]["repo"].lower(), rec["entry"]["revision"])
    for i, p in enumerate(parents):
        k = (p["repo"].lower(), p["revision"])
        if k == me:
            out.append(Problem("parent_self", "parent %d is the entry itself" % i, "parents/%d" % i))
        if k in seen:
            out.append(Problem("parents_duplicate", "%s@%s is named twice as a parent (parents %d and %d): name it once, with its total share" % (p["repo"], p["revision"][:8], seen[k], i), "parents/%d" % i))
        seen.setdefault(k, i)
    tot = sum(float(p["share"]) for p in parents)
    if abs(tot - 1.0) > SHARE_SUM_TOL + 1e-9:
        out.append(Problem("parent_shares", "the parent shares add up to %.3f, not 1 (+/- %.2f)" % (tot, SHARE_SUM_TOL), "parents"))
    return out


def walk_own_parents(parents: List[dict], depth: int = 0):
    """Every (parent, depth) below the parents that are `own` models, depth-first in record order, to MAX_OWN_DEPTH levels (a cycle cannot occur: the record is a tree)."""
    for p in parents:
        sub = p.get("parents") if p.get("kind") == "own" else None
        if sub and depth < MAX_OWN_DEPTH:
            for q in sub:
                yield q, depth + 1
            yield from walk_own_parents(sub, depth + 1)


# ------------------------------------------------------------ repo evidence ----

TEXT_EXT = (".md", ".json", ".yml", ".yaml", ".txt", ".toml", ".cfg", ".jinja")
SKIP_FILES = {"tokenizer.json", "model.safetensors.index.json", "reax-lineage.json", "chat_template.jinja",
              "chat_template.json", "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json",
              "generation_config.json", "preprocessor_config.json", "processor_config.json"}
FILE_EXT = {"py", "json", "md", "txt", "yaml", "yml", "safetensors", "bin", "pt", "png", "jpg", "jpeg", "gif", "jsonl", "csv",
            "tsv", "parquet", "gguf", "sh", "cfg", "toml", "html", "js", "ts", "pdf", "svg", "npz", "ckpt", "log", "ipynb", "jinja"}
DIR_WORDS = {"src", "tmp", "usr", "etc", "home", "var", "bin", "lib", "opt", "mnt", "workspace", "data", "models", "model",
             "scripts", "checkpoints", "checkpoint", "output", "outputs", "input", "inputs", "configs", "config", "logs",
             "runs", "results", "root", "dev", "proc", "sys", "app", "content", "kaggle", "drive", "datasets", "docs", "assets",
             "application", "text", "image", "audio", "video", "multipart", "font", "message", "and", "or", "yes", "no", "on",
             "off", "true", "false", "w", "n", "i", "o", "km", "mb", "gb", "tb", "kb", "s", "ms", "e", "g", "a", "b"}
ID_RE = re.compile(r"(?<![A-Za-z0-9_.@:/-])([A-Za-z0-9][A-Za-z0-9_.-]{0,95})/([A-Za-z0-9][A-Za-z0-9_.-]{0,95})(?![A-Za-z0-9_/-])")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
HF_URL_RE = re.compile(r"(?:huggingface\.co|hf\.co)/(datasets/|spaces/)?([A-Za-z0-9][A-Za-z0-9_.-]{0,95})/([A-Za-z0-9][A-Za-z0-9_.-]{0,95})")


def _strip_model_index(text: str) -> str:
    """Evaluation results in README front matter (model-index) name benchmarks, not training sources."""
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return text
    fm = m.group(1)
    fm2 = re.sub(r"(?ms)^model-index:.*?(?=^[A-Za-z_-]+:|\Z)", "", fm)
    return "---\n" + fm2 + "\n---" + text[m.end():]


def extract_candidate_ids(text: str, ignore_hf_urls_kind=False) -> Dict[str, str]:
    """-> {id lower: 'model'|'dataset'|'?'} candidate org/name strings in one text file."""
    out: Dict[str, str] = {}
    for m in HF_URL_RE.finditer(text):
        kind = "dataset" if m.group(1) == "datasets/" else ("space" if m.group(1) == "spaces/" else "?")
        if kind != "space":
            out[(m.group(2) + "/" + m.group(3)).rstrip(".").lower()] = kind
    text = URL_RE.sub(" ", text)
    for m in ID_RE.finditer(text):
        a, b = m.group(1), m.group(2).rstrip(".")
        if not b or len(a) < 2 or len(b) < 2:
            continue
        if a.lower() in DIR_WORDS or b.lower() in DIR_WORDS:
            continue
        if "." in b and b.rsplit(".", 1)[-1].lower() in FILE_EXT:
            continue
        if re.fullmatch(r"[\d.,%-]+", a) or re.fullmatch(r"[\d.,%-]+", b):
            continue
        out.setdefault((a + "/" + b).lower(), "?")
    return out


def scan_evidence(files: Dict[str, str], resolver: Callable[[str, str], Optional[str]], skip_ids=()) -> Dict[str, dict]:
    """files: {path: text}. Keep only ids that resolve to a Hub model or dataset. -> {id: {kind, files:[...]}}."""
    cands: Dict[str, dict] = {}
    for path, text in files.items():
        base = os.path.basename(path)
        if base in SKIP_FILES or path.startswith(".eval_results"):
            continue
        if base.lower().startswith("readme"):
            text = _strip_model_index(text)
        for i, kind in extract_candidate_ids(text).items():
            if i in skip_ids:
                continue
            cands.setdefault(i, {"kind": kind, "files": []})["files"].append(path)
    found = {}
    for i, d in sorted(cands.items()):
        kind = resolver(i, d["kind"])
        if kind:
            found[i] = {"kind": kind, "files": sorted(set(d["files"]))}
    return found


def declared_ids(rec: dict) -> Dict[str, str]:
    """Every id the record names, lower case -> field."""
    out = {}

    def add(s, field):
        if not s:
            return
        for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)", str(s)):
            out.setdefault(m.group(1).rstrip(".").lower(), field)

    add(rec.get("entry", {}).get("repo"), "entry")
    add((rec.get("start") or {}).get("repo"), "start")
    for p in rec.get("parents") or []:
        if isinstance(p, dict):
            add(p.get("repo"), "parents")
    for q, _ in walk_own_parents([p for p in (rec.get("parents") or []) if isinstance(p, dict)]):
        add(q.get("repo"), "parents")
    for m in rec.get("own_models", []):
        add(m.get("repo"), "own_models")
    for t in rec.get("teachers", []):
        add(t.get("repo_or_api"), "teachers")
        add(t.get("name"), "teachers")
    for d in rec.get("datasets", []):
        add(d.get("source"), "datasets")
        add(d.get("name"), "datasets")
    for m in rec.get("mentioned_not_used", []):
        add(m.get("id"), "mentioned_not_used")
    return out


def undeclared_ids(found: Dict[str, dict], rec: dict) -> List[dict]:
    decl = declared_ids(rec)
    return [{"id": i, "kind": d["kind"], "files": d["files"]} for i, d in sorted(found.items()) if i not in decl]


# ------------------------------------------------------ config + tokenizer ----

STRUCT_FIELDS = ["architectures", "model_type", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                 "intermediate_size", "num_hidden_layers", "layer_types", "vocab_size", "rope_theta", "rope_scaling",
                 "rope_parameters", "rope_local_base_freq", "tie_word_embeddings", "hidden_act", "sliding_window",
                 "max_position_embeddings", "rms_norm_eps", "attention_bias", "full_attention_interval",
                 "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                 "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                 "query_pre_attn_scalar", "final_logit_softcapping", "attn_logit_softcapping", "partial_rotary_factor"]
FORBIDDEN_CONFIG = ["auto_map"]          # remote code (RULES-v2 2.3)


def text_config(cfg: dict) -> dict:
    for k in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(k), dict):
            return {**{kk: vv for kk, vv in cfg.items() if kk not in ("text_config", "vision_config")}, **cfg[k]}
    return cfg


# RULES-v2 2.4: "A text-only export of a start that also has a vision part is allowed without a declaration."
# The one documented multimodal-wrapper difference is the class name: a start's `XForConditionalGeneration` (with `text_config` and a
# `vision_config` / `audio_config`) is exported as `XForCausalLM` with the text config at the top level (Gemma-4: Gemma4ForConditionalGeneration ->
# Gemma4ForCausalLM, model_type gemma4 -> gemma4_text; real pairs in tests/golden/configs). Everything else (sizes, layers, rope, vocabulary,
# tying ...) must still equal the start's, and `model_type` of the export must be the start's text model type; wrapper-only keys
# (vision_config, audio_config, *_token_id, ...) are not structural fields and were never compared.
WRAPPER_SUFFIX = "ForConditionalGeneration"
TEXT_SUFFIX = "ForCausalLM"
VISION_KEYS = ("vision_config", "audio_config")


def text_only_export(entry_cfg: dict, start_cfg: dict) -> Optional[dict]:
    """If `entry_cfg` is a text-only export of a multimodal `start_cfg` in the documented sense, return what was recognised, else None."""
    st_archs, en_archs = start_cfg.get("architectures"), entry_cfg.get("architectures")
    if not (isinstance(st_archs, list) and isinstance(en_archs, list) and len(st_archs) == 1 and len(en_archs) == 1):
        return None
    sa, ea = str(st_archs[0]), str(en_archs[0])
    has_text_part = isinstance(start_cfg.get("text_config"), dict)
    start_multimodal = has_text_part and any(isinstance(start_cfg.get(k), dict) for k in VISION_KEYS)
    entry_text_only = not isinstance(entry_cfg.get("text_config"), dict) and not any(isinstance(entry_cfg.get(k), dict) for k in VISION_KEYS)
    if not (start_multimodal and entry_text_only and sa.endswith(WRAPPER_SUFFIX)):
        return None
    if ea != sa[:-len(WRAPPER_SUFFIX)] + TEXT_SUFFIX:
        return None
    if entry_cfg.get("model_type") != start_cfg["text_config"].get("model_type"):
        return None
    return {"start_architecture": sa, "entry_architecture": ea, "model_type": entry_cfg.get("model_type"), "ignored_field": "architectures"}


def config_diff(entry_cfg: dict, start_cfg: dict, arch: dict) -> List[Problem]:
    """Structural config fields must equal the start's, except those named in architecture_changes and the documented text-only export."""
    out = []
    declared = dict((arch or {}).get("config_fields") or {})
    wrapper_only = text_only_export(entry_cfg, start_cfg) is not None
    if (arch or {}).get("added_tokens"):
        declared.setdefault("vocab_size", None)
    if (arch or {}).get("layer_map") is not None or (arch or {}).get("pruned"):
        for k in ("num_hidden_layers", "layer_types", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                  "hidden_size", "head_dim"):
            declared.setdefault(k, None)
    e, s = text_config(entry_cfg), text_config(start_cfg)
    for k in STRUCT_FIELDS:
        ev, sv = e.get(k), s.get(k)
        if ev == sv:
            continue
        if k == "architectures" and wrapper_only:
            continue                                    # the documented multimodal-wrapper field of a text-only export (RULES 2.4)
        if k in declared:
            if declared[k] is not None and declared[k] != ev:
                out.append(Problem("config", "declared %s=%r but config.json has %r" % (k, declared[k], ev), "architecture_changes.config_fields." + k))
            continue
        out.append(Problem("config", "config.json %s = %r differs from the start's %r and is not declared" % (k, ev, sv), k))
    for k in FORBIDDEN_CONFIG:
        if k in entry_cfg and k not in start_cfg:
            out.append(Problem("config", "config.json has %s (remote code is not allowed)" % k, k))
    return out


def _tok_parts(tok: dict):
    model = tok.get("model", {}) or {}
    vocab = model.get("vocab")
    if isinstance(vocab, list):                  # unigram: [[token, score], ...]
        vocab = {t[0]: i for i, t in enumerate(vocab)}
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", []) or []}
    return (vocab or {}), added, tok.get("normalizer"), tok.get("pre_tokenizer"), model.get("merges")


def tokenizer_diff(entry_tok: Optional[dict], start_tok: Optional[dict], arch: dict, same_sha: bool = False) -> List[Problem]:
    """Every token string of the start keeps its id; added tokens must be declared; normaliser/pre-tokeniser unchanged."""
    if same_sha:
        return []
    if entry_tok is None or start_tok is None:
        return [Problem("tokenizer", "tokenizer.json missing in the entry or in the start: cannot compare", "tokenizer")]
    out = []
    ev, ea, en, ep, em = _tok_parts(entry_tok)
    sv, sa, sn, sp, sm = _tok_parts(start_tok)
    declared = set((arch or {}).get("added_tokens") or [])
    # merged ONCE: v1.4.1 rebuilt both merged dicts for every token inside the comprehensions (O(V^2): minutes at V = 262,144, finding F1 of
    # the intake replay); added tokens still override vocabulary entries of the same string, exactly as before
    entry_all, start_all = {**ev, **ea}, {**sv, **sa}
    moved = [t for t, i in start_all.items() if entry_all.get(t) != i]
    if moved:
        out.append(Problem("tokenizer", "%d start tokens changed id or vanished (e.g. %r)" % (len(moved), moved[0]), "tokenizer"))
    new = [t for t in entry_all if t not in start_all]
    undeclared = [t for t in new if t not in declared]
    if undeclared:
        out.append(Problem("tokenizer", "%d added tokens are not in architecture_changes.added_tokens (e.g. %r)" % (len(undeclared), undeclared[0]), "tokenizer"))
    if en != sn:
        out.append(Problem("tokenizer", "normalizer changed", "tokenizer.normalizer"))
    if ep != sp:
        out.append(Problem("tokenizer", "pre_tokenizer changed", "tokenizer.pre_tokenizer"))
    if em != sm and not new:
        out.append(Problem("tokenizer", "BPE merges changed", "tokenizer.merges"))
    return out


# ------------------------------------------------------- several parents ----

# RULES 2.2: "All parents must be served by the same engine template of our engine (in practice: the same architecture)". The template is the architecture and the
# layer structure the weights are stored in: if one of these differs between two parents their tensors cannot be merged one-to-one. Context settings (rope, window,
# max positions, norm epsilon) and the vocabulary size are NOT part of it: the entry's own config is compared with every parent (`config_diff`), so a difference there
# is a NEEDS_CORRECTION unless declared, never a refusal. The documented text-only export of a multimodal parent counts as the same template (RULES 2.4).
TEMPLATE_FIELDS = ["architectures", "model_type", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
                   "num_hidden_layers", "layer_types", "tie_word_embeddings", "hidden_act", "full_attention_interval", "linear_num_key_heads",
                   "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok",
                   "moe_intermediate_size", "query_pre_attn_scalar", "final_logit_softcapping", "attn_logit_softcapping", "partial_rotary_factor", "attention_bias"]


def template_signature(cfg: dict) -> dict:
    """The template fields of a config.json (the text part of a multimodal config; the wrapper class `XForConditionalGeneration` read as `XForCausalLM`)."""
    t = text_config(cfg)
    sig = {k: t.get(k) for k in TEMPLATE_FIELDS}
    archs = cfg.get("architectures")
    if isinstance(archs, list) and isinstance(cfg.get("text_config"), dict):
        sig["architectures"] = [a[:-len(WRAPPER_SUFFIX)] + TEXT_SUFFIX if isinstance(a, str) and a.endswith(WRAPPER_SUFFIX) else a for a in archs]
    return sig


def template_differences(cfgs: Dict[str, dict]) -> List[dict]:
    """cfgs = {parent label: config.json}. Every parent is compared with the first one; -> [{"parent", "against", "fields": {field: [first, other]}}] (empty: one template)."""
    out = []
    labels = list(cfgs)
    if len(labels) < 2:
        return out
    base = template_signature(cfgs[labels[0]])
    for lab in labels[1:]:
        sig = template_signature(cfgs[lab])
        diff = {k: [base[k], sig[k]] for k in TEMPLATE_FIELDS if base[k] != sig[k]}
        if diff:
            out.append({"parent": lab, "against": labels[0], "fields": diff})
    return out


def merge_parent_problems(per_parent: Dict[str, List[Problem]]) -> List[Problem]:
    """The findings of the config / tokenizer comparison against several parents. An identical finding against several parents is ONE problem; a finding that carries the
    parent's own value (the start's value, a count of tokens) stays one problem per parent. With more than one parent every message names the parents it is about."""
    seen: Dict[tuple, dict] = {}
    for lab, probs in per_parent.items():
        for p in probs:
            seen.setdefault((p.code, p.field, p.message), {"p": p, "labels": []})["labels"].append(lab)
    out = []
    for d in seen.values():
        p, labs = d["p"], d["labels"]
        out.append(Problem(p.code, p.message + ((" [against parent %s]" % ", ".join(labs)) if len(per_parent) > 1 else ""), p.field))
    return out
