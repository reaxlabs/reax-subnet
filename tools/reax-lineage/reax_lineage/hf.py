"""Hugging Face I/O: token from env, <= 8 parallel requests, back-off on 429, presigned-URL cache.

Sources (HFSource, LocalSource) expose the same small interface so the sampler does not care where
the bytes come from:
    list_files()            -> {path: FileInfo}
    read_file(path, max_bytes)
    read_range(path, start, end_inclusive)
    tensor_index()          -> {tensor name: TensorRef}
"""
import hashlib
import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests

from . import UA

MAX_PARALLEL_HARD_CAP = 8
HF = "https://huggingface.co"
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".npz", ".npy", ".gguf", ".ckpt", ".onnx", ".msgpack", ".h5")
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I16": 2,
            "I32": 4, "I64": 8, "BOOL": 1}
FLOAT_OK = ("BF16", "F16", "F32")


ERROR_KINDS = ("repo_unreadable", "transient")


def kind_for_status(status):
    """error_kind of an HTTP status. Any 4xx except 408 / 425 / 429 means the repo (or file, or revision) cannot be read by us: private, gated, missing,
    removed (401, 403, 404, 410, 416, 451, ...): `repo_unreadable`. 429, 5xx, 408 / 425 and anything unexpected can pass: `transient`. An unreadable
    or temporarily failing repo is never a reason to refuse an entry (RULES 2.7: a temporary error on our side or at Hugging Face never refuses)."""
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
        return "repo_unreadable"
    return "transient"


def scrub_url(text):
    """Drop query strings (presigned CDN URLs carry signatures) from a message before it is logged or put into a result."""
    return re.sub(r"\?[^\s'\"<>)]+", "?...", str(text))


class HFError(RuntimeError):
    """A read failed. `status` = the last HTTP status (None for timeouts / connection errors / local problems); `kind` = "repo_unreadable" (private, gated,
    missing, removed, not a readable model repo: retrying will not help until somebody changes something) or "transient" (429 / 5xx / timeouts /
    connection errors / a server that ignores Range: try again later). The CLI prints both as machine-readable fields (`error_kind`, `http_status`)."""

    def __init__(self, msg, status=None, kind=None):
        super().__init__(msg)
        self.status = status
        self.kind = kind if kind in ERROR_KINDS else kind_for_status(status)


@dataclass
class FileInfo:
    path: str
    size: int
    sha256: Optional[str] = None      # known for LFS files (HF tree API) and for local files


@dataclass
class TensorRef:
    file: str
    offset: int       # absolute byte offset of the tensor data inside the file
    nbytes: int
    dtype: str
    shape: tuple


def hf_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


class HFClient:
    """Thin requests wrapper. One global semaphore (<= 8), shared cool-down after a 429."""

    def __init__(self, token="env", max_parallel=None, base=HF, ua=UA, max_tries=10, url_ttl=1500):
        self.token = hf_token() if token == "env" else token
        n = max_parallel or int(os.environ.get("REAX_LINEAGE_MAX_PARALLEL", MAX_PARALLEL_HARD_CAP))
        self.max_parallel = max(1, min(int(n), MAX_PARALLEL_HARD_CAP))
        self.base, self.ua, self.max_tries, self.url_ttl = base, ua, max_tries, url_ttl
        self._sem = threading.BoundedSemaphore(self.max_parallel)
        self._local = threading.local()
        self._cool_until = 0.0
        self._lock = threading.Lock()
        self._url_cache: Dict[str, tuple] = {}
        self.stats = {"requests": 0, "retries_429": 0, "bytes": 0, "peak_inflight": 0}
        self._inflight = 0

    # -- plumbing -----------------------------------------------------------
    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4))
            self._local.s = s
        return s

    def _headers(self, auth=True, rng=None):
        h = {"User-Agent": self.ua}
        if auth and self.token:
            h["Authorization"] = "Bearer " + self.token
        if rng is not None:
            h["Range"] = "bytes=%d-%d" % rng
        return h

    def _request(self, method, url, auth=True, rng=None, allow_redirects=True, timeout=(15, 120), ok=(200, 206)):
        """One request under the semaphore with back-off. Returns the response (status in `ok`, or a 3xx when
        allow_redirects is False)."""
        last, last_status, last_kind = None, None, "transient"
        for attempt in range(self.max_tries):
            wait = self._cool_until - time.time()
            if wait > 0:
                time.sleep(wait)
            with self._sem:
                with self._lock:
                    self._inflight += 1
                    self.stats["requests"] += 1
                    self.stats["peak_inflight"] = max(self.stats["peak_inflight"], self._inflight)
                try:
                    r = self._session().request(method, url, headers=self._headers(auth, rng),
                                                allow_redirects=allow_redirects, timeout=timeout)
                except requests.RequestException as e:
                    last, last_status, last_kind = scrub_url(e), None, "transient"        # timeout / connection error
                    r = None
                finally:
                    with self._lock:
                        self._inflight -= 1
            if r is None:
                time.sleep(min(2 * (attempt + 1), 20))
                continue
            if r.status_code in ok or (not allow_redirects and r.status_code in (301, 302, 303, 307, 308)):
                self.stats["bytes"] += len(r.content) if method == "GET" else 0
                return r
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                try:
                    delay = float(ra) if ra else min(5 * 2 ** attempt, 120)
                except ValueError:
                    delay = 30
                with self._lock:
                    self.stats["retries_429"] += int(r.status_code == 429)
                    self._cool_until = max(self._cool_until, time.time() + delay)
                last, last_status, last_kind = HFError("HTTP %d for %s" % (r.status_code, url.split("?")[0])), r.status_code, "transient"
                continue
            if r.status_code in (401, 403, 404, 410, 416, 451):         # final: private, gated, missing, removed, legal block, range past the end
                raise HFError("HTTP %d for %s" % (r.status_code, url.split("?")[0]), status=r.status_code)
            last, last_status = HFError("HTTP %d for %s" % (r.status_code, url.split("?")[0])), r.status_code
            last_kind = kind_for_status(r.status_code)
            time.sleep(min(2 * (attempt + 1), 20))
        raise HFError("giving up after %d tries: %s" % (self.max_tries, last), status=last_status, kind=last_kind)

    # -- API ----------------------------------------------------------------
    def api(self, path, params=None):
        url = self.base + "/api/" + path.lstrip("/")
        if params:
            url += "?" + "&".join("%s=%s" % kv for kv in params.items())
        r = self._request("GET", url, ok=(200,))
        try:
            return r.json()
        except ValueError:
            raise HFError("HTTP %d for %s: the answer is not JSON" % (r.status_code, url.split("?")[0]), status=r.status_code, kind="transient")

    def resolve_revision(self, repo, rev="main"):
        """Full commit sha and last-modified time for repo@rev (abbreviated shas work on the Hub)."""
        d = self.api("models/%s/revision/%s" % (repo, rev))
        return {"sha": d["sha"], "last_modified": d.get("lastModified"), "gated": d.get("gated"),
                "tags": d.get("tags", []), "card": d.get("cardData") or {}}

    def tree(self, repo, rev):
        out, url = [], "%s/api/models/%s/tree/%s?recursive=1&expand=0" % (self.base, repo, rev)
        r = self._request("GET", url, ok=(200,))
        out.extend(r.json())
        while 'rel="next"' in r.headers.get("Link", ""):
            nxt = r.headers["Link"].split(">")[0].lstrip("<")
            r = self._request("GET", nxt, ok=(200,))
            out.extend(r.json())
        return out

    def commit_time(self, repo, rev):
        """Commit time of the revision (UTC ISO string) -- what 'public before the commitment' means for public models."""
        return self.api("models/%s/revision/%s" % (repo, rev)).get("lastModified")

    # -- file reads ---------------------------------------------------------
    def _file_url(self, repo, rev, path):
        return "%s/%s/resolve/%s/%s" % (self.base, repo, rev, path)

    def read(self, repo, rev, path, rng=None):
        """Read a file (or the inclusive byte range) through the cached presigned URL."""
        key = "%s@%s/%s" % (repo, rev, path)
        url = self._file_url(repo, rev, path)
        with self._lock:
            hit = self._url_cache.get(key)
        if hit and hit[1] > time.time():
            try:
                r = self._request("GET", hit[0], auth=False, rng=rng, ok=(200, 206))
                return self._check(r, rng)
            except HFError:
                with self._lock:
                    self._url_cache.pop(key, None)
        r = self._request("GET", url, rng=rng, allow_redirects=False, ok=(200, 206))
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers["Location"]
            if loc.startswith("/"):
                loc = self.base + loc
            cdn = not loc.startswith(self.base)
            if cdn:       # presigned CDN URL: reuse for the next range reads, without the HF token
                with self._lock:
                    self._url_cache[key] = (loc, time.time() + self.url_ttl)
            r = self._request("GET", loc, auth=not cdn, rng=rng, ok=(200, 206))
        return self._check(r, rng)

    @staticmethod
    def _check(r, rng):
        if rng is not None and r.status_code != 206:
            raise HFError("server ignored the Range request (HTTP %d)" % r.status_code, status=r.status_code, kind="transient")
        return r.content


class HFSource:
    kind = "hf"

    def __init__(self, repo, rev, client: HFClient):
        self.repo, self.rev, self.client = repo, rev, client
        self.name = "%s@%s" % (repo, rev)
        self._files = None
        self._tensors = None

    def list_files(self):
        if self._files is None:
            fs = {}
            for t in self.client.tree(self.repo, self.rev):
                if t.get("type") != "file":
                    continue
                lfs = t.get("lfs") or {}
                fs[t["path"]] = FileInfo(t["path"], int(t.get("size", 0)), lfs.get("oid"))
            self._files = fs
        return self._files

    def read_file(self, path, max_bytes=None):
        data = self.client.read(self.repo, self.rev, path)
        return data

    def read_range(self, path, start, end):
        return self.client.read(self.repo, self.rev, path, (start, end))

    def tensor_index(self):
        if self._tensors is None:
            files = self.list_files()
            names = [p for p in files if p.endswith(".safetensors") and "/" not in p]
            idx = None
            if "model.safetensors.index.json" in files:
                try:
                    wm = json.loads(self.read_file("model.safetensors.index.json"))["weight_map"]
                    idx = sorted(set(wm.values()))
                except Exception:
                    idx = None
            shards = idx or sorted(p for p in names if p.startswith("model"))
            self._tensors = read_headers(self.read_range, shards)
        return self._tensors


def parse_header(first_bytes, fetch_more):
    """first_bytes >= 8 bytes; fetch_more(start, end_incl) used if the header is longer than what we have."""
    n = struct.unpack("<Q", first_bytes[:8])[0]
    if n > 100_000_000:
        raise HFError("implausible safetensors header length %d" % n, kind="repo_unreadable")
    have = first_bytes[8:8 + n]
    if len(have) < n:
        have += fetch_more(8 + len(have), 8 + n - 1)
    hdr = json.loads(have)
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


def read_headers(read_range, shards, first=262143):
    out = {}
    for f in shards:
        b = read_range(f, 0, first)
        base, hdr = parse_header(b, lambda s, e, f=f: read_range(f, s, e))
        for k, v in hdr.items():
            a, z = v["data_offsets"]
            out[k] = TensorRef(f, base + a, z - a, v["dtype"], tuple(v["shape"]))
    return out


class LocalSource:
    """A local directory with safetensors shards (a checkpoint, or a synthetic test repo)."""
    kind = "local"

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise HFError("not a directory: %s" % path, kind="repo_unreadable")
        self.repo, self.rev = os.path.basename(self.path.rstrip("/")), "local"
        self.name = "local:" + self.path
        self._files = None
        self._tensors = None

    def list_files(self):
        if self._files is None:
            fs = {}
            for root, _, names in os.walk(self.path):
                for n in names:
                    p = os.path.join(root, n)
                    rel = os.path.relpath(p, self.path)
                    if rel.startswith(".git" + os.sep):
                        continue
                    fs[rel] = FileInfo(rel, os.path.getsize(p), None)
            self._files = fs
        return self._files

    def file_sha256(self, path):
        h = hashlib.sha256()
        with open(os.path.join(self.path, path), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    def read_file(self, path, max_bytes=None):
        with open(os.path.join(self.path, path), "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def read_range(self, path, start, end):
        with open(os.path.join(self.path, path), "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    def tensor_index(self):
        if self._tensors is None:
            files = self.list_files()
            shards = None
            if "model.safetensors.index.json" in files:
                shards = sorted(set(json.loads(self.read_file("model.safetensors.index.json"))["weight_map"].values()))
            if not shards:
                shards = sorted(p for p in files if p.endswith(".safetensors") and "/" not in p)
            self._tensors = read_headers(self.read_range, shards)
        return self._tensors


def file_sha256_map(source, extra_only=True):
    """sha256 of every non-shard weight file ('readout etc.'). LFS oid for HF, computed for local files."""
    files = source.list_files()
    shard_names = set(r.file for r in source.tensor_index().values())
    out = {}
    for p, fi in files.items():
        if not p.lower().endswith(WEIGHT_EXT) or p in shard_names:
            continue
        if fi.sha256:
            out[p] = fi.sha256
        elif source.kind == "local":
            out[p] = source.file_sha256(p)
        else:
            out[p] = "unknown-size-%d" % fi.size
    return out
