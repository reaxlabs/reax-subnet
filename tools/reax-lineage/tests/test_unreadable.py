"""F2 (intake replay, 13 Nov): a private, gated or missing repo (HTTP 401 / 403 / 404) made `check` end with an uncaught HFError (traceback, exit 1). Now it ends
with one clean JSON line {"error", "error_kind", "http_status"} on stdout and exit 2; error_kind is `repo_unreadable` (4xx: private, gated, missing, removed)
or `transient` (429, 5xx, timeouts, connection errors). A fake Hugging Face on a local port answers with the status the repo name asks for."""
import http.server
import json
import socket
import threading
import types

import pytest

import reax_lineage.hf as hf_mod
from reax_lineage import cli
from reax_lineage.hf import HFClient, HFError
from reax_lineage.sources import Hub

SHA = "a" * 40


class FakeHF(http.server.BaseHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeHF.hits.append(self.path)
        parts = self.path.split("?")[0].strip("/").split("/")          # api / models / org / name / revision|tree / rev
        name = parts[3] if len(parts) > 3 else ""
        status = {"r401": 401, "r403": 403, "r404": 404, "r410": 410, "r429": 429, "r500": 500, "r503": 503, "r400": 400}.get(name)
        if status is None:
            self.send_response(200)
            body = json.dumps({"sha": SHA, "lastModified": "2026-01-01T00:00:00Z"}) if parts[4] == "revision" else json.dumps([])
            if name == "rhtml":
                body = "<html>not json</html>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
            return
        self.send_response(status)
        if status == 429:
            self.send_header("Retry-After", "0")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


@pytest.fixture
def fake_hf(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHF)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    FakeHF.hits.clear()
    # no real sleeping in the retry loops (the back-off itself is tested in test_hf.py)
    monkeypatch.setattr(hf_mod, "time", types.SimpleNamespace(time=__import__("time").time, sleep=lambda s: None))
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    client = HFClient(token=None, base=base, max_tries=3)
    monkeypatch.setattr(cli, "Hub", lambda hub_dir=None: Hub(client=client))
    yield base
    srv.shutdown()
    srv.server_close()


def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    assert "Traceback" not in cap.out + cap.err
    lines = cap.out.strip().splitlines()
    return code, (json.loads(lines[-1]) if lines else None), cap


CHECK = ["check", "--hotkey", "hk", "--block", "1", "--no-tier-b", "--jobs", "1"]


@pytest.mark.parametrize("name,status", [("r401", 401), ("r403", 403), ("r404", 404), ("r410", 410)])
@pytest.mark.parametrize("rev", ["main", SHA])
def test_private_gated_missing_repo_is_a_clean_repo_unreadable_error(fake_hf, capsys, name, status, rev):
    """rev=main: the first call (revision) fails; rev=<40 hex>: that call is skipped and the file listing (tree) fails instead."""
    code, out, cap = run_cli(capsys, *CHECK, "--repo", "org/" + name, "--rev", rev)
    assert code == 2
    assert out["error_kind"] == "repo_unreadable" and out["http_status"] == status
    assert ("HTTP %d" % status) in out["error"] and "org/" + name in out["error"]
    assert set(out) == {"error", "error_kind", "http_status"}
    assert cap.err == ""
    assert len(FakeHF.hits) == 1                       # 4xx is final: no retries


@pytest.mark.parametrize("name,status", [("r429", 429), ("r500", 500), ("r503", 503)])
def test_rate_limit_and_server_errors_are_a_clean_transient_error_after_the_retries(fake_hf, capsys, name, status):
    code, out, cap = run_cli(capsys, *CHECK, "--repo", "org/" + name, "--rev", "main")
    assert code == 2
    assert out["error_kind"] == "transient" and out["http_status"] == status
    assert out["error"].startswith("giving up after 3 tries")
    assert len(FakeHF.hits) == 3                       # the back-off loop ran to its end first


def test_a_connection_error_is_transient(monkeypatch, capsys):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                          # nothing listens here any more
    monkeypatch.setattr(hf_mod, "time", types.SimpleNamespace(time=__import__("time").time, sleep=lambda s: None))
    client = HFClient(token=None, base="http://127.0.0.1:%d" % port, max_tries=2)
    monkeypatch.setattr(cli, "Hub", lambda hub_dir=None: Hub(client=client))
    code, out, _ = run_cli(capsys, *CHECK, "--repo", "org/x", "--rev", "main")
    assert code == 2 and out["error_kind"] == "transient" and out["http_status"] is None and "giving up after 2 tries" in out["error"]


def test_an_answer_that_is_not_json_is_transient(fake_hf, capsys):
    code, out, _ = run_cli(capsys, *CHECK, "--repo", "org/rhtml", "--rev", "main")
    assert code == 2 and out["error_kind"] == "transient" and "not JSON" in out["error"]


def test_dry_run_and_fingerprint_end_the_same_way(fake_hf, capsys):
    code, out, _ = run_cli(capsys, "dry-run", "--repo", "org/r404", "--rev", "main")
    assert code == 2 and out["error_kind"] == "repo_unreadable"
    code, out, _ = run_cli(capsys, "fingerprint", "--repo", "org/r403", "--rev", "main")
    assert code == 2 and out["error_kind"] == "repo_unreadable" and out["http_status"] == 403


def test_error_text_never_carries_a_query_string(monkeypatch):
    """Presigned CDN URLs carry signatures in the query string: neither HFError messages nor the CLI error may repeat them."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(hf_mod, "time", types.SimpleNamespace(time=__import__("time").time, sleep=lambda s: None))
    c = HFClient(token=None, base="http://127.0.0.1:%d" % port, max_tries=2)
    from reax_lineage.hf import scrub_url           # new in v1.5.0 (imported here so the CLI tests above also run, and fail, on v1.4.1)
    with pytest.raises(HFError) as ei:
        c._request("GET", "http://127.0.0.1:%d/file?X-Amz-Signature=SECRETSIG&Expires=1" % port, auth=False)
    assert "SECRETSIG" not in str(ei.value) and ei.value.kind == "transient"
    assert scrub_url("see https://x/y?a=1&b=2 (and) ok") == "see https://x/y?... (and) ok"
    assert scrub_url("really?") == "really?"


def test_kind_for_status_table():
    from reax_lineage.hf import kind_for_status
    assert {s: kind_for_status(s) for s in (400, 401, 403, 404, 410, 416, 451)} == {s: "repo_unreadable" for s in (400, 401, 403, 404, 410, 416, 451)}
    assert {s: kind_for_status(s) for s in (None, 200, 408, 425, 429, 500, 502, 503, 504)} == {s: "transient" for s in (None, 200, 408, 425, 429, 500, 502, 503, 504)}
    assert HFError("x").kind == "transient" and HFError("x", status=404).kind == "repo_unreadable" and HFError("x", kind="repo_unreadable").kind == "repo_unreadable"
    assert HFError("x", status=404, kind="bogus").kind == "repo_unreadable"        # an unknown kind falls back to the status
