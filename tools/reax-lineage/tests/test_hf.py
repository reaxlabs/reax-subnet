"""HF client: at most 8 parallel requests, Retry-After back-off on 429, range reads, presigned-URL reuse without the token."""
import http.server
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from reax_lineage.hf import HFClient, HFError, parse_header


class Handler(http.server.BaseHTTPRequestHandler):
    DATA = bytes(range(256)) * 400
    state = {"inflight": 0, "max": 0, "n429": 0, "to_429": 0, "auth_cdn": [], "auth_hf": [], "hits": 0, "ua": set()}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_GET(self):
        S = Handler.state
        with Handler.lock:
            S["inflight"] += 1
            S["max"] = max(S["max"], S["inflight"])
            S["hits"] += 1
            S["ua"].add(self.headers.get("User-Agent"))
        try:
            time.sleep(0.02)
            host = self.headers.get("Host", "")
            if self.path.startswith("/cdn/"):
                S["auth_cdn"].append(self.headers.get("Authorization"))
            else:
                S["auth_hf"].append(self.headers.get("Authorization"))
            if S["to_429"] > 0:
                with Handler.lock:
                    S["to_429"] -= 1
                    S["n429"] += 1
                self.send_response(429)
                self.send_header("Retry-After", "0.2")
                self.end_headers()
                return
            if "/resolve/" in self.path:
                self.send_response(302)
                self.send_header("Location", "http://localhost:%d/cdn/file.bin?sig=1" % self.server.server_port)
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, b = rng[6:].split("-")
                a, b = int(a), min(int(b), len(self.DATA) - 1)
                body = self.DATA[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, len(self.DATA)))
            else:
                body = self.DATA
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with Handler.lock:
                S["inflight"] -= 1


@pytest.fixture()
def server():
    Handler.state.update(inflight=0, max=0, n429=0, to_429=0, auth_cdn=[], auth_hf=[], hits=0, ua=set())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_port
    srv.shutdown()


def test_parallel_cap_is_eight_even_if_more_is_asked(server):
    c = HFClient(token="secret-token", max_parallel=64, base=server)
    assert c.max_parallel == 8
    with ThreadPoolExecutor(32) as ex:
        out = list(ex.map(lambda i: c.read("o/m", "r", "f%d.bin" % (i % 3), (i * 10, i * 10 + 9)), range(64)))
    assert all(o == Handler.DATA[i * 10:i * 10 + 10] for i, o in enumerate(out))
    assert Handler.state["max"] <= 8 and c.stats["peak_inflight"] <= 8
    assert Handler.state["ua"] == {"reax-lineage-check/1.0"}


def test_token_goes_to_hf_not_to_the_cdn(server):
    c = HFClient(token="secret-token", base=server)
    c.read("o/m", "r", "f.bin", (0, 9))
    c.read("o/m", "r", "f.bin", (10, 19))                  # second read reuses the presigned URL: no resolver hit
    assert all(a == "Bearer secret-token" for a in Handler.state["auth_hf"])
    assert all(a is None for a in Handler.state["auth_cdn"])
    assert len(Handler.state["auth_hf"]) == 1 and len(Handler.state["auth_cdn"]) == 2


def test_429_backoff_honours_retry_after(server):
    c = HFClient(token=None, base=server)
    Handler.state["to_429"] = 3
    t = time.time()
    out = c.read("o/m", "r", "f.bin", (0, 4))
    assert out == Handler.DATA[:5] and c.stats["retries_429"] == 3 and time.time() - t >= 0.2


def test_gives_up_on_permanent_errors():
    c = HFClient(token=None, base="http://127.0.0.1:9", max_tries=2)
    with pytest.raises(HFError):
        c.read("o/m", "r", "f.bin", (0, 4))


def test_header_parser_reads_long_headers_in_two_requests():
    import struct
    hdr = json.dumps({"a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}, "pad": "x" * 500}).encode()
    blob = struct.pack("<Q", len(hdr)) + hdr
    calls = []
    base, h = parse_header(blob[:100], lambda s, e: (calls.append((s, e)), blob[s:e + 1])[1])
    assert base == 8 + len(hdr) and "a" in h and len(calls) == 1
