"""--vllm mode: download the pinned revision, normalise it like the preview does, serve it with the preview's vLLM flags.

Only the server process this module started is ever signalled (its own process group).
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Environment the preview serves with (vLLM 0.33.0 on one RTX PRO 6000 96 GB).
SERVE_ENV = {
    "MAX_JOBS": "4",
    "VLLM_USE_FLASHINFER_MOE_FP16": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
}
VLLM_PIN = "0.33.0"


def serve_command(vllm_bin, model_dir, served_name, port, dtype="bfloat16", max_model_len=16384, max_num_seqs=256,
                  gpu_mem=0.92, quantization=None, multimodal_arch=False):
    """Same flags as the preview."""
    cmd = [str(vllm_bin), "serve", str(model_dir), "--served-model-name", served_name, "--host", "127.0.0.1",
           "--port", str(port), "--dtype", dtype, "--gpu-memory-utilization", str(gpu_mem), "--enable-prefix-caching",
           "--seed", "0", "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_num_seqs)]
    if quantization:
        cmd += ["--quantization", quantization]
    if multimodal_arch:
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0, "video": 0})]
    return cmd


def is_multimodal_arch(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return "ConditionalGeneration" in " ".join(cfg.get("architectures") or [])


def stage(repo, revision, work_dir, log=print):
    """Download repo@revision into work_dir/model (real files, so the normaliser can rewrite them) and normalise."""
    from huggingface_hub import snapshot_download  # optional dependency (present with vLLM)
    dest = Path(work_dir) / "model"
    marker = Path(work_dir) / "staged.json"
    if not marker.exists():
        log(f"downloading {repo}@{revision} ...")
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest))
        log("normalising (text-only causal LM, as the preview does) ...")
        subprocess.run([sys.executable, "-I", str(Path(__file__).resolve().parent / "normalize_model.py"), str(dest)], check=True)
        marker.write_text(json.dumps({"repo": repo, "revision": revision}))
    return dest


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OwnedServer:
    def __init__(self, cmd, workdir, port):
        self.cmd, self.workdir, self.port, self.proc = list(cmd), Path(workdir), int(port), None

    def __enter__(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY")}
        env.update(SERVE_ENV)
        (self.workdir / "serve-command.json").write_text(json.dumps(self.cmd))
        self.log = open(self.workdir / "vllm.log", "a")
        self.proc = subprocess.Popen(self.cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        return self

    def wait_ready(self, timeout=1800):
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode}; see {self.workdir / 'vllm.log'}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"vLLM not healthy after {timeout} s")
            time.sleep(2)

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        self.log.close()
        return False
