"""Local-only HTTP worker; launched exclusively inside the existing Docker sandbox."""
from __future__ import annotations

import argparse
import asyncio
import json
import os

ENV_FAIL = "REAX-JOINT-ENV-FAIL"
MODEL_FAIL = "REAX-JOINT-MODEL-FAIL"
MAX_REQUEST_BYTES = 64 << 20


def environment_check() -> None:
    """Check fixed image dependencies and the sandbox's CUDA access without reading a candidate."""
    import aiohttp
    import accelerate
    import torch
    import tokenizers
    import safetensors.torch
    from transformers import PreTrainedTokenizerFast
    from transformers.masking_utils import create_recurrent_attention_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()


def load_runner(gpu_fraction: float):
    # Only the candidate-independent environment check can report an image fault.
    try:
        environment_check()
    except Exception as exc:
        print(f"{ENV_FAIL}: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(70) from exc
    try:
        from .joint import JointRunner
        return JointRunner.load("/model", gpu_fraction=gpu_fraction)
    except Exception as exc:
        # The fixed environment passed independently; candidate-controlled loading is model work.
        print(f"{MODEL_FAIL}: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(65) from exc


def create_app(runner):
    from aiohttp import web
    lock = asyncio.Lock()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)

    async def health(_request):
        return web.json_response({"status": "ok"})

    async def decide(request):
        rows = await request.json()
        if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
            raise web.HTTPBadRequest()
        async with lock:
            try:
                result = await asyncio.to_thread(runner.decide, rows)
            except Exception:
                result = [None] * len(rows)
        return web.json_response(result, dumps=lambda x: json.dumps(x, allow_nan=False))

    app.router.add_get("/health", health)
    app.router.add_post("/decide", decide)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-fraction", type=float)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        try:
            environment_check()
        except Exception as exc:
            print(f"{ENV_FAIL}: {type(exc).__name__}: {exc}", flush=True)
            raise SystemExit(70) from exc
        return
    runner = load_runner(args.gpu_fraction)
    from aiohttp import web
    os.umask(0)
    web.run_app(create_app(runner), path="/run/vllm/vllm.sock", print=None)


if __name__ == "__main__":
    main()
