"""Joint readout in the same pinned image, resource limits and lifecycle as causal entries."""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace
import shutil
from typing import Any, Mapping, Sequence

from . import vllm as V
from .joint_server import ENV_FAIL, MODEL_FAIL

class JointModelUnavailable(V.ModelUnavailable):
    """Candidate load failure after the independent joint image preflight passed."""


_PREFLIGHT_OK: set[tuple[str, str, str]] = set()


def build_joint_command(spec: V.ServeSpec) -> list[str]:
    command = V.build_docker_command(spec)
    # Preserve every sandbox flag; replace only the reviewed process started after the pinned image.
    image_index = command.index(spec.image)
    return command[:image_index] + ["-e", "PYTHONPATH=/aux", spec.image, "-c",
        'umask 000; exec python3 -m reax_subnet.spec.joint_server "$@"', "--",
        "--gpu-fraction", str(spec.gpu_util)]


class JointDockerBackend(V.VllmDockerBackend):
    async def __aenter__(self) -> JointDockerBackend:
        memory_bytes = int(V.parse_container_mem(self.memory)[:-1]) * V.GIB
        if memory_bytes < self.weight_bytes + max(4 * V.GIB, self.weight_bytes // 10):
            raise V.HostUnavailable("joint container memory is below weights plus load margin")
        await V.check_gpu_free(self.gpu_device, self.required_gib)
        self._gpu_preflight_ok = True
        self.run_dir, self.sock_dir, aux_dir = V.prepare_run_dir(
            self.work_dir, template="qwen.jinja", disable_uva=False)
        try:
            # Only our installed package code is mounted; never any source from the candidate repository.
            source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            shutil.copytree(source, os.path.join(aux_dir, "reax_subnet"),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for root, dirs, files in os.walk(aux_dir):
                os.chmod(root, 0o755)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o644)
            spec = V.ServeSpec(self.name, self.model_dir, self.sock_dir, self.util, self.gpu_device, self.image,
                               aux_dir=aux_dir, template="qwen.jinja", user=self.user, runtime=self.runtime,
                               cpus=self.cpus, memory=self.memory)
            await self._image_preflight(spec)
            code, out = await V._run(*build_joint_command(spec), timeout=120)
            if code:
                raise V.HostUnavailable(f"docker run failed: {out[-300:]}")
            import aiohttp
            self._http = aiohttp.ClientSession(
                connector=aiohttp.UnixConnector(path=os.path.join(self.sock_dir, "vllm.sock")),
                timeout=aiohttp.ClientTimeout(total=V.REQUEST_TIMEOUT_S))
            await self._wait_healthy()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def _image_preflight(self, spec: V.ServeSpec) -> None:
        # The empty runtime selects Docker's default, exactly as for causal entries.
        # Never reuse a probe across configured runtimes, images or GPU devices.
        key = (spec.image, spec.runtime, spec.gpu_device)
        if key in _PREFLIGHT_OK:
            return
        probe = replace(spec, name=spec.name + "-preflight")
        command = build_joint_command(probe)
        # Foreground probe, same sandbox/GPU/image as serving, no candidate load.
        command.remove("-d")
        command = command[:-2] + ["--preflight"]
        try:
            code, out = await V._run(*command, timeout=120)
            if code:
                raise V.HostUnavailable(f"joint image preflight failed: {out[-400:]}")
            _PREFLIGHT_OK.add(key)
        finally:
            await V._run("docker", "rm", "-f", probe.name, timeout=60)

    async def _wait_healthy(self) -> None:
        try:
            await super()._wait_healthy()
        except V.HostUnavailable:
            raise  # Docker inspection/logging and socket faults retain their host attribution.
        except V.ModelUnavailable as exc:
            code, state = await V._run("docker", "inspect", "-f", "{{.State.OOMKilled}}", self.name, timeout=20)
            if code:
                raise V.HostUnavailable("joint container state unavailable") from exc
            # __aenter__ checked container memory against weights plus the load margin.
            if state.strip() == "true":
                raise V.ModelBudgetOOM("joint model OOM-killed within the checked container memory budget") from exc
            code, logs = await V._run("docker", "logs", "--tail", str(V.STARTUP_LOG_TAIL), self.name, timeout=20)
            if code or (ENV_FAIL in logs and MODEL_FAIL not in logs):
                raise V.HostUnavailable(f"joint image/environment startup failure: {logs[-400:]}") from exc
            if self._gpu_preflight_ok and "CUDA out of memory" in logs:
                raise V.ModelBudgetOOM(f"joint model exceeded the checked GPU budget: {logs[-400:]}") from exc
            # A missing marker after a successful probe is also a failed candidate load.
            raise JointModelUnavailable(f"invalid weights (joint readout): {logs[-400:]}") from exc

    async def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]:
        assert self._http is not None
        import aiohttp
        results = []
        # At most four serial prefills per 120s request; a long/bad group fails only its items.
        for start in range(0, len(rows), 4):
            group = list(rows[start:start + 4])
            try:
                async with self._http.post("http://vllm/decide", json=group) as response:
                    if response.status != 200:
                        results.extend([None] * len(group))
                        continue
                    answer = await V._read_json(response)
                if not isinstance(answer, list) or len(answer) != len(group):
                    answer = [None] * len(group)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, ValueError):
                answer = [None] * len(group)
            results.extend(answer)
        return results
