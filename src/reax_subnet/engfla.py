"""Fail closed on a missing FLA config in the pinned competition reference runtime."""

import os


def require_fla_configs() -> None:
    """FLA 0.5.2 strict mode itself still autotunes on a miss; forbid that fallback.

    Serving keeps its existing behaviour unless strict mode is explicitly requested.
    This runs before JevK5 construction, covering CUDA graph capture and later shapes.
    """
    if os.environ.get("FLA_CACHE_MODE") != "strict":
        return
    from fla.ops.utils import cache

    load = cache.load_cached_config

    def required(kernel_name, autotune_key=None):
        config = load(kernel_name, autotune_key)
        if config is None:
            raise RuntimeError(f"unpublished FLA config: {kernel_name}; reference session must be VOID")
        return config

    cache.load_cached_config = required
