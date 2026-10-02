"""Make the rollout-side registration reach *engine worker* processes.

Why this file exists
--------------------
vLLM v1 does the actual modelling in a **separate process** (``EngineCore``, and for
multi-GPU one per rank), not in the process that runs your script.  Registering an
architecture on the driver is therefore not enough -- the worker's
``ModelRegistry`` is still empty and it fails with ``Model architecture X is not
supported``.  The same trap was hit on the training side of this repo (see
``stage2_rl 侧的注册（gdar_package 的 code/verl_plugin）``): a registration that never leaves the driver
is the single most common way to get this wrong.

Python imports ``sitecustomize`` at interpreter start-up, and both ``vLLM`` and
``multiprocessing(spawn)`` children inherit the parent's environment, so putting this
directory on ``PYTHONPATH`` is enough:

    export ROLLOUT_PLUGIN_AUTOLOAD=1
    export PYTHONPATH=<repo>/src

Deliberately opt-in: without ``ROLLOUT_PLUGIN_AUTOLOAD`` this file does nothing, so
it is harmless to leave it on the path.  Everything is wrapped in ``try/except`` so a
broken plugin can never stop an interpreter from starting.
"""

from __future__ import annotations

import os

_ENV = "ROLLOUT_PLUGIN_AUTOLOAD"


def _install() -> None:
    if os.environ.get(_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        from .register_model import register_all

        report = register_all(verbose=bool(int(os.environ.get("ROLLOUT_PLUGIN_VERBOSE", "1"))))
        if os.environ.get("ROLLOUT_PLUGIN_VERBOSE", "1") not in {"0", ""}:
            print(
                f"[shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.sitecustomize] pid={os.getpid()} registered "
                f"{len(report['registered'])} architectures via {report['impl']}",
                flush=True,
            )
    except Exception as exc:  # pragma: no cover - must never break interpreter start
        if os.environ.get("ROLLOUT_PLUGIN_VERBOSE", "1") not in {"0", ""}:
            print(
                f"[shensi.recipes.paper.gated_delta_attn_res.common.models.vllm.sitecustomize] pid={os.getpid()} registration FAILED: {exc!r}",
                flush=True,
            )


_install()
