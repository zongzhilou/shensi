"""vLLM worker 的 sitecustomize：把插件路径注入子进程。"""


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
