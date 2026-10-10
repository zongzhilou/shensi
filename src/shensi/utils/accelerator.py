"""Explicit device selection and early Megatron bootstrap for CUDA and Ascend.

SHENSI_BACKEND=auto|cuda|npu|cpu; explicit NPU requests never fall back to CPU.
Importing this module does not import Megatron or apply vendor patches.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Mapping
from contextlib import contextmanager


def backend_name(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    selected = env.get("SHENSI_BACKEND", "auto").lower()
    selected = {"ascend": "npu", "nvidia": "cuda"}.get(selected, selected)
    if selected not in ("auto", "cuda", "npu", "cpu"):
        raise ValueError(f"Unknown SHENSI_BACKEND={selected!r}; use auto, cuda, npu or cpu")
    if selected != "auto":
        return selected
    # A sourced CANN environment must fail visibly if torch_npu is missing/broken.
    if env.get("ASCEND_HOME_PATH") or importlib.util.find_spec("torch_npu") is not None:
        return "npu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def device(backend: str | None = None, index: int | None = None):
    import torch

    backend = backend or backend_name()
    if backend == "npu":
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is False")
    elif backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if backend == "cpu":
        return torch.device("cpu")
    index = int(os.environ.get("LOCAL_RANK", "0")) if index is None else index
    api = getattr(torch, backend)
    api.set_device(index)
    return torch.device(backend, index)


_megatron_ready = False


def bootstrap_megatron() -> None:
    """Apply the vendor adaptor BEFORE the first megatron/TE import; fail closed."""
    global _megatron_ready
    if _megatron_ready or backend_name() != "npu":
        return
    device("npu")
    # MegatronAdaptor itself installs transfer_to_npu and optional MindSpeed patches.
    importlib.import_module("megatron_adaptor")
    _megatron_ready = True


def visible_devices(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    backend = backend_name(env)
    keys = (
        ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")
        if backend == "npu"
        else ("CUDA_VISIBLE_DEVICES",)
    )
    for key in keys:
        if key in env:
            return max(1, len([x for x in env[key].split(",") if x.strip() and x != "-1"]))
    if backend == "cpu":
        return 1
    import torch

    if backend == "npu":
        if importlib.util.find_spec("torch_npu") is None:
            return 1  # Permit command construction on a development host.
        importlib.import_module("torch_npu")
    return max(1, getattr(torch, backend).device_count())


def configure_verl_env(env: dict[str, str]) -> dict[str, str]:
    """Choose the vendor platform without disabling Ascend's vLLM plugins."""
    if backend_name(env) == "npu":
        if env.get("VERL_PLATFORM") not in (None, "huawei", "ascend"):
            raise ValueError(
                "NPU requires VERL_PLATFORM=huawei or ascend; remove the CUDA platform override"
            )
        env.setdefault("VERL_PLATFORM", "huawei")
        env["SHENSI_BACKEND"] = "npu"
        # An empty CUDA-side override disables every vLLM plugin, including
        # vllm-ascend and this recipe's model registration entry point.
        if env.get("VLLM_PLUGINS") == "":
            env.pop("VLLM_PLUGINS")
    else:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("VERL_PLATFORM", "nvidia_noipc")
        env.setdefault("VLLM_PLUGINS", "")
    return env


def extend_distributed_backend_choices(parser):
    """Mcore's generated CLI permits only nccl/gloo; expose HCCL on Ascend."""
    if backend_name() == "npu":
        for action in parser._actions:  # noqa: SLF001 - argparse has no public action lookup
            if (
                action.dest == "distributed_backend"
                and action.choices is not None
                and "hccl" not in action.choices
            ):
                action.choices = [*action.choices, "hccl"]
    return parser


@contextmanager
def isolated_rng(seed: int):
    """Preserve CPU and the selected accelerator's RNG during layer initialization."""
    import torch

    backend = backend_name()
    if backend == "npu":
        importlib.import_module("torch_npu")
    api = getattr(torch, backend, None)
    devices = list(range(api.device_count())) if backend != "cpu" else []
    kwargs = {"device_type": backend} if backend != "cpu" else {}
    with torch.random.fork_rng(devices=devices, **kwargs):
        torch.manual_seed(int(seed) % (2**31 - 1))
        if devices:
            api.manual_seed_all(int(seed) % (2**31 - 1))
        yield
