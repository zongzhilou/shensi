"""shensi 与 mcore / Megatron-Bridge / verl 的运行时对接点。

只做加法：登记模块与注册表、补齐符号、必要时包一层函数；不往第三方包里写文件、不改它们的源码。

调用方式：`import shensi.runtime` 即生效（模块级调用一次）。配方入口显式 import 它；
verl 侧按 verl 自己的约定用 `VERL_USE_EXTERNAL_MODULES=shensi.runtime` 让每个 import verl 的进程
（driver、ray worker、vLLM server）都走一遍。
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

# 第三方按名字 import、FL fork 里没有的模块；值是我们这边的实现
MODULE_ALIASES: dict[str, str] = {
    "megatron.core.transformer.mla_qk_norm_config": "megatron_ext.core.transformer.mla_qk_norm_config",
    "megatron.training.models.gpt": "megatron_ext.training.models.gpt",
}

# 导入即注册的模块（Bridge 的 bridge 表、emerging_optimizers 的标量优化器表）
REGISTRATIONS: tuple[str, ...] = (
    "megatron_ext.bridge.models.shensi",
    "shensi.utils.optimizer.ademamix",
)

_applied = False


def _install_aliases() -> None:
    for name, target in MODULE_ALIASES.items():
        if name in sys.modules:
            continue
        module = importlib.import_module(target)
        parent_name, _, leaf = name.rpartition(".")
        setattr(importlib.import_module(parent_name), leaf, module)
        sys.modules[name] = module


def _install_registrations() -> None:
    for name in REGISTRATIONS:
        importlib.import_module(name)


def _patch_verl_flat_buffer_guard() -> None:
    """Verl 在 `use_distributed_optimizer=False` 时没有 flat param buffer，`load_megatron_model_to_gpu`
    无条件解引用 `param_data`；同一文件里上游自己按 `param_data is None` 判过，这里补同样的判断。"""
    from verl.utils import megatron_utils

    if getattr(megatron_utils, "_shensi_flat_buffer_patch", False):
        return
    original = megatron_utils.load_megatron_model_to_gpu

    def load_megatron_model_to_gpu(models, load_grad=True, load_frozen_params=False):
        saved = []
        for chunk in models:
            buffers = list(getattr(chunk, "buffers", []) or [])
            buffers += list(getattr(chunk, "expert_parallel_buffers", []) or [])
            for buffer in buffers:
                if getattr(buffer, "param_data", None) is not None:
                    continue
                # 没有 flat buffer 就没有要搬的东西：给个零长度占位（原函数还会读
                # param_data.cpu_data，所以要挂在这个张量上），调用完原样还原
                stub = _zeros(0)
                stub.cpu_data = _zeros(0)
                saved.append((buffer, ("param_data", None), ("param_data_size", buffer.param_data_size)))
                buffer.param_data = stub
                buffer.param_data_size = 0
        try:
            return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
        finally:
            for buffer, *restores in saved:
                for name, value in restores:
                    setattr(buffer, name, value)

    megatron_utils.load_megatron_model_to_gpu = load_megatron_model_to_gpu
    megatron_utils._shensi_flat_buffer_patch = True


def _zeros(numel: int):
    import torch

    return torch.zeros(numel)


def _register_noipc_platform() -> None:
    """WSL2 上跨进程 CUDA IPC 不可用；注册一个走共享内存的 CUDA 平台（`VERL_PLATFORM=nvidia_noipc`）。"""
    from verl.plugin.platform.platform_cuda import PlatformCUDA
    from verl.plugin.platform.platform_manager import PlatformRegistry

    try:

        @PlatformRegistry.register(platform="nvidia_noipc")
        class PlatformCUDANoIPC(PlatformCUDA):
            def is_ipc_supported(self) -> bool:
                return False

    except ValueError:  # 已注册
        pass


def setup() -> None:
    """依次做完上面几件事；每步失败只告警，不影响调用方。"""
    global _applied
    if _applied:
        return
    for step in (
        _install_aliases,
        # 补齐必须先于注册：我们的 bridge 模块一进来就会 import Bridge，而 Bridge 要那些符号
        lambda: importlib.import_module("megatron_ext.core.backfill").apply(),
        _install_registrations,
        _patch_verl_flat_buffer_guard,
        _register_noipc_platform,
    ):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "shensi runtime: %s 跳过（%s: %s）",
                getattr(step, "__name__", step),
                type(exc).__name__,
                exc,
            )
    _applied = True


setup()
