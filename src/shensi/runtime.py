"""shensi 与 Megatron-Bridge / verl 的运行时对接点。

只做加法：登记注册表、给上游的一个已知缺陷补一层判断；不往第三方包里写文件、不改它们的源码。
mcore 侧不需要任何补齐：模型、层规格、配置都在 Bridge 的 `models/shensi/` 里，而上游 mcore main
自带 `set_default_log_ranks`、`get_backend`、grouped-mxfp8 那套符号。

调用方式：`import shensi.runtime` 即生效（模块级调用一次）。配方入口显式 import 它；
verl 侧按 verl 自己的约定用 `VERL_USE_EXTERNAL_MODULES=shensi.runtime` 让每个 import verl 的进程
（driver、ray worker、vLLM server）都走一遍。
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# 导入即注册的模块：Bridge 的 HF↔Megatron 桥表、emerging_optimizers 的标量优化器表
REGISTRATIONS: tuple[str, ...] = (
    "megatron.bridge.models.shensi",
    "shensi.utils.optimizer.ademamix",
)

_applied = False


def _install_registrations() -> None:
    for name in REGISTRATIONS:
        importlib.import_module(name)


def _install_mcore_legacy_shims() -> None:
    """补两个 mcore main 上已不存在的模块：verl 的 v012 兼容层在**跑版本守卫之前**就 import 它们
    （`verl/models/mcore/patch.py: apply_patch_megatron_v012_with_torch_v28_v29`），于是装着 mcore main
    时 import verl 会直接 ModuleNotFoundError——守卫写在 import 之后，永远走不到。

    这里按 FL fork 的语义给两个最小实现（`_disable_gc` 就是 gc 开关的 contextmanager，
    `_process_memory` 返回当前进程 RSS）。我们这条线用 mcore main，v012 的补丁本体不会被应用，
    这两个符号只会被 import 到、不会被调用。
    """
    import sys
    import types

    base = "megatron.core.dist_checkpointing.strategies"
    try:
        parent = importlib.import_module(base)
    except Exception:  # noqa: BLE001
        return

    if base + ".async_utils" not in sys.modules:
        import gc
        from contextlib import contextmanager

        @contextmanager
        def _disable_gc():
            """Temporarily disables GC."""
            gc_enabled = gc.isenabled()
            try:
                if gc_enabled:
                    gc.disable()
                yield
            finally:
                if gc_enabled:
                    gc.enable()

        module = types.ModuleType(base + ".async_utils")
        module._disable_gc = _disable_gc
        sys.modules[base + ".async_utils"] = module
        parent.async_utils = module

    if base + ".filesystem_async" not in sys.modules:
        import os
        import types as _types

        def _process_memory() -> int:
            """当前进程的 RSS（字节）。"""
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss

        module = _types.ModuleType(base + ".filesystem_async")
        module._process_memory = _process_memory
        sys.modules[base + ".filesystem_async"] = module
        parent.filesystem_async = module


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
                saved.append(
                    (buffer, ("param_data", None), ("param_data_size", buffer.param_data_size))
                )
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


def _patch_fsdp_symbol_for_verl() -> None:
    """把 mcore 的 FSDP **工厂函数**换成一个类，供这个版本的 verl 做类型判断。

    mcore main 里 `mcore_fsdp_adapter.FullyShardedDataParallel` 是工厂函数（docstring 原话：
    "This is a factory function, not a wrapper type. Use the explicit V1 or V2 implementation
    classes for type checks."），而这份 verl 把它当类用：
    `isinstance(model[0], megatron_FSDP | DDP)`、`ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, ..., _MegatronFSDP, ...)`
    都会 `function | type` / `isinstance() arg 2 must be a type` 报错。工厂的默认版本就是 V1，
    所以把该模块属性指到 `FullyShardedDataParallelV1`——必须在 verl 的
    `verl.utils.megatron_utils` 首次 import 之前做（它的元组是模块级建的）。
    """
    adapter = importlib.import_module("megatron.core.distributed.fsdp.mcore_fsdp_adapter")
    v1 = getattr(adapter, "FullyShardedDataParallelV1", None)
    if v1 is None or isinstance(adapter.FullyShardedDataParallel, type):
        return
    adapter.FullyShardedDataParallel = v1


def _fallback_shensi_dsa_backend() -> None:
    """没装 DSA 融合内核时，把 Bridge provider 的 `dsa_kernel_backend` 默认改成 `none`。

    RL 侧的模型配置由 Bridge 从 HF config 建（verl 的 `use_mbridge`），没有 args 让它走
    训练入口那条回退；而 mcore 给 `dsv4_hybrid` 的默认是 `cudnn`（要 flash_mla），
    构造配置时就抛错。这里包一层 provider 的 `finalize`：值仍是 None 且内核缺失时补上。
    """
    from megatron.bridge.models.shensi.shensi_provider import ShensiModelProvider

    from shensi.utils.dsa import dsa_backend_fallback

    if getattr(ShensiModelProvider.finalize, "_shensi_dsa_fallback", False):
        return
    original = ShensiModelProvider.finalize

    def finalize(self):
        chosen = dsa_backend_fallback(getattr(self, "dsa_kernel_backend", None))
        if chosen != getattr(self, "dsa_kernel_backend", None):
            self.dsa_kernel_backend = chosen
            logger.warning(
                "shensi runtime: 本机没有 flash_mla / nvidia-cudnn-frontend[cutedsl] → "
                "dsa_kernel_backend=none（PyTorch 回退实现，数值同口径、速度慢）"
            )
        return original(self)

    finalize._shensi_dsa_fallback = True
    ShensiModelProvider.finalize = finalize


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
    # 顺序有讲究：
    #  - shims / FSDP 符号都得在任何 verl 导入之前（verl 的类名元组是模块级建的，v012 兼容层在 import 期就取符号）
    #  - no-IPC 平台要在**任何会解析平台名的 verl 导入之前**注册（verl 的 engine 模块一导入就会查 VERL_PLATFORM）
    for step in (
        _install_mcore_legacy_shims,
        _patch_fsdp_symbol_for_verl,
        _register_noipc_platform,
        _install_registrations,
        _fallback_shensi_dsa_backend,
        _patch_verl_flat_buffer_guard,
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
