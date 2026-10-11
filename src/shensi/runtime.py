"""与 Megatron-Bridge / verl 的运行时对接点（导入即生效）。"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# 导入即注册的模块：Bridge 的 HF↔Megatron 桥表、优化器口径（AdaMuon + AdEMAMix / GrokFastAdamW）
REGISTRATIONS: tuple[str, ...] = (
    "megatron.bridge.models.shensi",
    "shensi.utils.optimizer",
)

_applied = False


def _install_registrations() -> None:
    for name in REGISTRATIONS:
        importlib.import_module(name)


def _install_mcore_legacy_shims() -> None:
    """补两个 mcore main 上已不存在的模块。

    verl 的 v012 兼容层在跑版本守卫**之前**就 import 它们（`verl/models/mcore/patch.py`
    的 `apply_patch_megatron_v012_with_torch_v28_v29`），装着 mcore main 时 import
    verl 会直接 ModuleNotFoundError——守卫写在 import 之后，永远走不到。
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
    """Verl 在 `use_distributed_optimizer=False` 时没有 flat param buffer，
    `load_megatron_model_to_gpu` 无条件解引用 `param_data`；上游在别处按
    `param_data is None` 判过，这里补同样的判断。
    """
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
    """把 mcore 的 FSDP **工厂函数**换成一个类，供这个版本的 verl 做类型判断。"""
    adapter = importlib.import_module("megatron.core.distributed.fsdp.mcore_fsdp_adapter")
    v1 = getattr(adapter, "FullyShardedDataParallelV1", None)
    if v1 is None or isinstance(adapter.FullyShardedDataParallel, type):
        return
    adapter.FullyShardedDataParallel = v1


def _fallback_shensi_dsa_backend() -> None:
    """没装 DSA 融合内核时，把 Bridge provider 的 `dsa_kernel_backend` 默认改成 `none`。"""
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
    from shensi.utils.accelerator import backend_name

    if backend_name() == "npu":
        return
    from verl.plugin.platform.platform_cuda import PlatformCUDA
    from verl.plugin.platform.platform_manager import PlatformRegistry

    try:

        @PlatformRegistry.register(platform="nvidia_noipc")
        class PlatformCUDANoIPC(PlatformCUDA):
            def is_ipc_supported(self) -> bool:
                return False

    except ValueError:  # 已注册
        pass


def _patch_verl_muon_algorithms() -> None:
    """让 verl 把 Muon 家族的旋钮也透传给 AdaMuon。"""
    from verl.utils.megatron import optimizer as verl_optimizer

    algorithms = getattr(verl_optimizer, "_MUON_ALGORITHMS", None)
    if algorithms is None or "adaptive_muon" in algorithms:
        return
    verl_optimizer._MUON_ALGORITHMS = (*algorithms, "adaptive_muon")


# aclnnGroupedMatmulV5 的 tensorList 长度上限（torch_npu 在调用前校验，超出报 EZ1001/161002）。
# 这是**单次调用**的限制，不是单卡容量：把本卡专家按 ≤128 个一组分多次调用，数学完全等价。
GROUPED_MATMUL_LIMIT = 128


def _grouped_matmul_counts(group_list, group_list_type, rows: int):
    """把 group_list 归一成每组的行数；格式不能确定时返回 None（调用方退回原调用）。"""
    gl = group_list.tolist()
    if group_list_type == 1:
        counts = [int(c) for c in gl]
    else:
        # cumulative（默认）：每项是该组的**结束行号**，末项 = 总行数
        if not gl or int(gl[-1]) != rows:
            # 不是 cumulative；若各项之和恰为总行数则按 count 解
            counts = [int(c) for c in gl] if sum(int(c) for c in gl) == rows else None
        else:
            counts = [int(gl[0])] + [int(gl[i] - gl[i - 1]) for i in range(1, len(gl))]
    if counts is not None and sum(counts) != rows:
        return None
    return counts


def _chunked_grouped_matmul(original, args, kwargs):
    import torch

    """把一次 >128 组的调用拆成 ≤128 组的多次调用，输出按行拼回。

    只支持 MoE 的行切分组（group_type 未给或为 0）：x 按组沿 dim0 切，权重按组取切片，
    group_list 按原格式（cumulative/count）重发。其余形态一律原样透传。
    """
    a = list(args)
    x = kwargs.get("x", a[0] if len(a) > 0 else None)
    weight = kwargs.get("weight", a[1] if len(a) > 1 else None)
    group_list = kwargs.get("group_list", a[2] if len(a) > 2 else None)
    if not isinstance(x, (list, tuple)) or not isinstance(weight, (list, tuple)):
        return None
    n_groups = len(weight)
    if n_groups <= GROUPED_MATMUL_LIMIT or group_list is None:
        return None
    group_type = kwargs.get("group_type")
    if group_type not in (None, 0):
        return None
    rows = int(x[0].shape[0])
    counts = _grouped_matmul_counts(group_list, kwargs.get("group_list_type"), rows)
    if counts is None or len(counts) != n_groups:
        return None
    if rows == 0:  # 没有任何 token：留给原调用按它自己的口径处理
        return None

    cumulative = kwargs.get("group_list_type", 0) != 1
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    is_tensor_x = torch.is_tensor(x)
    x_list = [x] if is_tensor_x else list(x)

    outs: list = [[] for _ in x_list]
    pending_empty: list = []
    for start in range(0, n_groups, GROUPED_MATMUL_LIMIT):
        end = min(start + GROUPED_MATMUL_LIMIT, n_groups)
        r0, r1 = prefix[start], prefix[end]
        if r1 == r0:  # 整块 0 token：留空位，输出形状等第一个非空块出来再补
            pending_empty.append(len(outs[0]))
            for slot in outs:
                slot.append(None)
            continue
        if cumulative:
            chunk_gl = torch.tensor(
                [prefix[start + i + 1] - r0 for i in range(end - start)],
                dtype=group_list.dtype,
                device=group_list.device,
            )
        else:
            chunk_gl = group_list[start:end]
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["group_list"] = chunk_gl
        chunk_out = original([t[r0:r1] for t in x_list], list(weight[start:end]), *a[3:], **chunk_kwargs)
        for slot, piece in zip(outs, chunk_out):
            slot.append(piece)

    n_cols = next(p.shape[1] for p in outs[0] if p is not None)
    merged = [
        torch.cat([p if p is not None else p0.new_zeros((0, n_cols)) for p in slot], dim=0)
        for slot, p0 in zip(outs, x_list)
    ]
    return merged[0] if is_tensor_x else merged


def _assert_grouped_gemm_fits(*, moe_grouped_gemm, num_moe_experts, expert_model_parallel_size) -> None:
    """本卡专家数 >128 且分块 shim 没装上时直接报错（NPU 口径留在本仓，Bridge 保持上游干净）。"""
    if not moe_grouped_gemm:
        return
    import sys

    torch_npu = sys.modules.get("torch_npu")
    if torch_npu is None:  # 非 NPU 机器没有这个上限
        return
    experts = int(num_moe_experts or 0)
    ep = max(1, int(expert_model_parallel_size or 1))
    if experts // ep > 128 and not getattr(torch_npu, "_shensi_grouped_matmul_chunk", False):
        raise ValueError(
            f"moe_grouped_gemm 会把本卡 {experts // ep} 个专家塞进一次 npu_grouped_matmul，而昇腾"
            "算子单次最多 128 个张量。shensi.runtime 导入时会自动装好分块 shim"
            "（torch_npu._shensi_grouped_matmul_chunk 标记）；它不在说明 shim 安装失败了——"
            f"查 runtime 日志，或把 expert_model_parallel_size 提到 {-(-experts // 128)}。"
        )


def _check_grouped_gemm_limit_on_finalize() -> None:
    """provider finalize 前兜底校验本卡专家数（分块 shim 之后装，标记检查才有意义）。

    Bridge 是 NVIDIA 维护的库，NPU 相关判断留在本仓：包一层 `ShensiModelProvider.finalize`，
    把「shim 缺位 + 本卡专家 >128」拦在第一个 MoE forward 之前。
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return
    from megatron.bridge.models.shensi.shensi_provider import ShensiModelProvider

    if getattr(ShensiModelProvider.finalize, "_shensi_grouped_gemm_check", False):
        return
    original = ShensiModelProvider.finalize

    def finalize(self):
        _assert_grouped_gemm_fits(
            moe_grouped_gemm=bool(getattr(self, "moe_grouped_gemm", False)),
            num_moe_experts=getattr(self, "num_moe_experts", None),
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1),
        )
        return original(self)

    finalize._shensi_grouped_gemm_check = True
    ShensiModelProvider.finalize = finalize


def _patch_npu_grouped_matmul_chunk128() -> None:
    """昇腾的 grouped GEMM 一次最多吃 128 组：本卡专家更多时自动按 128 分块多次调用。

    MoE 走 grouped GEMM 时 TE-NPU 把本卡**全部**专家塞进一次 `npu_grouped_matmul`，EP=1 的
    256 专家直接超限（EP=2 恰好卡在上限）。分块调用数学等价、省掉 EP≥2 的两次 all-to-all
    和 DP 变小带来的优化器状态损失。CUDA 机器上没有 torch_npu，静默跳过。
    """
    import sys

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return
    if getattr(torch_npu, "_shensi_grouped_matmul_chunk", False):
        return
    original = torch_npu.npu_grouped_matmul

    def npu_grouped_matmul(*args, **kwargs):
        if len(args) > 1 and isinstance(args[1], (list, tuple)) and len(args[1]) > GROUPED_MATMUL_LIMIT:
            merged = _chunked_grouped_matmul(original, args, kwargs)
            if merged is not None:
                return merged
        return original(*args, **kwargs)

    npu_grouped_matmul._shensi_original = original
    torch_npu.npu_grouped_matmul = npu_grouped_matmul
    torch_npu._shensi_grouped_matmul_chunk = True
    # TE-NPU 若在导入期 `from torch_npu import npu_grouped_matmul` 拿走的是直引用，
    # 已导入的 transformer_engine 模块里的同名绑定一并换掉。
    for name, module in list(sys.modules.items()):
        if "transformer_engine" in name and getattr(module, "npu_grouped_matmul", None) is original:
            module.npu_grouped_matmul = npu_grouped_matmul
    logger.info(
        "shensi runtime: npu_grouped_matmul 已加 %d 组分块（本卡专家数可 > %d，EP=1 保持不变）",
        GROUPED_MATMUL_LIMIT,
        GROUPED_MATMUL_LIMIT,
    )


def setup() -> None:
    """依次做完上面几件事；每步失败只告警，不影响调用方。"""
    global _applied
    if _applied:
        return
    from shensi.utils.accelerator import bootstrap_megatron

    bootstrap_megatron()
    # 顺序有讲究：
    #  - shims / FSDP 符号都得在任何 verl 导入之前（verl 的类名元组是模块级建的，v012 兼容层在 import 期就取符号）
    #  - no-IPC 平台要在**任何会解析平台名的 verl 导入之前**注册（verl 的 engine 模块一导入就会查 VERL_PLATFORM）
    #  - grouped-matmul 分块要在任何 TE/MoE 前向**调用**之前挂上（改的是 torch_npu 的模块属性，
    #    晚于 TE 导入也没事，但要在第一次前向前）
    for step in (
        _install_mcore_legacy_shims,
        _patch_fsdp_symbol_for_verl,
        _register_noipc_platform,
        _install_registrations,
        _patch_npu_grouped_matmul_chunk128,
        _check_grouped_gemm_limit_on_finalize,
        _fallback_shensi_dsa_backend,
        _patch_verl_flat_buffer_guard,
        _patch_verl_muon_algorithms,
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
