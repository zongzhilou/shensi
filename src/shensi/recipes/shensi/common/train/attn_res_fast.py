"""AttnRes 读路径的训练侧加速（不改模型数学，只改执行方式）。

读路径是这套几何的训练瓶颈：在真实几何模型（当时为 mHC 多流版，2.78B/1.24B）上把读换成恒等再测同一前向，
读占前向 65–70%（128 / 2048 token 两档），且访存受限——单模块 profiler 里
157 次逐元素 kernel 占 55.6%。两条加速都只动执行方式：

* ``compile``：把读模块的 forward 交给 ``torch.compile(dynamic=True)``。
  实测（生产几何、M=32768 位置、nb=10）：46.4 → 24.6 ms（1.89×）；
  算术层面 ``num_blocks=0``（不做采样）时 eager 与编译输出**逐位一致**，
  差别只来自采样序列（参考实现本来就每次读重新抽签，属于同一估计器的另一次实现）。
* ``high``（TF32，可选）：fp32 matmul 精度降到 10 位尾数。单测同上 1.16×、
  与编译叠加 2.59×。但门链里的 ``exp(softplus(·))`` 会把尾数误差放大到 ~1e-2
  相对量级（与 bf16 门同量级），属于**精度**改动，上线前要用 pilot_lr/loss_gate
  的框架做短 A/B，故默认保持 ``highest``。

两条都只在 CUDA 上生效；NPU 侧 torch_npu 的 inductor 支持弱，等价手段是
CANN 图模式/融合算子（见 stage README 的吞吐一节）。
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["apply_from_args", "enable_attn_res_fast_path"]

_MARK = "_shensi_attn_res_fast_path"


def _read_modules(chunks: Iterable) -> list:
    """收集所有 AttnRes 读模块（mcore 侧的 ``ShensiAttentionResidual``）。"""
    from megatron.bridge.models.shensi.modeling_shensi import ShensiAttentionResidual

    found = []
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, ShensiAttentionResidual):
                found.append(module)
    return found


def _rank0(log):
    """只在 rank0 打印（hook 在每个 rank 上都会跑）。"""
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return lambda *a, **k: None
    except Exception:  # noqa: BLE001 - 拿不到分布式状态就照常打印
        pass
    return log


def enable_attn_res_fast_path(
    chunks: Iterable,
    *,
    mode: str = "compile",
    matmul_precision: str | None = None,
    log=print,
) -> dict:
    """对一个模型 chunk 列表启用读路径加速；幂等，返回生效摘要。

    Args:
        chunks: ``pre_wrap_hook`` 拿到的模型 chunk 列表。
        mode: ``compile``（默认）或 ``none``。
        matmul_precision: 给了就设 ``torch.set_float32_matmul_precision``
            （``highest`` 保持 fp32 对齐；``high`` = TF32，需 A/B）。
        log: 日志回调（内部只让 rank0 说话）。

    """
    log = _rank0(log)
    summary = {"mode": mode, "matmul_precision": None, "compiled": 0, "skipped": None}
    if matmul_precision:
        current = torch.get_float32_matmul_precision()
        if matmul_precision != current:
            torch.set_float32_matmul_precision(matmul_precision)
        summary["matmul_precision"] = torch.get_float32_matmul_precision()

    if mode == "none":
        return summary
    if mode != "compile":
        raise ValueError(f"未知的读路径加速档：{mode!r}（可选 none / compile）")
    if not torch.cuda.is_available():
        summary["skipped"] = "no-cuda"
        log("[shensi][attn_res_fast] 跳过 compile：没有 CUDA（NPU 侧用 CANN 图模式/融合算子）")
        return summary

    modules = _read_modules(chunks)
    for module in modules:
        if getattr(module, _MARK, False):
            continue
        # 包 forward 而不是整模块：state_dict / 检查点键完全不变；
        # dynamic=True 免去按序列长度重编译，fullgraph=False 允许采样等 Python 侧断点。
        module.forward = torch.compile(module.forward, dynamic=True, fullgraph=False)
        setattr(module, _MARK, True)
        summary["compiled"] += 1
    if summary["compiled"]:
        log(
            f"[shensi][attn_res_fast] 读路径 compile 生效：{summary['compiled']} 个模块"
            f"（dynamic=True；算术逐位一致，采样序列为另一次实现）"
        )
    return summary


def apply_from_args(model, args) -> dict:
    """按 CLI 开关给一个刚构建好的模型 chunk 启用读路径加速。

    由 `builders.ShensiModelBuilder.build_model` 在"构建后、DDP 包装前"调用——每个 rank 都会走到，
    幂等（重复调用只算一次）。
    """
    mode = getattr(args, "shensi_attn_res_fast_path", "none") or "none"
    precision = getattr(args, "shensi_attn_res_matmul_precision", None) or None
    if mode == "none" and precision in (None, "highest"):
        return {"mode": "none", "matmul_precision": None, "compiled": 0, "skipped": "disabled"}
    return enable_attn_res_fast_path([model], mode=mode, matmul_precision=precision)
