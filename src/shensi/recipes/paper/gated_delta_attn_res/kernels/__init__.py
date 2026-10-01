"""白化的内核侧实现（B6）：Triton 协方差 + Newton–Schulz + 融合读。

用法（训练入口的开关）：

    python train.py --config ... --set train.system.gdar_whiten_impl=fused        # 融合读（等价，推荐先试）
    python train.py --config ... --set train.system.gdar_whiten_impl=fused_apply  # 连 @W 也进 kernel
    python train.py --config ... --set train.system.gdar_whiten_impl=per_head # 逐头档换实现（上游默认档）
    python train.py --config ... --set train.system.gdar_whiten_impl=ns      # 只换 full 档的白化（有精度包线！）
    python train.py --config ... --set train.system.gdar_whiten_impl=triton  # ns + Triton 协方差

等价性与计时：`test_whiten.py`、`test_whiten_extra.py`、`bench_whiten*.py`。
**开 ns/triton 前先读 `kernels/README.md` 的"精度包线"**：full 档的协方差条件数大，NS 达不到
等价精度（残差会当场报出来，不静默）。
"""

from __future__ import annotations

from . import whiten_ns, whiten_triton

#: 可用后端；`fused`/`per_head` 是等价实现，`ns`/`triton` 只在良态数据上等价（见 README 包线）
IMPLS = ("eager", "fused", "fused_apply", "per_head", "ns", "triton")

__all__ = ["IMPLS", "enable", "whiten_ns", "whiten_triton"]


def enable(impl: str) -> str:
    """把白化/读换成 `impl`；返回实际生效的实现名。"""
    if impl not in IMPLS:
        raise ValueError(f"whiten impl must be one of {IMPLS}, got {impl!r}")
    from shensi.recipes.paper.gated_delta_attn_res.models.megatron import gdar_connection

    from . import whiten_fused, whiten_per_head

    whiten_ns.uninstall()
    whiten_per_head.uninstall()
    whiten_fused.uninstall()
    if impl == "eager":
        return "eager"
    if impl == "fused":
        whiten_fused.install("eager_whiten")  # 白化矩阵与参考同源，只融合读
        return "fused"
    if impl == "fused_apply":
        whiten_fused.install("eager_whiten", fuse_apply=True)  # 连 values/query @ W 也进 kernel
        return "fused_apply"
    if impl == "per_head":
        whiten_per_head.install()
        return "per_head"
    if impl == "ns":
        whiten_ns.install()
        print(
            "[gdar][whiten] ns 档只在良态协方差上等价（full 档残差不达容差会正在训练里报出来）",
            flush=True,
        )
        return "ns"

    def _triton_transform(values, mode, ridge):
        return whiten_triton.whitening_transform_triton(values, mode, ridge)

    whiten_ns.install()
    gdar_connection._whitening_transform = _triton_transform
    print("[gdar][whiten] triton 档同上：full 档有精度包线，别拿它出论文数", flush=True)
    return "triton"
