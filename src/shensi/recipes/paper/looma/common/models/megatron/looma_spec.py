"""Looma 的层 spec 预设，经 Megatron 官方 ``--spec`` 扩展点装载（``<模块> <对象名>``）。

``--spec`` 只 import 该对象并原样取用、不调用它，故每个预设都是模块级的 ``ModuleSpec``，Looma
旋钮放在 ``params`` 里，由 ``LoomaTransformerLayer`` 在 ``__init__`` 中读取。
"""

from __future__ import annotations

import math

from megatron.core.transformer.spec_utils import ModuleSpec

from .looma_layer import PLACEHOLDER_SUBMODULES, LoomaTransformerLayer, build_looma_submodules

__all__ = [
    "DESIGN_ABLATIONS",
    "looma_layer_spec",
    "looma_layer_spec_carrier0",
    "looma_layer_spec_fixed_count",
    "looma_layer_spec_flat_ladder",
    "looma_layer_spec_grad0",
    "looma_layer_spec_heads1",
    "looma_layer_spec_iter64",
    "looma_layer_spec_lambda_free",
    "looma_layer_spec_no_loop",
    "looma_layer_spec_no_output_route",
    "looma_layer_spec_rank16",
    "looma_layer_spec_rankfull",
    "looma_layer_spec_tau05",
    "looma_layer_spec_tol1e3",
    "make_looma_spec",
]

_DEFAULT = dict(
    looma_max_iter=8,
    looma_tol=1.0e-2,
    looma_stop_mode="rel",
    looma_tau=1.0,
    looma_grad_steps=1,
    looma_rank=64,
    looma_read_heads=8,
    looma_lambda_clamp=-0.5,
    looma_write_carrier_bias=-4.0,
)


def make_looma_spec(**knobs) -> ModuleSpec:
    """按给定的 ``looma_*`` 旋钮构造 layer spec，未给的取 ``_DEFAULT``。

    submodules 放 ``PLACEHOLDER_SUBMODULES``：模块级 spec 拿不到 config，只能带一份占位实例，
    由层在构建时按真 config 重建出与原生构建器逐字相同的子模块。
    """
    params = dict(_DEFAULT)
    params.update(knobs)
    return ModuleSpec(module=LoomaTransformerLayer, submodules=PLACEHOLDER_SUBMODULES, params=params)


# 主行：默认旋钮（8 次迭代 / 相对残差 1e-2 / 一步 phantom 梯度 + 连接族默认值）
looma_layer_spec: ModuleSpec = make_looma_spec()

looma_layer_spec_no_loop: ModuleSpec = make_looma_spec(looma_max_iter=1)
looma_layer_spec_iter64: ModuleSpec = make_looma_spec(looma_max_iter=64)
looma_layer_spec_tol1e3: ModuleSpec = make_looma_spec(looma_tol=1.0e-3)
# 容差压到不可能触及，循环恒跑满 max_iter：停止点不随批量统计变化，便于精确复现
looma_layer_spec_fixed_count: ModuleSpec = make_looma_spec(looma_tol=1.0e-9)
looma_layer_spec_tau05: ModuleSpec = make_looma_spec(looma_tau=0.5)
looma_layer_spec_grad0: ModuleSpec = make_looma_spec(looma_grad_steps=0)

looma_layer_spec_rank16: ModuleSpec = make_looma_spec(looma_rank=16)
# 全秩：rank 取得远大于 hidden，构造时被夹到 hidden
looma_layer_spec_rankfull: ModuleSpec = make_looma_spec(looma_rank=1 << 30)
looma_layer_spec_heads1: ModuleSpec = make_looma_spec(looma_read_heads=1)
looma_layer_spec_no_output_route: ModuleSpec = make_looma_spec(looma_output_route=False)
looma_layer_spec_lambda_free: ModuleSpec = make_looma_spec(looma_lambda_clamp=None)
# 阶梯顶端取 e：decay_tau = linspace(0, 1, H)，多时间尺度退化成一条斜坡
looma_layer_spec_flat_ladder: ModuleSpec = make_looma_spec(looma_decay_tau_max=math.e)
# 写载体偏置归零的对照：tanh(0) = 0，写门尺度在初值处梯度为零而不再更新
looma_layer_spec_carrier0: ModuleSpec = make_looma_spec(looma_write_carrier_bias=0.0)

# 设计矩阵：``EXPERIMENT_MATRIX`` 里 Looma 的每一行 → spec 预设；每行只与主行差一处
DESIGN_ABLATIONS: dict[str, ModuleSpec] = {
    "main": looma_layer_spec,
    "max_iter=1": looma_layer_spec_no_loop,
    "max_iter=64": looma_layer_spec_iter64,
    "tol=1e-3": looma_layer_spec_tol1e3,
    "tol=fixed": looma_layer_spec_fixed_count,
    "tau=0.5": looma_layer_spec_tau05,
    "grad_steps=0": looma_layer_spec_grad0,
    "rank=64": looma_layer_spec,
    "rank=16": looma_layer_spec_rank16,
    "rank=full": looma_layer_spec_rankfull,
    "read_heads=8": looma_layer_spec,
    "read_heads=1": looma_layer_spec_heads1,
    "output_route=on": looma_layer_spec,
    "output_route=off": looma_layer_spec_no_output_route,
    "lambda_clamp=-0.5": looma_layer_spec,
    "lambda_clamp=None": looma_layer_spec_lambda_free,
    "ladder=2L": looma_layer_spec,
    "ladder=flat": looma_layer_spec_flat_ladder,
    "carrier_bias=-4": looma_layer_spec,
    "carrier_bias=0": looma_layer_spec_carrier0,
}
