"""Looma 的层规格预设（主行与消融行）。"""


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
    """按主行 / 消融行生成层规格。"""
    params = dict(_DEFAULT)
    params.update(knobs)
    return ModuleSpec(module=LoomaTransformerLayer, submodules=PLACEHOLDER_SUBMODULES, params=params)



looma_layer_spec: ModuleSpec = make_looma_spec()

looma_layer_spec_no_loop: ModuleSpec = make_looma_spec(looma_max_iter=1)
looma_layer_spec_iter64: ModuleSpec = make_looma_spec(looma_max_iter=64)
looma_layer_spec_tol1e3: ModuleSpec = make_looma_spec(looma_tol=1.0e-3)

looma_layer_spec_fixed_count: ModuleSpec = make_looma_spec(looma_tol=1.0e-9)
looma_layer_spec_tau05: ModuleSpec = make_looma_spec(looma_tau=0.5)
looma_layer_spec_grad0: ModuleSpec = make_looma_spec(looma_grad_steps=0)

looma_layer_spec_rank16: ModuleSpec = make_looma_spec(looma_rank=16)

looma_layer_spec_rankfull: ModuleSpec = make_looma_spec(looma_rank=1 << 30)
looma_layer_spec_heads1: ModuleSpec = make_looma_spec(looma_read_heads=1)
looma_layer_spec_no_output_route: ModuleSpec = make_looma_spec(looma_output_route=False)
looma_layer_spec_lambda_free: ModuleSpec = make_looma_spec(looma_lambda_clamp=None)

looma_layer_spec_flat_ladder: ModuleSpec = make_looma_spec(looma_decay_tau_max=math.e)

looma_layer_spec_carrier0: ModuleSpec = make_looma_spec(looma_write_carrier_bias=0.0)


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
