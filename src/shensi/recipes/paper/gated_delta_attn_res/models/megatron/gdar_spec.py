"""GDAR 的层规格预设：论文主行、形态扫描、低秩/门结构/初始化等设计矩阵行。"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .gdar_layer import GdarTransformerLayer

__all__ = [
    "gdar_layer_spec_paper",
    "gdar_layer_spec_paper_sublayer",
    "gdar_layer_spec_paper_b2",
    "gdar_layer_spec_paper_b4",
    "gdar_layer_spec_paper_b8",
    "gdar_layer_spec_paper_b16",
    "gdar_layer_spec_paper_rank16",
    "gdar_layer_spec_paper_rankfull",
    "gdar_layer_spec_paper_decay_free",
    "gdar_layer_spec_paper_lambda_free",
    "gdar_layer_spec_paper_ladder0",
    "gdar_layer_spec_paper_gate_prefix",
    "gdar_layer_spec_paper_gate_delta",
    "gdar_layer_spec_paper_address_state",
    "gdar_layer_spec_paper_address_novelty",
    "gdar_layer_spec_paper_update_reference",
    "gdar_layer_spec_paper_heads1",
    "gdar_layer_spec_paper_null_off",
    "gdar_layer_spec_paper_whiten_diag",
    "gdar_layer_spec_paper_whiten_off",
    "gdar_layer_spec_paper_mix_whitened",
    "gdar_layer_spec_paper_no_output_route",
    "gdar_layer_spec_paper_carrier0",
    "gdar_layer_spec_paper_init_paper",
    "gdar_layer_spec_paper_init_uniform",
    "gdar_layer_spec_paper_init_half",
    "gdar_layer_spec",
    "gdar_layer_spec_reference",
    "gdar_layer_spec_fullrank",
    "gdar_layer_spec_theory",
    "gdar_layer_spec_upstream",
    "gdar_layer_spec_block2",
    "gdar_layer_spec_block4",
    "gdar_layer_spec_block8",
    "gdar_layer_spec_block16",
    "gdar_layer_spec_block4_r64",
    "gdar_layer_spec_block4_r16",
    "gdar_layer_spec_gate_prefix",
    "gdar_layer_spec_gate_delta",
    "gdar_layer_spec_decay_projected",
    "gdar_layer_spec_lambda_free",
    "gdar_layer_spec_read_whitened_mix",
    "gdar_layer_spec_paper_noladder",
    "gdar_layer_spec_no_output_route",
    "DESIGN_ABLATIONS",
    "make_gdar_spec",
]

_DEFAULT = dict(
    gdar_block_size=1,
    gdar_output_route=True,
    gdar_gate_param="deviation",
    gdar_update="shensi",
    gdar_address="state",
    gdar_write_carrier_bias=-4.0,
    gdar_decay_ladder=0,
    gdar_read_heads=1,
    gdar_read_null=False,
    gdar_read_whiten="off",
    gdar_gate_rank=64,
    gdar_q_rank=64,
    gdar_k_rank=64,
)

_PAPER = dict(
    _DEFAULT,
    gdar_block_size=4,
    gdar_update="objective",
    gdar_decay_ladder=64,
    gdar_address="delta",
    gdar_read_heads=8,
    gdar_read_null=True,
    gdar_read_whiten="full",
    gdar_decay_positivity="project",
)


def make_gdar_spec(**knobs) -> ModuleSpec:
    params = dict(_DEFAULT)
    params.update(knobs)
    return ModuleSpec(module=GdarTransformerLayer, submodules=None, params=params)


def _paper(**one_knob) -> ModuleSpec:
    return make_gdar_spec(**dict(_PAPER, **one_knob))


gdar_layer_spec_paper: ModuleSpec = _paper()
gdar_layer_spec_paper_sublayer: ModuleSpec = _paper(gdar_block_size=1)
gdar_layer_spec_paper_b2: ModuleSpec = _paper(gdar_block_size=2)
gdar_layer_spec_paper_b4: ModuleSpec = _paper(gdar_block_size=4)
gdar_layer_spec_paper_b8: ModuleSpec = _paper(gdar_block_size=8)
gdar_layer_spec_paper_b16: ModuleSpec = _paper(gdar_block_size=16)

gdar_layer_spec_paper_rank16: ModuleSpec = _paper(gdar_gate_rank=16, gdar_q_rank=16, gdar_k_rank=16)
gdar_layer_spec_paper_rankfull: ModuleSpec = _paper(
    gdar_gate_rank=None, gdar_q_rank=None, gdar_k_rank=None
)

gdar_layer_spec_paper_decay_free: ModuleSpec = _paper(gdar_decay_positivity="free")
gdar_layer_spec_paper_lambda_free: ModuleSpec = _paper(gdar_lambda_clamp=None)
gdar_layer_spec_paper_ladder0: ModuleSpec = _paper(gdar_decay_ladder=0)
gdar_layer_spec_paper_gate_prefix: ModuleSpec = _paper(gdar_gate_source="prefix")
gdar_layer_spec_paper_gate_delta: ModuleSpec = _paper(gdar_gate_source="delta")
gdar_layer_spec_paper_address_state: ModuleSpec = _paper(gdar_address="state")
gdar_layer_spec_paper_address_novelty: ModuleSpec = _paper(gdar_address="novelty")
gdar_layer_spec_paper_update_reference: ModuleSpec = _paper(gdar_update="reference")
gdar_layer_spec_paper_heads1: ModuleSpec = _paper(gdar_read_heads=1)
gdar_layer_spec_paper_null_off: ModuleSpec = _paper(gdar_read_null=False)
gdar_layer_spec_paper_whiten_diag: ModuleSpec = _paper(gdar_read_whiten="diag")
gdar_layer_spec_paper_whiten_off: ModuleSpec = _paper(gdar_read_whiten="off")
gdar_layer_spec_paper_mix_whitened: ModuleSpec = _paper(gdar_read_mix="whitened")
gdar_layer_spec_paper_no_output_route: ModuleSpec = _paper(gdar_output_route=False)
gdar_layer_spec_paper_carrier0: ModuleSpec = _paper(gdar_write_carrier_bias=0.0)
gdar_layer_spec_paper_init_paper: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="paper"
)
gdar_layer_spec_paper_init_uniform: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="uniform", gdar_gate_init_bias=-20.0
)
gdar_layer_spec_paper_init_half: ModuleSpec = _paper(
    gdar_gate_param="sigmoid", gdar_gate_init="zero"
)

gdar_layer_spec: ModuleSpec = make_gdar_spec()

gdar_layer_spec_reference: ModuleSpec = ModuleSpec(
    module=GdarTransformerLayer,
    submodules=None,
    params=dict(
        _DEFAULT,
        gdar_gate_param="sigmoid",
        gdar_gate_init="paper",
        gdar_gate_rank=None,
        gdar_q_rank=None,
        gdar_k_rank=None,
    ),
)

gdar_layer_spec_fullrank: ModuleSpec = make_gdar_spec(
    gdar_gate_rank=None, gdar_q_rank=None, gdar_k_rank=None
)

gdar_layer_spec_theory: ModuleSpec = _paper(gdar_decay_positivity="free")

gdar_layer_spec_upstream: ModuleSpec = _paper(gdar_read_whiten="per_head")

gdar_layer_spec_block2: ModuleSpec = make_gdar_spec(gdar_block_size=2)
gdar_layer_spec_block4: ModuleSpec = make_gdar_spec(gdar_block_size=4)
gdar_layer_spec_block8: ModuleSpec = make_gdar_spec(gdar_block_size=8)
gdar_layer_spec_block16: ModuleSpec = make_gdar_spec(gdar_block_size=16)
gdar_layer_spec_block4_r64: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=64, gdar_q_rank=64, gdar_k_rank=64
)
gdar_layer_spec_block4_r16: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=16, gdar_q_rank=16, gdar_k_rank=16
)

gdar_layer_spec_gate_prefix: ModuleSpec = gdar_layer_spec_paper_gate_prefix
gdar_layer_spec_gate_delta: ModuleSpec = gdar_layer_spec_paper_gate_delta
gdar_layer_spec_decay_projected: ModuleSpec = gdar_layer_spec_paper
gdar_layer_spec_lambda_free: ModuleSpec = gdar_layer_spec_paper_lambda_free
gdar_layer_spec_read_whitened_mix: ModuleSpec = gdar_layer_spec_paper_mix_whitened
gdar_layer_spec_paper_noladder: ModuleSpec = gdar_layer_spec_paper_ladder0
gdar_layer_spec_no_output_route: ModuleSpec = gdar_layer_spec_paper_no_output_route

DESIGN_ABLATIONS: dict[str, ModuleSpec] = {
    "main": gdar_layer_spec_paper,
    "block_size=1": gdar_layer_spec_paper_sublayer,
    "block_size=2": gdar_layer_spec_paper_b2,
    "block_size=4": gdar_layer_spec_paper_b4,
    "block_size=8": gdar_layer_spec_paper_b8,
    "block_size=16": gdar_layer_spec_paper_b16,
    "rank=64": gdar_layer_spec_paper,
    "rank=16": gdar_layer_spec_paper_rank16,
    "rank=full": gdar_layer_spec_paper_rankfull,
    "init=identity": gdar_layer_spec_paper,
    "init=paper": gdar_layer_spec_paper_init_paper,
    "init=uniform": gdar_layer_spec_paper_init_uniform,
    "init=half": gdar_layer_spec_paper_init_half,
    "gate_source=state": gdar_layer_spec_paper,
    "gate_source=prefix": gdar_layer_spec_paper_gate_prefix,
    "gate_source=delta": gdar_layer_spec_paper_gate_delta,
    "update=objective": gdar_layer_spec_paper,
    "update=reference": gdar_layer_spec_paper_update_reference,
    "address=delta": gdar_layer_spec_paper,
    "address=state": gdar_layer_spec_paper_address_state,
    "address=novelty": gdar_layer_spec_paper_address_novelty,
    "lambda_clamp=-0.5": gdar_layer_spec_paper,
    "lambda_clamp=None": gdar_layer_spec_paper_lambda_free,
    "decay_positivity=project": gdar_layer_spec_paper,
    "decay_positivity=free": gdar_layer_spec_paper_decay_free,
    "ladder=64": gdar_layer_spec_paper,
    "ladder=0": gdar_layer_spec_paper_ladder0,
    "read_heads=8": gdar_layer_spec_paper,
    "read_heads=1": gdar_layer_spec_paper_heads1,
    "read_null=on": gdar_layer_spec_paper,
    "read_null=off": gdar_layer_spec_paper_null_off,
    "read_whiten=full": gdar_layer_spec_paper,
    "read_whiten=diag": gdar_layer_spec_paper_whiten_diag,
    "read_whiten=off": gdar_layer_spec_paper_whiten_off,
    "read_mix=raw": gdar_layer_spec_paper,
    "read_mix=whitened": gdar_layer_spec_paper_mix_whitened,
    "output_route=on": gdar_layer_spec_paper,
    "output_route=off": gdar_layer_spec_paper_no_output_route,
    "carrier_bias=-4": gdar_layer_spec_paper,
    "carrier_bias=0": gdar_layer_spec_paper_carrier0,
}
