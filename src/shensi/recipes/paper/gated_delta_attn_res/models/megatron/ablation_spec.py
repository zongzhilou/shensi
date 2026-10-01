"""E2/E3/E6 消融臂的层规格：门结构子集与块粒度扫描。"""

from __future__ import annotations

import torch

from megatron.core.transformer.spec_utils import ModuleSpec

from .gdar_connection import AttentionResidual as GdarAttentionResidual
from .gdar_layer import GdarTransformerLayer
from .gdar_spec import _DEFAULT as _GDAR_DEFAULT
from .gdar_spec import _PAPER as _GDAR_PAPER

__all__ = [
    "gdar_half_init_layer_spec",
    "gdar_uniform_init_layer_spec",
    "ABLATION_GATE_CHANNELS",
    "GATED_AR_BLOCK",
    "AblationGdarLayer",
    "gate_selecting_gates",
    "gated_ar_layer_spec",
    "gated_ar_layer_spec_block2",
    "gated_ar_layer_spec_block6",
    "gated_ar_layer_spec_block8",
    "gated_ar_layer_spec_block12",
    "gated_ar_layer_spec_block16",
    "gated_ar_layer_spec_decay_erase",
    "gated_ar_layer_spec_decay_only",
    "gated_ar_layer_spec_erase_only",
    "gated_ar_layer_spec_erase_write",
    "gated_ar_layer_spec_no_gate",
    "gated_ar_layer_spec_scalar",
    "gated_ar_layer_spec_write_decay",
    "gated_ar_layer_spec_write_only",
    "make_ablation_spec",
    "gdar_paper_gates_d",
    "gdar_paper_gates_e",
    "gdar_paper_gates_w",
    "gdar_paper_gates_de",
    "gdar_paper_gates_dw",
    "gdar_paper_gates_ew",
    "gdar_paper_gates_scalar",
    "gdar_paper_gates_none",
    "PAPER_GATE_SPECS",
]

ABLATION_GATE_CHANNELS = ("dew", "d", "e", "w", "de", "dw", "ew", "scalar", "none")

GATED_AR_BLOCK = 4

_BASE = dict(_GDAR_DEFAULT, gdar_block_size=GATED_AR_BLOCK)


def gate_selecting_gates(module: GdarAttentionResidual, channels: str):
    if channels not in ABLATION_GATE_CHANNELS:
        raise ValueError(f"gate channels {channels!r} not in {ABLATION_GATE_CHANNELS}")
    if not isinstance(module, GdarAttentionResidual):
        return module
    base_gates = module._gates

    def _gates(state, _base=base_gates, _channels=channels):
        decay, erase, write = _base(state)
        if _channels == "dew":
            return decay, erase, write
        if _channels == "none":
            ones = torch.ones_like(decay)
            return ones, torch.zeros_like(erase), ones
        channels_ = "w" if _channels == "scalar" else _channels
        if _channels == "scalar":
            write = write.mean(dim=-1, keepdim=True)
        if "d" not in channels_:
            decay = torch.ones_like(decay)
        if "e" not in channels_:
            erase = torch.zeros_like(erase)
        if "w" not in channels_:
            write = torch.ones_like(write)
        return decay, erase, write

    module._gates = _gates
    module.gate_channels = channels
    return module


class AblationGdarLayer(GdarTransformerLayer):

    def __init__(self, *args, **kwargs):
        channels = kwargs.pop("ablation_gate_channels", "dew")
        if channels not in ABLATION_GATE_CHANNELS:
            raise ValueError(f"ablation_gate_channels={channels!r} not in {ABLATION_GATE_CHANNELS}")
        super().__init__(*args, **kwargs)
        self.ablation_gate_channels = channels
        if channels == "dew":
            return
        for name in ("self_attention_attn_res", "mlp_attn_res", "output_attn_res"):
            module = getattr(self, name, None)
            if module is not None:
                gate_selecting_gates(module, channels)


def make_ablation_spec(**knobs) -> ModuleSpec:
    params = dict(_BASE)
    params.update(knobs)
    params.setdefault("ablation_gate_channels", "dew")
    return ModuleSpec(module=AblationGdarLayer, submodules=None, params=params)


gdar_uniform_init_layer_spec: ModuleSpec = make_ablation_spec(
    gdar_gate_init="uniform", gdar_gate_init_bias=-20.0, gdar_gate_param="sigmoid"
)

gdar_half_init_layer_spec: ModuleSpec = make_ablation_spec(
    gdar_gate_init="zero", gdar_gate_param="sigmoid"
)


gated_ar_layer_spec: ModuleSpec = make_ablation_spec()


gated_ar_layer_spec_block2: ModuleSpec = make_ablation_spec(gdar_block_size=2)
gated_ar_layer_spec_block6: ModuleSpec = make_ablation_spec(gdar_block_size=6)
gated_ar_layer_spec_block8: ModuleSpec = make_ablation_spec(gdar_block_size=8)
gated_ar_layer_spec_block16: ModuleSpec = make_ablation_spec(gdar_block_size=16)
gated_ar_layer_spec_block12: ModuleSpec = make_ablation_spec(gdar_block_size=12)


gated_ar_layer_spec_no_gate: ModuleSpec = make_ablation_spec(ablation_gate_channels="none")

gated_ar_layer_spec_decay_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="d")
gated_ar_layer_spec_erase_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="e")
gated_ar_layer_spec_write_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="w")
gated_ar_layer_spec_decay_erase: ModuleSpec = make_ablation_spec(ablation_gate_channels="de")
gated_ar_layer_spec_write_decay: ModuleSpec = make_ablation_spec(ablation_gate_channels="dw")
gated_ar_layer_spec_erase_write: ModuleSpec = make_ablation_spec(ablation_gate_channels="ew")

gated_ar_layer_spec_scalar: ModuleSpec = make_ablation_spec(ablation_gate_channels="scalar")


def _paper_gates(channels: str) -> ModuleSpec:
    return make_ablation_spec(**dict(_GDAR_PAPER, ablation_gate_channels=channels))


gdar_paper_gates_d: ModuleSpec = _paper_gates("d")
gdar_paper_gates_e: ModuleSpec = _paper_gates("e")
gdar_paper_gates_w: ModuleSpec = _paper_gates("w")
gdar_paper_gates_de: ModuleSpec = _paper_gates("de")
gdar_paper_gates_dw: ModuleSpec = _paper_gates("dw")
gdar_paper_gates_ew: ModuleSpec = _paper_gates("ew")
gdar_paper_gates_scalar: ModuleSpec = _paper_gates("scalar")
gdar_paper_gates_none: ModuleSpec = _paper_gates("none")

PAPER_GATE_SPECS: dict[str, ModuleSpec] = {
    "d": gdar_paper_gates_d,
    "e": gdar_paper_gates_e,
    "w": gdar_paper_gates_w,
    "de": gdar_paper_gates_de,
    "dw": gdar_paper_gates_dw,
    "ew": gdar_paper_gates_ew,
    "scalar": gdar_paper_gates_scalar,
    "none": gdar_paper_gates_none,
}
