"""RealFormer 的层规格预设：deviation / 恒等 / 上游原样 / running mean 四档。"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .realformer_attention import RealFormerCarry
from .realformer_layer import RealFormerTransformerLayer

__all__ = [
    "make_realformer_spec",
    "realformer_layer_spec",
    "realformer_layer_spec_identity",
    "realformer_layer_spec_mean",
    "realformer_layer_spec_reference",
]

_DEFAULT = dict(realformer_gate="deviation", realformer_mean=False)


def make_realformer_spec(**knobs) -> ModuleSpec:
    params = dict(_DEFAULT)
    params.update(knobs)
    params.setdefault("realformer_carry", RealFormerCarry())
    return ModuleSpec(module=RealFormerTransformerLayer, submodules=None, params=params)


realformer_layer_spec: ModuleSpec = make_realformer_spec()

realformer_layer_spec_identity: ModuleSpec = make_realformer_spec(realformer_gate="zero")

realformer_layer_spec_reference: ModuleSpec = make_realformer_spec(realformer_gate="one")

realformer_layer_spec_mean: ModuleSpec = make_realformer_spec(
    realformer_gate="deviation", realformer_mean=True
)
