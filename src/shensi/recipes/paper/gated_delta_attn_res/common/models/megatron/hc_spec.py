"""HC 的层规格预设。"""


from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .hc_layer import HcTransformerLayer

__all__ = [
    "hc_layer_spec",
    "hc_layer_spec_force_identity",
    "hc_layer_spec_published",
    "make_hc_spec",
    "mhc_layer_spec",
    "mhc_layer_spec_lite",
    "mhc_layer_spec_force_identity",
    "mhc_layer_spec_published",
]

_BASE = dict(hc_chunk_size=None)


def make_hc_spec(**knobs) -> ModuleSpec:
    params = dict(_BASE)
    params.update(knobs)
    return ModuleSpec(module=HcTransformerLayer, submodules=None, params=params)


hc_layer_spec: ModuleSpec = make_hc_spec(hc_family="hc", hc_init="identity", hc_read="simplex")

mhc_layer_spec: ModuleSpec = make_hc_spec(hc_family="mhc", hc_init="identity", hc_read="simplex")


hc_layer_spec_published: ModuleSpec = make_hc_spec(
    hc_family="hc", hc_init="official", hc_read="sigmoid", hc_contract="learned"
)

mhc_layer_spec_published: ModuleSpec = make_hc_spec(
    hc_family="mhc", hc_init="official", hc_read="sigmoid", hc_contract="learned"
)


hc_layer_spec_force_identity: ModuleSpec = make_hc_spec(
    hc_family="hc", hc_init="official", hc_read="sigmoid", hc_force_identity=True
)
mhc_layer_spec_force_identity: ModuleSpec = make_hc_spec(
    hc_family="mhc", hc_init="official", hc_read="sigmoid", hc_force_identity=True
)


mhc_layer_spec_lite: ModuleSpec = make_hc_spec(
    hc_family="mhc", hc_init="identity", hc_num_streams=2
)
