"""深度连接族的层规格工厂。"""


from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .depth_layer import DepthTransformerLayer

__all__ = [
    "ar_layer_spec",
    "ar_layer_spec_block2",
    "ar_layer_spec_block4",
    "ar_layer_spec_block6",
    "ar_layer_spec_block8",
    "ar_layer_spec_block12",
    "ar_layer_spec_reference",
    "dar_layer_spec",
    "dar_layer_spec_block2",
    "dar_layer_spec_block4",
    "dar_layer_spec_block6",
    "dar_layer_spec_block8",
    "dar_layer_spec_block12",
    "dar_layer_spec_null_source",
    "dar_layer_spec_reference",
    "dar_layer_spec_reference_block4",
    "denseformer_layer_spec",
    "denseformer_layer_spec_official",
    "denseformer_layer_spec_period4",
    "make_depth_spec",
    "mudd_layer_spec",
    "mudd_layer_spec_official_norm",
    "mudd_layer_spec_random",
    "mudd_layer_spec_reference",
]

_BASE = dict(depth_block_size=1, depth_output_route=True)


def make_depth_spec(**knobs) -> ModuleSpec:
    """按连接族与配置生成层规格。"""
    params = dict(_BASE)
    params.update(knobs)
    return ModuleSpec(module=DepthTransformerLayer, submodules=None, params=params)


ar_layer_spec: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep"
)

ar_layer_spec_reference: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=False, depth_ar_reset="zero"
)


dar_layer_spec: ModuleSpec = make_depth_spec(depth_variant="dar", depth_identity=True)

dar_layer_spec_reference: ModuleSpec = make_depth_spec(depth_variant="dar", depth_identity=False)

dar_layer_spec_null_source: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_use_null_source=True
)


denseformer_layer_spec: ModuleSpec = make_depth_spec(
    depth_variant="denseformer", depth_dwa_param="deviation"
)

denseformer_layer_spec_official: ModuleSpec = make_depth_spec(
    depth_variant="denseformer", depth_dwa_param="official"
)

ar_layer_spec_block4: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=4
)
dar_layer_spec_block4: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=4
)
dar_layer_spec_reference_block4: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=False, depth_block_size=4
)

denseformer_layer_spec_period4: ModuleSpec = make_depth_spec(
    depth_variant="denseformer", depth_block_size=4
)


mudd_layer_spec: ModuleSpec = make_depth_spec(depth_variant="mudd", depth_mudd_param="deviation")

mudd_layer_spec_reference: ModuleSpec = make_depth_spec(
    depth_variant="mudd", depth_mudd_param="official"
)

mudd_layer_spec_random: ModuleSpec = make_depth_spec(
    depth_variant="mudd", depth_mudd_param="random"
)

mudd_layer_spec_official_norm: ModuleSpec = make_depth_spec(
    depth_variant="mudd",
    depth_mudd_use_pre_norm=True,
    depth_mudd_use_post_norm=True,
    depth_mudd_param="official",
)


ar_layer_spec_block2: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=2
)
ar_layer_spec_block6: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=6
)
ar_layer_spec_block8: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=8
)
ar_layer_spec_block12: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=12
)

dar_layer_spec_block2: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=2
)
dar_layer_spec_block6: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=6
)
dar_layer_spec_block8: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=8
)
dar_layer_spec_block12: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=12
)

