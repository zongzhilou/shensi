# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer specs for GDAR, exposed through Megatron's official ``--spec`` extension point.

``--spec`` takes a ``<module> <object>`` pair and ``spec_utils.import_module`` returns
that module-level *object* verbatim (it is not called), so a spec has to exist before
the config does.  That is why every preset here is a plain module-level
``ModuleSpec`` carrying the GDAR knobs in its ``params``: ``GdarTransformerLayer``
picks the knobs up in ``__init__`` (and builds its submodules from the *config*, so
nothing config-dependent is frozen at import time).

Enable it from a FlagScale task YAML::

    model:
      spec:
        - flagscale.models.megatron.gdar.gdar_spec
        - gdar_layer_spec

which ``flatten_dict_to_args`` renders as
``--spec flagscale.models.megatron.gdar.gdar_spec gdar_layer_spec``.

Presets (all of them ``block_size == 1`` = one depth source per sublayer write,
the reference decoder's granularity; ``block_size=None`` disables GDAR entirely and
becomes plain Qwen3)::

    gdar_layer_spec             deviation gates (GDAR(0) == Qwen3 *bit-exactly*),
                                shensi update, low-rank 64        <- the training default
    gdar_layer_spec_fullrank    same, full-rank gate/q/k projections (the reference's
                                parameterisation, ~10 H^2 extra parameters per layer)
    gdar_layer_spec_reference   the shensi repository as-is: sigmoid gates with the
                                "paper" init, full-rank projections (no exact identity)
    gdar_layer_spec_theory      the theory preset (objective update, 8 read heads,
                                Softmax_1, full whitening, learned decay ladder, r=64)
    gdar_layer_spec_block{2,4}  deviation gates with block-granularity snapshots

Anything else: ``make_gdar_spec(**knobs)`` for programmatic use (the offline
identity/gradient checks use it).
"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .gdar_layer import GdarTransformerLayer

__all__ = [
    "gdar_layer_spec",
    "gdar_layer_spec_block2",
    "gdar_layer_spec_block4",
    "gdar_layer_spec_fullrank",
    "gdar_layer_spec_reference",
    "gdar_layer_spec_theory",
    "make_gdar_spec",
]

#: default knobs of the connection.  See ``gdar_connection.GdarConfig``.
_DEFAULT = dict(
    gdar_block_size=1,
    gdar_output_route=True,
    # --- write side: exact identity at init (GDAR(0) == the plain residual update)
    gdar_gate_param="deviation",
    gdar_update="shensi",
    gdar_address="state",
    gdar_write_carrier_bias=-4.0,
    gdar_decay_ladder=0,
    # --- read side: the reference's plain single-head softmax read
    gdar_read_heads=1,
    gdar_read_null=False,
    gdar_read_whiten="off",
    # --- parameterisation: the redo plan's low-rank fix (full rank costs ~10 H^2/layer).
    # r=64 is the training default the docstring above, the recipe (RECIPE.md item 1) and
    # check_gdar.py all promise; `gdar_layer_spec_fullrank` is the explicit full-rank arm.
    gdar_gate_rank=64,
    gdar_q_rank=64,
    gdar_k_rank=64,
)


def make_gdar_spec(**knobs) -> ModuleSpec:
    """A GDAR layer spec with ``gdar_*`` knobs overriding the defaults."""
    params = dict(_DEFAULT)
    params.update(knobs)
    return ModuleSpec(module=GdarTransformerLayer, submodules=None, params=params)


gdar_layer_spec: ModuleSpec = make_gdar_spec()

#: the shensi repository as-is (no exact identity, full-rank projections)
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

#: deviation gates with the reference's full-rank projections
gdar_layer_spec_fullrank: ModuleSpec = make_gdar_spec(
    gdar_gate_rank=None, gdar_q_rank=None, gdar_k_rank=None
)

#: the theory preset (see ``Qwen3GDARConfig.theory_preset``)
#: --- design-ablation presets (see GDAR_ABLATION_DESIGN.md) -----------------------
#: Each flips exactly ONE design decision of GDAR, so every row can be diffed against
#: `gdar_layer_spec` directly.  A1/A2 need a trained-like gate_proj to be visible (the
#: deviation init zeroes it); A5 needs `update="objective"` (that is where lambda lives).
gdar_layer_spec_gate_prefix: ModuleSpec = make_gdar_spec(gdar_gate_source="prefix")
gdar_layer_spec_gate_delta: ModuleSpec = make_gdar_spec(gdar_gate_source="delta")
gdar_layer_spec_decay_projected: ModuleSpec = make_gdar_spec(gdar_decay_positivity="project")
gdar_layer_spec_lambda_free: ModuleSpec = make_gdar_spec(gdar_update="objective", gdar_lambda_clamp=None)
gdar_layer_spec_read_whitened_mix: ModuleSpec = make_gdar_spec(
    gdar_read_mix="whitened", gdar_read_whiten="full"
)
#: The paper's **main-table** GDAR row: the theory configuration (objective update, decay
#: ladder, address=delta, 8 read heads, Softmax_1, full whitening, r=64) *plus* the decay
#: positivity projection, so that the bounded-decay claim in the paper holds by
#: construction (without it the learned scale was measured to go negative in 24/24 modules
#: of one run).  See GDAR_ABLATION_DESIGN.md A3.
gdar_layer_spec_paper: ModuleSpec = make_gdar_spec(
    gdar_update="objective",
    gdar_decay_ladder=64,
    gdar_address="delta",
    gdar_read_heads=8,
    gdar_read_null=True,
    gdar_read_whiten="full",
    gdar_decay_positivity="project",
)

#: name -> spec, for a driver that iterates the design ablation
DESIGN_ABLATIONS: dict[str, ModuleSpec] = {
    "gdar_layer_spec": gdar_layer_spec,
    "paper_main": gdar_layer_spec_paper,
    "gate_prefix": gdar_layer_spec_gate_prefix,
    "gate_delta": gdar_layer_spec_gate_delta,
    "decay_projected": gdar_layer_spec_decay_projected,
    "lambda_free": gdar_layer_spec_lambda_free,
    "read_whitened_mix": gdar_layer_spec_read_whitened_mix,
}

gdar_layer_spec_theory: ModuleSpec = make_gdar_spec(
    gdar_update="objective",
    gdar_decay_ladder=64,
    gdar_address="delta",
    gdar_read_heads=8,
    gdar_read_null=True,
    gdar_read_whiten="full",
)

#: block-granularity snapshots (one source per ``block_size`` layers)
gdar_layer_spec_block2: ModuleSpec = make_gdar_spec(gdar_block_size=2)
#: block-granularity snapshots at the sizes the block-size ablation asks for
#: (review 2 W8: B in {2,4,8,16}); B=4 is `gdar_layer_spec_block4` below.
gdar_layer_spec_block8: ModuleSpec = make_gdar_spec(gdar_block_size=8)
gdar_layer_spec_block16: ModuleSpec = make_gdar_spec(gdar_block_size=16)
#: block-4 variants at an explicit rank, for scans whose iso-FLOP plan must know the
#: parameterisation (the default r=64 and the parameter-matched r=16 of the paper).
gdar_layer_spec_block4_r64: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=64, gdar_q_rank=64, gdar_k_rank=64
)
gdar_layer_spec_block4_r16: ModuleSpec = make_gdar_spec(
    gdar_block_size=4, gdar_gate_rank=16, gdar_q_rank=16, gdar_k_rank=16
)
gdar_layer_spec_block4: ModuleSpec = make_gdar_spec(gdar_block_size=4)
