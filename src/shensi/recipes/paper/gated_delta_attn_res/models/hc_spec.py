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
"""Layer specs for HC / mHC (official Megatron hyper-connection mechanism).

Enable from a FlagScale task YAML::

    model:
      spec:
        - flagscale.models.megatron.depth.hc_spec
        - mhc_layer_spec          # or hc_layer_spec

Presets -- every variant comes as an *identity anchor* and as the *published*
initialisation, so both can be measured:

===========================  ==============================  ====================
preset                       initialisation                  ``HC(0) == Qwen3``
===========================  ==============================  ====================
``hc_layer_spec``            identity anchor (``HC`` paper)  bit-exact
``mhc_layer_spec``           identity anchor (Sinkhorn)      bit-exact
``hc_layer_spec_published``  megatron-core defaults          no (measured)
``mhc_layer_spec_published`` megatron-core defaults          no (measured)
``*_layer_spec_force_identity``  run-time guard              bit-exact by fiat
===========================  ==============================  ====================

"identity anchor" = ``alpha = 0`` (the official ``mhc_init_gating_factor`` knob,
so the dynamic branch contributes exactly nothing while the xavier
``mapping_proj`` stays a non-zero carrier and keeps a live gradient), the
softmax read (exact ``1/n``), ``H_post = 2*sigmoid(0) = 1``, ``H_res = I`` for HC
and ``Sinkhorn(0)`` at ``eps = 0`` = the uniform doubly-stochastic matrix for
mHC, and the official mean contraction.

"published" = ``HyperConnectionModule._init_weights`` exactly: xavier
``mapping_proj``, ``alpha = mhc_init_gating_factor`` (0.01), zero static biases
(``H_pre = sigmoid(0) + 1e-6 = 0.500001`` per stream, ``H_post = 1``,
``H_res = Sinkhorn(0)`` at the released ``eps = 1e-6``) and
``learned_output_contract`` with the block's ``torch.randn`` head.  For plain HC
(unprojected ``H_res``) a zero matrix would delete the stream, so ``H_res`` is
anchored at ``I`` there -- the one deviation, and it is the HC paper's own static
``A_r = I``.
"""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec

from .hc_layer import HcTransformerLayer

__all__ = [
    "hc_layer_spec",
    "hc_layer_spec_force_identity",
    "hc_layer_spec_published",
    "make_hc_spec",
    "mhc_layer_spec",
    "mhc_layer_spec_force_identity",
    "mhc_layer_spec_published",
]

#: one n-stream residual for the whole decoder (the official layout:
#: ``TransformerBlock`` expands at the entry and contracts at the exit).
_BASE = dict(hc_chunk_size=None)


def make_hc_spec(**knobs) -> ModuleSpec:
    """An HC/mHC layer spec with ``hc_*`` knobs overriding the defaults."""
    params = dict(_BASE)
    params.update(knobs)
    return ModuleSpec(module=HcTransformerLayer, submodules=None, params=params)


# --- identity anchors (used by the tiny runs) -----------------------------

#: HC (no manifold projection) pinned to the exact plain residual at step 0
hc_layer_spec: ModuleSpec = make_hc_spec(hc_family="hc", hc_init="identity", hc_read="simplex")

#: mHC (Sinkhorn-Knopp doubly-stochastic H_res) pinned to the exact plain residual
mhc_layer_spec: ModuleSpec = make_hc_spec(hc_family="mhc", hc_init="identity", hc_read="simplex")

# --- published initialisation (measured, not the identity) ----------------

#: megatron-core 0.18.2 defaults for HC: sigmoid read, alpha = 0.01, zero biases,
#: zeros H_res logits anchored at I, learned output contraction
hc_layer_spec_published: ModuleSpec = make_hc_spec(
    hc_family="hc", hc_init="official", hc_read="sigmoid", hc_contract="learned"
)

#: the released mHC module as-is: Sinkhorn H_res at eps = 1e-6, learned contraction
mhc_layer_spec_published: ModuleSpec = make_hc_spec(
    hc_family="mhc", hc_init="official", hc_read="sigmoid", hc_contract="learned"
)

# --- the run-time identity guard (``_force_identity`` semantics) -----------

hc_layer_spec_force_identity: ModuleSpec = make_hc_spec(
    hc_family="hc", hc_init="official", hc_read="sigmoid", hc_force_identity=True
)
mhc_layer_spec_force_identity: ModuleSpec = make_hc_spec(
    hc_family="mhc", hc_init="official", hc_read="sigmoid", hc_force_identity=True
)
