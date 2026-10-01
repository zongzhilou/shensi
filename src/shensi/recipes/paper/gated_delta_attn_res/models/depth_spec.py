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
"""Layer specs for AR / DAR / DenseFormer / MUDD, exposed through ``--spec``.

``--spec`` takes a ``<module> <object>`` pair and ``spec_utils.import_module``
returns that module-level *object* verbatim (it is not called), so every preset
here is a plain module-level ``ModuleSpec`` carrying the knobs in its ``params``;
``DepthTransformerLayer`` picks them up in ``__init__`` and builds its submodules
from the *config*, so nothing config-dependent is frozen at import time.

Enable one from a FlagScale task YAML::

    model:
      spec:
        - flagscale.models.megatron.depth.depth_spec
        - ar_layer_spec          # or dar_layer_spec / denseformer_layer_spec / mudd_layer_spec

which ``flatten_dict_to_args`` renders as
``--spec flagscale.models.megatron.depth.depth_spec ar_layer_spec``.

Each variant comes in two flavours:

* ``*_layer_spec`` -- the **identity anchor** used for the tiny runs: at step 0
  the whole model is bit-exactly the equivalent plain Qwen3 (``torch.equal``),
  so every variant starts from the same point and can only move away from it.
  For AR/DAR that means a zero-initialised scalar gate on the read (and, for AR,
  ``ar_reset="keep"``); DenseFormer/MUDD are anchored by their *official*
  initialisation (``one_hot(last)`` prior + ``W2 = 0``), which is the identity.
* ``*_layer_spec_reference`` / ``*_layer_spec_official`` / ``*_layer_spec_random``
  -- the published operator as-is (no gate, ``ar_reset="zero"`` for AR), i.e.
  what the reference repositories do.  These are **not** the identity at step 0;
  the deviation is measured in ``flagscale_runs/depth_check.py``.
"""

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

#: one source per sublayer write (the reference's granularity for the tiny runs)
_BASE = dict(depth_block_size=1, depth_output_route=True)


def make_depth_spec(**knobs) -> ModuleSpec:
    """A depth layer spec with ``depth_*`` knobs overriding the defaults."""
    params = dict(_BASE)
    params.update(knobs)
    return ModuleSpec(module=DepthTransformerLayer, submodules=None, params=params)


# --- AR -------------------------------------------------------------------

#: identity anchor: gated replacement read + the stream kept at block boundaries
ar_layer_spec: ModuleSpec = make_depth_spec(depth_variant="ar", depth_identity=True, depth_ar_reset="keep")

#: the Kimi reference as-is: ungated replacement read, accumulator zeroed at
#: block boundaries (``X(0) != Qwen3``; the deviation is measured, see the report)
ar_layer_spec_reference: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=False, depth_ar_reset="zero"
)

# --- DAR ------------------------------------------------------------------

#: identity anchor: the additive read is scaled by a zero-initialised scalar
dar_layer_spec: ModuleSpec = make_depth_spec(depth_variant="dar", depth_identity=True)

#: the reference as-is (``output = partial_block + selected``), no gate
dar_layer_spec_reference: ModuleSpec = make_depth_spec(depth_variant="dar", depth_identity=False)

#: the reference's learnable null source on top of the identity anchor
dar_layer_spec_null_source: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_use_null_source=True
)

# --- DenseFormer ----------------------------------------------------------

#: ``dwa_param="deviation"``: identical forward value to the official
#: initialisation, with the identity a construction guarantee
denseformer_layer_spec: ModuleSpec = make_depth_spec(depth_variant="denseformer", depth_dwa_param="deviation")

#: the released code literally (``alpha.zero_(); alpha[-1] = 1``), same forward value
denseformer_layer_spec_official: ModuleSpec = make_depth_spec(
    depth_variant="denseformer", depth_dwa_param="official"
)

#: block-granularity snapshots (one source per ``block_size`` layers, matching the
#: DAR-Block(B=4) / GDAR-Block(B=4) setting of the plan).  At ``block_size=1`` the packed
#: width is 8H-10H and 0.6B/seq1024 does not fit in 24 GB; B=4 cuts the source count by 4.
ar_layer_spec_block4: ModuleSpec = make_depth_spec(
    depth_variant="ar", depth_identity=True, depth_ar_reset="keep", depth_block_size=4
)
dar_layer_spec_block4: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=True, depth_block_size=4
)
dar_layer_spec_reference_block4: ModuleSpec = make_depth_spec(
    depth_variant="dar", depth_identity=False, depth_block_size=4
)

#: block-periodic packing: one DWA every 4 blocks, packed width (1 + ceil(L/4)) * H
denseformer_layer_spec_period4: ModuleSpec = make_depth_spec(
    depth_variant="denseformer", depth_block_size=4
)

# --- MUDD -----------------------------------------------------------------

#: official initialisation (``W2 = 0``, ``prior = one_hot(last)``) = identity
mudd_layer_spec: ModuleSpec = make_depth_spec(depth_variant="mudd", depth_mudd_param="deviation")

#: the same point as a plain parameter (the official JAX release: zero kernel + one-hot bias)
mudd_layer_spec_reference: ModuleSpec = make_depth_spec(depth_variant="mudd", depth_mudd_param="official")

#: the released PyTorch port initialises the static prior with ``randn`` -- *not* identity
mudd_layer_spec_random: ModuleSpec = make_depth_spec(depth_variant="mudd", depth_mudd_param="random")

#: official JAX flags on (PreDANorm / PostDANorm); note PostDANorm makes the model
#: non-identity by construction (``prior = 0`` and ``X + Norm(DA)``)
mudd_layer_spec_official_norm: ModuleSpec = make_depth_spec(
    depth_variant="mudd",
    depth_mudd_use_pre_norm=True,
    depth_mudd_use_post_norm=True,
    depth_mudd_param="official",
)

# --- E6: block-size sweep for the two ungated arms --------------------------
#
# The gated arm of the sweep lives in ``ablation_spec.gated_ar_layer_spec_block{B}``
# (``B = 4`` is also here: ``dar_layer_spec_block4`` / ``ar_layer_spec_block4``
# above), so these are the *ungated* complement of the same axis -- they are what
# makes "source x gate" separable from "how often the source is refreshed":
#
#   AR  at B: cumulative snapshots, one per block (B = 1: one per layer)
#   DAR at B: the differences between consecutive snapshots (B = 1: per sublayer)
#
# ``block_size`` is the only thing that changes: at B = 1 both variants switch to
# per-sublayer bookkeeping (see ``depth_layer.py``), which is why the B = 1 arms are
# ``ar_layer_spec`` / ``dar_layer_spec`` themselves.

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

# .. warning:: the ``dar`` arms above (and the pre-existing ``dar_layer_spec_block4``)
#    are currently **inert**: ``DepthTransformerLayer._forward_snapshot`` appends the
#    block snapshots only for ``variant == "ar"`` (``if ar and block_closed``), and its
#    ``_sublayer_sources`` differencing path needs at least two of them.  With no
#    snapshots the read short-circuits, so every connection parameter is unreachable
#    from the loss -- measured with ``flagscale_runs/depth_check.py`` on the 8-layer
#    block4 config: ``51 connection tensors: 0 non-zero grads, 51 unused (grad None)``,
#    i.e. the DAR-block arms would train as plain Qwen3.  The presets are kept because
#    the E6 sweep needs the names, but the E6 DAR rows cannot be run before
#    ``depth_layer.py`` learns the DAR block-mode append (one condition in
#    ``_forward_snapshot``, outside this file).  The AR arms and the gated arm
#    (``ablation_spec.gated_ar_layer_spec_block*``, which goes through the GDAR layer)
#    are unaffected: both were verified to pass ``depth_check.py``.
