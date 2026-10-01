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
"""Layer specs for the E2/E3/E6 ablation arms.

The GDAR package (``flagscale.models.megatron.gdar``) already owns the *operator*
and its specs; this module adds the arms the ablation tables need, on top of it and
without touching it:

``gated_ar_layer_spec``      **E2, fourth cell**: the GDAR operator fed by
                             cumulative stream snapshots plus the three gates --
                             i.e. the AR source structure with GDAR's gates.  In
                             this layer the source structure *is* the block size
                             (``block_size > 1`` appends the stream itself at every
                             block boundary and reads those snapshots; ``block_size
                             == 1`` appends the sublayer outputs, i.e. deltas), so
                             the preset pins ``gdar_block_size`` and leaves the
                             operator untouched.
``gated_ar_layer_spec_block{2,6,8,12}``
                             **E6**: the same arm at other block granularities
                             (``B = 4`` is the preset's own default, matching
                             ``depth_spec.gdar_layer_spec_block4`` /
                             ``gdar_spec.gdar_layer_spec_block4``).
``gated_ar_layer_spec_*``    **E3**: the gate-structure rows -- which of decay /
                             erase / write exist -- from ``no_gate`` (the update is
                             then the DAR rule, exactly) to ``scalar`` (the write
                             gate collapses to one value per token) and the seven
                             gate subsets in between.

The gate subset is applied to the *existing* connection module by binding a
``_gates`` wrapper on it (``gate_selecting_gates``), not by building a different
module: the arm then differs from the reference in the gate selection alone, with
bit-identical parameters and initialisation (no extra RNG draws, no state-dict
remap).  The maths mirrors ``models/modeling_qwen3_gdar.py:_apply_gate_channels``
-- that function is the source of truth, this is its Megatron twin.

Enable one from a FlagScale task YAML::

    model:
      spec:
        - flagscale.models.megatron.depth.ablation_spec
        - gated_ar_layer_spec_block8

which ``flatten_dict_to_args`` renders as
``--spec flagscale.models.megatron.depth.ablation_spec gated_ar_layer_spec_block8``.

Not here:  E4 (gate initialisation) is a *value* ablation and is covered by the
existing ``gdar_gate_init`` / ``gdar_gate_init_bias`` knobs (the exact-identity and
0.5-init rows); the 0-init ("uniform" bias) row exists on the HF side
(``attn_res_gate_init="uniform"``).  E5 (rank) is ``gdar_gate_rank`` /
``gdar_q_rank`` / ``gdar_k_rank``, also already there.
"""

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

#: the gate subsets of E3; identical tuple to the HF ``configuration_qwen3_gdar.GATE_CHANNELS``
ABLATION_GATE_CHANNELS = ("dew", "d", "e", "w", "de", "dw", "ew", "scalar", "none")

#: block granularity that makes the read contexts cumulative stream snapshots (the
#: AR source) instead of per-sublayer deltas -- the E2 "Gated-AR" cell.
GATED_AR_BLOCK = 4

#: the operator's knobs, taken verbatim from the GDAR package (one source of truth),
#: with one change: the block size defaults to the Gated-AR cell's granularity, so an
#: unqualified ``make_ablation_spec(ablation_gate_channels=...)`` varies the gate
#: structure *inside* the E2 fourth cell instead of silently switching the source
#: structure back to the per-sublayer (GDAR) one.  Pass ``gdar_block_size=1`` for a
#: per-sublayer arm, or any other ``gdar_*`` knob to change the operator.
_BASE = dict(_GDAR_DEFAULT, gdar_block_size=GATED_AR_BLOCK)


def gate_selecting_gates(module: GdarAttentionResidual, channels: str):
    """Wrap ``module._gates`` so only the gates in ``channels`` survive.

    The removed gates are replaced by their identity constant (decay 1, erase 0,
    write 1) *after* the projection, exactly as the HF switch does, so every E3 arm
    keeps the reference's parameter set and differs only in the gate structure (the
    removed slices receive zero gradient).  ``"scalar"`` keeps the write gate and
    collapses it over channels; ``"none"`` pins all three, which makes the stream
    update the DAR rule.
    """
    if channels not in ABLATION_GATE_CHANNELS:
        raise ValueError(f"gate channels {channels!r} not in {ABLATION_GATE_CHANNELS}")
    if not isinstance(module, GdarAttentionResidual):
        # ``DepthRead`` (the final output routing) is a pure read: it has no gates to
        # select, exactly as on the HF side where it is a separate module class.
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

    # an instance attribute, so the *same* module -- same parameters, same init --
    # changes only its gate selection
    module._gates = _gates
    module.gate_channels = channels
    return module


class AblationGdarLayer(GdarTransformerLayer):
    """``GdarTransformerLayer`` + the E3 gate subset (``ablation_gate_channels``).

    Everything else -- packing, output routing, dropout takeover, the isolated-RNG
    initialisation of the connection modules -- is the GDAR layer, untouched; this
    class only rebinds the gates of the modules the base class just built, so the
    shared parameters stay bit-identical to a plain ``gdar_layer_spec`` run at the
    same seed (which is what ``flagscale_runs/depth_check.py`` asserts).
    """

    def __init__(self, *args, **kwargs):
        channels = kwargs.pop("ablation_gate_channels", "dew")
        if channels not in ABLATION_GATE_CHANNELS:
            raise ValueError(f"ablation_gate_channels={channels!r} not in {ABLATION_GATE_CHANNELS}")
        super().__init__(*args, **kwargs)
        self.ablation_gate_channels = channels
        if channels == "dew":  # the reference: nothing to do
            return
        for name in ("self_attention_attn_res", "mlp_attn_res", "output_attn_res"):
            module = getattr(self, name, None)
            if module is not None:
                gate_selecting_gates(module, channels)


def make_ablation_spec(**knobs) -> ModuleSpec:
    """A gated-AR / ablation layer spec with ``gdar_*`` and ``ablation_*`` knobs."""
    params = dict(_BASE)
    params.update(knobs)
    params.setdefault("ablation_gate_channels", "dew")
    return ModuleSpec(module=AblationGdarLayer, submodules=None, params=params)


# --- E4: initialisation rows (mirrors the HF `attn_res_gate_init` values) ----

#: ``"uniform"`` = all three gate heads at bias ``-20`` with the *sigmoid* gate: the gates start
#: at ``sigmoid(-20) = 2.1e-9``, i.e. the "0-init" row that the old draft's failure mode came from.
gdar_uniform_init_layer_spec: ModuleSpec = make_ablation_spec(
    gdar_gate_init="uniform", gdar_gate_init_bias=-20.0, gdar_gate_param="sigmoid"
)

#: ``"zero"`` = all biases zero, i.e. every gate at ``sigmoid(0) = 0.5`` (the "0.5-init" row).
gdar_half_init_layer_spec: ModuleSpec = make_ablation_spec(
    gdar_gate_init="zero", gdar_gate_param="sigmoid"
)

# --- E2: the fourth cell ----------------------------------------------------

#: Gated-AR: cumulative block snapshots (the AR source) read through GDAR's gates.
#: Knob-wise this is ``gdar_spec.gdar_layer_spec_block4`` (the operator and the block
#: size are the same); the preset exists to *name* the cell.  ``attn_res_address``
#: has no counterpart here: the Megatron port takes the erase direction from the
#: state by default (``gdar_address="state"``, the reference), which is the same
#: AR-flavoured choice the HF ``gated_ar_preset()`` pins.  The per-sublayer (delta
#: source, i.e. GDAR) arm of the same operator is ``gdar_spec.gdar_layer_spec``.
gated_ar_layer_spec: ModuleSpec = make_ablation_spec()

# --- E6: block size ---------------------------------------------------------

gated_ar_layer_spec_block2: ModuleSpec = make_ablation_spec(gdar_block_size=2)
gated_ar_layer_spec_block6: ModuleSpec = make_ablation_spec(gdar_block_size=6)
gated_ar_layer_spec_block8: ModuleSpec = make_ablation_spec(gdar_block_size=8)
gated_ar_layer_spec_block16: ModuleSpec = make_ablation_spec(gdar_block_size=16)
gated_ar_layer_spec_block12: ModuleSpec = make_ablation_spec(gdar_block_size=12)

# --- E3: gate structure -----------------------------------------------------
#
# All eight rows below keep the cell's source structure (block 4, set in ``_BASE``),
# so the E3 table varies *only* the gate structure.

#: no gate at all: the update is the DAR rule exactly ("no gate" row of E3)
gated_ar_layer_spec_no_gate: ModuleSpec = make_ablation_spec(ablation_gate_channels="none")

#: the seven gate subsets (``d``/``e``/``w`` and their unions); ``gated_ar_layer_spec``
#: itself is the ``dew`` row
gated_ar_layer_spec_decay_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="d")
gated_ar_layer_spec_erase_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="e")
gated_ar_layer_spec_write_only: ModuleSpec = make_ablation_spec(ablation_gate_channels="w")
gated_ar_layer_spec_decay_erase: ModuleSpec = make_ablation_spec(ablation_gate_channels="de")
gated_ar_layer_spec_write_decay: ModuleSpec = make_ablation_spec(ablation_gate_channels="dw")
gated_ar_layer_spec_erase_write: ModuleSpec = make_ablation_spec(ablation_gate_channels="ew")

#: single scalar gate: the write gate averaged over channels (one value per token)
gated_ar_layer_spec_scalar: ModuleSpec = make_ablation_spec(ablation_gate_channels="scalar")


# ---------------------------------------------------------------------------
# A8：主配置（main）上的门结构行 —— 7 个子集 + scalar + none，全部基于 ``_GDAR_PAPER``
# （即 main ± 只改门结构），所以每一行都能与主行直接 diff（EXPERIMENT_MATRIX.json 的
# "gates=…" 列表；``dew`` 就是 main 本身，见 ``gdar_spec.gdar_layer_spec_paper``）。
# ---------------------------------------------------------------------------


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
