"""HF <-> Megatron weight conversion for the seven depth-connection variants.

Why this package exists: ``verl``'s Megatron path builds the model through
``mbridge``, which has no weight mapping for any of the connection tensors *and*
whose backbone entries assume the TransformerEngine fused-norm layout while our
specs build the local one.  The registration work
(``code/VERL_REGISTRATION.md``) got the model built; this gets the weights in.

Layout
------
=========================================  =========================================
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.tables`         the names: one generated table per variant
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.convert`        the functions: state_dict -> state_dict
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.mbridge_patch`  the hook: bridge.load/export_weights
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.profiles`       the configurations the audit converts
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.hf_reference`   the HF half (runs under transformers 5)
:mod:`~shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.audit`          the proof: 17 configs, both directions
=========================================  =========================================

Quick use
---------
Offline (no Megatron needed)::

    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert import AttnLayout, build_table, hf_to_mcore, mcore_to_hf

    table = build_table("gdar", hf_config, hf_config.num_hidden_layers)
    mcore_sd, report = hf_to_mcore(hf_sd, table, layout=AttnLayout.from_config(hf_config),
                                   padded_vocab_size=151936)
    assert report.ok, report.describe()

In the verl run: ``verl_plugin.install()`` is enough -- the bridge registered for
each ``model_type`` already carries :class:`WeightConversionMixin`, so
``bridge.load_weights`` and ``bridge.export_weights`` are the converted ones.

The verification run::

    PYTHONPATH=$PWD .venv-flagos/bin/python -m shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.audit

It builds every profile's Megatron model for real, converts a real HF checkpoint
both ways, compares the two models' logits on a fixed input, and prints the
per-profile counts.  A green run has ``FAIL 0``; what "passes", "declared gap" and
"failure" mean is spelled out in ``code/VERL_CONVERTER.md`` §3.
"""

from __future__ import annotations

from .convert import AttnLayout, ConversionReport, compare, hf_to_mcore, mcore_to_hf
from .tables import ALL_VARIANTS, KINDS, Pair, SynthesisPolicy, Table, build_table

__all__ = [
    "ALL_VARIANTS",
    "AttnLayout",
    "ConversionReport",
    "KINDS",
    "Pair",
    "SynthesisPolicy",
    "Table",
    "build_table",
    "compare",
    "hf_to_mcore",
    "mcore_to_hf",
]
