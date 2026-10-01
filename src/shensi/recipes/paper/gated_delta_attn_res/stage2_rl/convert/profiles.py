"""The configurations the audit converts, one profile per interesting shape.

Shared by the two halves of the audit: :mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.hf_reference`
(runs under ``.venv``, where transformers 5 can import the modeling files) and
:mod:`shensi.recipes.paper.gated_delta_attn_res.stage2_rl.convert.audit` (runs under ``.venv-flagos``, where Megatron
lives).  They must agree on the knobs or the comparison is meaningless, so the
definitions live in one place and both halves import them.

Every profile is deliberately *small* (2 layers, hidden 64) and *non-default*
where a knob changes the tensor set -- a profile that only exercises the default
path proves the default path works and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TINY", "Profile", "PROFILES", "profile", "GDAR_LADDER"]

#: the shape used everywhere in the audit
TINY = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)

#: ladder length for the GDAR multi-timescale profile
GDAR_LADDER = 8


@dataclass(frozen=True)
class Profile:
    """One model configuration to convert."""

    name: str
    variant: str
    #: HF config overrides on top of :data:`TINY`
    knobs: dict = field(default_factory=dict)
    #: what this profile is for
    about: str = ""
    #: **expected asymmetries**: substrings of tensors the reference has and the
    #: Megatron port does not (or vice versa) that a check may report as an
    #: offender without the profile failing.  Anything the audit finds that is not
    #: covered by one of these strings is still a failure, and a string that never
    #: fires is a failure too -- a declaration is a claim, and the audit verifies it.
    expect_unsupported: tuple[str, ...] = ()
    #: **expected synthesis**: substrings of the Megatron-only rows the converter is
    #: allowed to invent values for (a zero gate, a zero bias, a null source).
    #: Synthesising anything not declared -- or declaring something that never gets
    #: synthesised -- fails, so "nothing is invented silently" is checked, not
    #: asserted in a comment.
    expect_synthesized: tuple[str, ...] = ()
    #: **declared forward gap**: the reason this configuration's logits *cannot*
    #: match the reference even though every tensor was converted exactly.  Only
    #: for differences that live in the Megatron port's arithmetic rather than in
    #: the name/shape/data mapping -- the audit requires the gap to be real (a
    #: declaration that stops differing is a failure), and prints it in the summary
    #: next to the measured delta.
    expect_forward_gap: str | None = None
    #: skip the mcore build entirely, with this reason
    skip_build: str | None = None

    def config_kwargs(self) -> dict:
        return {**TINY, **self.knobs}


PROFILES: tuple[Profile, ...] = (
    Profile("gdar", "gdar", dict(attn_res_block_size=1), "the reference case, full rank (both sides default)"),
    Profile(
        "gdar_lowrank",
        "gdar",
        dict(attn_res_block_size=1, attn_res_gate_rank=16, attn_res_q_rank=16, attn_res_k_rank=16),
        "low rank: HF Sequential indices vs Megatron .down/.up",
        # 2 layers x 2 sublayers x {q,k} + the read-only output module's q = 9
        expect_synthesized=("q_proj.up.bias", "k_proj.up.bias"),
    ),
    Profile(
        "gdar_ladder",
        "gdar",
        dict(attn_res_block_size=1, attn_res_decay_ladder=GDAR_LADDER),
        "learned per-channel time constants (decay_tau)",
    ),
    Profile(
        "gdar_deviation",
        "gdar",
        dict(attn_res_block_size=1, attn_res_gate_param="deviation"),
        "the deviation gate form: three extra scalars per sublayer, on both sides",
    ),
    Profile(
        "ar",
        "ar",
        dict(attn_res_block_size=1),
        "Kimi AR router + Megatron identity gate",
        expect_synthesized=("read_scale",),
        expect_forward_gap=(
            "the port's AR at block size 1 uses its per-sublayer source mode (3 sources per "
            "layer) while HF appends 1 per layer; the tensors are all mapped exactly, the "
            "arithmetic is not the same operator. Measured delta 2.71e-01 at read_scale=1.0 "
            "and 1.13e-01 with the port's 'keep' default; AR at block size 2 matches (ar_block2)"
        ),
    ),
    Profile(
        "ar_block2",
        "ar",
        dict(attn_res_block_size=2),
        "AR in *block* mode (P>1): one source per layer, which is what HF's AR does",
        expect_synthesized=("read_scale",),
    ),
    Profile(
        "dar",
        "dar",
        dict(attn_res_block_size=1),
        "DAR delta router + Megatron identity gate",
        expect_synthesized=("read_scale",),
    ),
    Profile(
        "dar_null",
        "dar",
        dict(attn_res_block_size=1, attn_res_use_null_source=True),
        "learnable null source",
        # the table's own note: "the Megatron output router carries a null source
        # that HF's output read does not have, so the final read is not equivalent"
        expect_unsupported=("output router",),
        expect_synthesized=("read_scale", "null_source"),
        expect_forward_gap=(
            "the per-sublayer null sources are real HF tensors and convert exactly, but the "
            "Megatron *output* router prepends one too (synthesised as zeros) where HF's output "
            "read has none, and an extra softmax entry changes the mixture even at zero: measured "
            "delta 2.88e-03, three orders of magnitude above the 3e-07 the other DAR profiles show"
        ),
    ),
    Profile("denseformer", "denseformer", dict(attn_res_block_size=1), "per-event weighted average (deviation form)"),
    Profile(
        "denseformer_official",
        "denseformer",
        dict(attn_res_block_size=1, attn_res_dwa_param="official"),
        "the released parameterisation (alpha, one-hot at init)",
    ),
    Profile(
        "hc",
        "hc",
        dict(attn_res_block_size=2, hc_read="simplex"),
        "hyper-connections, one chunk over the whole model",
        expect_forward_gap=(
            "HF's HC write is `hc_write='linear'` by default (the paper's unconstrained static "
            "mixing) and the port implements the mHC write ('sigmoid * 2') only; measured delta "
            "1.42e-01 with every tensor mapped exactly. mHC -- whose default *is* sigmoid2 -- "
            "matches to 2.4e-07"
        ),
    ),
    Profile(
        "hc_linear",
        "hc",
        dict(attn_res_block_size=2),
        "HC's own default read ('linear'); the Megatron port refuses it",
        # each marker has to match one of the independent statements: the table's
        # `unsupported` notes ("hc_read='linear': ...", "hc_write='linear': ...") and
        # the Megatron build error ("read must be 'simplex' or 'sigmoid', got 'linear'")
        expect_unsupported=("hc_read", "hc_write", "got 'linear'"),
        skip_build=(
            "HcConfig only accepts hc_read in ('simplex', 'sigmoid'), while the HF "
            "default is 'linear' -- the Megatron model cannot be instantiated at all"
        ),
    ),
    Profile(
        "hc_learned",
        "hc",
        dict(attn_res_block_size=2, hc_read="simplex", hc_output_contract="learned"),
        "hyper-connections with the learned block contraction",
        expect_forward_gap=(
            "same write-operator gap as `hc` (HF `hc_write='linear'` vs the port's sigmoid2); "
            "measured delta 1.70e-01"
        ),
    ),
    Profile(
        "mhc",
        "mhc",
        dict(attn_res_block_size=2, hc_read="simplex"),
        "mHC (Sinkhorn H_res)",
    ),
    Profile(
        "mhc_learned",
        "mhc",
        dict(attn_res_block_size=2, hc_read="simplex", hc_output_contract="learned"),
        "mHC with the learned block contraction",
    ),
    Profile(
        "mudd",
        "mudd",
        dict(attn_res_block_size=1, mudd_num_ways=1),
        "single-stream MUDD (the only stream count the port implements)",
    ),
    Profile(
        "mudd_pn",
        "mudd",
        dict(attn_res_block_size=1, mudd_num_ways=1, mudd_pre_norm=True, mudd_post_norm=True),
        "MUDD with the official Pre/PostDANorm switches",
        expect_unsupported=("state_norm",),
    ),
)

_BY_NAME = {p.name: p for p in PROFILES}


def profile(name: str) -> Profile:
    return _BY_NAME[name]
