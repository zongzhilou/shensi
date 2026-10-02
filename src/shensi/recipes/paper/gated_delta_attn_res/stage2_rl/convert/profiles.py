"""转换档位：不同几何与变体的转换配置。"""


from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TINY", "Profile", "PROFILES", "profile", "GDAR_LADDER"]


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


GDAR_LADDER = 8


@dataclass(frozen=True)
class Profile:

    """一个转换档：几何、变体与对应的权重表。"""
    name: str
    variant: str

    knobs: dict = field(default_factory=dict)

    about: str = ""





    expect_unsupported: tuple[str, ...] = ()





    expect_synthesized: tuple[str, ...] = ()






    expect_forward_gap: str | None = None

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


        expect_unsupported=("output router",),
        expect_synthesized=("read_scale", "null_source"),
        expect_forward_gap=(
            "the per-sublayer null sources are real HF tensors and convert exactly, but the "
            "Megatron *output* router prepends one too (synthesised as zeros) where HF's output "
            "read has none, and an extra softmax entry changes the mixture even at zero: measured "
            "delta 2.88e-03, three orders of magnitude above the 3e-07 the other DAR profiles show"
        ),
    ),
    Profile(
        "realformer",
        "realformer",
        dict(attn_res_realformer_gate="deviation"),
        "residual attention scores carried across layers (gate = identity anchor)",
        expect_synthesized=("carry_gate",),
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
    """按名字取转换档。"""
    return _BY_NAME[name]
