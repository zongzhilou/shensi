"""The 7 depth-routed Qwen3 variants, described once, for the rollout engine.

This is the single source of truth for

* which ``architectures`` string the engine has to dispatch on,
* which ``model_type`` sits in ``config.json`` (so ``trust_remote_code`` finds the
  right pair of files),
* which non-default connection knobs a *tiny* smoke checkpoint must carry so the
  connection is actually switched on (every variant gates its connection on
  ``attn_res_block_size``, whose default is ``None`` = connection disabled).

The knobs mirror ``models/test_autoclass.py``'s ``VARIANTS`` and its ``BASE``
config -- the one combination that is proven to construct, save and reload for all
7 variants.  Two of them are *load-bearing non-defaults*, not decoration:

* ``mudd_num_ways``: the documented values are 4 (qkvr) and 1 (single stream);
  ``2`` raises ``ValueError: not enough values to unpack`` inside the module, so a
  smoke test must avoid it.  ``1`` is the only non-default value that builds.
* ``hc_num_streams``: ``2`` instead of ``4`` only to keep the tiny checkpoint small.

Nothing here imports transformers, vLLM or torch, so it is importable from any of
the three environments in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    """One depth-routed variant, as the inference engine sees it."""

    key: str
    architecture: str
    """``config.json["architectures"][0]`` -- the engine's dispatch key."""

    model_type: str
    """``config.json["model_type"]`` -- what ``auto_map`` resolves against."""

    config_file: str
    config_class: str
    model_file: str
    model_class: str
    base_model_class: str

    tiny_knobs: dict = field(default_factory=dict)
    """Non-default knobs the tiny smoke checkpoint sets (connection ON)."""

    @property
    def auto_map(self) -> dict:
        return {
            "AutoConfig": f"{self.config_file}.{self.config_class}",
            "AutoModel": f"{self.model_file}.{self.base_model_class}",
            "AutoModelForCausalLM": f"{self.model_file}.{self.model_class}",
        }


VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="ar",
        architecture="Qwen3ARForCausalLM",
        model_type="qwen3_ar",
        config_file="configuration_qwen3_ar",
        config_class="Qwen3ARConfig",
        model_file="modeling_qwen3_ar",
        model_class="Qwen3ARForCausalLM",
        base_model_class="Qwen3ARModel",
        tiny_knobs=dict(attn_res_block_size=1, attn_res_output_route=False),
    ),
    Variant(
        key="dar",
        architecture="Qwen3DARForCausalLM",
        model_type="qwen3_dar",
        config_file="configuration_qwen3_dar",
        config_class="Qwen3DARConfig",
        model_file="modeling_qwen3_dar",
        model_class="Qwen3DARForCausalLM",
        base_model_class="Qwen3DARModel",
        tiny_knobs=dict(attn_res_block_size=1, attn_res_use_null_source=True),
    ),
    Variant(
        key="gdar",
        architecture="Qwen3GDARForCausalLM",
        model_type="qwen3_gdar",
        config_file="configuration_qwen3_gdar",
        config_class="Qwen3GDARConfig",
        model_file="modeling_qwen3_gdar",
        model_class="Qwen3GDARForCausalLM",
        base_model_class="Qwen3GDARModel",
        tiny_knobs=dict(
            attn_res_block_size=1,
            attn_res_gate_rank=16,
            attn_res_q_rank=16,
            attn_res_k_rank=16,
            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_address="delta",
            attn_res_decay_ladder=8,
            attn_res_read_heads=2,
            attn_res_read_null=True,
            attn_res_read_whiten="diag",
        ),
    ),
    Variant(
        key="hc",
        architecture="Qwen3HCForCausalLM",
        model_type="qwen3_hc",
        config_file="configuration_qwen3_hc",
        config_class="Qwen3HCConfig",
        model_file="modeling_qwen3_hc",
        model_class="Qwen3HCForCausalLM",
        base_model_class="Qwen3HCModel",
        tiny_knobs=dict(attn_res_block_size=1, hc_num_streams=2),
    ),
    Variant(
        key="mhc",
        architecture="Qwen3MHCForCausalLM",
        model_type="qwen3_mhc",
        config_file="configuration_qwen3_mhc",
        config_class="Qwen3MHCConfig",
        model_file="modeling_qwen3_mhc",
        model_class="Qwen3MHCForCausalLM",
        base_model_class="Qwen3MHCModel",
        tiny_knobs=dict(attn_res_block_size=1, mhc_sinkhorn_iterations=5),
    ),
    Variant(
        key="mudd",
        architecture="Qwen3MUDDForCausalLM",
        model_type="qwen3_mudd",
        config_file="configuration_qwen3_mudd",
        config_class="Qwen3MUDDConfig",
        model_file="modeling_qwen3_mudd",
        model_class="Qwen3MUDDForCausalLM",
        base_model_class="Qwen3MUDDModel",
        tiny_knobs=dict(attn_res_block_size=1, mudd_num_ways=1),
    ),
    Variant(
        key="denseformer",
        architecture="Qwen3DenseFormerForCausalLM",
        model_type="qwen3_denseformer",
        config_file="configuration_qwen3_denseformer",
        config_class="Qwen3DenseFormerConfig",
        model_file="modeling_qwen3_denseformer",
        model_class="Qwen3DenseFormerForCausalLM",
        base_model_class="Qwen3DenseFormerModel",
        tiny_knobs=dict(attn_res_block_size=1, attn_res_dwa_dilation=2),
    ),
)

BY_KEY: dict[str, Variant] = {v.key: v for v in VARIANTS}
BY_ARCH: dict[str, Variant] = {v.architecture: v for v in VARIANTS}
BY_MODEL_TYPE: dict[str, Variant] = {v.model_type: v for v in VARIANTS}


#: Backbone shapes a checkpoint can be built at.  ``tiny`` is the 2-layer smoke
#: shape all 7 variants are checked with; ``0.6b`` is Qwen3-0.6B's own geometry,
#: i.e. the width the real rollout models will have.
SHAPES: dict[str, dict] = {}


def tiny_base(vocab_size: int) -> dict:
    """The tiny backbone all 7 variants share (same numbers as ``test_autoclass.py``)."""
    return dict(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )


def qwen3_0p6b_base(vocab_size: int) -> dict:
    """Qwen3-0.6B's released backbone geometry, same numbers as the real config.

    ``28 x 1024 / 16 heads`` with ``head_dim=128`` (so ``q_proj`` is 2048 wide while
    ``k_proj``/``v_proj`` are 1024 -- GQA 16:8), which is what makes a checkpoint of
    this shape worth running: the attention geometry, the number of KV entries the
    engine allocates and the weight-loading path are all the real ones, unlike the
    2-layer toy.  The weights are random -- this is a *shape* test, not a model.
    """
    return dict(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        tie_word_embeddings=False,
    )


SHAPES.update(tiny=tiny_base, **{"0.6b": qwen3_0p6b_base})


def base_of(shape: str, vocab_size: int) -> dict:
    """``SHAPES[shape]`` applied to a vocab size, with a readable failure."""
    if shape not in SHAPES:
        raise KeyError(f"unknown shape {shape!r}; known: {sorted(SHAPES)}")
    return SHAPES[shape](vocab_size)


__all__ = [
    "Variant",
    "VARIANTS",
    "BY_KEY",
    "BY_ARCH",
    "BY_MODEL_TYPE",
    "tiny_base",
    "qwen3_0p6b_base",
    "SHAPES",
    "base_of",
]
