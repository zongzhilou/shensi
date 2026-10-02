"""vLLM 侧的几何与变体表（tiny 与 0.6B 基座）。"""


from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:

    """一个几何变体：基座配置 + 连接旋钮。"""
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





SHAPES: dict[str, dict] = {}


def tiny_base(vocab_size: int) -> dict:
    """极小几何基座（冒烟用）。"""
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
    """0.6B 几何基座。"""
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
    """按变体名取基座。"""
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
