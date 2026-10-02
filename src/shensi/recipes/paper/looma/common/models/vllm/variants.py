"""vLLM 侧的几何与变体表。"""


from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BY_ARCH", "BY_KEY", "BY_MODEL_TYPE", "VARIANTS", "LoomaVariant", "base_of", "tiny_base"]


@dataclass(frozen=True)
class LoomaVariant:

    """一个几何变体：基座配置 + 连接旋钮。"""
    key: str
    architecture: str
    model_type: str = "looma"
    lm_class: str = "LoomaForCausalLM"
    config_module: str = "configuration_looma"
    modeling_module: str = "modeling_looma"
    tiny_knobs: dict = field(default_factory=dict)

    @property
    def config_file(self) -> str:
        return self.config_module

    @property
    def model_file(self) -> str:
        return self.modeling_module

    @property
    def auto_map(self) -> dict:
        return {
            "AutoConfig": f"{self.config_module}.LoomaConfig",
            "AutoModel": f"{self.modeling_module}.LoomaModel",
            "AutoModelForCausalLM": f"{self.modeling_module}.{self.lm_class}",
        }


LOOMA = LoomaVariant(
    key="looma",
    architecture="LoomaForCausalLM",

    tiny_knobs=dict(
        looma_max_iter=4,
        looma_tol=1e-3,
        looma_rank=8,
        looma_read_heads=1,
    ),
)

VARIANTS: dict[str, LoomaVariant] = {LOOMA.key: LOOMA}
BY_KEY = dict(VARIANTS)
BY_ARCH = {v.architecture: v for v in VARIANTS.values()}
BY_MODEL_TYPE = {v.model_type: v for v in VARIANTS.values()}


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


def base_of(shape: str, vocab_size: int) -> dict:
    """按变体名取基座。"""
    if shape == "tiny":
        return tiny_base(vocab_size)
    if shape == "0.6b":
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
    raise ValueError(f"unknown shape {shape!r}（可用：tiny / 0.6b）")
