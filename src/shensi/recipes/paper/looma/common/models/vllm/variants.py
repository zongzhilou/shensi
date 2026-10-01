"""Looma 在推理引擎侧的静态登记信息：架构名、model_type、tiny 几何与旋钮。

引擎按 ``config.json["architectures"][0]`` 派发到 ``ModelRegistry``，命中登记过的类就用原生实现，
没登记时退回 ``config.json["auto_map"]`` 的远程代码。三张表都只用标准库（不 import
torch/vLLM/transformers），任何进程都能读它。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BY_ARCH", "BY_KEY", "BY_MODEL_TYPE", "VARIANTS", "LoomaVariant", "base_of", "tiny_base"]


@dataclass(frozen=True)
class LoomaVariant:
    """一个可加载的 Looma 档：架构名、model_type、配置/建模文件名。

    ``architecture`` 即 ``config.json["architectures"][0]``，也是 ``ModelRegistry`` 的派发键；
    ``tiny_knobs`` 是 tiny 档要覆盖的非默认旋钮，全取默认值就测不到连接与求解器。
    """

    key: str
    architecture: str
    model_type: str = "looma"
    lm_class: str = "LoomaForCausalLM"
    config_module: str = "configuration_looma"
    modeling_module: str = "modeling_looma"
    tiny_knobs: dict = field(default_factory=dict)

    @property
    def config_file(self) -> str:
        """变体检查点里 config.json 的内容。"""
        return self.config_module

    @property
    def model_file(self) -> str:
        """变体检查点的权重文件名。"""
        return self.modeling_module

    @property
    def auto_map(self) -> dict:
        """变体 config 的 auto_map（让引擎能走远程代码）。"""
        return {
            "AutoConfig": f"{self.config_module}.LoomaConfig",
            "AutoModel": f"{self.modeling_module}.LoomaModel",
            "AutoModelForCausalLM": f"{self.modeling_module}.{self.lm_class}",
        }


LOOMA = LoomaVariant(
    key="looma",
    architecture="LoomaForCausalLM",
    # 每个旋钮都挪离默认：求解器真迭代、低秩路径真走、读真走（钳制与写载体保持默认）
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
    """tiny 骨干几何：2 层 / hidden 64 / 2 头 1 KV / head_dim 32。"""
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
    """几何档：``tiny``（冒烟）或 ``0.6b``（真实规模 rollout）。"""
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
