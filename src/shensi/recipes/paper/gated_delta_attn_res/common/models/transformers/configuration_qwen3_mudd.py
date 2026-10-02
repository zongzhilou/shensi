"""MUDD 的 HF 配置定义。"""


from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def _strict_config(cls):
    try:
        from huggingface_hub.dataclasses import strict
    except ImportError:  # pragma: no cover - huggingface_hub without dataclasses
        return cls
    try:
        return strict(cls)
    except Exception:
        return cls


__all__ = ["Qwen3MUDDConfig"]


@_strict_config
class Qwen3MUDDConfig(Qwen3Config):

    model_type = "qwen3_mudd"





    auto_map = {
        "AutoConfig": "configuration_qwen3_mudd.Qwen3MUDDConfig",
        "AutoModel": "modeling_qwen3_mudd.Qwen3MUDDModel",
        "AutoModelForCausalLM": "modeling_qwen3_mudd.Qwen3MUDDForCausalLM",
    }

    def to_dict(self):
        output = super().to_dict()





        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output


    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True


    mudd_num_ways: int = 4
    mudd_act: str = "gelu"
    mudd_hidden_round: int = 64
    mudd_fix_last_layer: bool = True
    mudd_last_layer_expand: int = 4
    mudd_param: str = "deviation"
    mudd_dw_norm: str = "none"
    mudd_scale_dw: bool = False
    mudd_sepln: bool = True


    mudd_pre_norm: bool = False
    mudd_post_norm: bool = False
    mudd_ffn_depth_scaling: bool = False
    mudd_ffn_round: int = 128

    @classmethod
    def official_preset(cls, **overrides):
        preset = dict(
            mudd_pre_norm=True,
            mudd_post_norm=True,
            mudd_ffn_depth_scaling=True,
        )
        preset.update(overrides)
        return preset



_EXTRA_CONFIG_FIELDS = tuple(Qwen3MUDDConfig.__annotations__)




try:
    AutoConfig.register("qwen3_mudd", Qwen3MUDDConfig)
except ValueError:
    pass





try:
    Qwen3MUDDConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
