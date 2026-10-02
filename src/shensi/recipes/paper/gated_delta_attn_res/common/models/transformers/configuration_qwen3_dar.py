"""DAR 的 HF 配置定义。"""


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


__all__ = ["Qwen3DARConfig"]


@_strict_config
class Qwen3DARConfig(Qwen3Config):

    model_type = "qwen3_dar"





    auto_map = {
        "AutoConfig": "configuration_qwen3_dar.Qwen3DARConfig",
        "AutoModel": "modeling_qwen3_dar.Qwen3DARModel",
        "AutoModelForCausalLM": "modeling_qwen3_dar.Qwen3DARForCausalLM",
    }

    def to_dict(self):
        output = super().to_dict()





        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output

    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True
    attn_res_use_null_source: bool = False



_EXTRA_CONFIG_FIELDS = tuple(Qwen3DARConfig.__annotations__)




try:
    AutoConfig.register("qwen3_dar", Qwen3DARConfig)
except ValueError:
    pass





try:
    Qwen3DARConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
