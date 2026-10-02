"""GDAR 的 HF 配置：连接旋钮与消融开关的定义与校验。"""


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


__all__ = ["GATE_CHANNELS", "GATED_AR_SOURCE_BLOCK", "Qwen3GDARConfig"]








GATE_CHANNELS = ("dew", "d", "e", "w", "de", "dw", "ew", "scalar", "none")



GATED_AR_SOURCE_BLOCK = 4


@_strict_config
class Qwen3GDARConfig(Qwen3Config):

    model_type = "qwen3_gdar"





    auto_map = {
        "AutoConfig": "configuration_qwen3_gdar.Qwen3GDARConfig",
        "AutoModel": "modeling_qwen3_gdar.Qwen3GDARModel",
        "AutoModelForCausalLM": "modeling_qwen3_gdar.Qwen3GDARForCausalLM",
    }

    def to_dict(self):
        output = super().to_dict()





        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output


    attn_res_block_size: int | None = None
    attn_res_output_route: bool = True


    attn_res_gate_rank: int | None = None
















    attn_res_gate_init: str = "paper"
    attn_res_gate_init_bias: float = 4.0











    attn_res_gate_channels: str = "dew"





    attn_res_gate_param: str = "sigmoid"





    attn_res_write_carrier_bias: float = -4.0



    attn_res_decay_ladder: int = 0
    attn_res_decay_tau_max: float = 100.0








    attn_res_update: str = "shensi"





    attn_res_read_heads: int = 1




    attn_res_read_null: bool = False




    attn_res_read_whiten: str = "off"
    attn_res_read_ridge: float = 1e-3












    attn_res_address: str = "state"






    attn_res_gate_source: str = "state"






    attn_res_decay_positivity: str = "free"



    attn_res_lambda_clamp: float | None = -0.5



    attn_res_read_mix: str = "raw"

    @classmethod
    def theory_preset(cls, **overrides):
        preset = dict(

            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_decay_ladder=64,
            attn_res_address="delta",

            attn_res_read_heads=8,
            attn_res_read_null=True,
            attn_res_read_whiten="full",

            attn_res_gate_rank=64,
            attn_res_q_rank=64,
            attn_res_k_rank=64,
        )
        preset.update(overrides)
        return preset

    @classmethod
    def gated_ar_preset(cls, **overrides):
        preset = dict(
            attn_res_gate_param="deviation",
            attn_res_gate_channels="dew",
            attn_res_address="state",
            attn_res_block_size=GATED_AR_SOURCE_BLOCK,
        )
        preset.update(overrides)
        return preset


    attn_res_q_rank: int | None = None
    attn_res_k_rank: int | None = None



_EXTRA_CONFIG_FIELDS = tuple(Qwen3GDARConfig.__annotations__)




try:
    AutoConfig.register("qwen3_gdar", Qwen3GDARConfig)
except ValueError:
    pass





try:
    Qwen3GDARConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
