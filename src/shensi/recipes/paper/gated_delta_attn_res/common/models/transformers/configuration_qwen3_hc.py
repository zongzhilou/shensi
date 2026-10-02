"""HC 的 HF 配置定义。"""


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


__all__ = ["Qwen3HCConfig"]


@_strict_config
class Qwen3HCConfig(Qwen3Config):

    model_type = "qwen3_hc"





    auto_map = {
        "AutoConfig": "configuration_qwen3_hc.Qwen3HCConfig",
        "AutoModel": "modeling_qwen3_hc.Qwen3HCModel",
        "AutoModelForCausalLM": "modeling_qwen3_hc.Qwen3HCForCausalLM",
    }

    def to_dict(self):
        output = super().to_dict()





        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output


    attn_res_block_size: int | None = None


    hc_num_streams: int = 4







    hc_init: str = "identity"


    hc_dynamic: bool = True









    hc_read: str = "linear"


    hc_write: str = "linear"


    mhc_manifold: str = "none"
    mhc_sinkhorn_iterations: int = 10
    mhc_sinkhorn_eps: float = 1e-6
    mhc_init_gating_factor: float = 0.01
    mhc_compute_h_eps: float = 1e-6




    hc_output_contract: str = "mean"
    hc_output_contract_eps: float = 1e-6




    @classmethod
    def identity_preset(cls, **overrides) -> dict:
        preset = dict(
            hc_init="identity",
            hc_dynamic=True,
            hc_read="linear",
            hc_write="linear",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def paper_preset(cls, **overrides) -> dict:
        return cls.identity_preset(**overrides)

    @classmethod
    def megatron_preset(cls, **overrides) -> dict:
        preset = dict(
            hc_init="official",
            hc_dynamic=True,
            hc_read="sigmoid",
            hc_write="sigmoid2",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset

    @classmethod
    def lite_preset(cls, **overrides) -> dict:
        preset = dict(
            hc_init="identity",
            hc_dynamic=False,
            hc_read="linear",
            hc_write="linear",
            mhc_manifold="none",
            hc_output_contract="mean",
        )
        preset.update(overrides)
        return preset



_EXTRA_CONFIG_FIELDS = tuple(Qwen3HCConfig.__annotations__)




try:
    AutoConfig.register("qwen3_hc", Qwen3HCConfig)
except ValueError:
    pass





try:
    Qwen3HCConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
