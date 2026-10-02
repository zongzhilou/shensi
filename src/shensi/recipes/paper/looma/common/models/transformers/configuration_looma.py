"""Looma 的 HF 配置：连接与求解器旋钮的定义与校验。"""

from __future__ import annotations

from typing import Any

from transformers.models.llama.configuration_llama import LlamaConfig

__all__ = ["SOLVER_STOP_MODES", "LoomaConfig"]


SOLVER_STOP_MODES = ("rel", "abs")


def _strict_config(cls):
    try:
        from huggingface_hub.dataclasses import strict

        return strict(cls)
    except Exception:
        return cls


@_strict_config
class LoomaConfig(LlamaConfig):

    """Looma 的 HF 配置：连接与求解器旋钮的定义与校验。"""
    model_type = "looma"

    looma_max_iter: int = 8
    looma_tol: float = 1.0e-2
    looma_stop_mode: str = "rel"
    looma_tau: float = 1.0

    looma_grad_steps: int = 1

    looma_rank: int = 64
    looma_read_heads: int = 8
    looma_lambda_clamp: float | None = -0.5

    looma_write_carrier_bias: float = -4.0
    looma_output_route: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_looma()

    def _validate_looma(self) -> None:
        if self.looma_stop_mode not in SOLVER_STOP_MODES:
            raise ValueError(
                f"looma_stop_mode must be one of {SOLVER_STOP_MODES}, got {self.looma_stop_mode!r}"
            )
        if int(self.looma_max_iter) < 1:
            raise ValueError(f"looma_max_iter must be >= 1, got {self.looma_max_iter}")
        if float(self.looma_tol) <= 0.0:
            raise ValueError(f"looma_tol must be > 0, got {self.looma_tol}")
        if int(self.looma_rank) < 1:
            raise ValueError(f"looma_rank must be >= 1, got {self.looma_rank}")
        if int(self.looma_read_heads) < 1:
            raise ValueError(f"looma_read_heads must be >= 1, got {self.looma_read_heads}")
        if int(self.looma_grad_steps) not in (0, 1):
            raise ValueError(
                f"looma_grad_steps must be 0 or 1 (one-step phantom gradient), "
                f"got {self.looma_grad_steps}"
            )

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        for name in (*self.__class__.__annotations__, "auto_map"):
            output.setdefault(name, getattr(self, name, None))
        return output

    auto_map = {
        "AutoConfig": "configuration_looma.LoomaConfig",
        "AutoModel": "modeling_looma.LoomaModel",
        "AutoModelForCausalLM": "modeling_looma.LoomaForCausalLM",
    }


    config_module = "configuration_looma"
    modeling_module = "modeling_looma"



try:
    from transformers import AutoConfig

    AutoConfig.register("looma", LoomaConfig)
except Exception:
    pass


try:
    LoomaConfig.register_for_auto_class("AutoConfig")
except Exception:
    pass
