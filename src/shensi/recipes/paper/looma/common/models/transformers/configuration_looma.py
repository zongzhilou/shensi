from __future__ import annotations

from typing import Any

from transformers.models.llama.configuration_llama import LlamaConfig

__all__ = ["SOLVER_STOP_MODES", "LoomaConfig"]

# 求解器支持的停机统计量：rel 为 ‖Δ‖/‖x‖，abs 为 ‖Δ‖
SOLVER_STOP_MODES = ("rel", "abs")


def _strict_config(cls):
    """在支持 dataclass 配置的 transformers 上套用 ``strict`` 装饰器，否则原样返回。

    缺了这一步，类上新增的注解会被 dataclass 机制当作无默认值的字段，或从 ``to_dict`` 中
    丢掉；探测整体包在 try 中，因为该钩子在不同版本间有变动。
    """
    try:
        from huggingface_hub.dataclasses import strict

        return strict(cls)
    except Exception:
        return cls


@_strict_config
class LoomaConfig(LlamaConfig):
    """``LlamaConfig`` 加上 Looma 的求解器与深度连接旋钮。

    ``model_type`` 取 ``looma`` 而非 ``llama``：checkpoint 属于另一套架构，按原生 Llama
    加载会忽略 ``looma_*`` 并构造出与权重不匹配的模型，加载方需经 ``auto_map`` 解析建模文件。
    """

    model_type = "looma"

    looma_max_iter: int = 8
    looma_tol: float = 1.0e-2
    looma_stop_mode: str = "rel"
    looma_tau: float = 1.0
    # 1 = 一步 phantom 梯度（求解本身不带图，反向只经过对映射的一次额外求值）；0 = 不带梯度
    looma_grad_steps: int = 1

    looma_rank: int = 64
    looma_read_heads: int = 8
    looma_lambda_clamp: float | None = -0.5
    # write 载体的初始偏置：tanh(0) = 0 会让 write scale 卡死在零，故取 tanh(-4) ≈ -0.999
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
        """``PretrainedConfig.to_dict`` 之外再补上基础实现不会写出的字段。

        类上声明但未赋值的注解字段，以及作为普通类属性的 ``auto_map``，都要显式补进结果，
        否则这类字段在两种配置实现下都进不了 ``config.json``。
        """
        output = super().to_dict()
        for name in (*self.__class__.__annotations__, "auto_map"):
            output.setdefault(name, getattr(self, name, None))
        return output

    auto_map = {
        "AutoConfig": "configuration_looma.LoomaConfig",
        "AutoModel": "modeling_looma.LoomaModel",
        "AutoModelForCausalLM": "modeling_looma.LoomaForCausalLM",
    }

    # ``trust_remote_code`` 需要与 checkpoint 同目录的两个模块文件名
    config_module = "configuration_looma"
    modeling_module = "modeling_looma"


# 注册是尽力而为：导入本模块本身永远不应失败
try:
    from transformers import AutoConfig

    AutoConfig.register("looma", LoomaConfig)
except Exception:
    pass


try:
    LoomaConfig.register_for_auto_class("AutoConfig")
except Exception:
    pass
