"""Looma 的 HuggingFace 参考实现（`models/` 里唯一被引擎/评测直接加载的一侧）。

与其他变体一样，两个文件自成一体（Kimi-K3 布局）：``configuration_looma.py`` 只定义
配置，``modeling_looma.py`` 只从它自己的配置模块 import，不引用本配方的任何其他模块。
所以把这两个文件与权重放在一起，``trust_remote_code=True`` 就能加载——这也是
``train/export_hf.py`` 导出时**必须同时拷这两个 .py** 的原因。

    LoomaConfig             LlamaConfig + 求解器/深度连接旋钮（model_type = "looma"）
    LoomaModel              骨干（LlamaModel + 每层的 block 不动点）
    LoomaForCausalLM        LM 头（LlamaForCausalLM，只有 self.model 换成 Looma）
    LoomaDecoderLayer       一个 block：layer_step 解到不动点
    LoomaAttention          LlamaAttention + 迭代间复用首轮的 keys/values
    LoomaAttentionResidual  深度轴上的门控 delta 规则 + 白化 Softmax₁ 读
    layer_step              一次迭代（模块级函数：block 就是它自己的不动点）
    solve_block             不动点求解器（阻尼 + 逐 token 最低残差 + 一步 phantom 梯度）
"""

from .configuration_looma import SOLVER_STOP_MODES, LoomaConfig
from .modeling_looma import (
    LoomaAttention,
    LoomaAttentionResidual,
    LoomaDecoderLayer,
    LoomaForCausalLM,
    LoomaModel,
    LoomaPreTrainedModel,
    LoomaUnweightedRMSNorm,
    layer_step,
    solve_block,
)

__all__ = [
    "SOLVER_STOP_MODES",
    "LoomaAttention",
    "LoomaAttentionResidual",
    "LoomaConfig",
    "LoomaDecoderLayer",
    "LoomaForCausalLM",
    "LoomaModel",
    "LoomaPreTrainedModel",
    "LoomaUnweightedRMSNorm",
    "layer_step",
    "solve_block",
]
