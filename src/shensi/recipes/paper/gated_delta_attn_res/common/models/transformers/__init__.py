"""HF 参考实现与配置的导出（AutoConfig / AutoModel 分发）。"""


from .modeling_qwen3_ar import (
    Qwen3ARConfig,
    Qwen3ARDecoderLayer,
    Qwen3ARForCausalLM,
    Qwen3ARModel,
)
from .modeling_qwen3_dar import (
    Qwen3DARConfig,
    Qwen3DARDecoderLayer,
    Qwen3DARForCausalLM,
    Qwen3DARModel,
)
from .modeling_qwen3_gdar import (
    Qwen3GDARConfig,
    Qwen3GDARDecoderLayer,
    Qwen3GDARForCausalLM,
    Qwen3GDARModel,
)

from .modeling_qwen3_hc import Qwen3HCConfig, Qwen3HCForCausalLM
from .modeling_qwen3_mhc import Qwen3MHCConfig, Qwen3MHCForCausalLM
from .modeling_qwen3_mudd import Qwen3MUDDConfig, Qwen3MUDDForCausalLM
from .modeling_qwen3_denseformer import Qwen3DenseFormerConfig, Qwen3DenseFormerForCausalLM

__all__ = [
    "Qwen3ARDecoderLayer",
    "Qwen3DARDecoderLayer",
    "Qwen3GDARDecoderLayer",
    "Qwen3ARConfig",
    "Qwen3ARModel",
    "Qwen3ARForCausalLM",
    "Qwen3DARConfig",
    "Qwen3DARModel",
    "Qwen3DARForCausalLM",
    "Qwen3GDARConfig",
    "Qwen3GDARModel",
    "Qwen3GDARForCausalLM",
    "Qwen3HCConfig",
    "Qwen3HCForCausalLM",
    "Qwen3MHCConfig",
    "Qwen3MHCForCausalLM",
    "Qwen3MUDDConfig",
    "Qwen3MUDDForCausalLM",
    "Qwen3DenseFormerConfig",
    "Qwen3DenseFormerForCausalLM",
]
