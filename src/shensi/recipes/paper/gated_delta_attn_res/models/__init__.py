"""GDAR 配方的模型层：连接算子 + Megatron 层规格 + HF 参考实现。

- 本目录的 ``gdar_*`` / ``depth_*`` / ``hc_*`` 模块是 Megatron-Core 侧的实现：
  ``GdarTransformerLayer`` 等把深度连接接进标准 TransformerLayer，``gdar_spec`` /
  ``depth_spec`` / ``hc_spec`` / ``ablation_spec`` 给出 ``--spec <module> <object>``
  用的预设（论文主行是 ``gdar_layer_spec_paper``）。
- ``hf/`` 是七个变体的 HF 参考实现（configuration + modeling 成对、自包含），
  基座 Qwen3，tokenizer 也是 Qwen3 同款（配方根的 ``tokenizer/Qwen3-0.6B``）。

spec 预设在这里是**实打实的模块属性**（不是惰性 re-export）：mcore 的
``spec_utils.import_module`` 查 ``vars(module)[name]``，走不到 ``__getattr__``，
所以 ``--spec shensi.recipes.paper.gated_delta_attn_res.models gdar_layer_spec_paper``
必须靠这里的显式导入。首次 import 本包会连带 import mcore/torch，这是刻意为之
——能走到 --spec 的进程本来就要建模型。
"""

from __future__ import annotations

from .ablation_spec import (  # noqa: F401
    ABLATION_GATE_CHANNELS,
    GATED_AR_BLOCK,
    gated_ar_layer_spec,
    gated_ar_layer_spec_block12,
    gated_ar_layer_spec_block16,
    gated_ar_layer_spec_block2,
    gated_ar_layer_spec_block6,
    gated_ar_layer_spec_block8,
    gated_ar_layer_spec_decay_erase,
    gated_ar_layer_spec_decay_only,
    gated_ar_layer_spec_erase_only,
    gated_ar_layer_spec_erase_write,
    gated_ar_layer_spec_no_gate,
    gated_ar_layer_spec_scalar,
    gated_ar_layer_spec_write_decay,
    gated_ar_layer_spec_write_only,
    gdar_half_init_layer_spec,
    gdar_uniform_init_layer_spec,
    make_ablation_spec,
)
from .depth_spec import (  # noqa: F401
    ar_layer_spec,
    ar_layer_spec_block12,
    ar_layer_spec_block2,
    ar_layer_spec_block4,
    ar_layer_spec_block6,
    ar_layer_spec_block8,
    ar_layer_spec_reference,
    dar_layer_spec,
    dar_layer_spec_block12,
    dar_layer_spec_block2,
    dar_layer_spec_block4,
    dar_layer_spec_block6,
    dar_layer_spec_block8,
    dar_layer_spec_null_source,
    dar_layer_spec_reference,
    dar_layer_spec_reference_block4,
    denseformer_layer_spec,
    denseformer_layer_spec_official,
    denseformer_layer_spec_period4,
    make_depth_spec,
    mudd_layer_spec,
    mudd_layer_spec_official_norm,
    mudd_layer_spec_random,
    mudd_layer_spec_reference,
)
from .gdar_spec import (  # noqa: F401
    DESIGN_ABLATIONS,
    gdar_layer_spec,
    gdar_layer_spec_block16,
    gdar_layer_spec_block2,
    gdar_layer_spec_block4,
    gdar_layer_spec_block4_r16,
    gdar_layer_spec_block4_r64,
    gdar_layer_spec_block8,
    gdar_layer_spec_decay_projected,
    gdar_layer_spec_fullrank,
    gdar_layer_spec_gate_delta,
    gdar_layer_spec_gate_prefix,
    gdar_layer_spec_lambda_free,
    gdar_layer_spec_no_output_route,
    gdar_layer_spec_paper,
    gdar_layer_spec_paper_noladder,
    gdar_layer_spec_read_whitened_mix,
    gdar_layer_spec_reference,
    gdar_layer_spec_theory,
    make_gdar_spec,
)
from .hc_spec import hc_layer_spec, mhc_layer_spec  # noqa: F401
