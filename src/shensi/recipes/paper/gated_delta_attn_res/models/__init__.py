"""GDAR 配方的模型层：按框架分成三个自包含子树，各自对齐各自上游仓库的风格与注释。

- ``megatron/``     Megatron-Core 侧（``--spec`` 预设、深度连接层与消融臂；风格对齐 megatron-core）
- ``transformers/`` HF 参考实现（七个变体的 configuration/modeling 成对文件 + 单测；风格对齐 transformers）
- ``vllm/``         vLLM 推理侧（架构注册、Transformers-后端桥接、tiny ckpt 与 smoke；风格对齐 vllm）

``--spec`` 的写法（mcore 的 ``spec_utils.import_module`` 查 ``vars(module)[name]``，所以只能指到
真正持有该属性的模块）::

    --spec shensi.recipes.paper.gated_delta_attn_res.models.megatron.gdar_spec gdar_layer_spec_paper
    --spec shensi.recipes.paper.gated_delta_attn_res.models.megatron          gdar_layer_spec_paper

**本文件刻意不 import 任何子包**：vLLM 的引擎 worker 进程会 import ``models.vllm.*``，那里不该
被拖进 megatron/torch 的重依赖（三个子树因此各自独立，风格也各自独立）。
"""
