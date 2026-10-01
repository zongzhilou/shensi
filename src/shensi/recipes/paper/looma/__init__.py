"""Looma 配方：每个 layer 是一个解到不动点的 block（深度轴上的门控 delta 规则 + 白化 Softmax₁ 读）。

目录约定：stage 目录只放该段自己的入口与配置，其余（模型实现、训练/导出入口、分词器、公共件）
全部在 ``common/`` 下。见 ``README.md``。
"""
