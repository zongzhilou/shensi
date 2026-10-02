# 模型：HuggingFace 参考实现

八个变体的 HF 实现：用于评测、转换与 rollout（训练一律走 mcore 侧）。每个变体都建有数值对拍
闸门，保证与本变体应有的算子语义逐位一致。

## 总览

| 文件 | 变体 | 算子 |
|---|---|---|
| `modeling_qwen3_gdar.py` | GDAR（主行） | 深度轴门控 delta：decay/erase/write + 目标函数闭式更新 + 白化多头读 |
| `modeling_qwen3_ar.py` | AR | 深度读替换子层输入 |
| `modeling_qwen3_dar.py` | DAR | 深度读与输入相加 |
| `modeling_qwen3_denseformer.py` | DenseFormer | 深度加权平均（DWA） |
| `modeling_qwen3_mudd.py` | MUDD | 多路动态稠密 |
| `modeling_qwen3_hc.py` / `modeling_qwen3_mhc.py` | HC / mHC | 多残差流 + 流形投影 |
| `modeling_qwen3_realformer.py` | RealFormer | 跨层累加 softmax 前的注意力分数（可选 running mean） |

每个变体带一个配置类（`configuration_qwen3_*.py`），承载连接旋钮（`attn_res_*`）、注册
`model_type` 并声明 `auto_map`——`save_pretrained` 存下的检查点在**新进程**里用
`trust_remote_code=True` 就能加载（vLLM 与 verl 都走这条路）。

`upstream/` 下是本套实现里几个变体的比照参考件（逐字未改、带各自的许可证，校验见其
`PROVENANCE.md`）；对齐由 `test_upstream_alignment.py` 的数值对拍把关。

## 恒等性

所有变体的连接在初始化时都**逐位**等于普通残差流（构造保证，不是近似）：`GDAR(0) == plain
Qwen3`、AR/DAR 的读门为 0、DenseFormer 的 DWA 是恒等、RealFormer 的加法为 0。于是所有对照臂
从同一个函数出发，只有连接的学习决定差异。

## 快速开始

```bash
cd src/shensi/recipes/paper/gated_delta_attn_res/common/models/transformers
python test_theory.py              # 算子代数、恒等、门语义
python test_ablation_switches.py   # 每个旋钮的效应
python test_autoclass.py           # 配置 / auto_map / 序列化往返
python smoke_test.py               # 8 种配置的前向+反向与路由统计
python test_upstream_alignment.py  # 与参考实现的逐位对拍（13/13）
python test_realformer.py          # 恒等两档、转写对照、增量解码等价
```

## 判据

| 检查 | 结果 |
|---|---|
| 与参考实现对拍 | AR / MUDD / DenseFormer **逐位**；GDAR 在逐头白化配置下**逐位**（13/13） |
| RealFormer | 恒等两档逐位（`0.000e+00`）、gate=1 时在分数/概率/context/carry 四层逐位、running mean 语义、增量解码与全序列一致（核噪声 1.9e-7） |
| 理论 / 消融 / 自动类 | 58/58、64/64、42/42 |
| 前向+反向冒烟 | 8 种配置，回传 `sharpness` / `entropy` / `n_sources` / 门值 |

## 下一步

- [配方 README](../../../README.md) —— 这些实现在哪些环节被用到
- [vLLM rollout](../vllm/README.md) —— 用引擎服务这些模型
- [LIMITATIONS.md](../../../LIMITATIONS.md) —— 已知边界与处置
