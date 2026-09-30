# Shensi 训练仓

**DeepSeek-V4-Flash 路线的 Shensi 模型族**：训练配方（recipes）、Megatron 侧增量实现（`megatron_ext`）、
FlagScale 侧增量（`flagscale_ext`）与评测入口。单机 1~8 卡可跑，不依赖集群 launcher。

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## 仓库总览

```
shensi/
│
├── src/megatron_ext/     mcore / Megatron-Bridge 侧增量：Shensi 模型、AttnRes、mHC、MoE、bridge
├── src/flagscale_ext/    FlagScale 侧增量：训练入口 + 示例配置
├── src/shensi/           包本体：runtime（与第三方对接）、utils、recipes
│   └── recipes/          训练配方（从原始数据到成品模型的完整流水线）
│
└── 3rdparty/             上游依赖（git 子模块，只克隆不修改）：common / nvidia / ascend
```

### 该用哪一块？

| | **训练配方** | **megatron_ext** | **flagscale_ext** | **shensi/utils** |
|---|---|---|---|---|
| **用途** | 按阶段复现训练（预训练 → SFT → RL → 评测） | 给 mcore/Bridge 提"支持 Shensi"的最小增量 | 给 FlagScale 提"支持 Shensi"的入口与配置 | ckpt 摘要、优化器适配等自有工具 |
| **形态** | Python 包 + YAML 配置 + 命令行 | 与上游同路径的模块，直接 `from megatron_ext...` 导入 | FlagScale 的 `entrypoint` 与 experiment 配置 | 独立模块 |
| **位置** | [`src/shensi/recipes/`](src/shensi/recipes/) | [`src/megatron_ext/`](src/megatron_ext/) | [`src/flagscale_ext/`](src/flagscale_ext/) | [`src/shensi/utils/`](src/shensi/utils/) |

上游库按原样使用（`import megatron` / `import flagscale` / `import verl`），我们的实现**不合并进**
它们的命名空间：第三方需要的那几个模块与符号，由 `shensi.runtime` 在运行时显式登记。

---

## 模型

Shensi 是 DeepSeek-V4-Flash 路线的 MoE 混合注意力模型，几何按 HF 实现（`transformers/models/shensi`）计算：

- 35 个主干层 + 1 个 MTP 层；`hidden 4096`，64 头，`head_dim 512`（`qk_rope 64`）
- 混合注意力：CSA（压缩稀疏注意力）+ DSA（Lightning Indexer）、HCA（重压缩）+ 滑动窗口
- mHC 多流超连接（`hc_mult 16`，注意力/MLP 各一套）、AttnRes 深度连接
- MoE：256 专家选 6，共享专家 1，`moe_intermediate 2048`，`routed_scaling 2.5`
- 9.150B 总参 / 4.235B 每 token 激活；上下文 1M；词表 129280
- 标量优化器 AdEMAMix（`emerging_optimizers` 注册名 `ademamix`）+ Muon 混合
- 权重版式与 mcore 逐位对齐（HF ↔ mcore 对拍脚本见 `src/megatron_ext/tests/`）

---

## 训练配方

配方是"从原始数据到成品模型"的完整流水线：数据准备 → 训练 → 评测，各阶段有自己的配置、脚本与 README。

| 阶段 | 内容 | 走哪套栈 | 指南 |
|------|------|----------|------|
| **stage0 预训练** | 主预训练（9B/A4B 全量档 + tiny 档）→ 中训（DSA warmup / MTP draft）→ 长上下文（1M） | FlagScale + mcore | [README](src/shensi/recipes/shensi/stage0_pretrain/README.md) |
| **stage1 SFT** | 多域指令微调（DSV4 chat 版式，含编码/推理/工具调用） | FlagScale + mcore | [README](src/shensi/recipes/shensi/stage1_sft/README.md) |
| **stage2 RL** | RLVR（GRPO，21 环境可验证奖励）→ agentic → 对齐 → 世界模型 | verl + mcore actor + vLLM rollout | [README](src/shensi/recipes/shensi/stage2_rl/README.md) |
| **stage3 评测** | vLLM 起服务 + 基准评测 | vLLM | [README](src/shensi/recipes/shensi/stage3_eval/README.md) |

每个 stage 目录里是同样的四件事：

- `train.py` / `eval.py`：入口（`--profile` 选档、`--dry-run` 只打印命令、`--set` 点号覆写）
- `data_prep.py`：语料准备（`--discover` 看面貌、`--prepare` 产出训练用的 bin/idx 或 parquet）
- `config/`：`default.yaml`（全量档）+ `debug.yaml`（极小档，几步就能验证链路）+ 数据配比 json
- `README.md`：这个阶段的数据口径、超参对照、判据与踩过的坑

配方层共用两个模块：`recipes/shensi/common.py`（路径/配置/FlagScale 命令/语料准备）与
`recipes/shensi/rl.py`（yaml → verl CLI 的映射与启动）。

---

## 快速开始

```bash
git submodule update --init --recursive      # 3rdparty 子模块
uv sync                                      # 装依赖（详见 docs 里的环境说明）
uv pip install --no-deps 3rdparty/common/FlagScale   # FlagScale 用 pip 装（见 README 的说明）

cd src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain
python data_prep.py --discover                 # 语料面貌
python data_prep.py --prepare --blend config/data_prep/debug_sample.json
python train.py --profile debug                # 极小档真跑几步
```

RL（verl + Megatron actor + vLLM）的极小档：

```bash
cd src/shensi/recipes/shensi/stage2_rl/stage1_rlvr
python train.py --profile debug --data-dir <data_prep 产物目录> --set model.path=<HF ckpt>
```

---

## 环境与已知限制

- **上游依赖只在 `3rdparty/` 克隆**：安装、版本与本地补丁在
  [`src/README.md`](src/README.md) 与各 stage README 里说明；本仓不往它们的目录里写文件。
- **`shensi.runtime` 是唯一的对接点**：登记 `megatron_ext` 里第三方要 import 的模块、补齐 FL fork 缺的
  mcore-main 符号、打两个必要补丁（verl 的 flat-buffer 判空、WSL 的 no-IPC CUDA 平台）。verl 侧按它自己的
  约定用 `VERL_USE_EXTERNAL_MODULES=shensi.runtime` 加载。
- **单机口径**：配方按 1~8 卡写，`trainer.n_gpus_per_node` / `tensor_model_parallel_size` 等按机器改；
  多机需要自己接 launcher。
- 规模、昇腾路径与"登记未接"的清单见 [`src/shensi/recipes/shensi/README.md`](src/shensi/recipes/shensi/README.md)
  的"局限"一节。

---

## 许可

Apache 2.0，见 [LICENSE](LICENSE)。
