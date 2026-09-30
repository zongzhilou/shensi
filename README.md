# Shensi 训练仓

**DeepSeek-V4-Flash 路线的 Shensi 模型族**：Megatron-Bridge 侧的新模型贡献（`models/shensi/`）、
本地单机 1~8 卡的训练配方（recipes）、模型与第三方库的对接点（`shensi.runtime`）与评测入口。

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## 仓库总览

```
shensi/
│
├── src/shensi/           包本体
│   ├── runtime.py        与 Megatron-Bridge / verl 的对接点（登记注册表、一个上游缺陷的判断）
│   ├── utils/            ckpt 摘要、AdEMAMix 注册适配
│   └── recipes/          训练配方（从原始数据到成品模型的完整流水线）
│       └── shensi/
│           ├── train/    本地训练运行时：入口 + 上游 mcore 训练循环 + torchrun launcher
│           ├── config/   冒烟档（tiny.yaml）与 HF 参考几何（hf/9b_a4b.json）
│           └── stage0_pretrain/  stage1_sft/  stage2_rl/  stage3_eval/
│
├── patches/              上游二进制包在本机的补丁（只在 README 说明里使用，不进 3rdparty）
└── 3rdparty/             上游依赖（git 子模块，只克隆不修改）：common / ascend
```

Shensi 的模型实现**不在本仓**：它按官方文档贡献进了
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 的 `src/megatron/bridge/models/shensi/`。
本仓 `3rdparty/common/Megatron-Bridge` 指向**[我们 fork 的 `shensi` 分支](https://github.com/zongzhilou/Megatron-Bridge/tree/shensi)**
（三个提交：模型与桥 + 两处修复，PR 直接从这个分支开）；`/home/louzo/code/shensi/Megatron-Bridge` 那份克隆是同一份代码，
用来迭代与提 PR。本仓只留：配方、运行时对接、以及"怎么在本机把整条链路跑起来"的说明。

### 该用哪一块？

| | **训练配方** | **Bridge 的 models/shensi** | **shensi.runtime** | **shensi/utils** |
|---|---|---|---|---|
| **用途** | 按阶段复现训练（预训练 → SFT → RL → 评测） | 给 Bridge 提"支持 Shensi"：模型、层规格、mHC/AttnRes、HF↔Megatron 桥 | 让第三方（verl / 训练入口）拿到注册表与必要判断 | ckpt 摘要、优化器适配、DSA 内核口径 |
| **形态** | Python 包 + YAML 配置 + 命令行 | 与上游同路径的模块（提交在 fork 的 `shensi` 分支） | 一个模块，导入即生效 | 独立模块 |
| **位置** | [`src/shensi/recipes/`](src/shensi/recipes/) | `3rdparty/common/Megatron-Bridge/src/megatron/bridge/models/shensi/` | [`src/shensi/runtime.py`](src/shensi/runtime.py) | [`src/shensi/utils/`](src/shensi/utils/) |

上游库按原样使用（`import megatron` / `import megatron.bridge` / `import verl`），不往它们的目录里写文件。

---

## 模型

Shensi 是 DeepSeek-V4-Flash 路线的 MoE 混合注意力模型，几何与 HF 实现
（`transformers/models/shensi`）逐字段一致，权威参考是
[`src/shensi/recipes/shensi/config/hf/9b_a4b.json`](src/shensi/recipes/shensi/config/hf/9b_a4b.json)：

- 35 个主干层 + 3 个 MTP 层；`hidden 2560`，32 头，`head_dim 512`（`qk_rope 64`），`q_lora 640`
- 混合注意力：CSA（压缩稀疏注意力，16 层）+ HCA（重压缩，15 层）+ 滑窗（4 层）；
  DSA Lightning Indexer 挂在 CSA 层上（`index_topk 512`）
- mHC 多流超连接（`hc_mult 16`，活跃 4 / 固定 2）+ AttnRes 深度连接（`block_size 4`）
- MoE：256 专家选 6（`sqrtsoftplus`，`routed_scaling 1.5`），低秩专家 `rank 640`，前 3 层是 hash-MoE
- 9.150B 总参 / 4.235B 每 token 激活；上下文 1M；词表 129280
- 三个 loss：路由 aux（0.001）+ ERC（1.0 / α 0.5）+ DSA indexer KL（0.01）
- 优化器：Muon（矩阵参数）+ AdEMAMix（标量参数）混合，对齐 V4 / GLM-5 / Kimi K2 口径

HF ↔ Megatron 的转换、以及前向数值由 Bridge 侧的测试保证：AttnRes 与 HF 逐位一致（CPU/fp32），
tiny 权重双向转换无缺参，前向出有限 logits。

---

## 训练配方

配方是"从原始数据到成品模型"的完整流水线：数据准备 → 训练 → 评测，各阶段有自己的配置、脚本与 README。

| 阶段 | 内容 | 走哪套栈 | 指南 |
|------|------|----------|------|
| **stage0 预训练** | 主预训练（9B/A4B 全量档 + tiny 档）→ 中训（DSA warmup / MTP draft）→ 长上下文（1M） | 本仓 `train/`（上游 mcore 训练循环） | [README](src/shensi/recipes/shensi/stage0_pretrain/README.md) |
| **stage1 SFT** | 多域指令微调（DSV4 chat 版式，含编码/推理/工具调用） | 本仓 `train/`（mcore `--sft`） | [README](src/shensi/recipes/shensi/stage1_sft/README.md) |
| **stage2 RL** | RLVR（GRPO，21 环境可验证奖励）→ agentic → 对齐 → 世界模型 | verl + mcore actor + vLLM rollout | [README](src/shensi/recipes/shensi/stage2_rl/README.md) |
| **stage3 评测** | vLLM 起服务 + 基准评测 | vLLM | [README](src/shensi/recipes/shensi/stage3_eval/README.md) |

每个 stage 目录里是同样的四件事：

- `train.py` / `eval.py`：入口（`--profile` 选档、`--dry-run` 只打印命令、`--set` 点号覆写、`--smoke` 跑仓库内 tiny 档）
- `data_prep.py`：语料准备（`--discover` 看面貌、`--prepare` 产出训练用的 bin/idx 或 parquet）
- `config/`：`default.yaml`（全量档）+ `debug.yaml`（极小档，几步就能验证链路）+ 数据配比 json
- `README.md`：这个阶段的数据口径、超参对照、判据与踩过的坑

配方层共用两个模块：`recipes/shensi/common.py`（路径/配置/启动/语料准备）与
`recipes/shensi/rl.py`（yaml → verl CLI 的映射与启动）；训练侧的运行时在
[`recipes/shensi/train/`](src/shensi/recipes/shensi/train/)（`launcher.py` 摊平配置并起 torchrun，
`train_shensi.py` 是等价于上游 `pretrain_gpt.py` 的入口，模型来自 Bridge）。

---

## 快速开始

```bash
git submodule update --init --recursive      # 3rdparty 子模块（mcore / Bridge / verl / vllm / transformers / 昇腾四件套）

uv sync                                      # 上游依赖：mcore 按可编辑方式装（见"环境"一节）
uv pip install --no-deps -e /path/to/Megatron-Bridge    # 本地那份 Bridge（含 models/shensi）
```

`megatron-bridge` 不进 `[tool.uv.sources]`：它自己的 `pyproject.toml` 把 `megatron-core` 指到
`3rdparty/Megatron-LM/`（一个未初始化的子模块目录），uv 解算时会直接失败——所以按上面那一行单独装。
`transformers` / `vllm` / `verl` 这些重编的包，本机是用 `uv pip install --no-deps <path>` 装的
（vllm 与 TransformerEngine 从源码编一次要几十分钟，见 `src/README.md`）。

冒烟（不需要任何真实语料：2 层 / hidden 128 / mock 数据 / 5 步）：

```bash
cd src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain
python train.py --smoke                      # = recipes/shensi/config/tiny.yaml
```

真实语料上的极小档：

```bash
python data_prep.py --discover                 # 语料面貌
python data_prep.py --prepare --blend config/data_prep/debug_sample.json
python train.py --profile debug                # 极小档真跑几步
python train.py --profile debug --set train.model.train_iters=200 --early-stop 20
```

RL（verl + Megatron actor + vLLM）的极小档：

```bash
cd src/shensi/recipes/shensi/stage2_rl/stage1_rlvr
python train.py --profile debug --data-dir <data_prep 产物目录> --set model.path=<HF ckpt>
```

---

## 环境与已知限制

- **上游依赖只在 `3rdparty/` 克隆**：安装方式、版本与本地补丁见 [`src/README.md`](src/README.md) 与
  `pyproject.toml` 的注释；本仓不往它们的目录里写文件。
- **mcore 与 Megatron-Bridge 按可编辑方式装**（`pyproject.toml` 的 `[tool.uv.sources]`）：
  mcore 的 `core/datasets/Makefile` 不进 wheel，而 `compile_helpers()` 会在起训时 `make -C <包目录>`
  （缺 Makefile 直接 `sys.exit`）；顺带让 3rdparty 里的改动即时生效。
- **`pybind11` / `ninja` / `fast-hadamard-transform`** 是 mcore 的运行时依赖：前两个用于现场编 dataset
  helper，最后一个是 DSA indexer 的 Hadamard 旋转（mcore 的硬依赖）。
  `fast-hadamard-transform` 的上游 `setup.py` 没有 SM120 的 `-gencode`，在本机（RTX 50 系）起训会报
  `no kernel image is available`：用 [`patches/fast-hadamard-transform-sm120.patch`](patches/) 打一次补丁再本地编
  （见 [src/README.md](src/README.md) 的"本机补丁"一节）。
- **`TE_FL_PREFER=vendor`**：本机这套 TransformerEngine-FL / FlagGems 的 flagos 后端在 SM120 上会让前向
  段错误（`te_general_grouped_gemm`），launcher 默认改成 TE 自带 CUDA kernel；`CUDA_HOME` 也按需指向
  `/usr/local/cuda`（flashinfer 在 SM120 上要 JIT 补稀疏 MLA 内核）。
- **`shensi.runtime` 是唯一的对接点**：登记 Bridge 的桥表与 `emerging_optimizers` 的标量优化器表，并给
  verl 的一个缺陷补判断（`use_distributed_optimizer=False` 时 `param_data` 为空的解引用）。verl 侧按它
  自己的约定用 `VERL_USE_EXTERNAL_MODULES=shensi.runtime` 加载。
- **单机口径**：配方按 1~8 卡写，`experiment.runner.nproc_per_node` / `tensor_model_parallel_size` 等按机器
  改；多机需要自己接 launcher（`train/launcher.py` 只跑 `nnodes=1`）。
- **RL（stage2_rl）的注意力口径**：mcore 的 CSA 不接受显式 mask，而 verl 会把 response 右 padding
  到 `max_response_length`；Bridge 的 `ShensiModel.forward` 丢掉纯右 padding 的 mask、拒绝左
  padding。`stage1_rlvr` 的 debug 档已跑到 `step:1`，但尾部 pad 仍会通过压缩块参与计算（与 FL fork
  同口径）。详见 [`src/shensi/recipes/shensi/README.md`](src/shensi/recipes/shensi/README.md) 的
  "局限"第 4 条。
- 规模、昇腾路径与"登记未接"的清单见同一份 README 的"局限"一节。

---

## 许可

Apache 2.0，见 [LICENSE](LICENSE)。
