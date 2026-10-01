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
│   ├── utils/            ckpt 摘要、昇腾装配自查、DSA 内核口径、优化器接入（AdaMuon + AdEMAMix）
│   └── recipes/          训练配方（从原始数据到成品模型的完整流水线）
│       └── shensi/
│           ├── common/   公共件：路径与配置、启动与看门狗、语料、极小档、verl 接线
│           │   ├── train/      本地训练运行时：入口 + 上游 mcore 训练循环 + torchrun launcher
│           │   └── config/     配方级冒烟档（tiny.yaml）与 HF 参考几何（hf/9b_a4b.json）
│           ├── stage0_pretrain/  stage1_sft/  stage2_rl/  stage3_eval/
│           └── （每段：train.py / eval.py / data_prep.py / config/ / test_train.py / README.md）
│
├── patches/              上游二进制包在本机的补丁（用的时候 clone 上游源码再打，不进 3rdparty）
└── 3rdparty/             上游依赖（git 子模块，只克隆不修改）：common / ascend
```

Shensi 的模型实现**不在本仓**：它按官方文档贡献进了
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 的 `src/megatron/bridge/models/shensi/`。
本仓 `3rdparty/common/Megatron-Bridge` 指向我们 fork 的 `shensi` 分支（按贡献文档的布局组织、
带 DCO sign-off，PR 直接从这个分支开）；`/home/louzo/code/shensi/Megatron-Bridge` 那份克隆是同一份
代码，用来迭代与提 PR。本仓只留：配方、运行时对接、以及"怎么在本机把整条链路跑起来"的说明。

### 该用哪一块？

| | **训练配方** | **Bridge 的 models/shensi** | **shensi.runtime** | **shensi/utils** |
|---|---|---|---|---|
| **用途** | 按阶段复现训练（预训练 → SFT → RL → 评测） | 给 Bridge 提"支持 Shensi"：模型、层规格、mHC/AttnRes、HF↔Megatron 桥 | 让第三方（verl / 训练入口）拿到注册表与必要判断 | ckpt 摘要、优化器接入、DSA 内核口径、昇腾自查 |
| **形态** | Python 包 + YAML 配置 + 命令行 | 与上游同路径的模块（提交在 fork 的 `shensi` 分支） | 一个模块，导入即生效 | 独立模块 |
| **位置** | [`src/shensi/recipes/`](src/shensi/recipes/) | `3rdparty/common/Megatron-Bridge/src/megatron/bridge/models/shensi/` | [`src/shensi/runtime.py`](src/shensi/runtime.py) | [`src/shensi/utils/`](src/shensi/utils/) |

上游库按原样使用（`import megatron` / `import megatron.bridge` / `import verl`），不往它们的目录里写文件。

---

## 模型

Shensi 是 DeepSeek-V4-Flash 路线的 MoE 混合注意力模型，几何与 HF 实现
（`transformers/models/shensi`）逐字段一致，权威参考是
[`src/shensi/recipes/shensi/common/config/hf/9b_a4b.json`](src/shensi/recipes/shensi/common/config/hf/9b_a4b.json)：

- 35 个主干层 + 3 个 MTP 层；`hidden 2560`，32 头，`head_dim 512`（`qk_rope 64`），`q_lora 640`
- 混合注意力：CSA（压缩稀疏注意力，16 层）+ HCA（重压缩，15 层）+ 滑窗（4 层）；
  DSA Lightning Indexer 挂在 CSA 层上（`index_topk 512`）
- mHC 多流超连接（`hc_mult 16`，活跃 4 / 固定 2）+ AttnRes 深度连接（`block_size 4`）
- MoE：256 专家选 6（`sqrtsoftplus`，`routed_scaling 1.5`），低秩专家 `rank 640`，前 3 层是 hash-MoE
- 9.150B 总参 / 4.235B 每 token 激活；上下文 1M；词表 129280
- 三个 loss：路由 aux（0.001）+ ERC（1.0 / α 0.5）+ DSA indexer KL（0.01）
- 优化器：AdaMuon（矩阵腿）+ AdEMAMix（标量腿）混合；标量腿也可换 GrokFastAdamW
  （见 [`src/shensi/recipes/shensi/README.md`](src/shensi/recipes/shensi/README.md) 的「优化器」一节）

HF ↔ Megatron 的转换、以及前向数值由 Bridge 侧的测试保证：AttnRes 与 HF 逐位一致（CPU/fp32），
tiny 权重双向转换无缺参，前向出有限 logits。

---

## 训练配方

配方是"从原始数据到成品模型"的完整流水线：数据准备 → 训练 → 评测，各阶段有自己的配置、脚本与 README。

| 阶段 | 内容 | 走哪套栈 | 指南 |
|------|------|----------|------|
| **stage0 预训练** | 主预训练（9B/A4B 全量档 + tiny 档）→ 中训（DSA warmup / MTP draft）→ 长上下文（1M） | 本仓 `common/train/`（上游 mcore 训练循环） | [README](src/shensi/recipes/shensi/stage0_pretrain/README.md) |
| **stage1 SFT** | 多域指令微调（DSV4 chat 版式，含编码/推理/工具调用） | 本仓 `common/train/`（mcore `--sft`） | [README](src/shensi/recipes/shensi/stage1_sft/README.md) |
| **stage2 RL** | RLVR（GRPO，可验证奖励）→ agentic → 对齐 → 世界模型 | verl + mcore actor + vLLM rollout | [README](src/shensi/recipes/shensi/stage2_rl/README.md) |
| **stage3 评测** | vLLM 起服务 + 基准评测（LLM 基准走 OpenCompass） | vLLM + OpenCompass / local / harness / Gym / MRCR | [README](src/shensi/recipes/shensi/stage3_eval/README.md) |

每个 stage 目录里是同样的几件事：

- `train.py` / `eval.py`：入口（`--profile` 选档、`--dry-run` 只打印命令、`--set` 点号覆写、`--smoke` 跑仓库内 tiny 档）
- `data_prep.py`：语料准备（`--discover` 看面貌、`--prepare` 产出训练用的 bin/idx 或 parquet）
- `config/`：`default.yaml`（全量档）+ `tiny.yaml`（冒烟档，mock 数据）+ `debug.yaml`（极小几何 + 真实数据）+ 数据配比 json
- `test_train.py`：该段的集成门（极小几何跑几步，按日志判 PASS/FAIL）
- `README.md`：这个阶段的数据口径、超参、判据与踩过的坑

配方层共用的模块都在 [`recipes/shensi/common/`](src/shensi/recipes/shensi/common/)：
`common.py`（路径/配置合并/语料准备）、`rl.py`（yaml → verl CLI 的映射与启动）、`early_stop.py`（日志看门狗）、
`tiny_model.py`（极小几何的单一出处）、`tiny_test.py`（集成测试公共件）、`harness.py`、`codev3.py`，
以及训练运行时 [`common/train/`](src/shensi/recipes/shensi/common/train/)（`launcher.py` 摊平配置并起 torchrun，
`train_shensi.py` 是等价于上游 `pretrain_gpt.py` 的入口，模型来自 Bridge）。

---

## 快速开始

```bash
git submodule update --init --recursive      # 3rdparty 子模块（mcore / Bridge / verl / vllm / transformers / 昇腾四件套）

uv sync                                      # 上游依赖：mcore 与 Bridge 按可编辑方式装（见"装环境"）
```

冒烟（不需要任何真实语料：2 层 / hidden 128 / mock 数据 / 5 步）：

```bash
cd src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain
python train.py --smoke                      # = --profile tiny
```

真实语料上的极小档：

```bash
python data_prep.py --discover                 # 语料面貌
python data_prep.py --prepare --config config/data_prep/tiny.yaml
python train.py --profile debug                # 极小档真跑几步
python train.py --profile debug --set train.model.train_iters=200 --early-stop 20
```

RL（verl + Megatron actor + vLLM）的极小档：

```bash
cd src/shensi/recipes/shensi/stage2_rl/stage1_rlvr
python train.py --profile tiny --data-dir <data_prep 产物目录> --set model.path=<HF ckpt>
```

每段自带一份自检入口（几何换成 `common/tiny_model.py` 那份，mock 数据；`--profile tiny` 的冒烟档也走它）：

```bash
cd src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain
python test_train.py                      # 真跑 5 步 + 存 ckpt，日志里判 PASS/FAIL
python test_train.py --iters 10           # 想多跑几步
python ../../stage2_rl/stage1_rlvr/test_train.py   # RL 四段与评测走 preflight（拼命令 + 校验数据与 CLI）
```

---

## 装环境

### NVIDIA / CUDA 机（本机实际顺序）

```bash
uv lock                                    # 解算清单（验证 pyproject / source 成立）

# uv sync 是 exact 的：不在 lock 里的包会被卸掉，所以先 sync、后补这几个"本机另编/另装"的
uv sync --no-install-package vllm --no-install-package transformer-engine \
        --no-install-package fast-hadamard-transform
uv pip install --no-deps -e /path/to/Megatron-Bridge            # 本地那份 Bridge（含 models/shensi）
uv pip install transferqueue                                    # verl-core extra 里那个（RL 要用）
# 然后编 vllm 与 hadamard（见下），TE 见下
```

`megatron-bridge` 不进 `[tool.uv.sources]`：它自己的 `pyproject.toml` 把 `megatron-core` 指到
`3rdparty/Megatron-LM/`（一个未初始化的子模块目录），uv 解算 Bridge 的元数据时会直接失败——
所以 Bridge 按上面那一行单独装。`transformers` / `vllm` / `verl` 这些重编的包，本机是用
`uv pip install --no-deps <path>` 装的。

**vllm（本机实测可用的编法，约 40 分钟）**：

```bash
rm -rf 3rdparty/common/vllm/build          # 清掉被污染的 CMake 缓存（见下）
VLLM_VERSION_OVERRIDE=0.30.1rc0.dev360+g54c5060a1 MAX_JOBS=8 NVCC_THREADS=1 \
  uv pip install --no-build-isolation --no-deps --reinstall 3rdparty/common/vllm
```

四处绕行，都是上游打包 + 本机环境决定的，不是偏好：

- **vllm 的版本号必须带 `VLLM_VERSION_OVERRIDE`**：fork 只有分支没有 tag，vcs-versioning 会算出
  `0.1.devNNNN+g<sha>`，而 verl 的闸门要求 ≥0.18（`verl/third_party/vllm/__init__.py` 读的是
  dist 元数据）。不带 override 编出来的 wheel 装上去，RL 会直接 `ValueError: vllm version ... not supported`。
- **vllm 的 CMake 缓存坑**：隔离构建时它的 CMake 会去调一个已经不存在的构建环境 `bin/ninja`
  （uv 每次构建用新临时环境，而 CMake 缓存钉着上一次的路径），所以先删 `build/`、再用
  `--no-build-isolation` 编（venv 里有 cmake/ninja/setuptools-rust），并且 `MAX_JOBS` 压到 8
  （24 个并行 nvcc 把 47G 的 WSL 虚拟机打到重启过）。清单里也把构建期依赖列进了
  `[tool.uv.extra-build-dependencies] vllm`。
- **TransformerEngine**：清单按 Bridge 钉的 rev 走 git 源码（`NVTE_WITH_NCCL_EP=0`，因为 uv 的 git
  checkout 不会 init 它的 `nccl_ep` 子模块）。本机实测从源码编 >90 分钟没编完（单机笔记本），
  venv 里目前是 TE-FL 那份二进制，全部闸门都是在它上面跑通的。
- **`transferqueue` / `tile-kernels`**：`uv sync` 会把它们当"不在 lock 里"卸掉（`verl-core` 的
  extra 没被请求）。跑 RL 前按上面第 2 行补装。

### 昇腾 / NPU 机

清单是 `pyproject.ascend.toml`（与 NVIDIA 侧的差别只有依赖集、index/source、组件安装顺序三处；
两处相同的取向照旧：上游 Megatron-LM 用 main、mcore/Bridge 按可编辑方式装）。

| 层 | 要什么 |
| --- | --- |
| 硬件 | Atlas A2/A3 训练卡（组件文档的实测环境） |
| 驱动/固件 | 与 CANN 版本配套的那一套 + CANN 9.2.0（含 Ascend C、Bisheng 编译器、HCCL、HCOMM 的头与库） |
| 系统 | Linux + Python 3.12（TransformerEngineNPU 要 >=3.12，本仓 `requires-python` 也是 3.12） |
| torch | CPU wheel（`pytorch-cpu` index）+ `torch_npu` **严格配对**（组件文档里一组可用组合是 torch 2.13.0+cpu ↔ torch_npu 2.13.0rc1；MegatronAdaptor 的表写的是 CANN 9.2.0 ↔ torch_npu 26.2.0） |
| 适配层 | `MegatronAdaptor`（让 mcore 在 NPU 上跑）、`TransformerEngineNPU`（TE 的 NPU 后端） |
| mcore 侧补丁 | `MindSpeed`（对 mcore 打补丁）、`MindSpeed-Ops`（Triton-Ascend 融合算子，自带 `triton-ascend==3.2.2` 约束） |
| 可选内核 | `DeepGEMM-Ascend`、`DeepEP-Ascend`（EP / GEMM 走 DeepSeek 昇腾原生内核时） |
| 训练/推理栈 | `megatron-core`(main) · `megatron-bridge`(本仓检出) · `verl` · `verl-hardware-plugin` · `vllm` · `vllm-ascend` · `transformers` |
| 其它 | `emerging-optimizers` · `pytorch-optimizer`（Muon / AdEMAMix / GrokFastAdamW）· `ray[default]` · `omegaconf` · `pyarrow` · `zstandard` · `pybind11`/`ninja`/`cmake` |
| **不要装** | `fast-hadamard-transform`（CUDA 扩展）、`flashinfer-python`（CUDA 专用） |

```bash
# 1) 代码与子模块（mcore / Bridge / verl / vllm / transformers + 昇腾四件套 + 两个 DeepSeek 内核）
git clone <本仓> shensi && cd shensi
git submodule update --init --recursive

# 2) CANN 环境（提供 ASCEND_HOME_PATH；组件的 setup.py 都读它）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 3) venv + 上游依赖（清单换成昇腾侧那份）
cp pyproject.ascend.toml pyproject.toml
uv venv --python 3.12 && uv sync \
    --no-install-package vllm --no-install-package mindspeed-ops   # 这两个要本地编（见第 4 步）

# 4) 昇腾组件：顺序不能反（适配层 → mcore → 打补丁的 MindSpeed 系）
uv pip install -e 3rdparty/ascend/MegatronAdaptor
uv pip install -e 3rdparty/ascend/TransformerEngineNPU --no-build-isolation
uv pip install -e 3rdparty/common/Megatron-LM                 # 上游 main
uv pip install -e 3rdparty/common/Megatron-Bridge --no-deps    # shensi 模型在它里面
uv pip install -e 3rdparty/ascend/MindSpeed
uv pip install -e 3rdparty/ascend/MindSpeed-Ops --no-build-isolation --no-deps \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# 5) 两个可选内核（要 EP/GEMM 原生内核时；DeepGEMM-Ascend 的 install_requires 里还带 tilelang）
uv pip install --no-build-isolation 3rdparty/ascend/DeepGEMM-Ascend
uv pip install --no-build-isolation 3rdparty/ascend/DeepEP-Ascend

# 6) 装配自查：一句话把整条链 import 一遍（缺哪层就报哪层）
python -m shensi.utils.ascend_env

# 7) 本地产物 + 环境变量（与 NVIDIA 侧同一套）
export SHENSI_FS=/home/<user>/fsdata            # 数据/ckpt/模型 的根
python -m shensi.recipes.shensi.common.tiny_artifacts --out $SHENSI_FS   # tiny-tok / tiny-rl

# 8) 逐段跑（命令与 NVIDIA 侧一样）；RL 四段在昇腾上要设：
export VERL_PLATFORM=huawei                      # verl 的 NPU 平台名
export VERL_USE_EXTERNAL_MODULES=shensi.runtime
# 注意：NVIDIA 侧那两条本机专有开关在这里不要用（nvidia_noipc / VLLM_PLUGINS="" 都是 WSL2 与 fork 的绕行）
```

昇腾侧的五处已知差异（DSA 的 Hadamard 旋转替代路径、MindSpeed 与 mcore 的版本配对、numpy 约束、
两个 CUDA 专属包、**未上 NPU 实测**）由 `python -m shensi.utils.ascend_env` 逐项打印。

### 本机补丁

`patches/` 只放"上游二进制包在本机编不过"的补丁，用的时候 clone 上游源码、打补丁、本地编，
不打进 `3rdparty/`：

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform /tmp/fht
git -C /tmp/fht checkout f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
git -C /tmp/fht apply patches/fast-hadamard-transform-sm120.patch
uv pip install --python .venv/bin/python --no-build-isolation --no-deps --no-cache /tmp/fht
```

上游 `setup.py` 的 `-gencode` 列表没有 SM120，RTX 50 系上 DSA indexer 的 Hadamard 旋转会
`no kernel image is available`。（`uv sync` 会按清单里那个 git rev 重装成没打补丁的版本，
SM120 上要再走一遍这三行。）

---

## 环境与已知限制

- **上游依赖只在 `3rdparty/` 克隆**：安装方式、版本与本地补丁见上面的「装环境」与 `pyproject.toml`
  的注释；本仓不往它们的目录里写文件。
- **mcore 与 Megatron-Bridge 按可编辑方式装**（`pyproject.toml` 的 `[tool.uv.sources]`）：
  mcore 的 `core/datasets/Makefile` 不进 wheel，而 `compile_helpers()` 会在起训时 `make -C <包目录>`
  （缺 Makefile 直接 `sys.exit`）；顺带让 3rdparty 里的改动即时生效。
- **`pybind11` / `ninja` / `fast-hadamard-transform`** 是 mcore 的运行时依赖：前两个用于现场编 dataset
  helper，最后一个是 DSA indexer 的 Hadamard 旋转（mcore 的硬依赖），SM120 上要按上面那一节打补丁本地编。
- **`TE_FL_PREFER=vendor`**：本机这套 TransformerEngine-FL / FlagGems 的 flagos 后端在 SM120 上会让前向
  段错误（`te_general_grouped_gemm`），launcher 默认改成 TE 自带 CUDA kernel；`CUDA_HOME` 也按需指向
  `/usr/local/cuda`（flashinfer 在 SM120 上要 JIT 补稀疏 MLA 内核）。
- **`shensi.runtime` 是唯一的对接点**：登记 Bridge 的桥表与 `emerging_optimizers` 的标量优化器表，并给
  verl 的一个缺陷补判断（`use_distributed_optimizer=False` 时 `param_data` 为空的解引用）。verl 侧按它
  自己的约定用 `VERL_USE_EXTERNAL_MODULES=shensi.runtime` 加载。
- **单机口径**：配方按 1~8 卡写，`experiment.runner.nproc_per_node` / `tensor_model_parallel_size` 等按机器
  改；多机需要自己接 launcher（`common/train/launcher.py` 只跑 `nnodes=1`）。
- **RL（stage2_rl）的注意力口径**：mcore 的 CSA 不接受显式 mask，而 verl 会把 response 右 padding
  到 `max_response_length`；Bridge 的 `ShensiModel.forward` 丢掉纯右 padding 的 mask、拒绝左
  padding。`stage1_rlvr` 已跑到 `step:1`，但尾部 pad 仍会通过压缩块参与计算（与上游 fork 同口径）。
  详见 [`stage2_rl/README.md`](src/shensi/recipes/shensi/stage2_rl/README.md) 的「局限」一节。
- **昇腾 / NPU 路径**：清单是 `pyproject.ascend.toml`（拷成 `pyproject.toml` 用），组件按
  MegatronAdaptor → TransformerEngineNPU → mcore → MindSpeed → MindSpeed-Ops 的顺序装，
  装配自查 `python -m shensi.utils.ascend_env`；**未上 NPU 实测**。
- 规模、昇腾路径与"登记未接"的清单见 [`src/shensi/recipes/shensi/README.md`](src/shensi/recipes/shensi/README.md)
  的「局限」一节。

---

## 许可

Apache 2.0，见 [LICENSE](LICENSE)。
