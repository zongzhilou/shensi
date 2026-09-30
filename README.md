# Shensi

**DeepSeek-V4-Flash 的 text_model 架构 + GLM-5.x 的训练配方**，实现在一套可复现的工程栈上：训练用 FlagScale +
Megatron（`Megatron-LM-FL` 内核、`Megatron-Bridge` 权重桥），RL 用 verl，推理用 vLLM；NVIDIA（CUDA）与
昇腾（Ascend NPU）两条算力都支持。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 架构 | DeepSeek-V4-Flash 的 text_model：CSA + HCA 混合注意力（带 Lightning Indexer / DSA）、mHC 多流超连接、3 层参数共享 MTP、1M 上下文（[2606.19348](https://arxiv.org/abs/2606.19348)） |
| 配方 | GLM-5 / 5.1 / 5.2 / 5.3 的训练课程：三阶段预训练（稠密主干 → DSA 两段式 → 长上下文）+ SFT + 三段 RL（[2602.15763](https://arxiv.org/abs/2602.15763)） |
| 优化器 | 默认 **Muon（矩阵）混合 AdEMAMix（非矩阵）**：Muon Split（含 MLA 按头/按组切分）+ DeepSeek-V4 的 γ=0.18 归一，两条腿共用一条 LR 曲线（V4-Flash 峰值 2.7e-4） |
| 数据 | 预训练/SFT/RL 都用 Nemotron 系列公开集按域加权；用法参考 Nemotron-3（[2512.20848](https://arxiv.org/abs/2512.20848)、[2604.12374](https://arxiv.org/abs/2604.12374)、[2606.15007](https://arxiv.org/abs/2606.15007)） |
| 额外一条线 | `stage2_rl/stage4_world_model`：把「动作 → 观测」练成语言世界模型，当 `stage2_agentic` 的 Sim RL 环境用（AgentWorld 口径，[2606.24597](https://arxiv.org/abs/2606.24597)） |
| 交付形态 | 一个 Python 包 `shensi`（含 `_ext/` 下的上游增量 + `recipes/` 下的全部配方），`uv build` 出 wheel |

## 2. 模型家族

| 维度 | 取法 | 出处 |
| --- | --- | --- |
| 注意力 | **CSA + HCA 混合**：`compress_ratios` 里 `4` = Compressed Sparse Attention（带 Lightning Indexer / DSA），`128` = Heavily Compressed Attention，`0` = 纯滑窗 | DeepSeek-V4 text_model："a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)" |
| 残差/超连接 | **mHC**（多流 + 路由 + MLP 侧因果深度卷积 + Gram-Schmidt 正交支路）；深度连接（AttnRes）用 **GDAR**：门是零初始化的偏差（init 精确等于加性更新）、更新式是 gated delta rule 的闭式解，读侧白化 + 多头 + Softmax₁ | 同上（Manifold-Constrained Hyper-Connections / Attention Residuals） |
| 1M 上下文 | compressor 层 YaRN（factor 16 / 原位置 65536）+ 滑窗 | 同上（V4 全系 1M） |
| MoE | hash MoE（前若干层按 token 哈希路由，不训 router）+ latent MoE（低秩专家） | DeepSeek-V4 / V3 系一脉 |
| MTP | 3 层参数共享 | GLM-5：3 层共享，4 步投机解码接受长度 2.55 → 2.76 |
| 优化器 | Muon + AdEMAMix 混合（见第 3 节） | DeepSeek-V4、GLM-5、Kimi K2、Qwen3.8-Flash-Next |

## 3. 训练配方

```text
stage0_pretrain/stage1_pretrain   预训练①：稠密主干（csa_dense_mode=true），4K → 8K，27T 量级
stage0_pretrain/stage2_midtrain   预训练②：32K + DSA 两段式（dsa_warmup 1000 步 → sparse adaptation 20B）
stage0_pretrain/stage3_longctx    预训练③：128K/500B → 1M/50B，长文档 up-sample
stage1_sft                        SFT：FlagScale --sft + DeepSeek-V4 chat 编码，三口径产物
stage2_rl                         RL：verl GRPO + IcePop 截断；四个子 stage 见下
stage3_eval                       评测：dsh 走 NeMo Gym + 不依赖 Gym 的 local 套件
```

1. **三个 loss 全开**：router aux 0.001、ERC 熵正则 1.0 / α=0.5、indexer KL 0.01（DeepSeek-V3.2 §2.1）。
   本配方**不用** loss-free bias 负载均衡、**不用** IndexShare（显式关闭，理由见各 stage README）。
2. **两阶段 DSA**：稠密主干先训（stage1）→ 冻主干只训 Lightning Indexer（KL 目标 = 稠密注意力分布）→
   切稀疏全参训（KL 目标 = top-k 集合）。
3. **模型几何 = `ShensiConfig` 默认**：35 主干层 + 3 MTP 层的 `compress_ratios`，1M 上下文。
4. **优化器默认 Muon 混合**：2D 矩阵走 Muon（Muon Split + MLA 按头/按组切分 + `spectral` 尺度 + Nesterov +
   `polar_express` 系数 + 零冗余分布式），非矩阵（embedding / 输出头 / norm / MoE router / mHC 静态与
   AttnRes 门控）走 AdEMAMix；`muon_extra_scale_factor: 0.18` 把 Muon 更新幅度归一到 AdamW 量级，
   于是两条腿共用一条 LR（预训练三段默认；SFT 与 RL 仍走 Adam 系，理由见 `stage1_pretrain/README.md`）。

`stage2_rl` 的四个子 stage：

```text
stage1_rlvr        ① 多环境可验证奖励（数学/科学/代码/推理）
stage2_agentic     ② 长时程 agentic RL（多轮 + 工具 + 环境；环境可真机，也可世界模型）
stage3_align       ③ 偏好 / 指令 / 安全对齐（可接 GenRM 判分）
stage4_world_model ④ 世界模型：把「动作 → 观测」练成模型，产物就是 ② 的 Sim RL 环境
```

## 4. 用到的库与增量

| 库 | 角色 | 仓库 |
| --- | --- | --- |
| Megatron-LM-FL（mcore） | 训练内核（attention variant / MTP / MoE / 并行 / 分布式 ckpt / 优化器） | `flagos-ai/Megatron-LM-FL` |
| Megatron-Bridge | HF ↔ mcore 权重桥（转换 / provider / 量化桥） | `NVIDIA-NeMo/Megatron-Bridge` |
| FlagScale | PT/SFT 运行器（`flagscale.run`，两级配置） | `flagos-ai/FlagScale` |
| verl（官方） | RL（GRPO/DAPO、async rollout、agent loop；配套 `TransferQueue`） | `verl-project/verl` |
| vLLM（+ vllm-plugin-FL） | 推理与服务（Shensi 实现在 `zongzhilou/vllm@shensi`） | `zongzhilou/vllm@shensi`、`flagos-ai/vllm-plugin-FL` |
| transformers（shensi 分支） | HF 侧参考实现（`ShensiForCausalLM`） | `zongzhilou/transformers@shensi` |
| FlagGems / FlagCX / FlagTree | 算子 / 通信 / 编译器 | `flagos-ai/Flag*` |
| MindSpeed（昇腾） | Ascend 侧 Megatron 优化与适配 | `Ascend/MindSpeed` |
| DeepSeek Harness（`dsh`） | eval 与 agentic RL 的 agent 层 | `deepseek-ai/deepseek-harness` |

`src/megatron_ext/` 与 `src/flagscale_ext/` 是这两个库的 **shensi 增量**（= 给它们各提一个"适配 shensi"的最小
PR 时会包含的内容）：只做新增，不覆盖上游文件；要改上游行为，就从对应库 `import` 再**继承最小实现**。目录按上游
包路径摆放（`megatron_ext/core/transformer/shensi/…` 对应 `megatron/core/transformer/shensi/…`）。

两边进入运行时的方式不同：
- **megatron 侧直接导入**：`from megatron_ext... import ...`（不再借道 `megatron.core.*`，因此也不需要任何
  接管机制）；
- **flagscale 侧**由 `import shensi` 装的 finder 挂到 `flagscale.*` 上（FlagScale 的运行器按 `flagscale.*`
  导入入口），并且**入口文件必须真的在 FlagScale 树里**——用 `shensi apply-ext`（见 §5.1）或配方启动前拷一次。

我们自己的工具与机制（权重摘要、优化器适配等）放 `src/shensi/utils/`。`src/README.md` 有完整说明；
`python3 tmp/others/ext_invariant.py` 会检查"只新增、不覆盖"这条不变量（有同名上游文件就报错）。

## 5. 环境配置

两条算力路径的差异只有三处（清单、Python 版本、torch 来源），其余步骤相同。

清单是**真的能解析、能装**的：在干净目录里 `uv lock`（只解析不安装，几秒钟排掉清单问题）与
`uv sync` 都验证过。几个实测踩出来的点写在各小节和 §5.6：venv 要显式指定 Python 版本（不指定 uv 会挑允许的
最新版，比如 3.13）、FlagOS 的索引标成 `explicit`（否则它排在 PyPI 前会把 setuptools 这类通用包钉在旧版本上）、
`flagscale` / `transformer-engine` 关掉 build isolation 后要显式声明构建期依赖。

| | NVIDIA（默认） | 昇腾 Ascend |
| --- | --- | --- |
| 清单 | `pyproject.toml` | `pyproject.ascend.toml`（用前覆盖成 `pyproject.toml`） |
| Python | 3.12（`requires-python >=3.12`） | **3.11**（FlagOS 昇腾 wheel 是 cp311） |
| torch | PyPI 的 CUDA 轮子（可钉 CUDA 版本） | `download.pytorch.org/whl/cpu` 的 CPU 轮子 + `torch-npu` / `triton-ascend` |
| vLLM | `vllm`（`zongzhilou/vllm@shensi` 的 git 源，含 Shensi 实现）+ `vllm-plugin-fl`（`flagos-ai/vllm-plugin-FL` 的 git 源，仓库声明的名字就是这个） | 再加 `vllm-ascend`（华为镜像） |
| mcore | `flagos-ai/Megatron-LM-FL@main`（git 源） | FlagOS 的 cp311 发行版（钉在 `flagos-ascend` 索引） |

### 5.1 通用前置

```bash
# 工作区就是一份带子模块的 shensi 检出：上游库全在 3rdparty/ 下（只克隆、不改，改动一律进 src/*_ext）
mkdir -p /root/work/shensi && cd /root/work/shensi
git clone --recurse-submodules https://github.com/zongzhilou/shensi.git
# 3rdparty 分层：common/ 两平台共用；nvidia/、ascend/ 放平台专属
#   common: transformers vllm vllm-plugin-FL FlagScale FlagGems Megatron-LM-FL Megatron-Bridge verl verl-hardware-plugin
#   nvidia: DeepEP DeepGEMM        ascend: TransformerEngine-FL DeepEP-Ascend DeepGEMM-Ascend
#   （DeepEP/DeepGEMM 及其 Ascend 版、TileKernels 等是"参考/待用"的源码，不参与安装）

# 装完包后，把 flagscale 侧增量落到 FlagScale 树（megatron 侧不用落盘：直接 import megatron_ext）：
#   shensi apply-ext --root /root/work/shensi/3rdparty/common    # 先 --dry-run 看会写什么

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"                  # uv 装在这里
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897   # 直连环境可跳过

export SHENSI_FS=/root/work/filestorage               # 存储根（默认 /root/work/filestorage）
mkdir -p $SHENSI_FS/datasets/llm/pre-training $SHENSI_FS/datasets/llm/post-training $SHENSI_FS/models $SHENSI_FS/shensi
```

### 5.2 NVIDIA（CUDA）

```bash
cd /root/work/shensi/shensi
uv venv --python 3.12 --seed             # --seed 带 pip：flagscale 必须用 pip 构建（见 §5.6）
source .venv/bin/activate
# 受限网络下先按 §5.6 的「vllm 源码构建的加速钩子」导出一组 *_SRC_DIR（否则 uv sync 会卡在抓取上）
export VLLM_CUTLASS_SRC_DIR=... VLLM_FLASH_ATTN_SRC_DIR=...   # 完整清单见 §5.6
uv sync                                  # 清单里 torch 走 pytorch-cu130 索引、transformer-engine 走 PyPI，
                                         # 其余（mcore/vllm/transformers/verl/flag-gems/...）都是 3rdparty/common 的本地路径源

# flagscale 单独用 pip 装（不写进依赖）：uv 的构建子进程树会让它的 setup.py 崩
# （_get_ppid 扫 /proc 越界），pip 构建同一个包没问题 —— 这也是它自己 README 的装法
pip install 3rdparty/common/FlagScale
# 权重桥同样单独装：它声明 transformers<=5.15，而 shensi 侧是 shensi 分支（5.18），解析器会打架
pip install --no-deps 3rdparty/common/Megatron-Bridge
# 之后再动环境时带上 --inexact，否则 uv sync 会把不在清单里的 flagscale / megatron-bridge 卸掉
# uv sync --inexact
# 实测要点：FlagGems 的默认分支是 master（清单里写的就是 @master，要钉版本换成 tag）；
# transformer-engine 的 torch 扩展要源码构建，构建期依赖与变量在清单的
# [tool.uv.extra-build-dependencies] / [tool.uv.extra-build-variables] 里声明（见 §5.6）

# 换 CUDA 版本：清单里 torch 钉在 pytorch-cu130 索引（与 TE 的 core-cu13 配套），
# 要换就把它那段 index 的 url 与 extras 一起改掉再 uv sync

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
shensi
```

### 5.3 昇腾（Ascend NPU）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh    # CANN 环境（按机器实际路径）
cd /root/work/shensi/shensi
uv venv --python 3.11 --seed                        # 必须 3.11：FlagOS 昇腾的 mcore wheel 是 cp311；--seed 同 §5.2
source .venv/bin/activate
cp pyproject.ascend.toml pyproject.toml               # torch 走 CPU 轮子、加 torch-npu/vllm-ascend
uv sync
pip install 3rdparty/common/FlagScale         # 与 NVIDIA 侧同理：flagscale 单独装
pip install --no-deps 3rdparty/common/Megatron-Bridge   # 权重桥同理（transformers 上界的处理见 §5.2）
# 与 NVIDIA 侧的差异：mcore 钉在 flagos-ascend 索引（cp311 发行版）、多 vllm-ascend/torch-npu/triton-ascend，
# 其余（flagscale/verl/megatron-core 之外的部分、flag-gems、transformers、vllm-fl）与 nvidia 侧同源

python -c "import torch, torch_npu; print(torch.__version__, torch_npu.npu.is_available(), torch_npu.npu.device_count())"
shensi
```

`torch` / `torch_npu` / `vllm-ascend` / `triton-ascend` 四者要按 vllm-ascend 的 release notes 配套；
对不上就锁版本：`uv pip install "vllm-ascend==<版本>" "torch-npu==<版本>"`。昇腾侧的 Megatron 优化走 MindSpeed。

### 5.4 权重与 tokenizer（两平台共用）

```bash
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local_dir $SHENSI_FS/models/DeepSeek-V4-Flash-0731

# 或者 HuggingFace（需要 token；不要写进脚本或仓库）
export HF_TOKEN=<你的 token>
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir $SHENSI_FS/models/DeepSeek-V4-Flash-0731

export SHENSI_TOKENIZER=$SHENSI_FS/models/DeepSeek-V4-Flash-0731
```

### 5.5 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `SHENSI_ROOT` | `/root/work/shensi` | 代码工作区（shensi 检出；上游库在 `3rdparty/common/` 下） |
| `SHENSI_FS` | `/root/work/filestorage` | 存储根（语料 / 产物 / 权重） |
| `SHENSI_TOKENIZER` | `$SHENSI_FS/models/DeepSeek-V4-Flash-0731` | tokenizer 目录（权重同目录） |
| `CUDA_VISIBLE_DEVICES` | 全部 | NVIDIA 选卡（昇腾用 `ASCEND_RT_VISIBLE_DEVICES`） |

```text
$SHENSI_FS/
├── datasets/llm/pre-training/     # Nemotron 预训练集（目录名去掉 nvidia/ 前缀）
├── datasets/llm/post-training/    # Nemotron post-training v3 + UltraData + OpenCoder 等
├── models/DeepSeek-V4-Flash-0731/ # 权重与 tokenizer
└── shensi/{data,ckpt,logs,runs}/  # 语料产物、检查点、日志、运行目录
```

### 5.6 受限网络与本地克隆

- **上游都是本地路径源**：`[tool.uv.sources]` 里 `megatron-core`、`verl`、`verl-hardware-plugin`、
  `vllm-plugin-fl`、`flag-gems`、`transformers`、`vllm` 全指向 `3rdparty/common/...`（子模块把提交钉住；
  `vllm`/`transformers` 来自 `zongzhilou/*@shensi` 分支）；`torch` 钉在 pytorch 的 cu130 索引，
  `transformer-engine` 走 PyPI（见下一条）。想让某个库走开发中的克隆，改那一行的 `path` 即可。
- **`transformer-engine` 不能写成 git 源**：`flagos-ai/TransformerEngine-FL` 带四个递归子模块
  （cudnn-frontend / cutlass / googletest / nccl），受限网络下 `uv sync` 只会反复 `fatal: early EOF`、
  永远装不出来；NVIDIA 侧要的本来就是官方包，清单写 `transformer-engine[core-cu13,pytorch]`。
  但它在 PyPI 上的 `transformer_engine_torch` 只有 sdist（2026-09 实测：没有匹配 torch 2.13 的预编译
  轮子），必须本地编译，所以清单里配了 `no-build-isolation-package` +
  `extra-build-dependencies`（`setuptools>=80,<81`、`wheel`、`cmake`）+ `extra-build-variables`：
  `CPATH` 指向 venv 里 `nvidia/nccl/include`（它的 torch 扩展无条件 `#include <nccl.h>`，而 nvcc 只带
  toolkit 头目录）、`NVTE_PYTORCH_FORCE_BUILD=TRUE` 强制源码构建。venv 不在 `<包目录>/.venv` 时要改
  这个路径。昇腾侧照旧用 FL fork（NPU 后端在那份里）。
- **`transformers` 的上界和 verl 打架**：`verl 0.10.0.dev0` 把 `transformers` 声明在 5.5.3~5.13 之间，
  而 shensi 分支是 5.18，解析器直接判「无解」；清单用
  `[tool.uv] override-dependencies = ["transformers>=5.18"]` 在解析层换掉这条过期上界（不改上游代码，
  也不像 `Megatron-Bridge` 那样靠 `--no-deps` 绕过去）。
- **vllm 源码构建的加速钩子（受限网络必读）**：vllm 的 `cmake/external_projects/*.cmake` 会给每个上游依赖
  做 `git clone` + 递归子模块，实测只有 ~0.1MB/s，几个仓库加起来几小时都拉不完，而且 `vllm-flash-attn`
  还会去拉 **ROCm 专用**的 `third_party/aiter`（CUDA 构建根本用不到）。它给每个依赖都留了本地目录的环境变量
  钩子，指过去就完全跳过 git（CMake 只用那个目录的源码）：

  ```bash
  D=3rdparty/common/vllm/.deps        # 已经抓过的源码都在这里；首次可从任意一份构建缓存里拷
  export VLLM_CUTLASS_SRC_DIR=$D/cutlass-src
  export DEEPGEMM_SRC_DIR=$D/deepgemm-src
  export DEEPSELECT_SRC_DIR=$D/deepselect-src
  export FLASH_KDA_SRC_DIR=$D/flashkda-src
  export FLASH_MLA_SRC_DIR=$D/flashmla-src
  export FMHA_SM100_SRC_DIR=$D/fmha_sm100-src
  export QUTLASS_SRC_DIR=$D/qutlass-src
  export TML_FA4_SRC_DIR=$D/tml_fa4-src
  export VLLM_FLASH_ATTN_SRC_DIR=$D/vllm-flash-attn-src      # 跳过它那两个 ROCm 子模块
  export TRITON_KERNELS_SRC_DIR=$HOME/triton-v3.5.1/python/triton_kernels/triton_kernels
  uv sync
  ```

  三个配套要点：① `.deps` 里的 `*-subbuild`/`*-build` 是 CMake 的抓取缓存，**目录一搬动就会因
  "CMakeCache.txt directory is different" 报错**——只留 `*-src`、删掉这两类目录即可（钩子模式下不需要缓存）；
  ② **每次重跑 `uv sync` 前都要删 `3rdparty/common/vllm/build`**：uv 每轮的构建目录都不同，in-source 的
  `CMakeCache.txt` 里记着上一轮的 ninja/cmake 路径，不删会以 "failed with: .../ninja --version" 收场；
  ③ `python3 tmp/others/sync5.sh` 就是把这套（清缓存 + 导钩子 + 降并行 `MAX_JOBS=12` + uv sync）打好的脚本——
  并行数别开满 24：24 个并行 nvcc 约吃 48GB 内存，实测会被 OOM 杀掉（exit 15）。
  vllm 的 `deepgemm` 依赖还需要系统头 `elfutils/libdwfl.h`（Ubuntu：`sudo apt install libdw-dev`），
  否则编到 `_deep_gemm_C` 会以 `fatal error: elfutils/libdwfl.h: No such file or directory` 失败。
  triton 那份按需稀疏克隆（实测 6 秒）：

  ```bash
  git clone --depth 1 --branch v3.5.1 --filter=blob:none --sparse \
    https://github.com/triton-lang/triton.git ~/triton-v3.5.1
  cd ~/triton-v3.5.1 && git sparse-checkout set python/triton_kernels
  ```

  版本号要跟 cmake 里的 `TRITON_KERNELS_TAG` 对上，升级 vllm 后跟着改。
- **`nemo_gym` / `cosmos-xenna` 不在清单里**：Gym 那套归 stage3_eval —— 它的 `setup_env.sh` 会另建
  `$SHENSI_ROOT/.venv_gym` 装 `nemo-gym` + `deepseek-harness-sdk`，训练用的 `.venv` 不动；
  而且 `nemo-gym` 的 `openai<=2.7.2` / `mcp<2` 与 vllm 的 `openai>=2.25` / `mcp>=2` 直接互斥，
  放进基础依赖会让 `uv lock` 判无解。`cosmos-xenna` 只在 data_prep 配置的注释里出现过
  （语料特别大时改用 Nemotron 的 Ray/xenna 管线），不是运行时依赖。
- **`TransferQueue` 同样不在清单里**：它把 `tensordict` 钉到 `>=0.10.0`，这条链最终要求
  `packaging<26.0`，与 FlagGems 的 `packaging>=26.0` 互斥（`uv lock` 会直接把三者判成不相容）；
  本包源码里没有引用它，上一版环境也没装。要用 verl 的全异步管线时单独
  `uv pip install TransferQueue`，之后 `uv sync` 记得带 `--inexact`。
- **flashinfer 要顶版本**：vllm 的 `requirements/cuda.txt` 钉 `flashinfer-python==0.7.0`，
  但 SM120 上那版的稀疏 MLA 有缺口（`FLASHINFER_MLA_SPARSE_DSV4`，flashinfer#4380），
  `0.7.0.post1` 才修好，所以清单用 `override-dependencies` 顶掉它（`flashinfer-cubin` 不在 PyPI，
  vllm 的 `setup.py` 本来就把它排除在依赖外，不用管）。
- **源码构建的三个组件**不在清单里，按各自文档装：`FlagCX`（通信库，`flagos-ai/FlagCX` 的
  `docs/getting_started.md#build-and-installation`）、`flagtree`（Triton 发行版，`flagos-ai/flagtree`）、
  以及 FlagGems 的自定义算子（`flagos-ai/FlagGems` 的 install 文档）。
  只有 `FlagScale` 与 `Megatron-Bridge` 两个是 `pip install` 单独装的（前者构建被 uv 的进程树搞崩，
  后者声明的 `transformers` 上界与 shensi 分支打架），清单里刻意不含它们。
- **FlagOS 的 PyPI 被屏蔽**：`pyproject*.toml` 里留了 FlagOS 的 `[[tool.uv.index]]` 作兜底；
  只在需要 FlagOS 自建的 wheel 时才用到（昇腾侧的 mcore 就钉在那个索引上）。它上面也有 `flagscale`
  的 wheel（`flagos-pypi-hosted/simple`，上一个环境就是这么装的）——不想用本地 clone 构建的话，把
  清单里的 flagscale 改成 `flagscale = { index = "flagos" }` 即可，但那样跑的就是上游发行版、
  不是工作区 clone，`apply-ext` 的增量与运行器挂在 PYTHONPATH 上的 `flagscale/train` 就对不上了。
- **flagtree**（Triton 发行版）只有要用 FlagGems 自定义算子时才需要，按 `flagos-ai/FlagTree` 的说明装。
- **先在干净环境验清单**：`cd 干净目录 && uv lock`（几秒，只解析）能立刻暴露清单问题；本轮实测就是这样
  发现「FlagGems 默认分支是 master 不是 main」「FlagOS 索引把 setuptools 钉在 70.2」
  「`transformer-engine` 的 git 源在受限网络下卡在递归子模块」「verl 的 transformers 上界判无解」
  这四处并修掉的。
- **FlagOS 的索引是 `explicit`**：只有 `[tool.uv.sources]` 里显式钉到它的包才走它（昇腾侧 mcore 就是这样）；
  不标 explicit 它会参与全局解析，把通用包钉在 FlagOS 的旧版本上。
- **换成本地开发中的克隆**：`[tool.uv.sources]` 里那几行本来就是本地路径源，把 `path` 指到你正在改的 clone 即可
  （默认指 `3rdparty/common/...` 的子模块）。`flagscale` 不在依赖里（见下条），本来就按 clone 装。
- **从源码装 vllm-plugin-fl**要带后端环境变量（`VLLM_VENDOR=cuda` / `ascend`），否则装出来是纯 Python 版本。
- **flagscale 为什么不在依赖里**：uv 构建它时，它的 `setup.py` 会扫 `/proc` 找注解依赖，在 uv 的构建
  子进程树里会读到空内容并 `IndexError`（2026-09 实测，隔离/非隔离都复现）；**同一条命令用 pip 构建没问题**
  （`pip install ./FlagScale` 默认的隔离构建即可，1~2 秒）。所以它按自己的 README 用 pip 单独装，
  时序是「先 `uv sync`，再 `pip install 3rdparty/common/FlagScale`」；之后要再 `uv sync` 就加
  `--inexact`，免得把不在清单里的 flagscale 卸掉。
- **上游 FlagScale 的 `setup.py` 目前在干净环境构建会崩**（`_get_ppid` 读 `/proc/<pid>/stat` 遇到空内容时
  `IndexError`，2026-09 实测；git 源与本地 path 源都会命中，与清单无关）。临时办法是在 clone 里给
  `setup.py` 的 `_get_ppid` 加一层兜底（`FlagScale/setup.py`）：

  ```python
  parts = stat_content.split(")")
  if len(parts) < 2 or not parts[1].split():   # /proc 项可能已被回收，读到空内容
      return None
  return int(parts[1].split()[1])
  ```

  改完用 `flagscale = { path = "path/to/FlagScale" }` 装；上游修好后换回 git 源。改过之后记得
  `uv cache clean flagscale` 再装，否则会复用旧 sdist。
- **FlagOS 只发源码的组件**（FlagCX / FlagAttention / FlagDNN / FlagBLAS / FlagSparse / FlagTensor）：
  按各仓库说明 `pip install .`，本包只用它们的 Python 侧接口。

- **`transformer-engine` 必须从源码树构建**（本机实测：PyPI 的 `transformer-engine-torch` sdist
  **不含 C++ 源码/头**，单独编不出来）。清单里把 `transformer-engine` 指向
  `3rdparty/ascend/TransformerEngine-FL`（NVIDIA 与昇腾共用这一份 TE 源码），并在
  `[tool.uv.extra-build-variables]` 里给两个包配齐：
  `NVTE_WITH_NCCL_EP="0"`（fork 的 `libnccl_ep` 子构建要 `contrib/nccl_ep` 源码，
  上游自带这个开关）、`CUDACXX="/usr/local/cuda/bin/nvcc"`（否则 CMake 报找不到 CUDA 编译器）、
  `NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"`（Ninja 生成器下 RPATH 重链）、
  `CPATH`（nccl 头）、`NCCL_HOME`/`LIBRARY_PATH` 指向 `.venv/nccl-dev`
  （`nvidia-nccl-cu13` 轮子只有 `libnccl.so.2`，缺 `libnccl.so` 会让 `-lnccl` 失败——
  这个目录里放的是两个软链，`ln -s libnccl.so.2 libnccl.so` 即可）、`NVTE_PYTORCH_FORCE_BUILD="TRUE"`；
  另外 `cmake` 要在**项目依赖**里（no-isolation 的构建要用到它的可执行文件）。整个 TE 源码构建约 40 分钟。
- **`tile-kernels` 不在清单里**：它要求比 vllm 钉的 `tilelang==0.1.12` 更新的 API
  （`tilelang.utils.target`），装进来会让 vllm 的 pin 失配，所以刻意不加；
  要用它的内核就单独建环境。
- **两处本地上游补丁**（都在子模块里，未入库；上游对齐后可删）：
  1. `FlagScale/setup.py` 的 `_get_ppid` 兜底（见上一条）；
  2. `Megatron-Bridge` 的 `models/conversion/quant_bridge.py`：FL fork 的 mcore 没有
     native grouped mxfp8，把那三个名字的导入包一层 `try/except ImportError` 并按"不支持"处理
     （`is_grouped_mxfp8tensor` → False）。同时 **bridge 钉在 `a393057`**
     （"迁到 MCore main"那个提交之前，之后需要更多 mcore-main API）。
- **第三方库要 `import shensi` 之后再用**：Megatron-Bridge 会 import
  `megatron.training.models.gpt`、`megatron.core.transformer.mla_qk_norm_config` 这类
  "mcore main 才有的模块"，FL fork 里没有，由 `megatron_ext` 经 overlay 提供（见 `src/README.md`）。
  我们自己的代码一律直接 `from megatron_ext...`。
- **跑上游测试套件的姿势**（cwd、import 模式、平台插件都会影响解析）：
  `cd /tmp && VLLM_PLUGINS= python -c "import shensi, pytest, sys; sys.exit(pytest.main(['<vllm>/tests/models/shensi','-q','--import-mode=importlib']))"`
  —— vllm 侧 54 项：**53 过 / 1 跳**。`VLLM_PLUGINS=`（空串 = 一个插件都不加载）是关键：装了
  vllm-plugin-FL 时它会注册自己的加速器平台，而插件里 MoE 的补丁要等引擎启动
  （`load_general_plugins`）才打；单测进程走不到那一步，`select_unquantized_moe_backend` 就返回
  `(OOT, None)`（没有 kernel），`test_sparse_moe_block_matches_reference` 以 `is_monolithic` 的
  AttributeError 收场——不是 vllm 分支要补的洞，这条单测本来就该在原生 CUDA 平台上跑；
  Bridge 侧 `--noconftest` 跑（上游 conftest 的 fixture 会把 bridge 的 training/data 半边也拉进来，
  那半边是绑定 mcore main 的）：**10 项全过**。
- **增量怎么进到没 `import shensi` 的进程**：`python -m shensi.install_ext`（幂等）把
  `megatron_ext/**`、`flagscale_ext/**` 铺进 venv 的 site-packages（mcore 与 Bridge 本来就往
  `megatron/` 这一个包里写文件），并留一个 `shensi_ext_autoload.pth`；`import shensi` 另外装"导入后钩子"
  （`megatron.bridge` → 注册 ShensiBridge；`megatron.core.utils` / `mcore_fsdp_adapter` →
  `megatron_ext.core.backfill:apply()`，补 FL fork 缺的 mcore-main 符号）。**别**把 fork 树根塞进
  `PYTHONPATH`：树里的 `megatron` 是个普通包，会整体遮蔽 venv 里含 `bridge` 的那份，worker 报
  `No module named 'megatron.bridge'`。
- **vllm 的版本号要在清单里钉**：vllm 用 vcs-versioning 从 git tag 算版本，而我们的 fork 只推了分支
  （没有 tag），它会退化成 `0.1.dev22167+g<sha>`，verl 的版本闸门要求 ≥0.18.0，直接拒。
  清单用 vllm 官方支持的 `VLLM_VERSION_OVERRIDE` 钉成"基线 tag + 距离 + 提交"
  （`0.30.1rc0.dev360+g54c5060a1`，= fork 若带上游 tag 时 vllm 自己会算出的串）；改这一行后
  `uv sync` 会重编 vllm（源码编译，几十分钟量级）。
- **vllm-plugin-FL 的版本对不上这份 vllm**（子模块仍钉在它自己的 `main`）：`main` 是按更早的 vllm
  写的（`_fused_moe_pkg.FusedMoE`），而 vllm 0.28 起那个工厂改名叫 `FusedMoEFactory`，于是
  `register_model()` 一进来就 `AttributeError`；换成为 vllm 0.28 写的 `0.4.0-dev` 分支能过 MoE 那步，
  但在模型构造的 rope 上撞 `TypeError: unsupported operand type(s) for /: 'float' and 'Tensor'`
  （`deepseek_scaling_rope.py` 的 `1.0 / (scaling_factor * pos_freqs)`，只在 vllm worker 里复现，
  脱离 worker 单独建 rope 是好的）。结论：这份 vllm（上游 main 线）还没有配套的插件版本；
  要么把 vllm 退回 0.28 线，要么等插件跟上。
- **rollout 引擎在这台 SM120 机器上还没跑通**（2026-09 实测，三条路都试了，卡点各不相同；前面
  megatron 侧的建模型、加载权重都已经过了，停在做 dummy forward 之前）：
  1. 原生平台：DeepSeek-V4 系（shensi 的 CSA/HCA）第一次 forward 走 `fused_indexer_q_rope_quant`，
     `has_cutedsl()` 为真（venv 里有 `nvidia-cutlass-dsl`，flashinfer/quack 拉进来的）→ 进 cutedsl 实现
     → 需要 `fa4`；装上 PyPI 上唯一的 `fa4==4.0.0b3` 后与 `nvidia-cutlass-dsl==4.7.1` 的 API 对不上
     （`cutlass.cute.core.ThrMma` 没了），而 flashinfer[cu13]/quack 又要求 cutlass-dsl>=4.7，退不回 4.6；
     把 `cutlass` 从前缀路径摘掉则撞 flashinfer 自己 `No module named 'cutlass'`。
  2. vllm-plugin-FL：见上一条。
  3. 这份 vllm 构建时为了绕开 ROCm 子模块把 `VLLM_FLASH_ATTN_SRC_DIR` 指到了没有源码的目录，
     所以既没编出 `vllm.vllm_flash_attn` 的二进制、也没有 `flash_attn`（原生平台稠密层要它）。
  能选的解法：换用带 flash-attn 的 vllm 轮子、或把 vllm 退回插件支持的线、或等上游把
  fa4/cutlass-dsl 与插件对新 vllm 的支持补齐。
- **SM120 上 flashinfer 要 JIT**：环境里必须有 `CUDA_HOME` 和 `ninja`，否则 flashinfer 自报
  "kernels are disabled"，DSV4 稀疏 MLA 的 `(8,128)` specialization 查不到就 `RuntimeError`
  （配方里已经 `setdefault CUDA_HOME=/usr/local/cuda`）。

## 6. 验证与运行

```bash
cd /root/work/shensi/shensi
shensi                                    # 版本 + 接管了 flagscale 的多少个文件
shensi apply-ext --dry-run --root 3rdparty/common   # 看 flagscale 侧增量会写哪些文件
python3 tmp/others/ext_invariant.py       # 不变量：src/*_ext 只允许新增，不许覆盖上游同名文件

cd src/shensi/recipes/shensi/stage0_pretrain/stage1_pretrain
python data_prep.py --discover                                        # 列名 / 规模 / 权重
python data_prep.py --prepare --blend config/data_prep/debug_sample.json   # 极小档语料
python train.py --profile debug --dry-run                             # 只打印 flagscale.run 命令
python train.py --profile debug                                       # 真跑 5 步（需 GPU/NPU）
```

正式跑：

```bash
python data_prep.py --prepare
python train.py --tokens 27e12            # 单机 1~8 卡；多机在 experiment.runner 里加 hostfile
```

开发克隆（`shensi_tmp/`）里另有一组更细的闸门（`entrypoints/check_*.py`：mcore 冒烟、THD 打包、优化器、
indexer warmup、mbridge 对齐等），是这个工作区自用的，不随发布包分发。

## 7. 局限

1. **规模未验收**：所有配方在极小几何上跑通（loss 有限、ckpt 可存可续、闸门全过），
   全规模收敛曲线与 token 效率需要真机预算才能给结论。
2. **昇腾路径未上机**：昇腾清单的命令按清单与厂商文档编写，cp311 wheel 已随仓库预下，但没有 NPU 机器实测。
3. **若干“登记未接”**：V4.1 的 CSA2 跨层 KV 复用 / FP4 KV / Causal Encoder-Decoder、DeepSelect 的 DSA TopK 内核、
   IndexCache 的跨层 indexer 复用等，都只登记不启用（各自 README 里写明原因）。
4. **数据侧缺口**：GLM-5 长上下文段的自建长文档、合成长数据、MRCR 类数据在 Nemotron 集里没有对应物，
   当前用长文档筛选顶着。

每个 stage 的深入细节（数据口径、超参对照、判据、早停）见
[`src/shensi/recipes/shensi/README.md`](src/shensi/recipes/shensi/README.md) 与各 stage 自己的 README。
