# GDAR 配方：Gated Delta Attention Residuals

在 PreNorm transformer 的**深度轴**（残差流）上做「带门控的 delta 规则」——把深度记忆当作
可**编辑**的状态：decay / erase / write 三门 + 闭式解更新 + 白化读。基座固定 Qwen3，
只替换深度连接模块，所以变体间的比较是干净的（同 tokenizer、同数据顺序、同 seed、同超参，
只差 `--spec` 这一处）。

出处：`gdar_package`（2026-09-30 打包，2026-09-30 全流程 tiny 验证）——方案总纲
`code/train/RECIPE.md`、算法设计消融 `code/GDAR_ABLATION_DESIGN.md`、HF 参考实现
`code/models/`、FlagScale 侧实现 `code/FlagScale/flagscale/models/megatron/{gdar,depth}/`、
已跑实验报告 `code/E12_E13_RESULTS.md`、论文 `paper/gdar_draft.pdf`。本配方把其中可训练的
部分（模型 + 训练链路 + 消融矩阵）落进 shensi 仓库的配方约定；`models/` 是移植件（逐行
算法未动，适配本仓 mcore 0.20，见 `models/gdar_layer.py` 顶部的移植注记）。

## 1. 目录结构

```text
gated_delta_attn_res/
├── README.md                        本文件（口径、快速开始、消融矩阵）
├── common.py                        配方共用：路径 / 配置合并 / 启动 / 冒烟
├── train/
│   ├── train_gdar.py                训练入口：上游 mcore 训练循环 + GPTModel + --spec
│   ├── launcher.py                  yaml 的 train.{system,model,data} → mcore CLI → torchrun
│   └── checks.py                    离线校验：恒等初始化 / 前向恒等 / 参数开销 / 梯度流
├── models/                          Megatron-Core 侧（--spec 的全部预设）
│   ├── gdar_connection.py           连接算子（无 mcore 依赖）：AttentionResidual / DepthRead
│   ├── gdar_layer.py                GdarTransformerLayer：把连接接进标准 TransformerLayer
│   ├── gdar_spec.py                 spec 预设：gdar_layer_spec（训练默认）/ _paper（论文主行）/ …
│   ├── depth_connection.py / depth_layer.py / depth_spec.py   AR / DAR / DenseFormer / MUDD
│   ├── hc_layer.py / hc_spec.py     HC / mHC（官方 megatron-core 超连接机制 + 恒等锚定）
│   ├── ablation_spec.py             E2/E3/E6 消融臂（Gated-AR / 门结构 / 块粒度）
│   └── hf/                          七个变体的 HF 参考实现（configuration+modeling 成对、
│                                    自包含）+ test_theory 等单测（58/58）+ 下界守卫
├── pretrain/                        唯一的训练 stage（见 pretrain/README.md）
│   ├── train.py / data_prep.py / test_train.py
│   └── config/{default,gdar,ar,dar,debug,tiny}.yaml + ablations/*.yaml + data_prep/*.json
└── tokenizer/Qwen3-0.6B/            Qwen3 同款 tokenizer（vendor 自包，qwen3 口径）
```

## 2. 四条已定的口径

1. **Tokenizer = Qwen3 同款**（RECIPE.md §1：全实验统一 `Qwen/Qwen3-0.6B`）。本配方
   vendor 了一份在 `tokenizer/Qwen3-0.6B/`，配置默认指它，`$SHENSI_GDAR_TOKENIZER` 可覆盖。
   PT 混合：Ultra-FineWeb ~75% + UltraData-Code ~10% + UltraData-Math ~5% + 多语(中文) ~10%；
   同一规模内所有方法共享数据顺序与 seed（一条流，避免顺序差异污染对比）。
2. **优化器与调度全规模统一**：AdamW β=(0.9, 0.95)、wd 0.1、clip 1.0、BF16；
   0.6B 档峰值 LR 6e-4（先经 pilot sweep {3e-4, 6e-4, 1e-3} 校准）、warmup 2%。
   计划里的 WSD 课程（stable 90% → decay 10% 切高质量子集）在集群主跑执行；本配方配置档
   用 cosine→10% 峰值近似（mcore 无内建 WSD），差异写进对照说明。
3. **恒等锚定**：每个变体在 step 0 逐位等于 plain Qwen3（`torch.equal`），所以所有臂从
   同一点出发、只能离开它；"更差"由构造排除（HF 侧另有下界守卫 `hf/guarantee.py`）。
   验收命令见 §3 的 `train/checks.py`。
4. **参数匹配要报数**：连接模块的增量参数逐变体实测（`checks.py` 第 [4] 项），引用口径分
   "纯主干"与"含 untied 输出层"两个分母（RECIPE.md §10 的表）；低秩 r=64 是训练默认，
   r=16（参数匹配）与全秩都有 spec 预设。

## 3. 快速开始

```bash
# ① 冒烟：tiny 几何 + mock 数据 + 5 步（不碰语料，走 GDAR spec）
cd src/shensi/recipes/paper/gated_delta_attn_res/pretrain
python train.py --smoke

# ② 离线校验（不训练）：恒等 / 参数开销 / 梯度流
python -m shensi.recipes.paper.gated_delta_attn_res.train.checks --profile tiny

# ③ 集成测试：tiny 几何跑 5 步并自动判 PASS/FAIL
python test_train.py

# ④ 真实数据的极小档
python data_prep.py --discover               # 语料面貌
python data_prep.py --prepare --limit 1000   # tiny bin/idx（Qwen3 tokenizer）
python train.py --profile gdar               # 或 --profile debug

# ⑤ 正式档（0.6B × 10B tokens；--tokens 自动换算 train_iters）
python train.py --profile gdar   --tokens 10e9
python train.py --profile default --tokens 10e9   # base 对照臂
python train.py --profile dar    --tokens 10e9    # 对照臂同理
```

每个变体的最终配置与完整命令都落在 `<exp_dir>/config.yaml` 与 `<exp_dir>/run.sh`，
日志实时写 `<exp_dir>/logs/host_0_localhost.output`。

## 4. 规模阶梯与预算（RECIPE.md §3）

| 规模 | tokens | seq | peak LR | batch(tokens) | warmup | 用途 |
|---|---|---|---|---|---|---|
| 0.6B | 10B | 2048 | 6e-4 | 0.5M | 2% | **主对比（核心矩阵）** |
| 1.7B | 20B | 2048 | 4e-4 | 1M | 1% | 趋势确认 |
| 4B | 50B | 4096 | 3e-4 | 2M | 1% | 趋势确认 |
| 8B | 100B | 4096 | 3e-4 | 4M | 0.5% | 规模下限 + RULER |

实测对账（2×4090，0.6B/seq1024/gbs32，末步值）：base 34.2 TFLOP/s/GPU ≈ 5.7 天/10B-arm；
GDAR(r64,block4) 15.6 TFLOP/s/GPU ≈ 14.6 天/10B-arm——**GDAR 比 base 慢 2.19×**（无融合内核），
论文里必须带这个限定（评审 1 R1-6 要的 GDAR-Block 相对 AR 的量级）。本地只跑链路 + B0 的
1B-token 档；主表与消融在集群（同硬件）上跑。

## 5. 消融矩阵（spec 预设 ↔ `pretrain/config/ablations/*.yaml`）

设计消融（RECIPE.md §6.2 的 A 行，每行只改一处、与主行直接可比）：

| 行 | spec 预设 | 配置档 |
|---|---|---|
| A1a/b 门输入 | `gdar_layer_spec_gate_prefix` / `_gate_delta` | `a1a_gate_prefix` / `a1b_gate_delta` |
| A3 decay 正性 | `gdar_layer_spec_decay_projected`（主行已含 project） | `a3_decay_projected` |
| A4 λ 钳制 | `gdar_layer_spec_lambda_free` | `a4_lambda_free` |
| A6/A9 忠实参考 | `gdar_layer_spec_reference` | `a6_reference` |
| A9 门初始化 | `gdar_half_init_layer_spec` / `gdar_uniform_init_layer_spec` | `a9_half_init` / `a9_uniform_init` |
| A11 衰减阶梯 | `gdar_layer_spec_paper_noladder` | `a11_paper_noladder` |
| A13 块粒度 | `gdar_layer_spec_block{2,8,16}` | `a13_block8`（代表档） |
| A14 低秩/全秩 | `gdar_layer_spec_block4_r16`（参数匹配）/ `_fullrank` | `a14_r16` |
| A16 输出路由 | `gdar_layer_spec_no_output_route` | `a16_no_output_route` |
| E3/A8 门结构 | `gated_ar_layer_spec_*`（7 子集 + scalar + none） | `e3_scalar_gate`（代表档） |
| 连接模块矩阵 | `ar_layer_spec` / `dar_layer_spec` / `denseformer_layer_spec` / `mudd_layer_spec` / `hc_layer_spec` / `mhc_layer_spec` | `ar` / `dar` |

每个 spec 预设的语义见对应 `models/*_spec.py` 的 docstring；"跑什么、多少 seed、判什么"
的完整表在 gdar_package 的 `code/train/RECIPE.md` §6.2（含 B0–B3 的分批与三个决策门）。

## 6. 与 shensi 主配方的关系

- 模型来源不同：主配方是 DeepSeek-V4-Flash text_model（CSA/HCA/mHC/MoE，模型在
  Megatron-Bridge 的 `models/shensi/`）；本配方是 **Qwen3 稠密基座 + 深度连接**，
  模型自包含在本配方 `models/`（用户口径：模型直接放配方内）。
- 训练侧共用同一套运行时：launcher 摊平语义、`common.py` 的合并/启动/语料工具、
  `tiny_test` 判据、`early_stop` 看门狗都复用 `shensi.recipes.shensi` 的实现。
- RL / SFT 不在本配方范围（RECIPE.md §4 的 RL 只用于门面模型，不承担架构对比结论；
  verl 接入后置）。

## 7. 已验证与局限

已验证（本仓 venv，2026-10-01）：

- 13 个 spec 预设（gdar_paper / gdar / ar / dar / denseformer / mudd / hc / mhc /
  gated_ar / theory / block4 / 两档门初始化）单卡 GPU 建模 + 前向 + 反传全通过；
- HF 参考实现单测 `hf/test_theory.py` **58/58**；
- 冒烟（tiny + mock + 5 步）与集成测试见 `pretrain/test_train.py`。

局限（如实）：

- 集群真跑（主表 / ≈75 消融臂 / 多 seed）未做，数字是占位；本机只承担链路验证与 B0；
- `--spec` 路径用 local 实现（TE 融合不适用），吞吐按未融合口径估；
- vLLM rollout / verl 接入未随本配方落（包里 `code/rollout/`、`code/verl_plugin/` 有
  完整实现，接 verl 时再搬）；SFT/RL 阶段后置。
