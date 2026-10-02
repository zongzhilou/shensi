# DeepRecur 训练配方：DeepStack（arXiv 2406.04334），模型暂以 Qwen3-VL 占位

复现 [DeepStack](https://arxiv.org/abs/2406.04334)（"Deeply Stacking Visual Tokens is
Surprisingly Simple and Effective for LMMs"）的**训练配方**——数据与训练方法与论文一致；
**模型先以 Qwen3-VL 暂代**（论文是 CLIP-large-336 + MLP projector + Vicuna-7B/13B 的
LLaVA-1.5 路线），真实 DeepStack 模型落地后按 `paper:` 配置块换回去。

> 注意撞名：Qwen3-VL 视觉塔自带 `deepstack_merger_list`（Qwen 自己的多层视觉特征注入），
> 与论文的 DeepStack 方法**无关**；占位阶段它随视觉塔一起冻结/解冻，不另作处理。

## 论文配方对照（§4.1 + Table 10）

| 项 | PT（feature alignment） | SFT（DeepStack-L 主线） | SFT-V / HD 变体 |
|---|---|---|---|
| 数据 | LCS-558k（BLIP 重写的 LAION/CC/SBU caption） | LLaVA-mixed-665k | 748K 自组混合（Table 9） |
| 可训部分 | 只训 projector（"only the projection model tuned"） | 解冻 LLM，视觉编码器冻结 | 视觉编码器也训 |
| 学习率 | 1e-3 | 2e-5 | LLM 2e-5 + 视觉 2e-6 |
| global batch | 256 | 128 | 128 |
| 调度 | cosine / warmup 0.03 / 1 epoch / AdamW | 同左 | 同左 |
| 硬件 | 8×H100（Vicuna 档） | 同左 | 同左 |

论文里 vision lr 两处不一致：正文 1e-6（following LLaVA-NeXT），Table 10 写 2e-6；
`sft_v.yaml` 取 Table 10 的 2e-6，要按正文口径 `--set train.vision_lr=1e-6`。

架构侧记录在每档 `config` 的 `paper.stacking`：block length 1（逐层堆叠）、最优 4 个
stacking 层、高分辨率 token 按 2D 空间 4-邻域采样切成与全局 token 等长的组、原图与高分图
consistent resize、有效视觉 token 2880（HD 14400）。**占位模型不使用这些**；换真实模型时
由模型实现消费。

## 占位映射（Qwen3-VL ↔ 论文部件）

| 论文部件 | Qwen3-VL 对应 | PT | SFT | SFT-V |
|---|---|---|---|---|
| projector（MLP） | `model.visual.merger.*`（+ `deepstack_merger_list.*`） | 可训 | 可训 | 可训 |
| vision encoder | `model.visual.*` 其余（patch_embed/blocks/pos_embed） | 冻结 | 冻结 | 可训（2e-6） |
| LLM | `model.language_model.*` + `lm_head.*` | 冻结 | 可训 | 可训 |

## 目录

```text
deeprecur/
├── common/
│   ├── paths.py  config.py        路径 / 配置合并（base: 链 + --set 覆写）
│   ├── model.py                   Qwen3-VL 占位装载、tiny 随机几何、冻结与分组 LR
│   ├── data.py                    messages jsonl ↔ processor；assistant-only loss mask
│   ├── prep.py                    blend 配比 → 采集（HF hub / 本地）→ messages jsonl
│   ├── train.py                   HF Trainer 组装、早停、dry-run 计划
│   └── models/                    三镜像：引用上游 Qwen3-VL，变体只动 config
│       ├── variants.py            变体注册表：native / unified（encoder-free，+ Gemma 4 预算档）
│       ├── transformers/          unified（encoder-free 模型 + 处理器）与 GDAR/DeepRecur + 闸门
│       │   ├── configuration.py / modeling_qwen3_vl_unified.py / smoke_test.py
│       │   └── configuration_qwen3_vl_gdar.py / modeling_qwen3_vl_gdar.py / smoke_test_gdar.py
│       ├── vllm/                  引擎原生 qwen3_vl 引用 + tiny ckpt + 真权重对拍
│       └── megatron/              mbridge Qwen3VLModelProvider 引用（HF config 驱动）
├── stage0_pt/                     预训练/对齐：LCS-558k，只训 projector
└── stage1_sft/                    指令微调：LLaVA-mixed-665k（+ HD 的 748K 变体）
```

中间数据格式是引擎无关的 **messages jsonl**（`{"images": [...], "messages": [...与 image 部件...]}}），
换真实模型时数据管线原样可用。

## 三个模型镜像（qwen3_vl_unified = encoder-free）

unified 对齐 **Gemma 4 的 encoder-free 口径（arXiv 2607.02770）**：**删掉整个 ViT**——
合并后的 48×48×3 原始 patch（图像处理器 patch 16 × 3×3 merge 产出）经单个大 matmul
（`LN₁→Dense→LN₂→+因子化 2D 位置→LN₃→无缩放 RMSNorm→Linear`）直接投影进 Qwen3-VL 的
文本塔；视觉 token 预算沿用 Gemma 4 的离散档 `{70,140,280,560,1120}`（像素上限 = 预算 × m²，
m = 48）。视觉侧 knobs 直接复用 `Gemma4UnifiedVisionConfig`，语义与实现都以 transformers 的
`Gemma4UnifiedVisionEmbedder` 为准。

| 镜像 | 引用方式 | 变体落点 | 闸门 |
|---|---|---|---|
| `models/transformers` | 上游 `Qwen3VLTextModel`（文本塔）+ 自研 encoder-free 嵌入器；视觉侧复用 `Gemma4UnifiedVisionConfig` 与 Gemma 图像处理器 | `Qwen3VLUnifiedConfig`（`model_type=qwen3_vl_unified` + auto_map）；处理器 = Qwen tokenizer（VL 模板）+ Gemma 图像处理器 | `…transformers.smoke_test`：无 ViT/注意力/deepstack、占位契约（soft token == 占位 span，错配显式报错）、预算、前馈/反传、纯文本 |
| `models/vllm` | **没有** vLLM 原生架构（encoder-free 是自研件）；要上引擎需照 `vllm/model_executor/models/gemma4_unified.py` 做插件 | tiny ckpt 存盘 + 回读往返 | `…vllm.smoke_generate`：存盘→回读→前向；并显式声明「插件未做，不声称 vLLM 可用」 |
| `models/megatron` | 待做（mbridge 无 encoder-free 的 Qwen3-VL provider；native/GDAR 的 mcore 路径另有其表） | — | — |

预算与缩放：`max_soft_tokens=1120`，像素目标 T = 预算 × 48²（图像处理器按 Gemma 4
Algorithm 1 缩放：以预算为目标、保纵横比、按 48 取整），配 `model.variant: unified` +
`model.visual_token_budget: 1120`（两 stage 的 default 档已启用；`native` 走 Qwen 原生对照）。

**口径说明**：预算在 Gemma 4 里是**目标**而非上限——小图会被放大填到预算附近（如 192² →
33×33=1089 token），大图被缩到预算内；每次前向的 soft token 数由该图的最终 patch 数决定，
必须与文本里的占位 span 一致（由处理器负责展开，缺/多都会在模型入口显式报错）。

## 三臂对比口径（stage0_pt：继承 / 冻结 / 重随机）

论文的对比是**三臂同管线**（同一套 stage 脚本，只换 `model.variant`）：

| 臂 | `model.variant` | 视觉路径 | PT 继承 | PT 训练（其余冻结） |
|---|---|---|---|---|
| DeepStack 臂 | `native` | Qwen3-VL 原生：ViT + 多点（deepstack）注入 | 两塔权重；**projector（`visual.merger`）与 deepstack 权重不继承、按初始化口径重随机** | merger + deepstack mergers（1.3M 档） |
| encoder-free 臂 | `unified` | 单 matmul（Gemma 4 口径，无 ViT） | 文本塔；ViT 整段丢弃 | `vision_embedder`（自然的"新件"） |
| DeepRecur 臂 | `deeprecur`（或 `gdar`） | GDAR 两塔 + 每块 reinject/feedback | 两塔权重（deepstack 件剔除）；对齐件同样重随机 | recur 件（AR/feedback）+ merger |

公平性口径：与 Qwen 团队**有数据差距**，所以凡是"视觉↔语言的对齐件"都**不吃** Qwen 用自家
数据训出的权重——native 臂的 projector/deepstack 权重与 deeprecur 臂的 merger 都在 PT 前
按初始化口径重随机（`refresh_alignment_weights`，闸门保证只碰这些件、其余逐位不动）；
unified 臂的对齐件（单 matmul 嵌入器）本来就是新建。冻结的是两塔主干（vision encoder + LLM），
只训各臂的对齐/连接件——与 DeepStack 论文 PT 段"only the projection model tuned"同口径。

```bash
# 三臂同一套命令（tiny 档，完全离线）
python train.py --smoke --set model.variant=native
python train.py --smoke --set model.variant=unified      # 默认档
python train.py --smoke --set model.variant=deeprecur    # tiny 档 recur_blocks=2
```



## 可训规模（单卡昇腾 910B，只登记接口、不下载权重）

`common/model_zoo.py` 登记三档（几何取自各型号官方 `config.json`，2026-10 核对；
deepstack 注入点各型号不同，native 臂直用、deeprecur 臂置空）：

| key | 型号（HF / ModelScope 同名） | 参数 | 几何 | `recur_blocks` | 档位 |
|---|---|---|---|---|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | 2.1B | text 28×2048 · vision 24×1024 · deepstack [5,11,17] | 7（text 4/块 · vision 3/块） | **主档**：三臂 PT+SFT+多 seed |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | 4.4B | text 36×2560 · vision 24×1024 · deepstack [5,11,17] | 6（text 6 · vision 4；12 亦可） | 第二档：PT 必做；SFT 视显存 |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | 8.8B | text 36×4096 · vision 27×1152 · deepstack [8,16,24] | 9（text 4 · vision 3） | 趋势点：只 PT（冻结主干） |

30B-A3B / 32B / 235B-A22B 单卡放不下，排除。几何档放 `common/config/geoms/qwen3_vl_{2b,4b,8b}.yaml`
（跨 stage 共享；`--profile geoms/qwen3_vl_2b` 直接可选），三臂运行时用 `--set model.variant=…` 切：

```bash
python train.py --profile geoms/qwen3_vl_2b --set model.variant=native   --dry-run
python train.py --profile geoms/qwen3_vl_2b --set model.variant=deeprecur --dry-run
```

`--dry-run` 只合并配置、打印计划——**不装载模型、不触任何权重下载**（数据未准备也容忍）。
910B 训练口径：bf16 + 梯度检查点；PT 冻结主干（冻结件不建优化器状态，只训对齐/连接件）；
注意力走 SDPA（不依赖 flash-attn / flashinfer 等 CUDA 专属件）；4B 的 SFT 需 64GB 档卡，
8B 只做 PT。权重获取不代下：`placeholder.name` 填 HF/ModelScope 同名 id，或先下到本地目录
再填路径（`SHENSI_DEEPRECUR_MODEL` 可覆盖）。

## GDAR 两塔 + DeepRecur 顶层（qwen3_vl 上，冻结设计）

在 Qwen3-VL 上实现 DeepRecur 的三层：
**① Text GDAR ② Vision GDAR**（单流，无 HC）→ **③ 顶层 Block 交织**。

| 件 | 落点 | 说明 |
|---|---|---|
| 配置 | `common/models/transformers/configuration_qwen3_vl_gdar.py` | 上游 `Qwen3VLConfig` + 一组 `attn_res_*` 旋钮（与 gated_delta_attn_res 配方同名同默认，单点持有）+ `recur_blocks`（两塔共享的递归块数，null=关闭回上游） |
| 两塔 + 顶层 | `…/modeling_qwen3_vl_gdar.py` | `Qwen3VLGdar{Text,Vision}Model`（层换 GDAR）、`Qwen3VLGdarModel`（GDAR 塔 + 上游接线）、`Qwen3VLGdarDeepRecur{Model,ForConditionalGeneration}`（交织顶层） |
| 闸门 | `…/transformers/smoke_test_gdar.py` | 见下表 |

**复用的算子**：`AttentionResidual` / `DepthRead` / `record_router_stats` 直接从
`gated_delta_attn_res` 配方 import（流无关，两塔共用，不复制代码）。层内两段式：
`routed = read(state, bank) → sublayer(norm(routed)) → state, gates = update(state, out)`；
块边界（`floor(i×depth/n)`，两塔各自切、块数相同）把该层输入追加为银行一行。

**DeepRecur 冻结设计**（"深度轴上的跨模态递归"，块数 = 递归次数）：

```text
每块 i：vision chunk i
        →（末块：output_attn_res 定型 = vision_final）
        → merger 投影 → reinject：把最新视觉硬覆盖进语言的图像位（i=0 即取代占位填充）
        → language chunk i
        → feedback：语言回注给下一块的视觉（末块没有回边）
出口：语言 output_attn_res 深度读 + norm
```

- `reinject`（视觉→语言，硬覆盖）：函数式 `masked_scatter`，每块都把"当前深度的视觉"
  写进语言图像位；语言经自身因果注意力读它。
- `feedback`（语言→视觉，软门控）：逐 clip 让 patch 读自己 clip 的语言 token 段
  （`num_heads` 可配），乘 `tanh(gate)` 写回视觉流；门初值 0 = 回注关闭起步。
- 已删除（相对 ShensiVl 旧实现）：LoopBlock、阻尼迭代、收敛/停滞/轨道判据、
  `reasoning` 子系统、evidence 软检索（与 reinject 后的自注意力重复）。
- 边界：与 deepstack 多点注入未接线（`deepstack_visual_indexes` 必须为空，否则显式报错）；
  视频输入待接线；`recur_blocks` 必须非空。

| 闸门 | 结果 |
|---|---|
| `check_boundaries` | `floor(i×depth/n)`：4/2=[0,2]、4/4=[0,1,2,3]、27/9=步长 3、9/4 末块吃余数 |
| `check_off_parity` | 关闭态与上游 **state_dict 严格装载 + 文本/多模态 logits 逐位相同（max\|Δ\|=0）** |
| `check_gdar_forward_backward` | 开启态 AR 模块 16 个全部有梯度；前向有限 |
| `check_block_rows` | vision 银行行数 = `recur_blocks`（块边界写行生效） |
| `check_deeprecur` | 交织顶层前馈/反传有限、feedback 门有梯度、reinject 与 feedback 开关都真实改变前向 |

## 前置条件

| 项 | 要求 |
|---|---|
| Python | 仓库 venv（transformers ≥ 支持 Qwen3-VL，本机 5.18.0.dev0 / torch 2.13 cu130） |
| GPU | 冒烟与 debug 档单卡可跑；论文口径的 8B 全量 SFT 按 8×H100 档起 |
| 文件系统 | `SHENSI_FS` 指到数据/产物根（本机 `/home/louzo/fsdata`） |
| 占位权重 | 默认 `Qwen/Qwen3-VL-8B-Instruct`，`SHENSI_DEEPRECUR_MODEL` 可指本地目录；tiny/debug 档不下载权重 |

产物位置（`common/paths.py` 强制赋值）：

```text
${SHENSI_FS}/shensi/
├── data/deeprecur/<stage>/         messages jsonl（train/val）
├── runs/deeprecur/<stage>/<profile>/{config.yaml,run.sh,ckpt...}
└── ckpt/deeprecur/<stage>/<profile>/final/   HF 格式（含 processor）
```

## 快速开始

```bash
cd src/shensi/recipes/paper/deeprecur

# ① 冒烟：tiny 随机 Qwen3-VL + 合成数据 + 5 步（离线，不碰语料不下载权重）
python -m shensi.recipes.paper.deeprecur.stage0_pt.train --smoke
python -m shensi.recipes.paper.deeprecur.stage1_sft.train --smoke

# ② PT：LCS-558k
cd stage0_pt
python data_prep.py --discover                       # 看数据落位
python data_prep.py --prepare                        # 拉取 + 转换（558k 全量）
python train.py --tokens 0.14e9                      # 论文口径（1 epoch 的工程换算）

# ③ SFT：LLaVA-mixed-665k，接 PT 产物
cd ../stage1_sft
python data_prep.py --prepare
python train.py --tokens 0.9e9 --load <stage0_pt final 目录>

# 变体：DeepStack-V/HD（视觉编码器 2e-6 + 748K 混合）
python data_prep.py --blend data_blend_hd.json --prepare
python train.py --config sft_v --tokens 1.1e9 --load <SFT final 目录>
```

## 命令行（两 stage 同一套）

| 开关 | 说明 |
|---|---|
| `--profile <名字>` / `--config <路径>` | 选 `config/<名字>.yaml`（两者等价） |
| `--smoke` | tiny 随机模型 + 合成数据，全程离线 |
| `--tokens 0.9e9` | token 预算 → `max_steps≈tokens/(global_batch×assumed_seq_length)`；不给则按 1 epoch |
| `--load <目录>` | 接续的 HF 检查点（PT→SFT→变体） |
| `--set 键=值` | 点号键覆写，可多次，最后应用 |
| `--dry-run` | 打印训练计划（不装模型不拉权重） |
| `--early-stop N` / `--no-early-stop` | eval loss 早停耐心 / 关闭 |

数据准备：`data_prep.py --discover`（报落位）/ `--prepare`（`--config tiny` 小切片、
`--blend data_blend_hd.json` 换 HD 混合、`--offline` 只用本地已有）。

## 数据口径与采集

原始数据落 `$SHENSI_FS/datasets/llm/{pre,post}-training/<name>/`；配比 json 在各 stage 的
`config/data_prep/data_blend_*.json`，条目带 `count`（论文口径条数）与 `hf`（自动拉取的
dataset id）。要点：

- **LCS-558k**：`liuhaotian/LLaVA-Pretrain`（parquet 内嵌图像，自动落图）。
- **LLaVA-mixed-665k**：论文未给条目级拆解；按 LLaVA-1.5 官方口径配
  （LLaVA-Instruct-158K + VQAv2 83K + GQA 72K + OK-VQA 9K + OCR-VQA 80K + A-OKVQA 66K +
  RefCOCO 48K + VG 86K ≈ 602K，其余 ~63K 官方未逐条公开）。LLaVA-Instruct 的图像
  （COCO + VG）要手动放 `images/llava-instruct`。
- **HD 748K**：Table 9 原样 16 个条目（含 3 个 task prompt）；`hf: null` 的条目
  （LAION-GPT4V、DVQA、VG）没有可靠公开镜像，手动放置。
- 采集不到的数据集**显式报错**，不静默跳过；HF 镜像字段对不上时转换也会报错而不是出 0 条。

## 已验证

| 项 | 命令 | 结果 |
|---|---|---|
| PT 冒烟 | `stage0_pt.train --smoke` | 5 步，loss 12.1 → 11.31（≈ln vocab=11.9 起步，梯度非零）；只训 merger 1.3M 参数，vision 3.1M / llm 80.7M 冻结 |
| SFT 冒烟 | `stage1_sft.train --smoke` | 5 步，loss 11.86 → 10.45；llm+projector 可训、视觉塔冻结 |
| assistant-only mask | collator 单测 | 图像展开后区间不错位；assistant 区间外（prompt/图像/padding）全部 -100 |
| unified·transformers（encoder-free） | `…models.transformers.smoke_test` | 5/5：无 ViT/注意力/deepstack（视觉侧仅 472,896 参数的单 matmul 嵌入器）、占位契约（1089==1089，错配显式报错）、预算（1120 → 2,580,480 px）、前馈/反传（嵌入器全参数有梯度）、纯文本路径 |
| unified·vllm | `…models.vllm.smoke_generate`（离线） | tiny ckpt 存盘→回读→前向通过（logits (1,1101,151669)，占位 1089 == soft token）；并显式声明 vLLM 需插件（未做） |
| unified 全链路 | `stage0_pt.train --smoke` / `stage1_sft.train --smoke` | encoder-free 模型 + 处理器接进训练栈：PT 只训嵌入器（projector 481K 可训 / llm 41.8M 冻结），两条冒烟 loss 11.7 / 10.99，检查点落盘 |
| 数据落位 | `data_prep.py --discover`（两 stage） | 逐条报论文条数 / hf id / 本地落位 |
| 离线边界 | `data_prep.py --prepare --config tiny --offline` | 数据缺失时显式报错并给出落位路径 |
| 代码卫生 | `ruff check src/shensi/recipes/paper/deeprecur` | All checks passed |

## 边界（显式声明，不静默）

- **论文的 stacking 架构没有实现**：占位模型是原生 Qwen3-VL（动态分辨率），视觉 token 全部走
  输入层。`paper.stacking` 只是记录，等真实模型落地后由模型实现消费。
- **分辨率口径不同**：论文 CLIP-336 全局 576 token + 堆叠到 2880/14400；占位模型按
  Qwen3-VL 原生动态分辨率（`model.max_length` 截序列，数据量大时建议配 `--set` 限
  `max_pixels` 类参数，见占位 processor 的 min/max pixels）。
- **665k 的条目级拆解是 LLaVA-1.5 官方口径的推断**（论文只给了名字与总量），差异已写进
  blend json 的 note。
- **全量 SFT 的显存**：8B 全参 + AdamW 按 8×H100 档规划；本机 16GB 只够 tiny/debug 档与
  PT 档（projector-only）真跑。
- 评测未包含：论文的 25+ 基准评测不在本配方范围（复用 `recipes/shensi/stage3_eval` 时另接）。
