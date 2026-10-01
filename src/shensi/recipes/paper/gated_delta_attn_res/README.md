# GDAR 配方：Gated Delta Attention Residuals 全链路训练

GDAR = 在 PreNorm transformer 的**深度轴**（残差流）上做「带门控的 delta 规则」——把深度
记忆当作可**编辑**的状态（decay / erase / write 三门 + 闭式解更新 + 白化读）。基座固定
Qwen3，只换深度连接模块，所以同规模内所有模型算法的比较是干净的（同 tokenizer、同数据
顺序、同 seed、同超参，只差 `--model-algo` 这一处）。

组织与 shensi 主配方（`shensi.recipes.shensi`）一致，按训练环节拆 stage，且**组织与训练
配方一一对应**：

```text
预训练（2 段 stable+decay） → 中训练（2 段） → SFT（2 段） → RL（4 方向并行） → OPD（蒸馏回发布基座）

gated_delta_attn_res/
├── README.md                        本文件（口径、段位设计、算法注册表、快速开始）
├── common.py                        stage 定位 / 配置合并 / MODEL_ALGOS 算法注册表 / 启动
├── train/
│   ├── train_gdar.py                训练入口：mcore 训练循环 + GPTModel + --spec + --sft + KD
│   ├── launcher.py                  yaml 的 train.{system,model,data} → mcore CLI → torchrun
│   ├── data.py                      数据 provider（PT 走 GPTDataset，SFT 走上游打包 SFTDataset）
│   └── checks.py                    离线校验：恒等初始化 / 前向恒等 / 参数开销 / 梯度流
├── models/                          模型实现（--spec 的全部预设；mcore 原生件为主，见 §3）
│   ├── gdar_connection.py / gdar_layer.py / gdar_spec.py        GDAR 本体与设计消融
│   ├── depth_connection.py / depth_layer.py / depth_spec.py     AR / DAR / DenseFormer / MUDD
│   ├── hc_layer.py / hc_spec.py                                 HC / mHC
│   ├── ablation_spec.py                                         E2/E3/E6 消融臂
│   └── hf/                          七个变体的 HF 参考实现 + 单测（58/58、64/64、42/42）
├── tokenizer/Qwen3-0.6B/            Qwen3 同款 tokenizer（vendor，全链路统一）
├── stage0_pretrain/                 预训练：stage1_pretrain（stable→decay）+ stage2_midtrain
├── stage1_sft/                      SFT：SFT-1 deep-thinking → SFT-2 agent
├── stage2_rl/                       RL：stage2_{math,code,agent,writing} 四方向并行 teacher
│   └── */config/{default,dapo,drgrpo,token_baseline,critic,fsdp}.yaml   六种算法/路径档
├── stage3_opd/                      OPD：四 teacher 蒸馏回发布基座（mcore 原生 KD）
│   └── 发布：train/export_hf.py     mcore ckpt → HF 目录（评测/rollout/上线都读它）
├── stage4_eval/                     评测（**在 OPD 之后**）：受控深度检索（chance/Wilson/位置偏差）
├── LIMITATIONS.md                   局限清单：逐条"问题→处置→证据"（含升级路径）
└── MINICPM5_ALIGNMENT.md            与 MiniCPM5-2B 训练方案的逐项核对
```

与本文件并列的 **`MINICPM5_ALIGNMENT.md`** 是训练方案核对：逐项对齐 MiniCPM5-2B
（面壁智能/OpenBMB）公开的五段式配方（base stable+decay → mid → SFT 400B → RL teachers →
OPD），含发布几何 `minicpm5_2b.yaml` 的逐位反推验证与如实差异清单。

出处：`gdar_package`（方案总纲 `code/train/RECIPE.md`、HF 参考 `code/models/`、FlagScale 侧
实现 `code/FlagScale/**/megatron/{gdar,depth}/`、verl 接入 `code/verl_plugin/`、论文
`paper/gdar_draft.pdf`）。移植件逐行算法未动，适配本仓 mcore 0.20（见各 layer 顶部注记）。

## 1. 训练配方（0.6B 主对比档；其他规模等比缩放）

### 预训练 + 中训练（stage0_pretrain，2+2 段）

| 段 | profile | tokens | seq | LR | 语料 |
|---|---|---|---|---|---|
| PT-1 stable | `stage1_pretrain` default | 9B（90%） | 2048 | 6e-4 **恒定**（WSD stable），warmup 2% | Ultra-FineWeb(en+zh) 85% + Code 10% + Math 5% |
| PT-2 decay | `--profile decay` | 1B（10%） | 2048 | cosine 6e-4 → 6e-5 | ≥50% 高质量：UltraX 30% + L3 40% + Code/Math 30% |
| Mid-1 能力强化 | `stage2_midtrain` default | 0.5B（5%） | 4096 | 6e-5 恒定，warmup 1% | Code 40% + Math 30% + UltraX 30% |
| Mid-2 分布适配 | `--profile mid2` | 0.3B（3%） | 16384 | cosine 6e-5 → 3e-5 | Ultra-FineWeb-L3 70% + UltraX 30% |

设计依据（为什么 2+2）：stable/decay 两段是"逐级推进"的最小实现——恒定段保证稳定性读数
干净，退火段切高质量子集收尾（Nemotron/GLM 的 WSD 口径）；中训练把"能力强化"（升 4K、
代码/数学拉满）与"分布适配"（L3 长文档、16K、LR 末段再衰减）拆成两步，避免一次动三个变量。
LR 峰值 6e-4 是 0.6B 先验，正式跑前先 pilot {3e-4, 6e-4, 1e-3} 校准。逐段接续用 `--load`。

### SFT（stage1_sft，2 段）

| 段 | profile | 数据 | seq | LR |
|---|---|---|---|---|
| SFT-1 deep-thinking | default | UltraData-SFT-2605 | 8192（不打包） | 2e-5 cosine → 2e-6，warmup 1% |
| SFT-2 agent | `--profile sft2_agent` | UltraData-SFT-Agent-2609 | 8192（不打包） | 同 SFT-1（同量同配） |

旗舰档的 400B-token deep-thinking SFT 用 32K packing（RECIPE.md §4）；0.6B 对比档 ~2B
tokens / 8K 序列，等比缩放。`--sft` 走**不打包**口径（ShensiSFTDataset：一条对话一条样本
+ 右 padding——local 注意力不吃 THD 打包），loss mask 由 SFTTokenizer 按 Qwen3 chat 模板生成。

### RL（stage2_rl，4 方向并行）

从 **SFT-2 ckpt** 出发，四方向**并行分开训**专用 teacher（verl GRPO + Megatron actor，
LR 1e-6、GRPO 双侧截断）：`stage2_math`（答案比对）、`stage2_code`（单元测试执行）、
`stage2_agent`（多轮工具环境、任务成功标志）、`stage2_writing`（rubric/RM）。
verl 侧的 GDAR 注册在 `stage2_rl/gdar_bridge.py`（导入即注册：上游 `register_bridge` 的
七条 model_type → 本配方的连接层规格 provider → 由 `convert/tables.py` **生成**的权重表，
再加 rollout 同步方向的导出），`common.run_verl` 用 `VERL_USE_EXTERNAL_MODULES` 让每个 verl
进程都走一遍。端到端闸门 `stage2_rl/test_gdar_bridge.py`：装载零缺键、HF↔verl 建出的 mcore
对拍 `max|Δ|=2.4e-07`、导出（rollout 同步那条路）名字集合一致且**逐位相等**。

### OPD（stage3_opd）

把四个 RL teacher 蒸馏回**同一个发布模型**（学生 = SFT-2 基座）：学生 rollout → 各方向
teacher 打 token 级 logprob → 学生在**自己的 token** 上做 forward-KL。训练侧是 **mcore
原生 KD**（`--logits-load-dir`，`megatron.training.distillation.LossFuncCallable`），
`train.py --teacher-cache <目录>` 接线；LR 1e-5 cosine、0.5B tokens/轮，可迭代 2~4 轮，
多 teacher 按数据域路由或 logprob 平均。

## 2. 模型算法选择（`--model-algo`，所有 stage 通用）

算法即 spec 预设（注册表在 `common.py::MODEL_ALGOS`，默认 `qwen3_gdar_paper` = 论文主行）：

| 类别 | 名字 |
|---|---|
| GDAR 本体 | `qwen3_gdar_paper` ★ / `qwen3_gdar_upstream`（与上游 shensi 分支逐位对齐：逐头白化）/ `qwen3_gdar`（忠实参考版）/ `_theory` / `_fullrank` / `_block{2,4,8,16}` / `_r16`（参数匹配）/ `_noladder` / `_no_output_route` |
| 对照臂 | `base`（plain Qwen3）/ `qwen3_ar`（+`_block4`）/ `qwen3_dar`（+`_block4`） |
| 连接模块矩阵 | `qwen3_denseformer` / `qwen3_mudd` / `qwen3_hc` / `qwen3_mhc` / `qwen3_gated_ar` |
| 设计消融 | `a1a_gate_prefix` / `a1b_gate_delta` / `a3_decay_projected` / `a4_lambda_free` / `a6_reference` / `a9_half_init` / `a9_uniform_init` / `e3_scalar_gate` / `e3_no_gate` / `e3_decay_only` / `e3_erase_only` / `e3_write_only` |

```bash
python train.py --model-algo base             # plain Qwen3 对照臂
python train.py --model-algo qwen3_ar         # AR 臂
python train.py --model-algo a14_r16          # 设计消融行（参数匹配）
```

全链路 PT→SFT→RL→OPD 用同一个注册表：一个算法名贯穿所有 stage，架构对比才干净。

## 3. mcore 原生性（吞吐从哪来）

- **骨干全部 mcore 原生件**：embedding / RoPE / 注意力 / MLP / norm / MTP / 优化器
  （Muon 等 emerging 优化器）/ 分布式（TP/PP/CP/EP）/ torch_dist 检查点 / 数据管线
  （mcore datasets + bin/idx）都走 Megatron-Core 与其官方扩展点；模型侧通过
  `GPTModel + --spec`（官方 layer-spec 扩展点）接入，不是朴素 torch 手写 transformer。
- **唯一的自研算子是深度连接本身**（`gdar_connection.py`，研究贡献所在）：它包裹在
  mcore `TransformerLayer` 内部，注意力/MLP 等子层仍由 mcore spec 构造（
  `get_gpt_layer_local_submodules`，与 plain 模型逐位同初始化）；dropout 写路径复用层的
  `bias_dropout_add` callable（融合口径一致）。
- `hf/` 下的 HF 参考实现是研究/评测/转换用的朴素 torch 版（论文复现件），**不用于训练**；
  训练一律走 mcore 侧。
- **吞吐档：`config/perf.yaml`**（`--profile perf`）= TE 骨干 + **三个实测可用的融合**
  （bias_swiglu / bias_gelu / gradient_accumulation，smoke rc=0、5 步跑完）。切到 TE 后
  GDAR(0)==plain 达 bf16 舍入量级一致（实测 `max|Δlogit| = 9.8e-3`），**逐位**恒等仍以 local
  路径为准，所以论文恒等验收跑 local、集群吞吐跑 TE。剩下两个融合在本机不可用、保持关闭：
  `masked_softmax` 需要 apex 的 `scaled_masked_softmax_cuda`；`persist_layer_norm` 不被
  torch LayerNorm 支持（两条都是实测报错，见 `LIMITATIONS.md` A5）。
- GDAR 与上游 shensi 分支的对齐：默认 `gdar_layer_spec_paper` 用我们自己的**全局白化**；
  与上游**逐位对齐**（上游当前版是**逐头白化**）请用 `gdar_layer_spec_upstream`
  （`--model-algo qwen3_gdar_upstream`）。两者的实测差见 `models/transformers/test_upstream_alignment.py`
  的 "known deltas"（read_scale=1 时 max|diff| ≈ 2.7）。

## 3.5 设计矩阵的落地（EXPERIMENT_MATRIX.json 一行 = 一个 `--model-algo`）

* **`--model-algo` 覆盖矩阵的每一行**：`qwen3_gdar_main`（主行：theory+正性投影+r64，block 形态 B=4）
  与它的单旋钮族（`_sublayer/_b{2,8,16}/_rank16/_rankfull/_decay_free/_lambda_free/_ladder0/
  _gate_{prefix,delta}/_address_{state,novelty}/_update_reference/_heads1/_null_off/
  _whiten_{diag,off}/_mix_whitened/_no_output_route/_carrier0/_init_{paper,uniform,half}`）、
  门结构族（`_gates_{d,e,w,de,dw,ew,scalar,none}`），以及 `base/qwen3_ar*/qwen3_dar*/qwen3_hc*/`
  `qwen3_mhc{,_lite}*/qwen3_mudd*/qwen3_denseformer*`；每行都与主行只差一处，可直接 diff。
* **规模阶梯齐备**：`stage1_pretrain/config/geoms/{qwen3_1p7b,4b,8b,14b,30b_a3b}.yaml`
  （Qwen3 官方几何 + 各档 peak LR；30B-A3B 为 MoE 门面）。
* **每个 stage 的 README 末尾**都附了"跑完整论文实验"的代码块（PT 全矩阵 / Mid / 旗舰对 SFT /
  四方向 teacher / OPD），命令与矩阵逐行对应。

## 4. 快速开始

```bash
# ① 冒烟与校验（不需要语料）
cd stage0_pretrain/stage1_pretrain && python train.py --smoke
python -m shensi.recipes.paper.gated_delta_attn_res.train.checks   # 恒等/参数开销/梯度流
python test_train.py                                               # 集成测试（真实语料优先）

# ② 预训练 → 中训练（见 stage0_pretrain/README 的完整串接）
python train.py --tokens 9e9 && python train.py --profile decay --tokens 1e9 --load <ckpt>
cd ../stage2_midtrain && python train.py --tokens 5e8 --load <ckpt>   # Mid-1/Mid-2 同法

# ③ SFT
cd ../../stage1_sft && python data_prep.py --prepare && python train.py --tokens 2e9 --load <ckpt>

# ④ RL（四臂并行，起 SFT-2 ckpt）
cd ../stage2_rl/stage2_math && python data_prep.py --prepare && python train.py

# ⑤ OPD
cd ../../stage3_opd && python train.py --tokens 5e8 --load <SFT-2 ckpt> --teacher-cache <dir>
```

每个 run 的最终配置与完整命令落在 `<exp_dir>/config.yaml` 与 `<exp_dir>/run.sh`。

**早停默认开着**：所有训练 stage（PT/Mid/SFT/OPD 与 RL）步数/轮次都给到无限大，靠看门狗及时
收尾——PT/Mid/SFT/OPD 的指标是 `lm loss value`（min），RL 是 `val/reward`（max）；超耐心就
SIGTERM 训练进程组、把原因写进 `<exp_dir>/logs/early_stop.json`，launcher 按**成功**返回。
`--early-stop <patience>` 改耐心、`--no-early-stop` 关掉。

## 5. 已验证与局限

已验证（本仓 venv，2026-10-01，全部有运行日志）：

- **模型**：15 个 spec 预设（含 A11/A16 消融预设）单卡 GPU 前向+反传全过；
- **恒等锚定**：`train/checks.py` 27/27 参数逐位相等、logits `max|diff| = 0.000e+00`、
  增量参数表与梯度流全过；HF 单测 `test_theory` 58/58、`test_ablation_switches` 64/64、
  `test_autoclass` 42/42、`smoke_test` 全过；
- **端到端**：stage1_pretrain 与 stage2_midtrain **真实 bin/idx** 集成测试 PASS；
  stage1_sft **真实 messages jsonl**（`--profile debug`）与**合成 jsonl**（`--smoke`）
  两路都 5 步 PASS；stage3_opd smoke 与预检 PASS；stage2_rl 四臂预检 PASS
  （config→verl CLI / 奖励模块 / verl 导入全 ✓）；
- **算法切换**：12 项 dry-run 断言全过（9 个 stage/段 + 3 项 spec 优先级：
  `--model-algo` > profile 自带 spec > 默认算法，消融档不会被默认算法静默覆盖）；
- **各变体 vs 各自上游仓库**（`python models/transformers/test_upstream_alignment.py`）：
  AR vs Kimi-K3 官方算子**逐位相等**、MUDD vs MUDDFormer 官方 block **逐位相等**、
  DenseFormer vs 官方 DWAModules **逐位相等**、GDAR vs 本仓 shensi 分支在
  `per_head` 配置下**逐位相等**（13/13 检查过；已知 delta 单列）；
- **vLLM 侧**：七个变体登记成功，`smoke_generate --all` **7/7 生成且与纯 transformers
  参考逐 token 一致（lcp=16/16，fp32/tiny）**；
- **吞吐**：`--profile perf`（TE + 三融合）smoke rc=0 且 5 步跑完；
- **早停**：触发式实测（patience=0）——看门狗 SIGTERM 训练组、写报告（`best=10.93065` 为真
  loss）、launcher 返回 0；默认 metric 口径修正（`lm loss value` 而非 `validation loss`）；
- **评测**：`stage4_eval` 受控检索冒烟 40 题 3 秒出分（`chance=0.25`、`usable` 门就位）；
- **RL 算法档**：四个臂 × `default/dapo/drgrpo/token_baseline/critic/fsdp` **24 个 dry-run
  全绿**（每个都断言 `model.path` 与两份 `override_transformer_config` 覆盖真的进了 verl 命令）；
- **verl 通路（B1 结案）**：`stage2_rl/test_gdar_bridge.py` 12/12——auto_map 分发、层规格 =
  GDAR 连接层、**桥上规格与 `gdar_layer_spec_paper` 生效配置逐项相等**、装载/导出零缺键、
  logits 对拍 2.4e-07、导出逐位相等；`convert/test_convert_tiny.py` 往返位级 80/80；
- **发布/评测链**：`train/export_hf.py`（mcore ckpt → HF 目录）+ stage4_eval 评分端到端（tiny：导出 → 40 题 → `score.json`，chance/usable 就位）；导出目录自带 `auto_map` 与两个 `.py`，`trust_remote_code=True` 直接可加载（`model_type: qwen3_gdar`、22 个 `attn_res_*` 旋钮在位）；
- `ruff check` 全过。

局限（如实）：① OPD 的 rollout/打分脚本随 vLLM 接入补齐（KD 训练链路已接 mcore 原生
实现）；② 集群真跑（主表/≈75 消融臂/多 seed）未做；③ 连接算子的融合内核是后续吞吐项
（现状已是可用上限：TE 骨干 + 三融合 + vLLM rollout）。逐条见 `LIMITATIONS.md`。
