# 与 MiniCPM5-2B 训练配方的核对（2026-10-01）

> 来源（本次核对时检索）：OpenBMB/MiniCPM 仓库与 MiniCPM5 发布说明、HuggingFace `openbmb/MiniCPM5-2B`
> 模型卡、MiniCPM 系列论文（WSD 调度，arXiv:2404.06395）、Meshy 框架仓库（github.com/OpenBMB/Meshy）、
> JustRL II 博客、UltraData 分层数据治理。**所有条目只写检索到的公开信息**；我们这边的对应项都指到
> 本配方里的具体文件。

## 0. 一句话结论

MiniCPM5-2B 的公开训练方案是 **"base(stable+decay) → mid-training → SFT(400B) → RL teachers → OPD"**
五段式，数据全部来自 UltraData 家族（Ultra-FineWeb / Ultra-FineWeb-L3 / UltraX / UltraData-Code /
UltraData-Math / UltraData-SFT-2605 / UltraData-SFT-Agent-2609 / UltraData-RL-2609）。
本配方的 stage 组织与逐段口径对齐如下表；**唯一的刻意差异**是几何：论文的架构对比用 0.6B 阶梯
（`stage1_pretrain/config/default.yaml`），发布形状另有 `minicpm5_2b.yaml` 档（见 §5，几何已反推验证）。

## 1. 逐项对照

| # | MiniCPM5-2B（官方公开口径） | 本配方（对齐落点） | 状态 |
|---|---|---|---|
| 1 | **Base training = stable + decay** 两段，建立基础语言能力与训练稳定性 | `stage0_pretrain/stage1_pretrain`：`default`(stable) → `decay`(profile) | ✅ 对齐 |
| 2 | stable/decay 语料：**Ultra-FineWeb、UltraX**（decay 加 Ultra-FineWeb-L3） | stable=`stage1_pretrain/config/data_prep/default.yaml`（配比 `data_blend_raw.json`：Ultra-FineWeb en/zh + **UltraX** + Code + Math）；decay=`data_blend_decay.json`（**L3**+UltraX+Code+Math） | ✅ 对齐 |
| 3 | **Mid-training** 强化目标能力并适配目标分布 | `stage0_pretrain/stage2_midtrain`：Mid-1 能力强化(4K) → Mid-2 长文档分布适配(16K) | ✅ 对齐（分两段是我们的设计，官方未给段数） |
| 4 | Mid 的代码数据用 **L0–L3 分级的 UltraData-Code** | Mid-1 `config/data_prep/data_blend_raw.json`：UltraData-Code 用 `UltraData-Code-L2` + `UltraData-Code-L3` 两个分级 | ✅ 对齐 |
| 5 | **SFT 400B tokens**：200B deep-thinking + 200B hybrid-thinking（中文 README 口径），数据 **UltraData-SFT-2605** | `stage1_sft`：`default`（SFT-1，200B）→ `hybrid`（SFT-2，200B），配比 `data_blend_raw.json` / `data_blend_hybrid.json` | ✅ 对齐（旗舰档；0.6B 对比档等比缩放，README 注明） |
| 6 | Agent SFT：**UltraData-SFT-Agent-2609**（约 50 万条） | `stage1_sft --profile agent`，配比 `data_blend_agent.json` | ✅ 对齐 |
| 7 | **RL teachers**：数学 / 代码 / Agent / 写作 四个方向，从 SFT 后模型**并行分开训练** | `stage2_rl/stage2_{math,code,agent,writing}` 四臂，起点统一 SFT-3 ckpt | ✅ 对齐 |
| 8 | RL 数据 **UltraData-RL-2609**（>8 万条）；reasoning RL 用 **DAPO-Math-17k** + 两段式长度课程；另用 TriviaQA / NQ-Open / LongWriter-Zero-RLData / 合成的可验证 RLVR / 成对 RLHF | `stage2_rl/README.md` 的数据表与 `data_prep.py` 落点；四臂的 `reward.py` 覆盖 数学/代码/agent/写作 四类奖励 | ✅ 对齐（数据落位后即可跑；预检已通过） |
| 9 | RL 算法 **JustRL II**：GRPO + 加 Critic 做 token 级 credit assignment；三段数据过滤；训练框架 **Meshy**（无中心调度、无 Ray，TransferQueue 驱动，同步/异步/全异步） | 本配方 RL 走 verl GRPO（`stage2_rl/*/config/default.yaml`）；`README.md` 写明 JustRL II 的差异与升级点（critic 挂载位、三段过滤位、Meshy 的对应物是本仓 verl+TransferQueue 栈） | 🟡 结构对齐、算法待升级（见 §4） |
| 10 | **OPD**：把 **16 个专家 RL 模型（其中 5 个 Agent 专家）**蒸馏回同一个发布模型；**reverse KL**（全词表或 top-k logits 并集）当 advantage；蒸馏数据**复用训练各 teacher 用过的 prompts**；方法基于 Thinking Machines 的 OPD 与 "Rethinking On-Policy Distillation" 的改进 | `stage3_opd`：学生=发布基座（SFT-3）、老师=N 个 RL teacher；KD 走 mcore 原生 `--logits-load-dir`；`README.md` 与 `config/data_prep/default.yaml` 已按 reverse-KL / 复用 RL prompts / 可扩到 16 专家（5 agent）写 | ✅ 对齐（专家数从 4 扩到 16 就是加臂；本仓先落 4 臂 + 扩展说明） |
| 11 | 发布 checkpoint 分阶段：Base / Midtrain / SFT / aligned | 我们的产物目录按 stage 分：`ckpt/gated_delta_attn_res/stage1_pretrain/{default,decay}` → `stage2_midtrain/{default,mid2}` → `stage1_sft/{default,hybrid,agent}` → `stage2_rl/*` → `stage3_opd` → 发布（`train/export_hf.py` → HF 目录）→ `stage4_eval` | ✅ 对齐 |
| 12 | 数据治理框架 **UltraData 分层（L0–L4）**；开源 UltraData-Code（~400B L2 + ~150B L3，11 语言）、UltraX（5 子集 ~100B tokens） | 各 stage 的 `config/data_prep/*.json` 直接按数据集/分级组织（`config` 字段即分级），discover/prepare 输出按同口径 | ✅ 对齐 |
| 13 | WSD 学习率：warmup → stable → decay，decay 只占 ~10% tokens | PT-1 `lr_decay_style: constant`（stable）+ PT-2 `cosine→10% 峰值`（decay）；见 `stage0_pretrain/README.md` 的段位表 | ✅ 对齐 |
| 14 | 原生 **131,072** 上下文（非事后外推） | 0.6B 对比档：8K→16K 递增；`minicpm5_2b.yaml` 发布档：`max_position_embeddings: 131072`、Mid-2 `seq_length: 32768`（渐进到长上下文） | ✅ 对齐（档位不同，趋势一致） |
| 15 | 几何：2.52B 总参（1.98B 非嵌入），42 层、GQA 16Q/2KV、hidden 2048、ffn 6144、tied 词表 | `stage1_pretrain/config/minicpm5_2b.yaml`（§5 有反推验证） | ✅ 对齐（新增档） |

## 2. 官方数字与本配方数字的对照表（0.6B 对比档 vs 2.52B 发布档）

| 段 | 官方（2.52B） | 本配方 0.6B 对比档 | 本配方 `minicpm5_2b` 档 |
|---|---|---|---|
| stable | 未公开 token 数 | 9B tokens(90%) / seq 2048 / LR 6e-4 恒定 | 用 `--tokens` 定；LR 6e-4 恒定（2B 档建议先 pilot 校准） |
| decay | decay 只占 ~10% | 1B tokens(10%) / cosine→6e-5 | 同左（比例口径） |
| mid | 能力强化 + 分布适配（代码走 L0–L3） | Mid-1 0.5B(5%)@4K + Mid-2 0.3B(3%)@16K | Mid-2 升到 32K（向 128K 逼近） |
| SFT | 400B（200 deep + 200 hybrid）+ Agent 集 | ~2B 等比缩放（同比例：1B deep + 1B hybrid + agent） | 400B + agent |
| RL | 四方向 teacher + UltraData-RL-2609 | 四臂，同数据口径 | 同左 |
| OPD | 16 专家（5 agent）、reverse KL | 4 臂起步（可扩 16）、reverse KL | 同左 |

## 3. 数据混合（本配方实际写进 config 的比例）

- **PT-1 stable**（`stage1_pretrain/config/data_prep/data_blend_raw.json`）：Ultra-FineWeb(en) 62% +
  UltraX 15% + UltraData-Code 10% + Ultra-FineWeb(zh) 8% + UltraData-Math 5%。
- **PT-2 decay**（`data_blend_decay.json`）：Ultra-FineWeb-L3 45% + UltraX 25% + UltraData-Code 15% + UltraData-Math 15%。
- **Mid-1**（`stage2_midtrain/config/data_prep/data_blend_raw.json`）：UltraData-Code-**L2** 25% + UltraData-Code-**L3** 15%
  + UltraData-Math 30% + UltraX 30%。
- **Mid-2**（`data_blend_mid2.json`）：Ultra-FineWeb-L3 70% + UltraX 30%（长文档）。
- **SFT**：`data_blend_raw.json`（SFT-1 用 UltraData-SFT-2605 的 deep-thinking 子集）/ `data_blend_hybrid.json`（hybrid-thinking 子集）/
  `data_blend_agent.json`（UltraData-SFT-Agent-2609）。
- **RL**：`UltraData-RL-2609` 按方向切（数学/代码/Agent/写作），reasoning 臂带 DAPO 两段长度课程说明。
- **OPD**：学生 rollout（prompt 复用对应 teacher 的 RL prompts）。

## 4. 未完全对齐的三处（如实）

1. **RL 算法**：官方 JustRL II = GRPO + Critic（token 级 credit assignment）+ 三段数据过滤；
   本配方目前是 verl GRPO（双侧截断、无 critic）。升级点已标在 `stage2_rl/README.md`：
   `critic` 段与 `advantage` 段的配置位、过滤三段的数据管线位。
2. **训练框架**：官方 Meshy（去中心化，TransferQueue 驱动）；本仓是 verl + Ray/TransferQueue 的既有栈。
   配方层面（yaml 形状、数据 schema、rewards）已对齐，框架替换属于工程迁移。
3. **规模**：官方旗舰 2.52B / 400B SFT / 16 专家；本配方以 0.6B 比较档为主（论文口径），
   发布形状另有 `minicpm5_2b.yaml`。数字不同、口径一致。

## 5. MiniCPM5-2B 几何的反推验证（供 `minicpm5_2b.yaml` 用）

公开数字：总参 **2,516,756,480**、非嵌入 **1,981,982,720**、42 层、GQA 16 Q / 2 KV、head_dim 128（Llama 系）。

按 tied embedding 反推：每层 = q(2048×2048) + k(2048×256) + v(2048×256) + o(2048×2048)
+ gate/up/down(3×2048×6144) + 2×RMSNorm(2048)
= 4,194,304 + 524,288 + 524,288 + 4,194,304 + 37,748,736 + 4,096 = **47,190,016/层**
×42 + 末层 norm 2,048 = **1,981,982,720** ✅（与非嵌入参数**逐位相等**）
词表：2,516,756,480 − 1,981,982,720 = 534,773,760 = **261,120 × 2048**（tied，无独立 lm_head）

→ `hidden_size: 2048, num_layers: 42, ffn_hidden_size: 6144, num_attention_heads: 16,
num_query_groups: 2, kv_channels: 128`。词表由运行时的 tokenizer 决定（本配方 tokenizer 用
Qwen3 同款 → 151,936；若换 MiniCPM 官方 tokenizer 则 261,120）。

## 6. 复现口径

```bash
# 0.6B 比较档（论文主表口径）
cd stage0_pretrain/stage1_pretrain && python train.py --model-algo qwen3_gdar_paper --tokens 9e9
python train.py --profile decay --tokens 1e9 --load <stable ckpt>
# 发布形状（MiniCPM5-2B 几何）
python train.py --profile minicpm5_2b --tokens <stable 预算>
```
