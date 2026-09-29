# stage1_sft：有监督微调（SFT）

在预训练 + 中训练 + 长上下文的底座上做 SFT：让模型学会"按指令 / 多轮 / 工具调用的形状"回答，
数据取 post-training 的 SFT 集。形状上按 GLM-5 / Nemotron-3 的 SFT 口径：**只在 assistant 段算 loss**
（prompt 段 mask 掉），长样本按序列打包（packing）以提高吞吐——本栈的 CSA/HCA 稀疏注意力支持打包序列（THD），
语义与逐样本一致。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 目标 | 指令跟随 / 多轮 / 工具调用三项形状对齐；补长输出能力 |
| 模板 | **DeepSeek-V4 的 chat 编码**（`sft_tokenizer_prompt_format: deepseek_v4`），规格同 ModelScope `DeepSeek-V4-Flash-0731` 的 `encoding/encoding_dsv4.py`（随代码内联在本目录的 `encoding_dsv4.py`） |
| 产物 | 三种口径一次产出：mcore `--sft` 的 messages jsonl、verl 的 messages parquet、已拼好的 packed（含 `loss_mask`） |
| 长输出 | `--min-response-chars` 只留长应答样本（LongWriter 口径，[2408.07055](https://arxiv.org/abs/2408.07055)）；实测样例集 40 → 32 条，平均应答 2704 → 3210 token |
| 优化器 | **AdamW**（本段不切 Muon：Muon 的证据在预训练规模上，小数据微调要单独扫 LR） |

## 2. 数据

`config/data_prep/data_blend_raw.json` 里每条的 `name` 是 `$SHENSI_FS/datasets/llm/post-training/<数据集名>`
（`OpenCoder-Instruct` 也已挪到这个目录下），`config` 是数据集内部配置名。

| 组 | 数据集 | 权重合计 |
| --- | --- | --- |
| 通用/指令（Nemotron-Cascade-2） | `Nemotron-Cascade-2-SFT-Data`：chat / instruction_following / conversational_agent / math / science / safety | 0.33 |
| 代码 / agent | `Nemotron-SFT-SWE-v3`、`Nemotron-SFT-OpenCode-v1`、`Nemotron-SFT-Agentic-v2`、`Nemotron-SFT-CUDA-v1`、`OpenCoder-Instruct` | 0.26 |
| 数理与推理 | `Nemotron-SFT-Math-v4`、`Nemotron-SFT-Science-v2`、`Nemotron-SFT-Competitive-Programming-v2`、`Nemotron-Math-Proofs-v2` | 0.15 |
| 指令遵循 / 多语 / 安全 | `Nemotron-SFT-Instruction-Following-Chat-v3`、`Nemotron-SFT-Multilingual-v2`、`Nemotron-SFT-Safety-v2`、`Nemotron-SFT-ARC-AGI-v1` | 0.13 |
| 领域与中文 | `Nemotron-SpecializedDomains-Finance-v1`、`Fineweb-Edu-Chinese-V2.3`（`messages` 配置） | 0.05 |
| agent（UltraData） | `UltraData-SFT-Agent-2609`（Code/General/Search-Agent）、`UltraData-SFT-2605`（Code/IF/Math/Knowledge） | 0.13 |

权重按相对值理解（mcore 的 `data_path` 权重做归一化，和不必正好等于 1）。

`data_prep.py` 把各种形状（`messages` / `instruction-output` / `alpaca` / `conversations`）一次归一成三种口径：

```text
sft_train.jsonl / sft_val.jsonl    每行 {"messages": [...]}    → FlagScale/mcore 的 --sft（SFTDataset）
train.parquet / test.parquet       messages 口径               → verl 的 SFT 训练器
packed/train.parquet               input_ids + loss_mask       → 已按模板拼好的打包版（loss 只在 assistant 段）
```

```bash
python data_prep.py --discover
python data_prep.py --prepare --blend config/data_prep/data_blend_raw.json
python data_prep.py --prepare --min-response-chars 2000     # 只留长输出样本（LongWriter 口径）
```

## 3. 超参

| 项 | 值 | 说明 |
| --- | --- | --- |
| 序列长度 | 8192（`seq_length`） | 与 GLM-5 的 SFT 段同量级 |
| 全局批 | `global_batch_size: 64`（micro 1 × DP × 梯度累积） | 单机 8 卡够用 |
| 优化器 | AdamW β 0.9/0.95、wd 0.1、grad clip 1.0 | 沿用预训练的 AdamW 口径（预训练默认已换 Muon 混合，见 `../stage0_pretrain/stage1_pretrain/README.md`） |
| 学习率 | 1e-5 → 1e-6，warmup 1%、cosine | SFT 用比预训练更小的 LR |
| 模板 | DeepSeek-V4 的 chat 编码：prompt 形如 `<｜begin▁of▁sentence｜>system<｜User｜>…<｜Assistant｜></think>…<｜end▁of▁sentence｜>`；带 `reasoning_content` 的样本自动走 `<think>…</think>` 形态（`data_prep.py --thinking-mode auto`） | mcore 的 SFTTokenizer 里实现 |
| 续训点 | `experiment.load` → 预训练/长上下文 ckpt | 用 `finetune: true` 只接权重 |

## 4. 运行

```bash
python train.py --dry-run        # 只打印 flagscale 命令
python train.py                  # 正式跑（FlagScale 的 --sft + SFTDataset 打包）
```

## 5. 验收判据

1. 日志里 `lm loss` 在几十步内明显下降，`loss_mask` 生效（看 assistant 段的 loss 数量级）；
2. **打包与逐样本等价**：同一批样本，`micro_batch_size=1` 的打包前向与逐样本前向 logits 一致
   （本栈的 THD 打包路径已按此验证，见 `entrypoints/check_shensi_thd.py`）；
3. 抽测：拿训练里没见过的指令问模型，回答格式与训练形状一致（乱答说明模板/mask 配错了）；
4. checkpoint 能存能续（`torch_dist`），续跑 `lm loss` 接得上。

## 6. 局限

1. 权重是相对值，不同来源的相对规模与实际 token 占比未做对拍（要看 `--discover` 的实测条数）；
2. 打包口径要求样本按序列拼，超长样本会被截断（`seq_length: 8192`），长文档类 SFT 要单独调；
3. 本段不做 Muon（见第 1 节），因此与预训练段的优化器不连续——续训时优化器状态本来也不接续（`finetune: true`）。
