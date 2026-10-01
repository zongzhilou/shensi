# Stage 1: SFT（指令微调）

从预训练末段 ckpt 起做多域指令微调：数据是 messages jsonl（含编码 / 推理 / 工具调用），
用 mcore 的 `--sft`（SFTDataset + SFTTokenizer 按 chat 模板生成 loss mask），模型与训练循环跟 PT 同一条
（`../train/` 那个入口，模型来自 Megatron-Bridge 的 `models/shensi/`）。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--profile` 选档、`--smoke`（tiny + mock）、`--tokens`、`--set`；另负责把 `sft_train.jsonl` 注入 `data_path` |
| `test_train.py` | 集成测试：tiny 几何 5 步（走 `--sft` 的 THD 打包与 loss mask 路径） |
| `data_prep.py` | post-training 语料 → `{"messages": [...]}` jsonl（按 DeepSeek-V4 chat 模板，`truncation` 可配） |
| `encoding_dsv4.py` | DSV4 的编码/解码与 chat 模板实现（外部来源，保持原样） |
| `config/` | `default.yaml`（全量）+ `debug.yaml`（极小） |

| 项 | 值 | 出处 |
| --- | --- | --- |
| 起点 | `shensi/ckpt/stage3_1m`（1M 上下文基座） | `experiment.load` |
| 序列长度 | 32768 | 指令数据比预训练语料短，先不拉满上下文 |
| 全局批 | `global_batch_size: 32`，`train_iters: 5000` | 小数据微调：步数给够，靠 eval + 看门狗收尾 |
| 优化器 | **AdamW**（`adam_beta1/2 = 0.9/0.95`、wd 0、lr 1e-5 → 1e-6 cosine、warmup 1%） | Muon 的证据都在预训练规模上，微调先不换 |
| loss | 只算 assistant span（`SFTTokenizer` 生成 mask）；aux/ERC/indexer 三个 loss 按 SFT 口径自动关掉 | mcore `--sft` 的行为 |
| tokenizer | `SFTTokenizer` + `sft_tokenizer_prompt_format: deepseek_v4` | HF 的 chat 模板名（Nemotron 系列同款） |

## Quick Start

```bash
python test_train.py                    # 集成测试（tiny 几何 5 步）
python data_prep.py --discover          # post-training 语料面貌
python data_prep.py --prepare           # → $SHENSI_FS/shensi/data/stage1_sft/sft_{train,val}.jsonl
python train.py --smoke                 # 仓库内 tiny 档
python train.py --profile debug         # 真实数据的极小档
python train.py                         # 正式跑（default 档）
```

`data_prep.py` 的输出是**一行一条**的 `{"messages": [{"role": ..., "content": ...}, ...]}`；
mcore 的 `--sft` 直接读 jsonl（不用 bin/idx），loss mask 由 `SFTTokenizer` 按模板切。
`train.py` 会把 `sft_train.jsonl` 写进 `train.data.data_path`，不用手工拼。

## 数据

来源是 post-training 集（`$SHENSI_FS/datasets/llm/post-training/`）：Nemotron post-training v3、
UltraData、`OpenCoder-Instruct`（`OpenCoder-LLM/opc-sft-stage2`，instruction/output/code）等，
配比见 `config/data_prep/data_blend_raw.json`（`--discover` 看实际列名）。多轮样本按 DSV4 chat 模板
拼成单条 messages；超长样本按 `truncation` 策略处理（默认 `error`，即直接报错提醒调 `max_length`）。

THD 打包：`SFTDataset` 恒走打包序列（`--sft` 打开时 `is_packed_sequence=True`），所以 CSA 的
「逐段因果」路径会被走到；本栈的这条路径已按此离线验证过（`dataset` 侧产生的 cu_seqlens 与
mcore 的 THD 口径一致）。

## 验收判据

1. 极小档能续：`experiment.load` 指向 PT 的 debug ckpt 时，`no_load_optim/no_load_rng` 要显式给
   （PT 极小档是 `--no-save-optim` 存的）；
2. 日志里 `lm loss` 下行、`grad norm` 不炸；打包序列下 `num_tokens` 与 batch 的 span 数一致；
3. 指令跟随抽测（固定若干 prompt 生成，人工看格式与工具调用标签是否正确）；
4. **集成测试**：`python test_train.py` 5 步 PASS。

## 下一步

对齐 / RL 见 [Stage 2: RL](../stage2_rl/README.md)。

## 局限

1. SFT 仍走 Adam 系（Muon 只验证在预训练规模）；
2. `encoding_dsv4.py` 与 chat 模板是外部来源，模板改动要同步 HF 侧的同名实现（否则 loss mask 与训练侧不一致）；
3. 数据配比沿用 Nemotron / UltraData 的公开集，没有自建指令数据。
