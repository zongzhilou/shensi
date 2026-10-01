# 阶段 1：SFT（SFT-1 → SFT-2 → SFT-3，400B tokens）

在中训练 ckpt 上做指令微调，数据来自 UltraData SFT 系列：**SFT-1** deep-thinking
（`UltraData-SFT-2605`）、**SFT-2** hybrid-thinking、**SFT-3** agent
（`UltraData-SFT-Agent-2609`）。旗舰档 400B 的构成为 200B deep-thinking + 200B hybrid-thinking，
随后进 agent 段。对照臂用同样的数据、token 预算与 LR 计划，唯一变化的是模型算法。

## 总览

| 组件 | 说明 |
|---|---|
| `data_prep.py` | 把 UltraData SFT 的 parquet/jsonl 规整成 `{"messages": [...]}` jsonl（98/2 切 train/val） |
| `train.py` | Megatron-Core SFT（`--sft`，不打包口径），自动注入 `train.data.data_path` |
| `test_train.py` | 集成测试：合成 messages jsonl + tiny 几何跑 5 步 |

> **早停默认开**：迭代数给到无限大；loss 平台后看门狗（metric `lm loss value`）收尾，按成功处理。

## 档位

| 档 | 段 | 数据 | seq | LR | 预算（0.6B） |
|---|---|---|---|---|---|
| `default` | SFT-1 deep-thinking | UltraData-SFT-2605 | 8192（不打包） | 2e-5 cosine → 2e-6，warmup 1% | ~2B tokens（旗舰 200B） |
| `sft2_hybrid` | SFT-2 hybrid-thinking | 2605 的 hybrid 子集 | 8192 | 同上 | ~2B（旗舰 200B） |
| `sft3_agent` | SFT-3 agent | UltraData-SFT-Agent-2609 | 8192 | 同上 | ~0.2B（旗舰 20B） |
| `debug` / `tiny` | 链路验证 / mock 冒烟 | — | 2048 / tiny | — | 5 步 |

`geoms/*`（规模阶梯）在 PT / Mid / SFT 之间共享：`--profile geoms/qwen3_30b_a3b` 解析到
`stage0_pretrain/stage1_pretrain/config/geoms/` 下唯一的那一份。

## 数据口径：不打包 + chat 模板

- `--sft` 走**不打包**口径（`ShensiSFTDataset`：一条对话一条样本 + 右 padding）。本配方的注意力
  是 local 实现，`DotProductAttention` 断言 `packed_seq_params is None` —— THD 打包需要 TE 注意力
  那条路。这与 shensi 配方的取舍一致。
- loss mask 由 `SFTTokenizer` 按 tokenizer 目录自带的 **Qwen3 chat 模板**生成
  （`sft_tokenizer_prompt_format: default`）。
- `data_prep.py` 会原样保留每条消息的 `reasoning_content`（thinking）。

## 快速开始

```bash
cd stage1_sft
python data_prep.py --prepare --limit 1000            # 调试档 jsonl
python train.py --smoke                               # 合成 messages jsonl + tiny 几何，5 步
python data_prep.py --prepare --blend default.json    # SFT-1
python train.py --tokens 2e9 --load <Mid-2 ckpt>      # SFT-1
python data_prep.py --prepare --blend hybrid.json     # SFT-2
python train.py --profile sft2_hybrid --load <SFT-1 ckpt>
python data_prep.py --prepare --blend agent.json      # SFT-3
python train.py --profile sft3_agent --load <SFT-2 ckpt> \
    --data-jsonl <sft_train_agent.jsonl>
```

| 参数 | 说明 |
|---|---|
| `--profile <name>` | `default`、`sft2_hybrid`、`sft3_agent`、`geoms/*`、`debug`、`tiny` |
| `--model-algo <name>` | 与各 stage 同一份注册表（默认 `qwen3_gdar_paper`） |
| `--tokens <n>` | token 预算 → `train_iters` |
| `--load <ckpt>` / `--data-jsonl <file>` | 接续 ckpt / 显式指定 messages jsonl |
| `--smoke` / `--dry-run` | 合成 tiny 跑 / 只打印命令 |

## 判据

| 检查 | 命令 | 判据 |
|---|---|---|
| 集成测试 | `python test_train.py` | 没有语料时自己生成合成 jsonl；5 步：rc=0、`[after training is done]`、无 Traceback |
| 冒烟 | `python train.py --smoke` | 同判据 |

## 跑完整论文实验（EXPERIMENT_MATRIX.md §4：旗舰对）

SFT 只服务**旗舰对** —— `qwen3_gdar_main` 与它的 plain 残差孪生 `base`，都在 30B-A3B 几何上，
同数据 / 同 tokens / 同 LR。

```bash
cd stage1_sft
python data_prep.py --prepare --blend default.json     # SFT-1：UltraData-SFT-2605 deep-thinking
python data_prep.py --prepare --blend hybrid.json      # SFT-2：hybrid-thinking（200B + 200B = 400B）
python data_prep.py --prepare --blend agent.json       # SFT-3：UltraData-SFT-Agent-2609

for algo in qwen3_gdar_main base; do
  D=$SHENSI_FS/shensi/runs/gdar_30b_sft/$algo
  python train.py --profile geoms/qwen3_30b_a3b --model-algo $algo --tokens 2e11 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train.jsonl \
      --load <Mid-2(30B, $algo) ckpt> --set experiment.exp_dir=$D-sft1
  python train.py --profile sft2_hybrid --model-algo $algo --tokens 2e11 \
      --load $D-sft1/ckpt --set experiment.exp_dir=$D-sft2
  python train.py --profile sft3_agent --model-algo $algo --tokens 2e10 \
      --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train_agent.jsonl \
      --load $D-sft2/ckpt --set experiment.exp_dir=$D-sft3
done

# 评测（仅旗舰对）：mmlu(5-shot) / gsm8k(8-shot) / MATH / HumanEval / MBPP / CMMLU / C-Eval
```

产物按段分目录（`*-sft1 / *-sft2 / *-sft3`），与 `EXPERIMENT_MATRIX.json` 的 `sft1` / `sft2`
行一一对应。RL 阶段消费的是 **SFT-2**（agent 臂用 SFT-3）—— 但要先发布成 HF 目录，见
[RL README](../stage2_rl/README.md)。

## 延伸阅读

- [配方总 README](../README.md) —— 管线总览与 `--model-algo`
- [中训练](../stage0_pretrain/stage2_midtrain/README.md) —— SFT 接续的那一段
- [MINICPM5_ALIGNMENT.md](../MINICPM5_ALIGNMENT.md) —— 400B deep-thinking 的对齐说明
