# stage1_sft：SFT（deep-thinking → agent，两段）

先 SFT-1 用 **UltraData-SFT-2605**（deep-thinking）建立深度思考与通用对话能力，再 SFT-2
用 **UltraData-SFT-Agent-2609** 补 agent 多轮能力；两段**同量同配**（架构对比口径）。
起点 = 中训练末段 ckpt（stage2_midtrain/mid2）。


> **早停默认开**：本 stage 步数/轮次给到无限大，收敛与收尾交给看门狗（metric=`lm loss value`，
> patience 见 `config/default.yaml` 的 `early_stop` 段；超耐心 SIGTERM 收尾、按成功返回；
> `--no-early-stop` 可关；改耐心用 `--early-stop N`）。

## 段位设计（0.6B 对比档）

| 段 | profile | 数据 | seq | LR | 预算 |
|---|---|---|---|---|---|
| SFT-1 | default | UltraData-SFT-2605 | 8192（不打包） | 2e-5 cosine → 2e-6，warmup 1% | ~2B tokens（旗舰 8B 档为 400B，等比放大） |
| SFT-2 | sft2_agent | UltraData-SFT-Agent-2609 | 8192（不打包） | 同 SFT-1 | 同 SFT-1 |

## 数据口径

- mcore `--sft` 走**不打包口径**（复用 shensi 的 `ShensiSFTDataset`：一条对话一条样本 +
  右 padding）：本配方的注意力是 local 实现，`DotProductAttention` 断言
  `packed_seq_params is None`（THD 打包得换 TE 注意力），与 shensi 的取舍一致；
  loss mask 由 SFTTokenizer 按 **Qwen3 chat 模板**（`sft_tokenizer_prompt_format: default`
  = tokenizer 目录自带的模板）生成。
- data_prep.py 把 UltraData-SFT 集（parquet/jsonl）规整成 messages jsonl
  （`{"messages": [...]}` 每行一条，thinking 的 reasoning_content 原样保留），按 98/2 切
  sft_train.jsonl / sft_val.jsonl；`train.py` 自动注入 `train.data.data_path`。

## 跑法

```bash
cd stage1_sft
python data_prep.py --prepare --limit 1000                       # 调试档
python train.py --smoke                                          # 合成 messages jsonl + tiny 几何（5 步）
python train.py --tokens 2e9 --load <Mid-2 ckpt>                 # SFT-1
python data_prep.py --prepare --blend agent.json                 # SFT-2 数据
python train.py --profile sft2_agent --load <SFT-1 ckpt> \
    --data-jsonl $SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_sft/sft_train_agent.jsonl
```

说明：SFT-2 的 jsonl 文件名带 `agent` 后缀以区分两段产物（data_prep 以 --blend 决定来源，
文件名见其输出打印）。

---

## 跑完整论文实验（EXPERIMENT_MATRIX.md §4：旗舰对）

SFT 只为**旗舰对**服务：`GDAR(main)` 与它的 `base`（standard-residual 孪生），在 30B-A3B 上，
两臂**同量同配**（同数据、同 token 数、同 LR 计划）。

```bash
cd stage1_sft
python data_prep.py --prepare --blend default.json        # SFT-1：UltraData-SFT-2605 deep-thinking
python data_prep.py --prepare --blend hybrid.json         # SFT-1b：hybrid-thinking（200B+200B=400B 口径）
python data_prep.py --prepare --blend agent.json          # SFT-2：UltraData-SFT-Agent-2609

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

产物按段分目录（`*-sft1 / *-sft2 / *-sft3`），与 `EXPERIMENT_MATRIX.json` 的 sft1/sft2 行一一对应。
