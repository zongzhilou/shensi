# Stage 2.1: 多环境可验证奖励 RL（RLVR）

`stage2_rl` 的第一段：单轮为主、答案可校验。数据用 post-training 里带 verifier 的 RL 集与官方
`*-Training-Blends`（它们已经把多环境、多奖励混好了），奖励由 [`../reward.py`](../reward.py) 按 verifier 规则判。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口（`../rl.py` 的薄封装）：`--profile` / `--data-dir` / `--dry-run` / `--set` |
| `test_train.py` | 集成预检：配置→命令、数据 parquet、ray、GPU、import、环境变量 |
| `data_prep.py` | 语料 → verl 的 RL schema（`train.parquet` / `val.parquet`） |
| `config/` | `default.yaml` + `debug.yaml` + 算法档 `gspo.yaml` / `dapo.yaml` |

| 项 | 值 | 出处 |
| --- | --- | --- |
| 目标 | 在可校验任务上把「答对」的比例顶起来（数学 / 科学 / 代码 / 推理） |
| 数据 | `config/data_prep/data_blend_raw.json`：11 个可校验集，权重按本阶段归一 | Nemotron-3 的 21 环境 RLVR 集与 `*-Training-Blends` |
| 算法 | GRPO（verl 原生 `algorithm.adv_estimator`），不做 KL | GLM-5 的 RLVR 段 |
| 采样 | `rollout.n = 8`，`max_response_length = 32768`（推理链要够长） | GLM-5 报告 |
| LR | 1e-6 恒定，`clip_ratio_low/high = 0.2/0.28` | GLM-5 的 IcePop 式双侧截断 |
| 优化器 | verl 的 `actor.optim`（Adam 系） | 预训练用 Muon 混合的理由与 SFT 相同：小数据微调要单独扫 LR |

**算法档**：默认 GRPO；`config/gspo.yaml`（序列级重要性比，长思维链更稳）与 `config/dapo.yaml`
（clip-higher + 动态采样，group 拉到 16）都是 verl 原生的 `algorithm.adv_estimator`，`--profile gspo` /
`--profile dapo` 直接切。判分侧：代码类没有单测时可用模型判分（THUDM CodeRM-NT 的思路），
现在 `reward.py` 是 verifier 规则 + 兜底 0。

## Quick Start

```bash
python test_train.py --data-dir <parquet 目录>    # 集成预检（不跑完整 GRPO）
python data_prep.py --discover                     # 看数据在不在
python data_prep.py --prepare                      # → $SHENSI_FS/shensi/data/stage1_rlvr/{train,val}.parquet
python train.py --dry-run                          # 看 verl 命令
python train.py --profile debug --data-dir <目录>   # 极小档（1 epoch、少采样）
python train.py --set model.path=<sft ckpt>        # 正式跑：上一段 ckpt 用 --set 指过去
```

## 数据

11 个可校验集（数学 / 科学 / 代码 / 推理），每条数据行带 `verifier`（判分规则）与
`reward_model.ground_truth`；`--discover` 会打印每个数据集在不在、条数与字段名。
本机冒烟可以只用 40 条自造的小语料（`problem` + `answer`）跑通全链路。

## 验收判据

1. **集成预检 PASS**：`python test_train.py --data-dir <目录>`；
2. `critic/score/mean` 高于基线并上行；
3. `actor/entropy` 不塌到 0；
4. 同一 prompt 采样 8 次能看到不同解法（多样性还在）；
5. 早停：`../../early_stop.py --metric critic/score/mean --mode max`。

本机实测（debug 档 + 40 条小语料）：rc=0、`step:0`（验证）→ `step:19`（一个 epoch 跑满），
含优化器步与权重更新，无报错。

## 下一步

`stage2_agentic`（多轮 + 工具 + 环境）。

## 局限

1. 奖励只覆盖规则可判的部分（`string_match` / 精确 / 数字 / `pass_rate` 软标签）；
   开放式任务的判分要等 `stage3_align` 的 GenRM 通道；
2. 11 个集的相对权重按本阶段归一，实际 token 占比要看 `--discover` 的实测条数；
3. 注意力口径：verl 会把 response 右 padding，mcore 的 CSA 不接受显式 mask →
   丢掉右 padding 的 mask（等价），尾部 pad 仍会经压缩块参与计算（与 FL fork 同口径，见配方 README）。

## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

```bash
python data_prep.py --prepare --blend config/data_prep/debug_sample.json --limit 40
python -m shensi.recipes.shensi.train.export_hf \
    --ckpt $SHENSI_FS/shensi/ckpt/stage1_sft_debug --out $SHENSI_FS/shensi/models/sft-hf --tiny
python train.py --profile debug --data-dir $SHENSI_FS/shensi/data/stage1_rlvr \
  --set model.path=$SHENSI_FS/shensi/models/sft-hf
```

- 从**导出的 SFT ckpt** 起跑：19/19 步（`Training Progress: 100%`），
  权重同步 20 次，无报错；
- 末尾指标里能看到 `actor/entropy`、`training/rollout_probs_diff_*`（on-policy 一致性）与
  `global_seqlen/*`，说明 rollout 的 log-prob 与 actor 的重算是对齐的。
