# Stage 2.3: 偏好 / 指令 / 安全对齐

`stage2_rl` 的第三段：轨迹不再靠规则判分。对齐 Nemotron-3 的 RLHF + **GenRM** 判分段；
GRPO 与 IcePop 截断口径不变（见 [`../README.md`](../README.md)）。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口（RL 三段里最简单的一个：同一套 verl 命令） |
| `test_train.py` | 集成预检 |
| `data_prep.py` | 偏好 / 指令遵循 / 安全集 → parquet（`reward_model.ground_truth` 换成判分规格） |
| `config/` | `default.yaml` + `debug.yaml`（`base:` 继承 `../stage2_agentic/config`） |

| 项 | 值 | 说明 |
| --- | --- | --- |
| 目标 | 指令遵循（结构化输出 / 日历 / 多轮）、安全性（红线）、偏好质量 |
| 数据 | 指令遵循（结构化输出 / 日历 / 多轮）、InverseIFEval、安全（5 个集） | `config/data_prep/data_blend_raw.json` |
| 超参 | `base: ../stage2_agentic/config` 覆盖：`rollout.n: 8`、`max_response_length: 8192`、`lr: 5e-7`、`total_epochs: 2` | 与上一段同源，只调轨迹长度与 LR |
| 判分 | 打开 verl 的 `reward_model` 通道，`reward_model.ground_truth` 换成偏好/评分规格；判分模型（GenRM）可以是本仓库自己的 ckpt 或托管端点 | Nemotron-3 的 GenRM 做法 |

## Quick Start

```bash
python test_train.py --data-dir <parquet 目录>       # 集成预检
python data_prep.py --prepare && python train.py --dry-run && python train.py
```

## 验收判据

1. **集成预检 PASS**；
2. 安全类不退化（红线用例 0 命中）；
3. 指令遵循的结构化输出合法率上升；
4. `critic/score/mean` 不塌；
5. 早停：`../../early_stop.py --metric critic/score/mean --mode max`。

## 下一步

评测见 [Stage 3: 评测](../../stage3_eval/README.md)。

## 局限

1. GenRM 判分模型的选型与规模未做消融；判分器的自身偏好会直接进入策略（同源风险），
   要接托管端点或换更大判分器时先小规模对拍；
2. 判分端点属于外部依赖（本机用桩验过链路）：把 `reward.judge_model` 指向外部端点即可，
   预检会报端点与环境是否就位。
## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

```bash
python data_prep.py --prepare --blend config/data_prep/debug_sample.json --limit 40
python train.py --profile debug --data-dir $SHENSI_FS/shensi/data/stage3_align \
  --set model.path=$SHENSI_FS/shensi/models/sft-hf          # 由 export_hf.py 从 SFT ckpt 导出
```

- 从导出的 SFT ckpt 起跑：19/19 步通过，权重同步 20 次。
