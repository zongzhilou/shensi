# Stage 2.4: 世界模型（动作 → 观测）

`stage2_rl` 的第四个（与策略这条线**并行**的子 stage）：把「动作 → 观测」练成一个模型，
产物就是 `stage2_agentic --profile world_model` 的 Sim RL 环境。依据 Qwen-AgentWorld
（[2606.24597](https://arxiv.org/abs/2606.24597)）：Sim RL 用语言世界模型当环境，4k 个 OOD 环境上
Claw-Eval 65.4 → 69.7；可控扰动 +3.7 / +12.3；虚构世界让真实检索 F1 34.02 → 50.31；
单轮 LWM RL warm-up 也迁移到多轮工具调用（BFCL v4 62.29 → 71.25）。

## Overview

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--step cpt/sft/rl/all` 三段；`--profile` 选档；`--dry-run` |
| `test_train.py` | 集成预检（RL 段用 `config/rl/<档>.yaml`；另查 Sim RL 样例轨迹在不在） |
| `data_prep.py` | 轨迹 → ① CPT 纯文本 / ② SFT「历史 + 动作 → 观测」/ ③ RL 交互行 |
| `wm_common.py` | 三段共用的路径与数据口径 |
| `reward.py` | RL 段的奖励：AgentWorldBench 的五维判分 |
| `config/` | `default.yaml`（三段各自的上游档）+ `debug.yaml` + `rl/*.yaml` + `data_prep/sample_traj.jsonl` |

| 段 | 上游训练器 | 数据形态 | 目标 |
|----|-----------|---------|------|
| ① CPT `--step cpt` | `stage0_pretrain/stage2_midtrain`（mcore + Megatron-Bridge） | 轨迹文本（含动作与观测） | 注入环境知识：把交互轨迹当纯文本继续预训练 |
| ② SFT `--step sft` | `stage1_sft`（mcore `--sft`，DeepSeek-V4 编码） | `{"messages": [...]}` | 学「下一状态」：给历史 + 动作，输出 `**Environment Observation:**` + `<predicted_observation>` |
| ③ RL `--step rl` | verl（GRPO + 五维判分） | 交互行（含 `spec`） | 对齐模拟保真度 |

轨迹的来源：`stage2_agentic --profile world_model` 的工具配置里打开 `dump_dir`
（见 [`../stage2_agentic/README.md`](../stage2_agentic/README.md)），落盘的（动作, 观测）就是本段的语料；
`config/data_prep/sample_traj.jsonl` 是给冒烟用的一小份样例。

## Quick Start

```bash
python test_train.py                       # 集成预检
python data_prep.py --discover             # 轨迹面貌
python data_prep.py --prepare              # 三段的数据都产出
python train.py --step cpt --dry-run        # ① 环境知识
python train.py --step sft                 # ② 下一状态
python train.py --step rl                  # ③ 保真度对齐
python train.py --step all                  # 三段连着跑
```

## 验收判据

1. **集成预检 PASS**（含样例轨迹在位）；
2. ② 的 SFT：`<predicted_observation>` 的格式合法率上升；对留出轨迹的观测做文本相似度抽测；
3. ③ 的 RL：`critic/score/mean`（五维判分）上行；
4. 端到端：把 ③ 的产物接回 `stage2_agentic --profile world_model`，Sim 档与真机档的完成率差距收窄。

## 下一步

产物回给 `stage2_agentic`（Sim RL 环境）；策略这条线继续 `stage3_align` 与评测。

## 局限

1. 三段里本机只跑过 ①/② 的极小档口径（走的是 stage0/stage1 的那套入口），③ 只做预检；
2. 判分（AgentWorldBench 五维）依赖判分模型，判分器自身的偏好会进入世界模型；
3. 轨迹数据依赖真机 agentic 段落盘，量不够时世界模型会过拟合到少数域。

## 本机实跑记录（2026-10-01，WSL2 + RTX 5080 16G）

全部命令都在本机真跑过（单卡），日志与 run 目录在 `$SHENSI_FS/shensi/runs/`；极小档产物的生成见配方总览的「极小档要两个本地产物」。

```bash
# 数据：自带 7 条轨迹（每个域一条），三段一次做完
python data_prep.py --step all --blend config/data_prep/debug_sample.json --limit 40 --max-turn-chars 1200
# CPT（78 步）+ SFT（2 步）：都实跑过
python train.py --step cpt --profile debug --data-dir $SHENSI_FS/shensi/data/stage2_world_model
python train.py --step sft --profile debug --data-dir $SHENSI_FS/shensi/data/stage2_world_model
# RL：要先有判分端点（见下）
SHENSI_WORLD_MODEL_URL=http://127.0.0.1:8000/v1 SHENSI_WORLD_MODEL=world-model \
  python train.py --step rl --profile debug --data-dir $SHENSI_FS/shensi/data/stage2_world_model \
  --set model.path=$SHENSI_FS/shensi/models/tiny-rl
```

- **CPT / SFT 两段本机通过**（78 步 + 2 步，ckpt 落在 `ckpt/stage2_world_model/{cpt,sft}`）；
- **RL 段本机跑不了**：奖励是 LLM 裁判（`reward.py`），即要在同一张卡上多起一个模型服务。
  实测 16G 单卡上「判分端点 + rollout 引擎 + actor」会先把 WSL 的 GPU 驱动压爆
  （actor 前向里报 `CUDA driver error: device not ready`，`dmesg` 是 `dxgkio_make_resident:
  Ioctl failed: -12`）。要跑这段就另配一台判分服务（`SHENSI_JUDGE_URL`）或换大卡；
- 极小档的数据要 `--max-turn-chars` 掐一下：世界模型的 system prompt 本身就有 ~2.5 万字，
  真轨迹的 prompt 能到 2 万 token（tiny-rl 的 `max_position_embeddings` 要 ≥16k，
  `python -m shensi.recipes.shensi.tiny_artifacts --model-only --max-position-embeddings 16384`）。
