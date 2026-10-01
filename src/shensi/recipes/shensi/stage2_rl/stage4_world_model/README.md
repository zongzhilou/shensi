# Stage 2.4: 世界模型（动作 → 观测）

`stage2_rl` 的第四个（与策略这条线**并行**）：把「动作 → 观测」练成一个模型，产物就是
[`../stage2_agentic --profile world_model`](../stage2_agentic/README.md) 的 Sim RL 环境。

## 总览

| 组件 | 做什么 |
|------|--------|
| `train.py` | 入口：`--step cpt/sft/rl/all` 三段；`--profile` 选档；`--dry-run` |
| `test_train.py` | 集成预检（RL 段用 `config/rl/<档>.yaml`；另查样例轨迹在不在） |
| `data_prep.py` | 轨迹 → ① CPT 纯文本 / ② SFT「历史 + 动作 → 观测」/ ③ RL 交互行 |
| `wm_common.py` | 三段共用的路径与数据口径（含判分提示词的组装） |
| `reward.py` | RL 段的奖励：五维判分（0~1） |
| `bench.py` | 按 AgentWorldBench 口径给任意世界模型打分 |
| `stub_judge.py` | CPU 上的判分桩（`--mode hash` 按输入伪随机，保证组内有区分度） |
| `local_judge.py` | **CPU 上的真模型判分**：小模型逐维打分后组装官方 `<final_evaluation>` JSON，打印解析率 |
| `config/` | `default.yaml` + `tiny.yaml` + `debug.yaml` + `rl/*.yaml` + `data_prep/` |

| 段 | 上游训练器 | 数据形态 | 目标 |
|----|-----------|---------|------|
| ① CPT `--step cpt` | [`stage0_pretrain/stage2_midtrain`](../../stage0_pretrain/stage2_midtrain/README.md) | 轨迹文本（含动作与观测） | 注入环境知识 |
| ② SFT `--step sft` | [`stage1_sft`](../../stage1_sft/README.md)（mcore `--sft`） | `{"messages": [...]}` | 学「下一状态」：输出 `**Environment Observation:**` + `<predicted_observation>` |
| ③ RL `--step rl` | verl（GRPO + 五维判分） | 交互行（含 `spec`） | 对齐模拟保真度 |

轨迹来源：[`../stage2_agentic`](../stage2_agentic/README.md) 的工具配置里打开 `dump_dir`，落盘的
（动作, 观测）就是本段语料；`config/data_prep/sample_traj.jsonl` 是冒烟样例（七个域各一条）。

## 快速开始

```bash
python test_train.py                       # 集成预检
python data_prep.py --discover             # 轨迹面貌
python data_prep.py --prepare              # 三段的数据都产出
python train.py --step cpt --dry-run        # ① 环境知识
python train.py --step sft                 # ② 下一状态
python train.py --step rl                  # ③ 保真度对齐
python train.py --step all                  # 三段连着跑
```

端点从环境变量读：`SHENSI_WORLD_MODEL_URL` / `SHENSI_WORLD_MODEL`（世界模型）、
`SHENSI_JUDGE_URL` / `SHENSI_JUDGE_MODEL`（判分，默认同世界模型）。

### 判分端点（本机可全 CPU）

```bash
# 世界模型：桩（不占显存）或自己的服务
python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.stub_judge --port 9000 --mode hash
# 判分：CPU 小模型（逐维打分 → 官方五维 JSON；解析率会打印）
python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.local_judge \
    --model-dir $SHENSI_FS/models/SmolLM2-360M-Instruct --port 8000
export SHENSI_WORLD_MODEL_URL=http://127.0.0.1:9000/v1 SHENSI_WORLD_MODEL=stub-judge
export SHENSI_JUDGE_URL=http://127.0.0.1:8000/v1 SHENSI_JUDGE_MODEL=SmolLM2-360M-Instruct
```

## 验证

1. **集成预检 PASS**（含样例轨迹在位）；
2. ② 的 SFT：`<predicted_observation>` 格式合法率上升；对留出轨迹做文本相似度抽测；
3. ③ 的 RL：`critic/score/mean`（五维）上行；
4. 端到端：把 ③ 的产物接回 `stage2_agentic --profile world_model`，Sim 与真机完成率差距收窄。

本机实测：三段一次跑通（CPT 78 步 → SFT 2 步 → RL 3 步，判分来自桩端点）；`local_judge --check`
离线自检把 360M 模型的 `1 2 3 4 5` 组装成官方五维 JSON（五维键齐全，CPU、不占显存）。

## 局限

1. 判分器自身的偏好会进入世界模型；换裁判模型时先小规模对拍（`local_judge` 会报解析率，
   解析率低说明该换更大裁判）；
2. 轨迹数据依赖真机 agentic 段落盘，量不够时世界模型会过拟合到少数域；
3. 世界模型的"真裁判"若用外部大模型端点，速度与配额是外部约束；本机默认走 CPU 小模型或桩。

## 下一步

产物回给 [`../stage2_agentic`](../stage2_agentic/README.md)（Sim RL 环境）；策略这条线继续
[`../stage3_align`](../stage3_align/README.md) 与[评测](../../stage3_eval/README.md)。
