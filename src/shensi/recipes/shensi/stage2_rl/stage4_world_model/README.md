# stage4_world_model：把环境模拟练成模型

`stage2_rl` 的第四个子 stage。**语言世界模型（language world model）**：给它一段交互历史和一个动作，
它预测环境会返回什么观测。这一段与 `../stage2_agentic` 的 `--profile world_model` 是一对：
那边是**用**世界模型当环境，这里是**练**世界模型。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 对齐 | Qwen-AgentWorld（[2606.24597](https://arxiv.org/abs/2606.24597)）：七域（terminal / swe / search / mcp / android / web / os）、三阶段 CPT → SFT → RL、五维判分（Format / Factuality / Consistency / Realism / Quality） |
| 为什么值 | Sim RL 用 4k 个 OOD 环境：Claw-Eval 65.4 → 69.7、QwenClawBench 47.9 → 55.0；可控模拟（注入扰动）+3.7 / +12.3；虚构世界让真实检索 F1 34.02 → 50.31；单轮 LWM RL warm-up 迁移到多轮工具调用（Terminal-Bench 2.0 33.25 → 39.55、BFCL v4 62.29 → 71.25） |
| 与其它阶段的关系 | 从 `stage0_pretrain` 的基座出发，**不重写训练器**：三段分别复用 `stage2_midtrain` / `stage1_sft` / `rl`+verl |
| 产物 | 一个能当环境的模型（挂到 `stage2_agentic/config/world_model.yaml` 的 `base_url`），以及 AgentWorldBench 口径的评测报告 |

```text
stage2_rl（eval 前的那一段）
├── stage1_rlvr / stage2_agentic / stage3_align   ← 策略这条线（环境可真机、也可世界模型）
└── stage4_world_model                            ← 这一段：把「动作 → 观测」练成模型
        stage0_pretrain（基座）→ 本段 CPT/SFT/RL ──► stage2_agentic 的 Sim RL 环境
```

## 2. 三段与复用的训练器

| 段 | 复用什么 | 学什么 |
| --- | --- | --- |
| ① CPT `--step cpt` | `stage0_pretrain/stage2_midtrain`（FlagScale + Megatron） | 注入环境知识：把交互轨迹当纯文本继续预训练 |
| ② SFT `--step sft` | `stage1_sft`（FlagScale `--sft`，DeepSeek-V4 编码） | 学「下一状态」：给历史 + 动作，输出 `**Environment Observation:**` + `<predicted_observation>` |
| ③ RL `--step rl` | `stage2_rl` 的 `rl` + verl GRPO | 顶模拟保真度：奖励 = 五维判分（`reward.py`） |

## 3. 语料

来源两处，`config/data_prep/data_blend_raw.json` 配权重（和归一为 1.00）：

1. **自家 Sim RL 落盘的轨迹**（`stage2_agentic` 的工具配置里把 `dump_dir` 打开，落到
   `$SHENSI_FS/shensi/data/world_model_traj`）——这条是闭环：世界模型扩出来的轨迹再喂回世界模型；
2. post-training 的多轮 agentic 集（同一批 RL 语料，这里读它的（动作, 观测））。

`data_prep.py` 一次产出三种形态：CPT 的纯文本（走 `stage0_pretrain` 的切片 + tokenize + blend）、
SFT 的 messages（渲染/packed/loss_mask 交给 `stage1_sft` 的 `data_prep`）、RL 的
（历史 + 动作 → 真观测）parquet（`reward_model.style = world_model_judge`）。

```bash
python data_prep.py --discover --blend config/data_prep/debug_sample.json
python data_prep.py --step all --blend config/data_prep/debug_sample.json   # debug 档自带 7 个域各一条轨迹
python train.py --step all --profile debug --dry-run                        # 三段先看命令
python train.py --step sft --profile debug
```

## 4. 评测（AgentWorldBench 口径）

`bench.py` 给任意世界模型打分（Qwen 的现成模型、我们自训的、或中途的 ckpt 都行）：读上游 `*_test.jsonl`
（`{task, system_str, prompt[], response[], turn_idx}`）或自家轨迹，先让世界模型预测下一状态，
再让判分模型按五个维度打 1–5 分，汇总写 `summary.json`（分域 + overall；判分失败记 0 并计入 `invalid`，不当崩）。

```bash
# 世界模型端点（--language-model-only 是必须的：该 ckpt 只有语言权重，否则 vLLM 会去初始化视觉模块）
# vllm serve Qwen/Qwen-AgentWorld-35B-A3B --port 8000 --max-model-len 262144 \
#   --reasoning-parser qwen3 --language-model-only --trust-remote-code
python bench.py --data <上游 *.jsonl 或自家轨迹> --limit 200 \
  --lwm-url http://127.0.0.1:8000/v1 --judge-url <判分端点> --judge-model <判分模型>
python bench.py --stub        # 离线自测：假世界模型 + 假裁判，7 项判定
```

七域的系统提示词与五维判分（含 `JUDGE_USER_PROMPT` 与鲁棒解析）随代码内联在 `stage2_rl/agentworld/`
（Apache-2.0，原样保留上游文件）。判分端点建议比世界模型更强，用 `SHENSI_JUDGE_URL` / `SHENSI_JUDGE_MODEL` 指；
世界模型端点用 `SHENSI_WORLD_MODEL_URL` / `SHENSI_WORLD_MODEL`。

## 5. 验收判据

1. **CPT**：域内术语与工具名不再被拆错（看 tokenize 后的样例）；
2. **SFT**：held-out 观测的 token 级 loss 下行；预测里 `**Environment Observation:**` 格式合规率接近 1；
3. **RL**：`bench.py` 的五维总分上行，`invalid` 不涨（判分能被解析）；
4. **回灌**：把训好的 ckpt 挂到 `stage2_agentic` 的 `world_model.yaml` 档（`base_url` 指它），
   任务完成率对比用真环境的那一档。

## 6. 局限

1. **没有真机验证过**：`bench.py --stub` 与 `world_model.py check` 都是离线桩验证；接真模型（4 卡起）才能出分数；
2. 世界模型的上限是数据：轨迹来源只有 Sim RL 自产 + post-training 的 agentic 集，域覆盖不均（search/mcp 偏少）；
3. 判分依赖模型裁判，裁判自身偏好会进入 RL 奖励（与 `stage3_align` 同源风险）。
