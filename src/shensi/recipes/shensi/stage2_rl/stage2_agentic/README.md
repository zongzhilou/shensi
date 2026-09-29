# stage2_agentic：长时程 agentic RL

`stage2_rl` 的第二段：多轮 + 工具 + 环境，奖励来自环境 verifier。这一段是 GLM-5.2/5.3 全部增益的来源
（agentic RL + SAO 式单轨迹 async），也是 Nemotron-3 的 SWE-RL 段（容器内改仓库、跑测试）。

## 1. 摘要

| 项 | 结论 |
| --- | --- |
| 数据 | Agentic / SWE / 工具调用类 RL 集（6 个，见 `config/data_prep/data_blend_raw.json`），`agent_ref` 与 `verifier` 一起带进训练行 |
| 与 RLVR 的差别 | `base:` 继承 `../stage1_rlvr/config`，只覆盖几项：`rollout.n: 4`、`max_response_length: 65536`、`lr: 5e-7`、`total_epochs: 2` |
| harness（真机档） | 环境与工具层用 **NeMo Gym** 或 **DeepSeek Harness（dsh）**（都要一个 OpenAI/DeepSeek 兼容端点，就是我们的 `vllm serve`）；装法见 `../../stage3_eval/setup_env.sh` |
| harness（Sim RL 档） | `--profile world_model`：环境换成**语言世界模型**，观测由模型预测（见第 3 节） |
| 判据 | 任务完成率上行；轨迹长度分布稳定；工具调用格式错误率下降 |

```bash
python data_prep.py --prepare && python train.py --dry-run && python train.py
```

## 2. 真机档：环境与工具

多轮状态机与工具解析走上游 verl 的 `ToolAgentLoop`（`config` 里 `rollout.agent.default_agent_loop: tool_agent`），
工具用 Gym / dsh 提供的容器与工具面；训练侧只提供数据行的 `agent_ref` 与 `verifier`。

## 3. Sim RL 档：世界模型当环境

`--profile world_model` 把环境换成**语言世界模型**（Qwen-AgentWorld 口径，七域：terminal / swe / search /
mcp / android / web / os），观测由它预测而不是真机给。环境可以无限扩、可以注入扰动、可以是虚构世界；
真机那套（Gym/dsh）照旧——两档只差一个 `--profile`。

- `world_model.py`：环境本身。`WorldModelEnv` 是一个 session（历史逐轮累积），三种口径 `sim` /
  `control`（`spec.perturbations` 注入扰动）/ `fiction`（`spec.world` 虚构世界）；还能起 HTTP 环境服务
  给外部 harness：`python world_model.py serve --port 9000` → `GET /health`、`POST /reset`、`POST /step`。
  离线自测（假世界模型起端点，不需要 GPU）：`python world_model.py check`（15 项判定）。
- `world_model_tool.py` + `config/tools/world_model.yaml`：接成 **verl 工具**，多轮状态机与工具调用解析
  直接走上游 `ToolAgentLoop`，本仓库只实现一个工具；数据行可用
  `extra_info.tools_kwargs.env_action.create_kwargs` 逐行覆盖 `domain / mode / spec / task`。
  工具配置里把 `dump_dir` 打开，就把（动作, 观测）轨迹落盘——那是 `../stage4_world_model`（世界模型）的语料。
- `config/world_model.yaml`：Sim RL 档（`rollout.n: 8`、12 个动作轮、`tool_config_path` 指上面那份）。
  `max_assistant_turns` 是动作轮上限（每个动作算一轮），没有 user 轮；`format: hermes` 对应上游的 `ToolParser`。
  若本仓库的 verl 版本对 vLLM 多轮有限制，把 `rollout.name` 换成 `sglang` 即可（引擎选择与 Sim RL 无关）。

```bash
vllm serve Qwen/Qwen-AgentWorld-35B-A3B --port 8000 --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --language-model-only --trust-remote-code
export SHENSI_WORLD_MODEL_URL=http://127.0.0.1:8000/v1
python data_prep.py --prepare && python train.py --profile world_model --dry-run && python train.py --profile world_model
```

依据（Qwen-AgentWorld，[2606.24597](https://arxiv.org/abs/2606.24597)）：Sim RL 用 4k 个 OOD 环境，
Claw-Eval 65.4 → 69.7、QwenClawBench 47.9 → 55.0；可控扰动 +3.7 / +12.3；虚构世界让真实检索 F1 34.02 → 50.31；
单轮 LWM RL warm-up 也迁移到多轮工具调用（BFCL v4 62.29 → 71.25）。

## 4. 验收判据

1. 任务完成率（环境 verifier 给分）随步数上行；
2. 轨迹长度分布稳定（不塌成 1 轮、不顶到上限）；
3. 工具调用格式错误率下降；
4. Sim 档与真机档的完成率差距随世界模型变强而收窄（差距就是世界模型的建模误差）。

## 5. 局限

1. 真机档要 Gym / dsh 的容器与基准资产（见 `../../stage3_eval/README.md` 的"需要目标环境才能跑"一节）；
2. Sim 档的观测由世界模型生成，**保真度决定上限**：世界模型没见过的域（比如长尾 GUI）会系统性偏乐观，
   要在判据 4 里盯着；
3. 轨迹落盘（`dump_dir`）默认关，开之前先评估磁盘与后续语料清洗成本。
