# agentworld：世界模型的提示词与判分工具

一处**原样副本**（Apache-2.0，提交 `cd0aa83dc7a9c733695eb9c4652e0a68b6e6ecde`，2026-07-20），
随包分发以保证 Sim RL 与评测的口径一致。

## 内容

| 路径 | 内容 |
| --- | --- |
| `prompts/<域>/{system_prompt.txt,judge_system_prompt.txt}` | 七个域（terminal / swe / search / mcp / android / web / os）的世界模型系统提示词与判分提示词 |
| `eval/lwm_eval_utils/` | 输出解析与判分解析（五维 Format / Factuality / Consistency / Realism / Quality；`<predicted_observation>` / `<final_evaluation>` 两个标签的鲁棒提取） |
| `LICENSE` | 上游 Apache-2.0 许可原文 |

## 与上游的差异

两处非功能性改动：新增 `eval/__init__.py` 让 `eval.lwm_eval_utils` 可按包路径导入；补一份 `LICENSE`。
`lwm_eval_utils/judge_parser.py` 用 `Path(__file__).parent.parent.parent` 定位仓库根，所以这里保持
`eval/lwm_eval_utils/` 与 `prompts/` 的相对位置不变。

## 用法

| 位置 | 用法 |
| --- | --- |
| `stage2_rl.stage2_agentic.world_model` | 把世界模型当 RL 环境（Sim RL）；系统提示词从这里读 |
| `stage2_rl.stage4_world_model.bench` | 按 AgentWorldBench 口径给任意世界模型打分 |
| `stage2_rl.stage4_world_model.reward` | RL 的保真度奖励（五维总分 / 5，归一到 0~1） |

## 局限

本目录只包含提示词与判分工具，**不含权重与训练代码**——那部分用模型仓库的 ckpt 与我们的训练器
（[`../stage4_world_model`](../stage4_world_model/README.md)）。
