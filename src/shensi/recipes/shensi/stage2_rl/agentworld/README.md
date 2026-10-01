# agentworld: Prompts and Judge Utilities for the World Model

A verbatim copy of `QwenLM/Qwen-AgentWorld` (Apache-2.0, commit
`cd0aa83dc7a9c733695eb9c4652e0a68b6e6ecde`, 2026-07-20), vendored so that Sim RL and evaluation share one
protocol.

## Contents

| Path | Description |
|------|-------------|
| `prompts/<domain>/{system_prompt.txt,judge_system_prompt.txt}` | World-model system prompts and judge prompts for seven domains (terminal / swe / search / mcp / android / web / os) |
| `eval/lwm_eval_utils/` | AgentWorldBench output and judge parsing (the five dimensions Format / Factuality / Consistency / Realism / Quality; robust extraction of the `<predicted_observation>` and `<final_evaluation>` tags) |
| `LICENSE` | The upstream Apache-2.0 license text |

## Differences from upstream

Two non-functional changes only: an added `eval/__init__.py` so `eval.lwm_eval_utils` imports as a package
path, and the bundled `LICENSE`. Upstream's `lwm_eval_utils/judge_parser.py` locates the repository root
with `Path(__file__).parent.parent.parent`, so the relative layout of `eval/lwm_eval_utils/` and
`prompts/` is kept unchanged.

## Usage

| Location | Use |
|----------|-----|
| `shensi.recipes.shensi.stage2_rl.stage2_agentic.world_model` | The world model as an RL environment (Sim RL); system prompts are read from here |
| `shensi.recipes.shensi.stage2_rl.stage4_world_model.bench` | Scores any world model under the AgentWorldBench protocol (including trained ones) |
| `shensi.recipes.shensi.stage2_rl.stage4_world_model.reward` | The RL fidelity reward (five-dimension total / 5, normalized to 0–1) |

## Limitations

This directory contains the prompts and judge utilities only — **not the weights or training code**.
Those come from a model repository's checkpoint (e.g. `Qwen/Qwen-AgentWorld-35B-A3B`) and from our
trainer ([`../stage4_world_model`](../stage4_world_model/README.md)).
