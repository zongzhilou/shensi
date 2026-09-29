#!/usr/bin/env python3
# 世界模型的 RL 奖励：预测的下一状态 vs 真观测，按 AgentWorldBench 五维判分（0~1 送给 verl）。
# 判分端点默认 = 世界模型端点；判分模型建议换个更强的（$SHENSI_JUDGE_MODEL / $SHENSI_JUDGE_URL）。

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agentworld 在 stage2_rl/ 下

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wm_common  # noqa: E402
from agentworld.eval.lwm_eval_utils import (  # noqa: E402
    TASK_CONFIGS,
    parse_judge_output,
)


def score_from_judge_output(raw: str, domain: str) -> float:
    """裁判输出 → 0~1（五维总分 / 5）；解析不出来就是 0 分。"""
    parsed = parse_judge_output(raw, TASK_CONFIGS[domain]["judge_response_tag"])
    if not parsed.get("success"):
        return 0.0
    return max(0.0, min(1.0, float(parsed.get("total_score", 0)) / 5.0))


def _judge_client():
    import world_model as wm

    return wm.WorldModelClient(
        base_url=os.environ.get("SHENSI_JUDGE_URL") or os.environ.get("SHENSI_WORLD_MODEL_URL"),
        model=os.environ.get("SHENSI_JUDGE_MODEL") or os.environ.get("SHENSI_WORLD_MODEL"),
        temperature=0.0,  # 判分要稳定
    )


def judge_once(prediction: str, ground_truth: str, domain: str, current_prompt: str = "", context: str = "") -> str:
    job = {
        "task": domain,
        "prompt": [context or current_prompt],
        "response": [ground_truth],
        "turn_idx": 1,
        "current_prompt": current_prompt,
    }
    messages = wm_common.judge_messages(job, prediction, domain)
    return _judge_client().chat(messages)


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """verl 的自定义奖励接口：solution_str = 世界模型预测的观测，ground_truth = 真观测。"""
    extra_info = extra_info or {}
    domain = str(extra_info.get("domain") or "")
    if domain not in TASK_CONFIGS:
        domain = wm_common.domain_of({"data_source": data_source, "task": str(extra_info.get("source") or "")})
    prompt_text = str(extra_info.get("prompt_text") or "")
    context = ""
    if prompt_text:
        try:
            history = json.loads(prompt_text)
            context = "\n".join(str(m.get("content") or "") for m in history if isinstance(m, dict))
        except json.JSONDecodeError:
            context = prompt_text
    raw = judge_once(
        solution_str,
        ground_truth,
        domain,
        current_prompt=str(extra_info.get("current_prompt") or ""),
        context=context,
    )
    return score_from_judge_output(raw, domain)
