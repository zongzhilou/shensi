#!/usr/bin/env python3
"""数学 的可验证奖励（`reward.custom_reward_function.name=compute_score`）。

数学：抽取 \\boxed{} / 末个数值做答案比对（0/1）。生产可换 PRM 或多数投票比对，
接口不变。
"""

from __future__ import annotations

import math
import re


def _extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    pred = _extract_answer(solution_str)
    if pred is None:
        return 0.0
    try:
        return 1.0 if math.isclose(float(pred), float(ground_truth), rel_tol=1e-6) else 0.0
    except (ValueError, TypeError):
        return 1.0 if pred == str(ground_truth).strip() else 0.0
