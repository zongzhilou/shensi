"""数学方向的规则奖励：抽取最终答案并核对。"""

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
