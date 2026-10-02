"""agent 方向的规则奖励。"""

from __future__ import annotations


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    try:
        return float(json.loads(ground_truth)["success"])  # type: ignore[name-defined]
    except Exception:  # noqa: BLE001
        gt = str(ground_truth).strip().lower()
        return 1.0 if gt in ("1", "true", "success") else 0.0


import json  # noqa: E402  （放在函数后是刻意的：保持本文件"一眼看懂奖励"的形状）
