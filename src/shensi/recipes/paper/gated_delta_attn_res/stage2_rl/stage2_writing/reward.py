#!/usr/bin/env python3
"""写作 的奖励：rubric 打分（0~1）。

写作没有 0/1 的可验证答案：这里给一个**规则 rubric 骨架**（长度/结构覆盖），占位用。
生产口径是 reward model（成对偏好训练出的 RM）或 LLM-as-judge，接口不变（返回标量）。
"""

from __future__ import annotations

RUBRIC_MIN_WORDS = 200


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    words = len(solution_str.split())
    length_score = min(1.0, words / RUBRIC_MIN_WORDS)
    # 结构分：至少出现一个分段与一个收尾段（占位启发式，见 docstring 的诚实标注）
    structure_score = 1.0 if ("\n\n" in solution_str and words > RUBRIC_MIN_WORDS) else 0.5
    gt = str(ground_truth).strip()
    overlap = 0.0
    if gt:
        gt_words = set(gt.lower().split())
        overlap = len(gt_words & set(solution_str.lower().split())) / max(1, len(gt_words))
    return round(0.4 * length_score + 0.3 * structure_score + 0.3 * overlap, 4)
