"""写作方向的规则奖励。"""

from __future__ import annotations

RUBRIC_MIN_WORDS = 200


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    words = len(solution_str.split())
    length_score = min(1.0, words / RUBRIC_MIN_WORDS)
    structure_score = 1.0 if ("\n\n" in solution_str and words > RUBRIC_MIN_WORDS) else 0.5
    gt = str(ground_truth).strip()
    overlap = 0.0
    if gt:
        gt_words = set(gt.lower().split())
        overlap = len(gt_words & set(solution_str.lower().split())) / max(1, len(gt_words))
    return round(0.4 * length_score + 0.3 * structure_score + 0.3 * overlap, 4)
