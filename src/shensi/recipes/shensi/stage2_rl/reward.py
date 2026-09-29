#!/usr/bin/env python3
import json
import re


def _parse(spec):
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str) and spec.strip().startswith("{"):
        try:
            return json.loads(spec)
        except Exception:  # noqa: BLE001
            return {}
    return {}


# verifier 优先：string_match → 精确/数字 → pass_rate 软标签，兜底 0
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    extra = extra_info or {}
    spec = _parse(ground_truth) or _parse(extra.get("verifier"))
    sol = solution_str or ""

    # ① string_match：所有 marker 都要出现（Lightning 的 citation_format 这类）
    markers = spec.get("expected_markers") or spec.get("expected")
    if markers:
        want = [str(m) for m in markers]
        hit = sum(1 for m in want if m in sol)
        return hit / len(want)

    # ② 数字/精确/包含（数学与短答案类）
    gt = str(ground_truth or "").strip()
    if gt and not gt.startswith("{"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
        if gt in nums or gt == sol.strip():
            return 1.0
        return 1.0 if gt and gt in sol else 0.0

    # ③ pass_rate 之类的软标签：直接用数据集给的通过率当奖励（离线蒸馏式口径）
    if "pass_rate" in spec:
        try:
            return float(spec["pass_rate"])
        except (TypeError, ValueError):
            return 0.0

    # ④ 未知规格：不猜，返回 0 并让上层看到（需要外部执行器/沙箱判分）
    return 0.0
