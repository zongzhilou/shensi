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


# verifier 优先：MRCR → string_match → 精确/数字 → pass_rate 软标签，兜底 0
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    extra = extra_info or {}
    spec = _parse(ground_truth) or _parse(extra.get("verifier"))
    sol = solution_str or ""
    gt = str(ground_truth or "").strip()

    # ⓪ MRCR 类长文检索：答案开头的编号序列必须与「按出现顺序」的编号完全一致——顺序错、
    #    缺一个都算 0（MRCR 的口径是整篇全对才算 retrieval 成功）。
    if str(data_source or "").startswith("mrcr"):
        want = [x for x in re.split(r"[、,，;；\s]+", gt) if x]
        got = re.findall(r"\d{6,}", sol)
        return 1.0 if want and got[: len(want)] == want else 0.0

    # ① string_match：所有 marker 都要出现（Lightning 的 citation_format 这类）
    markers = spec.get("expected_markers") or spec.get("expected")
    if markers:
        want = [str(m) for m in markers]
        hit = sum(1 for m in want if m in sol)
        return hit / len(want)

    # ② 数字/精确：ground truth 是数字时只看答案里**最后一个**数字（乱答里凑巧出现同一串数字不算）；
    #    非数字答案（人名/地名这类短答案）才走「包含」兜底。
    if gt and not gt.startswith("{"):
        if gt == sol.strip():
            return 1.0
        nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", gt):
            return 1.0 if nums and nums[-1] == gt else 0.0
        return 1.0 if gt in sol else 0.0

    # ③ pass_rate 之类的软标签：直接用数据集给的通过率当奖励（离线蒸馏式口径）
    if "pass_rate" in spec:
        try:
            return float(spec["pass_rate"])
        except (TypeError, ValueError):
            return 0.0

    # ④ 未知规格：不猜，返回 0 并让上层看到（需要外部执行器/沙箱判分）
    return 0.0
