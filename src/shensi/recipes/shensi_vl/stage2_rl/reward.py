#!/usr/bin/env python3
"""Specialized RL 的奖励全家（论文 §2.5.2）。

verl 的 custom_reward_function 入口是 ``compute_score(data_source, solution_str,
ground_truth, extra_info)``。最终奖励 = format·w_f + quality·w_q + accuracy·w_a：
  Format RM（规则，0–1）  原语文法；TwG 查重复框（防 SFT 死循环）
  Quality RM（GRM，0/0.5/1） LLM 打分：冗余/思考-答案一致/自相矛盾/引用实体有意义/reward hacking
                            —— 需要外部判分端点（quality.url）；没配就中性 1.0 跳过（离线可复现）
  Accuracy RM（任务各一套） counting：R = α·exp(−β·|ŷ−y|/(|y|+1))，α=0.7 β=3
                            spatial/vqa：答案精确/数字匹配（+可选 GRM）
                            maze：因果探索进度 + 探索完整性 + 撞墙惩罚 + 终路径有效性 + 答案
                            trace：双向轨迹匹配 + 端点精度 + 连续性惩罚 + 答案
ground_truth 是数据池 jsonl 的 spec/ground_truth 字段整段（data_prep 时塞进 reward_model.ground_truth）。
"""

from __future__ import annotations

import json
import math
import re
import urllib.request

from shensi.recipes.shensi_vl.common import primitives as P

ALPHA, BETA = 0.7, 3.0  # 计数奖励系数（论文实测值）
QUALITY_URL = None  # 由 stage2_rl 启动时用环境变量 SHENSI_VL_QUALITY_URL 注入


# ---------------- 基础解析 ----------------


def _spec(ground_truth) -> dict:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str) and ground_truth.strip().startswith("{"):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}
    return {}


def _boxed(sol: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", sol or "")
    return m[-1].strip() if m else ""


def _last_number(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


# ---------------- Format RM ----------------


def format_reward(sol: str, primitive: str) -> float:
    return P.format_score(sol, primitive=primitive, no_duplicate_boxes=(primitive == "box"))


# ---------------- Quality RM（GRM，可选） ----------------

QUALITY_PROMPT = """你是"thinking with visual primitives"过程的质检员。按以下维度评审思考与最终回答：
1) 回答是否冗余 2) 思考与最终回答是否一致 3) 思考过程是否自相矛盾
4) 框引用的对象是否有意义 5) 是否有 reward hacking（硬造与自身预测相同的假 ground truth）。
只输出 0.0、0.5 或 1.0。"""


def quality_reward(sol: str, gt: dict, extra: dict) -> float:
    """LLM 生成式奖励（三档）。没配端点时中性 1.0（等于跳过这一路，训练口径里注明）。"""
    url = extra.get("quality_url") or QUALITY_URL
    if not url:
        return 1.0
    payload = json.dumps(
        {
            "prompt": QUALITY_PROMPT,
            "thinking": extra.get("thinking", ""),
            "response": sol,
            "question": extra.get("question", ""),
        }
    ).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}),
            timeout=20,
        ) as resp:
            text = resp.read().decode()
        m = re.search(r"0\.5|[01](?:\.0)?", text)
        return float(m.group(0)) if m else 0.5
    except Exception:  # noqa: BLE001  判分端点抖动不该炸训练
        return 0.5


# ---------------- Accuracy RM：counting ----------------


def counting_reward(sol: str, gt: dict) -> float:
    want = gt.get("count")
    if want is None:
        return 0.0
    got = _last_number(_boxed(sol) or sol)
    if got is None:
        return 0.0
    y_hat, y = abs(float(got)), abs(float(want))
    return ALPHA * math.exp(-BETA * abs(y_hat - y) / (abs(y) + 1.0))


# ---------------- Accuracy RM：spatial / general vqa ----------------


def vqa_reward(sol: str, gt: dict) -> float:
    want = str(gt.get("answer", "")).strip().lower()
    if not want:
        return 0.0
    got = (_boxed(sol) or sol).strip().lower()
    if got == want:
        return 1.0
    if want in ("true", "false"):
        hit = re.search(r"\btrue\b", got) if want == "true" else re.search(r"\bfalse\b", got)
        return 1.0 if hit else 0.0
    return 1.0 if want in got else 0.0


# ---------------- Accuracy RM：maze ----------------


def maze_reward(sol: str, gt: dict) -> float:
    """五项加权（论文）：因果探索进度 / 探索完整性 / 撞墙惩罚 / 终路径有效性 / 答案正确。
    gt 即任务池行的 spec（serialize_maze 产物：walls/centers/neighbors/start/goal/solvable）。"""
    maze = P.deserialize_maze(gt)
    start, goal = tuple(maze["start"]), tuple(maze["goal"])
    points = P.parse_points(sol)
    cells, invalid = [], 0
    for pt in points:
        cell = P.nearest_cell(pt, maze)
        if cell is None:
            invalid += 1
        else:
            cells.append(cell)
    reachable = _reachable(maze, start)
    solvable = bool(gt.get("solvable"))

    # ① 因果探索进度：第一个撞墙转移之后全部截断；分数 = 已合法探索的最短距离贴近度
    explore_frac, violated_at = 0.0, len(cells)
    for i in range(1, len(cells)):
        a, b = cells[i - 1], cells[i]
        if P.maze_wall_between(maze, a, b):
            violated_at = i
            break
    legal = cells[:violated_at]
    if legal:
        seen = set(legal)
        if solvable:
            sol_path = _bfs(maze, start, goal) or []
            pos = max((i for i, c in enumerate(sol_path) if c in seen), default=0)
            explore_frac = (pos + 1) / max(1, len(sol_path))
        else:
            explore_frac = 1.0  # 可解项对不可解迷宫恒 1
    progress = explore_frac if solvable else 1.0

    # ② 探索完整性（只对不可解迷宫）：因果合法的探索覆盖了多少可达格
    completeness = (len(set(legal) & reachable) / max(1, len(reachable))) if not solvable else 1.0

    # ③ 撞墙惩罚：全轨迹里非法转移占比（撞墙永不清零代价）
    n_moves = max(1, len(cells) - 1)
    wall_pen = 1.0 - (sum(
        1 for i in range(1, len(cells)) if P.maze_wall_between(maze, cells[i - 1], cells[i])
    ) + invalid) / n_moves

    # ④ 终路径有效性：声称可解时要给出从 start 到 goal 的连续合法路径
    if solvable:
        path_cells = [P.nearest_cell(pt, maze) for pt in _final_path_points(sol)]
        path_cells = [c for c in path_cells if c]
        ok_path = (
            bool(path_cells)
            and path_cells[0] == start
            and path_cells[-1] == goal
            and all(not P.maze_wall_between(maze, a, b) for a, b in zip(path_cells, path_cells[1:]))
        )
        path_score = 1.0 if ok_path else 0.0
    else:
        path_score = 1.0

    # ⑤ 答案正确
    ans = _boxed(sol).lower()
    answer = 1.0 if ans == ("true" if solvable else "false") else 0.0

    return 0.25 * progress + 0.2 * completeness + 0.2 * wall_pen + 0.15 * path_score + 0.2 * answer


def _reachable(maze: dict, start) -> set:
    from collections import deque

    seen, q = {start}, deque([start])
    while q:
        cur = q.popleft()
        for n in maze["neighbors"].get(cur, []):
            if n not in seen and not P.maze_wall_between(maze, cur, n):
                seen.add(n)
                q.append(n)
    return seen


def _bfs(maze: dict, start, goal):
    from collections import deque

    q, prev = deque([start]), {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for n in maze["neighbors"].get(cur, []):
            if n in prev or P.maze_wall_between(maze, cur, n):
                continue
            prev[n] = cur
            q.append(n)
    return None


def _final_path_points(sol: str) -> list[list[int]]:
    m = re.search(r"\*\*Final Path\*\*.*?(" + re.escape(P.POINT_OPEN) + r".*?" + re.escape(P.POINT_CLOSE) + r")", sol, re.S)
    return P.parse_points(m.group(1)) if m else P.parse_points(sol)[-30:]


# ---------------- Accuracy RM：path tracing ----------------


def trace_reward(sol: str, gt: dict) -> float:
    """四项加权（论文）：双向轨迹 + 端点 + 连续性 + 答案。坐标都是 0–999 系。"""
    poly = gt["target_polyline"]
    tol = float(gt.get("tolerance", 30.0))
    pts = P.parse_points(sol)
    # ① 双向轨迹匹配：预测→真值（罚偏航），真值→预测（罚漏段）
    fwd = sum(P.polyline_min_dist(p, poly) for p in pts) / len(pts) if pts else 999.0
    rev = sum(P.polyline_min_dist(p, pts) for p in poly) / len(poly) if pts else 999.0
    traj = sum(_decay(d, tol) for d in (fwd, rev)) / 2.0
    # ② 端点精度：起终点离真值端点的距离衰减（endpoints[标签] = [起点, 终点]）
    label = gt["labels"][gt["target"]] if gt.get("labels") else ""
    target_ends = (gt.get("endpoints") or {}).get(label)
    end_score = 0.0
    if target_ends and len(pts) >= 2:
        d0 = math.hypot(pts[0][0] - target_ends[0][0], pts[0][1] - target_ends[0][1])
        d1 = math.hypot(pts[-1][0] - target_ends[1][0], pts[-1][1] - target_ends[1][1])
        end_score = (_decay(d0, tol) + _decay(d1, tol)) / 2.0
    # ③ 连续性：轨迹末点跳到声称端点 = 惩罚（论文：不许只画一小段然后"跳"到答案）
    cont = 1.0
    if pts and len(pts) >= 2 and math.hypot(pts[-1][0] - poly[-1][0], pts[-1][1] - poly[-1][1]) > tol * 2:
        cont = 0.0
    # ④ 答案
    ans = _boxed(sol)
    answer = 1.0 if ans and ans == label else 0.0
    return 0.4 * traj + 0.25 * end_score + 0.15 * cont + 0.2 * answer


def _decay(dist: float, tol: float) -> float:
    return max(0.0, 1.0 - dist / tol)


# ---------------- verl 入口 ----------------

_PRIMITIVE_OF = {"count": "box", "spatial": "box", "vqa": "box", "maze": "point", "trace": "point"}


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    extra = extra_info or {}
    gt = _spec(ground_truth)
    task = str(data_source or "").split("/")[-1]
    if task not in _PRIMITIVE_OF:
        return 0.0
    prim = _PRIMITIVE_OF[task]
    acc = {
        "count": counting_reward,
        "spatial": vqa_reward,
        "vqa": vqa_reward,
        "maze": maze_reward,
        "trace": trace_reward,
    }[task](solution_str, gt)
    fmt = format_reward(solution_str, prim)
    qual = quality_reward(solution_str, gt, extra)
    w_f = float(extra.get("w_format", 0.2))
    w_q = float(extra.get("w_quality", 0.0 if not extra.get("quality_url") else 0.2))
    w_a = 1.0 - w_f - w_q
    return w_f * fmt + w_q * qual + w_a * acc


if __name__ == "__main__":  # 自检：迷宫/计数/轨迹各造一个正反例
    from shensi.recipes.shensi_vl.common import synth_maze as SM

    rng = __import__("random").Random(0)
    maze = SM.gen_maze(rng, "dfs", "rect", 6, 6)
    cells = list(maze["centers"])
    start, goal = cells[0], cells[-1]
    path = SM.bfs_path(maze, start, goal)
    spec = P.serialize_maze({**maze, "start": start, "goal": goal, "solvable": True})
    good = "**Final Path** " + P.render_point_primitive([maze["centers"][c] for c in path]) + " \\boxed{True}"
    bad = "**Final Path** " + P.render_point_primitive([maze["centers"][c] for c in reversed(path)]) + " \\boxed{False}"
    print("maze good/bad:", maze_reward(good, spec), maze_reward(bad, spec))
    print("count:", counting_reward("There are 8 dogs. \\boxed{8}", {"count": 8}),
          counting_reward("There are 20 dogs. \\boxed{20}", {"count": 8}))
