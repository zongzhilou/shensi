"""视觉原语（visual primitives）：格式、坐标与校验 —— 对齐论文 §2.3.4 / §2.4。

两种原语（论文把空间标记升格成"最小思考单元"）：
  box   ``<|ref|>TARGET<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>``   引用名 + 框（多实例从左到右）
  point ``<|point|>[[x1,y1],[x2,y2],...]<|/point|>``                 不带引用名（抽象引用/轨迹）
坐标一律归一到 0–999 的离散整数（论文口径），与像素分辨率解耦。
"""

from __future__ import annotations

import json
import math
import re

COORD_MAX = 999
BOX_OPEN, BOX_CLOSE = "<|box|>", "<|/box|>"
REF_OPEN, REF_CLOSE = "<|ref|>", "<|/ref|>"
POINT_OPEN, POINT_CLOSE = "<|point|>", "<|/point|>"

# 数据侧（jsonl 里）统一用的纯文本占位：写数据时避免与 tokenizer 特殊 token 混淆，
# 落到训练文本前由 render 换成真正的特殊 token。
P_BOX_OPEN, P_BOX_CLOSE = "<box>", "</box>"
P_REF_OPEN, P_REF_CLOSE = "<ref>", "</ref>"
P_POINT_OPEN, P_POINT_CLOSE = "<point>", "</point>"

_BOX_RE = re.compile(
    re.escape(P_REF_OPEN) + r"(.*?)" + re.escape(P_REF_CLOSE) + re.escape(P_BOX_OPEN)
    + r"(\[.*?\])" + re.escape(P_BOX_CLOSE),
    re.S,
)
_POINT_RE = re.compile(re.escape(P_POINT_OPEN) + r"(\[.*?\])" + re.escape(P_POINT_CLOSE), re.S)


# ---------------- 归一 / 反归一 ----------------


def norm_px(v: float, size: int) -> int:
    """像素 → 0–999 整数（论文：normalized to discrete integers ranging from 0 to 999）。"""
    if size <= 0:
        return 0
    return max(0, min(COORD_MAX, int(round(v * COORD_MAX / size))))


def denorm(v: int, size: int) -> int:
    """0–999 → 像素（推理渲染框/点时用）。"""
    return int(round(v * size / COORD_MAX))


def boxes_from_px(raw_boxes: list[list[float]], w: int, h: int) -> list[list[int]]:
    """检测标注（xyxy 像素，按 x1 排序 → 多实例从左到右）→ 归一化框。"""
    out = []
    for b in raw_boxes:
        x1, y1, x2, y2 = b[:4]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            [
                norm_px(x1, w),
                norm_px(y1, h),
                norm_px(x2, w),
                norm_px(y2, h),
            ]
        )
    return sorted(out, key=lambda b: (b[0], b[1]))


def points_from_px(raw_points: list[list[float]], w: int, h: int) -> list[list[int]]:
    return [[norm_px(x, w), norm_px(y, h)] for x, y in raw_points]


# ---------------- 渲染（数据侧占位 → 训练侧原语文本）----------------


def render_box_primitive(target: str, boxes: list[list[int] | list[float]]) -> str:
    coords = ",".join("[" + ",".join(str(int(v)) for v in b) + "]" for b in boxes)
    return f"{P_REF_OPEN}{target}{P_REF_CLOSE}{P_BOX_OPEN}[{coords}]{P_BOX_CLOSE}"


def render_point_primitive(points: list[list[int] | list[float]]) -> str:
    coords = ",".join("[" + ",".join(str(int(v)) for v in p) + "]" for p in points)
    return f"{P_POINT_OPEN}[{coords}]{P_POINT_CLOSE}"


def to_train_text(text: str) -> str:
    """数据侧占位（<box>…）→ 训练侧特殊 token（<|box|>…）。"""
    return (
        text.replace(P_BOX_OPEN, BOX_OPEN)
        .replace(P_BOX_CLOSE, BOX_CLOSE)
        .replace(P_REF_OPEN, REF_OPEN)
        .replace(P_REF_CLOSE, REF_CLOSE)
        .replace(P_POINT_OPEN, POINT_OPEN)
        .replace(P_POINT_CLOSE, POINT_CLOSE)
    )


# ---------------- 解析与校验（RL 的 Format RM / 冷启动数据闸门共用）----------------


def parse_boxes(text: str) -> list[list[int]]:
    """从模型输出（训练侧原语文本）里解析全部框。"""
    out = []
    for m in _TRAIN_BOX_RE.finditer(text or ""):
        try:
            arr = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        for b in arr:
            if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                out.append([int(v) for v in b])
    return out


_TRAIN_BOX_RE = re.compile(
    re.escape(REF_OPEN) + r"(.*?)" + re.escape(REF_CLOSE) + re.escape(BOX_OPEN)
    + r"(\[.*?\])" + re.escape(BOX_CLOSE),
    re.S,
)
_TRAIN_POINT_RE = re.compile(re.escape(POINT_OPEN) + r"(\[.*?\])" + re.escape(POINT_CLOSE), re.S)


def parse_points(text: str) -> list[list[int]]:
    out = []
    for m in _TRAIN_POINT_RE.finditer(text or ""):
        try:
            arr = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for p in arr:
            if isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p):
                out.append([int(v) for v in p])
    return out


def format_score(text: str, *, primitive: str, no_duplicate_boxes: bool = True) -> float:
    """Format RM（论文 §2.5.2，0–1）：原语文法是否正确；TwG 额外查重复框（防 SFT 死循环）。"""
    if primitive == "box":
        opens = len(re.findall(re.escape(REF_OPEN), text))
        closes = len(re.findall(re.escape(REF_CLOSE), text))
        boxes = parse_boxes(text)
        pairs = len(re.findall(re.escape(BOX_OPEN) + r".*?" + re.escape(BOX_CLOSE), text, re.S))
        if opens == 0 or pairs == 0 or opens != closes or pairs != opens:
            return 0.0
        score = 1.0
        if no_duplicate_boxes:
            uniq = {tuple(b) for b in boxes}
            if len(uniq) < len(boxes):  # 有重复框 → 按比例扣
                score *= len(uniq) / len(boxes)
        return score
    if primitive == "point":
        pts = parse_points(text)
        opens = len(re.findall(re.escape(POINT_OPEN), text))
        closes = len(re.findall(re.escape(POINT_CLOSE), text))
        if opens == 0 or opens != closes:
            return 0.0
        return 1.0 if pts else 0.0
    raise ValueError(f"unknown primitive: {primitive}")


# ---------------- 迷宫 / 路径追踪的几何校验（Accuracy RM 的底座）----------------


def nearest_cell(pt: list[int], maze: dict) -> tuple[int, int] | None:
    """0–999 坐标 → 最近的迷宫格。拓扑（矩形/环形/蜂窝）各自的格心都存在 centers 里；
    距离超过半格距视为没落在格上（RM 口径：非法转移）。"""
    best, bd = None, float("inf")
    for cell, ctr in maze["centers"].items():
        d = (pt[0] - ctr[0]) ** 2 + (pt[1] - ctr[1]) ** 2
        if d < bd:
            best, bd = cell, d
    if best is None or bd > (maze.get("cell_dist", 40) ** 2):
        return None
    return best


def maze_wall_between(maze: dict, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """a、b 之间是否有墙（walls 记录"该格哪些邻格方向的墙还在"，单侧记即算有墙）。
    邻接性查 neighbors 表（拓扑无关：矩形/环形/蜂窝通用）；不在表里 = 非法转移。"""
    neigh = maze.get("neighbors") or {}
    if b not in neigh.get(a, ()):  # neighbors 缺失时退回矩形曼哈顿判据
        if neigh:
            return True
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return True
    walls = maze.get("walls", {})
    return b in walls.get(a, set()) or a in walls.get(b, set())


def polyline_min_dist(pt: list[float], poly: list[list[float]]) -> float:
    """点到折线的最小距离（0–999 坐标系）。"""
    best = float("inf")
    px, py = pt
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
        ex, ey = x1 + t * dx - px, y1 + t * dy - py
        best = min(best, math.hypot(ex, ey))
    if len(poly) == 1:
        best = min(best, math.hypot(poly[0][0] - px, poly[0][1] - py))
    return best


def open_wall(maze: dict, a: tuple[int, int], b: tuple[int, int]) -> None:
    """打通 a/b 之间的墙（合成器用：生成时先全墙，走一步挖一步）。"""
    (c1, r1), (c2, r2) = a, b
    maze["walls"].setdefault((c1, r1), set()).discard((c2, r2))
    maze["walls"].setdefault((c2, r2), set()).discard((c1, r1))


def init_maze(grid_w: int, grid_h: int) -> dict:
    """全墙迷宫：walls[c,r] = 四邻集合（界内）。"""
    walls = {}
    for c in range(grid_w):
        for r in range(grid_h):
            walls[(c, r)] = {
                (c + dc, r + dr)
                for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= c + dc < grid_w and 0 <= r + dr < grid_h
            }
    return {"grid_w": grid_w, "grid_h": grid_h, "walls": walls}


def serialize_maze(maze: dict) -> dict:
    """Tuple 键（walls/centers）→ json 可存结构（进 ground_truth / RL spec）。"""
    out = {k: v for k, v in maze.items() if k not in ("walls", "centers", "neighbors")}
    out["walls"] = [[c, r, sorted([list(d) for d in dirs])] for (c, r), dirs in maze["walls"].items()]
    out["centers"] = [[c, r, list(ctr)] for (c, r), ctr in maze["centers"].items()]
    out["neighbors"] = [[c, r, [list(n) for n in ns]] for (c, r), ns in maze.get("neighbors", {}).items()]
    for key in ("start", "goal"):
        if key in out and isinstance(out[key], tuple):
            out[key] = list(out[key])
    return out


def deserialize_maze(spec: dict) -> dict:
    maze = {k: v for k, v in spec.items() if k not in ("walls", "centers", "neighbors")}
    maze["walls"] = {
        (int(c), int(r)): {tuple(d) for d in dirs} for c, r, dirs in spec["walls"]
    }
    maze["centers"] = {(int(c), int(r)): [int(x), int(y)] for c, r, (x, y) in spec["centers"]}
    maze["neighbors"] = {
        (int(c), int(r)): [tuple(n) for n in ns] for c, r, ns in spec.get("neighbors", [])
    }
    for key in ("start", "goal"):
        if key in maze and isinstance(maze[key], list):
            maze[key] = tuple(maze[key])
    return maze
