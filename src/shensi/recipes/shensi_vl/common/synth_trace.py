"""路径追踪冷启动合成（论文 §2.4.4）。

对齐点：
  - 多条 Bézier 曲线，每条连一个带标签的起点到终点；交叉处要靠"局部几何连续性"判断分支；
  - 保证端点不被无关线压住/穿过（违反就重新生成，保证 primitive 真的被测到）；
  - uniform 模式：所有线同色同宽，只靠曲率连续性（防颜色捷径）；
  - 难度 = 线条数 × 曲率幅度；分辨率/配色/线型/端点标记/背景随机；
  - 思维链 = 沿目标曲线采样的坐标序列（曲率大的地方点更密——人眼在复杂区域放慢），
    定位起点 → 途经点 → 终点；答案端点标签进 \\boxed{}。

产物 jsonl 行：{"image", "question", "thinking", "response", "spec": {target 线的真值折线/端点}}。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from shensi.recipes.shensi_vl.common import paths as vl_paths
from shensi.recipes.shensi_vl.common import primitives as P

PALETTES = [
    (230, 120, 120), (120, 160, 230), (120, 200, 140), (220, 180, 90),
    (180, 120, 220), (110, 200, 200), (200, 140, 90),
]
ICONS = ["crown", "octopus", "fish", "star", "moon", "leaf", "key", "bell", "duck", "gem"]


def bezier(pts: list[list[float]], t: float) -> tuple[float, float]:
    """De Casteljau 求值。"""
    tmp = [list(p) for p in pts]
    while len(tmp) > 1:
        tmp = [
            [tmp[i][0] + (tmp[i + 1][0] - tmp[i][0]) * t, tmp[i][1] + (tmp[i + 1][1] - tmp[i][1]) * t]
            for i in range(len(tmp) - 1)
        ]
    return tmp[0][0], tmp[0][1]


def sample_curve(ctrl: list[list[float]], n: int) -> list[list[float]]:
    return [list(bezier(ctrl, i / (n - 1))) for i in range(n)]


def curvature_density(ctrl: list[list[float]]) -> list[list[float]]:
    """沿曲线自适应采样：曲率大的地方点更密（论文：复杂区域放慢、点更细）。"""
    dense = sample_curve(ctrl, 200)
    # 以相邻点的转角作为密度权重
    weights = [1.0]
    for i in range(1, len(dense) - 1):
        a = math.atan2(dense[i][1] - dense[i - 1][1], dense[i][0] - dense[i - 1][0])
        b = math.atan2(dense[i + 1][1] - dense[i][1], dense[i + 1][0] - dense[i][0])
        weights.append(1.0 + 6.0 * abs((b - a + math.pi) % (2 * math.pi) - math.pi))
    weights.append(1.0)
    total = sum(weights)
    picks, acc = [], 0.0
    for i, w in enumerate(weights):
        acc += w / total
        picks.append((i, acc))
    out, target, n_out = [], 0.0, 60
    for i, acc in picks:
        if acc >= target and len(out) < n_out:
            out.append([round(v) for v in dense[i]])
            target += 1.0 / n_out
    if not out or out[-1] != [round(v) for v in dense[-1]]:
        out.append([round(v) for v in dense[-1]])
    return out


def random_curve(rng: random.Random, start: list[float] | None = None,
                 end: list[float] | None = None) -> list[list[float]]:
    """一条 Bézier 控制点：中段沿起终点弦的走廊做**法向偏移**（幅度=曲率）。
    直接在画幅里撒点会让曲线糊满画布、别的端点必然被压；走廊式生成保交叉、不糊 canvas。"""
    amp = rng.uniform(120, 420)  # 曲率幅度随难度上升
    start = list(start) if start else [rng.uniform(30, 960), rng.uniform(30, 960)]
    end = list(end) if end else [rng.uniform(30, 960), rng.uniform(30, 960)]
    n_mid = rng.randint(2, 4)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length  # 弦的法向
    ctrl = [list(start)]
    for i in range(1, n_mid + 1):
        t = i / (n_mid + 1)
        off = rng.uniform(-amp, amp) * 0.5
        ctrl.append(
            [
                min(999.0, max(0.0, start[0] + dx * t + px * off + rng.uniform(-40, 40))),
                min(999.0, max(0.0, start[1] + dy * t + py * off + rng.uniform(-40, 40))),
            ]
        )
    ctrl.append(list(end))
    return ctrl


def place_endpoints(rng: random.Random, n: int, sep: float = 90.0) -> list[list[float]]:
    """泊松式布 2n 个彼此分开的端点（sep 挤了就衰减，保证能布满）。"""
    pts: list[list[float]] = []
    s = sep
    while len(pts) < 2 * n:
        p = [rng.uniform(30, 960), rng.uniform(30, 960)]
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= s for q in pts):
            pts.append(p)
        if rng.random() < 0.02:  # 偶尔放松（防 sep 过大死循环）
            s = max(45.0, s * 0.95)
    return pts


def build_curves(rng: random.Random, n_lines: int, tol: float = 30.0) -> list:
    """先布好分开的端点 → 随机配对 → 逐条走廊生成；
    某条压到别的端点就单独重掷它（单曲线变量，收敛快），40 次后取违规最小者。"""
    pts = place_endpoints(rng, n_lines)
    rng.shuffle(pts)
    starts, ends = pts[:n_lines], pts[n_lines:]
    curves: list = []
    for i in range(n_lines):
        best, best_bad = None, float("inf")
        for _ in range(40):
            cand = random_curve(rng, starts[i], ends[i])
            others = [e for j, e in enumerate(zip(starts, ends)) if j != i]
            flat = [p for pair in others for p in pair]
            poly = sample_curve(cand, 100)
            bad = min((math.hypot(p[0] - e[0], p[1] - e[1]) for p in poly for e in flat), default=999.0)
            if bad >= tol:
                best = cand
                break
            if bad < best_bad:
                best, best_bad = cand, bad
        curves.append(best)
    return curves




def render_trace(curves: list, size: int, style: dict) -> Image.Image:
    img = Image.new("RGB", (size, int(size * style["aspect"])), style["bg"])
    dr = ImageDraw.Draw(img)
    sc_x, sc_y = size / P.COORD_MAX, img.height / P.COORD_MAX

    def xy(p):
        return (p[0] * sc_x, p[1] * sc_y)

    for i, ctrl in enumerate(curves):
        poly = sample_curve(ctrl, 160)
        color = style["line_color"] if style["uniform"] else style["colors"][i]
        width = style["width"]
        dr.line([xy(p) for p in poly], fill=color, width=width)
        # 端点标记：起点圈 + 终点圈（标签写在旁边）
        for k, p in enumerate((ctrl[0], ctrl[-1])):
            x, y = xy(p)
            r = style["marker_r"]
            if k == 0:
                dr.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
            else:
                dr.ellipse([x - r, y - r, x + r, y + r], fill=color)
            dr.text((x + r + 2, y - 6), style["labels"][i], fill=(20, 20, 20))
    return img


def synth_one(rng: random.Random, difficulty: str) -> dict:
    n_lines = {"easy": 3, "normal": 5, "hard": 8, "nightmare": 12}[difficulty] + rng.randint(0, 2)
    curves = build_curves(rng, n_lines)
    labels = rng.sample(ICONS, min(n_lines, len(ICONS))) + [
        f"line{i}" for i in range(max(0, n_lines - len(ICONS)))
    ]
    target = rng.randrange(n_lines)
    uniform = rng.random() < 0.5  # uniform 模式：同色同宽，只靠曲率连续性
    style = {
        "uniform": uniform,
        "colors": [rng.choice(PALETTES) for _ in range(n_lines)],
        "line_color": rng.choice(PALETTES),
        "width": rng.choice([3, 4, 5]),
        "marker_r": rng.choice([8, 10, 12]),
        "labels": labels,
        "bg": rng.choice([(255, 253, 250), (250, 250, 255), (250, 253, 250)]),
        "aspect": rng.uniform(0.7, 1.3),
    }
    # 思维链：定位起点 → 沿线采样（密度自适应）→ 终点
    pts = curvature_density(curves[target])
    thinking = (
        f"I find the starting point you mentioned, it's located here: "
        f"{P.render_point_primitive([pts[0]])}.\nFollowing this line, the visual path I observe is:\n"
        + P.render_point_primitive(pts)
        + f"\nFollowing this path, it connects to: {P.render_point_primitive([pts[-1]])}."
    )
    response = f"The line connects to the {labels[target]} icon. \\boxed{{{labels[target]}}}"
    question = (
        f"Where does the {labels[target]} icon connect to? Show the path and "
        f"output \\boxed{{endpoint}}."
    )
    return {
        "task": "trace",
        "difficulty": difficulty,
        "question": question,
        "thinking": thinking,
        "response": response,
        "spec": {
            "target": target,
            "labels": labels,
            "target_polyline": [[round(v) for v in p] for p in sample_curve(curves[target], 120)],
            "endpoints": {
                labels[i]: [[round(v) for v in c[0]], [round(v) for v in c[-1]]]
                for i, c in enumerate(curves)
            },
            "uniform": uniform,
        },
        "curves": curves,
        "style": style,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="路径追踪冷启动合成（§2.4.4）")
    ap.add_argument("--out", default=None, help="产物目录，默认 <data>/stage1_sft/trace")
    ap.add_argument("--n", type=int, default=1000, help="样本数（正式档 125_000）")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--limit-size", type=int, default=768)
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    out = Path(args.out or paths["data"] / "stage1_sft/trace")
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        diff = rng.choice(["easy", "normal", "hard", "nightmare"])
        row = synth_one(rng, diff)
        if not args.no_render:
            size = rng.randint(384, args.limit_size)
            fp = img_dir / f"trace_{i:07d}.png"
            render_trace(row.pop("curves"), size, row.pop("style")).save(fp)
            row["image"] = str(fp)
        else:
            row.pop("curves")
            row.pop("style")
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"[trace] {i + 1}/{args.n}")
    with open(out / "trace.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[trace] {len(rows)} 条 → {out / 'trace.jsonl'}（图像在 {img_dir}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
