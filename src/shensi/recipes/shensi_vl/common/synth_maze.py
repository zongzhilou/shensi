"""迷宫导航冷启动合成（论文 §2.4.3）。

对齐点：
  - 生成算法 DFS / Prim / Kruskal 三种；矩形网格、同心环×角扇区（环形）、蜂窝三种拓扑；
  - 可解迷宫给完整 DFS 探索轨迹（含死路与回溯）；不可解迷宫 = 先生成可解、再在路径中段
    （避开起终点附近）补墙断连——表面可解、实际要搜完才知道；
  - 难度 = 网格规模（easy → nightmare），分辨率随机、宽高比连续采样、视觉风格随机；
  - 思维链：每一步用 <point> 原语锚定格心（检查连通、前进、回退），结尾给 Final Path 与
    \\boxed{True/False}。

产物 jsonl 行：{"image": 落地路径, "question", "thinking", "response", "spec": 迷宫结构}。
图像由 ``--render`` 落盘（真实训练要图；冒烟可 --no-render 只出文本与 spec）。
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

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# 难度档：网格规模（论文：easy 链几个局部检查，nightmare 要上百个原语操作的持续组合）
DIFFICULTY = {"easy": (5, 8), "normal": (10, 14), "hard": (18, 24), "nightmare": (28, 40)}
TOPOLOGIES = ("rect", "ring", "hex")
ALGOS = ("dfs", "prim", "kruskal")


# ---------------- 拓扑：邻接 + 格心 ----------------


def cells_and_neighbors(topo: str, gw: int, gh: int):
    """返回 (cells, neighbors, centers, cell_dist)。环形拓扑：gh=环数、gw=每环扇区数。"""
    if topo == "rect":
        cells = [(c, r) for r in range(gh) for c in range(gw)]
        neigh = {
            cell: [
                (cell[0] + dc, cell[1] + dr)
                for dc, dr in DIRS
                if 0 <= cell[0] + dc < gw and 0 <= cell[1] + dr < gh
            ]
            for cell in cells
        }
        centers = {
            (c, r): [int((2 * c + 1) * P.COORD_MAX / (2 * gw)), int((2 * r + 1) * P.COORD_MAX / (2 * gh))]
            for c, r in cells
        }
        return cells, neigh, centers, P.COORD_MAX / gw
    if topo == "ring":
        cells, neigh, centers = [], {}, {}
        for ring in range(gh):  # gh = 环数（含中心圈）
            radius = (ring + 1) / gh * 400 + 60
            for sec in range(gw):
                ang = 2 * math.pi * sec / gw + (math.pi / gw if ring % 2 else 0.0)
                cells.append((sec, ring))
                centers[(sec, ring)] = [
                    int(500 + radius * math.cos(ang)),
                    int(500 + radius * math.sin(ang)),
                ]
        for c, r in cells:
            ns = [((c + 1) % gw, r), ((c - 1) % gw, r)]  # 同环相邻扇区
            if r + 1 < gh:
                ns.append((c, r + 1))
            if r - 1 >= 0:
                ns.append((c, r - 1))
            neigh[(c, r)] = [n for n in ns if (n[0], n[1]) in centers]
        return cells, neigh, centers, P.COORD_MAX / gw
    if topo == "hex":
        cells, neigh, centers = [], {}, {}
        for r in range(gh):
            for c in range(gw):
                cells.append((c, r))
                x = (2 * c + 1 + (r % 2)) * P.COORD_MAX / (2 * gw + 1)
                y = (2 * r + 1) * P.COORD_MAX / (2 * gh + 1)
                centers[(c, r)] = [int(x), int(y)]
        for c, r in cells:
            # odd-r 偏移布局：奇数行右移半格；斜向邻居按行奇偶镜像（保证邻接对称）
            ns = [(c - 1, r), (c + 1, r)]
            if r % 2 == 1:
                diag = [(c, r - 1), (c + 1, r - 1), (c, r + 1), (c + 1, r + 1)]
            else:
                diag = [(c - 1, r - 1), (c, r - 1), (c - 1, r + 1), (c, r + 1)]
            neigh[(c, r)] = [n for n in ns + diag if n in centers]
        return cells, neigh, centers, P.COORD_MAX / gw
    raise ValueError(topo)


# ---------------- 生成：三种算法产出同一套 maze 结构 ----------------


def gen_maze(rng: random.Random, algo: str, topo: str, gw: int, gh: int) -> dict:
    cells, neigh, centers, cell_dist = cells_and_neighbors(topo, gw, gh)
    maze = {"topo": topo, "grid_w": gw, "grid_h": gh, "cell_dist": cell_dist,
            "centers": centers, "walls": {}, "neighbors": neigh}
    # 全墙
    for cell in cells:
        maze["walls"][cell] = set(neigh[cell])
    start = rng.choice(cells)

    def carve(a, b):
        maze["walls"][a].discard(b)
        maze["walls"][b].discard(a)

    if algo == "dfs":
        stack, seen = [start], {start}
        while stack:
            cur = stack[-1]
            cand = [n for n in neigh[cur] if n not in seen]
            if not cand:
                stack.pop()
                continue
            nxt = rng.choice(cand)
            carve(cur, nxt)
            seen.add(nxt)
            stack.append(nxt)
    elif algo in ("prim", "kruskal"):
        if algo == "prim":
            seen, frontier = {start}, [(start, n) for n in neigh[start]]
            while frontier:
                i = rng.randrange(len(frontier))
                a, b = frontier.pop(i)
                if b in seen:
                    continue
                carve(a, b)
                seen.add(b)
                frontier += [(b, n) for n in neigh[b] if n not in seen]
        else:
            parent = {c: c for c in cells}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            edges = [(a, b) for a in cells for b in neigh[a] if a < b]
            rng.shuffle(edges)
            for a, b in edges:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                    carve(a, b)
    else:
        raise ValueError(algo)
    maze["centers"] = {k: list(v) for k, v in centers.items()}
    return maze


def bfs_path(maze: dict, start, goal) -> list | None:
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
        for n in maze["neighbors"][cur]:
            if n in prev or P.maze_wall_between(maze, cur, n):
                continue
            prev[n] = cur
            q.append(n)
    return None


def make_unsolvable(rng: random.Random, maze: dict, start, goal) -> None:
    """可解 → 在解路径中段补墙断连（避开起终点附近，表面可解、实际无路）。"""
    path = bfs_path(maze, start, goal)
    assert path and len(path) >= 5
    mid = path[2:-2]
    rng.shuffle(mid)
    for a in mid:
        for b in maze["neighbors"][a]:
            if b in path and not P.maze_wall_between(maze, a, b):
                maze["walls"][a].add(b)
                maze["walls"][b].add(a)
                if bfs_path(maze, start, goal) is None:
                    return
    raise RuntimeError("补墙失败：迷宫仍然可解")


# ---------------- 探索轨迹（思维链）与答案 ----------------


def _dir_name(a, b) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(1, 0): "right", (-1, 0): "left", (0, 1): "down", (0, -1): "up"}.get(
        (dx, dy), {(1, 1): "lower-right", (-1, 1): "lower-left", (1, -1): "upper-right", (-1, -1): "upper-left"}.get((dx, dy), "forward")
    )


def explore_narration(rng: random.Random, maze: dict, start, goal) -> tuple[str, list]:
    """DFS 全程解说：前进 / 死路 / 回溯都用 <point> 锚格心；返回 (思维链文本, 解路径或 [])。"""
    pts = maze["centers"]
    lines = [
        (
            f"I'll explore this maze with a trial-and-error strategy. Start: {P.render_point_primitive([pts[start]])}, "
            f"destination: {P.render_point_primitive([pts[goal]])}."
        ),
        "**Start Exploring**:",
    ]
    visited = {start}
    stack = [start]
    step = 0

    def emit_cur():
        cur = stack[-1]
        exits = [n for n in maze["neighbors"][cur] if n not in visited and not P.maze_wall_between(maze, cur, n)]
        return cur, exits

    while stack:
        cur, exits = emit_cur()
        if cur == goal:
            lines.append(
                f"**Step{step}**: Arriving {P.render_point_primitive([pts[cur]])} — this is the destination!"
            )
            break
        if not exits:
            stack.pop()
            step += 1
            if stack:
                lines.append(
                    f"**Step{step}**: All directions from {P.render_point_primitive([pts[cur]])} are explored/dead ends, "
                    f"backtrack to {P.render_point_primitive([pts[stack[-1]]])}."
                )
            continue
        nxt = rng.choice(exits)
        step += 1
        lines.append(
            f"**Step{step}**: Moving {_dir_name(cur, nxt)} from {P.render_point_primitive([pts[cur]])} "
            f"to {P.render_point_primitive([pts[nxt]])}."
        )
        visited.add(nxt)
        stack.append(nxt)
    final_path = bfs_path(maze, start, goal) or []
    solvable = bool(final_path)
    if solvable:
        coords = [pts[c] for c in final_path]
        lines.append("**Final Path**: " + P.render_point_primitive(coords))
    else:
        lines.append(
            "**Conclusion**: All reachable cells are explored, no path connects start to destination."
        )
    lines.append(f"The maze is {'solvable' if solvable else 'unsolvable'}: \\boxed{{{solvable}}}")
    return "\n".join(lines), final_path


# ---------------- 渲染（PIL）----------------


def render_maze(maze: dict, start, goal, path_cells: list, size: int, style: dict) -> Image.Image:
    img = Image.new("RGB", (size, size), style["bg"])
    if style.get("gradient"):  # 纵向渐变背景（论文的 gradient 风格之一）
        for y in range(size):
            t = y / size
            dr_row = (
                int(style["bg"][0] * (1 - t * 0.25)),
                int(style["bg"][1] * (1 - t * 0.15)),
                style["bg"][2],
            )
            ImageDraw.Draw(img).line([(0, y), (size, y)], fill=dr_row)
    dr = ImageDraw.Draw(img)
    sc = size / P.COORD_MAX

    def xy(cell):
        c = maze["centers"][cell]
        return (c[0] * sc, c[1] * sc)

    # 墙：每格把"还在的墙"画一遍（画半段避免重复加粗：只画 (b>a) 的方向）
    w = style["wall_width"]
    for a, walls in maze["walls"].items():
        for b in walls:
            if b > a:
                dr.line([xy(a), xy(b)], fill=style["wall"], width=w)
    for cell in path_cells or []:
        x, y = xy(cell)
        dr.ellipse([x - 3 * w / 2, y - 3 * w / 2, x + 3 * w / 2, y + 3 * w / 2], fill=style["trace"])
    x, y = xy(start)
    r = 4 * w
    dr.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=style["start"])
    x, y = xy(goal)
    dr.ellipse([x - r, y - r, x + r, y + r], fill=style["goal"])
    return img


def random_style(rng: random.Random) -> dict:
    bg = rng.choice([(250, 250, 245), (245, 245, 252), (252, 248, 240)])
    return {
        "bg": bg,
        "wall": rng.choice([(40, 60, 90), (30, 30, 30), (70, 90, 60)]),
        "wall_width": rng.choice([3, 4, 6]),  # extra-thick walls 也是风格之一
        "trace": rng.choice([(230, 160, 60), (120, 180, 90)]),
        "start": rng.choice([(90, 170, 90), (60, 140, 200)]),
        "goal": (200, 70, 70),
        "gradient": rng.random() < 0.3,
        "rotation": rng.uniform(-8, 8),
    }


# ---------------- 批量产出 ----------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="迷宫导航冷启动合成（§2.4.3）")
    ap.add_argument("--out", default=None, help="产物目录，默认 $SHENSI_FS/shensi/data/shensi_vl/stage1_sft/maze")
    ap.add_argument("--n", type=int, default=1000, help="样本数（正式档 460_000，分难度均匀）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-render", action="store_true", help="不落图（只出文本+spec，冒烟用）")
    ap.add_argument("--limit-size", type=int, default=768, help="最长边像素")
    args = ap.parse_args(argv)

    paths = vl_paths.env_paths()
    out = Path(args.out or paths["data"] / "stage1_sft/maze")
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        diff = rng.choice(list(DIFFICULTY))
        lo, hi = DIFFICULTY[diff]
        topo = rng.choice(TOPOLOGIES)
        gw = rng.randint(lo, hi)
        gh = rng.randint(lo, hi) if topo != "ring" else rng.randint(3, 8)
        maze = gen_maze(rng, rng.choice(ALGOS), topo, gw, gh)
        cells = list(maze["centers"])
        start, goal = cells[0], cells[-1]
        solvable = rng.random() < 0.7  # 论文：可解为主、掺不可解
        if not solvable:
            make_unsolvable(rng, maze, start, goal)
        thinking, path_cells = explore_narration(rng, maze, start, goal)
        q = (
            "Is there a feasible way to get from the green marker to the red circle? "
            "Explain the path if any. Output \\boxed{True} or \\boxed{False}."
        )
        response = (
            f"The maze is solvable. The verified path is: "
            f"{P.render_point_primitive([maze['centers'][c] for c in path_cells])}\n\\boxed{{True}}"
            if solvable
            else "No path exists between the start and the destination.\n\\boxed{False}"
        )
        style = random_style(rng)
        size = rng.randint(384, args.limit_size)
        row = {
            "task": "maze",
            "difficulty": diff,
            "image": None,
            "question": q,
            "thinking": thinking,
            "response": response,
            "spec": P.serialize_maze({**maze, "start": start, "goal": goal, "solvable": solvable}),
        }
        if not args.no_render:
            fp = img_dir / f"maze_{i:07d}.png"
            render_maze(maze, start, goal, path_cells, size, style).save(fp)
            row["image"] = str(fp)
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"[maze] {i + 1}/{args.n}")
    with open(out / "maze.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[maze] {len(rows)} 条 → {out / 'maze.jsonl'}（图像在 {img_dir}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
