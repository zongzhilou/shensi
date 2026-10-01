#!/usr/bin/env python3
# stage5 语料准备：拿交互轨迹（自家 Sim RL 落盘的 + post-training 的 agentic 集）产出三种形态——
# CPT 的纯文本、SFT 的「下一状态」messages、RL 的（历史+动作 → 真观测）行。

import argparse
import importlib.util
import json
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common, rl
from shensi.recipes.shensi.stage2_rl.stage4_world_model import wm_common

_HERE = Path(__file__).resolve().parent
# recipes/shensi/：stage2_rl/stage4_world_model → stage2_rl(0) → shensi(1)
_RECIPES = _HERE.parents[1]

STAGE = "stage2_world_model"


def _stage1_sft_main():
    """按文件路径加载 stage1_sft/data_prep.py：SFT 的渲染/packed 口径直接复用，别再写一份。"""
    f = _RECIPES / "stage1_sft/data_prep.py"
    spec = importlib.util.spec_from_file_location("shensi_stage1_sft_data_prep", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(
    files: list[Path],
    limit: int | None,
    min_obs_chars: int,
    max_system_chars: int | None = None,
) -> list[tuple[str, str, list]]:
    """→ [(域, system, [(动作, 观测)...])]。"""
    out = []
    for row in rl.iter_rows(files, limit):
        turns = wm_common.to_turns(row)
        if not turns:
            continue
        turns = [(a, o) for a, o in turns if len(o) >= min_obs_chars]
        if not turns:
            continue
        domain = wm_common.domain_of(row)
        system = wm_common.clip_turn(wm_common.system_of(row, domain), max_system_chars)
        out.append((domain, system, turns))
    return out


def dataset_files(root: Path, d: dict, spec_dir: Path) -> list[Path]:
    """Blend 条目两种写法：`path`（自带的样本文件，相对配比 json）或 `name`（在语料根下找目录）。"""
    if d.get("path"):
        p = Path(d["path"])
        return [(spec_dir / p).resolve()] if not p.is_absolute() else [p]
    return rl.files_of(root, d)


def sources(args, paths, spec, spec_dir: Path) -> list[Path]:
    files: list[Path] = []
    traj_dir = Path(args.traj_dir or paths["data"] / "world_model_traj")
    if traj_dir.is_dir():
        files += sorted(f for f in traj_dir.glob("**/*") if f.suffix in (".json", ".jsonl"))
    root = Path(args.root or paths["post"])
    for d in spec["datasets"]:
        files += dataset_files(root, d, spec_dir)
    return files


def write_cpt(trajs, out: Path, tokenizer: str) -> None:
    src = out / "cpt_src" / "world_model_traj"
    src.mkdir(parents=True, exist_ok=True)
    jsonl = src / "world_model_traj.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for domain, system, turns in trajs:
            doc = wm_common.render_cpt(domain, system, turns)
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    # 走 stage0_pretrain 的 prepare：切片 + tokenize + blend.json 都在那边，这里只喂语料
    spec = {
        "datasets": [{"name": "world_model_traj", "config": "", "weight": 1.0, "text_key": "text"}]
    }
    common.prepare(
        spec, root=out / "cpt_src", out=out / "cpt_bins", tokenizer=tokenizer, limit=None, workers=8
    )
    print(f"[world_model] CPT：{len(trajs)} 条轨迹 → {out}/cpt_bins/blend.json")


def write_sft(
    trajs,
    out: Path,
    tokenizer: str,
    max_history: int,
    val_ratio: float,
    max_turn_chars: int | None = None,
) -> None:
    src = out / "sft_src" / "world_model_next_state"
    src.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src / "next_state.jsonl", "w", encoding="utf-8") as fh:
        for domain, system, turns in trajs:
            for messages in wm_common.sft_messages(
                domain, system, turns, max_history, max_turn_chars
            ):
                fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                n += 1
    # 渲染成 token + loss_mask、packed、parquet 都交给 stage1_sft 的 data_prep（口径同一份）
    spec = {"datasets": [{"name": "world_model_next_state", "config": "", "weight": 1.0}]}
    (out / "sft_src" / "blend.json").write_text(
        json.dumps(spec, ensure_ascii=False), encoding="utf-8"
    )
    sft = _stage1_sft_main()
    rc = sft.main(
        [
            "--prepare",
            "--root",
            str(out / "sft_src"),
            "--out",
            str(out / "sft"),
            "--blend",
            str(out / "sft_src" / "blend.json"),
            "--tokenizer",
            tokenizer,
            "--val-ratio",
            str(val_ratio),
        ]
    )
    if rc:
        raise SystemExit(f"[world_model] stage1_sft 的 data_prep 返回 {rc}")
    print(f"[world_model] SFT：{len(trajs)} 条轨迹 → {n} 条「下一状态」样本")


def write_rl(
    trajs, out: Path, max_history: int, val_ratio: float, max_turn_chars: int | None = None
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for domain, system, turns in trajs:
        for i in range(len(turns)):
            rows.append(
                wm_common.rl_row(
                    domain, system, turns[: i + 1], len(rows), max_history, max_turn_chars
                )
            )
    if len(rows) < 2:
        raise SystemExit("[world_model] RL 样本不足 2 条，拆不出 val（增加轨迹或换 --limit）")
    n_val = max(1, int(len(rows) * val_ratio))
    table = pa.Table.from_pylist(rows)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(n_val, len(rows) - n_val), out / "train.parquet")
    pq.write_table(table.slice(0, n_val), out / "val.parquet")
    print(f"[world_model] RL：{len(rows) - n_val} 训练 + {n_val} 验证 → {out}/train.parquet")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi stage2_rl/stage4_world_model 语料准备")
    ap.add_argument("--step", default="all", choices=("cpt", "sft", "rl", "all"))
    ap.add_argument("--discover", action="store_true")
    ap.add_argument(
        "--blend", default=None, help="换一份配比 json（默认 config/data_prep/data_blend_raw.json）"
    )
    ap.add_argument(
        "--traj-dir",
        default=None,
        help="Sim RL 落盘的轨迹目录，默认 $SHENSI_FS/shensi/data/world_model_traj",
    )
    ap.add_argument(
        "--root", default=None, help="轨迹语料根，默认 $SHENSI_FS/datasets/llm/post-training"
    )
    ap.add_argument("--out", default=None, help=f"产物目录，默认 $SHENSI_FS/shensi/data/{STAGE}")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-history", type=int, default=4, help="SFT/RL 里带几轮历史")
    ap.add_argument("--min-obs-chars", type=int, default=8, help="丢掉观测短于这个字符数的轮次")
    ap.add_argument(
        "--max-turn-chars",
        type=int,
        default=None,
        help="每轮动作/观测掐到多少字符（极小档用：真轨迹的 prompt 能到 2 万 token，"
        "单卡极小档放不下）；不给就不截",
    )
    ap.add_argument(
        "--max-system-chars",
        type=int,
        default=None,
        help="每个域的 system prompt 掐到多少字符（那几份是两万多字、约 9k token，极小档放不下）；"
        "不给就不截",
    )
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args(argv)

    paths = common.env_paths()
    out = Path(args.out or paths["data"] / STAGE)
    tokenizer = args.tokenizer or paths["tokenizer"]
    spec_path = Path(args.blend) if args.blend else _HERE / "config/data_prep/data_blend_raw.json"
    if not spec_path.is_file():
        raise SystemExit(f"[world_model] 没有这份配比：{spec_path}（相对路径按当前目录解析）")
    spec = common.load_blend_spec(spec_path)
    root = Path(args.root or paths["post"])
    files = sources(args, paths, spec, spec_path.resolve().parent)
    if args.discover:
        print(f"[world_model] 轨迹目录：{args.traj_dir or paths['data'] / 'world_model_traj'}")
        print(f"[world_model] 语料根：{root}")
        for d in spec["datasets"]:
            found = dataset_files(root, d, spec_path.resolve().parent)
            print(
                f"  {'✅' if found else '❌'} {d['name']:<44} 文件 {len(found):<4} weight={d.get('weight')}"
            )
        print(f"  {'✅' if files else '❌'} 轨迹文件合计 {len(files)}（含 Sim RL 落盘）")
        return 0
    if not files:
        raise SystemExit(
            "[world_model] 一条轨迹都没找到：先跑 Sim RL（工具里开 dump_dir）或给 --blend/--traj-dir"
        )
    trajs = collect(files, args.limit, args.min_obs_chars, args.max_system_chars)
    if not trajs:
        raise SystemExit("[world_model] 轨迹都解析不出 (动作, 观测)：看看 --discover 打出来的字段")
    by_domain = {d: sum(1 for x in trajs if x[0] == d) for d in wm_common.DOMAINS}
    print(f"[world_model] 轨迹 {len(trajs)} 条，按域：{by_domain}")
    if args.step in ("cpt", "all"):
        write_cpt(trajs, out, tokenizer)
    if args.step in ("sft", "all"):
        write_sft(trajs, out, tokenizer, args.max_history, args.val_ratio, args.max_turn_chars)
    if args.step in ("rl", "all"):
        write_rl(trajs, out, args.max_history, args.val_ratio, args.max_turn_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
