"""不变量：src/*_ext 相对上游（子模块 HEAD 里的内容）默认只允许新增；少数"登记回填"例外。

比的是上游仓库 HEAD 里的文件，不是工作区——回填会动到上游同路径的文件，拿工作区当真值会误报。
登记回填必须写清原因：教训是"旧快照覆盖"会把上游代码冻在旧版本上，lint 抓不到。

用法：`python3 tools/ext_invariant.py [shensi 检出根]`（默认取环境变量 SHENSI_ROOT，否则当前仓）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

R = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("SHENSI_ROOT", Path(__file__).resolve().parents[1])
)
SRC = R / "src"

BACKFILL = {
    "megatron_ext/core/models/backends.py": "Bridge 要 get_backend（mcore main 的 API）；FL fork 的 backends.py 里只有三个 provider，缺这个函数",
    "megatron_ext/core/_rank_utils.py": "FL fork 缺 set_default_log_ranks（Bridge@a393057 / verl 在导入期就 import 它）；fork 那一版 + main 的默认日志 rank 支持",
}

MAP = {
    "megatron_ext": R / "3rdparty/common/Megatron-LM-FL/megatron",
    "flagscale_ext": R / "3rdparty/common/FlagScale/flagscale",
    "megatron_ext/core": R / "3rdparty/common/Megatron-LM-FL/megatron/core",
    "megatron_ext/training": R / "3rdparty/common/Megatron-LM-FL/megatron/training",
    "megatron_ext/bridge": R / "3rdparty/common/Megatron-Bridge/src/megatron/bridge",
    "flagscale_ext/train": R / "3rdparty/common/FlagScale/flagscale/train",
    "flagscale_ext/examples": R / "3rdparty/common/FlagScale/examples",
}

# ext 树自己的包标记，不是增量：不落盘、也不覆盖上游的包入口（踩过：0 字节的
# src/flagscale_ext/__init__.py 曾把上游 flagscale/__init__.py（算 __version__ 的那份）盖空）
MARKERS = {"megatron_ext/__init__.py", "flagscale_ext/__init__.py"}


def upstream_head_has(upstream_dir: Path, rel: Path) -> bool:
    repo = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not repo:
        return (upstream_dir / rel).exists()
    try:
        inner = Path(upstream_dir, *rel.parts[:-1]).relative_to(repo).as_posix()
    except ValueError:
        return (upstream_dir / rel).exists()
    return (
        subprocess.run(
            ["git", "-C", repo, "cat-file", "-e", f"HEAD:{inner}/{rel.parts[-1]}"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


bad, checked, allowed, markers = [], 0, 0, 0
for ours_rel, upstream in MAP.items():
    ours = SRC / ours_rel
    if not ours.is_dir() or not upstream.is_dir():
        continue
    for f in ours.rglob("*"):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(ours)
        checked += 1
        if not upstream_head_has(upstream, rel):
            continue
        key = f"{ours_rel}/{rel}"
        if key in MARKERS:
            markers += 1
        elif key in BACKFILL:
            allowed += 1
        else:
            bad.append(key)

print(f"[invariant] 比对 {checked} 个增量文件（登记回填 {allowed} 个，包标记 {markers} 个）")
for k in sorted(MARKERS):
    print(f"[invariant]   包标记 {k}（不落盘、只存在于 ext 树里）")
for k, why in BACKFILL.items():
    print(f"[invariant]   回填 {k}\n              原因：{why}")
if bad:
    print(f"[invariant] ✗ 未登记的覆盖件 {len(bad)} 个：")
    for b in bad:
        print("   ", b)
    raise SystemExit(1)
print("[invariant] ✓ 除登记回填外全部是新增")
