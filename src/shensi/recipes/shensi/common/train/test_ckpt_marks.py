#!/usr/bin/env python3
"""纯权重检查点标记的闸门：保存侧落盘、加载侧点名、过期清理与训练循环钩子。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shensi.recipes.shensi.common.train import ckpt_marks  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def read_marker(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="shensi_ckpt_marks_"))
    save_dir = tmp / "ckpt"

    marker = ckpt_marks.write_marker(save_dir, no_save_optim=True, no_save_rng=False, iteration=7)
    info = read_marker(Path(marker)) if marker else {}
    check(
        "权重存盘：标记落盘且口径正确",
        marker
        and info.get("pure_weight") is True
        and info.get("top_level_keys") == ["model", "rng_state"]
        and info.get("iteration") == 7,
        f"{marker} -> {info}",
    )
    check(
        "加载侧识别：ckpt 根目录",
        ckpt_marks.is_pure_weight(save_dir) == marker,
        str(ckpt_marks.is_pure_weight(save_dir)),
    )
    check(
        "加载侧识别：指到具体迭代（往上一级找）",
        ckpt_marks.is_pure_weight(save_dir / "iter_0000007") == marker,
        str(ckpt_marks.is_pure_weight(save_dir / "iter_0000007")),
    )
    check("空路径不误判", ckpt_marks.is_pure_weight("") is None)

    liar = save_dir / "iter_0000009"
    liar.mkdir(parents=True, exist_ok=True)
    (liar / ckpt_marks.PURE_WEIGHT_MARKER).write_text(
        json.dumps(
            {
                "pure_weight": True,
                "top_level_keys": ["model", "optimizer", "rng_state"],
            }
        ),
        encoding="utf-8",
    )
    check(
        "脏标记不认（顶层键声称有 optimizer）",
        ckpt_marks.is_pure_weight(liar) is None,
        str(ckpt_marks.is_pure_weight(liar)),
    )

    with_rng_off = ckpt_marks.write_marker(save_dir, no_save_optim=True, no_save_rng=True)
    check(
        "no_save_rng 时顶层键不含 rng_state",
        read_marker(Path(with_rng_off)).get("top_level_keys") == ["model"],
        str(read_marker(Path(with_rng_off))),
    )

    removed = ckpt_marks.write_marker(save_dir, no_save_optim=False)
    check(
        "非权重存盘：过期标记被清掉",
        removed is None
        and not Path(marker).exists()
        and ckpt_marks.is_pure_weight(save_dir) is None,
        f"marker_exists={Path(marker).exists()}",
    )

    from megatron.training import global_vars
    from megatron.training import training as training_loop

    from shensi.recipes.shensi.common.train import args as shensi_args
    from shensi.recipes.shensi.common.train import probes

    calls: list[int] = []
    stub = lambda iteration, *a, **kw: calls.append(iteration)  # noqa: E731
    original = training_loop.save_checkpoint
    old_args = global_vars._GLOBAL_ARGS  # noqa: SLF001 - 测试注入 mcore 的全局 args
    try:
        training_loop.save_checkpoint = stub
        installed = probes.install_pure_weight_marker()
        again = probes.install_pure_weight_marker()
        global_vars._GLOBAL_ARGS = SimpleNamespace(  # noqa: SLF001
            save=str(save_dir), no_save_optim=True, no_save_rng=False
        )
        training_loop.save_checkpoint(3)
        check(
            "钩子：装上、幂等、存盘后按实况落标记",
            installed and not again and calls == [3],
            f"installed={installed} again={again} calls={calls}",
        )
        check(
            "钩子：标记带迭代号",
            read_marker(save_dir / ckpt_marks.PURE_WEIGHT_MARKER).get("iteration") == 3,
            str(read_marker(save_dir / ckpt_marks.PURE_WEIGHT_MARKER)),
        )
    finally:
        training_loop.save_checkpoint = original
        global_vars._GLOBAL_ARGS = old_args  # noqa: SLF001

    load_args = SimpleNamespace(load=str(save_dir), no_load_optim=False, no_load_rng=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        shensi_args.warn_pure_weight_load(load_args)
    said = buffer.getvalue()
    check(
        "加载侧点名：未给开关时提示 --no-load-optim",
        "no-load-optim" in said and "纯权重" in said,
        said.strip().splitlines()[-1][:90] if said.strip() else "(无输出)",
    )

    load_args.no_load_optim = True
    load_args.no_load_rng = True
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        shensi_args.warn_pure_weight_load(load_args)
    said = buffer.getvalue()
    check(
        "给了开关就不唠叨",
        "原样不动" in said,
        said.strip().splitlines()[-1][:90] if said.strip() else "(无输出)",
    )

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'SOME CHECKS FAILED'}")
    if FAILS:
        print("FAILED:", FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
