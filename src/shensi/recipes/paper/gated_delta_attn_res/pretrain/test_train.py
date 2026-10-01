#!/usr/bin/env python3
"""GDAR 配方 · 集成测试：tiny 几何跑 5 步并按日志判 PASS/FAIL。

判定口径与 shensi 配方的 tiny_test 一致：rc=0、跑到最后一次 iteration、出现
`[after training is done]`、日志无 Traceback/Error。数据有 data_prep 产物就用真实
bin/idx，没有就退回 mock 冒烟档（config/tiny.yaml），日志里会说明用的哪种。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common

_ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
_BAD = re.compile(r"Traceback|^.*\bERROR\b.*|AssertionError|ValueError|RuntimeError", re.M)


def main() -> int:
    iters = 5
    paths = common.env_paths()
    data_dir = Path(paths["data"])
    has_data = (data_dir / "blend.json").is_file()
    if has_data:
        cfg = common.build_config("debug", [], data_dir)
        print(f"[test_train:gdar] 数据：{data_dir}（真实 bin/idx）")
    else:
        cfg = common.smoke_config("tiny")
        print(
            f"[test_train:gdar] 数据：mock（没找到 {data_dir / 'blend.json'}；"
            "想跑真实数据先 python data_prep.py --prepare）"
        )
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / f"{common.STAGE}_tiny")
    cfg["experiment"]["exp_name"] = f"{common.STAGE}_tiny"
    common._set_dotted(cfg, "train.model.train_iters", iters)
    common._set_dotted(cfg, "train.system.checkpoint.save_interval", iters)
    cfg = common.resolve_cfg(cfg)

    log = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    if log.exists():
        log.unlink()
    print(f"[test_train:gdar] 跑 tiny 档（GDAR 主 spec，{iters} 步）…")
    rc = common.run(cfg, dry_run=False)

    text = log.read_text(encoding="utf-8", errors="ignore") if log.is_file() else ""
    steps = [int(m.group(1)) for m in _ITER.finditer(text)]
    done = "[after training is done]" in text
    bad = [m.group(0)[:120] for m in _BAD.finditer(text) if "error_injection" not in m.group(0)]
    last = steps[-1] if steps else 0
    ckpt = Path(cfg["train"]["system"]["checkpoint"]["save"])

    print(f"[test_train:gdar] rc={rc} 最后 iteration={last} 收尾标记={done} 报错行={len(bad)}")
    for line in [ln for ln in text.splitlines() if "iteration" in ln and "/" in ln][-1:]:
        print("  " + line.strip()[:160])
    ok = rc == 0 and done and last == iters and not bad
    if not ok:
        print("[test_train:gdar] FAIL")
        for line in bad[:3]:
            print("  ! " + line)
        return 1
    print(f"[test_train:gdar] PASS（ckpt 在 {ckpt}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
