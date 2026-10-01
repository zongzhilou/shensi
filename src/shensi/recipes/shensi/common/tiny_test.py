"""集成测试的共用部分：跑一个 stage 的 tiny 档，按日志判 PASS/FAIL。

各 stage 的 `test_train.py` 只声明自己的 stage 名与 profile，判定逻辑都在这里：

1. 用 `tiny_model.as_cli_overrides()` 把极小几何覆盖到该 stage 的 profile 上；
2. 走该 stage 自己的 `train.py`（= 生产入口），日志落 `<exp_dir>/logs/host_0_localhost.output`；
3. 判据：返回码为 0、日志里出现最后一次 iteration、出现 `[after training is done]`、
   没有 `Traceback` / `Error`；
4. 顺带把最终的 iteration 行与 ckpt 路径打出来，便于人工核对。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.shensi.common import common, tiny_model

_ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
_BAD = re.compile(r"Traceback|^.*\bERROR\b.*|AssertionError|ValueError|RuntimeError", re.M)


def run_stage_tiny(
    stage: str,
    profile: str = "debug",
    *,
    override: list[str] | None = None,
    iters: int | None = None,
    mtp_layers: int = 0,
) -> int:
    """跑 `stage` 的 `profile` 档（几何换成 tiny），判 PASS/FAIL 并返回退出码。

    数据：该 stage 的 `data_prep.py` 产物（`<data>/<stage>/blend.json`）在就用真实 bin/idx；
    没准备就退回仓库内的 mock 数据档（`config/tiny.yaml`），并在日志里说清楚用的哪种。

    `iters=None` 时用 profile 自己的 `train_iters`：各段的迭代号是跨阶段连续计数的
    （stage1 到 5、stage2 到 10、stage3 到 15），覆盖成固定值会让 `train_samples` 小于已消费数。
    """
    paths = common.env_paths()
    data_dir = Path(paths["data"]) / stage
    has_data = (data_dir / "blend.json").is_file()
    if has_data:
        cfg = common.build_config(stage, profile, [], data_dir)
        print(f"[test_train:{stage}] 数据：{data_dir}（真实 bin/idx）")
    else:
        cfg = common.smoke_config("tiny")
        print(
            f"[test_train:{stage}] 数据：mock（没找到 {data_dir / 'blend.json'}；"
            f"想跑真实数据先 python data_prep.py --prepare）"
        )
    cfg["experiment"]["exp_dir"] = str(Path(paths["runs"]) / f"{stage}_{profile}_tiny")
    cfg["experiment"]["exp_name"] = f"{stage}_{profile}_tiny"
    overrides = tiny_model.as_cli_overrides(mtp_layers=mtp_layers, iters=iters)
    overrides += list(override or [])
    for item in overrides:
        key, _, val = item.partition("=")
        common._set_dotted(cfg, key, common._coerce(val))  # noqa: SLF001  配方内部的点号覆写
    cfg = common.resolve_cfg(cfg)

    log = Path(cfg["experiment"]["exp_dir"]) / "logs/host_0_localhost.output"
    if log.exists():
        log.unlink()
    expected_iters = int(iters if iters is not None else cfg["train"]["model"]["train_iters"])
    print(f"[test_train:{stage}] 跑 {profile} 档（tiny 几何，止于 iteration {expected_iters}）…")
    rc = common.run(cfg, stage, profile, dry_run=False)

    text = log.read_text(encoding="utf-8", errors="ignore") if log.is_file() else ""
    steps = [int(m.group(1)) for m in _ITER.finditer(text)]
    done = "[after training is done]" in text
    bad = [m.group(0)[:120] for m in _BAD.finditer(text) if "error_injection" not in m.group(0)]
    last = steps[-1] if steps else 0
    ckpt = Path(cfg["train"]["system"]["checkpoint"]["save"])

    print(f"[test_train:{stage}] rc={rc} 最后 iteration={last} 收尾标记={done} 报错行={len(bad)}")
    for line in [ln for ln in text.splitlines() if "iteration" in ln and "/" in ln][-1:]:
        print("  " + line.strip()[:160])
    ok = rc == 0 and done and last == expected_iters and not bad
    if not ok:
        print(f"[test_train:{stage}] FAIL")
        for line in bad[:3]:
            print("  ! " + line)
        return 1
    print(f"[test_train:{stage}] PASS（检查点：{ckpt}）")
    return 0


def expect(cond: bool, name: str, detail: str = "") -> bool:
    """一行式判定打印（preflight 用）。"""
    print(f"  {'✓' if cond else '✗'} {name}" + (f" — {detail}" if detail else ""))
    return cond


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def ray_preflight(init_kwargs: dict | None = None) -> bool:
    """起一次 ray 再关掉（RL 的 driver 就是用它调度 actor/vLLM）。"""
    try:
        import ray

        ray.init(ignore_reinit_error=True, include_dashboard=False, **(init_kwargs or {}))
        res = ray.cluster_resources()
        ray.shutdown()
        return expect(
            res.get("GPU", 0) >= 1, "ray 集群", f"CPU={res.get('CPU')} GPU={res.get('GPU')}"
        )
    except Exception as exc:  # noqa: BLE001
        return expect(False, "ray 集群", f"{type(exc).__name__}: {exc}")


def env_preflight(vars_: tuple[str, ...]) -> bool:
    """检查 RL/评测要用的环境变量（有就给值，没有就列出缺的）。"""
    missing = [v for v in vars_ if not os.environ.get(v)]
    for v in vars_:
        if os.environ.get(v):
            print(f"  ✓ env:{v}={os.environ[v]}")
    if missing:
        print(f"  ○ env 未设置（跑真训练前按 README 的「环境注意事项」补）：{missing}")
    return True
