#!/usr/bin/env python3
"""世界模型集成预检。"""

import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记（顺带验证注册表生效）
from shensi.recipes.shensi.common import rl, tiny_test

STAGE = "stage4_world_model"


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Shensi {STAGE} 集成预检")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--model-path", default=None, help="HF ckpt（默认取 profile 里的）")
    ap.add_argument("--data-dir", default=None, help="data_prep.py 产出的 parquet 目录")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    results = []
    print(f"[test_train:{STAGE}] 预检（profile={args.profile}）")

    try:
        import shensi.recipes.shensi.common as pretrain_common

        # 世界模型的 RL 段用的是 config/rl/<档>.yaml（config/<档>.yaml 里还带 cpt/world_model 两段，不是 verl 的键）
        profile = (
            args.profile if (here / "config/rl" / f"{args.profile}.yaml").is_file() else "default"
        )
        cfg = pretrain_common.resolve_cfg(  # noqa: SLF001
            rl._load_with_base(here / "config/rl" / f"{profile}.yaml")  # noqa: SLF001
        )
        if args.model_path:
            cfg.setdefault("model", {})["path"] = args.model_path
        rl.resolve_paths(cfg, here)
        data_dir = (
            Path(args.data_dir)
            if args.data_dir
            else Path(pretrain_common.env_paths()["data"]) / STAGE
        )
        cmd = rl.build_command(cfg, STAGE, data_dir, here.parent / "reward.py")
        results.append(tiny_test.expect(bool(cmd), "配置解析 + 命令拼装", f"{len(cmd)} 个参数"))
    except Exception as exc:  # noqa: BLE001
        results.append(
            tiny_test.expect(False, "配置解析 + 命令拼装", f"{type(exc).__name__}: {exc}")
        )

    if args.data_dir:
        ok = True
        for name in ("train.parquet", "val.parquet"):
            f = Path(args.data_dir) / name
            if not f.is_file():
                ok = tiny_test.expect(False, f"数据 {name}", f"没有：{f}") and ok
                continue
            try:
                import pyarrow.parquet as pq

                ok = (
                    tiny_test.expect(
                        True, f"数据 {name}", f"{pq.ParquetFile(f).metadata.num_rows} 行"
                    )
                    and ok
                )
            except Exception as exc:  # noqa: BLE001
                ok = tiny_test.expect(False, f"数据 {name}", f"{type(exc).__name__}: {exc}") and ok
        results.append(ok)
    else:
        print("  ○ 数据检查跳过（没给 --data-dir；真跑前先 python data_prep.py --prepare）")

    results.append(tiny_test.ray_preflight({"num_cpus": 8}))
    results.append(tiny_test.expect(tiny_test.gpu_available(), "GPU 可见"))
    for mod in ("verl", "vllm", "megatron.core", "megatron.bridge", "shensi.runtime"):
        try:
            __import__(mod)
            results.append(tiny_test.expect(True, f"import {mod}"))
        except Exception as exc:  # noqa: BLE001
            results.append(tiny_test.expect(False, f"import {mod}", f"{type(exc).__name__}: {exc}"))

    tiny_test.env_preflight(("VERL_USE_EXTERNAL_MODULES", "VERL_PLATFORM", "TE_FL_PREFER"))

    ok = all(results)
    print(f"[test_train:{STAGE}] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
