#!/usr/bin/env python3
"""把 `src/megatron_ext`、`src/flagscale_ext` 的增量文件挂进 venv 里已安装的 megatron / flagscale。

为什么需要这一步：清单里 `module-name` 只把 `src/megatron_ext` 装成顶层包 `megatron_ext`，而 mcore 与
Megatron-Bridge 都在 `megatron` 这一个普通包里（Bridge 的 wheel 同样把自己的文件铺进 `megatron/bridge/`）。
`import shensi` 的 meta-path overlay 只能在显式 import shensi 的进程里生效，torchrun / ray worker /
vLLM 子进程不一定走那条路，所以这里把增量文件**铺进 site-packages**，任何进程都拿得到。

用法：`python -m shensi.install_ext`（幂等；内容相同就跳过）。配方启动 RL 前会自动调一次。
"""

from __future__ import annotations

import argparse
import shutil
import site
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
MAP = ((SRC / "megatron_ext", "megatron"), (SRC / "flagscale_ext", "flagscale"))
# 让"没显式 import shensi"的进程也拿到 overlay 与第三方注册（site 启动时执行，shensi 本身很轻）
AUTOLOAD = "shensi_ext_autoload.pth"
AUTOLOAD_LINE = "import shensi\n"


def site_packages() -> Path:
    """当前解释器的 site-packages 目录。"""
    for p in list(site.getsitepackages() or []) + [site.getusersitepackages()]:
        if Path(p).is_dir():
            return Path(p)
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def mount(dst_root: Path | None = None, verbose: bool = False) -> list[tuple[str, str]]:
    """返回 [(包, 文件)] 形式的落地清单。"""
    dst_root = Path(dst_root) if dst_root else site_packages()
    done: list[tuple[str, str]] = []
    for src, pkg in MAP:
        dst = dst_root / pkg
        if not src.is_dir() or not dst.is_dir():
            if verbose:
                print(f"跳过 {src}（或目标 {dst} 不存在）")
            continue
        for f in sorted(src.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix != ".py":
                continue
            rel = f.relative_to(src)
            if rel.as_posix() == "__init__.py":
                # 我们 ext 树的包标记而已：上游的 megatron/__init__.py、flagscale/__init__.py
                # 是人家的包入口（flagscale 那版还会算 __version__），不能盖
                continue
            target = dst / rel
            if target.exists() and target.read_bytes() == f.read_bytes():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
            for pyc in target.parent.glob(f"__pycache__/{target.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)
            done.append((pkg, str(rel)))
    pth = dst_root / AUTOLOAD
    if dst_root.is_dir() and (not pth.is_file() or pth.read_text(encoding="utf-8") != AUTOLOAD_LINE):
        pth.write_text(AUTOLOAD_LINE, encoding="utf-8")
        done.append(("", AUTOLOAD))
    return done


def main(argv: list[str] | None = None) -> int:
    """命令行入口：`python -m shensi.install_ext [-v] [--dst PATH]`。"""
    ap = argparse.ArgumentParser(description="把 shensi 增量挂进已安装的 megatron / flagscale")
    ap.add_argument("--dst", default=None, help="目标 site-packages（默认当前解释器的）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    done = mount(args.dst, args.verbose)
    if not done:
        print("已是最新（无需挂载）")
        return 0
    for pkg, rel in done:
        print(f"  {pkg}/{rel}" if pkg else f"  {rel}")
    print(f"挂载 {len(done)} 个文件到 {args.dst or site_packages()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
