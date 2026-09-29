import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import shutil
import site
import sys
import sysconfig
import tempfile
from pathlib import Path

__version__ = "0.1.0"

_SRC_ROOT = Path(__file__).resolve().parent.parent
# megatron 侧：我们自己的代码一律直接 `from megatron_ext...` 导入；overlay 是给**第三方**用的——
# Megatron-Bridge 会 import `megatron.training.models.gpt`、`megatron.core.transformer.mla_qk_norm_config`
# 这类「mcore main 才有的模块」，FL fork 里没有，只能由 megatron_ext 提供
_OVERLAYS = ((_SRC_ROOT / "megatron_ext", "megatron"), (_SRC_ROOT / "flagscale_ext", "flagscale"))
_LIBRARIES = tuple(lib for _, lib in _OVERLAYS)
_OVERLAY_BASES = {lib: Path(base) for base, lib in _OVERLAYS}


def _installed_roots() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    roots = {paths.get("purelib"), paths.get("platlib")}
    roots.update(getattr(site, "getsitepackages", lambda: [])())
    return tuple(sorted(str(Path(root).resolve()) for root in roots if root))


class _ExtFinder(importlib.abc.MetaPathFinder):
    """把 src/megatron_ext、src/flagscale_ext 补到装在 site-packages 里的同名库上。

    包（带 __init__.py 的目录）会把我们的目录排在包自己 __path__ 的最前、已装的目录留在后面，
    于是同名文件被盖住、其余兄弟模块照常从已装的库里取。显式放在 sys.path 上的树（PYTHONPATH、
    工作区 clone、可编辑安装）一律让位——那些树本来就带着同一份增量。
    """

    def __init__(self, overlays=_OVERLAYS, installed: tuple[str, ...] | None = None) -> None:
        self.bases = {lib: Path(base) for base, lib in overlays}
        self.libraries = tuple(self.bases)
        self.installed = _installed_roots() if installed is None else tuple(installed)

    def _split(self, fullname: str) -> tuple[str, tuple[str, ...]] | None:
        for lib in self.libraries:
            if fullname == lib:
                return lib, ()
            head = f"{lib}."
            if fullname.startswith(head):
                return lib, tuple(fullname[len(head) :].split("."))
        return None

    def find_spec(self, fullname, path=None, target=None):
        split = self._split(fullname)
        if split is None:
            return None
        lib, parts = split
        if not parts:
            # 顶层包（megatron / flagscale）本身一律让给上游：这里若接管，包的 __path__ 就只剩我们的
            # 目录，上游子模块（megatron.core.transformer.module 这类）会整片找不到。
            return None
        base = self.bases[lib]
        package = base.joinpath(*parts, "__init__.py")
        module = base.joinpath(*parts).with_suffix(".py")
        if package.is_file():
            spec = importlib.util.spec_from_file_location(
                fullname, package, submodule_search_locations=[str(package.parent)]
            )
        elif module.is_file():
            spec = importlib.util.spec_from_file_location(fullname, module)
        else:
            return None
        upstream = self._upstream(fullname, path)
        if upstream is not None and upstream.origin and not self._is_installed(upstream.origin):
            return None
        if spec.submodule_search_locations is not None and upstream is not None:
            for location in upstream.submodule_search_locations or ():
                if location not in spec.submodule_search_locations:
                    spec.submodule_search_locations.append(location)
        return spec

    def _is_installed(self, origin: str) -> bool:
        resolved = str(Path(origin).resolve())
        return any(resolved == root or resolved.startswith(root + os.sep) for root in self.installed)

    def _upstream(self, fullname: str, path):
        if path is None:
            return importlib.machinery.PathFinder.find_spec(fullname)
        bases = tuple(str(base) for base in self.bases.values())
        upstream = [entry for entry in path if not str(entry).startswith(bases)]
        if not upstream:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, upstream)


def _install_overlay() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _ExtFinder) and finder.bases == _OVERLAY_BASES:
            return
    sys.meta_path.insert(0, _ExtFinder())


_install_overlay()


def activate(env: dict) -> dict:
    """让子进程也接管增量：写一个只有 `import shensi` 的 sitecustomize，并把它所在目录挂到 PYTHONPATH 最前。"""
    if not any(base.is_dir() for base, _ in _OVERLAYS):
        return env
    stub = Path(tempfile.gettempdir()) / "shensi_activate" / "sitecustomize.py"
    if not stub.is_file():
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("import shensi\n", encoding="utf-8")
    head = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{stub.parent}{os.pathsep}{head}" if head else str(stub.parent)
    return env


# 逻辑路径（<顶层包>/<包内相对路径>）→ 工作区里的仓库路径（现在只有 flagscale）
LAYOUT = (
    ("megatron/tests/", "Megatron-Bridge/tests/"),
    ("flagscale/examples/", "FlagScale/examples/"),
    ("flagscale/", "FlagScale/flagscale/"),
)


def _ext_files():
    """(逻辑路径, 实际文件)：现在只有 flagscale（src/flagscale_ext）。"""
    for base, lib in _OVERLAYS:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                yield f"{lib}/{path.relative_to(base).as_posix()}", path


def _shipped(lib: str) -> int:
    return sum(1 for logical, _ in _ext_files() if logical.startswith(f"{lib}/"))


def apply_ext(root: Path, dry_run: bool = False) -> int:
    """把增量按 LAYOUT 落盘到工作区各 clone；只写有差异的文件。"""
    if not any(base.is_dir() for base, _ in _OVERLAYS):
        raise SystemExit("没有增量目录")
    if not root.is_dir():
        raise SystemExit(f"工作区不存在：{root}（用 SHENSI_ROOT 或 --root 指定）")
    copied, same, skipped = 0, 0, []
    for logical, source in _ext_files():
        destination = None
        for prefix, workspace in LAYOUT:
            if logical.startswith(prefix):
                destination = root / workspace / logical[len(prefix) :]
                break
        if destination is None:
            if not logical.startswith("megatron/"):   # megatron 侧由 overlay 提供，不落盘
                skipped.append(logical)
            continue
        if destination.is_file() and destination.read_bytes() == source.read_bytes():
            same += 1
            continue
        print(f"  {'写入' if destination.exists() else '新增'} {destination.relative_to(root)}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        copied += 1
    print(f"[apply-ext] 目标工作区：{root}")
    print(f"[apply-ext] {'将' if dry_run else '已'}写入 {copied} 个文件，{same} 个已一致")
    if skipped:
        print(f"[apply-ext] 未匹配布局（跳过）：{skipped}")
    if not dry_run and not (root / "FlagScale/flagscale/train/megatron/train_shensi.py").is_file():
        print("[apply-ext] 提醒：FlagScale 树里没有 train_shensi.py，配方的 entrypoint 起不来")
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["apply-ext"]:
        root = Path(os.environ.get("SHENSI_ROOT", "/root/work/shensi"))
        if "--root" in argv:
            root = Path(argv[argv.index("--root") + 1])
        raise SystemExit(apply_ext(root, dry_run="--dry-run" in argv))
    print(f"shensi {__version__}")
    for base, lib in _OVERLAYS:
        if not base.is_dir():
            print(f"  {lib}: 未随包提供（{base}）")
            continue
        try:
            importlib.import_module(lib)
        except Exception:
            print(f"  {lib}: 环境未安装")
            continue
        print(f"  {lib}: 已接管 {_shipped(lib)} 个文件（{base}）")
    print("  （把增量落盘到工作区：shensi apply-ext [--root PATH] [--dry-run]）")
