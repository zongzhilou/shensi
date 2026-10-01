"""昇腾（NPU）装配自查：把「装环境（昇腾 / NPU 机）」一节里的清单跑成一遍可读的检查。

在 NPU 机上装完环境后第一件事：

    python -m shensi.utils.ascend_env

逐项打印 ✓ / ✗ / 提示：CANN、torch 与 torch_npu 的版本配对、设备可见性、五个组件能否 import、
以及五处已知差异（DSA 的 Hadamard、MindSpeed 与 mcore 的版本配对、numpy、两个 CUDA 专属包、
DSA 后端回退）。脚本只读环境，不改任何东西。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

# torch ↔ torch_npu 的配对（与 pyproject.ascend.toml 的注释同源）
PAIRS = (
    ("2.13", "2.13.0rc1"),
    ("2.9", "2.9.0"),
    ("2.8", "2.8.0"),
)

CUDA_ONLY_PACKAGES = ("flashinfer-python", "fast-hadamard-transform")


def _try(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - 自查脚本：任何异常都算"没就位"
        return default if default is not None else f"{type(exc).__name__}: {exc}"


def _version(pkg: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _module_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    ok = True
    print("== 昇腾装配自查 ==")

    # 1) CANN 环境
    home = os.environ.get("ASCEND_HOME_PATH")
    print(f"[1] CANN：ASCEND_HOME_PATH={home or '（未设置）'}")
    if not home:
        ok = False
        print("    → 先 source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh")
    else:
        print(f"    驱动包目录存在：{Path(home).is_dir()}")
    npu_smi = shutil.which("npu-smi")
    print(f"    npu-smi：{npu_smi or '（没有，容器里常见）'}")

    # 2) torch 与 torch_npu 的版本配对
    t = _version("torch")
    tn = _version("torch_npu")
    print(f"[2] torch={t} torch_npu={tn}")
    if t and tn:
        pair = next((p for p in PAIRS if t.startswith(p[0])), None)
        expect = pair[1] if pair else None
        if expect and not tn.startswith(expect):
            ok = False
            print(f"    ✗ 建议配对：torch {t} ↔ torch_npu {expect}（详见 pyproject.ascend.toml）")
        else:
            print("    ✓ 版本配对在表内")

    # 3) 设备可见
    devs = _try(lambda: __import__("torch").npu.device_count(), None)
    print(f"[3] NPU 设备数：{devs if isinstance(devs, int) else '不可用（torch_npu 未装或没在 NPU 机上）'}")
    if not isinstance(devs, int):
        print("    → 这是 CUDA 机器或 torch_npu 没装；本项在 NPU 机上才应为 >0")

    # 4) 组件 import
    print("[4] 组件 import：")
    for mod, hint in (
        ("megatron.core", "mcore（上游 main，可编辑安装）"),
        ("megatron.bridge", "Megatron-Bridge（models/shensi 在这里）"),
        ("mindspeed", "MindSpeed"),
        ("mindspeed_ops", "MindSpeed-Ops（--no-build-isolation 编）"),
        ("transformer_engine_npu", "TransformerEngineNPU"),
        ("verl", "verl"),
        ("vllm", "vllm（要带 VLLM_VERSION_OVERRIDE 编）"),
    ):
        has = _module_ok(mod)
        ok &= has or mod in ("vllm",)  # vllm 只在推理机上必须
        print(f"    {'✓' if has else '✗'} {mod:<24} {hint}")

    # 5) 五处已知差异
    print("[5] 已知差异：")
    hadamard = _module_ok("fast_hadamard_transform")
    from shensi.utils.dsa import dsa_backend_fallback

    chosen = dsa_backend_fallback(None)
    print(
        f"    DSA 的 Hadamard：fast_hadamard_transform {'在' if hadamard else '不在'}；"
        f"dsa_kernel_backend 会取 {chosen!r}（NPU 上没有融合内核时按 none 走 PyTorch 实现）"
    )
    mcore_v = _version("megatron-core")
    mind_v = _try(lambda: subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3] / "3rdparty/ascend/MindSpeed"),
         "describe", "--tags", "--always"], capture_output=True, text=True, check=False
    ).stdout.strip(), "")
    print(f"    MindSpeed 与 mcore：mcore={mcore_v}，MindSpeed={mind_v or '（没在 3rdparty 下）'}"
          "（官方配 core_v0.12.1，本仓用上游 main）")
    numpy_v = _version("numpy")
    print(f"    numpy={numpy_v}（MindSpeed 要 <2，verl/vllm 这条线要 2.x：按机器分工装）")
    on_npu = bool(home)
    for pkg in CUDA_ONLY_PACKAGES:
        has = _version(pkg) is not None
        if has and on_npu:
            ok = False
            print(f"    ✗ {pkg} 已装：NPU 机上要 `uv sync --no-install-package {pkg}` 排掉")
        else:
            print(f"    ✓ {pkg}：{'已装（CUDA 机上本来就该有）' if has else '未装'}")
    print("    未上 NPU 实测：本脚本就是上机后的第一遍检查（清单与命令见 README 的昇腾一节）")

    print("== 结束：", "全部就位" if ok else "有项目未就位（见上面的 ✗ 与提示）", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
