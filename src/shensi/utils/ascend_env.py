"""Ascend preflight with real allocation/backprop and machine-readable results."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _probe(code: str, timeout: int = 120) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-6000:],
            "stderr": proc.stderr[-6000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect(scope: str = "all") -> dict:
    """Missing packages, inaccessible devices and broken imports all fail preflight."""
    checks = {}
    cann_home = os.environ.get("ASCEND_HOME_PATH", "")
    checks["cann"] = {"ok": bool(cann_home) and Path(cann_home).is_dir(), "path": cann_home}
    # Import and a device probe are mandatory: package metadata is not ABI validation.
    checks["torch_npu"] = _probe(
        "import torch, torch_npu; "
        "print('torch', torch.__version__, 'torch_npu', torch_npu.__version__); "
        "assert torch.npu.is_available(), 'No available NPU'; "
        "assert torch.npu.device_count() > 0, 'Zero NPU devices'; "
        "torch.npu.set_device(0); "
        "x = torch.ones(4, 4, device='npu', requires_grad=True); "
        "loss = (x @ x).sum(); loss.backward(); torch.npu.synchronize(); "
        "assert torch.isfinite(x.grad).all().item(); "
        "print('device_count', torch.npu.device_count(), 'backward_loss', loss.item())"
    )
    if scope in ("train", "all"):
        checks["megatron"] = _probe(
            "import torch, torch_npu, megatron_adaptor; "
            "import transformer_engine.pytorch; import megatron.core; "
            "import megatron.training; "
            "print('MegatronAdaptor + TransformerEngineNPU + Megatron imports OK')"
        )
    if scope == "all":
        checks["megatron_bridge"] = _probe(
            "import megatron.bridge; print('Megatron Bridge imports OK')"
        )
        checks["verl"] = _probe(
            "import os; os.environ['VERL_PLATFORM']='huawei'; "
            "import torch, torch_npu, megatron_adaptor; import verl.trainer.main_ppo; "
            "print('verl imports OK')"
        )
        checks["vllm_ascend"] = _probe("import vllm, vllm_ascend; print('vLLM Ascend imports OK')")
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "scope": scope,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: _version(name)
            for name in (
                "torch",
                "torch-npu",
                "transformers",
                "megatron-core",
                "megatron-adaptor",
                "transformer-engine",
                "mindspeed",
                "mindspeed-ops",
                "verl",
                "vllm",
                "vllm-ascend",
            )
        },
        "checks": checks,
        "note": "Import/one-device preflight only; not proof of recipe or HCCL training success.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("reference", "train", "all"), default="all")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = inspect(args.scope)
    for name, check in report["checks"].items():
        print(f"[{'PASS' if check['ok'] else 'FAIL'}] {name}")
        for key in ("stdout", "stderr", "error"):
            if check.get(key):
                print(check[key].strip())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ASCEND PREFLIGHT: " + ("PASS" if report["ok"] else "FAIL"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
