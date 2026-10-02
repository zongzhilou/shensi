#!/usr/bin/env python3
"""把某个臂起成 OpenAI 兼容端点（lmms-eval 接它做评测）。

- native：vLLM 原生 `Qwen3VLForConditionalGeneration`，无需登记；
- unified / gdar / deeprecur：先跑 `common/models/vllm/registration.py` 登记（桥），
  再以 `--trust_remote_code` 加载 —— 主干（注意力/KV/融合）仍走 vLLM 原生。

python serve.py --arm deeprecur --ckpt <HF 目录> --port 8000 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_command(*, ckpt: str, port: int, served_name: str, extra: list[str] | None = None) -> list[str]:
    """VLLM OpenAI 端点的启动命令（用 vllm 的 api_server 模块，避免依赖 PATH 上的二进制）。"""
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        ckpt,
        "--trust-remote-code",
        "--served-model-name",
        served_name,
        "--port",
        str(port),
    ] + list(extra or [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="起一个臂的评测端点（vLLM）")
    parser.add_argument("--arm", default="deeprecur", choices=["native", "unified", "gdar", "deeprecur"])
    parser.add_argument("--ckpt", required=True, help="HF 检查点目录")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-name", default=None, help="端点模型名（默认 = --arm）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[], help="透传给 vLLM 的参数")
    args = parser.parse_args(argv)
    served_name = args.served_name or args.arm

    if args.arm != "native":
        from shensi.recipes.paper.deeprecur.common.models.vllm.registration import register_all

        report = register_all()
        assert not report["failed"], f"[deeprecur·eval] 桥登记失败：{report['failed']}"

    command = build_command(ckpt=args.ckpt, port=args.port, served_name=served_name, extra=args.extra)
    print(f"[deeprecur·eval] 端点（arm={args.arm}）：{' '.join(command)}", flush=True)
    if args.dry_run:
        print(f"[deeprecur·eval] 起来后 lmms-eval 用 --base-url http://127.0.0.1:{args.port}/v1")
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
