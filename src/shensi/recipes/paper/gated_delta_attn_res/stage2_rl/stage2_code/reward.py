#!/usr/bin/env python3
"""代码 的可验证奖励：单元测试执行通过率（0/1 或按用例比例）。

⚠️ 诚实标注：这里的执行是**直接 subprocess**，生产要换隔离沙箱（firejail / 容器 /
nsjail），防止生成代码做坏事；接口不变。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 10


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """ground_truth = JSON：{"tests": "<pytest 文件内容>"}；代码块从 solution 提取。"""
    try:
        tests = json.loads(ground_truth).get("tests", "")
    except (ValueError, AttributeError):
        tests = str(ground_truth)
    code = solution_str
    if "```" in code:
        blocks = [b.split("```")[0] for b in code.split("```python")[1:]] or [""]
        code = blocks[0]
    with tempfile.TemporaryDirectory() as td:
        Path(td, "solution.py").write_text(code, encoding="utf-8")
        Path(td, "test_solution.py").write_text(
            f"from solution import *\n\n{tests}", encoding="utf-8"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=td,
                capture_output=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if proc.returncode == 0 else 0.0
