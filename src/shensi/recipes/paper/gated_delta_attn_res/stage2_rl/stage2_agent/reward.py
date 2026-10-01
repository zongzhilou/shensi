#!/usr/bin/env python3
"""Agent 的奖励：任务成功标志（环境给出）。

agent 臂走 verl 的多轮 rollout（config 里 rollout.multi_turn.enable），奖励由环境
在轨迹结束时给出 0/1（任务完成）+ 可选的部分得分；本模块只做字段归一。
环境接入（NeMo Gym / 自建工具环境）落位后把 env 回填进 ground_truth 字段即可。
"""

from __future__ import annotations


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    try:
        return float(json.loads(ground_truth)["success"])  # type: ignore[name-defined]
    except Exception:  # noqa: BLE001
        gt = str(ground_truth).strip().lower()
        return 1.0 if gt in ("1", "true", "success") else 0.0


import json  # noqa: E402  （放在函数后是刻意的：保持本文件"一眼看懂奖励"的形状）
