#!/usr/bin/env python3
# 世界模型接成 verl 工具：多轮状态机与工具解析走上游 ToolAgentLoop，这里只实现一个工具。

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .world_model import DOMAINS, MODES, WorldModelEnv

DEFAULT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_action",
        "description": (
            "Execute an action in the simulated environment and get the environment observation. "
            "The observation is predicted by a language world model, not by a real machine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to execute, e.g. a shell command or a tool call with its arguments.",
                }
            },
            "required": ["action"],
        },
    },
}


class WorldModelTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(DEFAULT_SCHEMA)
        super().__init__(config or {}, tool_schema)
        self.envs: dict[str, WorldModelEnv] = {}
        self.dump_dir = Path(self.config["dump_dir"]) if self.config.get("dump_dir") else None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _env_kwargs(self, create_kwargs: dict) -> dict:
        """默认取工具配置，数据行的 tools_kwargs 可逐字段覆盖（域 / 口径 / 任务 / 设定）。"""
        cfg = {**self.config, **(create_kwargs or {})}
        return {
            "base_url": cfg.get("base_url"),
            "model": cfg.get("model"),
            "domain": cfg.get("domain") or "terminal",
            "mode": cfg.get("mode") or "sim",
            "spec": cfg.get("spec"),
            "task": cfg.get("task"),
            "max_turns": int(cfg.get("max_turns") or 8),
        }

    async def create(self, instance_id: str | None = None, **kwargs):
        kwargs_env = self._env_kwargs(kwargs.get("create_kwargs"))
        if kwargs_env["domain"] not in DOMAINS or kwargs_env["mode"] not in MODES:
            raise ValueError(
                f"[world_model] 域/口径不对：{kwargs_env['domain']} / {kwargs_env['mode']}"
            )
        env = WorldModelEnv(**kwargs_env)
        env.reset()
        iid = instance_id or uuid4().hex
        self.envs[iid] = env
        return iid, ToolResponse(
            text=f"[simulated environment ready] domain={env.domain} mode={env.mode}"
        )

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        env = self.envs.get(instance_id)
        if env is None:
            raise ValueError(f"[world_model] 没有这个 instance：{instance_id}")
        action = str((parameters or {}).get("action") or "").strip()
        if not action:
            return ToolResponse(text="[world_model] action 为空，请给出要执行的动作。"), 0.0, {}
        state = await asyncio.to_thread(
            env.step, self.config.get("action_name") or self.name, None, action
        )
        text = state["observation"]
        if state["done"]:
            text += "\n\n[simulation ended]"
        metrics = {"world_model_turns": state["turn"], "world_model_done": float(state["done"])}
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Sim RL 的判分仍走数据行的 verifier（reward_model.ground_truth + reward.py），工具本身不给分
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        env = self.envs.pop(instance_id, None)
        if env is None or self.dump_dir is None or not env.history:
            return
        # 落盘的是（动作, 观测）轨迹：Sim RL 滚出来的数据就是下一轮世界模型的语料
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        out = self.dump_dir / f"{instance_id}.json"
        out.write_text(
            json.dumps(
                {
                    "domain": env.domain,
                    "mode": env.mode,
                    "system": env.system,
                    "trajectory": env.history,
                    "env": {"base_url": env.client.base_url, "model": env.client.model},
                    "os_pid": os.getpid(),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
