"""RL 的 harness 工具：把外部 agent harness 接进 rollout。"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from shensi.recipes.shensi.common import harness

DEFAULT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_action",
        "description": "Execute an action in the environment and get the observation back.",
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

DEFAULT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_action",
        "description": "Execute an action in the environment and get the observation back.",
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


class HarnessTool(BaseTool):
    """agent 方向的 harness 工具：在 rollout 里调用外部 harness。"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(DEFAULT_SCHEMA)
        super().__init__(config or {}, tool_schema)
        self.cfg = dict(config or {})
        self.timeout = float(self.cfg.get("timeout") or 60.0)
        self.base_url = self.cfg.get("base_url")
        self.model = self.cfg.get("model")
        self.env = harness.harness_env(self.cfg, base_url=self.base_url or "", model=self.model)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs):
        for cmd in harness.setup_commands(self.cfg, base_url=self.base_url, model=self.model):
            if str(cmd).startswith("pip install"):
                continue
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        action = str((parameters or {}).get("action") or "").strip()
        if not action:
            return ToolResponse(text="empty action"), 0.0
        command = (
            [harness.DEFAULT_COMMAND, "exec", "--", action]
            if self.cfg.get("as_exec")
            else shlex.split(action)
        )
        try:
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={**os.environ, **self.env},
                ),
            )
            out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        except FileNotFoundError:
            out = f"[harness] {harness.DEFAULT_COMMAND} 不在 PATH（沙箱里先 {harness.SDK_PACKAGE}）"
        except subprocess.TimeoutExpired:
            out = f"[harness] timeout after {self.timeout}s"
        return ToolResponse(text=out[: self.cfg.get("max_chars") or 8192]), 0.0

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        return None
