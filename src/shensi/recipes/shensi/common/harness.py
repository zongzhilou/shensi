#!/usr/bin/env python3
"""外部 harness 的统一接线（默认 DeepSeek Harness）。"""

import os
import shutil
from pathlib import Path

HARNESS_DEFAULT = "dsh"
SDK_PACKAGE = "deepseek-harness-sdk"
DEFAULT_COMMAND = "dsh"
DEFAULT_HOME_SUFFIX = "shensi/dsh-home"


def harness_cfg(cfg: dict) -> dict:
    """取 `harness:` 段（各段 yaml 里可选），缺项补默认值。"""
    return dict(cfg.get("harness") or {})


def harness_name(cfg: dict) -> str:
    return str(harness_cfg(cfg).get("name") or HARNESS_DEFAULT)


def dsh_home(cfg: dict) -> str:
    """`$DSH_HOME`：dsh 按它下面的 profile 启动（默认落在 $SHENSI_FS 里）。"""
    home = harness_cfg(cfg).get("home")
    if home:
        return str(home)
    fs = os.environ.get("SHENSI_FS") or os.environ.get("SHENSI_ROOT") or "/root/work/filestorage"
    return str(Path(fs) / DEFAULT_HOME_SUFFIX)


def setup_commands(
    cfg: dict, *, base_url: str | None = None, model: str | None = None
) -> list[str]:
    """进沙箱/容器先做的事：装 harness、导出 DSH_HOME、把端点与模型名接上。"""
    sec = harness_cfg(cfg)
    commands = [f"pip install {sec.get('install') or SDK_PACKAGE}"]
    commands.append(f"export DSH_HOME={dsh_home(cfg)}")
    if base_url:
        commands.append(f"export OPENAI_BASE_URL={base_url}")
    if model:
        commands.append(f"export OPENAI_MODEL={model}")
    commands += [str(c) for c in sec.get("setup_commands") or []]
    return commands


def harness_env(cfg: dict, *, base_url: str, model: str | None = None) -> dict[str, str]:
    """直接跑 harness（不经 Gym）时给它的环境：端点 + 模型名 + DSH_HOME。"""
    env = {str(k): str(v) for k, v in (harness_cfg(cfg).get("env") or {}).items()}
    env.setdefault("DSH_HOME", dsh_home(cfg))
    env.setdefault("OPENAI_BASE_URL", base_url)
    if model:
        env.setdefault("OPENAI_MODEL", model)
    return env


def gym_agent_overrides(
    cfg: dict, *, base_url: str | None = None, model: str | None = None
) -> list[str]:
    """Gym 侧：把 HarnessAgent 指到 dsh（字段名见 Gym 的 harness_agent/app.py::HarnessAgentConfig）。"""
    harness = harness_name(cfg)
    if not harness:
        return []
    root = "policy_model.responses_api_agents.harness_agent"
    over = [f"++{root}.agent={harness}"]
    agent_kwargs = dict(harness_cfg(cfg).get("agent_kwargs") or {})
    if base_url:
        agent_kwargs.setdefault("base_url", base_url)
    if model:
        agent_kwargs.setdefault("model", model)
    for key, value in agent_kwargs.items():
        over.append(f"++{root}.agent_kwargs.{key}={value}")
    return over


def command(cfg: dict, *, task: str | None = None, extra: list[str] | None = None) -> list[str]:
    """不经 Gym、直接跑 harness 的命令（本地/无 Gym 时的统一路径；profile 决定具体行为）。"""
    sec = harness_cfg(cfg)
    base = str(sec.get("command") or f"{harness_name(cfg)}")
    cmd = base.split()
    if sec.get("profile"):
        cmd += ["--profile", str(sec["profile"])]
    if task:
        cmd += [str(task)]
    cmd += [str(x) for x in extra or []]
    return cmd


def preflight(cfg: dict) -> dict:
    """报 harness 的可用性（外部依赖）：缺什么、怎么装——两段的预检都用它。"""
    name = harness_name(cfg)
    home = dsh_home(cfg)
    binary = shutil.which(name)
    home_exists = Path(home).is_dir()
    missing = []
    if not binary:
        missing.append(f"{name} 不在 PATH")
    if not home_exists:
        missing.append(f"DSH_HOME 目录不存在（{home}）")
    try:
        import importlib.util

        if importlib.util.find_spec("deepseek_harness") is None:
            missing.append(f"{SDK_PACKAGE} 未安装")
    except Exception:  # noqa: BLE001 - 探测失败就当没装
        missing.append(f"{SDK_PACKAGE} 未安装")
    return {
        "ok": not missing,
        "name": name,
        "home": home,
        "binary": binary,
        "missing": missing,
        "hint": (
            f"pip install {SDK_PACKAGE} && mkdir -p {home}（把 harness 的 profile 放进去；"
            "Gym 宿主用 `agent=dsh` 起它，端点指向本机的 vllm serve）"
        ),
    }
