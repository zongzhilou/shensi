#!/usr/bin/env python3
"""三段共用的路径与数据：轨迹归一、三种训练形态的渲染、AgentWorldBench 判分提示词。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # agentworld

_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent / "stage2_agentic"
if str(_AGENTIC) not in sys.path:
    sys.path.insert(0, str(_AGENTIC))

import world_model as wm  # noqa: E402

DOMAINS = wm.DOMAINS
OBS_MARKER = wm.OBS_MARKER


def wrap_observation(text: str) -> str:
    """训出来的世界模型要和评分口径一致：`**Environment Observation:**` + `<predicted_observation>`。"""
    return f"{OBS_MARKER}\n<predicted_observation>\n{text.strip()}\n</predicted_observation>"


def clip_turn(text: str, max_chars: int | None) -> str:
    """把一轮动作/观测掐到 `max_chars` 字符（头尾各留一半）。"""
    text = text.strip()
    if not max_chars or len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return f"{text[:half]}\n…（中间省略 {len(text) - 2 * half} 字）…\n{text[-half:]}"


def domain_of(row: dict, default: str = "terminal") -> str:
    task = str(row.get("task") or "")
    cand = task.split("/")[0] if "/" in task else task
    if cand in DOMAINS:
        return cand
    for key in ("domain", "subtask", "environment", "env"):
        if str(row.get(key) or "") in DOMAINS:
            return str(row[key])
    return default


def to_turns(row: dict) -> list[tuple[str, str]] | None:
    """各种轨迹形状 → [(动作, 观测)]；认不出来返回 None。"""
    traj = row.get("trajectory") or row.get("turns")
    if isinstance(traj, list) and traj:
        if isinstance(traj[0], dict):
            out = [
                (
                    str(t.get("action") or t.get("prompt") or ""),
                    str(t.get("observation") or t.get("response") or ""),
                )
                for t in traj
            ]
            out = [(a, o) for a, o in out if a.strip() and o.strip()]
            return out or None
        if isinstance(traj[0], (list, tuple)) and len(traj[0]) == 2:
            return [(str(a), str(o)) for a, o in traj]
    prompts, responses = row.get("prompt"), row.get("response")
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]
    if isinstance(prompts, list) and isinstance(responses, list) and prompts and responses:
        pairs = [
            (str(a), str(o))
            for a, o in zip(prompts, responses, strict=False)
            if str(a).strip() and str(o).strip()
        ]
        return pairs or None
    if row.get("action") and row.get("observation"):
        return [(str(row["action"]), str(row["observation"]))]
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, list) and msgs:
        pairs, pending = [], None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role, content = str(m.get("role") or ""), str(m.get("content") or "")
            if role == "assistant" and content.strip():
                pending = content
            elif role in ("user", "tool") and content.strip() and pending:
                pairs.append((pending, content))
                pending = None
        return pairs or None
    return None


def system_of(row: dict, domain: str) -> str:
    """上游 AgentWorldBench 用 system_str，我们自家轨迹用域提示词重建；口径一致才能比数。"""
    given = str(row.get("system_str") or row.get("system") or "").strip()
    if given:
        return given
    return wm.build_system_message(
        domain, row.get("mode") or "sim", row.get("spec"), row.get("task")
    )


def agentworld_job(row: dict) -> dict | None:
    """上游 AgentWorldBench 样本行：{task, system_str, prompt[], response[], turn_idx}。"""
    if (
        isinstance(row.get("prompt"), list)
        and isinstance(row.get("response"), list)
        and row["prompt"]
        and row["response"]
    ):
        return row
    return None


def lwm_input(job: dict) -> list[dict]:
    """跑世界模型用的 messages：历史轮照给，最后接当前轮（照 AgentWorldBench 的输入）。"""
    messages = []
    if job.get("system_str"):
        messages.append({"role": "system", "content": str(job["system_str"])})
    prompts, responses = job["prompt"], job["response"]
    turn = max(int(job.get("turn_idx") or 1) - 1, 0)
    for prompt, response in zip(prompts[:turn], responses[:turn], strict=False):
        messages.extend(
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)},
            ]
        )
    current = str(job.get("current_prompt") or (prompts[turn] if turn < len(prompts) else ""))
    messages.append({"role": "user", "content": current})
    return messages


def ground_truth_of(job: dict) -> str:
    responses = job["response"]
    turn = max(int(job.get("turn_idx") or 1) - 1, 0)
    return str(responses[turn]) if turn < len(responses) else ""


def judge_messages(
    job: dict, model_output: str, domain: str, system_prompt: str | None = None
) -> list[dict]:
    """判分提示词照 AgentWorldBench：历史上下文 + 当前轮 + 模拟输出 + 真值。"""
    from agentworld.eval.lwm_eval_utils import JUDGE_USER_PROMPT, clean_response_marker

    prompts, responses = job["prompt"], job["response"]
    turn = max(int(job.get("turn_idx") or 1) - 1, 0)
    context = "".join(
        f"{prompts[i]}\n{responses[i]}\n\n"
        for i in range(turn)
        if i < len(prompts) and i < len(responses)
    )
    context = f"# Context (Historical Interactions):\n\n{context}" if context else ""
    current = str(job.get("current_prompt") or (prompts[turn] if turn < len(prompts) else ""))
    user_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        world_model_input=f"# Current Turn:\n\n{current}",
        predicted_observation=f"**World Model Output (Simulated):**\n```\n{clean_response_marker(model_output, domain)}\n```",
        ground_truth=f"**Ground Truth (Real Output):**\n```\n{clean_response_marker(ground_truth_of(job), domain)}\n```",
    ).strip()
    if system_prompt is None:
        system_prompt = wm.load_judge_system_prompt(domain)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def render_cpt(domain: str, system: str, turns: list[tuple[str, str]]) -> str:
    """CPT 文档：一段交互轨迹的纯文本（让基座先熟悉这七个域里的动作与观测长什么样）。"""
    body = "\n\n".join(f"{action.strip()}\n{observation.strip()}" for action, observation in turns)
    return f"{system.strip()}\n\n# Interaction\n\n{body}"


def sft_messages(
    domain: str,
    system: str,
    turns: list[tuple[str, str]],
    max_history: int = 4,
    max_turn_chars: int | None = None,
) -> list[dict]:
    """SFT 行：给历史 + 一个动作，学「下一状态」；观测带标记，和评测口径对齐。"""
    out = []
    for i, (action, observation) in enumerate(turns):
        messages = [{"role": "system", "content": system}]
        for prev_action, prev_obs in turns[max(0, i - max_history) : i]:
            messages.append({"role": "user", "content": clip_turn(prev_action, max_turn_chars)})
            messages.append(
                {
                    "role": "assistant",
                    "content": wrap_observation(clip_turn(prev_obs, max_turn_chars)),
                }
            )
        messages.append({"role": "user", "content": clip_turn(action, max_turn_chars)})
        messages.append(
            {
                "role": "assistant",
                "content": wrap_observation(clip_turn(observation, max_turn_chars)),
            }
        )
        out.append(messages)
    return out


def rl_row(
    domain: str,
    system: str,
    turns: list[tuple[str, str]],
    index: int,
    max_history: int = 4,
    max_turn_chars: int | None = None,
) -> dict:
    """RL 行：prompt = 历史 + 当前动作，ground_truth = 真观测，verifier 交给世界模型裁判。"""
    action, observation = turns[-1]
    messages = [{"role": "system", "content": system}]
    for prev_action, prev_obs in turns[max(0, len(turns) - 1 - max_history) : -1]:
        messages.append({"role": "user", "content": clip_turn(prev_action, max_turn_chars)})
        messages.append(
            {"role": "assistant", "content": wrap_observation(clip_turn(prev_obs, max_turn_chars))}
        )
    messages.append({"role": "user", "content": clip_turn(action, max_turn_chars)})
    return {
        "prompt": messages,
        "data_source": f"world_model_{domain}",
        "agent": None,
        "verifier": {"type": "world_model_judge", "domain": domain},
        "reward_model": {"style": "world_model_judge", "ground_truth": observation.strip()},
        "extra_info": {
            "source": f"world_model_{domain}",
            "split": "train",
            "index": index,
            "domain": domain,
            "prompt_text": json.dumps(messages[:-1], ensure_ascii=False),
            "current_prompt": action.strip(),
        },
    }
