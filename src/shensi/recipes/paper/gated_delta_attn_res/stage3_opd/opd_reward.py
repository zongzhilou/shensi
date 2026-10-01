#!/usr/bin/env python3
"""OPD 的 **RL 式** reward：`reward = −mean_t [logπ_student(a_t) − logπ_teacher(a_t)]`。

这就是 on-policy 蒸馏的 advantage 形式（Thinking Machines 的 OPD / MiniCPM5 的 OPD 口径）：
学生在**自己采样出来的 token** 上被 teacher 打分，逐 token 的 reverse KL 的负值当奖励 ——
GRPO 组内归一化之后，advantage 正的方向是"比 teacher 更自信的 token"，负的方向是"teacher 更
自信而我们没跟上"的 token。与静态 KD（`train/reverse_kl.py`，在缓存 logits 上做 KL）的分工：

* 静态 KD：teacher 的 top-k 概率落盘，学生按 token 学（本目录 ③ 步，mcore 原生）；
* **本模块**：走 RL 循环（verl，`reward.custom_reward_function`），teacher 在线打分 ——
  学生自己 rollout 的分布进了循环，OPD 的 "on-policy" 才真正成立。

用法（与 stage2_rl 的臂同构：verl + mcore actor，只是 reward 换成这里）::

    export OPD_STUDENT_URL=http://127.0.0.1:8001/v1      # 学生的 vLLM 端点（= rollout 引擎）
    export OPD_TEACHER_URL=http://127.0.0.1:8002/v1      # 该方向 teacher 的 vLLM 端点
    python -m shensi.recipes.paper.gated_delta_attn_res.stage3_opd.opd_reward --selftest
    # 起训（把 reward 指到本文件；其余与 stage2_rl 的臂一致，见 config/opd_rl.yaml）
    cd stage2_rl/stage2_math && python train.py --set reward.custom_reward_function.path=$PWD/../../stage3_opd/opd_reward.py

两个端点都读 `choices[0].logprobs.token_logprobs`（OpenAI 兼容的 `/v1/completions`，
`echo=true, max_tokens=0, logprobs=1`）。**对齐方式**：先问"prompt 单独"拿到 prompt 的 token
数与 token 串，再问"prompt+response"，要求前 P 个 token **逐个相同**（tokenizer 在拼接处不会
改主意），不满足就报错而不是猜 —— 猜错会把 KL 算到别的 token 上。逐条带内存缓存，
同一 (prompt, response, teacher) 只打一次端点。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

__all__ = [
    "OPDConfig",
    "completion_token_logprobs",
    "compute_score",
    "reverse_kl_advantage",
]


# ---------------------------------------------------------------------------
# 纯函数：KL → reward
# ---------------------------------------------------------------------------
def reverse_kl_advantage(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    *,
    clip: float | None = None,
) -> float:
    """`−mean_t [logπ_s(a_t) − logπ_t(a_t)]` —— 采样 token 上的 reverse KL 的负值。

    Args:
        student_logprobs: 学生在**采样到的 token** 上的 log 概率（自然对数）。
        teacher_logprobs: 同一批 token 上 teacher 的 log 概率。
        clip: 每 token 的差值限幅（防单点爆炸；``None`` 不限）。

    Returns:
        reward 标量：0 表示两边对这批 token 一样自信；正值表示学生更自信（KL 更小）。

    Raises:
        ValueError: 两个序列长度不一致或为空 —— 静默截断会把 KL 算到别的 token 上。

    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"student/teacher 的 token 数不同（{len(student_logprobs)} vs {len(teacher_logprobs)}）；"
            "对齐失败时宁可直接失败，也不要把 KL 算错位置"
        )
    if not student_logprobs:
        raise ValueError("空序列：没有可打分的 token")
    total = 0.0
    for s, t in zip(student_logprobs, teacher_logprobs):
        delta = float(s) - float(t)
        if clip is not None:
            delta = max(-clip, min(clip, delta))
        total += delta
    return -total / len(student_logprobs)


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
class OPDConfig:
    """从环境读端点配置（`OPD_*`），并把缓存挂在实例上。"""

    def __init__(
        self,
        student_url: str | None = None,
        teacher_url: str | None = None,
        *,
        student_model: str | None = None,
        teacher_model: str | None = None,
        clip: float | None = None,
        timeout: float = 120.0,
    ):
        self.student_url = (student_url or os.environ.get("OPD_STUDENT_URL") or "").rstrip("/")
        self.teacher_url = (teacher_url or os.environ.get("OPD_TEACHER_URL") or "").rstrip("/")
        self.student_model = student_model or os.environ.get("OPD_STUDENT_MODEL") or "student"
        self.teacher_model = teacher_model or os.environ.get("OPD_TEACHER_MODEL") or "teacher"
        env_clip = os.environ.get("OPD_KL_CLIP")
        self.clip = clip if clip is not None else (float(env_clip) if env_clip else None)
        self.timeout = timeout
        self.mode = (os.environ.get("OPD_MODE") or "reverse_kl").lower()
        if self.mode not in ("reverse_kl", "teacher_only"):
            raise ValueError(f"OPD_MODE={self.mode!r} 只能是 reverse_kl / teacher_only")
        self._cache: dict[tuple, list[float]] = {}
        self.hits = 0
        self.misses = 0

    def require(self, role: str) -> str:
        url = self.student_url if role == "student" else self.teacher_url
        if not url:
            raise RuntimeError(
                f"OPD 的 {role} 端点没配置：设 OPD_{role.upper()}_URL=http://host:port/v1"
                "（学生 = rollout 引擎那个 vLLM；teacher = 该方向 teacher 的 vLLM）"
            )
        return url


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 端点的错误体里通常有真正的原因
        body = exc.read().decode("utf-8", errors="ignore")[:400]
        raise RuntimeError(f"{url} 返回 {exc.code}：{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连不上 {url}：{exc}") from exc


def _echo_logprobs(
    base_url: str, model: str, text: str, timeout: float
) -> tuple[list[str], list[float]]:
    """`echo=true, max_tokens=0, logprobs=1` 拿整段文本的逐 token logprob。"""
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0.0,
    }
    body = _post_json(f"{base_url}/completions", payload, timeout)
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens")
    values = lp.get("token_logprobs")
    if not tokens or values is None:
        raise RuntimeError(
            f"{base_url}/completions 没返回 token_logprobs（拿到 {list(choice)}）；"
            "该端点需要支持 echo=true + logprobs=1"
        )
    return list(tokens), [float(v) for v in values]


def completion_token_logprobs(cfg: OPDConfig, role: str, prompt: str, response: str) -> list[float]:
    """Response 那一段的逐 token logprob（带缓存；对齐失败直接报错）。"""
    if not response:
        raise ValueError("空 response：OPD 的 reward 要在学生采样出来的 token 上算")
    base = cfg.require(role)
    model = cfg.student_model if role == "student" else cfg.teacher_model
    key = (role, prompt, response)
    cached = cfg._cache.get(key)
    if cached is not None:
        cfg.hits += 1
        return cached
    cfg.misses += 1
    p_tokens, _ = _echo_logprobs(base, model, prompt, cfg.timeout)
    all_tokens, all_lp = _echo_logprobs(base, model, prompt + response, cfg.timeout)
    p = len(p_tokens)
    if p >= len(all_tokens):
        raise RuntimeError(
            f"{role} 端点：prompt+response 的 token 数（{len(all_tokens)}）不比 prompt 多（{p}）；"
            "response 是空的或端点截断了输入"
        )
    if all_tokens[:p] != p_tokens:
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(all_tokens[:p], p_tokens)) if a != b), None
        )
        raise RuntimeError(
            f"{role} 端点：拼接处在第 {first_bad} 个 token 上分叉"
            f"（prompt 单独={p_tokens[max(0, (first_bad or 0) - 1) : (first_bad or 0) + 2]}，"
            f"拼接后={all_tokens[max(0, (first_bad or 0) - 1) : (first_bad or 0) + 2]}）。"
            "tokenizer 在边界处改主意了，不能按长度切 —— 换用不引入边界的拼法或先自查 tokenize"
        )
    out = all_lp[p:]
    cfg._cache[key] = out
    return out


# ---------------------------------------------------------------------------
# verl 的 custom_reward_function 入口
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Verl 的 reward 入口：`−mean_t [logπ_s − logπ_t]`（采样 token 上的 reverse KL 取负）。

    prompt 从 `extra_info` 里取（键名 `OPD_PROMPT_KEY`，默认 ``"prompt"``；本目录的 data_prep
    会把 prompt 写进这一列）。日志里会打印端点命中/未命中计数（缓存生效情况）。
    """
    prompt_key = os.environ.get("OPD_PROMPT_KEY", "prompt")
    prompt = None
    if isinstance(extra_info, dict):
        prompt = extra_info.get(prompt_key)
    if prompt is None:
        # data_source 在本目录的约定里就是 prompt（data_prep 写成了 "opd" 时用 extra_info）
        prompt = kwargs.get("prompt") or (data_source if data_source not in ("opd", None) else None)
    if not prompt:
        raise RuntimeError(
            f"拿不到 prompt：extra_info[{prompt_key!r}] 为空。本目录的 data_prep 会把 prompt 写进"
            "这一列；用别的数据源时设 OPD_PROMPT_KEY。"
        )
    cfg = _CONFIG
    if (cfg.mode or "reverse_kl").lower() == "teacher_only":
        # 显式降级：拿不到学生端点（或有意只跑 teacher 打分）时，reward = mean logπ_t。
        # 注意它**不是**完整的 reverse-KL advantage（少了 −logπ_s 那一项）；GRPO 的组内归一化
        # 会把同一 prompt 组内共享的偏移消掉，所以组内比较仍有意义，但绝不要说成 OPD。
        teacher_lp = completion_token_logprobs(cfg, "teacher", prompt, solution_str)
        reward = sum(teacher_lp) / len(teacher_lp)
    else:
        student_lp = completion_token_logprobs(cfg, "student", prompt, solution_str)
        teacher_lp = completion_token_logprobs(cfg, "teacher", prompt, solution_str)
        reward = reverse_kl_advantage(student_lp, teacher_lp, clip=cfg.clip)
    if os.environ.get("OPD_VERBOSE", ""):
        print(
            f"[opd_reward] {len(student_lp)} tok | −KL={reward:+.4f} "
            f"| cache hits={cfg.hits} misses={cfg.misses}",
            flush=True,
        )
    return reward


#: 进程级配置（端点 + 缓存）；`main` 的 selftest 会替换它
_CONFIG = OPDConfig()


def set_config(cfg: OPDConfig) -> None:
    """换一份配置（测试与多 teacher 路由用）。"""
    global _CONFIG
    _CONFIG = cfg


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _selftest() -> int:
    """没有端点时也能自检的部分：解析与 KL 数学。"""
    import math

    s = [math.log(0.7), math.log(0.5), math.log(0.9)]
    t = [math.log(0.5), math.log(0.5), math.log(0.3)]
    r = reverse_kl_advantage(s, t)
    expect = (
        -((math.log(0.7) - math.log(0.5)) + 0.0 + (math.log(0.9) - math.log(0.3))) / 3
    )  # −mean(log0.7−log0.5, 0, log0.9−log0.3)
    print(f"  reverse_kl_advantage = {r:+.6f}（期望 {expect:+.6f}）")
    assert abs(r - expect) < 1e-12
    assert reverse_kl_advantage(s, s) == 0.0
    assert reverse_kl_advantage(s, t, clip=0.1) > r  # 限幅把大项压小 → 奖励更高
    try:
        reverse_kl_advantage(s, t[:2])
        raise AssertionError("长度不一致应当报错")
    except ValueError:
        pass
    print("  [OK] KL→reward、限幅、长度校验")
    for role in ("student", "teacher"):
        try:
            OPDConfig(student_url="", teacher_url="").require(role)
            raise AssertionError("缺端点应当报错")
        except RuntimeError as exc:
            assert role.upper() in str(exc)
    print("  [OK] 端点缺失时报错信息包含变量名")
    assert OPDConfig(student_url="", teacher_url="").mode == "reverse_kl"
    os.environ["OPD_MODE"] = "teacher_only"
    assert OPDConfig().mode == "teacher_only"
    del os.environ["OPD_MODE"]
    print("  [OK] OPD_MODE 解析（reverse_kl / teacher_only）")
    print(f"  [OK] 自检完成（{time.strftime('%H:%M:%S')}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="OPD 的 RL 式 reward（reverse-KL advantage）")
    ap.add_argument("--selftest", action="store_true", help="不连端点，自检 KL 数学与配置校验")
    ap.add_argument("--prompt", default=None, help="打一条：prompt")
    ap.add_argument("--response", default=None, help="打一条：response")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.prompt and args.response:
        print(f"{compute_score('opd', args.response, None, {'prompt': args.prompt}):+.6f}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
