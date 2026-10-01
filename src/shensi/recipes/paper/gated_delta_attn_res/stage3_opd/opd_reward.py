"""OPD 段的 opd_reward.py 模块。"""

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


def reverse_kl_advantage(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    *,
    clip: float | None = None,
) -> float:
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"student/teacher 的 token 数不同（{len(student_logprobs)} vs {len(teacher_logprobs)}）；"
            "对齐失败时宁可直接失败，也不要把 KL 算错位置"
        )
    if not student_logprobs:
        raise ValueError("空序列：没有可打分的 token")
    # reward = -mean_t [logπ_s(a_t) - logπ_t(a_t)]：采样 token 上的 reverse KL 取负
    total = 0.0
    for s, t in zip(student_logprobs, teacher_logprobs):
        delta = float(s) - float(t)
        if clip is not None:
            delta = max(-clip, min(clip, delta))
        total += delta
    return -total / len(student_logprobs)


class OPDConfig:
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
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:400]
        raise RuntimeError(f"{url} 返回 {exc.code}：{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连不上 {url}：{exc}") from exc


def _echo_logprobs(
    base_url: str, model: str, text: str, timeout: float
) -> tuple[list[str], list[float]]:
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
    # 拼接处必须逐 token 一致：宁可报错，也不要把 KL 算到别的 token 上
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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    prompt_key = os.environ.get("OPD_PROMPT_KEY", "prompt")
    prompt = None
    if isinstance(extra_info, dict):
        prompt = extra_info.get(prompt_key)
    if prompt is None:
        prompt = kwargs.get("prompt") or (data_source if data_source not in ("opd", None) else None)
    if not prompt:
        raise RuntimeError(
            f"拿不到 prompt：extra_info[{prompt_key!r}] 为空。本目录的 data_prep 会把 prompt 写进"
            "这一列；用别的数据源时设 OPD_PROMPT_KEY。"
        )
    cfg = _CONFIG
    if (cfg.mode or "reverse_kl").lower() == "teacher_only":
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


_CONFIG = OPDConfig()


def set_config(cfg: OPDConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def _selftest() -> int:
    import math

    s = [math.log(0.7), math.log(0.5), math.log(0.9)]
    t = [math.log(0.5), math.log(0.5), math.log(0.3)]
    r = reverse_kl_advantage(s, t)
    expect = -((math.log(0.7) - math.log(0.5)) + 0.0 + (math.log(0.9) - math.log(0.3))) / 3
    print(f"  reverse_kl_advantage = {r:+.6f}（期望 {expect:+.6f}）")
    assert abs(r - expect) < 1e-12
    assert reverse_kl_advantage(s, s) == 0.0
    assert reverse_kl_advantage(s, t, clip=0.1) > r
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
