#!/usr/bin/env python3
"""本机判分服务：CPU 上的小模型当裁判，不占显存、也不要第二个模型服务。

AgentWorldBench 的裁判提示词要求输出 `<final_evaluation>` JSON（format / factuality /
consistency / realism / quality 五维，1~5）。小模型直接产 JSON 不稳，这里走"逐维打分"：
把轨迹与真值压进一段提示，让模型给出五个 1~5 的整数，解析成功就按官方格式组装；
解析失败记一次 fallback（用 3.0 顶），并把解析率打出来（解析率低说明该换更大的裁判模型）。

用法（判分端点，默认端口 8000）：

    python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.local_judge \
        --model-dir $SHENSI_FS/models/SmolLM2-360M-Instruct --port 8000
    # 离线自检（不服务，直接打一次判分并打印解析结果）
    python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.local_judge --check
"""

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from shensi import runtime  # noqa: F401

DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")

RUBRIC = (
    "You are grading a simulated environment observation against the real one.\n"
    "Rate 1-5 on five axes, in this exact order: format, factuality, consistency, realism, quality.\n"
    "Reply with five integers separated by spaces and nothing else.\n\n"
    "{payload}"
)

_STATS = {"calls": 0, "parsed": 0}


class LocalJudge:
    """CPU 上的 HF 因果语言模型裁判（按需加载，只跑 generate）。"""

    def __init__(self, model_dir: str, max_new_tokens: int = 24):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32, device_map=None
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.name = model_dir.rstrip("/").split("/")[-1]

    def _ask(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a strict evaluator. Output only five integers."},
            {"role": "user", "content": RUBRIC.format(payload=user_text[-6000:])},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def scores(self, payload: dict) -> dict:
        """一次判分：先让模型给五个整数，失败就用 3.0 顶（并记一次 fallback）。"""
        _STATS["calls"] += 1
        text = json.dumps(payload, ensure_ascii=False)
        raw = self._ask(text)
        nums = [int(x) for x in re.findall(r"\b([1-5])\b", raw)][:5]
        if len(nums) == 5:
            _STATS["parsed"] += 1
            return {d: float(v) for d, v in zip(DIMENSIONS, nums)}
        return {d: 3.0 for d in DIMENSIONS}

    def answer(self, payload: dict) -> str:
        body = {
            "scores": self.scores(payload),
            "strengths": ["本机 CPU 裁判（local_judge）"],
            "weaknesses": ["解析失败时按 3.0 顶，见服务日志的解析率"],
        }
        return "<final_evaluation>\n" + json.dumps(body, ensure_ascii=False) + "\n</final_evaluation>"


def _handler_for(judge: LocalJudge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shensi-local-judge/0.1"

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/v1/models"):
                self._send(200, {"data": [{"id": judge.name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "bad json"})
                return
            user = ""
            for m in reversed(req.get("messages") or []):
                if m.get("role") == "user":
                    user = str(m.get("content") or "")
                    break
            content = judge.answer({"user": user})
            pct = 100.0 * _STATS["parsed"] / max(_STATS["calls"], 1)
            print(f"[local_judge] 第 {_STATS['calls']} 次（解析率 {pct:.0f}%）", flush=True)
            self._send(
                200,
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "model": judge.name,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}
                    ],
                },
            )

        def log_message(self, *args) -> None:  # 静音默认访问日志
            return

    return Handler


def check(model_dir: str) -> int:
    """离线自检：打一次合成判分，打印解析结果（1~5 五维）。"""
    judge = LocalJudge(model_dir)
    payload = {
        "user": "**World Model Output (Simulated):**\n```\n$ ls\nfile.txt\n```\n"
                "**Ground Truth (Real Output):**\n```\n$ ls\nfile.txt\n```"
    }
    raw = judge.answer(payload)
    print("[local_judge] 模型：", judge.name)
    print("[local_judge] 输出：", raw.replace("\n", " "))
    ok = all(f'"{d}"' in raw for d in DIMENSIONS)
    print("[local_judge] 五维键齐全：", ok)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="本机 CPU 判分服务（AgentWorldBench 五维）")
    ap.add_argument("--model-dir", default=None, help="本地 HF 模型目录（CPU 上跑）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--check", action="store_true", help="只做一次离线自检，不服务")
    args = ap.parse_args()
    if not args.model_dir:
        raise SystemExit("给一个本地模型目录：--model-dir $SHENSI_FS/models/<小模型>")
    if args.check:
        return check(args.model_dir)
    judge = LocalJudge(args.model_dir)
    srv = ThreadingHTTPServer((args.host, args.port), _handler_for(judge))
    print(f"[local_judge] {judge.name} 已在 http://{args.host}:{args.port}（CPU）", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
