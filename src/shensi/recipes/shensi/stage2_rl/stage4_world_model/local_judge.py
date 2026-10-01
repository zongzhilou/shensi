#!/usr/bin/env python3
"""本机判分服务：CPU 上的小模型当裁判，不占显存、也不要第二个模型服务。"""

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from shensi import runtime  # noqa: F401

DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")

RUBRIC = (
    "Compare the two outputs below (simulated vs real).\n"
    "Give five ratings from 1 to 5 for: format, factuality, consistency, realism, quality.\n"
    "Answer with five digits separated by spaces, for example: 3 4 3 5 4\n"
    "Do not write any other words.\n\n"
    "SIMULATED:\n{simulated}\n\nREAL:\n{real}\n"
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
        simulated, _, real = user_text.partition("REAL:")
        messages = [
            {"role": "system", "content": "You are a strict evaluator. Output only five digits."},
            {
                "role": "user",
                "content": RUBRIC.format(simulated=simulated[-3000:], real=real[-3000:]),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def scores(self, payload: dict) -> dict:
        """一次判分：模型给五个数字就用它们，只给出一部分就按已给的均值补齐（并记解析率）。"""
        _STATS["calls"] += 1
        raw = self._ask(json.dumps(payload, ensure_ascii=False))
        nums = [int(x) for x in re.findall(r"\b([1-5])\b", raw)][:5]
        if len(nums) == 5:
            _STATS["parsed"] += 1
        elif nums:  # 部分解析：按已给分数求均值补齐，避免整条被 3.0 覆盖
            nums = nums + [round(sum(nums) / len(nums))] * (5 - len(nums))
        else:
            nums = [3] * 5
        self.last_raw = raw
        return {d: float(v) for d, v in zip(DIMENSIONS, nums)}

    def answer(self, payload: dict) -> str:
        body = {
            "scores": self.scores(payload),
            "strengths": ["本机 CPU 裁判（local_judge）"],
            "weaknesses": ["解析失败时按 3.0 顶，见服务日志的解析率"],
        }
        return (
            "<final_evaluation>\n" + json.dumps(body, ensure_ascii=False) + "\n</final_evaluation>"
        )


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
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
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
    print("[local_judge] 模型原始输出：", (judge.last_raw or "").replace("\n", " ")[:160])
    print("[local_judge] 组装输出：", raw.replace("\n", " "))
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
