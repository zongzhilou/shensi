"""桩判分端点：一个只靠标准库、跑在 CPU 上的 OpenAI 兼容小服务，按 AgentWorldBench 的
判分输出格式返回固定的（或按输入伪随机的）五维分数。

用途：世界模型的 RL 段奖励要一个 LLM 裁判（`reward.py` 走 `SHENSI_JUDGE_URL` /
`SHENSI_WORLD_MODEL_URL`）。真裁判是另一个模型服务，本机 16G 单卡上和 actor + rollout 挤不下
（会把 WSL 的 GPU 驱动压爆，见 stage 的 README）；用这个桩可以**不占 GPU** 把
「rollout → 判分 → 优势 → actor 更新」这条链路完整跑通，分数本身没有意义。

判分格式对齐 `agentworld/eval/lwm_eval_utils`：内容里要有
`<final_evaluation>{"scores": {...}}</final_evaluation>`，`total_score` 由解析器按各维均值算。

用法：
    python -m shensi.recipes.shensi.stage2_rl.stage4_world_model.stub_judge --port 8000
    # 另开一个终端：
    SHENSI_WORLD_MODEL_URL=http://127.0.0.1:8000/v1 SHENSI_WORLD_MODEL=stub-judge \
      python train.py --step rl --profile debug --data-dir $SHENSI_FS/shensi/data/stage2_world_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")


def _scores_for(payload: dict, mode: str, fixed: float) -> dict[str, float]:
    """按模式给五维分数：fixed 全给同一个值；hash 按请求内容取伪随机（GRPO 需要组内有区分度）。"""
    if mode == "fixed":
        return {d: float(fixed) for d in DIMENSIONS}
    seed = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).digest()
    return {d: float(seed[i] % 6) for i, d in enumerate(DIMENSIONS)}  # 0~5


def _judge_content(payload: dict, mode: str, fixed: float) -> str:
    scores = _scores_for(payload, mode, fixed)
    body = {
        "scores": scores,
        "strengths": ["（桩判分）链路连通性"],
        "weaknesses": ["（桩判分）分数与预测内容无关"],
    }
    return "<final_evaluation>\n" + json.dumps(body, ensure_ascii=False) + "\n</final_evaluation>"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "shensi-stub-judge/0.1"

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802  （http.server 的接口）
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(
                200, {"object": "list", "data": [{"id": self.server.model, "object": "model"}]}
            )
        elif self.path.rstrip("/").endswith(("/health", "/ready")):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {"raw": "unparsable"}
        content = _judge_content(payload, self.server.mode, self.server.fixed)
        self._send(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": self.server.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, fmt: str, *args) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)


def serve(host: str, port: int, model: str, mode: str, fixed: float, verbose: bool = False):
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.model = model
    httpd.mode = mode
    httpd.fixed = fixed
    httpd.verbose = verbose
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi 世界模型 RL 的桩判分端点（CPU、标准库）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--model", default="stub-judge", help="返回的模型名（与 SHENSI_WORLD_MODEL 对上）"
    )
    ap.add_argument(
        "--mode",
        default="hash",
        choices=("hash", "fixed"),
        help="hash：按请求内容伪随机给 0~5（组内有区分度）；fixed：全给 --score",
    )
    ap.add_argument("--score", type=float, default=3.0, help="--mode fixed 时的分数")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    httpd = serve(args.host, args.port, args.model, args.mode, args.score, args.verbose)
    print(
        f"[stub-judge] 起在 http://{args.host}:{args.port}/v1（model={args.model} mode={args.mode}）",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[stub-judge] 停", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
