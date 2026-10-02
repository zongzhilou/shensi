"""OPD 奖励的单测：解析解、缓存与报错路径。"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/home/louzo/code/shensi/shensi/src")

from shensi.recipes.paper.gated_delta_attn_res.stage3_opd import opd_reward  # noqa: E402

OK = True


def check(name, ok, detail=""):
    global OK
    OK &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56} {detail}")


class _Stub(BaseHTTPRequestHandler):
    table: dict = {}
    calls = 0

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode("utf-8"))
        _Stub.calls += 1
        text = payload["prompt"]
        tokens = text.split()
        role_bias = 0.5 if payload.get("model") == "teacher" else 0.0
        lps = [-(0.1 * (len(tok) + role_bias)) for tok in tokens]
        body = {
            "choices": [
                {
                    "logprobs": {"tokens": tokens, "token_logprobs": lps},
                    "text": text,
                }
            ]
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def _serve() -> tuple[str, HTTPServer]:
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/v1", srv


def main() -> int:
    base, srv = _serve()
    cfg = opd_reward.OPDConfig(student_url=base, teacher_url=base)
    prompt, response = "hello world", " again more"

    _Stub.calls = 0
    s_lp = opd_reward.completion_token_logprobs(cfg, "student", prompt, response)
    t_lp = opd_reward.completion_token_logprobs(cfg, "teacher", prompt, response)
    check("response 段取到的 token 数正确", len(s_lp) == len(response.split()), f"{len(s_lp)} 个")
    expect_s = [-(0.1 * len(tok)) for tok in response.split()]
    check(
        "student 的解析正确（桩函数可手算）",
        all(abs(a - b) < 1e-12 for a, b in zip(s_lp, expect_s)),
        "",
    )
    expect_t = [-(0.1 * (len(tok) + 0.5)) for tok in response.split()]
    check("teacher 的解析正确", all(abs(a - b) < 1e-12 for a, b in zip(t_lp, expect_t)), "")
    r = opd_reward.reverse_kl_advantage(s_lp, t_lp)
    check("reward = −mean(logp_s − logp_t)（手算）", abs(r - (-(0.05))) < 1e-12, f"{r:+.6f}")

    cfg2 = opd_reward.OPDConfig(student_url=base, teacher_url=base)
    _Stub.calls = 0
    for _ in range(3):
        opd_reward.completion_token_logprobs(cfg2, "student", prompt, response)
    check(
        "同一 Query 只打一次端点（缓存生效）",
        _Stub.calls == 2,
        f"端点调用 {_Stub.calls} 次（prompt + 拼接）",
    )

    _Stub.calls = 0
    opd_reward.set_config(opd_reward.OPDConfig(student_url=base, teacher_url=base))
    score = opd_reward.compute_score("opd", response, None, {"prompt": prompt})
    check("compute_score 走通并给出 reward", abs(score - (-0.05)) < 1e-12, f"{score:+.6f}")

    class _Shifted(_Stub):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            tokens = payload["prompt"].split()
            if len(tokens) > 2:
                tokens[1] = "SPLIT"
            body = {
                "choices": [
                    {
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": [-0.1 for _ in tokens],
                        }
                    }
                ]
            }
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv2 = HTTPServer(("127.0.0.1", 0), _Shifted)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    cfg3 = opd_reward.OPDConfig(
        student_url=f"http://127.0.0.1:{srv2.server_port}/v1",
        teacher_url=f"http://127.0.0.1:{srv2.server_port}/v1",
    )
    try:
        opd_reward.completion_token_logprobs(cfg3, "student", "a b c d", "e f")
        check("拼接处分叉时报错（不猜对齐）", False, "没有报错")
    except RuntimeError as exc:
        check("拼接处分叉时报错（不猜对齐）", "分叉" in str(exc), str(exc)[:60])

    try:
        opd_reward.compute_score("opd", "", None, {"prompt": "x"})
        check("空 response 报错", False, "没有报错")
    except ValueError as exc:
        check("空 response 报错", "空 response" in str(exc), str(exc)[:40])
    try:
        opd_reward.set_config(opd_reward.OPDConfig(student_url="", teacher_url=""))
        opd_reward.compute_score("opd", "x y", None, {"prompt": "p"})
        check("缺端点的报错信息可照做", False, "没有报错")
    except RuntimeError as exc:
        check("缺端点的报错信息可照做", "OPD_STUDENT_URL" in str(exc), str(exc)[:60])
    finally:
        opd_reward.set_config(opd_reward.OPDConfig())

    check("--selftest 返回 0", opd_reward.main(["--selftest"]) == 0, "")

    srv.shutdown()
    srv2.shutdown()
    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
