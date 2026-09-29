#!/usr/bin/env python3
# 世界模型环境后端：把「语言世界模型」（Qwen-AgentWorld 口径）当成 agentic RL 的环境，观测由世界模型预测（Sim RL）。
# 三种口径对应 AgentWorld 报告的三类用法：sim 直接模拟环境、control 注入扰动、fiction 虚构世界。
# 用法：`python world_model.py check` 离线自测（带假世界模型）；`python world_model.py serve` 起 HTTP 环境服务。

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agentworld 在 stage2_rl/ 下
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agentworld 就在 stage2_rl/ 下

DOMAINS = ("terminal", "swe", "search", "mcp", "android", "web", "os")
MODES = ("sim", "control", "fiction")
OBS_MARKER = "**Environment Observation:**"
DONE_MARKER = "[TASK_DONE]"

FICTION_PREAMBLE = (
    "This simulation takes place in a fictional world that does not exist in the real world. "
    "Follow the world specification below exactly and stay consistent with it across turns:"
)
CONTROL_PREAMBLE = (
    "Additional environment settings for this simulation. They apply to every action and must be followed exactly:"
)


def _aw_root() -> Path:
    import agentworld as aw

    return Path(aw.__file__).parent


def _task_config(domain: str) -> dict:
    from agentworld.eval.lwm_eval_utils import TASK_CONFIGS

    if domain not in TASK_CONFIGS:
        raise SystemExit(f"[world_model] 不认识的域 {domain}，可选：{list(TASK_CONFIGS)}")
    return TASK_CONFIGS[domain]


def load_system_prompt(domain: str) -> str:
    rel = _task_config(domain)["system_prompt_path"]
    return (_aw_root() / rel).read_text(encoding="utf-8")


def load_judge_system_prompt(domain: str) -> str:
    rel = _task_config(domain)["judge_system_prompt_path"]
    return (_aw_root() / rel).read_text(encoding="utf-8")


def build_system_message(domain: str, mode: str = "sim", spec: dict | None = None, task: str | None = None) -> str:
    """世界模型的 system：先写本轮的模拟口径（扰动 / 虚构世界），再接该域的系统提示词。"""
    if mode not in MODES:
        raise SystemExit(f"[world_model] 不认识的口径 {mode}，可选：{list(MODES)}")
    spec = spec or {}
    parts = []
    if mode == "fiction":
        world = str(spec.get("world") or "").strip()
        if not world:
            raise SystemExit("[world_model] fiction 口径要在 spec 里给 world")
        parts.append(f"{FICTION_PREAMBLE}\n{world}")
    elif mode == "control":
        items = spec.get("perturbations") or []
        if isinstance(items, str):
            items = [items]
        items = [str(x) for x in items if str(x).strip()]
        if not items:
            raise SystemExit("[world_model] control 口径要在 spec 里给 perturbations")
        parts.append(CONTROL_PREAMBLE + "\n" + "\n".join(f"- {x}" for x in items))
    parts.append(load_system_prompt(domain))
    if task:
        parts.append(f"# Task\n{task}")
    return "\n\n".join(parts)


def format_action(name: str = "act", arguments=None, text: str | None = None) -> str:
    """动作渲染成世界模型认识的文本：首行 `Action: <工具名>`，随后逐行 `Key: value`。"""
    head = f"Action: {name or 'act'}"
    if text:
        return f"{head}\n{text}"
    if not arguments:
        return head
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{head}\n{arguments}"
    if not isinstance(arguments, dict):
        return f"{head}\n{arguments}"
    return "\n".join([head, *(f"{str(k).capitalize()}: {v}" for k, v in arguments.items())])


def parse_observation(raw: str, domain: str) -> str:
    from agentworld.eval.lwm_eval_utils import clean_response_marker, parse_model_output

    cfg = _task_config(domain)
    text = parse_model_output(raw, cfg.get("response_tag", "predicted_observation"))
    return clean_response_marker(text, domain)


class WorldModelClient:
    """世界模型的 OpenAI 兼容客户端（vLLM/SGLang 起 Qwen-AgentWorld，或我们自训的 shensi-world）。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ):
        self.base_url = (base_url or os.environ.get("SHENSI_WORLD_MODEL_URL") or "http://127.0.0.1:8000/v1").rstrip(
            "/"
        )
        self.model = model or os.environ.get("SHENSI_WORLD_MODEL") or "world-model"
        self.api_key = api_key or os.environ.get("SHENSI_WORLD_MODEL_KEY") or "EMPTY"
        self.timeout, self.temperature, self.top_p, self.max_tokens = timeout, temperature, top_p, max_tokens

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise SystemExit(f"[world_model] 连不上世界模型 {self.base_url}：{e}") from e
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content") or msg.get("reasoning_content") or ""

    def predict(self, messages: list[dict], domain: str) -> tuple[str, str]:
        raw = self.chat(messages)
        return parse_observation(raw, domain), raw


class WorldModelEnv:
    """一段 reset→step 的轨迹：每步动作都问一次世界模型，历史（动作、观测）逐轮累积。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        domain: str = "terminal",
        mode: str = "sim",
        spec: dict | None = None,
        task: str | None = None,
        max_turns: int = 8,
        client=None,
        done_marker: str = DONE_MARKER,
    ):
        self.client = client or WorldModelClient(base_url, model=model)
        self.domain, self.mode, self.spec = domain, mode, spec or {}
        self.max_turns, self.done_marker = int(max_turns), done_marker
        self.system = build_system_message(domain, mode, self.spec, task)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]
        self.turn, self.done, self.history = 0, False, []

    def reset(self, task: str | None = None, spec: dict | None = None) -> dict:
        if task is not None or spec is not None:
            self.spec = spec if spec is not None else self.spec
            self.system = build_system_message(self.domain, self.mode, self.spec, task)
        self.messages = [{"role": "system", "content": self.system}]
        self.turn, self.done, self.history = 0, False, []
        return self.state()

    def step(self, name: str = "act", arguments=None, text: str | None = None, action: str | None = None) -> dict:
        if self.done:
            return self.state()
        action_text = action or format_action(name, arguments, text)
        self.messages.append({"role": "user", "content": action_text})
        observation, raw = self.client.predict(self.messages, self.domain)
        self.messages.append({"role": "assistant", "content": raw})
        self.turn += 1
        self.done = (self.done_marker in observation) or self.turn >= self.max_turns
        self.history.append({"action": action_text, "observation": observation})
        return self.state()

    def state(self) -> dict:
        return {
            "domain": self.domain,
            "mode": self.mode,
            "turn": self.turn,
            "done": self.done,
            "max_turns": self.max_turns,
            "observation": self.history[-1]["observation"] if self.history else "",
        }


SESSIONS: dict[str, WorldModelEnv] = {}
SESSIONS_LOCK = threading.Lock()


class _EnvHandler(BaseHTTPRequestHandler):
    server_version = "shensi-world-model-env/1"

    def log_message(self, *args):  # 别把每个请求打到 stderr
        pass

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("health"):
            self._send({"ok": True, "sessions": len(SESSIONS), "domains": list(DOMAINS)})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        try:
            req = self._body()
            path = self.path.rstrip("/")
            if path.endswith("reset"):
                env = WorldModelEnv(
                    base_url=req.get("base_url"),
                    model=req.get("model"),
                    domain=req.get("domain", "terminal"),
                    mode=req.get("mode", "sim"),
                    spec=req.get("spec"),
                    task=req.get("task"),
                    max_turns=int(req.get("max_turns", 8)),
                )
                sid = uuid.uuid4().hex
                with SESSIONS_LOCK:
                    SESSIONS[sid] = env
                self._send({"session": sid, **env.state()})
            elif path.endswith("step"):
                with SESSIONS_LOCK:
                    env = SESSIONS.get(str(req.get("session")))
                if env is None:
                    self._send({"error": "unknown session"}, 404)
                    return
                self._send(
                    env.step(
                        name=req.get("name", "act"),
                        arguments=req.get("arguments"),
                        text=req.get("text"),
                        action=req.get("action"),
                    )
                )
            else:
                self._send({"error": "not found"}, 404)
        except SystemExit as e:
            self._send({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 9000) -> None:
    """HTTP 环境服务：`/health`、`/reset`、`/step`，harness（Gym / dsh）按 JSON 调用即可。"""
    srv = ThreadingHTTPServer((host, int(port)), _EnvHandler)
    print(f"[world_model] env server http://{host}:{srv.server_address[1]}（/health /reset /step）", flush=True)
    srv.serve_forever()


class StubWorldModel:
    """自测用的假世界模型：OpenAI 兼容端点，把最后一个动作回显成观测，并记录收到的请求。"""

    def __init__(self):
        self.seen: list[dict] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.seen.append(req)
                user_turns = [m for m in req.get("messages", []) if m.get("role") == "user"]
                last = user_turns[-1]["content"] if user_turns else ""
                tail = last.splitlines()[-1]
                body_text = f"$ {tail.partition(': ')[2] or tail}\nfile1 file2"
                if "submit" in last:
                    body_text += f"\n{DONE_MARKER}"
                content = f"{OBS_MARKER}\n<predicted_observation>\n{body_text}\n</predicted_observation>"
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


JUDGE_SAMPLE = (
    "<think>先看格式再看事实</think>\n"
    "<final_evaluation>\n"
    "```json\n"
    '{"strengths": ["命令回显与提示符一致"], "weaknesses": ["少了一个空行"], '
    '"scores": {"format": 4, "factuality": "5", "consistency": 4.0, "realism": 3, "quality": 4}}\n'
    "```\n"
    "</final_evaluation>"
)


def _tool_gate(stub, gate, tmp: Path) -> None:
    """工具这条腿：verl 不在就跳过（env 后端本身不依赖 verl）。"""
    import asyncio

    try:
        from shensi.recipes.shensi.stage2_rl.stage2_agentic.world_model_tool import WorldModelTool
    except ImportError as e:
        print(f"  [跳过] G14 Sim RL 工具（没装 verl）：{e}", flush=True)
        return

    tool = WorldModelTool(
        {
            "type": "native",
            "base_url": stub.base_url,
            "domain": "terminal",
            "mode": "sim",
            "action_name": "execute_bash",
            "max_turns": 3,
            "dump_dir": str(tmp / "traj"),
        },
        None,
    )
    iid, ready = asyncio.run(tool.create(create_kwargs={"domain": "swe", "task": "把测试跑绿"}))
    text, reward, metrics = asyncio.run(tool.execute(iid, {"action": "pytest -q"}))
    asyncio.run(tool.release(iid))
    dumped = list((tmp / "traj").glob("*.json"))
    gate(
        "G14 Sim RL 工具：按数据行覆盖域 + 观测 + 轨迹落盘",
        ready.text.endswith("domain=swe mode=sim")
        and text.text.startswith("$ pytest -q")
        and metrics["world_model_turns"] == 1
        and len(dumped) == 1
        and json.loads(dumped[0].read_text(encoding="utf-8"))["trajectory"][0]["action"].endswith("pytest -q"),
        f"观测 {text.text!r} 轨迹 {dumped[0].name if dumped else '无'}",
    )


def check() -> int:
    """离线自测：不需要 GPU、不需要真世界模型（用 StubWorldModel 起假端点）。"""
    import tempfile

    from agentworld.eval.lwm_eval_utils import (
        SCORE_DIMENSIONS,
        TASK_CONFIGS,
        parse_judge_output,
    )

    results = {}

    def gate(name, ok, detail=""):
        results[name] = {"passed": bool(ok), "detail": str(detail)}
        print(f"  [判定] {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        return ok

    gate("G0 七域提示词都在 vendor 里", len(TASK_CONFIGS) == len(DOMAINS) == 7, f"域 {sorted(TASK_CONFIGS)}")
    short = [d for d in DOMAINS if len(load_system_prompt(d)) < 1000 or len(load_judge_system_prompt(d)) < 1000]
    gate("G1 每域 system/judge 提示词非空", not short, f"过短的域 {short}")

    stub = StubWorldModel()
    try:
        env = WorldModelEnv(base_url=stub.base_url, domain="terminal", task="找出 /tmp 下的日志")
        env.reset()
        gate(
            "G2 任务与域提示词进 system",
            "找出 /tmp 下的日志" in env.system and "world model" in env.system.lower(),
            f"system 长度 {len(env.system)}",
        )
        st = env.step(name="execute_bash", arguments={"command": "ls -la /tmp"})
        gate(
            "G3 观测来自世界模型且剥掉响应标记",
            st["observation"].startswith("$ ls -la /tmp") and OBS_MARKER not in st["observation"],
            f"观测 {st['observation']!r}",
        )
        env.step(name="execute_bash", arguments={"command": "cat /tmp/a.log"})
        last_req = stub.seen[-1]["messages"]
        joined = json.dumps(last_req, ensure_ascii=False)
        gate(
            "G4 多轮历史逐轮累积",
            "ls -la /tmp" in joined and "cat /tmp/a.log" in joined and len(last_req) == 4,
            f"第 2 次请求消息数 {len(last_req)}",
        )
        gate(
            "G5 动作渲染成世界模型格式",
            last_req[-1]["content"] == "Action: execute_bash\nCommand: cat /tmp/a.log",
            last_req[-1]["content"],
        )

        ctrl = WorldModelEnv(
            base_url=stub.base_url, domain="swe", mode="control", spec={"perturbations": ["网络不可用"]}
        )
        gate(
            "G6 control：扰动进 system",
            "网络不可用" in ctrl.system and CONTROL_PREAMBLE[:20] in ctrl.system,
            "扰动已注入",
        )
        fic = WorldModelEnv(
            base_url=stub.base_url, domain="search", mode="fiction", spec={"world": "只有三个站点的内部网"}
        )
        gate("G7 fiction：虚构世界进 system", "只有三个站点的内部网" in fic.system, "设定已注入")
        bad = []
        for kw, dom in (("fiction", "terminal"), ("control", "terminal"), ("sim", "not_a_domain")):
            try:
                WorldModelEnv(base_url=stub.base_url, domain=dom, mode=kw, spec={})
                bad.append(kw)
            except SystemExit:
                pass
        gate("G8 缺设定 / 未知域会硬失败", not bad, f"没拦住 {bad}")

        done_env = WorldModelEnv(base_url=stub.base_url, domain="terminal", max_turns=6)
        done_env.reset()
        st = done_env.step(name="submit", arguments={"answer": "done"})
        gate("G9 停止位：观测带结束标记即 done", st["done"] and st["turn"] == 1, f"done={st['done']}")
        cap = WorldModelEnv(base_url=stub.base_url, domain="terminal", max_turns=2)
        cap.reset()
        cap.step(name="execute_bash", arguments={"command": "ls"})
        st = cap.step(name="execute_bash", arguments={"command": "ls"})
        gate("G10 轮数封顶", st["done"] and st["turn"] == 2, f"turn={st['turn']}")
        gate("G11 结束后不再发请求", cap.step(name="x")["turn"] == 2, "已停")

        import urllib.request as u

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _EnvHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        health = json.load(u.urlopen(f"{base}/health", timeout=10))  # noqa: S310
        payload = json.dumps(
            {"domain": "terminal", "task": "看日志", "max_turns": 3, "base_url": stub.base_url}
        ).encode()
        created = json.load(
            u.urlopen(
                u.Request(f"{base}/reset", data=payload, headers={"Content-Type": "application/json"}), timeout=30
            )  # noqa: S310
        )
        step_payload = json.dumps(
            {"session": created["session"], "name": "execute_bash", "arguments": {"command": "whoami"}}
        ).encode()
        stepped = json.load(
            u.urlopen(
                u.Request(f"{base}/step", data=step_payload, headers={"Content-Type": "application/json"}), timeout=30
            )  # noqa: S310
        )
        srv.shutdown()
        srv.server_close()
        gate(
            "G12 HTTP 环境服务端到端（/health /reset /step）",
            health.get("ok") is True and created.get("session") and "whoami" in stepped.get("observation", ""),
            f"turn={stepped.get('turn')}",
        )

        judged = parse_judge_output(JUDGE_SAMPLE, TASK_CONFIGS["terminal"]["judge_response_tag"])
        gate(
            "G13 AgentWorldBench 判分链路（五维 + 总分）",
            judged.get("success")
            and set(judged.get("scores", {})) == set(SCORE_DIMENSIONS)
            and abs(float(judged.get("total_score", 0)) - 4.0) < 1e-6,
            f"scores={judged.get('scores')} total={judged.get('total_score')}",
        )

        with tempfile.TemporaryDirectory() as tmp:
            _tool_gate(stub, gate, Path(tmp))
    finally:
        stub.stop()

    out = Path(os.environ.get("SHENSI_FS", "/root/work/filestorage")) / "shensi/logs/world_model_check.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:  # 本机跑自测时 SHENSI_FS 可能不可写，报告落不下来不影响判定
        out = None
    n_fail = sum(1 for v in results.values() if not v["passed"])
    print(f"\n  -> {'全部 PASS' if n_fail == 0 else f'{n_fail} 项 FAIL'}" + (f"（报告：{out}）" if out else ""))
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi 世界模型环境（Sim RL 的 env 后端）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="离线自测（假世界模型，不需要 GPU）")
    s = sub.add_parser("serve", help="起 HTTP 环境服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9000)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return check()
    return serve(args.host, args.port) or 0


if __name__ == "__main__":
    raise SystemExit(main())
