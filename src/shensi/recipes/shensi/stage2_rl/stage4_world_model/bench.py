#!/usr/bin/env python3
# 世界模型的评测：照 AgentWorldBench 的口径给「下一状态预测」打分（五维 Format / Factuality / Consistency /
# Realism / Quality），既评 Qwen-AgentWorld 这类现成模型，也评我们自己训的 shensi-world。
# 数据吃上游的 *_test.jsonl（{task, system_str, prompt[], response[], turn_idx}），也吃自家轨迹。

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common, rl
from shensi.recipes.shensi.stage2_rl.agentworld.eval.lwm_eval_utils import (
    SCORE_DIMENSIONS,
    TASK_CONFIGS,
    parse_judge_output,
)
from shensi.recipes.shensi.stage2_rl.stage2_agentic import world_model as wm
from shensi.recipes.shensi.stage2_rl.stage4_world_model import wm_common


def to_job(row: dict) -> dict | None:
    """自家的轨迹行也归一成 AgentWorldBench 的 job 形状（prompt/response/turn_idx）。"""
    job = wm_common.agentworld_job(row)
    if job:
        return job
    turns = wm_common.to_turns(row)
    if not turns:
        return None
    domain = wm_common.domain_of(row)
    system = wm_common.system_of(row, domain)
    return {
        "task": domain,
        "system_str": system,
        "prompt": [a for a, _ in turns],
        "response": [o for _, o in turns],
        "turn_idx": len(turns),
        "current_prompt": turns[-1][0],
    }


def load_jobs(path: Path, limit: int | None) -> list[dict]:
    files = (
        [path]
        if path.is_file()
        else sorted(f for f in path.glob("**/*") if f.suffix in (".jsonl", ".json", ".parquet"))
    )
    jobs = []
    for row in rl.iter_rows(files, limit):
        job = to_job(row)
        if job:
            jobs.append(job)
    return jobs


def run(
    jobs: list[dict],
    lwm_url: str | None,
    lwm_model: str | None,
    judge_url: str | None,
    judge_model: str | None,
):
    lwm = wm.WorldModelClient(base_url=lwm_url, model=lwm_model, temperature=0.6)
    judge = wm.WorldModelClient(
        base_url=judge_url or lwm_url, model=judge_model or lwm_model, temperature=0.0
    )
    records = []
    for i, job in enumerate(jobs):
        domain = wm_common.domain_of(job)
        pred_raw = lwm.chat(wm_common.lwm_input(job))
        prediction = wm.parse_observation(pred_raw, domain)
        judge_raw = judge.chat(wm_common.judge_messages(job, prediction, domain))
        parsed = parse_judge_output(judge_raw, TASK_CONFIGS[domain]["judge_response_tag"])
        records.append(
            {
                "index": i,
                "task": str(job.get("task") or domain),
                "domain": domain,
                "prediction": prediction,
                "scores": parsed.get("scores") or {},
                "total_score": float(parsed.get("total_score") or 0.0),
                "success": bool(parsed.get("success")),
            }
        )
        print(
            f"  [{i + 1}/{len(jobs)}] {domain:<9} total={records[-1]['total_score']:.2f} "
            f"{'' if records[-1]['success'] else '(判分失败)'}",
            flush=True,
        )
    return records


def summarize(records: list[dict], settings: dict) -> dict:
    per_domain: dict[str, dict] = {}
    for domain in sorted({r["domain"] for r in records}):
        rows = [r for r in records if r["domain"] == domain]
        valid = [r for r in rows if r["success"]]
        per_domain[domain] = {
            "n": len(rows),
            "n_valid": len(valid),
            "total_score": sum(r["total_score"] for r in valid) / len(valid) if valid else 0.0,
            **{
                dim: (sum(r["scores"].get(dim, 0) for r in valid) / len(valid) if valid else 0.0)
                for dim in SCORE_DIMENSIONS
            },
        }
    valid_all = [r for r in records if r["success"]]
    return {
        "n": len(records),
        "n_valid": len(valid_all),
        "invalid": len(records) - len(valid_all),
        "overall": {
            "total_score": sum(r["total_score"] for r in records) / len(records)
            if records
            else 0.0,
            "total_score_valid_only": sum(r["total_score"] for r in valid_all) / len(valid_all)
            if valid_all
            else 0.0,
        },
        "per_domain": per_domain,
        "settings": settings,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


class StubJudge:
    """自测用的假裁判：五维全 5，指定的一条返回垃圾（验证「判分失败不当崩」）。"""

    def __init__(self, bad_index: int = 0):
        self.bad_index = bad_index
        self.count = 0
        outer = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                json.loads(self.rfile.read(n) or b"{}")
                idx = outer.count
                outer.count += 1
                if idx == outer.bad_index:
                    content = "抱歉，我无法完成这次评估。"
                else:
                    content = (
                        "<final_evaluation>\n```json\n"
                        '{"strengths": ["格式对"], "weaknesses": [], "scores": {"format": 5, "factuality": 5, '
                        '"consistency": 5, "realism": 5, "quality": 5}}\n```\n</final_evaluation>'
                    )
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def stub_samples() -> list[dict]:
    out = []
    for domain in wm_common.DOMAINS:
        out.append(
            {
                "task": f"{domain}/tiny",
                "system_str": wm.load_system_prompt(domain),
                "prompt": [f"Action: act\nCommand: step-1 {domain}"],
                "response": ["step-1 output"],
                "turn_idx": 1,
                "current_prompt": f"Action: act\nCommand: step-1 {domain}",
            }
        )
    return out


def stub_check(jobs: list[dict]) -> int:
    from reward import score_from_judge_output

    results = {}

    def gate(name, ok, detail=""):
        results[name] = {"passed": bool(ok), "detail": str(detail)}
        print(f"  [判定] {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        return ok

    lwm_stub = wm.StubWorldModel()
    judge_stub = StubJudge(bad_index=2)
    bad_domain = wm_common.DOMAINS[2]
    try:
        gate(
            "B0 七个域都造了样本",
            {wm_common.domain_of(j) for j in jobs} == set(wm_common.DOMAINS),
            f"{len(jobs)} 条",
        )
        records = run(jobs, lwm_stub.base_url, "stub-lwm", judge_stub.base_url, "stub-judge")
        summary = summarize(records, {"stub": True})
        gate(
            "B1 每条都有五维分（判分失败的不算）",
            all(set(r["scores"]) == set(SCORE_DIMENSIONS) for r in records if r["success"])
            and sum(1 for r in records if r["success"]) == 6,
            f"有效 {summary['n_valid']}/{summary['n']}",
        )
        gate(
            "B2 判分失败记 invalid 且不当崩",
            summary["invalid"] == 1,
            f"invalid={summary['invalid']}",
        )
        gate(
            "B3 汇总按域算对（有效样本 5 分满分，坏的那域记 0）",
            all(
                abs(v["total_score"] - 5.0) < 1e-9
                for d, v in summary["per_domain"].items()
                if d != bad_domain
            )
            and abs(summary["per_domain"][bad_domain]["total_score"]) < 1e-9,
            f"除 {bad_domain} 外全 5.0，{bad_domain} 0.0",
        )
        gate(
            "B4 overall 把 invalid 也算进去",
            abs(summary["overall"]["total_score"] - 30.0 / 7) < 1e-6
            and abs(summary["overall"]["total_score_valid_only"] - 5.0) < 1e-6,
            f"overall={summary['overall']['total_score']:.4f} valid_only={summary['overall']['total_score_valid_only']:.1f}",
        )
        gate(
            "B5 奖励归一化与判分一致（4 分 → 0.8）",
            abs(score_from_judge_output(wm.JUDGE_SAMPLE, "terminal") - 0.8) < 1e-9
            and abs(score_from_judge_output("垃圾输出", "terminal")) < 1e-9,
            "4/5=0.8；解析不了的记 0",
        )
        messages = wm_common.judge_messages(jobs[0], "pred", "terminal")
        user = messages[-1]["content"]
        gate(
            "B6 判分提示词带上下文 / 当前轮 / 模拟输出 / 真值",
            all(
                k in user
                for k in (
                    "# Current Turn:",
                    "**World Model Output (Simulated):**",
                    "**Ground Truth (Real Output):**",
                )
            )
            and messages[0]["content"] == wm.load_judge_system_prompt("terminal"),
            "槽位齐全",
        )
    finally:
        lwm_stub.stop()
        judge_stub.stop()

    n_fail = sum(1 for v in results.values() if not v["passed"])
    print(f"\n  -> {'全部 PASS' if n_fail == 0 else f'{n_fail} 项 FAIL'}")
    return 0 if n_fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi 世界模型评测（AgentWorldBench 口径）")
    ap.add_argument(
        "--data", default=None, help="上游 *_test.jsonl 或自家轨迹（文件/目录），--stub 时不用给"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lwm-url", default=None, help="世界模型端点，默认 $SHENSI_WORLD_MODEL_URL")
    ap.add_argument("--lwm-model", default=None)
    ap.add_argument("--judge-url", default=None, help="判分端点，默认同世界模型")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--out", default=None, help="报告路径，默认 <out>/world_model_bench.json")
    ap.add_argument("--stub", action="store_true", help="离线自测：假世界模型 + 假裁判")
    args = ap.parse_args(argv)

    if args.stub:
        return stub_check(stub_samples())
    if not args.data:
        ap.error("--data 或 --stub 至少给一个")
    jobs = load_jobs(Path(args.data), args.limit)
    if not jobs:
        raise SystemExit(f"[bench] {args.data} 里没有可评的样本（认 {TASK_CONFIGS} 的七个域）")
    print(f"[bench] {len(jobs)} 条样本，开始评 ...")
    settings = {
        "data": str(args.data),
        "lwm_url": args.lwm_url or "(env)",
        "judge_url": args.judge_url or args.lwm_url or "(同世界模型)",
        "dimensions": SCORE_DIMENSIONS,
    }
    records = run(jobs, args.lwm_url, args.lwm_model, args.judge_url, args.judge_model)
    if args.out:
        out = Path(args.out)
    else:
        paths = common.env_paths()
        out = paths["logs"] / "world_model_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, settings)
    summary["records"] = records
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"[bench] overall={summary['overall']['total_score']:.2f}（有效 {summary['n_valid']}/{summary['n']}）→ {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
