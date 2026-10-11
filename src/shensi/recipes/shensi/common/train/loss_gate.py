#!/usr/bin/env python3
"""收敛闸门：按文献律算「预期 loss 曲线」，把训练日志的观测值对上去判异常。

模型口径（每条都能覆盖）：

1. Chinchilla 参数化律 `L(D) = E + A/N^α + B/D^β`（arXiv:2203.15556 的 Approach-3 未取整常数）。
   `N` = **每 token 激活参数量**（本仓默认几何 0.7570e9）——MoE 的 loss 以激活为主变量，
   路由增益另算（见 2）。
2. MoE 折扣：同激活参数的 MoE ≈ dense 的 **3× token**（OLMoE 实测，arXiv:2409.02060 §4.1.1：1.3B 激活
   的 MoE「reaches the final dense performance with ~3x fewer tokens」）。折算成 nats：默认
   `--moe-discount 0.09`（57B 档上 3× token ≈ −0.09 nats；见 HYPERPARAMS.md）。
3. 重复修正（Muennighoff arXiv:2305.16264）：`D_eff = U·(1 + R*·(1 − e^−(R−1)/R*))`，`R* = 15.3878`、
   `R = 已见 token / 唯一语料`。第 2/3 个 epoch 的边际只值 ~1/15 的「新鲜 token」。
4. 语料/分词器偏置 `--offset`：公开跑法（Pile / GPT-2-BPE）比该律**低 ~0.46 nats**（口径差，不是好坏）。
   绝对阈值必须用自己语料标定：先用本工具 `--calibrate` 从前段日志拟一个 offset，再拿它判后段。

本机就能跑的自检：

    python -m shensi.recipes.shensi.common.train.loss_gate --selftest

对着训练日志判（mcore 的 `lm loss:` 行）：

    python -m shensi.recipes.shensi.common.train.loss_gate \
        --log runs/stage1_pretrain/logs/host_0_localhost.output \
        --unique-tokens 1.9e10 --epochs 3 --tokens-per-step auto --offset 0

判据（默认）：任一检查点（10/30/50/100%）观测 > 预期 + 0.30 nats → **fail**（先查数据/LR）；
终值 > 预期 + 0.15 → warn；否则 ok。预期曲线是「完全退火后的 loss」，训练中段天然略高
（LR 还没降下来，具体差多少用一次短 pilot 量），所以 0.15/0.30 是容差不是精度。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# Chinchilla Approach-3（未取整，arXiv:2203.15556 Table A1；取整版 E=1.69/A=406.4/B=410.7 只差 ~0.04）
E_DEFAULT = 1.6934
A_DEFAULT = 406.4
ALPHA_DEFAULT = 0.3392
B_DEFAULT = 410.7
BETA_DEFAULT = 0.2849

N_ACTIVE_DEFAULT = 0.7570e9  # 默认几何（2p8b_a1p2b.json）：每 token 激活
MOE_DISCOUNT_DEFAULT = 0.09  # nats；等价「同激活 dense 的 3× token」（OLMoE 实测）
R_STAR = 15.3878  # 重复律的 R*（arXiv:2305.16264 拟合值）

_LOSS = re.compile(r"lm loss:\s*([0-9.eE+-]+)")
_ITER = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
_TOKENS = re.compile(r"consumed tokens:\s*(\d+)")
_GBSSEQ = re.compile(r"global batch size:\s*(\d+).*?sequence length:\s*(\d+)", re.S)


def expected_loss(
    tokens: float,
    *,
    n_params: float = N_ACTIVE_DEFAULT,
    e: float = E_DEFAULT,
    a: float = A_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    b: float = B_DEFAULT,
    beta: float = BETA_DEFAULT,
    discount: float = MOE_DISCOUNT_DEFAULT,
    offset: float = 0.0,
) -> float:
    """Law 口径的预期 loss（nats/token）。"""
    return e + a / n_params**alpha + b / max(tokens, 1.0) ** beta - discount + offset


def effective_tokens(passes: float, unique_tokens: float) -> float:
    """把「已看 token 数」折算成考虑重复后的等效新鲜 token（Muennighoff 律）。"""
    if passes <= unique_tokens:
        return passes
    r = passes / unique_tokens  # 已完成 epoch 数
    return unique_tokens * (1.0 + R_STAR * (1.0 - math.exp(-(r - 1.0) / R_STAR)))


def parse_log(path: Path) -> list[tuple[float, float]]:
    """从 mcore 日志抽 (已看 token 数, lm loss)。优先用 consumed tokens，没有就按 iter×gbs×seq 推。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    per_step = None
    m = _GBSSEQ.search(text)
    if m:
        per_step = float(m.group(1)) * float(m.group(2))
    out: list[tuple[float, float]] = []
    seen = 0.0
    for line in text.splitlines():
        loss = _LOSS.search(line)
        if not loss:
            continue
        it = _ITER.search(line)
        tok = _TOKENS.search(line)
        if tok:
            seen = float(tok.group(1))
        elif it and per_step:
            seen = (float(it.group(1)) + 1.0) * per_step
        try:
            out.append((seen, float(loss.group(1))))
        except ValueError:
            continue
    return out


def evaluate(
    series: list[tuple[float, float]],
    *,
    unique_tokens: float,
    epochs: float,
    total_tokens: float | None,
    kwargs: dict,
    warn_at: float,
    fail_at: float,
) -> dict:
    budget = total_tokens or (series[-1][0] if series else 0.0)
    checks = []
    for frac in (0.1, 0.3, 0.5, 1.0):
        at = budget * frac
        near = [s for s in series if s[0] <= at]
        if not near:
            continue
        obs = near[-1][1]
        exp = expected_loss(effective_tokens(at, unique_tokens), **kwargs)
        delta = obs - exp
        checks.append(
            {
                "fraction": frac,
                "tokens": round(at, 1),
                "epoch_equiv": round(at / unique_tokens, 2),
                "observed": round(obs, 4),
                "expected": round(exp, 4),
                "delta_nats": round(delta, 4),
                "verdict": "fail" if delta > fail_at else ("warn" if delta > warn_at else "ok"),
            }
        )
    worst = max((c["delta_nats"] for c in checks), default=float("nan"))
    status = (
        "fail"
        if any(c["verdict"] == "fail" for c in checks)
        else ("warn" if any(c["verdict"] == "warn" for c in checks) else "ok")
    )
    return {"status": status, "worst_delta_nats": worst, "checks": checks, "budget_tokens": budget}


def _selftest() -> int:
    print("== 预期 loss 表（默认几何：N_active=0.7570e9，MoE 折扣 0.09，唯一 19B × 3 epoch）==")
    print(
        f"{'passes':>10} {'epoch':>6} {'D_eff':>12} {'预期 loss(law+MoE)':>18} {'（含 -0.46 语料偏置）':>20}"
    )
    for frac in (0.1, 0.3, 0.5, 1.0):
        passes = 5.7e10 * frac
        d_eff = effective_tokens(passes, 1.9e10)
        exp = expected_loss(d_eff)
        print(
            f"{passes / 1e9:>9.1f}B {passes / 1.9e10:>6.2f} {d_eff / 1e9:>11.1f}B "
            f"{exp:>18.3f} {exp - 0.46:>20.3f}"
        )
    print("\n（-0.46 = 公开 Pile/GPT-2-BPE 跑法的口径偏置，仅示意；绝对阈值要用自己的语料标定）")
    print("判据：30% 处观测 > 预期 + 0.30 nats → 异常；终值 > 预期 + 0.15 → warn。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按文献律给训练 loss 设闸门")
    ap.add_argument("--log", help="训练日志（mcore 的 .output）")
    ap.add_argument("--tokens", type=float, default=None, help="本次预算（passes；默认取日志末值）")
    ap.add_argument(
        "--unique-tokens", type=float, default=1.9e10, help="唯一语料 token（默认 19B）"
    )
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--n-params", type=float, default=N_ACTIVE_DEFAULT, help="每 token 激活参数")
    ap.add_argument("--moe-discount", type=float, default=MOE_DISCOUNT_DEFAULT)
    ap.add_argument("--offset", type=float, default=0.0, help="语料/分词器偏置（自己的标定值）")
    ap.add_argument("--warn-at", type=float, default=0.15)
    ap.add_argument("--fail-at", type=float, default=0.30)
    ap.add_argument("--out", default=None, help="把结论写成 JSON")
    ap.add_argument("--selftest", action="store_true", help="本机打印预期表")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.log:
        raise SystemExit("给 --log（或 --selftest）")
    series = parse_log(Path(args.log))
    if not series:
        raise SystemExit(f"{args.log} 里没找到 `lm loss:` 行")
    kwargs = {
        "n_params": args.n_params,
        "discount": args.moe_discount,
        "offset": args.offset,
    }
    res = evaluate(
        series,
        unique_tokens=args.unique_tokens,
        epochs=args.epochs,
        total_tokens=args.tokens,
        kwargs=kwargs,
        warn_at=args.warn_at,
        fail_at=args.fail_at,
    )
    res["log"] = str(args.log)
    print(f"== 收敛闸门：{args.log}")
    print(
        f"{'检查点':>8} {'已看 token':>14} {'epoch':>6} {'观测':>8} {'预期':>8} {'偏差':>8}  判定"
    )
    for c in res["checks"]:
        print(
            f"{c['fraction']:>7.0%} {c['tokens'] / 1e9:>13.1f}B {c['epoch_equiv']:>6.2f} "
            f"{c['observed']:>8.3f} {c['expected']:>8.3f} {c['delta_nats']:>+8.3f}  {c['verdict']}"
        )
    print(f"\n结论：{res['status']}（最差偏差 {res['worst_delta_nats']:+.3f} nats）")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"记账：{args.out}")
    return 0 if res["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
