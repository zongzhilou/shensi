#!/usr/bin/env python3
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Watchdog:
    log: str
    metric: str
    patience: int = 3
    min_delta: float = 1e-4
    mode: str = "min"  # min：越小越好；max：越大越好
    target: float | None = None  # 到了就直接停
    grace: float = 600.0  # 启动宽限：这段时间内不判耐心
    poll: float = 20.0
    max_wait: float = 0.0  # >0 时总时长上限（秒），到点无论好坏都收尾
    stop_cmd: str | None = None  # 自定义收尾命令（默认给进程组发 SIGTERM）
    dry_run: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)

    def _read(self) -> list[float]:
        if not os.path.isfile(self.log):
            return []
        with open(self.log, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        pat = re.compile(re.escape(self.metric) + r"[^0-9\-+]*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
        out = []
        for m in pat.finditer(text):
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
        return out

    def _better(self, cur: float, best: float) -> bool:
        return cur < best - self.min_delta if self.mode == "min" else cur > best + self.min_delta

    def run(self) -> int:
        t0 = time.time()
        best = None
        bad = 0
        seen = 0
        while True:
            vals = self._read()
            if len(vals) > seen:
                for v in vals[seen:]:
                    if best is None or self._better(v, best):
                        best, bad = v, 0
                    else:
                        bad += 1
                    self.history.append((time.time() - t0, v))
                    print(
                        f"[early_stop] t={self.history[-1][0]:.0f}s {self.metric}={v:g} "
                        f"best={best:g} 未改善={bad}/{self.patience}",
                        flush=True,
                    )
                    if self.target is not None and (
                        (self.mode == "min" and v <= self.target)
                        or (self.mode == "max" and v >= self.target)
                    ):
                        print(f"[early_stop] 达到目标 {self.target}，收尾", flush=True)
                        return self._stop("target")
                    if bad >= self.patience and (time.time() - t0) >= self.grace:
                        print(
                            f"[early_stop] 连续 {bad} 次未改善（best={best:g}），收尾", flush=True
                        )
                        return self._stop("patience")
                seen = len(vals)
            if self.max_wait and (time.time() - t0) > self.max_wait:
                print("[early_stop] 到总时长上限，收尾", flush=True)
                return self._stop("max_wait")
            if os.path.exists("STOP_TRAINING"):
                print("[early_stop] 看到 STOP_TRAINING 标记，收尾", flush=True)
                return self._stop("flag")
            time.sleep(self.poll)

    def _stop(self, why: str) -> int:
        if self.stop_cmd:
            print(f"[early_stop] 执行收尾命令：{self.stop_cmd}", flush=True)
            if not self.dry_run:
                subprocess.call(self.stop_cmd, shell=True)
            return 0
        print(f"[early_stop] 给训练进程组发 SIGTERM（{why}）", flush=True)
        if self.dry_run:
            return 0
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[early_stop] 发信号失败（{exc}）；改用手工收尾：写 STOP_TRAINING 或 kill",
                file=sys.stderr,
            )
            return 1
        return 0


# 指标名与方向由 --metric/--mode 决定；RL 用 critic/score/mean + --mode max
def main() -> int:
    ap = argparse.ArgumentParser(description="训练早停看门狗（PT/SFT/RL 通用）")
    ap.add_argument(
        "--log",
        required=True,
        help="训练日志路径（launcher 写到 <exp_dir>/logs/host_0_localhost.output）",
    )
    ap.add_argument(
        "--metric", default="validation loss", help="日志里的指标名（PT/SFT 默认 validation loss）"
    )
    ap.add_argument("--mode", choices=("min", "max"), default="min")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--grace", type=float, default=600.0)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--max-wait", type=float, default=0.0)
    ap.add_argument("--stop-cmd", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return Watchdog(
        log=args.log,
        metric=args.metric,
        patience=args.patience,
        min_delta=args.min_delta,
        mode=args.mode,
        target=args.target,
        grace=args.grace,
        poll=args.poll,
        max_wait=args.max_wait,
        stop_cmd=args.stop_cmd,
        dry_run=args.dry_run,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
