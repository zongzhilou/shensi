#!/usr/bin/env python3
"""stage0 集成测试：tiny 几何 + 假 grounding 数据跑 3 步，判 PASS/FAIL。

约定与 shensi 各 stage 的 test_train.py 一致：结构跑通（加载 → 前向反传 → 存 ckpt）。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, test_common


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="shensi_vl_t0_"))
    llm = test_common.tiny_llm()
    print(f"[test] tiny llm = {llm}")
    tok_dir, vocab = test_common.tiny_tokenizer(llm, tmp / "tok")
    data = test_common.make_fake_grounding(tmp / "tiny.jsonl", n=8)
    cfg = config.build_config(
        "stage0_pretrain",
        "tiny",
        [
            f"train.model.tokenizer_dir={tok_dir}",
            f"train.model.vocab_size={vocab}",
            f"train.model.llm_path={llm}",
            f"train.data.train_jsonl={data}",
            f"experiment.exp_dir={tmp / 'run'}",
        ],
    )
    from shensi.recipes.shensi_vl.common.train import train_loop

    rc = train_loop.run(cfg, mode="pretrain")
    ok = rc == 0 and (Path(cfg["experiment"]["exp_dir"]) / "final").exists()
    print(f"[test] stage0_pretrain tiny：{'PASS' if ok else 'FAIL'}（rc={rc}，产物 {tmp}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
