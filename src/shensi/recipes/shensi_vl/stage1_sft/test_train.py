#!/usr/bin/env python3
"""stage1 集成测试：tiny 几何 + 合成假数据（迷宫文本 + 假 grounding 图）跑 3 步 SFT。"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.common import config, test_common
from shensi.recipes.shensi_vl.common import primitives as P


def fake_sft_jsonl(tmp: Path) -> Path:
    """带思维链的假样本：grounding（图+框）与迷宫（spec 缩小版）各 4 条。"""
    grounding = test_common.make_fake_grounding(tmp / "g.jsonl", n=4)
    with open(grounding, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        r["thinking"] = (
            "1. **Deconstructing the query**\nCount the blocks.\n"
            f"2. **Sweeping**\n{r['response']}\n3. **Tally**\n1."
        )
    for i in range(4):
        rows.append(
            {
                "task": "maze",
                "image": None,
                "question": "Is there a path? Output \\boxed{True} or \\boxed{False}.",
                "thinking": (
                    f"Exploring. Start {P.render_point_primitive([[100, 500]])}, "
                    f"next {P.render_point_primitive([[200, 500]])}, dead end, backtrack."
                ),
                "response": "No path exists.\n\\boxed{False}",
                "source": "tiny_fake",
            }
        )
    fp = tmp / "tiny_sft.jsonl"
    with open(fp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return fp


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="shensi_vl_t1_"))
    llm = test_common.tiny_llm()
    tok_dir, vocab = test_common.tiny_tokenizer(llm, tmp / "tok")
    data = fake_sft_jsonl(tmp)
    cfg = config.build_config(
        "stage1_sft",
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

    rc = train_loop.run(cfg, mode="sft")
    ok = rc == 0 and (Path(cfg["experiment"]["exp_dir"]) / "final").exists()
    print(f"[test] stage1_sft tiny：{'PASS' if ok else 'FAIL'}（rc={rc}，产物 {tmp}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
