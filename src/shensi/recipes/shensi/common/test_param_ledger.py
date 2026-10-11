#!/usr/bin/env python3
"""参数账闸门：2p8b 几何 meta 实测对账（2.2930B / 0.7570B / 8 组专家 / 1 张 deepemb），
外加审计的失败模式（未共享、组内混对象、多张表）用假层验证。

背景：机上实测 4.53B 与账面 2.2930B 对不上，差额恰是"16 组专家未共享 + 3 张 deepemb
未共表"——本闸门把这类事故钉在构建期。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[5]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from shensi.recipes.shensi.common import param_ledger as pl  # noqa: E402

HF_CONFIG = REPO / "src/shensi/recipes/shensi/common/config/hf/2p8b_a1p2b.json"

FAILS: list[str] = []


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def expect_audit_error(name: str, fn, needle: str):
    try:
        fn()
        check(name, False, "没有抛")
    except ValueError as exc:
        check(name, needle in str(exc), str(exc)[:70])


class FakeExperts:
    def __init__(self, n: int = 256):
        self.gate_up_proj = torch.zeros(n, 16, 8)
        self.down_proj = torch.zeros(n, 8, 8)

    def parameters(self):
        yield self.gate_up_proj
        yield self.down_proj


def fake_layer(*, hash_table=None, experts=None):
    mlp = SimpleNamespace()
    if hash_table is not None:
        mlp.deepemb = hash_table
    if experts is not None:
        mlp.experts = experts
    return SimpleNamespace(mlp=mlp)


def main() -> int:
    print("== 参数账审计 ==")
    with tempfile.TemporaryDirectory():
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(HF_CONFIG)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)

        types = list(cfg.attn_res_block_layer_types)
        report = pl.audit_hf_model(model)
        check("2p8b：8 组专家 / 1 张表", report["expert_sets"] == 8 and report["deepemb_tables"] == 1,
              f"sets={report['expert_sets']} tables={report['deepemb_tables']}")
        dedup_b = report["total_dedup"] / 1e9
        sd_b = report["total_state_dict"] / 1e9
        active_b = report["active"] / 1e9
        check("总参（dedup）= 2.2930B", abs(dedup_b - 2.2930) / 2.2930 < 0.005, f"{dedup_b:.4f}B")
        check("state_dict 口径 = 3.9512B（不 dedup，用于对机上口径）",
              abs(sd_b - 3.9512) / 3.9512 < 0.005, f"{sd_b:.4f}B")
        check("每 token 激活 = 0.7570B", abs(active_b - 0.7570) / 0.7570 < 0.005, f"{active_b:.4f}B")

        # 块计划与 tie_mlp 同源
        check("块计划：write 层 = [0] + 8 个",
              len([t for t in types if t == "block_write_layer"]) == 9, f"n_write={types.count('block_write_layer')}")

        # 失败模式（假层，全栈 19 层计划）
        full_types = types
        e0, e1 = FakeExperts(), FakeExperts()
        t0 = torch.nn.Embedding(128, 8)
        # 正常：write 层在 0,3,5,7…，层 3/4 同块共享 e0、层 5 属下一块独占 e1
        layers = [fake_layer(hash_table=t0), fake_layer(experts=e0), fake_layer(experts=e1)]
        idx = [3, 4, 5]
        pl.audit_layers(layers, idx, mlp_layer_types=full_types, attn_res_block_layer_types=full_types)
        check("审计：正常共享通过", True)

        # 未共享：同块两 experts 对象
        expect_audit_error(
            "审计：同块未共享被拦（tie_mlp 没生效）",
            lambda: pl.audit_layers(
                [fake_layer(experts=e0), fake_layer(experts=e1)], [3, 4],
                mlp_layer_types=full_types, attn_res_block_layer_types=full_types,
            ),
            "tie_mlp",
        )
        # 组内混对象（三层里两共享一独占）
        expect_audit_error(
            "审计：组内混对象被拦",
            lambda: pl.audit_layers(
                [fake_layer(experts=e0), fake_layer(experts=e1), fake_layer(experts=e0)], [3, 4, 5],
                mlp_layer_types=full_types, attn_res_block_layer_types=full_types,
            ),
            "不同 experts 对象",
        )
        # 多张 deepemb 表
        t1 = torch.nn.Embedding(128, 8)
        expect_audit_error(
            "审计：deepemb 多表被拦",
            lambda: pl.audit_layers(
                [fake_layer(hash_table=t0), fake_layer(hash_table=t1)], [0, 1],
                mlp_layer_types=full_types, attn_res_block_layer_types=full_types,
            ),
            "共表没生效",
        )
        # PP 分 stage：层号带全局偏移仍成立（5、6 层在一块）
        pl.audit_layers(
            [fake_layer(experts=e0), fake_layer(experts=e0)], [5, 6],
            mlp_layer_types=full_types, attn_res_block_layer_types=full_types,
        )
        check("审计：PP 偏移层号正确解析", True)

        # Bridge 侧入口：假 ShensiModel（config 驱动的类型计划 + decoder.layers）
        bridge_model = SimpleNamespace(
            config=SimpleNamespace(moe_n_hash_layers=3, attn_res_block_size=2, num_layers=19),
            decoder=SimpleNamespace(layers=[
                SimpleNamespace(mlp=SimpleNamespace(deepemb=torch.nn.Embedding(128, 8)), layer_number=1),
                SimpleNamespace(mlp=SimpleNamespace(deepemb=None), layer_number=2),  # hash 层但无 experts? 见下
            ]),
        )
        # 用 4 层 tiny 计划更贴实际：hash 1 + moe 3 层共享一组
        bridge_model = SimpleNamespace(
            config=SimpleNamespace(moe_n_hash_layers=1, attn_res_block_size=4, num_layers=4),
            decoder=SimpleNamespace(layers=[
                SimpleNamespace(mlp=SimpleNamespace(deepemb=torch.nn.Embedding(128, 8)), layer_number=1),
                SimpleNamespace(mlp=SimpleNamespace(experts=e0), layer_number=2),
                SimpleNamespace(mlp=SimpleNamespace(experts=e0), layer_number=3),
                SimpleNamespace(mlp=SimpleNamespace(experts=e0), layer_number=4),
            ]),
        )
        rep = pl.audit_bridge_model(bridge_model)
        check("审计：Bridge 入口（tiny 计划 1 组）", rep["expert_sets"] == 1 and rep["deepemb_tables"] == 1,
              f"sets={rep['expert_sets']} tables={rep['deepemb_tables']}")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项未过：{FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
