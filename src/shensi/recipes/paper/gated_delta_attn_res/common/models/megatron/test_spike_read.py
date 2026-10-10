#!/usr/bin/env python3
"""spike 度量读的跨后端闸门：HF 与 mcore 两算子同权重、同 RNG 对拍，并校验看护规则。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron import (  # noqa: E402
    gdar_connection as mc,
)
from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.configuration_qwen3_gdar import (  # noqa: E402
    Qwen3GDARConfig,
)
from shensi.recipes.paper.gated_delta_attn_res.common.models.transformers.modeling_qwen3_gdar import (  # noqa: E402
    AttentionResidual as HFAttentionResidual,
)

FAILS: list[str] = []


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


H, RANK, LAYERS = 32, 8, 4
SPIKE_KNOBS = dict(
    attn_res_gate_param="deviation",
    attn_res_update="objective",
    attn_res_address="delta",
    attn_res_decay_ladder=H,
    attn_res_read_whiten="spike",
    attn_res_read_heads=1,
    attn_res_read_null=False,
    attn_res_gate_rank=RANK,
    attn_res_q_rank=RANK,
    attn_res_k_rank=RANK,
)


def build_pair() -> tuple[HFAttentionResidual, mc.AttentionResidual]:
    """同权重的一对算子：HF 的 .0/.1 ↔ mcore 的 down/up，缩放与阶梯逐项照搬。"""
    torch.manual_seed(0)
    hf = HFAttentionResidual(
        Qwen3GDARConfig(hidden_size=H, num_hidden_layers=LAYERS, rms_norm_eps=1e-6, **SPIKE_KNOBS)
    ).eval()
    mc_mod = mc.AttentionResidual(
        H,
        mc.GdarConfig(
            gate_param="deviation",
            update="objective",
            address="delta",
            decay_ladder=H,
            decay_tau_max=100.0,
            read_whiten="spike",
            gate_rank=RANK,
            q_rank=RANK,
            k_rank=RANK,
        ).validated(),
        eps=1e-6,
    ).eval()

    with torch.no_grad():
        for hf_proj, mc_proj in (
            (hf.gate_proj, mc_mod.gate_proj),
            (hf.q_proj, mc_mod.q_proj),
            (hf.k_proj, mc_mod.k_proj),
        ):
            mc_proj.down.weight.copy_(hf_proj[0].weight)
            mc_proj.up.weight.copy_(hf_proj[1].weight)
            if mc_proj.down.bias is not None:
                mc_proj.down.bias.copy_(hf_proj[0].bias)
            if mc_proj.up.bias is not None:
                mc_proj.up.bias.copy_(hf_proj[1].bias if hf_proj[1].bias is not None else 0.0)
        for name in ("decay_scale", "erase_scale", "write_scale", "read_scale"):
            getattr(mc_mod, name).copy_(getattr(hf, name))
        mc_mod.decay_tau.copy_(hf.decay_tau)
    return hf, mc_mod


def main() -> int:
    hf, mc_mod = build_pair()

    torch.manual_seed(0)
    prefix = torch.randn(6, H)
    delta = torch.randn(6, H)
    blocks = torch.randn(6, 3, H)

    # 读吃全局 RNG（两次 randint + svd_lowrank 内部 randn）：每轮两侧各自重播种
    with torch.no_grad():
        for read_scale in (0.0, 1.0):
            hf.read_scale.fill_(read_scale)
            mc_mod.read_scale.fill_(read_scale)
            torch.manual_seed(123)
            hf_upd, hf_gates = hf.update(prefix.clone(), delta.clone())
            hf_read, _ = hf.read(prefix.clone(), blocks.clone(), updated=hf_upd)
            torch.manual_seed(123)
            mc_upd, mc_gates = mc_mod.update(prefix.clone(), delta.clone())
            mc_read, _ = mc_mod.read(prefix.clone(), blocks.clone(), updated=mc_upd)
            gates_d = max((a - b).abs().max().item() for a, b in zip(hf_gates, mc_gates))
            check(
                f"写路逐位（read_scale={read_scale:g}）",
                torch.equal(hf_upd, mc_upd) and gates_d == 0.0,
                f"max|Δupdated| = {(hf_upd - mc_upd).abs().max().item():.3e}",
            )
            check(
                f"spike 读逐位（read_scale={read_scale:g}）",
                torch.equal(hf_read, mc_read),
                f"max|Δread| = {(hf_read - mc_read).abs().max().item():.3e}",
            )

    check(
        "无块时读退化为 prefix（两侧）",
        torch.equal(hf.read(prefix.clone(), None)[0], prefix)
        and torch.equal(mc_mod.read(prefix.clone(), None)[0], prefix),
        "blocks=None",
    )

    for label, knobs in (
        ("HF", {"attn_res_read_heads": 2}),
        ("HF", {"attn_res_read_null": True}),
    ):
        try:
            HFAttentionResidual(
                Qwen3GDARConfig(
                    hidden_size=H,
                    num_hidden_layers=LAYERS,
                    rms_norm_eps=1e-6,
                    **{**SPIKE_KNOBS, **knobs},
                )
            )
        except ValueError:
            ok = True
        else:
            ok = False
        check(f"{label} 校验拒绝 spike + {list(knobs)[0].removeprefix('attn_res_')}", ok)

    for knobs in (dict(read_heads=2), dict(read_null=True)):
        try:
            mc.GdarConfig(read_whiten="spike", **knobs).validated()
        except ValueError:
            ok = True
        else:
            ok = False
        check(f"mcore 校验拒绝 spike + {list(knobs)[0]}", ok)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'SOME CHECKS FAILED'}")
    if FAILS:
        print("FAILED:", FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
