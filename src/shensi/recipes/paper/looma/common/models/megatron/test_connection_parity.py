"""对拍：mcore 侧与 HF 侧的 Looma 连接层在同一权重、同一输入下逐位一致。

两侧结构与参数名逐一刻度镜像、算子都在 fp32 里算，故可直接 ``torch.equal`` 比较。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:  # 包内导入
    from shensi.recipes.paper.looma.common.models.megatron.looma_connection import (
        LoomaAttentionResidual as MCoreConnection,
        LoomaConfig as MCoreConfig,
        solve_block as mcore_solve,
    )
    from shensi.recipes.paper.looma.common.models.transformers.configuration_looma import LoomaConfig
    from shensi.recipes.paper.looma.common.models.transformers.modeling_looma import (
        LoomaAttentionResidual as HFConnection,
        solve_block as hf_solve,
    )
except ImportError:  # 直接运行本文件：把 src 加进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    from shensi.recipes.paper.looma.common.models.megatron.looma_connection import (
        LoomaAttentionResidual as MCoreConnection,
        LoomaConfig as MCoreConfig,
        solve_block as mcore_solve,
    )
    from shensi.recipes.paper.looma.common.models.transformers.configuration_looma import LoomaConfig
    from shensi.recipes.paper.looma.common.models.transformers.modeling_looma import (
        LoomaAttentionResidual as HFConnection,
        solve_block as hf_solve,
    )


def _hf_config(hidden: int, layers: int, **knobs) -> LoomaConfig:
    base = dict(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=max(1, hidden // 32),
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    base.update(knobs)
    return LoomaConfig(**base)


def _mcore_config(layers: int, **knobs) -> MCoreConfig:
    base = dict(decay_tau_max=2.0 * layers, init_std=0.02)
    base.update(knobs)
    return MCoreConfig(**base).validated()


def _pair(hidden: int, layers: int = 2, seed: int = 0, **knobs):
    """同种子、同权重的一对连接（权重从 HF 侧拷进 mcore 侧，键名逐一对齐）。"""
    torch.manual_seed(seed)
    hf = HFConnection(_hf_config(hidden, layers, **knobs)).eval()
    # 两侧的旋钮名只差一个前缀（HF: looma_read_heads / mcore: read_heads），这里做转换；
    # mcore 独有的（decay_tau_max / init_std）原样透传。
    mc_knobs = {k[len('looma_'):] if k.startswith('looma_') else k: v for k, v in knobs.items()}
    mc = MCoreConnection(hidden, _mcore_config(layers, **mc_knobs), eps=1e-6).eval()
    hf_sd = hf.state_dict()
    mc_sd = mc.state_dict()
    missing = [k for k in mc_sd if k not in hf_sd]
    shape_bad = [k for k in mc_sd if k in hf_sd and hf_sd[k].shape != mc_sd[k].shape]
    assert not missing, f"mcore 侧多出来的张量：{missing}"
    assert not shape_bad, f"形状不一致：{shape_bad}"
    mc.load_state_dict(hf_sd, strict=False)
    return hf, mc


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    return ok


def check_forward(hidden: int = 64, **knobs) -> bool:
    """C1：四路输入下的前向逐位。"""
    hf, mc = _pair(hidden, **knobs)
    torch.manual_seed(1)
    prefix = torch.randn(2, 6, hidden)
    delta = torch.randn(2, 6, hidden)
    blocks = torch.randn(2, 6, 3, hidden)
    weight = torch.rand(hidden) * 0.5 + 0.5
    with torch.no_grad():
        a = hf(prefix, delta, blocks, output_norm_weight=weight)
        b = mc(prefix, delta, blocks, output_norm_weight=weight)
    same = torch.equal(a, b)
    return _report(
        f"前向逐位（hidden={hidden}）", same, f"max|Δ| = {float((a - b).abs().max()):.3e}"
    )


def check_backward(hidden: int = 64) -> bool:
    """C2：逐参数比梯度的反传逐位。"""
    hf, mc = _pair(hidden)
    torch.manual_seed(1)
    prefix = torch.randn(2, 6, hidden)
    delta = torch.randn(2, 6, hidden)
    blocks = torch.randn(2, 6, 3, hidden)
    weight = torch.rand(hidden) * 0.5 + 0.5
    hf(prefix, delta, blocks, output_norm_weight=weight).sum().backward()
    mc(prefix, delta, blocks, output_norm_weight=weight).sum().backward()
    bad = []
    for name, param in hf.named_parameters():
        g_hf, g_mc = param.grad, dict(mc.named_parameters())[name].grad
        if g_hf is None or g_mc is None or not torch.equal(g_hf, g_mc):
            bad.append(name)
    return _report(
        "反传逐位（逐参数比梯度）",
        not bad,
        f"{len(list(hf.named_parameters()))} 个张量" + (f"；不一致：{bad}" if bad else "全部逐位相等"),
    )


def check_solver() -> bool:
    """C3：同一映射下两侧求解器的返回值与梯度逐位一致。"""
    torch.manual_seed(0)
    dim = 16
    a = torch.randn(dim, dim) * 0.05

    def build(fn_solve, param):
        def func(x):
            return [x @ a + param]

        state = [torch.zeros(1, dim)]
        out = fn_solve(func, state, max_iter=16, tol=1e-8, stop_mode="rel", tau=1.0, grad_steps=1)
        out[0].sum().backward()
        return out[0].detach(), param.grad.clone()

    p1 = torch.nn.Parameter(torch.randn(1, dim))
    p2 = torch.nn.Parameter(p1.detach().clone())
    v1, g1 = build(hf_solve, p1)
    v2, g2 = build(mcore_solve, p2)
    return _report(
        "求解器值+梯度逐位",
        torch.equal(v1, v2) and torch.equal(g1, g2),
        f"值 max|Δ| = {float((v1 - v2).abs().max()):.3e}，梯度 max|Δ| = {float((g1 - g2).abs().max()):.3e}",
    )


def check_edges() -> bool:
    """C4：空 blocks / read_heads 退化 / lambda 自由 三种边界口径一致。"""
    ok = True
    hf, mc = _pair(48, looma_read_heads=8)  # 48 能被 8 整除，读头不退化
    torch.manual_seed(2)
    prefix = torch.randn(2, 5, 48)
    delta = torch.randn(2, 5, 48)
    empty = torch.zeros(1, 5, 0, 48).expand(2, -1, -1, -1)
    with torch.no_grad():
        a, b = hf(prefix, delta, empty), mc(prefix, delta, empty)
    ok &= _report("空 blocks（无行可读）", torch.equal(a, b), f"max|Δ| = {float((a - b).abs().max()):.3e}")

    # hidden=64 与 heads=6 不整除，两侧都退化为 1 头
    hf, mc = _pair(64, looma_read_heads=6)
    assert hf.read_heads == mc.read_heads == 1, (hf.read_heads, mc.read_heads)
    torch.manual_seed(3)
    prefix = torch.randn(2, 4, 64)
    blocks = torch.randn(2, 4, 2, 64)
    with torch.no_grad():
        a, b = hf(prefix, prefix, blocks), mc(prefix, prefix, blocks)
    ok &= _report(
        "read_heads 不整除时两侧都退化为 1 头", torch.equal(a, b), f"max|Δ| = {float((a - b).abs().max()):.3e}"
    )

    hf, mc = _pair(64, looma_lambda_clamp=None)
    with torch.no_grad():
        a, b = hf(prefix, prefix, blocks), mc(prefix, prefix, blocks)
    ok &= _report(
        "lambda_clamp=None", torch.equal(a, b), f"max|Δ| = {float((a - b).abs().max()):.3e}"
    )
    return bool(ok)


def check_init(hidden: int = 64) -> bool:
    """C5：两边的零初始化锚点一致（g_scale、载体偏置、decay_tau 阶梯与两侧权重半区）。"""
    torch.manual_seed(0)
    hf = HFConnection(_hf_config(hidden, 2))
    mc = MCoreConnection(hidden, _mcore_config(2), eps=1e-6)
    checks = {
        "g_scale 全零": bool((hf.g_scale == 0).all() and (mc.g_scale == 0).all()),
        "decay_tau 阶梯": bool(torch.allclose(hf.decay_tau, mc.decay_tau)),
        "写载体 -4": bool(
            hf.gate_proj[1].bias[2 * hidden :].eq(-4).all()
            and mc.gate_proj[1].bias[2 * hidden :].eq(-4).all()
        ),
        "up 半边零": bool(
            (hf.gate_proj[1].weight == 0).all() and (mc.gate_proj[1].weight == 0).all()
        ),
        "down 半边非零（不锁死）": bool(
            (hf.gate_proj[0].weight != 0).any() and (mc.gate_proj[0].weight != 0).any()
        ),
    }
    bad = [k for k, v in checks.items() if not v]
    return _report("初始化锚定一致", not bad, "、".join(checks) if not bad else f"不满足：{bad}")


def main(argv: list[str] | None = None) -> int:
    """依次跑 C5 初始化、C1 前向、C2 反传、C3 求解器、C4 边界并汇总，返回退出码。"""
    ap = argparse.ArgumentParser(description="Looma 连接层：mcore vs HF 逐位对拍")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    _ = args

    print("[looma · parity] mcore 侧 models/megatron vs HF 侧 models/transformers（fp32，逐位）")
    ok = True
    print("\nC5 初始化锚定")
    ok &= check_init()
    print("\nC1 前向")
    ok &= check_forward(64)
    ok &= check_forward(32)  # hidden 小于 rank(64)：两侧都要把 rank 夹到 hidden
    print("\nC2 反传")
    ok &= check_backward()
    print("\nC3 求解器")
    ok &= check_solver()
    print("\nC4 边界口径")
    ok &= check_edges()
    print(f"\n[looma · parity] {'全部通过' if ok else '有失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
