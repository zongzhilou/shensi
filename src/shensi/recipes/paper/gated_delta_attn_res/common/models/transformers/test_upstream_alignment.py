#!/usr/bin/env python3
"""与各自上游实现的逐位对拍（AR / MUDD / DenseFormer / GDAR）。"""

from __future__ import annotations

import ast
import math
import pathlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[7]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
PKG = "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers"

RESULTS: list[tuple[str, bool, str]] = []
KNOWN_DELTAS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    return bool(ok)


def note(name: str, detail: str) -> None:
    print(f"  [INFO] {name:<52} {detail}")


def load_from_vendored(file: str, names: list[str], extra_globals: dict | None = None) -> dict:
    text = (UPSTREAM / file).read_text(encoding="utf-8")
    tree = ast.parse(text)
    picked: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            picked.append(ast.get_source_segment(text, node))
    found = {n for n in names if n in text}
    missing = [n for n in names if not any(f"class {n}" in s or f"def {n}" in s for s in picked)]
    if missing:
        raise RuntimeError(f"{file}: {missing} not found at top level (found={sorted(found)})")
    ns: dict = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Tensor": torch.Tensor,
        "list": list,
        "tuple": tuple,
        "dict": dict,
    }
    if extra_globals:
        ns.update(extra_globals)
    try:
        from einops import rearrange  # noqa: PLC0415

        ns["rearrange"] = rearrange
    except Exception:  # pragma: no cover
        pass
    exec(compile("\n\n".join(picked), f"<vendored:{file}>", "exec"), ns)
    return {n: ns[n] for n in names}


def ours(module: str, name: str):
    import importlib

    return getattr(importlib.import_module(f"{PKG}.{module}"), name)


class StubRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps


def test_ar() -> None:
    print("\n=== AR：vs MoonshotAI/Kimi-K3 的 _apply_attn_res（官方逐行）===")
    (official,) = load_from_vendored("kimi_modeling_kimi_linear.py", ["_apply_attn_res"]).values()
    mi = ours("modeling_qwen3_ar", "_apply_attn_res")

    torch.manual_seed(0)
    T, H, N = 3, 8, 5
    prefix = torch.randn(T, H)
    blocks = torch.randn(T, N, H)
    norm = StubRMSNorm(H)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(H) + 0.5)
    proj = nn.Linear(H, 1, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.randn_like(proj.weight))

    a = official(prefix.clone(), blocks.clone(), proj, norm)
    b, probs = mi(prefix.clone(), blocks.clone(), proj, norm, return_probs=True)
    check(
        "AR: official operator == ours (bit-exact)",
        torch.equal(a, b),
        f"max|diff| = {(a - b).abs().max().item():.3e}; probs.sum() = {probs.sum().item():.6f}",
    )


def test_mudd() -> None:
    print("\n=== MUDD：vs MUDDFormer 官方 MultiwayDynamicDenseBlock + layer_mix ===")
    OfficialRMS, OfficialBlock = load_from_vendored(
        "muddformer_modeling.py", ["RMSnormNoscale", "MultiwayDynamicDenseBlock"]
    ).values()
    Mi = ours("modeling_qwen3_mudd", "MultiwayDynamicDense")
    Cfg = ours("modeling_qwen3_mudd", "Qwen3MUDDConfig")

    D, LIDX = 16, 2
    L = LIDX + 2
    cfg = types.SimpleNamespace(
        dim=D, norm_eps=1e-6, dense_type=["l"], expand_last=False, round64=False
    )
    official = OfficialBlock(cfg, lidx=LIDX)
    note(
        "MUDD: official geometry",
        f"C={official.C}, w1 {tuple(official.w1.weight.shape)}, w2 {tuple(official.w2.weight.shape)}",
    )

    ours_cfg = Cfg(
        hidden_size=D,
        rms_norm_eps=1e-6,
        mudd_num_ways=1,
        mudd_hidden_round=0,
        mudd_param="official",
    )
    mine = Mi(ours_cfg, num_states=L)
    with torch.no_grad():
        mine.prior.zero_()
        mine.w1.weight.copy_(official.w1.weight)
        mine.w2.weight.copy_(official.w2.weight)

    torch.manual_seed(0)
    T = 3
    x = torch.randn(1, T, D)
    hids = [torch.randn(1, T, D) for _ in range(L)]

    dw_off = official(x)
    out_off = official.layer_mix(hids, dw_off)[0]

    dw_mine = mine(x[0])
    (out_mine,) = mine.aggregate(dw_mine, torch.stack(hids, dim=2)[0])

    check(
        "MUDD: generated weights dw == official (bit-exact)",
        torch.equal(dw_mine[:, 0, :], dw_off[0, 0]),
        f"ours {tuple(dw_mine.shape)}, official {tuple(dw_off.shape)}",
    )
    d = (out_off[0] - out_mine).abs().max().item()
    check("MUDD: layer_mix output ≈ official", d <= 1e-6, f"max|diff| = {d:.3e}")

    official2 = OfficialBlock(cfg, lidx=LIDX)
    note(
        "MUDD: official released init",
        f"|w2|max = {official2.w2.weight.abs().max().item():.3e} → NOT identity "
        f"(identity 是我们的 deviation/official 重参数化，已在 docs 标注)",
    )


def test_denseformer() -> None:
    print("\n=== DenseFormer：vs 官方 DWAModules（按官方 models_denseformer.py 的驱动序列）===")
    InPlace, apply_set, OfficialDWA = load_from_vendored(
        "denseformer_denseformer.py", ["InPlaceSetSlice", "apply_inplace_set", "DWAModules"]
    ).values()

    Dwa = ours("modeling_qwen3_denseformer", "DepthWeightedAverage")
    Cfg = ours("modeling_qwen3_denseformer", "Qwen3DenseFormerConfig")

    N_REPEAT, D = 3, 16
    official = OfficialDWA(N_REPEAT, dilation=1, period=1)

    torch.manual_seed(0)
    xs = [torch.randn(4, D) for _ in range(N_REPEAT + 1)]
    official.init_accumulators(xs[0])
    out_off = xs[0]
    for rep in range(1, N_REPEAT + 1):
        out_off = official(xs[rep], rep - 1)

    mine = Dwa(Cfg(hidden_size=D, attn_res_dwa_param="official"), num_sources=N_REPEAT + 1)
    alpha = official.alphas[N_REPEAT - 1].weight.view(-1)
    with torch.no_grad():
        mine.alpha.copy_(alpha)
    out_mine = mine(torch.stack(xs, dim=1))

    d = (out_off - out_mine).abs().max().item()
    check(
        "DenseFormer: DWA output == official (same alphas)",
        d <= 1e-6,
        f"max|diff| = {d:.3e}; alpha = {[round(v, 4) for v in alpha.tolist()]}",
    )
    check(
        "DenseFormer: official init alpha == one_hot(last)（恒等）",
        torch.equal(alpha.float(), torch.tensor([0.0, 0.0, 0.0, 1.0])),
        f"official alpha after _init_weights = {alpha.tolist()}",
    )


def test_gdar() -> None:
    print("\n=== GDAR：vs 本仓 shensi 分支的 ShensiAttentionResidual（上游当前版）===")
    try:
        from transformers.models.shensi.configuration_shensi import ShensiConfig  # noqa: PLC0415
        from transformers.models.shensi.modular_shensi import (  # noqa: PLC0415
            ShensiAttentionResidual,
        )
    except Exception as exc:  # pragma: no cover
        check("GDAR: upstream module importable", False, f"{type(exc).__name__}: {exc}")
        return

    D, LAYERS = 32, 4
    # 本机 huggingface_hub 对上游配置的哨兵默认值做严格校验：显式给合法 int（不影响数值）
    cfg = ShensiConfig(n_shared_experts=1)
    cfg.hidden_size = D
    cfg.num_hidden_layers = LAYERS
    cfg.routed_expert_hidden_size = max(8, D // 4)
    cfg.rms_norm_eps = 1e-6
    up = ShensiAttentionResidual(cfg)

    torch.manual_seed(0)
    prefix = torch.randn(6, D)
    delta = torch.randn(6, D)
    blocks = torch.randn(6, 3, D)

    # 无块时读退化为 0，初始化仍恒等；有块时读在 init 即注入 blocks 的 softmax 加权均值
    torch.manual_seed(11)
    out0, _ = up(prefix.clone(), delta.clone(), blocks[:, :0].clone(), None, 0)
    d = (out0 - (prefix + delta)).abs().max().item()
    check(
        "GDAR: 上游无块时初始化恒等（decay 1 / erase 0 / write 1 / read 0）",
        d <= 1e-6,
        f"max|upstream(0, 无块) − (prefix+delta)| = {d:.3e}",
    )

    src = up.forward.__code__
    facts = {
        "q/k/delta low-rank a/b 投影": hasattr(up, "q_a_proj")
        and hasattr(up, "k_a_proj")
        and hasattr(up, "q_b_proj")
        and hasattr(up, "k_b_proj"),
        "g_scale(4) + 正性直通投影(clamp min=0)": hasattr(up, "g_scale")
        and up.g_scale.numel() == 4,
        "t 阶梯（linspace·log2L，learned）": hasattr(up, "t") and up.t.numel() == D,
        "写路 objective 闭式（normalize / addcmul / lam clamp）": {"normalize", "addcmul", "clamp"}
        <= set(src.co_names),
        "抽样度量读（randint + svd_lowrank + unravel_index）": {
            "randint",
            "svd_lowrank",
            "unravel_index",
        }
        <= set(src.co_names),
        "读的软路由与门（softmax + sigmoid(lse)·read_scale）": {"softmax", "logsumexp", "sigmoid"}
        <= set(src.co_names),
    }
    for k, v in facts.items():
        check(f"GDAR: 上游具备 {k}", v)

    Ar = ours("modeling_qwen3_gdar", "AttentionResidual")
    GCfg = ours("modeling_qwen3_gdar", "Qwen3GDARConfig")
    R_UP = up.q_a_proj.out_features

    def build_mine(read_scale: float, whiten: str = "spike", heads: int = 1, null: bool = False):
        cfg_m = GCfg(
            hidden_size=D,
            num_hidden_layers=LAYERS,
            rms_norm_eps=1e-6,
            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_address="delta",
            attn_res_decay_ladder=D,
            attn_res_read_heads=heads,
            attn_res_read_null=null,
            attn_res_read_whiten=whiten,
            attn_res_read_mix="raw",
            attn_res_gate_rank=R_UP,
            attn_res_q_rank=R_UP,
            attn_res_k_rank=R_UP,
        )
        m = Ar(cfg_m)
        m.eval()
        with torch.no_grad():
            m.gate_proj[0].weight.copy_(up.g_a_proj.weight)
            m.gate_proj[0].bias.copy_(up.g_a_proj.bias)
            m.gate_proj[1].weight.copy_(up.g_b_proj.weight)
            m.gate_proj[1].bias.copy_(up.g_b_proj.bias)
            m.q_proj[0].weight.copy_(up.q_a_proj.weight)
            m.q_proj[1].weight.copy_(up.q_b_proj.weight)
            m.k_proj[0].weight.copy_(up.k_a_proj.weight)
            m.k_proj[1].weight.copy_(up.k_b_proj.weight)
            m.decay_scale.fill_(float(up.g_scale[0]))
            m.erase_scale.fill_(float(up.g_scale[1]))
            m.write_scale.fill_(float(up.g_scale[2]))
            m.read_scale.fill_(read_scale)
            if m.decay_tau is not None:
                m.decay_tau.copy_(up.t)
        return m

    def upstream_forward(seed: int):
        torch.manual_seed(seed)
        return up(prefix.clone(), delta.clone(), blocks.clone(), None, blocks.shape[1])

    def mirror_forward(m, seed: int):
        torch.manual_seed(seed)
        return m.forward(prefix.clone(), delta.clone(), blocks.clone(), None, blocks.shape[1])

    # 写前状态两侧共用（上游 forward 内部就是 norm(prefix+delta)，抄公式避开私有访问）
    pooled = prefix + delta
    state = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)

    # 抽样度量读吃全局 RNG（两次 randint + svd_lowrank 内部 randn）：两侧各自重播种
    for read_scale in (0.0, 1.0):
        with torch.no_grad():
            up.g_scale[3].fill_(read_scale)
        mine = build_mine(read_scale)
        out_up, upd_up = upstream_forward(1234)
        out_mine, upd_mine, _, _ = mirror_forward(mine, 1234)
        d_write = (upd_mine - upd_up).abs().max().item()
        d_all = (out_mine - out_up).abs().max().item()
        check(
            f"GDAR: 同权重点处 vs 上游（spike 度量读, read_scale={read_scale:g}, 同 RNG 重播种）|diff| ≤ 1e-6",
            max(d_all, d_write) <= 1e-6,
            f"max|Δout| = {d_all:.3e}, max|Δupdated| = {d_write:.3e}",
        )

    # 读路单独对拍：我们的 read(..., updated=写后) − prefix == 上游 out − updated
    with torch.no_grad():
        up.g_scale[3].fill_(1.0)
    mine = build_mine(1.0)
    out_up, upd_up = upstream_forward(77)
    torch.manual_seed(77)
    upd_mine, _ = mine.update(prefix.clone(), delta.clone())
    read_mine, _ = mine.read(prefix.clone(), blocks.clone(), state=state, updated=upd_mine)
    d_read = ((read_mine - prefix) - (out_up - upd_up)).abs().max().item()
    check(
        "GDAR: 读路单独对拍 vs 上游（spike 度量, read_scale=1, 同权重同 RNG）|diff| ≤ 1e-6",
        d_read <= 1e-6,
        f"max|Δread| = {d_read:.3e}",
    )

    # 论文主行（全局白化 + Softmax₁ 读）vs 上游抽样度量读：设计差异，记录不判失败
    paper = build_mine(1.0, whiten="full", heads=8, null=True)
    out_paper, upd_paper, _, _ = mirror_forward(paper, 5)
    out_up5, _ = upstream_forward(5)
    d_paper = (out_paper - out_up5).abs().max().item()
    KNOWN_DELTAS.append(
        (
            "GDAR 主行读（全局白化 + Softmax₁, read_whiten='full'）vs 上游抽样度量读",
            f"max|diff| = {d_paper:.3e}（设计差异；逐位对齐用 gdar_layer_spec_upstream / read_whiten='spike'）",
        )
    )
    note(
        "GDAR 参数映射账本",
        "上游 g_scale(4) ↔ 我们 (decay_scale, erase_scale, write_scale, read_scale)；"
        "上游 t(log τ 阶梯, linspace·log2L) ↔ 我们 decay_tau（ladder=hidden）；"
        "上游 q/k/g 的 a/b 两层 ↔ 我们 q_proj/k_proj/gate_proj 的 .0/.1。",
    )
    note(
        "GDAR 上游读的映射",
        "上游读＝抽样度量（K=min(2H, M·B) 行、二阶矩向 1 收缩、一次 svd_lowrank + MP 边 + BBP 去偏），"
        "度量只进查询，末位源是写后状态，其权重乘 g=sigmoid(lse)·read_scale；"
        "对应我们 read_whiten='spike'（层内读出点尚无写后状态时末位源退回 prefix）。",
    )
    note(
        "GDAR 已知设计差异",
        "上游读已从逐头白化换成抽样度量；论文主行的全局白化 + Softmax₁ 读与它有量级可见的差异"
        "（见 known deltas），属设计差异。上游没有 null 源（values = blocks + updated 平铺），"
        "我们 read_null 是可选开关（本对照已关）。",
    )


def main() -> int:
    print("=" * 104)
    print("upstream alignment: our ports vs each variant's own upstream repository")
    print("=" * 104)
    test_ar()
    test_mudd()
    test_denseformer()
    test_gdar()

    hard = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 104)
    if KNOWN_DELTAS:
        print("known deltas (measured, tracked):")
        for name, detail in KNOWN_DELTAS:
            print(f"  - {name}: {detail}")
    print(f"summary: {len(RESULTS) - len(hard)}/{len(RESULTS)} checks passed")
    if hard:
        print("FAILED:")
        for name, _, detail in hard:
            print(f"  - {name}: {detail}")
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
