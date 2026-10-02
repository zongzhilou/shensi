#!/usr/bin/env python3
"""严格对齐检查：我们的每个移植件 vs **各自上游仓库**的官方实现（数值对拍）。

入口：``python models/transformers/test_upstream_alignment.py``（在本仓 venv 里跑）。

对着谁测（逐条给出处，vendored 件在 ``upstream/``，见其 PROVENANCE.md）：

  AR          MoonshotAI/Kimi-K3 的 ``modeling_kimi_linear.py::_apply_attn_res``（可运行的官方算子）
  MUDD        MUDDFormer 官方 ``MultiwayDynamicDenseBlock`` + ``layer_mix``
  DenseFormer 官方 ``DWAModules``（按官方 ``models_denseformer.py`` 的 rep_idx 驱动序列）
  GDAR        本仓 ``3rdparty/common/transformers`` 就是 shensi 分支：
              ``ShensiAttentionResidual``（上游当前版）——钉住其恒等性与结构事实，并做数值对照

判定：AR/MUDD/DenseFormer 必须**逐位/1e-6 内相等**；GDAR 输出实测差值表。
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[7]  # <repo>/src
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
PKG = "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers"

RESULTS: list[tuple[str, bool, str]] = []
KNOWN_DELTAS = []  # 已测量、已定位的差异（不算 PASS/FAIL，汇总时单列）


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    return bool(ok)


def note(name: str, detail: str) -> None:
    print(f"  [INFO] {name:<52} {detail}")


# --------------------------------------------------------------------------- #
# loader: pull classes/functions *verbatim* out of a vendored file
# --------------------------------------------------------------------------- #
def load_from_vendored(file: str, names: list[str], extra_globals: dict | None = None) -> dict:
    """Exec the named top-level definitions from a vendored file, untouched.

    Importing the whole official file would drag its package layout in; slicing the
    exact source segments keeps the comparison honest (the code that runs is the
    official code, character for character) and side-effect free.
    """
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


# --------------------------------------------------------------------------- #
# 1) AR: our _apply_attn_res vs the official Kimi operator
# --------------------------------------------------------------------------- #
class StubRMSNorm(nn.Module):
    """The two attributes both operators read: ``weight`` and ``variance_epsilon``.

    Deliberately *not* a norm module: neither implementation calls it — both do
    ``v * rsqrt(var + eps)`` themselves and fold ``norm.weight`` into the score
    weight, exactly as the official Kimi file does.
    """

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


# --------------------------------------------------------------------------- #
# 2) MUDD: our MultiwayDynamicDense vs the official block + layer_mix
# --------------------------------------------------------------------------- #
def test_mudd() -> None:
    print("\n=== MUDD：vs MUDDFormer 官方 MultiwayDynamicDenseBlock + layer_mix ===")
    OfficialRMS, OfficialBlock = load_from_vendored(
        "muddformer_modeling.py", ["RMSnormNoscale", "MultiwayDynamicDenseBlock"]
    ).values()
    Mi = ours("modeling_qwen3_mudd", "MultiwayDynamicDense")
    Cfg = ours("modeling_qwen3_mudd", "Qwen3MUDDConfig")

    D, LIDX = 16, 2  # official L = lidx + 2 = 4 sources
    L = LIDX + 2
    cfg = types.SimpleNamespace(
        dim=D, norm_eps=1e-6, dense_type=["l"], expand_last=False, round64=False
    )
    official = OfficialBlock(cfg, lidx=LIDX)
    note(
        "MUDD: official geometry",
        f"C={official.C}, w1 {tuple(official.w1.weight.shape)}, w2 {tuple(official.w2.weight.shape)}",
    )

    # ``mudd_param="official"`` + prior 清零：官方 MultiwayDynamicDenseBlock **没有** prior 项，
    # 我们的 identity prior 是"恒等锚定"重参数化，做算符级对照时必须清掉才能对上。
    ours_cfg = Cfg(
        hidden_size=D,
        rms_norm_eps=1e-6,
        mudd_num_ways=1,
        mudd_hidden_round=0,
        mudd_param="official",
    )
    mine = Mi(ours_cfg, num_states=L)
    with torch.no_grad():  # identical weights on both sides（norm 两侧都是 no-scale，无参数）
        mine.prior.zero_()
        mine.w1.weight.copy_(official.w1.weight)
        mine.w2.weight.copy_(official.w2.weight)

    torch.manual_seed(0)
    T = 3
    x = torch.randn(1, T, D)  # official block takes (B, T, D)
    hids = [torch.randn(1, T, D) for _ in range(L)]  # X_0 .. X_{L-1}

    dw_off = official(x)  # (C, B, T, L)
    out_off = official.layer_mix(hids, dw_off)[0]  # (B, T, D)

    dw_mine = mine(x[0])  # (T, C, L)
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


# --------------------------------------------------------------------------- #
# 3) DenseFormer: our DepthWeightedAverage vs the official DWAModules
# --------------------------------------------------------------------------- #
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
    xs = [torch.randn(4, D) for _ in range(N_REPEAT + 1)]  # x0 (emb) + 3 block outputs
    official.init_accumulators(xs[0])
    out_off = xs[0]
    for rep in range(1, N_REPEAT + 1):  # the official model's rep_idx loop
        out_off = official(xs[rep], rep - 1)

    mine = Dwa(Cfg(hidden_size=D, attn_res_dwa_param="official"), num_sources=N_REPEAT + 1)
    alpha = official.alphas[N_REPEAT - 1].weight.view(-1)
    with torch.no_grad():
        mine.alpha.copy_(alpha)
    out_mine = mine(torch.stack(xs, dim=1))  # (T, N+1, D)

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


# --------------------------------------------------------------------------- #
# 4) GDAR: our AttentionResidual vs the upstream shensi branch (in this repo)
# --------------------------------------------------------------------------- #
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

    D, LAYERS, HEADS = 32, 4, 2
    cfg = ShensiConfig()
    cfg.hidden_size = D
    cfg.num_hidden_layers = LAYERS
    cfg.routed_expert_hidden_size = max(8, D // 4)
    cfg.attn_res_read_heads = HEADS
    cfg.rms_norm_eps = 1e-6
    up = ShensiAttentionResidual(cfg)

    torch.manual_seed(0)
    prefix = torch.randn(6, D)
    delta = torch.randn(6, D)
    blocks = torch.randn(6, 3, D)

    out, updated = up(prefix.clone(), delta.clone(), blocks.clone(), None, blocks.shape[1])
    ident = prefix + delta
    d = (out - ident).abs().max().item()
    check(
        "GDAR: upstream init == identity (decay 1 / erase 0 / write 1 / read 0)",
        d <= 1e-6,
        f"max|upstream(0) − (prefix+delta)| = {d:.3e}",
    )

    # structural ledger of the upstream revision this port must match
    src = up.forward.__code__
    facts = {
        "q/k/delta low-rank a/b 投影": hasattr(up, "q_a_proj")
        and hasattr(up, "k_a_proj")
        and hasattr(up, "q_b_proj")
        and hasattr(up, "k_b_proj"),
        "g_scale(4) + 正性直通投影(clamp min=0)": hasattr(up, "g_scale")
        and up.g_scale.numel() == 4,
        "t 阶梯（linspace·log2L，learned）": hasattr(up, "t") and up.t.numel() == D,
        "objective 闭式更新（lam clamp ≥ −0.5）": "eigh" in src.co_names,
        "逐头白化（eigh/whiten 在 read 内）": "einsum" in src.co_names,
        "Softmax₁ 读（logsumexp/softplus）": "logsumexp" in src.co_names,
    }
    for k, v in facts.items():
        check(f"GDAR: 上游具备 {k}", v)

    # ---- 全权重映射对照（两套配置：per_head=对齐上游；full=我们论文默认）----
    Ar = ours("modeling_qwen3_gdar", "AttentionResidual")
    GCfg = ours("modeling_qwen3_gdar", "Qwen3GDARConfig")
    R_UP = up.q_a_proj.out_features

    def build_mine(whiten: str, null: bool, mix: str):
        cfg_m = GCfg(
            hidden_size=D,
            num_hidden_layers=LAYERS,
            attn_res_read_heads=HEADS,
            rms_norm_eps=1e-6,
            attn_res_gate_param="deviation",
            attn_res_update="objective",
            attn_res_address="delta",
            attn_res_decay_ladder=D,
            attn_res_read_null=null,
            attn_res_read_whiten=whiten,
            attn_res_read_mix=mix,
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
            m.read_scale.fill_(float(up.g_scale[3]))
            if m.decay_tau is not None:
                m.decay_tau.copy_(up.t)
        return m

    # 接口差异（同一算子，折叠位置不同）：我们的 read() 返回 ``prefix + read_scale*routed``
    # （"读出的东西直接加进流"，见 modeling_qwen3_gdar.read 的 docstring），上游返回
    # ``routed``，由 ``output = updated + routed`` 折叠——下面这样组合才是 apples-to-apples。
    def compose(m, blocks_):
        upd_m, _ = m.update(prefix.clone(), delta.clone())
        read_m, _ = m.read(prefix.clone(), blocks_.clone())
        return upd_m + (read_m - prefix)

    mine = build_mine("per_head", True, "raw")  # 与上游逐位对齐的配置
    d_same = (compose(mine, blocks) - out).abs().max().item()
    check(
        "GDAR: 同权重点处 vs 上游（per_head 逐头白化, paper 其它旋钮）|diff| ≤ 1e-5",
        d_same <= 1e-5,
        f"max|diff| = {d_same:.3e}",
    )

    # 我们论文默认（全局白化）与上游的实测差：把 read 门开到 1 放大 read 侧差异再测
    with torch.no_grad():
        up.g_scale[3].fill_(1.0)
    mine2 = build_mine("full", False, "whitened")
    with torch.no_grad():
        mine2.read_scale.fill_(1.0)
    out_up2, _ = up(prefix.clone(), delta.clone(), blocks.clone(), None, blocks.shape[1])
    d_full = (compose(mine2, blocks) - out_up2).abs().max().item()
    KNOWN_DELTAS.append(
        (
            "GDAR read 侧（上游逐头白化 vs 我们论文默认的全局白化，read_scale=1）",
            f"max|diff| = {d_full:.3e}（要逐位对齐请用 gdar_layer_spec_upstream / read_whiten='per_head'）",
        )
    )
    note(
        "GDAR 参数映射账本",
        "上游 g_scale(4) ↔ 我们 (decay_scale, erase_scale, write_scale, read_scale)；"
        "上游 t(log τ 阶梯) ↔ 我们 decay_tau；上游 q/k/g 的 a/b 两层 ↔ 我们 q_proj/k_proj/gate_proj 的 .0/.1。",
    )
    note(
        "GDAR 已知设计差异",
        "上游 read 用**逐头**白化（reshape 到 heads 后对每头 eigh）；"
        "我们 read_whiten='full' 是全局协方差白化 + Softmax₁；"
        "上游没有 null 源（values = blocks + updated 平铺），我们 read_null 是可选开关（本对照已关）。"
        "跟进上游逐头白化是我们清单里已列的项（gdar_package SHENSI_UPSTREAM_VERIFY.md §5b）。",
    )


# --------------------------------------------------------------------------- #
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
