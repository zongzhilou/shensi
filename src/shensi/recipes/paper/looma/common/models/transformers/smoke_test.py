"""Looma 的 HF 侧冒烟与锚定检查（纯脚本，不依赖 pytest）。

以 ``python -m shensi.recipes.paper.looma.common.models.transformers.smoke_test`` 运行，
依次检查前向/反传、初始化恒等锚定、读的静默性、求解器、参数开销与存档往返。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

try:
    from .configuration_looma import LoomaConfig
    from .modeling_looma import LoomaAttentionResidual, LoomaForCausalLM, solve_block
except ImportError:
    # 直接运行文件时把本目录加入 sys.path，再按顶层模块导入
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configuration_looma import LoomaConfig
    from modeling_looma import LoomaAttentionResidual, LoomaForCausalLM, solve_block


def tiny_config(**overrides) -> LoomaConfig:
    """构造 tiny 几何的配置（2 层 / hidden 64 / 2 头 / 1 个 KV 头 / head_dim 32）。"""
    knobs = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )
    knobs.update(overrides)
    return LoomaConfig(**knobs)


def check_identity(config: LoomaConfig) -> tuple[bool, str]:
    """S2：连接模块在初始化处的输出逐位等于 ``norm(x) * weight``。"""
    torch.manual_seed(0)
    hidden = config.hidden_size
    module = LoomaAttentionResidual(config).eval()
    weight = torch.rand(hidden) * 0.5 + 0.5
    x = torch.randn(3, 5, hidden)
    # blocks 有内容，但 read_scale = 0 时读不进来
    blocks = torch.randn(3, 5, 2, hidden)
    prefix = torch.randn(3, 5, hidden)

    with torch.no_grad():
        out = module(x, None, blocks, output_norm_weight=weight)
        reference = nn.functional.rms_norm(x, (hidden,), weight, eps=config.rms_norm_eps)

    same = torch.equal(out, reference)
    detail = f"max|Δ| = {float((out - reference).abs().max()):.3e}"
    return same, detail


def check_read_silent(config: LoomaConfig) -> tuple[bool, str]:
    """S3：``read_scale = 0`` 时 blocks 的内容完全不影响输出。"""
    torch.manual_seed(0)
    hidden = config.hidden_size
    module = LoomaAttentionResidual(config).eval()
    prefix = torch.randn(2, 4, hidden)
    a = torch.zeros(2, 4, 3, hidden)
    b = torch.randn(2, 4, 3, hidden) * 5.0
    with torch.no_grad():
        out_a = module(prefix, prefix, a)
        out_b = module(prefix, prefix, b)
    same = torch.equal(out_a, out_b)
    return same, f"max|Δ| = {float((out_a - out_b).abs().max()):.3e}"


def check_solver() -> list[tuple[str, bool, str]]:
    """S4：求解器在 ``x -> A x + b`` 上收敛到 ``(I - A)^-1 b``，梯度口径符合声明。"""
    torch.manual_seed(0)
    dim = 16
    a = torch.randn(dim, dim) * 0.05
    b = torch.randn(dim, dim)

    def func(x):
        return [x @ a + b]

    x0 = [torch.zeros(dim, dim)]
    out = solve_block(func, x0, max_iter=64, tol=1e-10, stop_mode="rel", tau=1.0, grad_steps=0)[0]
    exact = b @ torch.linalg.inv(torch.eye(dim) - a)
    converged = bool(torch.allclose(out, exact, atol=1e-6))
    results = [("收敛到解析不动点", converged, f"max|Δ| = {float((out - exact).abs().max()):.3e}")]

    # 一步 phantom 梯度 == 一步展开的梯度（对参数 b 求导）
    param = nn.Parameter(b.clone())

    def func_p(x):
        return [x @ a + param]

    state = [torch.zeros(dim, dim)]
    fused = solve_block(func_p, state, max_iter=1, tol=1e-10, grad_steps=1)[0]
    fused.sum().backward()
    one_step = param.grad.clone()

    param2 = nn.Parameter(b.clone())

    def func_p2(x):
        return [x @ a + param2]

    z = solve_block(func_p2, [torch.zeros(dim, dim)], max_iter=1, tol=1e-10, grad_steps=0)[0]
    no_grad_ok = not z.requires_grad and param2.grad is None
    # 一步展开：z1 = f(0) = b，∂z1/∂b = 1（单位张量）
    results.append(("grad_steps=1 的梯度 == 一步展开", bool(torch.allclose(one_step, torch.ones_like(b))), f"mean = {float(one_step.mean()):.3f}"))
    results.append(("grad_steps=0 不产生梯度", no_grad_ok, ""))
    return results


def _grad_norms(model) -> dict[str, float]:
    return {
        n: float(p.grad.abs().sum())
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def check_forward_backward(config: LoomaConfig, device: str) -> list[tuple[str, bool, str]]:
    """S1：前向/反传跑通，且初始化处的梯度结构与设计一致。

    初始化时只有 deviation scale 与读门带着活梯度，门的权重与 q/k 投影在零点处梯度为零；
    把 scale 推出零点之后，门的权重与 q/k 投影必须拿到非零梯度，也不允许有永久死掉的张量。
    """
    torch.manual_seed(0)
    model = LoomaForCausalLM(config).to(device)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    out = model(input_ids=input_ids, labels=input_ids.clone())
    out.loss.backward()
    loss = float(out.loss.detach())
    finite = torch.isfinite(out.loss).item()

    grads = _grad_norms(model)
    alive = {n for n, v in grads.items() if v > 0}
    escape = {n for n in alive if n.endswith("g_scale")}
    frozen = [n for n, v in grads.items() if v == 0 and "attn_res" in n]
    n_gscale = sum(1 for n in grads if n.endswith("g_scale"))

    results = [
        ("前向 loss 有限", bool(finite), f"loss = {loss:.4f}"),
        (
            "初始化只放活逃逸路径（g_scale）",
            len(escape) == n_gscale and n_gscale > 0,
            f"{len(escape)}/{n_gscale} 个 g_scale 有梯度，其余连接张量 {len(frozen)} 个为零",
        ),
    ]

    # 梯度逐级苏醒：scale → decay 门 → up 半边 → down 半边，每步显式把活着的一级推一步
    alive_names: set[str] = set()
    step1 = None
    for step in range(4):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "attn_res" not in n:
                    continue
                if step == 0:
                    if n.endswith("g_scale"):
                        # 四个 scale 一起转正
                        p.fill_(0.05)
                elif n in alive_names or n.endswith(("g_scale", "decay_tau")):
                    # 把已有梯度的那一级往前推一步，相当于训练中的前几次 optimizer step
                    p.add_(torch.randn_like(p) * 0.05)
        model.zero_grad()
        out = model(input_ids=input_ids, labels=input_ids.clone())
        out.loss.backward()
        alive = {n for n, v in _grad_norms(model).items() if v > 0}
        if step == 1:
            step1 = alive
        alive_names |= alive
    conn = [n for n in _grad_norms(model) if "attn_res" in n]
    awake = [n for n in conn if n in alive_names]
    results.append(
        (
            "scale 转正后 decay 门/时间常数醒过来",
            any(n.endswith("decay_tau") for n in (step1 or set())),
            f"第二步新增 {len(step1 or set()) - len(escape)} 个活张量",
        )
    )
    # down 半边要多一步：它的梯度 ∝ up 的权重，up 动过之后才非零
    dead = [n for n in conn if n not in awake]
    results.append(
        (
            "梯度流跑满之后没有永久死掉的连接张量",
            not dead,
            f"{len(awake)}/{len(conn)} 个有梯度" + (f"；仍为零：{dead}" if dead else ""),
        )
    )

    # scale <= 0 时 decay 恒为 1、门无梯度，但 straight-through clamp 让 scale 自己仍有梯度
    torch.manual_seed(0)
    probe = LoomaAttentionResidual(config)
    with torch.no_grad():
        probe.g_scale.fill_(-0.05)
    x = torch.randn(2, 4, config.hidden_size)
    probe(x, x, x.unsqueeze(-2)).sum().backward()
    gate_bias_live = float(probe.gate_proj[1].bias.grad[: config.hidden_size].abs().sum())
    scale_live = float(probe.g_scale.grad[0].abs())
    results.append(
        (
            "decay scale <= 0 时门惰性但 scale 不卡死",
            gate_bias_live == 0.0 and scale_live > 0.0,
            f"门偏置梯度 {gate_bias_live:.1e}，scale 梯度 {scale_live:.3e}",
        )
    )
    return results


def param_report(config: LoomaConfig) -> str:
    """S5：相对同几何 plain Llama 的增量参数报告。"""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    llama_cfg = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    plain = LlamaForCausalLM(llama_cfg)
    looma = LoomaForCausalLM(config)

    def count(m, pred=lambda n: True):
        return sum(p.numel() for n, p in m.named_parameters() if pred(n))

    plain_total = count(plain)
    looma_total = count(looma)
    extra = [
        (n, p.numel())
        for n, p in looma.named_parameters()
        if "attn_res" in n or ".output_attn_res" in n
    ]
    lines = [
        f"plain Llama  {plain_total:>10,}",
        f"Looma        {looma_total:>10,}  (+{looma_total - plain_total:,}, "
        f"{100.0 * (looma_total - plain_total) / plain_total:.1f}%)",
        f"其中连接模块 {sum(v for _, v in extra):>10,}（{len(extra)} 个张量）",
    ]
    return "\n".join(lines)


def preset_config(**overrides) -> LoomaConfig:
    """非默认旋钮的 tiny 配置：用于验证旋钮本身也随 ``config.json`` 往返。

    每个旋钮都落在真实生效的位置上，例如 ``looma_output_route=False`` 会删掉末端连接，
    所以"逐位相等"不是靠默认值碰巧对上。
    """
    knobs = dict(
        looma_max_iter=3,
        looma_tol=1e-3,
        looma_stop_mode="abs",
        looma_tau=0.5,
        looma_read_heads=4,
        looma_rank=8,
        looma_lambda_clamp=None,
        looma_output_route=False,
        looma_write_carrier_bias=-2.0,
    )
    knobs.update(overrides)
    return tiny_config(**knobs)


def check_save_load(config: LoomaConfig, device: str) -> list[tuple[str, bool, str]]:
    """S6：``save_pretrained`` → ``from_pretrained``（不传 config）逐位相等。"""
    torch.manual_seed(0)
    model = LoomaForCausalLM(config).to(device).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 12), device=device)
    knobs_in = {k: getattr(model.config, k) for k in config.__class__.__annotations__ if k.startswith("looma_")}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "looma-tiny"
        model.save_pretrained(out, safe_serialization=True)
        files = sorted(p.name for p in out.iterdir())
        has_code = (
            "configuration_looma.py" in files
            and "modeling_looma.py" in files
            and "config.json" in files
        )
        from transformers import AutoModelForCausalLM

        reloaded = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).to(device).eval()
        with torch.no_grad():
            a = model(input_ids).logits
            b = reloaded(input_ids).logits
        same = torch.equal(a, b)
        knobs_out = {
            k: getattr(reloaded.config, k) for k in config.__class__.__annotations__ if k.startswith("looma_")
        }
        detail = f"max|Δ| = {float((a - b).abs().max()):.3e}；导出文件 {files}"
    mismatched = {k: (knobs_in[k], knobs_out[k]) for k in knobs_in if knobs_in[k] != knobs_out[k]}
    return [
        ("导出目录自带建模代码", has_code, ""),
        ("旋钮随 config.json 往返", not mismatched, str(mismatched) if mismatched else f"{len(knobs_in)} 个旋钮一致"),
        ("往返逐位相等", same, detail),
    ]


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    return ok


def main(argv: list[str] | None = None) -> int:
    """命令行入口：依次运行各项检查，全部通过时返回 0。"""
    ap = argparse.ArgumentParser(description="Looma HF 冒烟与锚定检查")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--skip-save", action="store_true", help="跳过存档往返（要 import vLLM/慢盘时有用）")
    args = ap.parse_args(argv)

    ok = True
    config = tiny_config()
    print("[looma · HF] tiny 几何：2 层 / hidden 64 / 2 heads / 1 KV / head_dim 32")

    print("\nS2 恒等锚定（连接输出 == 加权 RMSNorm，逐位）")
    _ok, detail = check_identity(config)
    ok &= _report("Looma(0) == Llama pre-norm", _ok, detail)

    print("\nS3 读在初始化处静默")
    _ok, detail = check_read_silent(config)
    ok &= _report("read_scale=0 ⇒ blocks 内容不影响输出", _ok, detail)

    print("\nS4 求解器")
    for name, _ok, detail in check_solver():
        ok &= _report(name, _ok, detail)

    print(f"\nS1 前向/反传（device={args.device}）")
    for name, _ok, detail in check_forward_backward(config, args.device):
        ok &= _report(name, _ok, detail)

    print("\nS5 参数开销（同几何对照）")
    print(param_report(config))

    if not args.skip_save:
        print("\nS6 存档往返（save_pretrained → from_pretrained，非默认旋钮）")
        for name, _ok, detail in check_save_load(preset_config(), args.device):
            ok &= _report(name, _ok, detail)

    print(f"\n[looma · HF] {'全部通过' if ok else '有失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
