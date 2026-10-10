"""读路径加速（attn_res_fast）的契约测试：算术不变、幂等、开关与精度设置生效。

对应 benchmarks（生产几何、M=32768 位置、nb=10）：eager 46.4 ms → compile 24.6 ms（1.89x），
+TF32 17.9 ms（2.59x）；`num_blocks=0`（不做采样）时 eager 与编译**逐位一致**。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from megatron.bridge.models.shensi.modeling_shensi import ShensiAttentionResidual

from shensi.recipes.shensi.common.train import attn_res_fast

HIDDEN, RANK, LAYERS, EPS = 64, 16, 4, 1e-6


def _module() -> ShensiAttentionResidual:
    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        routed_expert_hidden_size=RANK,
        num_layers=LAYERS,
        num_hidden_layers=LAYERS,
        layernorm_epsilon=EPS,
        attn_res_pp_state_transfer=False,
        recompute_granularity=None,
        fp32_keep=None,
        bf16=False,
        fp16=False,
        params_dtype=torch.float32,
        bf16_attention=False,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=None,
        fp8_param=False,
    )
    return ShensiAttentionResidual(cfg).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="读路径 compile 只在 CUDA 上生效")
def test_compile_is_idempotent_and_reported():
    module = _module().cuda()
    chunks = [module]

    first = attn_res_fast.enable_attn_res_fast_path(chunks, mode="compile", log=lambda *a, **k: None)
    assert first["compiled"] == 1
    assert first["skipped"] is None
    assert getattr(module, attn_res_fast._MARK) is True

    second = attn_res_fast.enable_attn_res_fast_path(chunks, mode="compile", log=lambda *a, **k: None)
    assert second["compiled"] == 0, "重复调用不应二次包裹"


def test_mode_none_is_a_noop():
    module = _module()
    summary = attn_res_fast.enable_attn_res_fast_path([module], mode="none", log=lambda *a, **k: None)
    assert summary == {"mode": "none", "matmul_precision": None, "compiled": 0, "skipped": None}
    assert not hasattr(module, attn_res_fast._MARK)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="未知的读路径加速档"):
        attn_res_fast.enable_attn_res_fast_path([_module()], mode="fastest", log=lambda *a, **k: None)


def test_matmul_precision_is_applied_and_reported():
    before = torch.get_float32_matmul_precision()
    try:
        summary = attn_res_fast.enable_attn_res_fast_path(
            [_module()], mode="none", matmul_precision="high", log=lambda *a, **k: None
        )
        assert summary["matmul_precision"] == "high"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="读路径 compile 只在 CUDA 上生效")
def test_compiled_read_is_bit_exact_without_sampling():
    """不做采样（num_blocks=0）时，编译与 eager 的输出逐位一致：只换了执行方式。"""
    module = _module().cuda().to(torch.bfloat16)
    torch.manual_seed(0)
    prefix = torch.randn(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    slots = [torch.zeros(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    with torch.no_grad():
        eager_out, eager_updated = module(prefix, delta, slots, None, 0)
    attn_res_fast.enable_attn_res_fast_path([module], mode="compile", log=lambda *a, **k: None)
    with torch.no_grad():
        torch.manual_seed(0)
        fast_out, fast_updated = module(prefix, delta, slots, None, 0)

    assert torch.equal(eager_out, fast_out)
    assert torch.equal(eager_updated, fast_updated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="读路径 compile 只在 CUDA 上生效")
def test_compiled_read_keeps_the_write_path_exact():
    """采样那部分换的是"抽签序列"（同一估计器的另一次实现）；写入路径仍逐位一致。"""
    module = _module().cuda().to(torch.bfloat16)
    torch.manual_seed(1)
    prefix = torch.randn(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    slots = [torch.randn(2, 5, 3, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    with torch.no_grad():
        torch.manual_seed(7)
        eager_out, eager_updated = module(prefix, delta, slots, None, 3)
    attn_res_fast.enable_attn_res_fast_path([module], mode="compile", log=lambda *a, **k: None)
    with torch.no_grad():
        torch.manual_seed(7)
        fast_out, fast_updated = module(prefix, delta, slots, None, 3)

    assert torch.equal(eager_updated, fast_updated)
    assert torch.isfinite(fast_out).all()
    diff = (eager_out.float() - fast_out.float()).abs().mean()
    scale = eager_out.float().abs().mean() + 1e-6
    assert float(diff) < 0.2 * float(scale), f"采样差异超出预期量级：{float(diff)} vs {float(scale)}"


def test_apply_from_args_follows_the_flags():
    """关档（none/highest）时不动任何东西；开档时进入编译路径。"""
    module = _module()
    off = attn_res_fast.apply_from_args(module, SimpleNamespace())
    assert off["mode"] == "none" and off["compiled"] == 0
    assert not hasattr(module, attn_res_fast._MARK)

    args = SimpleNamespace(shensi_attn_res_fast_path="compile", shensi_attn_res_matmul_precision="highest")
    summary = attn_res_fast.apply_from_args(module, args)
    if torch.cuda.is_available():
        assert summary["compiled"] == 1
        assert getattr(module, attn_res_fast._MARK) is True
    else:
        assert summary["skipped"] == "no-cuda"
