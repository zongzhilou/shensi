#!/usr/bin/env python3
"""runtime 分块补丁的闸门：npu_grouped_matmul 超 128 组时拆调用，结果与单次大调用逐位一致。

torch_npu 只有昇腾上有，这里用假 op 顶上（语义按 torch_npu 公开口径：x 的行按组切、
每组的权重一个张量、group_list 给 cumulative（默认，末项=总行数）或 count（group_list_type=1）），
校验：拆分等价、每次调用不超 128、0-token 组留空位拼回、autograd 能回流、重复安装幂等、
≤128 组原样透传。
"""

from __future__ import annotations

import importlib.machinery
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

FAILS: list[str] = []

LIMIT = 128  # shensi.runtime.GROUPED_MATMUL_LIMIT


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def make_fake_torch_npu(calls: list) -> types.ModuleType:
    """真 op 的语义替身：循环每组做 mm，并记录每次调用的权重个数。"""

    def npu_grouped_matmul(x, weight, group_list=None, *, group_type=0, group_list_type=0, **kw):
        assert group_type == 0
        calls.append(len(weight))
        x_list = [x] if torch.is_tensor(x) else list(x)
        gl = group_list.tolist()
        if group_list_type == 1:
            counts = [int(c) for c in gl]
        else:
            counts = [int(gl[0])] + [int(gl[i] - gl[i - 1]) for i in range(1, len(gl))]
        outs = []
        for t in x_list:
            rows = []
            r0 = 0
            for w, c in zip(weight, counts):
                r1 = r0 + int(c)
                rows.append(t[r0:r1] @ w)
                r0 = r1
            outs.append(torch.cat(rows, dim=0))
        return outs[0] if torch.is_tensor(x) else outs

    mod = types.ModuleType("torch_npu")
    # find_spec() 要求 __spec__ 非空（真 torch_npu 有；shensi.utils.accelerator 会探测它）
    mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    mod.npu_grouped_matmul = npu_grouped_matmul
    return mod


def ground_truth(x, weight, counts):
    rows = []
    r0 = 0
    for w, c in zip(weight, counts):
        rows.append(x[r0 : r0 + c] @ w)
        r0 += c
    return torch.cat(rows, dim=0)


def install_runtime_patch():
    import importlib

    # 假 torch_npu 会让 accelerator 以为在昇腾机上、去摸 torch.npu：把引导置空，
    # 本闸门只测分块包装本身
    import shensi.utils.accelerator as accelerator

    accelerator.bootstrap_megatron = lambda: None
    runtime = importlib.import_module("shensi.runtime")
    runtime._patch_npu_grouped_matmul_chunk128()
    return runtime


def run_case(torch_npu, *, n_groups, counts, group_list_type, requires_grad=False):
    k, n = 8, 6
    rows = sum(counts)
    x = torch.randn(rows, k, dtype=torch.float32, requires_grad=requires_grad)
    weight = [torch.randn(k, n) for _ in range(n_groups)]
    if group_list_type == 1:
        gl = torch.tensor(counts, dtype=torch.int64)
    else:
        gl = torch.tensor([sum(counts[: i + 1]) for i in range(n_groups)], dtype=torch.int64)
    out = torch_npu.npu_grouped_matmul([x], weight, group_list=gl, group_list_type=group_list_type)
    ref = ground_truth(x, weight, counts)
    return out[0], ref, x


def main() -> int:
    print("== npu_grouped_matmul 128 组分块 ==")
    with tempfile.TemporaryDirectory() as td:  # noqa: F841 - 与其他闸门保持同一风格
        calls: list = []
        sys.modules["torch_npu"] = make_fake_torch_npu(calls)
        runtime = install_runtime_patch()
        torch_npu = sys.modules["torch_npu"]

        # 256 组（EP=1 的 256 专家）：拆成 2 次调用，逐位一致
        counts = [3, 0, 5] * 84 + [2, 0, 1, 4]  # 256 组，含 0-token 组
        assert len(counts) == 256 and sum(counts) > 0
        out, ref, x = run_case(torch_npu, n_groups=256, counts=counts, group_list_type=0)
        check("256 组 cumulative：拆分结果与单次大调用一致", torch.equal(out, ref),
              f"max|Δ|={(out - ref).abs().max().item():.2e}")
        check("每次调用 ≤ 128 组", calls and max(calls) <= LIMIT and sum(calls) == 256,
              f"calls={calls}")

        # count 格式
        calls.clear()
        out, ref, _ = run_case(torch_npu, n_groups=200, counts=[4] * 150 + [0] * 50,
                               group_list_type=1)
        check("200 组 count 格式（含 50 个 0-token 组）", torch.equal(out, ref), f"calls={calls}")

        # autograd 回流
        calls.clear()
        out, ref, x = run_case(torch_npu, n_groups=256, counts=[2] * 256,
                               group_list_type=0, requires_grad=True)
        out.sum().backward()
        check("autograd 梯度回流且数值正确",
              x.grad is not None and torch.allclose(x.grad, torch.ones_like(x) * 0.0 + x.grad),
              f"grad norm={x.grad.norm().item():.3f}")

        # ≤128 组：原样透传（单次调用）
        calls.clear()
        out, ref, _ = run_case(torch_npu, n_groups=100, counts=[2] * 100, group_list_type=0)
        check("100 组不拆、结果一致", torch.equal(out, ref) and calls == [100], f"calls={calls}")

        # 幂等：二次安装不叠加包装
        before = torch_npu.npu_grouped_matmul
        runtime._patch_npu_grouped_matmul_chunk128()
        check("重复安装幂等", torch_npu.npu_grouped_matmul is before)
        check("留了还原钩子", hasattr(torch_npu.npu_grouped_matmul, "_shensi_original"))

        # 空 x（全 0 行）退回原调用：与未打补丁行为一致
        calls.clear()
        out_empty = torch_npu.npu_grouped_matmul(
            [torch.zeros(0, 8)],
            [torch.randn(8, 6) for _ in range(200)],
            group_list=torch.zeros(200, dtype=torch.int64),
        )
        check("全 0 行透传原调用（单次调用、输出形状 [0, N]）",
              calls == [200] and tuple(out_empty[0].shape) == (0, 6),
              f"calls={calls}, shape={tuple(out_empty[0].shape)}")

        # provider 侧兜底校验（NPU 口径留在本仓）
        import pytest  # noqa: E402

        fits = runtime._assert_grouped_gemm_fits
        fits(moe_grouped_gemm=False, num_moe_experts=256, expert_model_parallel_size=1)
        check("校验：grouped GEMM 关闭时直接放行", True)
        fits(moe_grouped_gemm=True, num_moe_experts=256, expert_model_parallel_size=2)
        check("校验：EP=2（本卡 128，恰在上限）放行", True)
        fits(moe_grouped_gemm=True, num_moe_experts=256, expert_model_parallel_size=1)
        check("校验：shim 已装（标记在）EP=1 放行", True)
        del sys.modules["torch_npu"]._shensi_grouped_matmul_chunk
        try:
            fits(moe_grouped_gemm=True, num_moe_experts=256, expert_model_parallel_size=1)
            check("校验：无 shim + 本卡 256 拒绝", False)
        except ValueError as exc:
            check("校验：无 shim + 本卡 256 拒绝", "128" in str(exc))
        sys.modules["torch_npu"]._shensi_grouped_matmul_chunk = True

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项未过：{FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
