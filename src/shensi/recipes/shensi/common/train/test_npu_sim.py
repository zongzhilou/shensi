#!/usr/bin/env python3
"""昇腾（TE-NPU）路径的**模拟测试**：不用真 NPU，把 recipe 的 NPU 分支真跑一遍。

背景：目标机是 8×910B2，`transformer_impl=transformer_engine` 在那边解析成 TE-NPU，device 选择、
分布式后端（hccl）、verl 平台、vendor adaptor 的装载顺序、DSA 内核回退这些都在 recipe 侧有 NPU 分支。
本机没有 NPU，所以用 `npu_sim/` 里的 `torch_npu` / `megatron_adaptor` 替身把它们跑起来——
替身只覆盖**选择与决策**层（device 枚举、后端名、平台名、hccl choice、adaptor 装载顺序、
fail-closed 语义），不提供真实算子/通信；内核与通信由真机的预检
（`validation/2026-10-02/ascend_preflight.*`）兜。

跑法（本机即可）：

    python -m shensi.recipes.shensi.common.train.test_npu_sim

覆盖（每条都打 `[PASS]`）：
  ① `backend_name()` 自动识别（`torch_npu` 在 → npu）、显式覆盖（cuda/ascend/非法值）
  ② `device()` 选中 npu 并 set_device；`LOCAL_RANK` 生效
  ③ fail-closed：显式 NPU 但不可用 → 报错，绝不静默回退 CPU
  ④ `visible_devices()` 认 `ASCEND_RT_VISIBLE_DEVICES`
  ⑤ `configure_verl_env()`：NPU 走 `VERL_PLATFORM=huawei` 并保留 vLLM 插件；CUDA 侧清空插件
  ⑥ `bootstrap_megatron()` 在第一次 megatron/TE 导入前装载 `megatron_adaptor`（且只装一次）
  ⑦ `extend_distributed_backend_choices()` 给 mcore 的 parser 补上 `hccl`
  ⑧ `isolated_rng()` 在 npu 后端下可用且用完复位
  ⑨ recipe 侧的真决策：`runtime.setup()` 在 NPU 下不炸；DSA 内核回退把 `dsa_kernel_backend` 定成 `none`
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path

PASSES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSES
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"FAILED: {name}")
    PASSES += 1


@contextlib.contextmanager
def env(**kv: str):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def install_sim() -> str:
    """把 npu_sim 放到 sys.path 最前面并导入替身；返回它所在的目录（收尾时撤销）。"""
    sim_dir = str(Path(__file__).resolve().parent / "npu_sim")
    if sim_dir in sys.path:
        sys.path.remove(sim_dir)
    sys.path.insert(0, sim_dir)
    importlib.import_module("torch_npu")
    return sim_dir


def uninstall_sim(sim_dir: str) -> None:
    """撤销替身：进程不许一直"以为自己在昇腾上"。"""
    import torch

    for name in ("megatron_adaptor", "torch_npu"):
        sys.modules.pop(name, None)
    if sim_dir in sys.path:
        sys.path.remove(sim_dir)
    if hasattr(torch, "npu"):
        delattr(torch, "npu")


def fake_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed-backend", dest="distributed_backend", choices=["nccl", "gloo"]
    )
    return parser


def main() -> int:
    import torch

    sim_dir = install_sim()
    try:
        check("昇腾替身已装上（torch.device('npu') 可构造）", torch.device("npu").type == "npu")

        from shensi.utils import accelerator as acc

        # ---- ① backend 识别与覆盖 ----
        check("自动识别：torch_npu 在 → npu", acc.backend_name() == "npu", acc.backend_name())
        check("显式 cuda 覆盖生效", acc.backend_name({"SHENSI_BACKEND": "cuda"}) == "cuda")
        check("ascend 别名 → npu", acc.backend_name({"SHENSI_BACKEND": "ascend"}) == "npu")
        check(
            "ASCEND_HOME_PATH 也能识别",
            acc.backend_name({"ASCEND_HOME_PATH": "/usr/local/Ascend"}) == "npu",
        )
        try:
            acc.backend_name({"SHENSI_BACKEND": "tpu"})
            check("非法 SHENSI_BACKEND 报错", False)
        except ValueError:
            check("非法 SHENSI_BACKEND 报错", True)

        # ---- ② device 选择 ----
        dev = acc.device()
        check("device() 选中 npu:0", str(dev) == "npu:0", str(dev))
        check("set_device 落到 torch.npu", torch.npu.current_device() == 0)
        with env(LOCAL_RANK="3"):
            dev3 = acc.device()
        check("LOCAL_RANK 生效", str(dev3) == "npu:3", str(dev3))

        # ---- ③ fail-closed：显式 NPU 不可用时不许回退 CPU ----
        with env(SHENSI_BACKEND="npu", SHENSI_SIM_NPU_AVAILABLE="0"):
            try:
                acc.device()
                check("NPU 不可用时报错（不回退）", False)
            except RuntimeError as exc:
                check("NPU 不可用时报错（不回退）", "is_available" in str(exc), str(exc)[:60])

        # ---- ④ 可见设备 ----
        check(
            "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 → 4",
            acc.visible_devices({"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"}) == 4,
        )
        check("没有可见设备变量 → device_count", acc.visible_devices({}) == 1)

        # ---- ⑤ verl 平台选择 ----
        # 真实的输入来自 `dict(os.environ)`：键要么在、要么不在（不会出现 None 值）
        npu_env = acc.configure_verl_env({"VLLM_PLUGINS": ""})
        check(
            "NPU：VERL_PLATFORM=huawei",
            npu_env.get("VERL_PLATFORM") == "huawei",
            npu_env.get("VERL_PLATFORM", ""),
        )
        check(
            "NPU：空 VLLM_PLUGINS 被摘掉（否则 vllm-ascend 插件被禁）",
            "VLLM_PLUGINS" not in npu_env,
        )
        check("NPU：SHENSI_BACKEND 落成 npu", npu_env.get("SHENSI_BACKEND") == "npu")
        try:
            acc.configure_verl_env({"VERL_PLATFORM": "nvidia_noipc"})
            check("NPU 上给了 CUDA 平台 → 报错", False)
        except ValueError as exc:
            check("NPU 上给了 CUDA 平台 → 报错", "huawei" in str(exc), str(exc)[:60])
        cuda_env = acc.configure_verl_env({"SHENSI_BACKEND": "cuda"})
        check("CUDA 侧：nvidia_noipc + 清空插件", cuda_env.get("VERL_PLATFORM") == "nvidia_noipc")
        check("CUDA 侧：VLLM_PLUGINS 置空", cuda_env.get("VLLM_PLUGINS") == "")

        # ---- ⑥ vendor adaptor 装载顺序 ----
        check("adaptor 尚未装载", "megatron_adaptor" not in sys.modules)
        acc.bootstrap_megatron()
        check("bootstrap_megatron 装载 megatron_adaptor", "megatron_adaptor" in sys.modules)
        acc.bootstrap_megatron()  # 只装一次
        check("重复调用幂等", True)

        # ---- ⑦ hccl choice ----
        parser = acc.extend_distributed_backend_choices(fake_parser())
        choices = next(a.choices for a in parser._actions if a.dest == "distributed_backend")  # noqa: SLF001
        check("parser 补上 hccl", "hccl" in choices, str(choices))

        # ---- ⑧ isolated_rng 在 npu 后端可用 ----
        before = torch.get_rng_state().clone()
        with acc.isolated_rng(1234):
            inside = torch.get_rng_state().clone()
        after = torch.get_rng_state().clone()
        check(
            "isolated_rng 生效（内部 RNG 被换、外面复位）",
            bool((before == inside).all()) is False and bool((before == after).all()),
        )

        # ---- ⑨ recipe 侧的真决策 ----
        import shensi.runtime as rt

        rt.setup()
        check("runtime.setup() 在 NPU 下不炸", True)
        from megatron.bridge.models.shensi.shensi_provider import ShensiModelProvider

        check(
            "DSA 内核回退已挂到 provider.finalize",
            bool(getattr(ShensiModelProvider.finalize, "_shensi_dsa_fallback", False)),
        )
        from shensi.recipes.shensi.common import tiny_model

        provider = tiny_model.make_tiny_shensi_provider(seq_length=128)
        # 先把后端清成"未定"，再 finalize —— 验证的是**回退补丁真的在 NPU 上做决策**，
        # 而不是构造器里写死的值。
        provider.dsa_kernel_backend = None
        provider.finalize()
        check(
            "NPU 无 flash_mla/cudnn → finalize 把 dsa_kernel_backend 定成 none（PyTorch 回退）",
            str(getattr(provider, "dsa_kernel_backend", None)) == "none",
            str(getattr(provider, "dsa_kernel_backend", None)),
        )
        provider.dsa_kernel_backend = "none"  # 复位成可用状态，避免影响后续
    finally:
        uninstall_sim(sim_dir)

    print(f"\nALL CHECKS PASSED（{PASSES} 条；覆盖选择/决策层，内核与通信由真机预检兜）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
