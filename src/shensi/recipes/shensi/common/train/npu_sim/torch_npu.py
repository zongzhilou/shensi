"""测试专用的 `torch_npu` 替身：把 recipe 的 NPU 分支在没有 NPU 的机器上真跑一遍。

放置位置在 `npu_sim/` 里、由 `test_npu_sim.py` 临时加进 `sys.path`——**不要**把本目录常驻
`PYTHONPATH`（那会让所有进程以为自己在昇腾上）。真机不需要它（真 torch_npu 自带这些）。

行为刻意做成"看起来像昇腾"：
- `torch.utils.rename_privateuse1_backend("npu")` → `torch.device("npu", 0)` 可构造（真 torch_npu 同款）；
- `torch.npu` 命名空间：`is_available` / `device_count` / `set_device` / `current_device` /
  `synchronize` / `manual_seed_all` / `get_rng_state` / `set_rng_state`；
- `is_available()` 默认 True；置环境变量 `SHENSI_SIM_NPU_AVAILABLE=0` 可模拟"NPU 不在"，
  用来验证 recipe 的 fail-closed 分支（显式请求 NPU 时绝不静默回退 CPU）。

不提供的：真实的算子/内存/通信（`torch.zeros(..., device="npu")` 会失败）——模拟测试只覆盖
**选择与决策**这一层，内核层由真机的 `validation/*/ascend_preflight` 与预检脚本兜。
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

__version__ = "0.0.0+sim"

# 与真 torch_npu 一致：先把 privateuse1 改名为 npu，torch.device("npu") 才认得
torch.utils.rename_privateuse1_backend("npu")

_current = int(os.environ.get("SHENSI_SIM_NPU_DEVICE", "0"))
_rng_states: dict[int, torch.Tensor] = {}


def is_available() -> bool:
    return os.environ.get("SHENSI_SIM_NPU_AVAILABLE", "1") != "0"


def device_count() -> int:
    if not is_available():
        return 0
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        return max(1, len([x for x in visible.split(",") if x.strip() and x != "-1"]))
    return 1


def set_device(index: int) -> None:
    global _current
    _current = int(index)


def current_device() -> int:
    return _current


def get_device_name(index: int | None = None) -> str:
    return f"Ascend910B2-sim:{_current if index is None else int(index)}"


def synchronize(index: int | None = None) -> None:
    return None


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(int(seed) % (2**31 - 1))


def get_rng_state(device: int | str | None = None):
    """`torch.random.fork_rng(device_type="npu")` 会取它；给一个确定性的 CPU 张量即可。"""
    return _rng_states.get(_current, torch.zeros(8, dtype=torch.uint8)).clone()


def set_rng_state(state: torch.Tensor, device: int | str | None = None) -> None:
    _rng_states[_current] = state.clone()


def _reset_for_tests() -> None:
    """测试之间复位（真机上没有这个函数，测试里按需调用）。"""
    global _current
    _current = int(os.environ.get("SHENSI_SIM_NPU_DEVICE", "0"))
    _rng_states.clear()


class _NotRealNPUError(RuntimeError):
    pass


def _stub_op(name: str):
    """下游库（transformers 的 npu flash-attn 集成等）**导入期**会探测这些算子名。

    导入要能过、真调用要响亮地失败——所以给一个"调用即报错"的桩，而不是静默 dummy。
    """

    def _op(*args, **kwargs):
        raise _NotRealNPUError(
            f"torch_npu.{name} 是 npu_sim 的桩：本机没有真 NPU，模拟测试只覆盖选择/决策层；"
            "要跑这个算子请在 910B2 上装真 torch_npu。"
        )

    _op.__name__ = name
    return _op


_EXTRA_OP_NAMES = (
    "npu_fusion_attention",
    "npu_fusion_attention_grad",
    "npu_rms_norm",
    "npu_rotary_mul",
    "npu_swiglu",
    "npu_scaled_masked_softmax",
    "npu_apply_rotary_pos_emb",
    "npu_prompt_flash_attention",
    "npu_incre_flash_attention",
)


def __getattr__(name: str):
    """仅导入期可见的兜底：`from torch_npu import <算子>` 能过，调用时报错。"""
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _EXTRA_OP_NAMES:
        return _stub_op(name)
    raise AttributeError(
        f"npu_sim: torch_npu.{name} 未提供（模拟器只实现选择/决策层；真算子请上 910B2）"
    )


# 暴露成 `torch.npu`（真 torch_npu 导入后也是这个效果）
torch.npu = SimpleNamespace(
    is_available=is_available,
    device_count=device_count,
    set_device=set_device,
    current_device=current_device,
    get_device_name=get_device_name,
    synchronize=synchronize,
    manual_seed_all=manual_seed_all,
    get_rng_state=get_rng_state,
    set_rng_state=set_rng_state,
    _reset_for_tests=_reset_for_tests,
)
