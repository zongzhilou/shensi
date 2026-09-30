"""起训前的一次性装置：权重加载探针、CPU 平台兼容。

都只做加法（包一层函数 / 注册一个 override），重复调用是幂等的；装不上只告警。
"""

from __future__ import annotations

import json
import os
import re

from megatron.bridge.models.shensi.model import ShensiModel
from megatron.training import get_args, print_rank_0

from shensi.utils.ckpt_digest import (
    digest_state_dict,
    sample_keys,
    summarize,
    tensor_digest,
)


def probe_rank() -> int:
    """本进程的全局 rank（没起分布式就是 0）。"""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _is_extra_state(key: str) -> bool:
    """`_extra_state` / `_extra_state<N>`（专家按 EP 分片各带一份）都算元数据，不算权重。"""
    return bool(re.search(r"_extra_state\d*$", key))


def install_load_probe() -> bool:
    """加载检查点时打 missing/unexpected 与权重摘要，并按需落 JSON。

    `--load` 指到 HF 权重目录、或 ckpt 与模型几何不一致时，上游会把 strict 失败降级成
    strict=False 只打日志；这里显式把缺口写出来（`SHENSI_LOAD_PROBE_STRICT=1` 时直接报错）。
    """
    if getattr(ShensiModel.load_state_dict, "_shensi_load_probe", False):
        return False
    original = ShensiModel.load_state_dict

    def probe(self, state_dict, *args, **kwargs):
        model_keys = {k for k in self.state_dict() if not _is_extra_state(k)}
        ckpt_keys = {k for k in state_dict if not _is_extra_state(k)}
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        n_extra = len(state_dict) - len(ckpt_keys)
        ret = original(self, state_dict, *args, **kwargs)
        sd_now = self.state_dict()
        digest, n = digest_state_dict(sd_now)
        try:
            args_now = get_args()
            load_dir = getattr(args_now, "load", None)
            save_dir = getattr(args_now, "save", None)
        except Exception:
            load_dir, save_dir = None, None
        strict = kwargs.get("strict", args[0] if args else True)
        print_rank_0(
            f"[shensi][load] successfully loaded checkpoint from {load_dir}: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"（模型侧 {len(model_keys)} 个权重键 / 检查点侧 {len(ckpt_keys)} 个，"
            f"_extra_state {n_extra} 个已排除）| 参数摘要 sha256={digest}（{n} 个张量）"
        )
        if missing:
            print_rank_0(f"[shensi][load] missing 明细（前 10）：{missing[:10]}")
        if unexpected:
            print_rank_0(f"[shensi][load] unexpected 明细（前 10）：{unexpected[:10]}")
        probe_strict = str(os.environ.get("SHENSI_LOAD_PROBE_STRICT", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if probe_strict and (missing or unexpected):
            raise RuntimeError(
                "[shensi][load-probe] SHENSI_LOAD_PROBE_STRICT=1 且加载有缺口："
                f"missing={len(missing)}（前 10：{missing[:10]}）"
                f"unexpected={len(unexpected)}（前 10：{unexpected[:10]}）。"
                "上游 checkpointing 会把 strict 失败吞掉降级成 strict=False，所以这里显式拦下。"
            )
        if save_dir and probe_rank() == 0:
            try:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(save_dir, "shensi_load_probe.json")
                info = summarize(sd_now)
                info.update(
                    {
                        "load": load_dir,
                        "missing": missing,
                        "unexpected": unexpected,
                        "num_extra_state_ckpt": n_extra,
                        "strict": strict,
                        "probe_rank": probe_rank(),
                        "per_tensor_sha256": {
                            k: tensor_digest(sd_now, k) for k in sample_keys(sd_now, n=8)
                        },
                    }
                )
                with open(out, "w") as f:
                    json.dump(info, f, indent=1, ensure_ascii=False)
                print_rank_0(f"[shensi][load] 探针 JSON 已写：{out}")
            except Exception as exc:
                print_rank_0(f"[shensi][load] 探针 JSON 写入失败（忽略）：{exc!r}")
        return ret

    probe._shensi_load_probe = True
    ShensiModel.load_state_dict = probe
    print_rank_0("[shensi][load] 已安装检查点加载探针（missing/unexpected + sha256 摘要）")
    return True


def install_cpu_platform_compat() -> bool:
    """昇腾/CPU 上 `get_device_arch_version()` 会抛 NotImplementedError：能注册就注册。

    `megatron.plugin.*` 由 MindSpeed 提供（`pyproject.ascend.toml` 那条安装链）；
    NVIDIA 侧的上游 mcore 没有这个插件机制，直接跳过。
    """
    try:
        from megatron.plugin.decorators import register_override_method
    except Exception as exc:
        print_rank_0(f"[shensi][cpu] 跳过 CPU 平台补丁（没有上游插件机制：{exc!r}）")
        return False

    def get_device_arch_version_cpu_safe():
        from megatron.plugin.platform import get_platform

        cur_platform = get_platform()
        try:
            return cur_platform.get_device_properties(cur_platform.device(0)).major
        except NotImplementedError:
            return None

    register_override_method(
        "common_utils.get_device_arch_version", get_device_arch_version_cpu_safe
    )
    return True


def install_all(args) -> None:
    """起训前的收尾装置；每一步失败只告警。"""
    for step in (install_cpu_platform_compat, install_load_probe):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            print_rank_0(f"[shensi][probes] {step.__name__} 跳过（{type(exc).__name__}: {exc}）")
    if getattr(args, "shensi_erc_loss_coef", 0.0):
        from shensi.recipes.shensi.train.erc import erc_group_plan

        plan = erc_group_plan(args)
        print_rank_0(
            f"[shensi][erc] 分组计划 {len(plan)} 组 → 每组 MoE 层数 {sorted(len(v) for v in plan.values())}"
        )
