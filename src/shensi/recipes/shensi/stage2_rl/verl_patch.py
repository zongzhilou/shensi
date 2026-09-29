"""verl 运行时补丁：非分布式优化器下 flat buffer 为 None 的判空。

上游 `verl/utils/megatron_utils.py::load_megatron_model_to_gpu` 无条件解引用
`buffer.param_data.storage()`，但同一文件另一处（`get_megatron_model_device` 一带）上游自己判了
`if buffer.param_data is None:` 并注明「use_distributed_optimizer=False: no flat param buffer」——
也就是说这是上游同一文件内的不一致：一个调用点漏判空。

我们不改上游文件，只在调用前把 `param_data`/`grad_data` 为 None 的 buffer 临时换成一个零长度
占位张量（`param_data_size` 在这种配置下本就是 0），让那段逻辑变成 no-op；调用结束后原样还原，
因为其它函数要靠 `param_data is None` 判断「没有 flat buffer」。
"""

from __future__ import annotations

import os

import torch


def _stub(tensor_like) -> tuple:
    """为 None 的 flat buffer 造一个零长度占位（device 跟随模型）。"""
    device = "cpu"
    try:
        module = tensor_like
        device = next(module.parameters()).device
    except Exception:
        pass
    return torch.empty(0, device=device), torch.empty(0)


def apply() -> str:
    """打上补丁；返回实际动作（没装 verl / 已打过都安全跳过）。"""
    try:
        from verl.utils import megatron_utils as mu
    except Exception as exc:  # 没有 verl 的进程（纯推理、torchrun 等）直接跳过
        return f"跳过（{type(exc).__name__}）"

    if getattr(mu, "_shensi_none_buffer_patch", False):
        return "已打过"
    original = mu.load_megatron_model_to_gpu

    def load_megatron_model_to_gpu(models, load_grad=True, load_frozen_params=False):
        saved = []
        for model_chunk in models:
            for buffers in list(getattr(model_chunk, "buffers", []) or []) + list(
                getattr(model_chunk, "expert_parallel_buffers", []) or []
            ):
                for buffer in buffers:
                    if getattr(buffer, "param_data", 1) is None:
                        saved.append((buffer, "param_data", None))
                        buffer.param_data, buffer.cpu_data = _stub(getattr(model_chunk, "module", None))
                        if hasattr(buffer, "param_data_size"):
                            saved.append((buffer, "param_data_size", buffer.param_data_size))
                            buffer.param_data_size = 0
                    if load_grad and hasattr(buffer, "grad_data_size") and getattr(buffer, "grad_data", 1) is None:
                        saved.append((buffer, "grad_data", None))
                        buffer.grad_data = torch.empty(0, device=buffer.param_data.device)
                        saved.append((buffer, "grad_data_size", buffer.grad_data_size))
                        buffer.grad_data_size = 0
        try:
            return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
        finally:
            for buffer, attr, value in reversed(saved):
                setattr(buffer, attr, value)

    mu.load_megatron_model_to_gpu = load_megatron_model_to_gpu
    mu._shensi_none_buffer_patch = True
    return "已应用"


def inject(env: dict, run_dir) -> None:
    """把补丁挂到子进程的启动路径上：一个 .pth（site 启动时 import verl_patch）+ 模块所在目录。

    .pth 由 site 自动处理，因此 torchrun / ray worker / vLLM 进程只要 PYTHONPATH 上有这个目录就会
    自动打补丁；这里只写一个几行的文件，不依赖 sitecustomize 的先后顺序。
    """
    from pathlib import Path

    patch_dir = Path(run_dir) / "_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "zz_shensi_verl_patch.pth").write_text("import verl_patch\n", encoding="utf-8")
    here = str(Path(__file__).resolve().parent)
    head = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{here}{os.pathsep}{patch_dir}{os.pathsep}{head}" if head else f"{here}{os.pathsep}{patch_dir}"
    )


if __name__ != "__main__":
    try:
        apply()
    except Exception:  # 启动期绝不因为兜底补丁把进程带崩
        pass
