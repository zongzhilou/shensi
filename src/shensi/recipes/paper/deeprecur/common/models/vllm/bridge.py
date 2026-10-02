"""vLLM 桥：**原生推理路径 + 自研件留在 HF 侧**（榨干原生吞吐的正解）。

三臂的引擎落点：

- **native**（Qwen3-VL 原样）：vLLM 有**原生** ``Qwen3VLForConditionalGeneration``——不用本桥，
  注册表里已存在，直接按 Qwen3-VL 服务（paged attention / 图融合 / 原生权重加载全开）。
- **unified / gdar / deeprecur**（自研件）：走 vLLM 的 **Transformers 后端（多模态支）**
  ``TransformersMultiModalForCausalLM`` 作基类。引擎的融合/张量并行/分页注意力照常作用于主干，
  自研件（AR 读写、编码器-自由嵌入器、feedback 回注）在 ``recursive_replace`` 期间被遮成
  ``Identity``、之后原样恢复——即"主干原生、连接件自研"，避免为每套连接件手写插件。
"""

from __future__ import annotations

import contextlib

import torch.nn as nn

#: 自研件的名字标记（遮罩用；主干 self_attn/mlp/norm/embed 不在列，引擎照常融合）
CUSTOM_MARKERS = (
    "vision_embedder",
    "self_attention_attn_res",
    "mlp_attn_res",
    "output_attn_res_module",
    ".feedback",
)

#: 三臂在引擎里注册的架构名（= 我们模型类的类名，save_pretrained 会写进 config.architectures）
ARCHITECTURES = (
    "Qwen3VLGdarForConditionalGeneration",
    "Qwen3VLGdarDeepRecurForConditionalGeneration",
    "Qwen3VLUnifiedForConditionalGeneration",
)


def custom_module_names(model: nn.Module) -> list[str]:
    """列出要遮罩的自研模块（只取最外层，避免遮罩子模块）。"""
    names: list[str] = []
    for name, _module in model.named_modules():
        if not name or not any(marker in name for marker in CUSTOM_MARKERS):
            continue
        if any(name == kept or name.startswith(kept + ".") for kept in names):
            continue
        names.append(name)
    return names


@contextlib.contextmanager
def _masked(model: nn.Module, qualnames: list[str]):
    """把自研件临时换成 Identity（引擎的 recursive_replace 期间不受其干扰），退出时恢复。"""
    saved: list[tuple[nn.Module, str, nn.Module]] = []
    try:
        for qual in qualnames:
            parts = qual.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            module = getattr(parent, attr, None)
            if module is None:
                continue
            saved.append((parent, attr, module))
            setattr(parent, attr, nn.Identity())
        yield
    finally:
        for parent, attr, module in saved:
            setattr(parent, attr, module)


def _resolve(root: nn.Module, qualname: str):
    obj = root
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _base():
    """VLLM 的多模态 Transformers 后端基类（不可用则给显式报错类）。"""
    try:
        from vllm.model_executor.models.transformers import TransformersMultiModalForCausalLM

        return TransformersMultiModalForCausalLM
    except Exception:  # pragma: no cover - vLLM 不在本解释器时
        return None


_BASE = _base()

if _BASE is None:  # pragma: no cover

    class RecurTransformersForCausalLM(nn.Module):  # type: ignore[no-redef]
        """vLLM 缺失时的占位：显式报错，不静默。"""

        def __init__(self, *a, **kw):
            raise RuntimeError(
                "本解释器没有 vLLM 的多模态 Transformers 后端，桥不可用；"
                "先在装了 vllm 的环境里跑（同 .venv）"
            )

else:

    class RecurTransformersForCausalLM(_BASE):  # type: ignore[misc,valid-type]
        """主干走 vLLM 原生（paged attention / 融合 / TP），自研连接件走 HF 实现。"""

        #: 本次构建遮罩掉的自研件（构造后校验已恢复）
        custom_qualnames: list[str] = []

        def recursive_replace(self):
            model = self.model
            qualnames = custom_module_names(model)
            self.custom_qualnames = qualnames
            if not qualnames:
                return super().recursive_replace()
            with _masked(model, qualnames):
                super().recursive_replace()

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            missing = [q for q in self.custom_qualnames if _resolve(self.model, q) is None]
            if missing:  # pragma: no cover - 遮罩/恢复坏掉才会到这
                raise RuntimeError(f"自研件没有恢复：{missing}")

    def describe() -> str:
        """给文档/闸门用的一行说明。"""
        return (
            f"bridge={_BASE.__module__}.{_BASE.__name__}"
            f" | 自研件遮罩标记 {len(CUSTOM_MARKERS)} 条 | 注册架构 {len(ARCHITECTURES)} 个"
        )


__all__ = ["ARCHITECTURES", "CUSTOM_MARKERS", "RecurTransformersForCausalLM", "custom_module_names"]
