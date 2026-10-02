#!/usr/bin/env python3
"""megatron 镜像：引用 Megatron-Bridge 的 Qwen3-VL provider，unified 由 HF config 驱动。

mbridge 的 ``Qwen3VLModelProvider`` 明确写着：provider 自己的 ``deepstack_visual_indexes``
字段不直接生效，**模型按 ``hf_config.deepstack_visual_indexes`` 走**（qwen3_vl_provider.py）。
所以 unified 变体在 mcore 侧不需要新 provider：检查点的 HF config 带
``deepstack_visual_indexes: []``（transformers 镜像产出），provider 转换/训练时自动关 deepstack。

另：mbridge 里还有 ``gemma_vl.gemma4_vl_provider``（Gemma 4 的 mcore 件），unified 若要按
Gemma 4 原生口径对拍可以用它，本配方默认仍走 Qwen3-VL provider。

python -m shensi.recipes.paper.deeprecur.common.models.megatron.bridge
"""

from __future__ import annotations

#: provider 导入路径（mbridge 检出在 3rdparty/common/Megatron-Bridge）
QWEN3_VL_PROVIDER = "megatron.bridge.models.qwen_vl.qwen3_vl_provider.Qwen3VLModelProvider"
GEMMA4_VL_PROVIDER = "megatron.bridge.models.gemma_vl.gemma4_vl_provider.Gemma4VLModelProvider"


def _load_provider_class(dotted: str):
    """按点号路径导入 provider 类（mbridge 不在 python path 时给出装配指引）。"""
    import importlib

    module_path, _, class_name = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"[deeprecur·megatron] 导不了 {module_path}：{exc}。"
            "mbridge 在 3rdparty/common/Megatron-Bridge/src，先按仓库 README 装进环境"
            "（或把它加到 PYTHONPATH）"
        ) from exc
    return getattr(module, class_name)


def unified_provider(**overrides):
    """Unified 口径的 provider 实例：deepstack 关闭 + 语言侧覆盖。

    返回 mbridge ``Qwen3VLModelProvider`` 实例；``deepstack_visual_indexes=[]`` 只是
    保险——真正生效的是 HF config（mbridge 按 hf_config 走）。
    """
    provider_cls = _load_provider_class(QWEN3_VL_PROVIDER)
    provider = provider_cls(**overrides)
    provider.deepstack_visual_indexes = []
    return provider


def main() -> int:
    provider = unified_provider()
    assert list(provider.deepstack_visual_indexes) == [], "unified provider 仍带 deepstack 注入点"
    gemma4 = _load_provider_class(GEMMA4_VL_PROVIDER)  # 存在性检查（unified 的对拍路径）
    print("[deeprecur·megatron] ✓ Qwen3VLModelProvider 可导入，unified 覆盖已应用：")
    print(f"    provider = {type(provider).__name__}（mbridge，HF config 驱动 deepstack）")
    print(f"    deepstack_visual_indexes = {list(provider.deepstack_visual_indexes)}")
    print(f"    patch_size = {provider.patch_size} / spatial_merge_size = {provider.spatial_merge_size}")
    print(f"    gemma4 对拍路径可用：{gemma4.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
