"""mcore 侧三臂口径：**尽可能用原生件**（mbridge provider + gated_delta_attn_res 的 mcore 算子）。

| 臂 | mcore 路径 | 复用 |
|---|---|---|
| native（Qwen3-VL 原样） | mbridge ``Qwen3VLModelProvider`` | 全套原生（ViT/merger/文本层/并行/检查点） |
| gdar / deeprecur | 同一个 provider，**只把文本层 spec 换成 GDAR** | 视觉塔 / merger / 打包 / 并行 / 检查点全原生；GDAR 的连接算子直接 import ``gated_delta_attn_res`` 的 mcore 件 |
| unified（encoder-free） | 同一 provider + 单 matmul 视觉模块（未接线，显式报错） | 文本塔原样；视觉侧只有 LN/Dense/位置/投影 |

现成可复用的 mcore 自研件（``gated_delta_attn_res/common/models/megatron``）：
``gdar_connection``（门控 delta 连接算子）、``gdar_layer``（层）、``gdar_spec`` /
``ablation_spec``（层规格与消融预置）。它们是为 Qwen3 文本层写的，Qwen3-VL 的文本塔同族，
所以**文本塔的 GDAR 可以直接换 spec**；视觉塔的 GDAR 块与本配方的 DeepRecur 交织顶层是
后续件（见 README 的镜像状态表）。

用法：
    python -m shensi.recipes.paper.deeprecur.common.models.megatron.mcore           # 打印三臂口径
    python -m shensi.recipes.paper.deeprecur.common.models.megatron.mcore --check   # 复用件导入自查
"""

from __future__ import annotations

import argparse

#: 三臂的 mcore 落点（文档 + 自查共用）
ARMS = {
    "native": {
        "provider": "megatron.bridge.models.qwen_vl.qwen3_vl_provider.Qwen3VLModelProvider",
        "reuse": "全套原生（ViT / merger / deepstack / 文本层 / 并行 / 检查点）",
        "spec_override": None,
    },
    "gdar": {
        "provider": "megatron.bridge.models.qwen_vl.qwen3_vl_provider.Qwen3VLModelProvider",
        "reuse": "视觉塔 / merger / 打包 / 并行 / 检查点原生；只换文本层 spec",
        "spec_override": "gated_delta_attn_res.common.models.megatron.gdar_spec:gdar_layer_spec",
    },
    "deeprecur": {
        "provider": "megatron.bridge.models.qwen_vl.qwen3_vl_provider.Qwen3VLModelProvider",
        "reuse": "同上；另需 DeepRecur 块容器（视觉 chunk ↔ 文本 chunk 配对 + reinject/feedback）",
        "spec_override": "gated_delta_attn_res.common.models.megatron.gdar_spec:gdar_layer_spec",
    },
    "unified": {
        "provider": "megatron.bridge.models.qwen_vl.qwen3_vl_provider.Qwen3VLModelProvider",
        "reuse": "文本塔原生；视觉侧换单 matmul 嵌入器（LN→Dense→LN→+2D 位置→LN→RMSNorm→Linear）",
        "spec_override": None,
    },
}


def _import_attr(dotted: str):
    import importlib

    module_path, sep, attr = dotted.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr) if sep else module


def check() -> dict:
    """复用件自查：mbridge provider、GDAR 的 mcore spec/layer/connection 都能导入。"""
    report: dict = {"ok": [], "failed": {}}
    targets = {
        "qwen3_vl_provider": "megatron.bridge.models.qwen_vl.qwen3_vl_provider:Qwen3VLModelProvider",
        "gemma4_vl_provider（encoder-free 参考）": (
            "megatron.bridge.models.gemma_vl.gemma4_vl_provider:Gemma4VLModelProvider"
        ),
        "gdar_spec": "shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_spec:gdar_layer_spec",
        "gdar_layer": "shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer",
        "gdar_connection": "shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_connection",
    }
    for name, spec in targets.items():
        try:
            _import_attr(spec)
            report["ok"].append(name)
        except Exception as exc:
            report["failed"][name] = f"{type(exc).__name__}: {exc}"
    return report


def gdar_text_spec():
    """Qwen3-VL 文本塔的 GDAR 层规格（层类换 GDAR，子模块沿用 TE）。

    用法（与 mbridge provider 组合）::

        from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider
        provider = Qwen3VLModelProvider()
        provider.transformer_layer_spec = gdar_text_spec()   # 视觉塔/merger 等其余全原生
    """
    return _import_attr(ARMS["gdar"]["spec_override"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deeprecur 的 mcore 侧三臂口径")
    parser.add_argument("--check", action="store_true", help="复用件导入自查")
    args = parser.parse_args(argv)

    print("[deeprecur·megatron] 三臂 mcore 口径（能原生就原生）：")
    for arm, row in ARMS.items():
        print(f"  {arm:<9} provider={row['provider'].rsplit('.', 1)[-1]}")
        print(f"            复用：{row['reuse']}")
        if row["spec_override"]:
            print(f"            只换：transformer_layer_spec = {row['spec_override']}")
    if not args.check:
        return 0

    report = check()
    print("[deeprecur·megatron] 复用件自查：")
    for name in report["ok"]:
        print(f"  ✓ {name}")
    for name, err in report["failed"].items():
        print(f"  ✗ {name}: {err}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
