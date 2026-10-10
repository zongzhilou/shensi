#!/usr/bin/env python3
"""构造器契约测试：`ShensiModelBuilder` 必须走 Bridge 的 hybrid 路径（不再碰旧式 layer spec）。

本机没有真 Transformer Engine、建不了整模型（TE 内核路径要真 TE；`transformer_impl=local` 又被
DSv4-hybrid 明确拒绝），所以这里钉的是**契约**而不是整模型：

  ① `ShensiModelConfig` 是 `HybridModelConfig` 的子类，`builder` 字符串解析回本类；
  ② `build_model` 把层计划 / 混合栈规格整个委托给 Bridge 的 `build_shensi_model`：用记录器替换它，
     逐项核对 kwargs，并确认**没有**再传旧式的 `transformer_layer_spec` / `mtp_block_spec`；
  ③ 默认几何（19 层 + 1 MTP）能纯配置地推出层计划 `*×19 + /*`，栈规格能构出来（不建模块）。

整模型 5 步冒烟（含真实 TE）用 `test_train.py` 在目标机上跑——见 README 的「验证」。
"""

from __future__ import annotations

import types

import megatron.training.global_vars as gv
from megatron.bridge.models.shensi.shensi_hybrid import (
    get_shensi_hybrid_stack_spec,
    shensi_hybrid_layer_pattern,
)
from megatron.bridge.models.shensi.shensi_provider import ShensiTransformerConfig
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig

from shensi.recipes.shensi.common.train import builders as B


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def geometry_kwargs() -> dict:
    """默认几何（与 `common/config/hf/2p8b_a1p2b.json` 同值）。"""
    return dict(
        num_layers=19,
        mtp_num_layers=1,
        hidden_size=1280,
        num_attention_heads=16,
        q_lora_rank=320,
        o_groups=2,
        o_lora_rank=320,
        num_moe_experts=256,
        moe_router_topk=6,
        moe_ffn_hidden_size=800,
        moe_latent_size=320,
        moe_n_hash_layers=3,
        attn_res_block_size=2,
        csa_compress_ratios=[0, 0] + [4, 128] * 8 + [4, 0],
        csa_window_size=128,
        num_residual_streams=16,
        hc_active_streams=4,
        hc_fixed_streams=2,
        add_bias_linear=False,
        # 本机没有 flash_mla / cudnn 融合内核：走 PyTorch 回退（生产档在真机上由 runtime 自动选）
        dsa_kernel_backend="none",
    )


class _FakePG:
    class pp:  # noqa: N801
        @staticmethod
        def rank() -> int:
            return 0


def main() -> int:
    # ---- ① 类契约 ----
    check(
        "ShensiModelConfig 继承 HybridModelConfig",
        issubclass(B.ShensiModelConfig, HybridModelConfig),
    )
    check(
        "ShensiModelBuilder 继承 HybridModelBuilder",
        issubclass(B.ShensiModelBuilder, HybridModelBuilder),
    )

    transformer = ShensiTransformerConfig(**geometry_kwargs())
    finalize = getattr(transformer, "finalize", None)
    if callable(finalize):
        finalize()
    model_cfg = B.ShensiModelConfig(transformer=transformer, seq_length=4096, vocab_size=129280)
    check(
        "builder 字符串解析回本类",
        model_cfg.get_builder_cls() is B.ShensiModelBuilder,
        str(model_cfg.get_builder_cls()),
    )

    # ---- ② 委托契约：build_model 只把 kwargs 交给 Bridge 的 build_shensi_model ----
    recorded: dict = {}

    def fake_build_shensi_model(config, pg_collection, **kwargs):
        recorded["config"] = config
        recorded["pg"] = pg_collection
        recorded.update(kwargs)
        return types.SimpleNamespace(name="fake-shensi-model")

    original_args = gv._GLOBAL_ARGS  # noqa: SLF001  get_args() 读的就是它
    gv._GLOBAL_ARGS = types.SimpleNamespace(shensi_freeze="none")  # noqa: SLF001
    real_build = B.build_shensi_model
    B.build_shensi_model = fake_build_shensi_model
    try:
        model = B.ShensiModelBuilder(model_cfg).build_model(pg_collection=_FakePG())
    finally:
        B.build_shensi_model = real_build
        gv._GLOBAL_ARGS = original_args  # noqa: SLF001
    check("走的是 Bridge 的 build_shensi_model（返回了模型）", model.name == "fake-shensi-model")
    check("传的是 transformer 配置本身", recorded["config"] is transformer)
    check(
        "没有旧式 transformer_layer_spec / mtp_block_spec",
        "transformer_layer_spec" not in recorded and "mtp_block_spec" not in recorded,
    )
    for key in (
        "vocab_size",
        "max_sequence_length",
        "pre_process",
        "post_process",
        "fp16_lm_cross_entropy",
        "logit_dtype",
        "parallel_output",
        "share_embeddings_and_output_weights",
        "position_embedding_type",
        "rotary_percent",
        "rotary_base",
        "seq_len_interpolation_factor",
    ):
        check(f"kwargs 含 {key}", key in recorded, repr(recorded.get(key)))
    check(
        "actual_vocab_size 已注入（hash-MoE deepemb 用）",
        getattr(transformer, "actual_vocab_size", None) == 129280,
    )

    # ---- ③ 层计划与栈规格（纯配置，不建模块） ----
    pattern = shensi_hybrid_layer_pattern(transformer)
    check("层计划 = 19 个 * + MTP 段 /*", pattern == "*" * 19 + "/*", repr(pattern))
    spec = get_shensi_hybrid_stack_spec(transformer, vp_stage=None, pp_rank=0)
    check("混合栈规格可推出", spec is not None, type(spec).__name__)

    # ---- ④ 新 batch 口径（`get_batch` 返回 ContextParallelBatch）需要的配置字段 ----
    for field in ("linear_cp_layout", "attention_cp_layout"):
        check(
            f"finalize 后有 {field}",
            getattr(transformer, field, None) is not None,
            repr(getattr(transformer, field, None)),
        )

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
