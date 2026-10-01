"""RealFormer 的 mcore 层：残差注意力（分数跨层累加）。"""

from __future__ import annotations

from megatron.core.transformer.transformer_layer import TransformerLayer

from .realformer_attention import build_realformer_submodules, realformer_knobs_from_kwargs

__all__ = ["RealFormerTransformerLayer", "build_realformer_submodules", "realformer_knobs_from_kwargs"]


class RealFormerTransformerLayer(TransformerLayer):

    def __init__(
        self,
        config,
        submodules=None,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection=None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        carry = kwargs.pop("realformer_carry", None)
        knobs = realformer_knobs_from_kwargs(kwargs, config)
        if config.pipeline_model_parallel_size > 1:
            raise NotImplementedError(
                "RealFormer 的跨层状态是注意力分数矩阵 [b, heads, s, s]，过不了 pipeline 边界"
                "（p2p 的张量形状由 config 决定，装不下这层状态）。请用 "
                "pipeline_model_parallel_size=1。"
            )
        if getattr(config, "recompute_granularity", None) == "full":
            raise NotImplementedError(
                "RealFormer 不支持 recompute_granularity='full'：残差注意力要求层按序"
                "「层 1 重置 → 每层顺序消费」，整层重算会打乱这个时序。"
            )
        if getattr(config, "fp32_residual_connection", False):
            raise NotImplementedError("RealFormer 不支持 fp32_residual_connection。")
        if submodules is None:
            submodules = build_realformer_submodules(
                config,
                gate_mode=knobs["gate"],
                use_running_mean=knobs["mean"],
                carry=carry,
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )
