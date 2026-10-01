"""模型构造：上游 `ModelConfig`/`ModelBuilder` 接口 + Bridge 的 Shensi 构件。

上游 `pretrain()` 从 `PretrainConfigContainer.model` 拿 `ModelConfig`，再按它的 `builder`
ClassVar 找到 builder 类去建模型，所以这里只需要把 Shensi 的层规格与 ShensiModel 填进去。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.shensi.layer_specs import (
    get_shensi_decoder_block_spec,
    get_shensi_mtp_block_spec,
)
from megatron.bridge.models.shensi.model import ShensiModel
from megatron.bridge.models.shensi.transformer_config import (
    ShensiTransformerConfig,
    apply_shensi_overrides_from_args,
    shensi_config_from_args,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import get_args, print_rank_0
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from megatron.training.vocab_utils import calculate_padded_vocab_size


@dataclass(kw_only=True)
class ShensiModelConfig(GPTModelConfig):
    """`GPTModelConfig` + Shensi 的构造器（其余字段与上游一致）。"""

    builder: ClassVar[str] = "shensi.recipes.shensi.train.builders.ShensiModelBuilder"


class ShensiModelBuilder(GPTModelBuilder):
    """按 Shensi 的层规格建 `ShensiModel`。"""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> ShensiModel:
        cfg = self._model_config
        transformer = cfg.transformer
        if not isinstance(transformer, ShensiTransformerConfig):
            raise TypeError(
                f"ShensiModelBuilder 需要 ShensiTransformerConfig，得到 {type(transformer).__name__}"
            )
        args = get_args()
        use_te = transformer.transformer_impl == "transformer_engine"
        decoder_spec = get_shensi_decoder_block_spec(
            config=transformer,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
            use_moe=True,
        )
        mtp_spec = None
        if getattr(transformer, "mtp_num_layers", None):
            assert_mtp_on_last_stage(transformer)
            # 走 Bridge 的 helper：上游只认 block spec 或 module 恰为 TransformerLayer 的 spec
            mtp_spec = get_shensi_mtp_block_spec(
                transformer,
                use_transformer_engine=use_te,
                vp_stage=vp_stage,
                pp_rank=pg_collection.pp.rank(),
            )
        assert cfg.vocab_size is not None, "vocab_size 要在建模型前定下来"
        if cfg.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                cfg.vocab_size,
                cfg.make_vocab_size_divisible_by,
                transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = cfg.vocab_size
        # hash-MoE 的 deepemb 建在词表上（`ShensiHashMLP` 读 `config.actual_vocab_size`）；
        # 从 args 建的 TransformerConfig 上没有这个词表字段，这里补成模型实际用的 padding 后大小。
        transformer.actual_vocab_size = int(padded_vocab_size)

        model = ShensiModel(
            config=transformer,
            transformer_layer_spec=decoder_spec,
            mtp_block_spec=mtp_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=cfg.seq_length,
            fp16_lm_cross_entropy=cfg.fp16_lm_cross_entropy,
            logit_dtype=cfg.logit_dtype,
            parallel_output=cfg.parallel_output,
            share_embeddings_and_output_weights=cfg.share_embeddings_and_output_weights,
            position_embedding_type=cfg.position_embedding_type,
            rotary_percent=cfg.rotary_percent,
            rotary_base=cfg.rotary_base,
            rope_scaling=cfg.rope_scaling,
            rope_scaling_factor=cfg.rope_scaling_factor,
            seq_len_interpolation_factor=cfg.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=cfg.scatter_embedding_sequence_parallel,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        apply_shensi_freeze(model, getattr(args, "shensi_freeze", "none"))
        return model


def assert_mtp_on_last_stage(config) -> None:
    """MTP 必须与末 PP stage 同段（布局串里的 `m` 只能出现在最后一段）。"""
    layout = str(getattr(config, "pipeline_model_parallel_layout", "") or "")
    if not layout or "|" not in layout:
        return
    segs = layout.split("|")
    bad = [i for i, seg in enumerate(segs[:-1]) if "m" in seg.lower()]
    if bad:
        raise NotImplementedError(
            f"layout={layout!r} 把 MTP 放在非末 pp stage（段 {bad}）；本模型的 MTP 必须与末 stage 同段"
        )


def apply_shensi_freeze(model, mode: str) -> None:
    """按 `--shensi-freeze` 冻参数组（indexer / MTP 的取舍见 CLI help）。"""
    if mode in (None, "", "none"):
        return
    allowed = ("indexer", "non-indexer", "mtp", "non-mtp")
    if mode not in allowed:
        raise ValueError(f"--shensi-freeze 只认 {allowed}，得到 {mode!r}")
    n_frozen = n_train = 0
    for name, param in model.named_parameters():
        is_indexer = ".indexer." in name or name.endswith(".indexer")
        # MTP 头在 mcore 里叫 mtp / multi_token_prediction；冻主干、只训 draft 是 DeepSpec 口径
        is_mtp = ".mtp." in name or name.startswith("mtp") or "multi_token_prediction" in name
        if mode == "non-indexer":
            freeze = is_indexer
        elif mode == "non-mtp":
            freeze = is_mtp
        elif mode == "mtp":
            freeze = not is_mtp
        else:
            freeze = not is_indexer
        param.requires_grad_(not freeze)
        if freeze:
            n_frozen += 1
        else:
            n_train += 1
    if n_train == 0:
        raise ValueError(
            f"--shensi-freeze {mode} 把全部参数都冻上了：本模型里没有对应参数"
            "（indexer 只有 ratio==4 的 CSA 层才有；MTP 要 mtp_num_layers>0，检查层计划）"
        )
    print_rank_0(
        f"[shensi][freeze] mode={mode}：可训练 {n_train} 个参数、冻结 {n_frozen} 个"
        f"（可训练元素数 {sum(p.numel() for p in model.parameters() if p.requires_grad)}）"
    )


def build_shensi_transformer_config_from_args(args) -> ShensiTransformerConfig:
    """Args → `ShensiTransformerConfig`（含 `--shensi-*` 覆写）。"""
    config = shensi_config_from_args(args)
    apply_shensi_overrides_from_args(config, args)
    return config


# mcore 按 `{优化器名}_{构造参数}` 从 OptimizerConfig 上取标量优化器的超参
# （`_kwargs_from_config(..., prefix=eopt_name, config)`），而 AdEMAMix 这几项不是它的字段，
# 所以在这里补挂到 OptimizerConfig 上（不改上游 mcore）。
ADEMAMIX_KWARG_FIELDS = (
    "ademamix_betas",
    "ademamix_alpha",
    "ademamix_beta3",
    "ademamix_t_alpha_beta3",
)


def attach_scalar_optimizer_kwargs(container, args) -> list[str]:
    """把 `--ademamix-*` 挂到运行配置的 OptimizerConfig 上；返回实际挂上去的字段。"""
    opt_cfg = getattr(container, "optimizer", None)
    if opt_cfg is None:
        return []
    attached = []
    for name in ADEMAMIX_KWARG_FIELDS:
        value = getattr(args, name, None)
        if value is None:
            continue
        setattr(opt_cfg, name, tuple(value) if isinstance(value, list) else value)
        attached.append(name)
    if attached and str(getattr(opt_cfg, "muon_scalar_optimizer", "")) == "ademamix":
        print_rank_0(
            "[shensi] AdEMAMix 超参已挂到 OptimizerConfig："
            f"{ {n: getattr(opt_cfg, n) for n in attached} }"
        )
    return attached


__all__ = [
    "ADEMAMIX_KWARG_FIELDS",
    "ShensiModelBuilder",
    "ShensiModelConfig",
    "apply_shensi_freeze",
    "assert_mtp_on_last_stage",
    "attach_scalar_optimizer_kwargs",
    "build_shensi_transformer_config_from_args",
]
