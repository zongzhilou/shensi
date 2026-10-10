"""模型与配置构建、参数冻结、标量优化器超参挂载。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_hybrid import build_shensi_model
from megatron.bridge.models.shensi.shensi_provider import (
    ShensiTransformerConfig,
    apply_shensi_overrides_from_args,
    shensi_config_from_args,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import get_args, print_rank_0
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from megatron.training.vocab_utils import calculate_padded_vocab_size


@dataclass(kw_only=True)
class ShensiModelConfig(HybridModelConfig):
    """`HybridModelConfig` + Shensi 的构造器（其余字段与上游一致）。"""

    builder: ClassVar[str] = "shensi.recipes.shensi.common.train.builders.ShensiModelBuilder"


class ShensiModelBuilder(HybridModelBuilder):
    """按 Bridge 的混合栈规格建 `ShensiModel`（容器 = mcore 的 `HybridModel`）。

    层计划（含 MTP 的 `/` 段）与栈规格都由 Bridge 的 `shensi_hybrid.build_shensi_model` 推导——
    同一个函数也供 `ShensiModelProvider` 用，两条路径建出同一个模型；本类只负责词表补齐、
    `actual_vocab_size` 注入与 `--shensi-freeze` 参数冻结。
    """

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

        # 参数清单与 Bridge provider 的 `get_model` 逐项一致（provider 会传
        # scatter_embedding_sequence_parallel，那是 provider 自己的字段，HybridModel 默认 True 即同值）。
        model = build_shensi_model(
            transformer,
            pg_collection,
            vocab_size=int(padded_vocab_size),
            max_sequence_length=cfg.seq_length,
            vp_stage=vp_stage,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=cfg.fp16_lm_cross_entropy,
            logit_dtype=cfg.logit_dtype,
            parallel_output=cfg.parallel_output,
            share_embeddings_and_output_weights=cfg.share_embeddings_and_output_weights,
            position_embedding_type=cfg.position_embedding_type,
            rotary_percent=cfg.rotary_percent,
            rotary_base=cfg.rotary_base,
            seq_len_interpolation_factor=cfg.seq_len_interpolation_factor,
        )
        apply_shensi_freeze(model, getattr(args, "shensi_freeze", "none"))
        return model


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
    """Args → `ShensiTransformerConfig`（含 `--shensi-*` 覆写）。

    MCore 的 `core_transformer_config_from_args` 会把 args 上同名的**每个** dataclass 字段都当 kwarg 传
    （不区分 `init=False`）；较新的上游里这就会撞上 Bridge 侧的 `gtp_weight_remat_size` 这类
    「不让进构造器」的字段（`field(init=False)`）→ `TypeError: got an unexpected keyword argument`。
    这里在调用前把这类名字从 args 上临时摘掉、调用完原样恢复：它们的值都是常量默认（1），摘掉不丢信息，
    在老的 MCore 上这些名字根本不在 args 上（`hasattr` 为假）→ 本层是空操作，两边都安全。
    """
    import dataclasses

    blocked = {
        f.name
        for f in dataclasses.fields(ShensiTransformerConfig)
        if not f.init and hasattr(args, f.name)
    }
    saved = {name: getattr(args, name) for name in blocked}
    for name in blocked:
        delattr(args, name)
    try:
        config = shensi_config_from_args(args)
    finally:
        for name, value in saved.items():
            setattr(args, name, value)
    apply_shensi_overrides_from_args(config, args)
    # Bridge 的 TransformerConfig 把 MCore 的 `__post_init__` **延迟**到 `finalize()`（构造时不跑），
    # 派生字段（init_method / output_layer_init_method / MoE 检查 / weight shards …）都在那里才算；
    # provider 路径由 `provider.finalize()` 触发，训练这条 args 路径要自己补，否则建模型时
    # `config.init_method` 还是 None → `'NoneType' object is not callable`。放在覆写之后：finalize
    # 要按最终字段值算派生量。老一点的 Bridge 上 finalize 可能不存在 → 守卫。
    finalize = getattr(config, "finalize", None)
    if callable(finalize):
        finalize()
    return config


# 标量优化器的超参不是 OptimizerConfig 字段，在这里补挂（shensi.utils.optimizer 的标量腿按名字取用）
SCALAR_OPTIMIZER_KWARG_FIELDS = (
    "ademamix_betas",
    "ademamix_alpha",
    "ademamix_beta3",
    "ademamix_t_alpha_beta3",
    "grokfast_alpha",
    "grokfast_lamb",
    "grokfast_after_step",
)

# 这些名字是 shensi.utils.optimizer 注册的标量腿
_SCALAR_OPTIMIZER_NAMES = ("ademamix", "grokfastadamw")


def attach_scalar_optimizer_kwargs(container, args) -> list[str]:
    """把标量优化器的超参挂到运行配置的 OptimizerConfig 上；返回实际挂上去的字段。"""
    opt_cfg = getattr(container, "optimizer", None)
    if opt_cfg is None:
        return []
    attached = []
    for name in SCALAR_OPTIMIZER_KWARG_FIELDS:
        value = getattr(args, name, None)
        if value is None:
            continue
        setattr(opt_cfg, name, tuple(value) if isinstance(value, list) else value)
        attached.append(name)
    scalar_name = str(getattr(opt_cfg, "muon_scalar_optimizer", "") or "").lower()
    if attached and scalar_name in _SCALAR_OPTIMIZER_NAMES:
        print_rank_0(
            f"[shensi] {scalar_name} 超参已挂到 OptimizerConfig："
            f"{ {n: getattr(opt_cfg, n) for n in attached} }"
        )
    return attached


__all__ = [
    "SCALAR_OPTIMIZER_KWARG_FIELDS",
    "ShensiModelBuilder",
    "ShensiModelConfig",
    "apply_shensi_freeze",
    "attach_scalar_optimizer_kwargs",
    "build_shensi_transformer_config_from_args",
]
