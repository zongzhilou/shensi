#!/usr/bin/env python3
"""mcore 训练循环入口（等价上游 pretrain_gpt.py）。"""

from __future__ import annotations

import sys
import time
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.context_parallel import ContextParallelBatch, get_batches_on_this_cp_rank
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_tp_rank,
    get_te_version,
    get_torch_version,
)
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.global_vars import initialize_runtime_services

try:  # 新一点的 MCore 才有：运行配置要显式登记（见下方调用点的注释）
    from megatron.training.global_vars import set_run_config
except ImportError:  # pragma: no cover - 老 MCore 没有这个全局
    set_run_config = None
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import is_first_or_last_pipeline_stage

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.shensi.common.train import args as shensi_args
from shensi.recipes.shensi.common.train import erc, probes
from shensi.recipes.shensi.common.train.builders import (
    ShensiModelBuilder,
    ShensiModelConfig,
    attach_scalar_optimizer_kwargs,
    build_shensi_transformer_config_from_args,
)
from shensi.recipes.shensi.common.train.data import (
    is_dataset_built_on_rank,  # noqa: F401  供上游按名字取
    train_valid_test_datasets_provider,
)

_PROGRAM_START_TIME = time.time()
stimer = StragglerDetector()

BATCH_KEYS = [
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "hybrid_cp_group",
    "labels",
    "local_cp_size",
    "loss_mask",
    "max_seqlen",
    "position_ids",
    "tokens",
]


def get_batch(data_iterator, vp_stage: int | None = None):
    """取一个 micro-batch（与上游 `pretrain_hybrid.py` 同口径，返回 `ContextParallelBatch`）。

    与上游唯一的差别：`has_cu_seqlens` 里 **SFT 默认不算打包**——我们的 CSA/HCA 层断言
    `packed_seq_params is None`，SFT 走「一条对话一条样本 + 右 padding」的稠密档；只有
    `--shensi-sft-packed`（或 mock SFT）才是打包口径（那时 batch 里才有 cu_seqlens）。
    """
    args = get_args()
    config = core_transformer_config_from_args(args)

    if args.sequence_packing_scheduler is not None:
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            packed_seq_params,
            padding_mask,
        ) = get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=mtp_on_this_rank_func(
                layout=config.pipeline_model_parallel_layout,
                mtp_num_layers=config.mtp_num_layers,
                ignore_virtual=False,
                vp_stage=vp_stage,
            ),
            vp_stage=vp_stage,
        )
        return ContextParallelBatch.from_single_layout(
            config.linear_cp_layout,
            {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "padding_mask": padding_mask,
            },
            packed_seq_params,
        )

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_sft = args.sft
    sft_packed = is_sft and (getattr(args, "shensi_sft_packed", False) or args.mock_data)
    has_cu_seqlens = sft_packed or args.dataloader_inter_document_masking
    create_attention_mask_in_dataloader = args.create_attention_mask_in_dataloader
    mtp_on_this_rank = mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )
    is_hybrid_cp = args.hybrid_context_parallel

    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not mtp_on_this_rank
        and not has_cu_seqlens
    ):
        return ContextParallelBatch(
            boundary_layout=config.linear_cp_layout,
            batches_by_layout={config.linear_cp_layout: dict.fromkeys(BATCH_KEYS)},
            packed_seq_params_by_layout={config.linear_cp_layout: None},
        )

    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            batch[key] = (
                batch[key].cuda(non_blocking=True)
                if key in batch and batch[key] is not None
                else None
            )
    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        has_cu_seqlens=has_cu_seqlens,
        is_hybrid_cp=is_hybrid_cp,
        create_attention_mask_in_dataloader=create_attention_mask_in_dataloader,
        cp_size=cp_size,
        tp_rank=tp_rank,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=mtp_on_this_rank,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )
    batch = flatten_batch_for_packed_sequences(batch)

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank:
        assert has_cu_seqlens
        batch = {
            **dict.fromkeys(BATCH_KEYS),
            "cu_seqlens": batch["cu_seqlens"],
            "cu_seqlens_padded": batch["cu_seqlens_padded"],
            "max_seqlen": batch["max_seqlen"],
        }

    additional_layouts = set()
    if cp_size > 1 and config.linear_cp_layout != config.attention_cp_layout:
        additional_layouts.add(config.attention_cp_layout)
    return get_batches_on_this_cp_rank(
        batch,
        boundary_layout=config.linear_cp_layout,
        is_hybrid_cp=is_hybrid_cp,
        cp_group=get_context_parallel_group(),
        additional_layouts=additional_layouts,
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_per_sequence_balancing=args.dataloader_inter_document_masking and not is_sft,
        sequence_parallel=config.sequence_parallel,
        tp_group=mpu.get_tensor_model_parallel_group(),
        tp_cp_group=(
            mpu.get_tensor_and_context_parallel_group()
            if config.sequence_parallel and config.tensor_model_parallel_size > 1
            else None
        ),
        tokens_per_sample=args.seq_length,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel | None = None):
    """上游的 lm loss + Shensi 的 ERC 项（口径见 erc.py 的模块注释）。"""
    args = get_args()
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large, threshold=10, context="loss"
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )
    return erc.attach_erc_to_loss(
        loss, num_tokens, report, output_tensor=output_tensor, loss_mask=loss_mask, model=model
    )


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """取 batch → 前向 → 交给 loss_func（与上游同口径）。"""
    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        cp_batch = get_batch(data_iterator, vp_stage)
        batch = cp_batch.get_batch()
        attention_mask = batch.get("attention_mask")
        cu_seqlens = batch.get("cu_seqlens")
        labels = batch.get("labels")
        loss_mask = batch.get("loss_mask")
        position_ids = batch.get("position_ids")
        tokens = batch.get("tokens")
        packed_seq_params = cp_batch.get_packed_seq_params()
        padding_mask = batch.get("padding_mask")
        if cu_seqlens is not None:
            update_seqlen_stats_from_cu_seqlens(cu_seqlens.squeeze(0))
    timers("batch-generator").stop()

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm 开着才能返回 schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            cp_batch=cp_batch,
        )
    return output_tensor, partial(loss_func, loss_mask, model=model)


def get_embedding_ranks(pp_ranks):
    """Embedding 所在 stage：首 stage（tie 时还有末 stage）+ MTP 所在 stage。"""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        embedding_ranks.extend(get_mtp_ranks(pp_ranks, config))
    return sorted(set(embedding_ranks))


def main() -> None:
    main_entry_time = time.time()
    print_rank_0(f"> PyTorch version ................ {get_torch_version()}")
    print_rank_0(f"> Megatron-Core version .......... {mcore_version}")
    print_rank_0(f"> Transformer Engine version ... {get_te_version()}")
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)

    train_valid_test_datasets_provider.is_distributed = True
    # 上游这个辅助函数会自己再 parse 一遍 argv（不带 extra_args_provider），只有 argv[0] 不会撞
    # 参数冲突；真要用 --inprocess-restart 时下面显式报错，不静默失效。
    _saved_argv = sys.argv
    sys.argv = _saved_argv[:1]
    try:
        pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    finally:
        sys.argv = _saved_argv

    parsed = parse_and_validate_args(extra_args_provider=shensi_args.add_shensi_args)
    if getattr(parsed, "inprocess_restart", False):
        raise SystemExit(
            "本入口不支持 --inprocess-restart：上游的 inprocess_restart 辅助函数会自己重新解析 argv，"
            "拿不到 shensi 的参数集（choices 里的 ademamix 之类会被拒）。需要这个特性请用上游 "
            "pretrain_gpt.py 那套入口，或把需求提给我们。"
        )
    shensi_args.postprocess_args(parsed)
    probes.install_all(parsed)

    # Shensi 走 HybridModel 容器（层计划 + 混合栈规格由 Bridge 的 shensi_hybrid 推导）→
    # 配置侧用 mcore 的 `hybrid_config_from_args`（与 `gpt_config_from_args` 逐项对应）
    model_cfg = hybrid_config_from_args(
        parsed,
        config=build_shensi_transformer_config_from_args(parsed),
        model_config_cls=ShensiModelConfig,
        vocab_size_from_tokenizer=True,
    )
    # 上游按 `ModelConfig.builder` 这个字符串找构造器：改错了这里直接拦下
    assert model_cfg.get_builder_cls() is ShensiModelBuilder, model_cfg.get_builder_cls()
    full_config = pretrain_cfg_container_from_args(parsed, model_cfg)
    attach_scalar_optimizer_kwargs(full_config, parsed)
    # 较新的 MCore 要求先把运行配置登记给全局（上游入口的顺序：container → set_run_config →
    # initialize_runtime_services），否则 tensorboard/checkpoint 那几步会 assert 未初始化；
    # 老版本没有这个函数 → 导入守卫，拿到就不调用。
    if set_run_config is not None:
        set_run_config(full_config)
    initialize_runtime_services(parsed)
    resolve_tokenizer_vocab_size(full_config, parsed.padded_vocab_size)
    pretrain_fn(
        full_config,
        train_valid_test_datasets_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
