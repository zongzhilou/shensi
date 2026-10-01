#!/usr/bin/env python3
"""Looma 配方的训练入口：Megatron 训练循环 + 不动点 block 的 Llama 几何模型。

``--spec <module> <object>`` 指到 ``common/models/megatron/looma_spec.py`` 的层规格预设，
层是 ``LoomaTransformerLayer``（连接与求解器见 ``looma_layer.py``）；数据管线与 KD 方向
开关在同目录的 ``data.py`` / ``reverse_kl.py``。

torchrun 命令行由 ``common/runner.py`` 生成（不派生 ``--qk-layernorm``，Llama 没有 qk
norm），也可以照它写出的 ``<exp_dir>/run.sh`` 手工起。
"""

from __future__ import annotations

import time
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_te_version,
    get_torch_version,
)
from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import (
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.global_vars import initialize_runtime_services
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import is_first_or_last_pipeline_stage

from shensi import runtime  # noqa: F401
from shensi.recipes.paper.looma.common.train.data import (
    is_dataset_built_on_rank,  # noqa: F401
    train_valid_test_datasets_provider,
)
from shensi.recipes.shensi.common.train import args as shensi_args

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
    """按 ``BATCH_KEYS`` 的顺序取一个 micro-batch。

    TP rank 0 先取数据并搬上 GPU，再沿 TP 广播、按 CP 切分；本 stage 不持有 batch
    （流水线中段且无 MTP / cu_seqlens）时返回全 ``None`` 的占位。
    """
    args = get_args()
    config = core_transformer_config_from_args(args)

    if args.sequence_packing_scheduler is not None:
        return get_batch_on_this_rank_for_sequence_packing(
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

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    has_cu_seqlens = args.dataloader_inter_document_masking  # SFT 不打包，没有 cu_seqlens
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
        return [None for _ in BATCH_KEYS]

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
        return (
            None,
            batch["cu_seqlens"],
            batch["cu_seqlens_padded"],
            None,
            None,
            None,
            None,
            batch["max_seqlen"],
            None,
            None,
        )

    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=is_hybrid_cp,
        cp_group=get_context_parallel_group(),
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_per_sequence_balancing=args.dataloader_inter_document_masking,
    )
    return [batch[key] for key in BATCH_KEYS]


def add_looma_args(parser) -> None:
    """``extra_args_provider``：挂上 shensi 家族的参数，再加本配方自己的开关。

    先调 ``shensi_args.add_shensi_args`` 是必须的：AdaMuon / AdEMAMix 这些 optimizer
    的名字由它扩进 choices，否则在 mcore 的 parser 里是 invalid choice。
    """
    shensi_args.add_shensi_args(parser)
    group = parser.add_argument_group(title="Looma recipe")
    group.add_argument(
        "--logits-load-reverse-kl",
        action="store_true",
        help="KD 用 reverse KL（KL(student‖teacher)）而不是 mcore 的 forward KL",
    )
    return parser


_CACHED_KD_LOSS = None


def _kd_loss_func(loss_mask, output_tensor, model):
    """KD 分支的 loss：``--logits-load-dir`` 有 teacher 缓存 logprob 时走这条，懒构造并缓存。"""
    global _CACHED_KD_LOSS
    if _CACHED_KD_LOSS is None:
        if getattr(get_args(), "logits_load_reverse_kl", False):
            from shensi.recipes.paper.looma.common.train.reverse_kl import install_reverse_kl

            install_reverse_kl()
            print_rank_0("[looma] KD 方向：reverse KL（KL(student‖teacher)，OPD 口径）")
        from megatron.training.distillation import LossFuncCallable

        _CACHED_KD_LOSS = LossFuncCallable(
            logprobs_dir=get_args().logits_load_dir,
            decode_threads=get_args().logits_load_decode_threads,
            prefetch_factor=get_args().logits_load_prefetch_factor,
            msc_prefetch_depth=get_args().logits_load_msc_prefetch_depth,
            kd_loss_alpha=get_args().logits_load_kd_loss_alpha,
            ignore_errors=get_args().logits_load_ignore_errors,
        )
    return _CACHED_KD_LOSS(loss_mask, output_tensor, model=model)


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel | None = None):
    """纯 LM loss（没有 ERC / indexer 项），配了 teacher 缓存时转给 KD 分支。

    顺带按 ``--check-for-nan-in-loss-and-grad`` / ``--check-for-spiky-loss`` 做重跑校验。
    """
    args = get_args()
    if getattr(args, "logits_load_dir", None) is not None:
        return _kd_loss_func(loss_mask, output_tensor, model)
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
    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """取 batch → 前向 → 交给 ``loss_func``；``return_schedule_plan`` 时返回调度计划。"""
    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        batch = get_batch(data_iterator, vp_stage)
        (
            attention_mask,
            cu_seqlens,
            cu_seqlens_padded,
            hybrid_cp_group,
            labels,
            local_cp_size,
            loss_mask,
            max_seqlen,
            position_ids,
            tokens,
        ) = batch
        padding_mask = None
        packed_seq_params = None
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.squeeze(0)
            if cu_seqlens_padded is not None:
                cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
            update_seqlen_stats_from_cu_seqlens(cu_seqlens)
            cu_seqlens_for_params = (
                cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
            )
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_for_params,
                cu_seqlens_kv=cu_seqlens_for_params,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                max_seqlen_q=int(max_seqlen.item()),
                max_seqlen_kv=int(max_seqlen.item()),
                local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
                cp_group=hybrid_cp_group,
                tokens_per_sample=args.seq_length,
            )
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
        )
    return output_tensor, partial(loss_func, loss_mask, model=model)


def get_embedding_ranks(pp_ranks):
    """Embedding 所在 stage：首 stage（权重不 tie 时再加上末 stage）；本配方没有 MTP。"""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
    return sorted(set(embedding_ranks))


def main() -> None:
    """打印环境信息、解析参数、校验规格与 MTP 的互斥后启动 ``pretrain``。"""
    main_entry_time = time.time()
    print_rank_0(f"> PyTorch version ................ {get_torch_version()}")
    print_rank_0(f"> Megatron-Core version .......... {mcore_version}")
    print_rank_0(f"> Transformer Engine version ... {get_te_version()}")
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)

    train_valid_test_datasets_provider.is_distributed = True
    parsed = parse_and_validate_args(extra_args_provider=add_looma_args)
    if parsed.spec is not None:
        print_rank_0(f"> spec .......................... {' '.join(parsed.spec)}")
    else:
        print_rank_0("> spec .......................... (none) → plain Llama")
    if parsed.mtp_num_layers and parsed.spec:
        raise SystemExit(
            "[looma] MTP 与本配方的层规格不能同时用：mcore 组合 MTP 块时要求 "
            "spec.module is TransformerLayer（megatron/core/models/gpt/gpt_layer_specs.py），"
            "自定义层（本配方的 LoomaTransformerLayer）不在其列。要 MTP 就把 mtp_num_layers 设为 0；"
            "若要给 MTP 头单独用一个 stock 规格，那部分不走本配方的块循环。"
        )
    model_cfg = gpt_config_from_args(parsed, vocab_size_from_tokenizer=True)
    full_config = pretrain_cfg_container_from_args(parsed, model_cfg)
    initialize_runtime_services(parsed)
    resolve_tokenizer_vocab_size(full_config, parsed.padded_vocab_size)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
