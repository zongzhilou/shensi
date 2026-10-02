"""训练入口：上游 mcore 训练循环 + GPTModel + --spec 深度连接。"""

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

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res.common.train import optimizer_knobs
from shensi.recipes.paper.gated_delta_attn_res.common.train.data import (
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
    has_cu_seqlens = args.dataloader_inter_document_masking
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


def add_gdar_args(parser) -> None:
    group = parser.add_argument_group(title="GDAR recipe")
    optimizer_knobs.add_scalar_optimizer_args(group)
    group.add_argument(
        "--logits-load-reverse-kl",
        action="store_true",
        help="KD 用 reverse KL（KL(student‖teacher)）而不是 mcore 的 forward KL——"
        "MiniCPM5 的 OPD 口径（见 train/reverse_kl.py 的推导）。",
    )
    group.add_argument(
        "--gdar-whiten-impl",
        default="eager",
        choices=("eager", "ns", "triton"),
        help="白化实现：eager（参考）/ ns（免 LAPACK 的 Newton–Schulz）/ triton（再叠加 "
        "Triton 协方差）。见 kernels/README.md 的实测表。",
    )
    return parser


_CACHED_KD_LOSS = None


def _kd_loss_func(loss_mask, output_tensor, model):
    global _CACHED_KD_LOSS
    if _CACHED_KD_LOSS is None:
        if getattr(get_args(), "logits_load_reverse_kl", False):
            from shensi.recipes.paper.gated_delta_attn_res.common.train.reverse_kl import (
                install_reverse_kl,
            )

            install_reverse_kl()
            print_rank_0("[gdar] KD 方向：reverse KL（KL(student‖teacher)，OPD 口径）")
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
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
    return sorted(set(embedding_ranks))


def main() -> None:
    main_entry_time = time.time()
    print_rank_0(f"> PyTorch version ................ {get_torch_version()}")
    print_rank_0(f"> Megatron-Core version .......... {mcore_version}")
    print_rank_0(f"> Transformer Engine version ... {get_te_version()}")
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)

    train_valid_test_datasets_provider.is_distributed = True
    parsed = parse_and_validate_args(extra_args_provider=add_gdar_args)
    if parsed.spec is not None:
        print_rank_0(f"> spec .......................... {' '.join(parsed.spec)}")
    else:
        print_rank_0("> spec .......................... (none) → plain Qwen3（base 对照臂）")
    model_cfg = gpt_config_from_args(parsed, vocab_size_from_tokenizer=True)
    full_config = pretrain_cfg_container_from_args(parsed, model_cfg)
    optimizer_knobs.attach_to_container(full_config, parsed)
    _whiten_impl = getattr(parsed, "gdar_whiten_impl", "eager")
    if _whiten_impl != "eager":
        from shensi.recipes.paper.gated_delta_attn_res.common import kernels

        print_rank_0(f"> 白化实现 ....................... {kernels.enable(_whiten_impl)}")
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
