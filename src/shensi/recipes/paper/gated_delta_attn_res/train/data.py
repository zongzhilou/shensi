"""训练公共件的 data.py 模块。"""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage

from shensi.recipes.shensi.train.sft_dataset import ShensiSFTDataset


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    if is_packed_sequence:
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    tokenizer = build_tokenizer(args)
    blend: tuple[list[str], list[float] | None] | None
    blend_per_split: list[tuple[list[str], list[float] | None] | None] | None
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path) as f:
            sequences_per_dataset = json.load(f)
    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        inter_document_masking=args.dataloader_inter_document_masking,
        sft_mock_dataset_config_json=args.sft_mock_dataset_config_json,
        varlen_mock_dataset_config_json=args.varlen_mock_dataset_config_json,
        varlen_sbhd_validation=args.varlen_sbhd_validation,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)
    if args.sft:
        if args.mock_data:
            raise SystemExit(
                "[gdar] SFT 不能用 --mock-data：上游 MockSFTDataset 是 THD 打包口径，而本配方的\n"
                "        local 注意力断言 packed_seq_params is None（两者互斥）。链路自检用\n"
                "        `train.py --smoke`（合成 messages jsonl + tiny 几何），验证用\n"
                "        `--profile debug` + 真实 messages jsonl。"
            )
        dataset_type = ShensiSFTDataset
        is_packed_sequence = False
    else:
        dataset_type = MockGPTDataset if args.mock_data else GPTDataset
        is_packed_sequence = False
    print_rank_0("> building train, validation, and test datasets for GPT ...")
    is_dataset_built = partial(
        is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence
    )
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()
    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds
