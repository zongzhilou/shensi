"""极小几何的单一出处（launcher 覆写 + Bridge provider）。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# 键 = HF 侧 ShensiConfig 的字段，值 = 极小档取值
TINY: dict[str, object] = {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    # 16：FlashInfer 在 SM120 上的 DSV4 稀疏 prefill 只实例化了 num_heads ∈ {8,16,32,64,128}
    # （生产 32）；给 4 会分派不到 kernel，跑起来直接 CUDA illegal memory access
    "num_attention_heads": 16,
    # 512 = 生产几何的取值，也是 vLLM 在 SM120 上跑 DSv4 的硬要求：压缩器的 fused quant+cache
    # 只支持 head_dim ∈ {128, 512}，而输出侧的 fp8 量化要求 head_dim // 128 >= 4（=512）。
    # 换成 128/64 会在引擎初始化时报 arange's end argument must be greater than the start argument。
    "head_dim": 512,
    "q_lora_rank": 64,
    "o_groups": 4,
    "o_lora_rank": 32,
    "partial_rotary_factor": 0.125,
    "max_position_embeddings": 128,
    "vocab_size": 128,
    # 注意力层计划：一层 CSA（带 indexer）+ 一层 HCA
    "layer_types": ["compressed_sparse_attention", "heavily_compressed_attention"],
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    # 128 / 128 = 生产几何的取值，也是 FlashInfer 稀疏 prefill 的分块口径：选择是按 block
    # （64/128）来的，给 64/32 会落到没测过的分支（CUDA illegal memory access）
    "sliding_window": 128,
    "index_topk": 128,
    # 16 = deepgemm 在 SM120 上跑 indexer 的 mqa_logits 内核要求 num_heads ∈ {16, 32, 64}
    # （生产几何是 64）；给 8 会在 rollout 起引擎时断言失败
    "index_n_heads": 16,
    # 128 = 生产几何（hf/9b_a4b.json）的取值；也是 vLLM 那个 fused quant+cache 压缩器认的尺寸
    # （只支持 128/512，给 16 会直接 ValueError，rollout 起不来）
    "index_head_dim": 128,
    # MLP 层计划：一层 hash-MoE + 一层普通 MoE
    "mlp_layer_types": ["hash_moe", "moe"],
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "routed_expert_hidden_size": 32,
    "hc_mult": 16,
    "hc_active_streams": 4,
    "hc_fixed_streams": 2,
    "hc_conv_kernels": [4, 8, 12],
    "attn_res_block_size": 4,
    # loss 系数：能进 HF config 的都放这（indexer 的系数只在 mcore 侧：TINY_MCORE_ONLY）
    "router_aux_loss_coef": 0.001,
    "erc_loss_coef": 1.0,
    "erc_loss_alpha": 0.5,
    "num_nextn_predict_layers": 0,
    # AutoBridge 靠它认模型（`from_hf_config` 会校验 architectures 以 ForCausalLM 结尾）
    "architectures": ["ShensiForCausalLM"],
}

# 只在 mcore / Bridge 侧有的旋钮（HF config 里没有这个名字，由 `--shensi-*` 走 CLI）
TINY_MCORE_ONLY: dict[str, object] = {"indexer_loss_coeff": 0.01}

# 词表对齐粒度：mcore 按 make_vocab_size_divisible_by 补齐（哈希嵌入表读 actual_vocab_size），
# HF 侧要用同一个补齐值，否则导出的 ckpt 加载不上（deepemb.weight 行数不一致）。
VOCAB_ALIGN = 128


def aligned_vocab_size(n: int, multiple: int = VOCAB_ALIGN) -> int:
    """把词表大小对齐到 `multiple` 的整数倍（与 mcore 的补齐口径一致）。"""
    return int(-(-int(n) // int(multiple)) * int(multiple))


def tiny_shensi_config(**overrides):
    """极小档的 **HF 侧** `ShensiConfig`（transformers 的那份，Bridge 也认它）。"""
    from transformers.models.shensi import ShensiConfig

    names = {f.name for f in dataclasses.fields(ShensiConfig)}
    unknown = sorted(set(TINY) - names)
    if unknown:
        raise ValueError(f"TINY 里有 HF ShensiConfig 不认识的键（几何漂移了）：{unknown}")
    kwargs = dict(TINY)
    kwargs.update(overrides)
    return ShensiConfig(**kwargs)


@dataclass
class TinyShensiProviderMixin:
    """给 `ShensiModelProvider` 用的极小档字段（`make_tiny_shensi_provider` 组合它）。"""


def make_tiny_shensi_provider(seq_length: int = 128, mtp_layers: int = 0):
    """Bridge 侧的极小 provider（TP=PP=EP=1，单卡可建），供测试直接建模型。"""
    from megatron.bridge.models.shensi.shensi_bridge import ShensiBridge

    hf_cfg = tiny_shensi_config(num_nextn_predict_layers=mtp_layers)
    provider = ShensiBridge().provider_bridge_from_config(hf_cfg)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.seq_length = seq_length
    provider.dsa_kernel_backend = "none"  # 单测环境没有 flash_mla / cudnn 融合内核
    provider.bf16 = True
    provider.params_dtype = __import__("torch").bfloat16
    provider.finalize()
    return provider


def as_cli_overrides(
    *, mtp_layers: int = 0, seq_length: int = 128, iters: int | None = None
) -> list[str]:
    """把极小几何摊成 launcher 的 `--set` 覆盖项（`train.model.*` + 单卡并行口径）。"""
    overrides = [
        *geometry_overrides(mtp_layers=mtp_layers),
        f"train.model.seq_length={seq_length}",
        f"train.model.max_position_embeddings={seq_length}",
        "train.model.micro_batch_size=1",
        "train.model.global_batch_size=2",
        "train.model.eval_iters=0",
        "train.system.tensor_model_parallel_size=1",
        "train.system.pipeline_model_parallel_size=1",
        "train.system.expert_model_parallel_size=1",
        "train.system.context_parallel_size=1",
        "train.system.use_distributed_optimizer=false",
        "train.system.overlap_grad_reduce=false",
        "train.system.overlap_param_gather=false",
        "train.system.checkpoint.no_save_optim=true",
        "train.system.checkpoint.no_save_rng=true",
        "experiment.runner.nproc_per_node=1",
    ]
    if iters is not None:
        overrides.append(f"train.model.train_iters={iters}")
    return overrides


def geometry_overrides(*, mtp_layers: int = 0) -> list[str]:
    """只含**家族几何**的 `--set` 项（不含 seq/vocab/步数这类跟具体 stage 数据有关的值）。"""
    ratios = [4, 128] + [128] * mtp_layers
    return [
        f"train.model.num_layers={TINY['num_hidden_layers']}",
        f"train.model.hidden_size={TINY['hidden_size']}",
        f"train.model.num_attention_heads={TINY['num_attention_heads']}",
        f"train.model.mtp_num_layers={mtp_layers}",
        # 只用数值形式：tiny.yaml 里写的是类型名形式，两个一起给会被几何校验拦下
        "train.model.shensi_attn_layer_types=",
        f"train.model.shensi_compress_ratios={','.join(str(r) for r in ratios)}",
        f"train.model.shensi_mlp_layer_types={','.join(TINY['mlp_layer_types'])}",
        f"train.model.shensi_hc_mult={TINY['hc_mult']}",
        f"train.model.shensi_hc_active_streams={TINY['hc_active_streams']}",
        f"train.model.shensi_hc_fixed_streams={TINY['hc_fixed_streams']}",
        f"train.model.shensi_o_groups={TINY['o_groups']}",
        f"train.model.shensi_head_dim={TINY['head_dim']}",
        f"train.model.shensi_q_lora_rank={TINY['q_lora_rank']}",
        f"train.model.shensi_o_lora_rank={TINY['o_lora_rank']}",
        f"train.model.shensi_partial_rotary_factor={TINY['partial_rotary_factor']}",
        f"train.model.shensi_sliding_window={TINY['sliding_window']}",
        f"train.model.shensi_index_topk={TINY['index_topk']}",
        f"train.model.shensi_index_n_heads={TINY['index_n_heads']}",
        f"train.model.shensi_index_head_dim={TINY['index_head_dim']}",
        f"train.model.shensi_num_experts={TINY['n_routed_experts']}",
        f"train.model.shensi_num_experts_per_tok={TINY['num_experts_per_tok']}",
        f"train.model.shensi_moe_intermediate_size={TINY['moe_intermediate_size']}",
        f"train.model.shensi_routed_expert_hidden_size={TINY['routed_expert_hidden_size']}",
        f"train.model.shensi_attn_res_block_size={TINY['attn_res_block_size']}",
        f"train.model.shensi_erc_loss_coef={TINY['erc_loss_coef']}",
        f"train.model.shensi_erc_loss_alpha={TINY['erc_loss_alpha']}",
        f"train.model.shensi_indexer_loss_coeff={TINY_MCORE_ONLY['indexer_loss_coeff']}",
        f"train.model.moe_aux_loss_coeff={TINY['router_aux_loss_coef']}",
        "train.model.shensi_hf_config=",  # tiny 档不与全量 HF config 对拍
    ]
