"""极小几何的单一出处：单卡冒烟 / 集成测试（test_train.py）都用这里的定义。

两个用途：

- `as_cli_overrides()` 把几何摊成 launcher 的 `--set` 覆盖项，各 stage 的 `test_train.py` 直接拿去跑；
- `make_tiny_shensi_provider()` 给 Bridge 侧（`ShensiModelProvider`）一个同样机集合的 provider，
  便于在测试里直接建模型看形状/参数量。

几何原则：保留家族独有的每一处结构（CSA/HCA、DSA indexer、mHC 多流、AttnRes block、hash-MoE、
可选的 MTP），只是把规模压到单卡几十秒能跑完。改这里的值 = 改所有 tiny 档。
"""

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
    # 512 = 参考几何的取值，也是 vLLM 在 SM120 上跑 DSv4 的硬要求：压缩器的 fused quant+cache
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
    # 128 / 128 = 参考几何的取值，也是 FlashInfer 稀疏 prefill 的分块口径：选择是按 block
    # （64/128）来的，给 64/32 会落到没测过的分支（CUDA illegal memory access）
    "sliding_window": 128,
    "index_topk": 128,
    # 16 = deepgemm 在 SM120 上跑 indexer 的 mqa_logits 内核要求 num_heads ∈ {16, 32, 64}
    # （参考几何是 64）；给 8 会在 rollout 起引擎时断言失败
    "index_n_heads": 16,
    # 128 = 参考几何（hf/9b_a4b.json）的取值；也是 vLLM 那个 fused quant+cache 压缩器认的尺寸
    # （只支持 128/512，给 16 会直接 ValueError，rollout 起不来）
    "index_head_dim": 128,
    # MLP 层计划：一层 hash-MoE + 一层普通 MoE
    "mlp_layer_types": ["hash_moe", "moe"],
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "routed_expert_hidden_size": 32,
    # mHC / AttnRes
    "hc_mult": 16,
    "hc_active_streams": 4,
    "hc_fixed_streams": 2,
    "hc_conv_kernels": [4, 8, 12],
    "attn_res_block_size": 4,
    # loss 系数：能进 HF config 的都放这（indexer 的系数只在 mcore 侧，见 TINY_MCORE_ONLY）
    "router_aux_loss_coef": 0.001,
    "erc_loss_coef": 1.0,
    "erc_loss_alpha": 0.5,
    "num_nextn_predict_layers": 0,
}

# 只在 mcore / Bridge 侧有的旋钮（HF config 里没有这个名字，由 `--shensi-*` 走 CLI）
TINY_MCORE_ONLY: dict[str, object] = {"indexer_loss_coeff": 0.01}


def tiny_shensi_config(**overrides):
    """极小档的 **HF 侧** `ShensiConfig`（transformers 的那份，Bridge 也认它）。

    注意别拿 Bridge 的 `shensi_config.ShensiConfig` 来对字段：那是 mcore 侧的子类，
    字段名与 HF 会漂移（踩过：它没有 `qk_rope_head_dim`）。
    """
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
    """Bridge 侧的极小 provider（TP=PP=EP=1，单卡可建），供测试直接建模型。

    `mtp_layers>0` 时带上 MTP；注意我们的 MTP 目前只支持**单层**（见 recipes README 的局限）。
    """
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


def as_cli_overrides(*, mtp_layers: int = 0, seq_length: int = 128) -> list[str]:
    """把极小几何摊成 launcher 的 `--set` 覆盖项（`train.model.*`）。"""
    ratios = [4, 128] + [128] * mtp_layers
    overrides = [
        f"train.model.num_layers={TINY['num_hidden_layers']}",
        f"train.model.hidden_size={TINY['hidden_size']}",
        f"train.model.num_attention_heads={TINY['num_attention_heads']}",
        f"train.model.seq_length={seq_length}",
        f"train.model.max_position_embeddings={seq_length}",
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
        f"train.model.shensi_indexer_loss_coeff={TINY['indexer_loss_coeff']}",
        f"train.model.moe_aux_loss_coeff={TINY['router_aux_loss_coef']}",
        "train.model.shensi_hf_config=",  # tiny 档不与全量 HF config 对拍
        "train.model.micro_batch_size=1",
        "train.model.global_batch_size=2",
        "train.model.train_iters=5",
        "train.model.eval_iters=0",
        "train.system.checkpoint.no_save_optim=true",
        "train.system.checkpoint.no_save_rng=true",
    ]
    return overrides
