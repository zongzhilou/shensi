# Copyright (c) 2026 Zongzhi Lou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""移植件的逐位对拍：本仓库自建的模块 vs HF 参考实现。

这是"移植保真"最硬的判据 —— 用 ``torch.equal``（0 位差）而不是容差，毫秒级跑完，也不受
mcore 侧 TE 线性层 fp32=TF32 的影响（那是端到端对拍的下限来源）。

两侧的约定差异只有一处：HF 是 batch-first，本仓库统一 sequence-first，因此输入输出都做转置对齐；
权重按同名同形 1:1 复制。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import ShensiConfig
from transformers.models.shensi.modeling_shensi import (
    ShensiAttentionResidual as HFShensiAttentionResidual,
)
from transformers.models.shensi.modeling_shensi import ShensiHashMLP as HFShensiHashMLP
from transformers.models.shensi.modeling_shensi import (
    ShensiHyperConnection as HFShensiHyperConnection,
)
from transformers.models.shensi.modeling_shensi import ShensiHyperHead as HFShensiHyperHead

from shensi.core.transformer.shensi_attention_residual import ShensiAttentionResidual
from shensi.core.transformer.shensi_block import ShensiHyperHead
from shensi.core.transformer.shensi_hash_mlp import ShensiHashMLP
from shensi.core.transformer.shensi_hyper_connection import ShensiHyperConnection

HIDDEN = 32
STREAMS = 2
ACTIVE = 2
FIXED = 1
BLOCKS = 3
BATCH = 2
SEQ = 5
VOCAB = 64


class _MCoreConfig:
    """只提供移植件读取的字段（mcore 侧字段名），避免为了单测去建完整 TransformerConfig。"""

    def __init__(self, hf_config: ShensiConfig):
        self.hidden_size = hf_config.hidden_size
        self.layernorm_epsilon = hf_config.rms_norm_eps
        self.params_dtype = torch.float32
        self.mhc_num_residual_streams = hf_config.hc_mult
        self.hc_active_streams = hf_config.hc_active_streams
        self.hc_fixed_streams = hf_config.hc_fixed_streams
        self.hc_conv_kernels = list(hf_config.hc_conv_kernels)
        self.add_bias_linear = bool(hf_config.mlp_bias)
        self.vocab_size = hf_config.vocab_size
        self.actual_vocab_size = hf_config.vocab_size
        # hash MLP 的中间维就是潜维（mcore 的 latent MoE 字段），激活与 clamp 也取自 provider。
        self.moe_latent_size = hf_config.routed_expert_hidden_size
        self.activation_func = F.silu
        self.activation_func_clamp_value = hf_config.swiglu_limit


@pytest.fixture(scope="module")
def configs():
    torch.manual_seed(0)
    hf_config = ShensiConfig(
        num_hidden_layers=2,
        hidden_size=HIDDEN,
        head_dim=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=VOCAB,
        hc_mult=STREAMS,
        hc_active_streams=ACTIVE,
        hc_fixed_streams=FIXED,
        hc_conv_kernels=[2],
        mlp_bias=False,
    )
    return hf_config, _MCoreConfig(hf_config)


def _twin(hf_module, our_module):
    """按同名同形复制权重，返回并行的一对模块（fp32）。

    HF 侧有些参数是 ``torch.empty`` 建的、初始化放在模型级 ``_init_weights`` 里，单独构造模块
    时是未初始化内存（会出 NaN），所以先做一遍确定性初始化再复制 —— 取值本身无所谓，两侧一致
    即可。
    """
    torch.manual_seed(1234)
    for param in hf_module.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    hf_state, our_state = hf_module.state_dict(), our_module.state_dict()
    assert set(hf_state) == set(our_state), set(hf_state) ^ set(our_state)
    for name, tensor in hf_state.items():
        assert tensor.shape == our_state[name].shape, name
    our_module.load_state_dict(hf_state)
    return hf_module.eval(), our_module.eval()


def test_hyper_connection_matches_hf(configs):
    """超连接：塌缩（forward）与回写（write_back）都要逐位一致。"""
    hf_config, mc_config = configs
    hf, ours = _twin(
        HFShensiHyperConnection(hf_config, is_mlp=True).float(),
        ShensiHyperConnection(mc_config, is_mlp=True).float(),
    )
    streams = torch.randn(BATCH, SEQ, STREAMS, HIDDEN)
    sublayer = torch.randn(BATCH, SEQ, HIDDEN)

    with torch.no_grad():
        hf_collapsed = hf(streams)
        our_collapsed = ours(streams.transpose(0, 1))
        hf_written = hf.write_back(streams, sublayer)
        our_written = ours.write_back(streams.transpose(0, 1), sublayer.transpose(0, 1))

    assert torch.equal(our_collapsed, hf_collapsed.transpose(0, 1))
    assert torch.equal(our_written, hf_written.transpose(0, 1))


def test_hyper_connection_attention_site_matches_hf(configs):
    """注意力侧（``is_mlp=False``，``kr=1``）也要逐位一致。"""
    hf_config, mc_config = configs
    hf, ours = _twin(
        HFShensiHyperConnection(hf_config, is_mlp=False).float(),
        ShensiHyperConnection(mc_config, is_mlp=False).float(),
    )
    streams = torch.randn(BATCH, SEQ, STREAMS, HIDDEN)
    sublayer = torch.randn(BATCH, SEQ, HIDDEN)

    with torch.no_grad():
        hf_written = hf.write_back(streams, sublayer)
        our_written = ours.write_back(streams.transpose(0, 1), sublayer.transpose(0, 1))

    assert torch.equal(our_written, hf_written.transpose(0, 1))


# HF 的 has_router=False 表示"前面还没有块"，此时调用方传的 num_blocks 必然是 0（它的 forward
# 在 num_blocks>0 时会用 q_proj，而那时 q_proj 是 None），所以两者要配套。
@pytest.mark.parametrize("has_router,num_blocks", ((False, 0), (True, BLOCKS)))
def test_attention_residual_matches_hf(configs, has_router, num_blocks):
    """块记忆：三态输出（output / updated / blocks）逐位一致。"""
    hf_config, mc_config = configs
    hf, ours = _twin(
        HFShensiAttentionResidual(hf_config, has_router).float(),
        ShensiAttentionResidual(mc_config, has_router).float(),
    )
    delta = torch.randn(BATCH, SEQ, STREAMS, HIDDEN)
    library = torch.randn(BATCH, SEQ, STREAMS, BLOCKS, HIDDEN)
    prefix_sum = torch.randn(BATCH, SEQ, STREAMS, HIDDEN)
    norm_weight = torch.randn(HIDDEN)

    with torch.no_grad():
        hf_out = hf(
            delta, library, prefix_sum, output_norm_weight=norm_weight, num_blocks=num_blocks
        )
        our_out = ours(
            delta.transpose(0, 1),
            library.transpose(0, 1),
            prefix_sum.transpose(0, 1),
            output_norm_weight=norm_weight,
            num_blocks=num_blocks,
        )

    for our_tensor, hf_tensor in zip(our_out, hf_out):
        assert torch.equal(our_tensor, hf_tensor.transpose(0, 1))


def test_hash_mlp_matches_hf(configs):
    """hash 层：deepemb 门控逐位一致（含 input_ids 的 batch/sequence 约定转换）。"""
    hf_config, mc_config = configs
    hf, ours = _twin(HFShensiHashMLP(hf_config).float(), ShensiHashMLP(mc_config).float())
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))

    with torch.no_grad():
        hf_out = hf(hidden, input_ids)
        our_out = ours(hidden.transpose(0, 1), input_ids)

    assert torch.equal(our_out, hf_out.transpose(0, 1))


def test_hyper_head_matches_hf(configs):
    """出口的流收缩（``hc_fn`` / ``hc_base`` / ``hc_scale``，不加 eps）。"""
    hf_config, mc_config = configs
    hf, ours = _twin(HFShensiHyperHead(hf_config).float(), ShensiHyperHead(mc_config).float())
    streams = torch.randn(BATCH, SEQ, STREAMS, HIDDEN)

    with torch.no_grad():
        hf_out = hf(streams)
        our_out = ours(streams.transpose(0, 1))

    assert torch.equal(our_out, hf_out.transpose(0, 1))
