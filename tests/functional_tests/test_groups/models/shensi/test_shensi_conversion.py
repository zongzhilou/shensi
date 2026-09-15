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

"""Shensi 的权重转换与前向数值对拍。

覆盖 Megatron-Bridge ``docs/adding-new-models.md`` 第 5/7 步要的两类证据：

* HF→Megatron 载入不丢参数、Megatron→HF 导出与检查点**逐位一致**；
* 前向对拍：fp32 逐层三态 + logits 落在比对下限内（这个下限由 mcore 的 TE 线性层决定 —— 它在
  fp32 下走 TF32，相对误差 ~5e-3，而 HF 侧是真 fp32，所以判据取 atol 2e-5 + rtol 2e-3；
  架构级错误会是 O(0.1~1)，仍有 2~3 个数量级的判别力）。

fp32 对拍用"全 sliding"检查点：压缩层（CSA/HCA）的 Hadamard 旋转只支持 bf16，绕开它才能在
fp32 下比。逐位证据由 ``tests/unit_tests`` 的移植件对拍提供。
"""

from __future__ import annotations

import glob
import os

import pytest
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, ShensiConfig

import shensi.bridge  # noqa: F401  注册 bridge
from megatron.bridge import AutoBridge

from shensi.core.transformer.state import unpack_state

SEQ_LEN = 16
SLIDING = ["sliding_attention"] * 4
MIXED = [
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
]

#: tiny 配置：小到秒级可建，但把 Shensi 的结构分支都打开（hash 层 + MoE + 三态块读写 + 压缩路径）。
TINY = dict(
    architectures=["ShensiForCausalLM"],
    num_hidden_layers=4,
    hidden_size=32,
    head_dim=16,
    # 融合 MLA rope 内核要求 rope 维能被 4 整除（默认 64/512 在 head_dim=16 下只有 2 维）。
    partial_rotary_factor=0.5,
    num_attention_heads=2,
    num_key_value_heads=1,
    q_lora_rank=8,
    o_groups=2,
    o_lora_rank=8,
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=24,
    routed_expert_hidden_size=16,
    vocab_size=256,
    max_position_embeddings=4096,
    sliding_window=8,
    hc_mult=2,
    hc_active_streams=2,
    hc_fixed_streams=1,
    hc_conv_kernels=[2],
    attn_res_block_size=2,
    # Shensi v1 不启用 MTP：MTP + 超连接在上游要求 HybridModel 契约，未适配。
    num_nextn_predict_layers=0,
    mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="权重转换与前向对拍需要 GPU")


@pytest.fixture(autouse=True, scope="module")
def _dist_env():
    """mcore 需要进程组环境变量；单进程单卡即可。"""
    env = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29700 + os.getpid() % 400),
    }
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_ckpt(directory, layer_types, dtype) -> str:
    """按 tiny 配置随机初始化一个 HF 检查点（固定种子，保证判据可复现）。

    注意 dtype 要同时写进 config：它决定 provider 的 bf16 开关、进而决定 mcore 是否套
    ``Float16Module`` —— 想要真正的 fp32 前向，检查点本身就必须是 fp32。
    """
    config = ShensiConfig(**{**TINY, "layer_types": list(layer_types), "dtype": dtype})
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(dtype).save_pretrained(directory)
    return str(directory)


@pytest.fixture(scope="module")
def ckpt_mixed(tmp_path_factory) -> str:
    """混合 layer_types（bf16）：覆盖压缩稀疏/重压缩注意力、压缩器与 Indexer。"""
    return _write_ckpt(tmp_path_factory.mktemp("shensi_mixed"), MIXED, torch.bfloat16)


@pytest.fixture(scope="module")
def ckpt_sliding(tmp_path_factory) -> str:
    """全 sliding（fp32）：可以在两端都跑纯 fp32 的逐层对拍。"""
    return _write_ckpt(tmp_path_factory.mktemp("shensi_sliding"), SLIDING, torch.float32)


def _hf_model(path: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval().cuda()
    if dtype == torch.bfloat16:
        # HF 的 _keep_in_fp32_modules_strict 把 norm / 超连接等参数钉在 fp32，而前向会把这些
        # 激活提升回 fp32、与 bf16 权重冲突（参考实现的 bf16 前向本身跑不通）。按 dtype 语义统一。
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
    return model


def _megatron_model(path: str, dtype: torch.dtype):
    bridge = AutoBridge.from_hf_pretrained(path)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.params_dtype = dtype
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()
    bridge.load_hf_weights([model])
    inner = model.module if hasattr(model, "module") else model
    return bridge, model, inner


def _inputs():
    torch.manual_seed(0)
    ids = torch.randint(0, TINY["vocab_size"], (1, SEQ_LEN)).cuda()
    return ids, torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)


def _capture(model, layers, ids, position_ids, *, megatron: bool):
    """抓每层输出：HF 是 ``(streams, prefix_sum, library)``，mcore 是打包三态。"""
    acts = {}
    handles = [
        layer.register_forward_hook(lambda m, a, out, i=i: acts.__setitem__(i, out))
        for i, layer in enumerate(layers)
    ]
    with torch.no_grad():
        if megatron:
            out = model(ids, position_ids, None)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            logits = model(ids, position_ids=position_ids).logits
    for handle in handles:
        handle.remove()
    return logits, acts


def _checkpoint_tensors(path: str) -> dict:
    """按**文件里的键**读检查点。

    HF 的读层 router/experts 与写层共享同一份张量（tie_moe_groups），内存里的 state_dict 会带出
    这些别名键，而保存时会被 safetensors 去重 —— 参照物必须是文件内容。
    """
    tensors = {}
    for filename in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(filename, framework="pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def _max_abs_diff(got, ref) -> float:
    return float((got.float() - ref.float()).abs().max())


class TestShensiConversion:
    """HF ↔ Megatron 的权重覆盖与往返一致性。"""

    @pytest.fixture(scope="class")
    def roundtrip(self, ckpt_mixed):
        return _megatron_model(ckpt_mixed, torch.bfloat16)

    def test_load_covers_all_hf_params(self, ckpt_mixed, roundtrip):
        """载入成功即"无缺失"；再要求导出键集合与检查点完全一致（不多不少）。"""
        bridge, model, _ = roundtrip
        ckpt_keys = set(_checkpoint_tensors(ckpt_mixed))
        exported_keys = {
            name for name, _ in bridge.export_hf_weights([model], cpu=True, show_progress=False)
        }
        assert exported_keys == ckpt_keys

    def test_export_roundtrip_bit_exact(self, ckpt_mixed, roundtrip):
        bridge, model, _ = roundtrip
        ckpt_tensors = _checkpoint_tensors(ckpt_mixed)
        worst = 0.0
        for name, tensor in bridge.export_hf_weights([model], cpu=True, show_progress=False):
            worst = max(worst, _max_abs_diff(tensor, ckpt_tensors[name]))
        assert worst == 0.0, f"导出与检查点不逐位一致：最大绝对差 {worst:.3e}"


class TestShensiForwardParity:
    """HF 与 mcore 的前向数值对拍。"""

    def test_fp32_layerwise_and_logits(self, ckpt_sliding):
        """fp32：逐层三态与 logits 都落在比对下限内（判据量级见模块 docstring）。"""
        ids, position_ids = _inputs()
        hf = _hf_model(ckpt_sliding, torch.float32)
        hf_logits, hf_acts = _capture(hf, hf.model.layers, ids, position_ids, megatron=False)
        config = hf.config
        num_layers = int(config.num_hidden_layers)
        num_streams = int(config.hc_mult)
        num_blocks = sum(1 for t in config.attn_res_block_layer_types if t == "block_write_layer")
        hidden = int(config.hidden_size)
        del hf
        torch.cuda.empty_cache()

        _, model, inner = _megatron_model(ckpt_sliding, torch.float32)
        mc_logits, mc_acts = _capture(
            model, inner.decoder.layers, ids, position_ids, megatron=True
        )

        for index in range(num_layers):
            streams_ref, prefix_ref, library_ref = hf_acts[index]
            streams, prefix_sum, library = unpack_state(
                mc_acts[index][0], num_streams, num_blocks, hidden
            )
            for name, got, ref in (
                ("streams", streams, streams_ref.transpose(0, 1)),
                ("prefix_sum", prefix_sum, prefix_ref.transpose(0, 1)),
                ("library", library, library_ref.transpose(0, 1)),
            ):
                diff = _max_abs_diff(got, ref)
                budget = 2e-5 + 2e-3 * float(ref.float().abs().max())
                print(f"layer {index} {name}: 最大差={diff:.3e} 预算={budget:.3e}")
                assert diff <= budget, f"layer {index} {name} 超出比对下限：{diff:.3e}"

        diff = _max_abs_diff(mc_logits, hf_logits)
        budget = 2e-5 + 2e-3 * float(hf_logits.float().abs().max())
        print(f"logits: 最大差={diff:.3e} 预算={budget:.3e}")
        assert diff <= budget, f"logits 超出比对下限：{diff:.3e}"
