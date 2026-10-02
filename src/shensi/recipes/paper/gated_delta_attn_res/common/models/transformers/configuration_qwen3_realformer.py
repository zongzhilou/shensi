# coding=utf-8
# Copyright 2026 The FlagOS Contributors and HuggingFace Inc. team. All rights reserved.
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

"""RealFormer 的 HF 配置定义。"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3RealFormerConfig(Qwen3Config):

    model_type = "qwen3_realformer"



    attn_res_realformer_gate: str = "deviation"

    attn_res_realformer_mean: bool = False



    auto_map = {
        "AutoConfig": "configuration_qwen3_realformer.Qwen3RealFormerConfig",
        "AutoModel": "modeling_qwen3_realformer.Qwen3RealFormerModel",
        "AutoModelForCausalLM": "modeling_qwen3_realformer.Qwen3RealFormerForCausalLM",
    }

    def to_dict(self):
        output = super().to_dict()
        for name in _EXTRA_CONFIG_FIELDS + ("auto_map",):
            output.setdefault(name, getattr(self, name))
        return output



_EXTRA_CONFIG_FIELDS = tuple(Qwen3RealFormerConfig.__annotations__)

try:
    AutoConfig.register("qwen3_realformer", Qwen3RealFormerConfig)
except ValueError:
    pass

try:
    Qwen3RealFormerConfig.register_for_auto_class("AutoConfig")
except (AttributeError, ValueError):
    pass
