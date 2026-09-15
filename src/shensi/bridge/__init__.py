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

"""镜像 Megatron-Bridge ``src/megatron/bridge/`` 的部分。

导入本模块即完成 bridge 注册（与上游 ``megatron/bridge/__init__.py`` 触发模型注册同理）。
"""

from .models.shensi import ShensiBridge, ShensiModelProvider

__all__ = ["ShensiBridge", "ShensiModelProvider"]
