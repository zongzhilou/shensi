# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from .fp32_keep import GROUP_MHC, ShensiFp32KeepMixin


def _packed_bounds(packed_seq_params, total: int):
    if packed_seq_params is None:
        return None
    cu = packed_seq_params.cu_seqlens_kv_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_kv
    bounds = [int(v) for v in cu.tolist()]
    if bounds[-1] < total:
        bounds.append(total)
    return bounds


class ShensiUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )


class ShensiHyperConnection(ShensiFp32KeepMixin, MegatronModule):
    SHENSI_FP32_KEEP_GROUP = GROUP_MHC

    def __init__(self, config, layer_number: int = 1, is_mlp: bool = False) -> None:
        super().__init__(config)
        self.hc_mult = int(config.num_residual_streams)
        self.active_streams = int(config.hc_active_streams)
        self.fixed_streams = int(config.hc_fixed_streams)
        self.routed_streams = self.active_streams - self.fixed_streams
        hidden = int(config.hidden_size)
        if not 0 < self.fixed_streams <= self.active_streams <= self.hc_mult:
            raise ValueError(
                "需要 0 < hc_fixed_streams <= hc_active_streams <= hc_mult，得到 "
                f"fixed={self.fixed_streams} active={self.active_streams} mult={self.hc_mult}"
            )
        self.input_norm = ShensiUnweightedRMSNorm(eps=config.layernorm_epsilon)
        self.pre_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))
        self.route_norm = nn.LayerNorm(self.hc_mult * hidden)
        self.route_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))
        self.is_mlp = is_mlp
        self.hc_conv_kernels: Tuple[int, ...] = tuple(
            int(k) for k in config.hc_conv_kernels
        )
        self.kr = (len(self.hc_conv_kernels) + 1) if is_mlp else 1
        if is_mlp:
            if not self.hc_conv_kernels:
                raise ValueError(
                    "hc_conv_kernels 不能为空（MLP 侧 mHC 需要至少一个卷积核）"
                )
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden, hidden, ks, padding=ks - 1, groups=hidden, bias=False
                    )
                    for ks in self.hc_conv_kernels
                ]
            )
        self.post_fn = nn.Parameter(
            torch.empty(self.active_streams * self.kr, self.active_streams * hidden)
        )
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr))
        self.post_scale = nn.Parameter(torch.empty(1))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre = torch.sigmoid(
            F.linear(flat, self.pre_fn.float()) * self.pre_scale.float()
            + self.pre_base.float()
        )
        collapsed = (
            (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        )
        return collapsed

    def _select_active_streams(self, hidden_streams: torch.Tensor):
        s_len, b, hc, _ = hidden_streams.shape
        flat = self.route_norm(
            hidden_streams.flatten(start_dim=2).to(self.route_norm.weight.dtype)
        ).float()
        route_scores = torch.sigmoid(
            F.linear(flat, self.route_fn.float()) * self.route_scale.float()
            + self.route_base.float()
        )
        fixed_mask = torch.arange(hc, device=route_scores.device) < self.fixed_streams
        route_scores = route_scores.masked_fill(
            fixed_mask.view(1, 1, -1), float("-inf")
        )
        fixed_idx = (
            torch.arange(self.fixed_streams, device=hidden_streams.device)
            .view(1, 1, -1)
            .expand(s_len, b, -1)
        )
        routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
        active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
        p = torch.cat(
            [
                torch.ones_like(fixed_idx, dtype=route_scores.dtype),
                route_scores.gather(-1, routed_idx),
            ],
            dim=-1,
        )
        return active_idx, p

    def _augmented_features(
        self,
        sublayer_output: torch.Tensor,
        s_len: int,
        b: int,
        hidden: int,
        packed_seq_params=None,
    ):
        x = (
            sublayer_output.permute(1, 2, 0)
            .contiguous()
            .to(self.temporal_convs[0].weight.dtype)
        )
        bounds = _packed_bounds(packed_seq_params, x.size(-1))
        conv_outs = []
        for conv in self.temporal_convs:
            if bounds is None:
                conv_outs.append(conv(x)[..., :s_len])
            else:
                conv_outs.append(
                    torch.cat(
                        [
                            conv(x[..., s:e])[..., : e - s]
                            for s, e in zip(bounds[:-1], bounds[1:])
                            if e > s
                        ],
                        dim=-1,
                    )
                )
        ortho = []
        prevs = [x]
        for g in conv_outs:
            v = g
            for prev in prevs:
                denom = (
                    (prev * prev)
                    .sum(dim=1, keepdim=True)
                    .clamp_min(self.input_norm.eps)
                )
                v = v - ((prev * v).sum(dim=1, keepdim=True) / denom) * prev
            ortho.append(v)
            prevs.append(v)
        out_aug = (
            torch.cat([x] + ortho, dim=1)
            .transpose(1, 2)
            .reshape(b, s_len, self.kr, hidden)
            .float()
        )
        return out_aug.permute(1, 0, 2, 3)

    def write_back(
        self,
        hidden_streams: torch.Tensor,
        sublayer_output: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        s_len, b, _, hidden = hidden_streams.shape
        active_idx, p = self._select_active_streams(hidden_streams)
        if self.is_mlp:
            out_aug = self._augmented_features(
                sublayer_output, s_len, b, hidden, packed_seq_params
            )
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)
        active_streams = hidden_streams.gather(
            2, active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden)
        )
        post = 2 * torch.sigmoid(
            F.linear(
                self.input_norm(active_streams.flatten(start_dim=2).float()),
                self.post_fn.float(),
            ).view(s_len, b, self.active_streams, self.kr)
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )
        delta = torch.einsum("bskr,bsrh->bskh", post, out_aug) * p.unsqueeze(-1)
        return hidden_streams.scatter(
            2,
            active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden),
            delta.to(hidden_streams.dtype),
        )

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.pre_fn, mean=0.0, std=std)
            nn.init.zeros_(self.pre_base)
            nn.init.constant_(self.pre_scale, 0.01)
            nn.init.normal_(self.route_fn, mean=0.0, std=std)
            nn.init.zeros_(self.route_base)
            nn.init.ones_(self.route_scale)
            nn.init.normal_(self.post_fn, mean=0.0, std=std)
            nn.init.zeros_(self.post_base)
            nn.init.constant_(self.post_scale, 0.01)
            if self.is_mlp:
                for conv in self.temporal_convs:
                    nn.init.kaiming_uniform_(conv.weight, a=5**0.5)


class ShensiHyperHead(ShensiFp32KeepMixin, MegatronModule):
    SHENSI_FP32_KEEP_GROUP = GROUP_MHC

    def __init__(self, config, layer_number: int = 1) -> None:
        super().__init__(config)
        self.hc_mult = int(config.num_residual_streams)
        hidden = int(config.hidden_size)
        self.input_norm = ShensiUnweightedRMSNorm(eps=config.layernorm_epsilon)
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.hc_fn, mean=0.0, std=std)
            nn.init.zeros_(self.hc_base)
            nn.init.ones_(self.hc_scale)
