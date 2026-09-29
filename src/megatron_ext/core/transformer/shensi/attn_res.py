# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import os
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from .fp32_keep import GROUP_ATTN_RES, ShensiFp32KeepMixin


class ShensiAttentionResidual(ShensiFp32KeepMixin, MegatronModule):
    SHENSI_FP32_KEEP_GROUP = GROUP_ATTN_RES

    def __init__(self, config, layer_number: int = 1) -> None:
        super().__init__(config)
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.eps = float(config.layernorm_epsilon)
        self.read_heads = int(getattr(config, "attn_res_read_heads", 8) or 8)
        rank = int(config.routed_expert_hidden_size)
        self.g_a_proj = nn.Linear(hidden, rank, bias=True)
        self.g_b_proj = nn.Linear(rank, 3 * hidden, bias=True)
        self.q_a_proj = nn.Linear(hidden, rank, bias=False)
        self.q_b_proj = nn.Linear(rank, hidden, bias=False)
        self.k_a_proj = nn.Linear(hidden, rank, bias=False)
        self.k_b_proj = nn.Linear(rank, hidden, bias=False)
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.t = nn.Parameter(
            torch.linspace(0.0, 1.0, hidden) * math.log(2.0 * int(config.num_layers))
        )

    def forward(
        self,
        prefix: torch.Tensor,
        delta: Optional[torch.Tensor],
        blocks: torch.Tensor,
        output_norm_weight: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self._norm(prefix + (delta if delta is not None else 0.0))
        g0, g1 = self.g_a_proj, self.g_b_proj
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        gates = F.linear(
            F.linear(state, g0.weight.float(), g0.bias.float()), g1.weight.float(), g1.bias.float()
        )
        r_decay, r_erase, r_write = gates.reshape(*state.shape[:-1], 3, -1).unbind(-2)
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * self.t.exp())
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale
        k0, k1 = self.k_a_proj, self.k_b_proj
        khat = F.normalize(
            F.linear(F.linear(delta if delta is not None else state, k0.weight.float()), k1.weight.float()),
            dim=-1,
        )
        m = decay * prefix + write * (delta if delta is not None else 0.0)
        lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
        updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)
        if num_blocks > 0:
            values = torch.cat(
                [blocks[..., :num_blocks, :].float(), updated.unsqueeze(-2)], dim=-2
            )
            q0, q1 = self.q_a_proj, self.q_b_proj
            query = F.linear(F.linear(state, q0.weight.float()), q1.weight.float())
            dim = values.shape[-1]
            head_dim = dim // self.read_heads
            flat_v = values.reshape(-1, self.read_heads, head_dim)
            flat_q = query.reshape(-1, self.read_heads, head_dim)
            with torch.no_grad():
                V = flat_v.detach()
                n = V.shape[0]
                cov = torch.einsum("nhd,nhe->hde", V, V) / n
                scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
                ridge = max(head_dim, n) * torch.finfo(cov.dtype).eps * scale
                cov.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
                evals, evecs = torch.linalg.eigh(cov)
                floor = (evals[..., -1:] * head_dim * torch.finfo(cov.dtype).eps).clamp_min(torch.finfo(cov.dtype).tiny)
                whiten = evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor))) @ evecs.transpose(-1, -2)
                v = torch.einsum("nhd,hde->nhe", flat_v, whiten)
                q = torch.einsum("nhd,hde->nhe", flat_q, whiten)
            v = v.view(*values.shape[:-1], self.read_heads, -1)
            q = q.view(*query.shape[:-1], self.read_heads, -1)
            logits = (v * q.unsqueeze(-3)).sum(dim=-1) * torch.rsqrt(v.square().mean(dim=-1) + self.eps)
            s = torch.logsumexp(logits, dim=-2, keepdim=True)
            scores = torch.exp(logits - F.softplus(s))
            routed = (scores.unsqueeze(-1) * values.view_as(v)).sum(dim=-3).reshape(
                *values.shape[:-2], values.shape[-1]
            ) * read_scale
        else:
            routed = torch.zeros_like(updated)
        output = updated + routed
        if output_norm_weight is not None:
            reciprocal_std = torch.rsqrt(
                output.square().mean(dim=-1, keepdim=True) + self.eps
            )
            output = output * reciprocal_std * output_norm_weight.float()
        return output.to(out_dtype), updated.to(out_dtype)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)

    def init_shensi_weights(
        self, std: float = 0.02, hidden_size: Optional[int] = None
    ) -> None:
        d = int(hidden_size or self.hidden_size)
        with torch.no_grad():
            nn.init.zeros_(self.g_a_proj.weight)
            nn.init.zeros_(self.g_b_proj.weight)
            bias = self.g_b_proj.bias
            bias[:d] = 2.0
            bias[d : 2 * d] = -2.0
            bias[2 * d :] = -2.0
            nn.init.zeros_(self.q_a_proj.weight)
            nn.init.zeros_(self.q_b_proj.weight)
            nn.init.normal_(self.k_a_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.k_b_proj.weight, mean=0.0, std=std)

class ShensiAttnResState:
    def __init__(
        self,
        num_blocks: int,
        *,
        hc_mult: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.hc_mult = None if hc_mult is None else int(hc_mult)
        self.hidden_size = None if hidden_size is None else int(hidden_size)
        self.prefix_sum: Optional[torch.Tensor] = None
        self.residual: Optional[torch.Tensor] = None
        self.num_written = 0
        self.pp_imported = False

    def begin(self, stream: torch.Tensor) -> None:
        if stream.dim() != 4:
            raise ValueError(
                f"ShensiAttnResState.begin 期望 4D n 流输入，得到 {stream.shape}"
            )
        self.prefix_sum = stream
        s, b, hc, hidden = stream.shape
        self.residual = stream.new_zeros(s, b, hc, self.num_blocks, hidden)
        self.hc_mult, self.hidden_size = int(hc), int(hidden)
        self.num_written = 0
        self.pp_imported = False

    def write_block(self, slot: int, prefix_sum: torch.Tensor) -> None:
        if self.residual is None:
            raise RuntimeError("AttnRes 状态未初始化：block 第一层没有先调用 begin()")
        self.residual[..., slot, :] = prefix_sum.to(self.residual.dtype)
        self.num_written = max(self.num_written, int(slot) + 1)

    def check_ready(self, layer_number: int) -> None:
        if self.prefix_sum is None or self.residual is None:
            raise RuntimeError(
                f"AttnRes 状态未初始化（当前层 layer_number={layer_number} 不是全局第 1 层，"
                "也没有任何层调用过 begin()，且没有从上游 PP stage 导入）。见 "
                "attn_res 的 PP>1 说明。"
            )

    def state_for_pp(self) -> torch.Tensor:
        if self.prefix_sum is None or self.residual is None:
            raise RuntimeError("AttnRes 状态未初始化：无法导出 PP payload")
        ps, rs = self.prefix_sum, self.residual
        s, b, hc, hidden = ps.shape
        nb = int(rs.shape[-2])
        header = torch.tensor(
            [MAGIC, VERSION, s, b, hc, hidden, nb, self.num_written],
            dtype=torch.float32,
            device=ps.device,
        )
        payload = torch.cat(
            [header, ps.reshape(-1).float(), rs.reshape(-1).float()], dim=0
        )
        _LIVE_PAYLOADS.add(payload)
        return payload

    def load_from_pp(
        self,
        payload: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        expected_hidden_flat_numel: Optional[int] = None,
    ) -> Dict[str, object]:
        if not torch.is_tensor(payload):
            raise ValueError(
                f"state_for_pp payload 必须是张量，得到 {type(payload).__name__}"
            )
        if payload.dim() != 1:
            raise ValueError(f"payload 必须是一维张量，得到 {tuple(payload.shape)}")
        if payload.numel() < HEADER_LEN:
            raise ValueError(f"payload 太短（{payload.numel()} < header {HEADER_LEN}）")
        if payload.dtype != torch.float32:
            raise ValueError(f"payload 必须是 fp32（打包约定），得到 {payload.dtype}")
        header = payload[:HEADER_LEN].tolist()
        magic, version = int(header[0]), int(header[1])
        if magic != MAGIC:
            raise ValueError(
                f"payload 魔数不对（{magic} != {MAGIC}）：这不是 AttnRes 的 PP 状态载荷"
            )
        if version != VERSION:
            raise ValueError(f"payload 版本不支持（{version} != {VERSION}）")
        s, b, hc, hidden, nb = (int(v) for v in header[2:7])
        num_written = int(header[7])
        expect = HEADER_LEN + s * b * hc * hidden * (1 + nb)
        if int(payload.numel()) != expect:
            raise ValueError(
                f"payload 元素数不符：头部声明 {expect}（s={s}, b={b}, hc={hc}, H={hidden}, "
                f"nb={nb}），实际 {int(payload.numel())}；请检查两侧的 num_residual_streams / "
                "hidden_size / AttnRes block 数是否一致"
            )
        if expected_hidden_flat_numel is not None and s * b * hc * hidden != int(
            expected_hidden_flat_numel
        ):
            raise ValueError(
                f"payload 的 n 流元素数 {s * b * hc * hidden} 与本层收到的 hidden 元素数 "
                f"{int(expected_hidden_flat_numel)} 不一致"
            )
        if nb != self.num_blocks:
            raise ValueError(
                f"payload 的 block 槽数 {nb} 与本地 ShensiAttnResState.num_blocks "
                f"{self.num_blocks} 不一致：PP>1 时每个 stage 的栈必须按**全局** block 数建"
                "（bind_attn_res_state(num_blocks=num_attn_res_blocks(...))，不要按本 stage "
                "的层数算）"
            )
        if num_written < 0 or num_written > nb:
            raise ValueError(f"payload 的已写槽数非法：{num_written}（槽数 {nb}）")
        n1 = s * b * hc * hidden
        prefix_sum = (
            payload[HEADER_LEN : HEADER_LEN + n1].reshape(s, b, hc, hidden).clone()
        )
        residual = payload[HEADER_LEN + n1 :].reshape(s, b, hc, nb, hidden).clone()
        if dtype is not None:
            prefix_sum = prefix_sum.to(dtype)
            residual = residual.to(dtype)
        self.prefix_sum = prefix_sum
        self.residual = residual
        self.hc_mult, self.hidden_size = int(hc), int(hidden)
        self.num_written = num_written
        self.pp_imported = True
        return {
            "shape": (s, b, hc, hidden),
            "num_blocks": nb,
            "num_written": num_written,
            "numel": int(payload.numel()),
            "dtype": str(self.prefix_sum.dtype),
        }


def bind_attn_res_state(
    block, config, num_blocks: int, *, handoff=None, plan=None
) -> ShensiAttnResState:
    check_attn_res_pp_support(config, plan=plan)
    state = ShensiAttnResState(
        num_blocks=num_blocks,
        hc_mult=int(getattr(config, "num_residual_streams", 1) or 1),
        hidden_size=int(getattr(config, "hidden_size", 0) or 0) or None,
    )
    layers = getattr(block, "layers", None)
    if layers is None:
        raise AttributeError(
            f"{type(block).__name__} 没有 layers，无法绑定 AttnRes 状态"
        )
    for layer in layers:
        layer.attn_res_state = state
        layer.attn_res_handoff = handoff
    block.shensi_attn_res_state = state
    block.shensi_attn_res_handoff = handoff
    return state


def bind_mtp_attn_res_state(mtp_block, config) -> dict[int, ShensiAttnResState]:
    layers = getattr(mtp_block, "layers", None) or []
    states: dict[int, ShensiAttnResState] = {}
    for idx, mtp_layer in enumerate(layers):
        inner = getattr(mtp_layer, "mtp_model_layer", None)
        if inner is None:
            raise AttributeError(
                f"{type(mtp_layer).__name__} 没有 mtp_model_layer（mcore MTP 内层），无法绑 AttnRes"
            )
        state = ShensiAttnResState(
            num_blocks=1,
            hc_mult=int(getattr(config, "num_residual_streams", 1) or 1),
            hidden_size=int(getattr(config, "hidden_size", 0) or 0) or None,
        )
        inner.attn_res_state = state
        inner.attn_res_handoff = None
        states[idx] = state
    return states


def attn_res_block_layer_types(
    num_layers: int, n_hash_layers: int, block_size: int
) -> List[str]:
    return [
        (
            "block_write_layer"
            if i == 0 or (i >= n_hash_layers and (i - n_hash_layers) % block_size == 0)
            else "block_read_layer"
        )
        for i in range(num_layers)
    ]


def num_attn_res_blocks(num_layers: int, n_hash_layers: int, block_size: int) -> int:
    return attn_res_block_layer_types(num_layers, n_hash_layers, block_size).count(
        "block_write_layer"
    )


def prev_valid_blocks(
    layer_idx: int, num_layers: int, n_hash_layers: int, block_size: int
) -> int:
    types = attn_res_block_layer_types(num_layers, n_hash_layers, block_size)
    return sum(1 for t in types[:layer_idx] if t == "block_write_layer")


MAGIC = 21320
VERSION = 1
HEADER_LEN = 8
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float64: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class AttnResPPLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class AttnResBlock:
    index: int
    write_layer: int
    read_layers: Tuple[int, ...]

    @property
    def layers(self) -> Tuple[int, ...]:
        return (self.write_layer,) + self.read_layers


def blocks_of(
    num_layers: int, n_hash_layers: int, block_size: int
) -> Tuple[AttnResBlock, ...]:
    types = attn_res_block_layer_types(num_layers, n_hash_layers, block_size)
    writes = [i for i, t in enumerate(types) if t == "block_write_layer"]
    out: List[AttnResBlock] = []
    for bi, w in enumerate(writes):
        end = writes[bi + 1] if bi + 1 < len(writes) else num_layers
        out.append(
            AttnResBlock(index=bi, write_layer=w, read_layers=tuple(range(w + 1, end)))
        )
    return tuple(out)


def _upstream_layout_vp_ok(vpp_size: int) -> bool:
    if vpp_size <= 1:
        return True
    try:
        from megatron.core import parallel_state as ps

        return ps.get_virtual_pipeline_model_parallel_world_size() is not None
    except Exception:
        return False


def _stage_layers_from_layout(layout, pp_size: int, vpp_size: int):
    from megatron_ext.core.models.shensi.shensi_layer_specs import layer_ids_from_layout

    if _upstream_layout_vp_ok(vpp_size):
        return {
            (pp, vp): tuple(layer_ids_from_layout(layout, vp_stage=vp, pp_rank=pp))
            for vp in range(vpp_size)
            for pp in range(pp_size)
        }
    from megatron.core.transformer.enums import LayerType

    out = {}
    offset = 0
    for vp in range(vpp_size):
        for pp in range(pp_size):
            n = layout.layout[pp][vp].count(LayerType.decoder)
            out[(pp, vp)] = tuple(range(offset, offset + n))
            offset += n
    return out


@dataclass
class LayoutReport:
    ok: bool
    pp_size: int = 1
    vpp_size: int = 1
    num_layers: int = 0
    stage_layers: Dict[Tuple[int, int], Tuple[int, ...]] = field(default_factory=dict)
    boundaries: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(
        default_factory=list
    )
    crossing_blocks: List[Tuple[AttnResBlock, Tuple[Tuple[int, int], ...]]] = field(
        default_factory=list
    )
    handoff_required: bool = False
    handoff_enabled: Optional[bool] = None
    problems: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - 只为可读日志
        lines = [
            f"[attn_res_pp] 布局校验：{'PASS' if self.ok else 'FAIL'} "
            f"(pp={self.pp_size}, vpp={self.vpp_size}, num_layers={self.num_layers})",
            "  stage -> 全局层号: "
            + ", ".join(
                f"pp{pp}.vp{vp}=[{ids[0]}..{ids[-1]}]" if ids else f"pp{pp}.vp{vp}=[]"
                for (pp, vp), ids in sorted(self.stage_layers.items())
            ),
            f"  交接点（按全局层序相邻的 stage 对）= {self.boundaries}；需要状态交接 = "
            f"{self.handoff_required}（交接实现 = {self.handoff_enabled}）",
            "  跨 stage 的 block = "
            + (
                ", ".join(
                    f"block{b.index}(write={b.write_layer}, readers={list(b.read_layers)}) 跨 "
                    + "/".join(f"pp{p}.vp{v}" for p, v in stages)
                    for b, stages in self.crossing_blocks
                )
                or "无（每个 block 完整落在同一 stage）"
            ),
        ]
        for n in self.notes:
            lines.append(f"  [note] {n}")
        for i, p in enumerate(self.problems, 1):
            lines.append(f"  [problem {i}] {p}")
        for i, s in enumerate(self.suggestions, 1):
            lines.append(f"  [suggest {i}] {s}")
        return "\n".join(lines)


@dataclass
class AttnResStagePlan:
    num_layers: int
    n_hash_layers: int
    block_size: int
    pp_size: int
    vpp_size: int
    stage_layers: Dict[Tuple[int, int], Tuple[int, ...]]
    source: str = "layout"
    local_stage: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.stage_layers = {
            k: tuple(sorted(v)) for k, v in dict(self.stage_layers).items()
        }
        self._layer_to_stage = {
            layer: key for key, ids in self.stage_layers.items() for layer in ids
        }
        self.blocks = blocks_of(self.num_layers, self.n_hash_layers, self.block_size)

    @classmethod
    def from_stage_layers(
        cls,
        stage_layers: Dict[Tuple[int, int], Sequence[int]],
        *,
        num_layers: int,
        n_hash_layers: int,
        block_size: int,
        local_stage: Tuple[int, int] = (0, 0),
        source: str = "manual",
    ) -> "AttnResStagePlan":
        pp_size = 1 + max((k[0] for k in stage_layers), default=0)
        vpp_size = 1 + max((k[1] for k in stage_layers), default=0)
        return cls(
            num_layers=num_layers,
            n_hash_layers=n_hash_layers,
            block_size=block_size,
            pp_size=pp_size,
            vpp_size=vpp_size,
            stage_layers={k: tuple(sorted(v)) for k, v in stage_layers.items()},
            source=source,
            local_stage=local_stage,
        )

    @classmethod
    def from_config(
        cls, config, *, pp_rank: Optional[int] = None, vp_stage: Optional[int] = None
    ) -> "AttnResStagePlan":
        pp_size = int(getattr(config, "pipeline_model_parallel_size", 1) or 1)
        vpp_size = int(
            getattr(config, "virtual_pipeline_model_parallel_size", None) or 1
        )
        num_layers = getattr(config, "num_layers", None)
        if num_layers is None:
            raise AttnResPPLayoutError(
                "config 缺少 num_layers，无法校验 AttnRes 的 PP 布局（请传完整的 mcore "
                "TransformerConfig，或直接传 plan=AttnResStagePlan.from_stage_layers(...)）"
            )
        num_layers = int(num_layers)
        n_hash = int(getattr(config, "moe_n_hash_layers", 0) or 0)
        block_size = int(getattr(config, "attn_res_block_size", 4) or 4)
        layout = getattr(config, "pipeline_model_parallel_layout", None)
        stage_layers: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        if pp_size <= 1 and vpp_size <= 1:
            stage_layers[(0, 0)] = tuple(range(num_layers))
            source = "single-stage"
        elif layout is not None:
            stage_layers = _stage_layers_from_layout(layout, pp_size, vpp_size)
            source = "layout"
        else:
            from megatron_ext.core.models.shensi.shensi_layer_specs import local_layer_ids

            source = "offsets"
            for vp in range(vpp_size):
                for pp in range(pp_size):
                    stage_layers[(pp, vp)] = tuple(
                        local_layer_ids(config, vp_stage=vp, pp_rank=pp)
                    )
        if pp_rank is None:
            try:
                from megatron.core import parallel_state as ps

                if ps.is_initialized():
                    pp_rank = int(ps.get_pipeline_model_parallel_rank())
                    vp_stage = int(ps.get_virtual_pipeline_model_parallel_rank() or 0)
                else:
                    pp_rank, vp_stage = 0, 0
            except Exception:  # noqa: BLE001
                pp_rank, vp_stage = 0, 0
        return cls(
            num_layers=num_layers,
            n_hash_layers=n_hash,
            block_size=block_size,
            pp_size=pp_size,
            vpp_size=vpp_size,
            stage_layers=stage_layers,
            source=source,
            local_stage=(int(pp_rank or 0), int(vp_stage or 0)),
        )

    @property
    def local_layers(self) -> Tuple[int, ...]:
        return self.stage_layers.get(self.local_stage, ())

    def stage_of(self, layer: int) -> Optional[Tuple[int, int]]:
        return self._layer_to_stage.get(int(layer))

    def crossing_blocks(self) -> List[Tuple[AttnResBlock, Tuple[Tuple[int, int], ...]]]:
        out = []
        for block in self.blocks:
            stages = []
            for layer in block.layers:
                st = self.stage_of(layer)
                if st is not None and st not in stages:
                    stages.append(st)
            if len(stages) > 1:
                out.append((block, tuple(stages)))
        return out

    def block_of(self, layer: int) -> int:
        layer = int(layer)
        for block in self.blocks:
            end = (
                block.read_layers[-1] + 1
                if block.read_layers
                else block.write_layer + 1
            )
            if block.write_layer <= layer < end:
                return block.index
        return -1

    def block_crosses(self, layer: int) -> bool:
        bi = self.block_of(layer)
        return any(b.index == bi for b, _ in self.crossing_blocks())

    def payload_numel(
        self,
        *,
        seq_len: int,
        micro_batch: int,
        hc_mult: int,
        hidden_size: int,
        num_blocks: Optional[int] = None,
    ) -> int:
        nb = len(self.blocks) if num_blocks is None else int(num_blocks)
        return HEADER_LEN + int(seq_len) * int(micro_batch) * int(hc_mult) * int(
            hidden_size
        ) * (1 + nb)

    def boundaries(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        out: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for layer in range(1, self.num_layers):
            prev, cur = self.stage_of(layer - 1), self.stage_of(layer)
            if (
                prev is not None
                and cur is not None
                and prev != cur
                and (prev, cur) not in out
            ):
                out.append((prev, cur))
        return out

    def needs_import(self, layer: int) -> bool:
        layer = int(layer)
        if layer <= 0 or layer >= self.num_layers:
            return False
        prev, cur = self.stage_of(layer - 1), self.stage_of(layer)
        return prev is not None and cur is not None and prev != cur

    def needs_export(self, layer: int) -> bool:
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers - 1:
            return False
        nxt, cur = self.stage_of(layer + 1), self.stage_of(layer)
        return nxt is not None and cur is not None and nxt != cur

    def local_import_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_import(i))

    def local_export_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_export(i))

    def report(self, *, handoff_enabled: Optional[bool] = None) -> LayoutReport:
        problems: List[str] = []
        suggestions: List[str] = []
        notes: List[str] = []
        crossing = self.crossing_blocks()
        boundaries = self.boundaries()
        all_ids = [i for ids in self.stage_layers.values() for i in ids]
        if sorted(all_ids) != list(range(self.num_layers)):
            problems.append(
                f"layout 覆盖的 decoder 层号必须正好是 0..{self.num_layers - 1}（不重不漏），"
                f"实际 {len(all_ids)} 个（去重后 {len(set(all_ids))}，最大 {max(all_ids, default=-1)}）"
            )
            suggestions.append(
                "检查 pipeline_model_parallel_layout 的 decoder 字符数量是否等于 num_layers"
            )
        embedding_stages = [
            k for k, ids in self.stage_layers.items() if ids and min(ids) == 0
        ]
        hash_layers = set(range(self.n_hash_layers))
        if embedding_stages:
            emb_stage = embedding_stages[0]
            emb_ids = set(self.stage_layers[emb_stage])
            missing = sorted(hash_layers - emb_ids)
            if missing:
                problems.append(
                    f"hash 层 {missing} 不在带 embedding 的 stage pp{emb_stage[0]}.vp{emb_stage[1]} 内"
                    "（mcore 要求所有 hash MoE 层与 embedding 同 stage）"
                )
                suggestions.append(
                    f"把 {len(missing)} 个 hash 层放进第一个 stage（推荐骨架见 "
                    "attn_res.recommended_layout_str）"
                )
        if crossing:
            if handoff_enabled is False:
                problems.append(
                    "以下 block 跨 PP stage，而 config.attn_res_pp_state_transfer=False（状态跨 stage "
                    "交接被关掉）："
                    + "; ".join(
                        f"block{b.index}(write={b.write_layer}, readers={list(b.read_layers)}) -> "
                        + "/".join(f"pp{p}.vp{v}" for p, v in st)
                        for b, st in crossing
                    )
                )
                suggestions.append(
                    "两种修法：(a) 打开 attn_res_pp_state_transfer（默认开）让 block 状态跨 stage 交接；"
                    "或 (b) 改用 block 对齐的 layout："
                    + recommended_layout_str(
                        self.num_layers,
                        self.n_hash_layers,
                        self.block_size,
                        self.pp_size,
                        self.vpp_size,
                    )
                )
            else:
                notes.append(
                    f"有 {len(crossing)} 个 block 跨 stage（状态交接开启，走 handoff 支持）："
                    + "; ".join(f"block{b.index}" for b, _ in crossing)
                )
            for b, st in crossing:
                vps = {v for _, v in st}
                if len(vps) > 1:
                    problems.append(
                        f"block{b.index} 跨**虚拟** stage（可能涉及 {sorted(vps)} 个 vp stage）："
                        "交织调度下 block 状态没有确定的交接点，当前 handoff 只覆盖 pp 相邻 stage"
                    )
                    suggestions.append(
                        "把 block 边界与虚拟 stage 边界对齐（推荐骨架见 "
                        "attn_res.recommended_layout_str），或先用 vpp=1 验证"
                    )
                elif self.vpp_size > 1:
                    notes.append(
                        f"block{b.index} 跨 pp stage 但仍在同一 vp stage 内：handoff 走 pp 邻接通信"
                    )
        handoff_required = self.pp_size > 1 or self.vpp_size > 1
        if handoff_required and handoff_enabled is False:
            problems.append(
                f"PP>1（pp={self.pp_size}, vpp={self.vpp_size}）必须做 block 状态跨 stage 交接："
                "ShensiModel 的末层收缩（output_attn_res + hc_head）要读**完整** residual 栈"
                "（含前面 stage 写入的槽位），而状态是每 stage 私有的内存张量"
            )
            suggestions.append("打开 config.attn_res_pp_state_transfer（默认 True）")
        ok = not problems
        return LayoutReport(
            ok=ok,
            pp_size=self.pp_size,
            vpp_size=self.vpp_size,
            num_layers=self.num_layers,
            stage_layers={k: v for k, v in sorted(self.stage_layers.items())},
            boundaries=boundaries,
            crossing_blocks=crossing,
            handoff_required=handoff_required,
            handoff_enabled=handoff_enabled,
            problems=problems,
            suggestions=suggestions,
            notes=notes,
        )


def _block_spans(
    num_layers: int, n_hash_layers: int, block_size: int
) -> List[Tuple[int, int]]:
    blocks = blocks_of(num_layers, n_hash_layers, block_size)
    spans = []
    for bi, b in enumerate(blocks):
        end = blocks[bi + 1].write_layer if bi + 1 < len(blocks) else num_layers
        spans.append((b.write_layer, end))
    return spans


def recommended_layout_str(
    num_layers: int,
    n_hash_layers: int,
    block_size: int,
    pp_size: int,
    vpp_size: int = 1,
) -> str:
    n_chunks = int(pp_size) * int(vpp_size)
    if n_chunks < 1:
        raise ValueError(f"pp_size/vpp_size 非法：{pp_size}/{vpp_size}")
    spans = _block_spans(int(num_layers), int(n_hash_layers), int(block_size))
    if n_chunks > len(spans):
        raise AttnResPPLayoutError(
            f"PP*VPP={n_chunks} 大于 AttnRes block 数 {len(spans)}：无法让每个 stage 拿到完整 "
            f"block（num_layers={num_layers}, n_hash_layers={n_hash_layers}, "
            f"block_size={block_size}）。请减小 pp/vpp，或增大 num_layers / 减小 block_size。"
        )
    target = int(num_layers) / n_chunks
    counts: List[int] = []
    bi = 0
    for stage_i in range(n_chunks):
        remaining_stages = n_chunks - stage_i
        if remaining_stages == 1:
            take = len(spans) - bi
        else:
            take = 0
            acc = 0
            while bi + take < len(spans) - (remaining_stages - 1):
                size = spans[bi + take][1] - spans[bi + take][0]
                if take > 0 and acc + size > target:
                    break
                acc += size
                take += 1
            take = max(take, 1)
        layers = 0
        for k in range(take):
            w, e = spans[bi + k]
            layers += e - w
        counts.append(layers)
        bi += take
    if sum(counts) != num_layers:
        raise AttnResPPLayoutError(
            f"内部错误：stage 层数合计 {sum(counts)} != num_layers {num_layers}"
        )
    chunks = ["t" * c for c in counts]
    chunks[0] = "E" + chunks[0]
    chunks[-1] = chunks[-1] + "L"
    return "|".join(chunks)


RECOMPUTE_METHOD_BLOCK = "block"
RECOMPUTE_METHOD_UNIFORM = "uniform"


@dataclass(frozen=True)
class AttnResRecomputePlan:
    granularity: Optional[str]
    method: Optional[str]
    recompute_num_layers: Optional[int]
    num_layers_per_stage: int
    pp_size: int
    vpp_size: int
    chunks: Tuple[Tuple[int, int], ...]
    safe: bool
    reason: str

    def describe(self) -> str:
        chunks = ", ".join(f"[{a},{b})" for a, b in self.chunks)
        return (
            f"granularity={self.granularity!r} method={self.method!r} "
            f"recompute_num_layers={self.recompute_num_layers} "
            f"stage 层数={self.num_layers_per_stage} pp={self.pp_size} vpp={self.vpp_size} "
            f"chunks=({chunks}) 安全={self.safe}（{self.reason}）"
        )


def _norm_granularity(v) -> Optional[str]:
    if v is None:
        return None
    return str(v)


def build_attn_res_recompute_plan(config) -> AttnResRecomputePlan:
    gran = _norm_granularity(getattr(config, "recompute_granularity", None))
    method = (
        _norm_granularity(getattr(config, "recompute_method", None))
        or RECOMPUTE_METHOD_UNIFORM
    )
    k = getattr(config, "recompute_num_layers", None)
    k = None if k is None else int(k)
    pp = int(getattr(config, "pipeline_model_parallel_size", 1) or 1)
    vpp = int(getattr(config, "virtual_pipeline_model_parallel_size", None) or 1)
    n_stage = int(getattr(config, "num_layers", 0) or 0)
    if pp > 1:
        n_stage = int(getattr(config, "num_layers", 0) or 0)
    chunks: list = []
    if gran == "full" and n_stage > 0 and k:
        if method == RECOMPUTE_METHOD_UNIFORM:
            i = 0
            while i < n_stage:
                chunks.append((i, min(i + k, n_stage)))
                i += k
        elif method == RECOMPUTE_METHOD_BLOCK:
            chunks = [(i, i + 1) for i in range(min(k, n_stage))]
        else:
            chunks = []
    safe, reason = True, "未启用 full recompute（不重放层 forward）"
    if gran == "full":
        if pp > 1 or vpp > 1:
            safe, reason = (
                False,
                "PP/VPP>1：首层重放会再次 import_layer（recv 不幂等），且跨 stage 状态不在本 stage 的 saved tensors 里",
            )
        elif not k:
            safe, reason = (
                False,
                "full recompute 未给 recompute_num_layers（无法确定分块网格）",
            )
        elif method == RECOMPUTE_METHOD_UNIFORM and k >= n_stage:
            safe, reason = (
                True,
                f"uniform 单 chunk 覆盖整段（{k}>={n_stage}）：重放从首层 begin() 重建状态",
            )
        elif method == RECOMPUTE_METHOD_BLOCK and k == 1:
            safe, reason = (
                True,
                "block 只 checkpoint 首层（k=1）：其余层激活全保留，首层重放 begin() 重建状态",
            )
        elif method == RECOMPUTE_METHOD_UNIFORM:
            safe, reason = (
                False,
                f"uniform 分块起点不在 stage 首层（k={k}<{n_stage}）：非首块重放读到块末态",
            )
        elif method == RECOMPUTE_METHOD_BLOCK:
            safe, reason = (
                False,
                f"block k={k}>1：第 1..{k - 1} 层的重放发生在首层之前（反向序），状态未重建",
            )
        else:
            safe, reason = (
                False,
                f"未知 recompute_method={method!r}（mcore 只认 uniform/block）",
            )
    return AttnResRecomputePlan(
        granularity=gran,
        method=method if gran == "full" else None,
        recompute_num_layers=k,
        num_layers_per_stage=n_stage,
        pp_size=pp,
        vpp_size=vpp,
        chunks=tuple(chunks),
        safe=safe,
        reason=reason,
    )


def check_attn_res_recompute_support(config) -> AttnResRecomputePlan:
    plan = build_attn_res_recompute_plan(config)
    if plan.safe:
        return plan
    raise NotImplementedError(
        "[shensi][attn_res] recompute(full) 与 AttnRes 的 block 状态（side-channel）不兼容："
        f"{plan.describe()}。\n"
        "上游依据：megatron/core/transformer/transformer_block.py:830 "
        "（`if self.config.recompute_granularity == 'full' and self.training:` -> "
        "`checkpointed_forward`）、metagatron/core/recompute.py:21（定义）、"
        ":142-171（chunk 循环：uniform 按 recompute_num_layers 切、block 只包前 k 层）、"
        ":114-136（重放用 tensor_parallel.checkpoint / te_checkpoint，只保存层输入）。\n"
        "修法（按上游那套重算块语义）：\n"
        "  ① PP=1 且让整段 stage 落进一个重算块 —— `recompute_method: uniform` + "
        "`recompute_num_layers >= 层数`（或 `block` + `recompute_num_layers: 1`）：重放从首层 "
        "`state.begin()` 重建状态；\n"
        "  ② 关掉 full recompute（`recompute_granularity: null`），长序列改用 'selective' "
        "并只 checkpoint 需要的模块（别加 'mhc'）；\n"
        "  ③ 要做 PP>1 的 full recompute，必须把跨 stage 的状态纳入 saved tensors 并让 "
        "import 幂等（= 改上游 recompute/gpt_model 契约，本家族不改上游文件）。"
    )


def check_attn_res_pp_support(
    config,
    *,
    plan: Optional[AttnResStagePlan] = None,
    handoff_enabled: Optional[bool] = None,
    raise_on_error: bool = True,
) -> LayoutReport:
    if raise_on_error:
        check_attn_res_recompute_support(config)
    if handoff_enabled is None:
        handoff_enabled = bool(getattr(config, "attn_res_pp_state_transfer", True))
    if plan is None:
        try:
            plan = AttnResStagePlan.from_config(config)
        except AttnResPPLayoutError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AttnResPPLayoutError(
                "无法从 config 推导 AttnRes 的 PP 布局："
                f"{type(e).__name__}: {e}。PP>1 时需要完整的 mcore TransformerConfig"
                "（num_layers / pipeline_model_parallel_size / pipeline_model_parallel_layout）"
                "或直接传 plan=AttnResStagePlan.from_stage_layers(...)"
            ) from e
    report = plan.report(handoff_enabled=handoff_enabled)
    if report.ok or not raise_on_error:
        return report
    header = (
        f"AttnRes 的 PP 布局校验失败（pp={report.pp_size}, vpp={report.vpp_size}）。"
        "可读报告：\n" + str(report)
    )
    transfer_disabled = any("attn_res_pp_state_transfer" in p for p in report.problems)
    if transfer_disabled:
        raise NotImplementedError(
            header
            + "\n\n修法：把 config.attn_res_pp_state_transfer 置 True（默认）以启用 block 状态跨 "
            "stage 交接；或用上面 [suggest] 里的 block 对齐 layout。"
        )
    raise AttnResPPLayoutError(header)


_GROUP_CACHE: Dict[tuple, object] = {}
_flush_registry: List["ShensiAttnResPPHandoff"] = []
_PENDING_SENDS: List[tuple] = []
_PENDING_BYTES: List[int] = [0]
_RELEASED_BYTES: List[int] = [0]
_LIVE_PAYLOADS: "weakref.WeakSet" = weakref.WeakSet()


def _payload_nbytes(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else int(t.numel()) * int(t.element_size())


def pending_transfer_bytes() -> int:
    return int(_PENDING_BYTES[0])


def released_transfer_bytes() -> int:
    return int(_RELEASED_BYTES[0])


def live_payload_stats() -> Dict[str, int]:
    live = [t for t in _LIVE_PAYLOADS]
    return {
        "n_live": len(live),
        "bytes_live": sum(_payload_nbytes(t) for t in live),
        "n_registered": len(_LIVE_PAYLOADS),
    }


def _track_payloads() -> bool:
    return str(
        os.environ.get("SHENSI_ATTN_RES_PP_TRACK_PAYLOAD", "")
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def payload_device(hint: Optional[torch.Tensor] = None) -> torch.device:
    if hint is not None:
        return hint.device
    try:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - CUDA 驱动/上下文异常时退回 cpu
        pass
    return torch.device("cpu")


def _release_send(entry) -> None:
    work, buf, nbytes = entry
    work.wait()
    _PENDING_BYTES[0] -= int(nbytes)
    _RELEASED_BYTES[0] += int(nbytes)
    del buf


def _flush_all() -> None:
    while _PENDING_SENDS:
        _release_send(_PENDING_SENDS.pop(0))


class _ShensiAttnResPPBridge(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        stage_output: torch.Tensor,
        payload: torch.Tensor,
        dst,
        group,
        tag,
        use_isend: bool,
    ):
        ctx.dst, ctx.group, ctx.tag = dst, group, tag
        ctx.remote_numel = int(payload.numel())
        ctx.payload_device = payload.device
        ctx.buf = None
        ctx.work = None
        ctx.nbytes = 0
        buf = payload.detach().to(torch.float32).contiguous()
        if dst is not None:
            if use_isend:
                work = torch.distributed.isend(buf, dst=dst, group=group, tag=tag)
                ctx.buf, ctx.work = buf, work
                ctx.nbytes = _payload_nbytes(buf)
                _PENDING_BYTES[0] += ctx.nbytes
            else:
                torch.distributed.send(buf, dst=dst, group=group, tag=tag)
        return stage_output

    @staticmethod
    def backward(ctx, grad_stage_output: torch.Tensor):
        if ctx.dst is None:
            return grad_stage_output, None, None, None, None, None
        if ctx.work is not None:
            ctx.work.wait()
            _PENDING_BYTES[0] -= int(ctx.nbytes)
            _RELEASED_BYTES[0] += int(ctx.nbytes)
            ctx.work, ctx.buf, ctx.nbytes = None, None, 0
        grad = torch.empty(
            ctx.remote_numel, dtype=torch.float32, device=ctx.payload_device
        )
        torch.distributed.recv(grad, src=ctx.dst, group=ctx.group, tag=ctx.tag + 1)
        return grad_stage_output, grad, None, None, None, None


def _recv_payload(
    numel: int,
    src: Optional[int],
    group,
    tag: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    buf = torch.empty(
        int(numel),
        dtype=torch.float32,
        device=payload_device() if device is None else device,
    )
    torch.distributed.recv(buf, src=src, group=group, tag=tag)
    buf.requires_grad_(True)

    def _send_back(grad: torch.Tensor):
        g = grad.detach().to(torch.float32).contiguous()
        work = torch.distributed.isend(g, dst=src, group=group, tag=tag + 1)
        nbytes = _payload_nbytes(g)
        _PENDING_SENDS.append((work, g, nbytes))
        _PENDING_BYTES[0] += nbytes
        return None

    buf.register_hook(_send_back)
    return buf


class ShensiAttnResPPHandoff:
    def __init__(
        self,
        plan: AttnResStagePlan,
        *,
        group=None,
        local_stage: Optional[Tuple[int, int]] = None,
        use_isend: Optional[bool] = None,
        tag: int = 7900,
        logger=None,
        auto_group: bool = True,
    ) -> None:
        self.plan = plan
        self.group = group
        self.local_stage = tuple(local_stage or plan.local_stage)
        if use_isend is None:
            use_isend = not (
                str(os.environ.get("SHENSI_ATTN_RES_PP_NO_ISEND", "")).strip().lower()
                in ("1", "true", "yes", "on")
            )
        self.use_isend = bool(use_isend)
        self.tag = int(tag)
        self.logger = logger
        self.auto_group = bool(auto_group)
        self.pp_group_ranks: Dict[int, int] = {}
        self.records: List[dict] = []
        self.n_import = 0
        self.n_export = 0
        self.payload_numels: List[int] = []
        self._exported_payloads: List[torch.Tensor] = []
        self.n_graph_refs_released = 0
        _flush_registry.append(self)
        self.bind_group()

    def resolve_pp_group_ranks(self) -> Optional[Tuple[int, ...]]:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return None
        try:
            from megatron.core import parallel_state as ps

            pp_group = ps.get_pipeline_model_parallel_group()
            ranks = tuple(sorted(torch.distributed.get_process_group_ranks(pp_group)))
            if not ranks:
                raise RuntimeError("mcore 返回的 pp 组为空")
            return ranks
        except Exception as exc:  # noqa: BLE001 - 一律上抛（理由见 docstring）
            raise RuntimeError(
                "[attn_res_pp] 取不到 pipeline model parallel 组（"
                f"{type(exc).__name__}: {exc}）。AttnRes 的跨 stage 交接**必须**在真的 pp 组上"
                "进行：组内局部 rank 才等于 pp 相邻，而 world 视角的 rank±1 通常是 TP/DP 搭档"
                "（mcore 的 rank 序是 tp × pp × dp）—— 换成 world 会静默把状态发给错误的进程"
                "（错梯度）。因此这里**不再静默退回 world**。修法：确认 handoff 在 "
                "ps.initialize_model_parallel(...) 之后构造（构造期 bind_group 会缓存组），"
                "或显式传 group=.../auto_group=False（离线单测路径）。"
            ) from exc

    def bind_group(self) -> None:
        if self.group is not None or not self.auto_group:
            return
        if self.plan.pp_size <= 1 and self.plan.vpp_size <= 1:
            return
        ranks = self.resolve_pp_group_ranks()
        if ranks is None:
            return
        if ranks not in _GROUP_CACHE:
            _GROUP_CACHE[ranks] = torch.distributed.new_group(ranks=list(ranks))
        self.group = _GROUP_CACHE[ranks]
        self._record(
            "bind_group", -1, group_ranks=list(ranks), pp_size=self.plan.pp_size
        )

    def resolve_group(self):
        if (
            self.group is None
            and self.auto_group
            and (self.plan.pp_size > 1 or self.plan.vpp_size > 1)
        ):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                raise RuntimeError(
                    "[attn_res_pp] 交接通信组未在 bind/构造阶段建好（self.group=None）。"
                    "不要在层 forward 里补建：new_group 是集合操作，PP>1 时会与 mcore 的 "
                    "recv_forward 成环死锁（见 ShensiAttnResPPHandoff.bind_group 的实测记录）；"
                    "请让 handoff 在 parallel_state.initialize_model_parallel 之后构造，"
                    "或显式调用 handoff.bind_group()（可安全重入），或直接传 group=..."
                )
        return self.group

    def _neighbor_global_rank(self, delta: int) -> Optional[int]:
        tgt = self.local_stage[0] + int(delta)
        if tgt < 0 or tgt >= self.plan.pp_size:
            return None
        group = self.resolve_group()
        if group is None:
            return None
        local_idx = self.pp_group_ranks.get(tgt, tgt)
        return int(torch.distributed.get_global_rank(group, local_idx))

    def _dst_rank(self) -> Optional[int]:
        return self._neighbor_global_rank(+1)

    def _src_rank(self) -> Optional[int]:
        return self._neighbor_global_rank(-1)

    def _record(
        self, kind: str, layer: int, payload: Optional[torch.Tensor] = None, **extra
    ) -> None:
        import hashlib

        rec = {
            "kind": kind,
            "stage": f"pp{self.local_stage[0]}.vp{self.local_stage[1]}",
            "global_layer": int(layer),
            "payload_numel": int(payload.numel()) if payload is not None else 0,
        }
        if payload is not None:
            rec["payload_sha256"] = hashlib.sha256(
                payload.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
            ).hexdigest()[:16]
        rec.update(extra)
        self.records.append(rec)
        if self.logger is not None:
            self.logger(
                f"[attn_res_pp][{kind}] stage={rec['stage']} layer={rec['global_layer']} "
                f"numel={rec['payload_numel']}"
                + (
                    f" sha256={rec.get('payload_sha256')}"
                    if payload is not None
                    else ""
                )
                + (
                    " " + " ".join(f"{k}={v}" for k, v in extra.items())
                    if extra
                    else ""
                )
            )

    def export_layer(
        self,
        state,
        layer_idx: int,
        *,
        stage_output: Optional[torch.Tensor] = None,
        send: bool = True,
    ) -> Optional[torch.Tensor]:
        self.flush()
        if not self.plan.needs_export(layer_idx):
            return stage_output
        payload = state.state_for_pp()
        dst = self._dst_rank()
        out = payload if stage_output is None else stage_output
        if send and dst is not None:
            group = self.resolve_group()
            if stage_output is None:
                raise ValueError(
                    "export_layer 需要 stage_output（本层输出）才能把状态通道接到反向路径上；"
                    "只导出 payload 时 autograd 收不到它的梯度（见 _ShensiAttnResPPBridge）"
                )
            out = _ShensiAttnResPPBridge.apply(
                stage_output, payload, dst, group, self.tag, self.use_isend
            )
        self.n_export += 1
        self.payload_numels.append(int(payload.numel()))
        if _track_payloads():
            self._exported_payloads.append(payload)
        self._record(
            "export",
            layer_idx,
            payload,
            dst=dst,
            num_blocks=int(state.num_blocks),
            num_written=int(getattr(state, "num_written", -1)),
            block_index=self.plan.block_of(layer_idx),
            crossing=bool(self.plan.block_crosses(layer_idx)),
            payload_device=str(payload.device),
        )
        return out

    def import_layer(
        self,
        state,
        layer_idx: int,
        *,
        hidden_flat: Optional[torch.Tensor] = None,
        numel: Optional[int] = None,
        recv: bool = True,
    ) -> Optional[dict]:
        self.flush()
        if not self.plan.needs_import(layer_idx):
            return None
        if numel is None:
            if hidden_flat is None:
                raise ValueError(
                    "import_layer 需要 hidden_flat（用来推 payload 大小）或显式 numel"
                )
            numel = HEADER_LEN + int(hidden_flat.numel()) * (1 + int(state.num_blocks))
        src = self._src_rank()
        payload = None
        if recv:
            group = self.resolve_group()
            payload = _recv_payload(
                int(numel), src, group, self.tag, device=payload_device(hidden_flat)
            )
        if payload is None:
            raise RuntimeError(
                f"[attn_res_pp] stage pp{self.local_stage[0]}.vp{self.local_stage[1]} 的第 "
                f"{layer_idx} 层需要导入 AttnRes block 状态，但没有收到 payload（src={src}）"
            )
        info = state.load_from_pp(
            payload,
            dtype=None if hidden_flat is None else hidden_flat.dtype,
            expected_hidden_flat_numel=None
            if hidden_flat is None
            else int(hidden_flat.numel()),
        )
        self.n_import += 1
        self.payload_numels.append(int(payload.numel()))
        self._record(
            "import",
            layer_idx,
            payload,
            src=src,
            num_blocks=int(info["num_blocks"]),
            num_written=int(info["num_written"]),
            block_index=self.plan.block_of(layer_idx),
            crossing=bool(self.plan.block_crosses(layer_idx)),
            payload_device=str(payload.device),
        )
        return info

    def flush(self) -> None:
        _flush_all()

    def clear_graph_refs(self) -> None:
        n = len(self._exported_payloads)
        self._exported_payloads.clear()
        self.n_graph_refs_released += n
        return None

    def live_graph_refs(self) -> int:
        return len(self._exported_payloads)

    def pending_transfer_bytes(self) -> int:
        return pending_transfer_bytes()

    def payload_diagnostics(self) -> Dict[str, int]:
        stats = live_payload_stats()
        return {
            "container_refs": len(self._exported_payloads),
            "n_live_payloads": int(stats["n_live"]),
            "bytes_live_payloads": int(stats["bytes_live"]),
            "pending_bytes": pending_transfer_bytes(),
            "released_bytes": released_transfer_bytes(),
            "n_graph_refs_released": int(self.n_graph_refs_released),
        }

    def summary(self) -> dict:
        return {
            "stage": f"pp{self.local_stage[0]}.vp{self.local_stage[1]}",
            "pp_size": self.plan.pp_size,
            "vpp_size": self.plan.vpp_size,
            "local_layers": list(self.plan.local_layers),
            "import_layers": list(self.plan.local_import_layers()),
            "export_layers": list(self.plan.local_export_layers()),
            "n_import": self.n_import,
            "n_export": self.n_export,
            "payload_numels": self.payload_numels,
            "crossing_blocks": [
                {
                    "index": b.index,
                    "write_layer": b.write_layer,
                    "read_layers": list(b.read_layers),
                }
                for b, _ in self.plan.crossing_blocks()
            ],
            "records": self.records,
            "payload_diagnostics": self.payload_diagnostics(),
        }
