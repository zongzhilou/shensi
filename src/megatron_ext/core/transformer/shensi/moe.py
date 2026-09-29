# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_layer import LayerNormBuilder


@dataclass
class ShensiMoESubmodules(MoESubmodules):
    routed_expert_norm: Optional[LayerNormBuilder] = None


class ShensiMoELayer(MoELayer):
    def __init__(
        self,
        config,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection=None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        latent = config.moe_latent_size
        if not latent:
            raise ValueError(
                "ShensiMoELayer 需要 config.moe_latent_size = routed_expert_hidden_size "
                "（低秩路由专家的瓶颈维）；当前为 None。"
            )
        norm_spec = getattr(submodules, "routed_expert_norm", None)
        if norm_spec is None:
            raise ValueError(
                "ShensiMoESubmodules.routed_expert_norm 未设置：Shensi 的路由专家在 rank "
                "空间输出后必须先过 routed_expert_norm 再做 up 投影。"
            )
        self.routed_expert_norm = build_module(
            norm_spec,
            config=config,
            hidden_size=int(latent),
            eps=config.layernorm_epsilon,
        )

    def combine(self, output: torch.Tensor) -> torch.Tensor:
        return self.routed_expert_norm(super().combine(output))

    def erc_weights(self):
        router_weight = self.router.weight
        down_proj_weight = self.fc1_latent_proj.weight
        gate_up = torch.stack(
            [
                getattr(self.experts.linear_fc1, f"weight{i}")
                for i in range(self.num_local_experts)
            ],
            dim=0,
        )
        return router_weight, down_proj_weight, gate_up


def erc_gather_expert_weights(
    mlp,
    ep_group=None,
    *,
    grad_reduce: Optional[str] = None,
    verify_router_replica: Optional[bool] = None,
    verbose: bool = False,
):
    ep_size, ep_rank, ep_group = _resolve_parallel_group(
        "expert_model_parallel", ep_group
    )
    etp_size, etp_rank, etp_group = _resolve_parallel_group(
        "expert_tensor_parallel", None
    )
    if grad_reduce is None:
        grad_reduce = (
            os.environ.get("SHENSI_ERC_EP_GRAD_REDUCE", "slice").strip().lower()
        )
    if verify_router_replica is None:
        verify_router_replica = _env_flag("SHENSI_ERC_VERIFY_ROUTER_REPLICA", False)
    if _erc_legacy_local():
        router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
        n_local = int(getattr(mlp, "num_local_experts", 0) or 0)
        if ep_size > 1 and n_local > 0:
            router_weight = router_weight[ep_rank * n_local : (ep_rank + 1) * n_local]
        if verbose:
            print_rank0(
                "[shensi][erc] erc_gather_expert_weights：SHENSI_ERC_EP_ALLOW_LOCAL=1 -> "
                f"退回**逐卡本地**视图（专家 {n_local} 个，子块口径；对照/诊断用）"
            )
        return router_weight, down_proj_weight, gate_up_local
    router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
    if ep_size <= 1 and etp_size <= 1:
        return router_weight, down_proj_weight, gate_up_local
    if not hasattr(mlp, "fc1_latent_proj"):
        raise NotImplementedError(
            "erc_gather_expert_weights：ERC 需要低秩路由专家的 down/up 投影"
            "（mcore 的 moe_latent_size 路径 -> fc1_latent_proj）；当前 MoE 没有该模块，"
            "无法构造全局视图。请用 moe_latent_size=routed_expert_hidden_size 的配置。"
        )
    if verify_router_replica and ep_size > 1:
        _assert_replica(router_weight, ep_group, "router.weight")
        _assert_replica(down_proj_weight, ep_group, "fc1_latent_proj.weight")
    if ep_size > 1 and ep_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_model_parallel_size={ep_size} > 1，但拿不到 EP 组"
            "（parallel_state 未初始化/组为空）：无法构造全局视图，也**不会**静默退回本地口径"
            "（那会静默算出一个语义不同的损失）。请确认在 initialize_model_parallel 之后调用，"
            "或用 SHENSI_ERC_EP_ALLOW_LOCAL=1 显式要求本地口径。"
        )
    if etp_size > 1 and etp_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_tensor_parallel_size={etp_size} > 1，但拿不到 ETP 组："
            "ERC 的耦合矩阵要对**完整**中间维取范数，不能静默按切片算。"
        )
    gate_up = gate_up_local
    if ep_size > 1:
        gate_up = _ShensiEPAllGather.apply(gate_up, ep_group, 0, grad_reduce)
    if etp_size > 1:
        gate_up = _ShensiEPAllGather.apply(gate_up, etp_group, 1, grad_reduce)
    if verbose:
        print_rank0(
            f"[shensi][erc] 全局视图：ep={ep_size}（rank {ep_rank}，本地专家 "
            f"{getattr(mlp, 'num_local_experts', '?')} 个）etp={etp_size}"
            f"（rank {etp_rank}）gate_up_proj={tuple(gate_up.shape)} "
            f"grad_reduce={grad_reduce}"
        )
    return router_weight, down_proj_weight, gate_up


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _erc_legacy_local() -> bool:
    return _env_flag("SHENSI_ERC_EP_ALLOW_LOCAL", False)


def print_rank0(msg: str) -> None:
    try:
        from megatron.training import print_rank_0

        print_rank_0(msg)
    except Exception:  # noqa: BLE001
        rank = 0
        try:
            import torch.distributed as _dist

            if _dist.is_available() and _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:  # noqa: BLE001
            rank = 0
        if rank == 0:
            print(msg, flush=True)


def _resolve_parallel_group(kind: str, group):
    size, rank = 1, 0
    try:
        from megatron.core import parallel_state as _ps

        if not _ps.is_initialized():
            return size, rank, group
        if kind == "expert_model_parallel":
            size = int(_ps.get_expert_model_parallel_world_size())
            rank = int(_ps.get_expert_model_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_model_parallel_group()
        elif kind == "expert_tensor_parallel":
            size = int(_ps.get_expert_tensor_parallel_world_size())
            rank = int(_ps.get_expert_tensor_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_tensor_parallel_group()
    except Exception:  # noqa: BLE001
        return 1, 0, group
    return size, rank, group


def _all_gather_along_dim(t: torch.Tensor, group, dim: int) -> torch.Tensor:
    import torch.distributed as dist

    world = group.size() if hasattr(group, "size") else dist.get_world_size(group)
    if world <= 1:
        return t
    if dim == 0 and t.is_cuda:
        try:
            from megatron.core.tensor_parallel.mappings import _gather_along_first_dim

            return _gather_along_first_dim(t, group)
        except Exception:  # noqa: BLE001
            pass
    if dim != 0:
        order = (dim,) + tuple(i for i in range(t.dim()) if i != dim)
        inv = tuple(sorted(range(t.dim()), key=lambda i: order[i]))
        t_perm = t.permute(order).contiguous()
    else:
        order, inv, t_perm = None, None, t.contiguous()
    out_shape = list(t_perm.shape)
    out_shape[0] = out_shape[0] * world
    out = torch.empty(out_shape, dtype=t_perm.dtype, device=t_perm.device)
    torch.distributed.all_gather_into_tensor(out, t_perm, group=group)
    return out.permute(inv) if order is not None else out


def _slice_along_dim(
    t: torch.Tensor, group, dim: int, rank: Optional[int] = None
) -> torch.Tensor:
    import torch.distributed as dist

    world = group.size() if hasattr(group, "size") else dist.get_world_size(group)
    if world <= 1:
        return t
    rank = dist.get_rank(group) if rank is None else rank
    n = t.shape[dim] // world
    return t.narrow(dim, rank * n, n).contiguous()


class _ShensiEPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, dim: int, grad_reduce: str):
        if grad_reduce not in ("slice", "reduce_scatter_mean"):
            raise NotImplementedError(
                f"未知的 grad_reduce={grad_reduce!r}（可选 slice / reduce_scatter_mean）"
            )
        ctx.group, ctx.dim, ctx.grad_reduce = group, dim, grad_reduce
        return _all_gather_along_dim(tensor, group, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as dist

        if ctx.grad_reduce == "slice":
            return _slice_along_dim(grad_output, ctx.group, ctx.dim), None, None, None
        world = (
            ctx.group.size()
            if hasattr(ctx.group, "size")
            else dist.get_world_size(ctx.group)
        )
        dim = ctx.dim
        order = None
        g = grad_output
        if dim != 0:
            order = (dim,) + tuple(i for i in range(g.dim()) if i != dim)
            inv = tuple(sorted(range(g.dim()), key=lambda i: order[i]))
            g = g.permute(order).contiguous()
        out_shape = list(g.shape)
        out_shape[0] = out_shape[0] // world
        out = torch.empty(out_shape, dtype=g.dtype, device=g.device)
        try:
            dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)
        except (RuntimeError, ValueError) as exc:  # noqa: BLE001
            print_rank0(
                f"[shensi][erc] [warn] 本后端不支持 reduce_scatter_tensor（{exc!r}），"
                "退回 all_reduce+narrow 的等价实现（数值口径不变，通信量更大）"
            )
            full = g.contiguous().clone()
            dist.all_reduce(full, group=ctx.group)
            out = full.narrow(0, 0, out_shape[0]).clone()
        out = out / world
        return (out.permute(inv) if order is not None else out), None, None, None


def _assert_replica(t: torch.Tensor, group, name: str) -> None:
    import torch.distributed as dist

    world = group.size() if hasattr(group, "size") else dist.get_world_size(group)
    if world <= 1:
        return
    outs = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(outs, t.contiguous(), group=group)
    ref = outs[0]
    for i, o in enumerate(outs[1:], start=1):
        if not torch.equal(o, ref):
            raise RuntimeError(
                f"[shensi][erc] EP 组的 {name} 在 rank0 与 rank{i} 上**不一致**"
                f"（max|Δ|={float((o.float() - ref.float()).abs().max()):.3e}）。ERC 的全局视图"
                "要求这些量是复本（mcore 的 router / latent 投影按 'duplicated' 构造）；"
                "不一致通常来自各 rank 的初始化 RNG 不同步（EP 切分导致建模时的 RNG 消耗不同）。"
                "修法：从检查点加载权重（加载后即为复本），或用 "
                "--shensi-erc-ep-legacy-local 退回逐卡本地口径。"
            )


class ShensiHashMLP(MLP):
    def __init__(
        self,
        config,
        submodules,
        ffn_hidden_size: Optional[int] = None,
        name: str | None = None,
        **kwargs,
    ):
        vocab_size = getattr(config, "actual_vocab_size", None) or config.vocab_size
        hidden_intermediate = int(ffn_hidden_size or config.routed_expert_hidden_size)
        inner_config = copy.copy(config)
        inner_config.ffn_hidden_size = hidden_intermediate
        super().__init__(
            config=inner_config,
            submodules=submodules,
            ffn_hidden_size=hidden_intermediate,
            name=name,
        )
        self.deepemb = nn.Embedding(int(vocab_size), int(config.hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ):
        if input_ids is None:
            raise ValueError(
                "ShensiHashMLP 需要 input_ids 做 deepemb 门控；请确认 "
                "--moe-n-hash-layers（= Shensi 的 hash 层数）已设置，"
                "这样 GPTModel 才会把 input_ids 透传到 decoder/layer。"
            )
        output_with_bias = super().forward(hidden_states)
        output, bias = output_with_bias[0], output_with_bias[1]
        ids = input_ids.transpose(0, 1) if input_ids.dim() == 2 else input_ids
        return output * self.deepemb(ids), bias

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.deepemb.weight, mean=0.0, std=std)


def _layer_is_hash(layer_idx: int, n_hash_layers: int) -> bool:
    return layer_idx < n_hash_layers


def tie_moe_groups(
    block, n_hash_layers: int, block_size: int, num_layers: Optional[int] = None
) -> Dict[int, ShensiMoELayer]:
    from .attn_res import attn_res_block_layer_types, num_attn_res_blocks

    layers = block.layers
    global_idx = [
        int(getattr(layer, "layer_number", i + 1)) - 1 for i, layer in enumerate(layers)
    ]
    n_total = (
        int(num_layers) if num_layers else (max(global_idx) + 1 if global_idx else 0)
    )
    types = attn_res_block_layer_types(n_total, n_hash_layers, block_size)
    n_blocks_total = num_attn_res_blocks(n_total, n_hash_layers, block_size)
    write_layers = [i for i, t in enumerate(types) if t == "block_write_layer"]
    owners: Dict[int, ShensiMoELayer] = {}
    groups: Dict[int, List[int]] = {}
    skipped_hash: List[int] = []
    cross_stage: List[int] = []
    for idx, layer in enumerate(layers):
        g = global_idx[idx]
        if _layer_is_hash(g, n_hash_layers):
            skipped_hash.append(g)
            continue
        block_id = max((k for k, w in enumerate(write_layers) if w <= g), default=None)
        if block_id is None:
            raise RuntimeError(
                f"[shensi][moe] 全局层 {g} 落在任何 AttnRes block 之前（write 层={write_layers}）；"
                "请检查 attn_res_block_size / moe_n_hash_layers / layer_number 是否一致。"
            )
        mlp = layer.mlp
        if block_id not in owners:
            owners[block_id] = mlp
            groups[block_id] = [g]
            if write_layers[block_id] not in global_idx:
                cross_stage.append(block_id)
        else:
            owner = owners[block_id]
            groups[block_id].append(g)
            if mlp is not owner:
                mlp.router = owner.router
                mlp.experts = owner.experts
    for block_id, owner in owners.items():
        owner.sharing_layers = list(groups[block_id])
        owner.shensi_tie_cross_stage = block_id in cross_stage
    if cross_stage:
        print_rank0(
            f"[shensi][moe][warn] 有 {len(cross_stage)} 个 AttnRes block 被 PP 边界切开"
            f"（block_id={cross_stage}）：跨 stage 的层之间**无法共享** gate/experts"
            "（不同进程没有共同父模块；mcore 的 p2p 不传模块）。这与 HF 的"
            "全模型 tie_moe_groups **不等价**，已显式登记在 owner.shensi_tie_cross_stage 上。"
        )
    if skipped_hash:
        print_rank0(
            f"[shensi][moe] tie_moe_groups：本 stage 跳过 hash 层（全局层号 {skipped_hash}），"
            f"共 {len(owners)} 个共享组，组内全局层号="
            f"{ {k: v for k, v in groups.items()} }"
            f"（全局层数={n_total}，block 总数={n_blocks_total}）"
        )
    return owners
