"""ERC：路由专家的耦合正则（DeepSeek-V4 的 expert routing coupling）。

口径与 HF `ShensiForCausalLM.forward` 一致：每个 AttnRes block 的 MoE 组各算一次，
再对**全模型**的组数取平均；EP>1 时先 all-gather 出全局专家权重（Bridge 的
`erc_gather_expert_weights`），各 rank 噪声同步，保证 ERC 标量一致。
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from megatron.bridge.models.shensi.modeling_shensi import attn_res_block_layer_types
from megatron.bridge.models.shensi.modeling_shensi import erc_gather_expert_weights
from megatron.core.utils import unwrap_model
from megatron.training import get_args

_ERC_GROUP_AUDIT_LOGGED: set[str] = set()
_ERC_DISABLED_LOGGED = False


def erc_loss_func(
    router_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    alpha: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """HF 的口径：对路由权重加扰动后，惩罚专家间（除对角外）的响应耦合。"""
    R = router_weight
    norm_R = torch.norm(R, dim=1)
    distances = torch.cdist(R, R, p=2)
    distances = distances.masked_fill(
        torch.eye(R.size(0), dtype=torch.bool, device=distances.device), float("inf")
    )
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_R
    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    if noise is None:
        noise = torch.rand_like(R)
    R_tilde = (low + noise * (high - low)) * R
    proxy = F.linear(R_tilde, down_proj_weight)
    M = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)
    row_diff = M - alpha * torch.diag(M).unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)
    col_diff = M - alpha * torch.diag(M).unsqueeze(0)
    col_diff_clamped = torch.clamp(col_diff, min=0.0)
    mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask
    return total_diff.mean()


def _print_erc(msg: str) -> None:
    try:
        from megatron.training import print_rank_last

        print_rank_last(msg)
    except Exception:
        print(msg, flush=True)


def _resolve_ep():
    try:
        from megatron.core import parallel_state as _ps

        if not _ps.is_initialized():
            return 1, 0, None
        size = int(_ps.get_expert_model_parallel_world_size())
        if size <= 1:
            return 1, 0, None
        return (
            size,
            int(_ps.get_expert_model_parallel_rank()),
            _ps.get_expert_model_parallel_group(),
        )
    except Exception:
        return 1, 0, None


def _erc_noise_for(router_weight: torch.Tensor) -> torch.Tensor | None:
    ep_size, _, ep_group = _resolve_ep()
    if ep_size <= 1 or ep_group is None:
        return None
    if str(os.environ.get("SHENSI_ERC_EP_SYNC_NOISE", "1")).strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    import torch.distributed as dist

    src = dist.get_global_rank(ep_group, 0)
    if dist.get_rank() == src:
        noise = torch.rand_like(router_weight.detach().float())
    else:
        noise = torch.empty_like(router_weight.detach().float())
    dist.broadcast(noise, src=src, group=ep_group)
    return noise


def check_erc_ep_support(coef: float) -> None:
    """EP>1 时说明走的是全局口径（除非显式退回逐卡本地）。"""
    ep_size = 1
    try:
        from megatron.core import parallel_state as _ps

        if _ps.is_initialized():
            ep_size = int(_ps.get_expert_model_parallel_world_size())
    except Exception:
        ep_size = 1
    if ep_size <= 1 or float(coef) == 0.0:
        return
    allow_local = str(os.environ.get("SHENSI_ERC_EP_ALLOW_LOCAL", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if allow_local:
        _print_erc(
            f"[shensi][erc] [warn] SHENSI_ERC_EP_ALLOW_LOCAL=1：EP={ep_size}>1 时退回"
            "**逐卡本地** ERC（耦合矩阵只覆盖本地专家、梯度只进本地专家）—— 仅供诊断/对照，"
            "训练请用默认的全局口径。"
        )
        return
    _print_erc(
        f"[shensi][erc] EP={ep_size}>1：走**全局**口径（moe.erc_gather_expert_weights 的 "
        "all-gather + 反向按 rank 切片回切；各 rank 噪声同步 -> ERC 标量一致）。"
        "需要逐卡本地口径做对照时设 SHENSI_ERC_EP_ALLOW_LOCAL=1。"
    )


def erc_group_plan(cfg) -> dict[int, list[int]]:
    """AttnRes block 分组：每组从它的『写层』开始，到下一组之前的所有 MoE 层。"""
    num_layers = int(getattr(cfg, "num_layers", 0) or 0)
    if num_layers <= 0:
        return {}
    n_hash = int(getattr(cfg, "moe_n_hash_layers", 0) or 0)
    block_size = int(getattr(cfg, "attn_res_block_size", 0) or 0)
    if block_size <= 0:
        return {}
    types = attn_res_block_layer_types(num_layers, n_hash, block_size)
    write_layers = [i for i, t in enumerate(types) if t == "block_write_layer"]
    if not write_layers:
        return {}
    groups: dict[int, list[int]] = {}
    for i in range(num_layers):
        if i < n_hash:
            continue
        gid = max(w for w in write_layers if w <= i)
        groups.setdefault(gid, []).append(i)
    return groups


def _local_global_layer_indices(model) -> list[int] | None:
    layers = list(getattr(getattr(model, "decoder", None), "layers", None) or [])
    if not layers:
        return None
    nums = [getattr(ly, "layer_number", None) for ly in layers]
    if not all(n for n in nums):
        return None
    return sorted(int(n) - 1 for n in nums)


def check_erc_groups_complete(model, owners, verbose: bool = True) -> dict:
    """本 stage 必须覆盖全部 AttnRes 分组，否则平均口径就不是 HF 的。"""
    cfg = getattr(model, "config", None)
    plan = erc_group_plan(cfg) if cfg is not None else {}
    n_owners = len(owners or {})
    if not plan:
        if verbose and "skip_no_cfg" not in _ERC_GROUP_AUDIT_LOGGED:
            _ERC_GROUP_AUDIT_LOGGED.add("skip_no_cfg")
            _print_erc(
                "[shensi][erc] [warn] 全局分组对账**跳过**：config 里没有 "
                "num_layers/moe_n_hash_layers/attn_res_block_size（离线脚本或伪造 model）"
            )
        return {"status": "skipped", "global_groups": 0, "local_groups": n_owners, "layers": []}
    local_idx = _local_global_layer_indices(model)
    if local_idx is None:
        if verbose and "skip_no_layer_number" not in _ERC_GROUP_AUDIT_LOGGED:
            _ERC_GROUP_AUDIT_LOGGED.add("skip_no_layer_number")
            _print_erc(
                "[shensi][erc] [warn] 全局分组对账**跳过**：拿不到本 stage 的全局层号"
                "（decoder.layers 为空 / layer_number 缺失）"
            )
        return {
            "status": "skipped",
            "global_groups": len(plan),
            "local_groups": n_owners,
            "layers": [],
        }
    have = sorted(gid for gid, ls in plan.items() if set(ls) & set(local_idx))
    missing = sorted(set(plan) - set(have))
    if missing or n_owners != len(have):
        raise RuntimeError(
            "[shensi][erc] ERC 分组**不完整**：本 stage（= 真正算 ERC 的 PP 末 stage）只覆盖 "
            f"{len(have)}/{len(plan)} 个 AttnRes block 分组，缺组 {missing}"
            f"（本地层={local_idx}，moe_group_owners 条目={n_owners}）。"
            "ERC 的语义是『全模型每组算一次、再对全模型组数取平均』"
            "（HF `ShensiForCausalLM.forward`），而 `loss_func` 只在 PP 末 stage 执行 —— "
            "所以末 stage **必须**拥有每一组的参数，否则算出来的不是 HF 口径（会静默按本地"
            "组数平均）。三种处置：① `--shensi-erc-loss-coef 0` 关掉 ERC；② 改 "
            "`pipeline_model_parallel_layout` / ERC 相关层分配，让全部 AttnRes block 的 MoE "
            "层都落在末 stage；③ 需要 ERC 的 PP>1 生产配置改用 `--shensi-erc-loss-coef 0` + "
            "单独的阶段（当前实现不支持跨 stage 的 ERC）。"
        )
    if verbose and "ok" not in _ERC_GROUP_AUDIT_LOGGED:
        _ERC_GROUP_AUDIT_LOGGED.add("ok")
        _print_erc(
            f"[shensi][erc] 全局分组对账 PASS：本 stage 覆盖 {len(have)}/{len(plan)} 组"
            f"（组 = AttnRes block 写层 {have}；全局层号 {local_idx}）→ "
            "平均口径 = HF 的『按全模型组数平均』"
        )
    return {
        "status": "ok",
        "global_groups": len(plan),
        "local_groups": len(have),
        "layers": local_idx,
    }


def shensi_erc_loss(model, alpha: float = 1.0, verbose: bool = False) -> torch.Tensor | None:
    """本 stage 的 MoE 组各算一次 ERC，再按组数平均；没有组就返回 None。"""
    owners = getattr(model, "moe_group_owners", None)
    if not owners:
        _print_erc(
            "[shensi][erc] 本 stage 没有 MoE 组（moe_group_owners 为空）→ ERC 不上报、不反传；"
            "这是**本 rank 的分片事实**（PP>1 的非末 stage 天然如此），不是 erc_loss_coef=0。"
        )
        return None
    cfg = getattr(model, "config", None)
    check_erc_ep_support(float(getattr(cfg, "erc_loss_coef", 0.0) or 0.0))
    audit = check_erc_groups_complete(model, owners, verbose=verbose)
    total = None
    num_groups = 0
    for mlp in owners.values():
        router_weight, down_proj_weight, gate_up_proj = erc_gather_expert_weights(mlp)
        group_loss = erc_loss_func(
            router_weight.float(),
            down_proj_weight.float(),
            gate_up_proj.float(),
            alpha=alpha,
            noise=_erc_noise_for(router_weight),
        )
        if verbose:
            _print_erc(
                f"[shensi][erc] erc={float(group_loss.detach()):.6e} "
                f"gate_up={tuple(gate_up_proj.shape)} ep={_resolve_ep()[0]} "
                f"groups={audit.get('local_groups')}/{audit.get('global_groups')}"
            )
        total = group_loss if total is None else total + group_loss
        num_groups += 1
    if total is None or num_groups == 0:
        return None
    return total / num_groups


def attach_erc_loss(
    activation: torch.Tensor,
    erc: torch.Tensor | None,
    coef: float,
    calculate_per_token_loss: bool = True,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """把 ERC 挂到 per-token 的 loss 上（沿用 mcore 的 aux-loss 反传缩放机制）。"""
    if erc is None or coef == 0.0:
        return activation
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    scaled = erc * coef
    if calculate_per_token_loss:
        tokens = num_tokens if num_tokens is not None else activation.shape[0]
        return MoEAuxLossAutoScaler.apply(activation, scaled * tokens)
    return MoEAuxLossAutoScaler.apply(activation, scaled)


def verify_erc_attach_scaling(
    num_tokens: int = 7, coef: float = 3.0, main_loss_backward_scale: float = 1.0
) -> dict:
    """自证 `attach_erc_loss` 的反传缩放：ERC 项的梯度 = coef × 主损失缩放（× token 数）。"""
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    saved = MoEAuxLossAutoScaler.main_loss_backward_scale
    out: dict[str, object] = {
        "num_tokens": int(num_tokens),
        "coef": float(coef),
        "main_loss_backward_scale": float(main_loss_backward_scale),
    }
    try:
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            float(main_loss_backward_scale)
        )
        for per_token in (False, True):
            act = torch.ones(4, dtype=torch.float32, requires_grad=True)
            erc = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
            loss = attach_erc_loss(
                act, erc, coef, calculate_per_token_loss=per_token, num_tokens=num_tokens
            )
            loss.sum().backward()
            expected = (
                float(coef)
                * float(main_loss_backward_scale)
                * (float(num_tokens) if per_token else 1.0)
            )
            tag = "per_token" if per_token else "no_per_token"
            out[f"{tag}_grad_erc"] = float(erc.grad)
            out[f"{tag}_expected"] = expected
            out[f"{tag}_identity_ok"] = bool(torch.equal(loss.detach(), act.detach()))
            out[f"{tag}_grad_activation_ok"] = bool(torch.equal(act.grad, torch.ones(4)))
            out[f"{tag}_ok"] = bool(
                abs(float(erc.grad) - expected) < 1e-6
                and torch.equal(loss.detach(), act.detach())
                and torch.equal(act.grad, torch.ones(4))
            )
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = saved
    out["ok"] = bool(out["per_token_ok"] and out["no_per_token_ok"])
    return out


def erc_of(model):
    """(模型对象, ERC 标量或 None, coef, alpha, 为什么没有)。"""
    modules = unwrap_model(model)
    model_obj = modules[0] if isinstance(modules, (list, tuple)) else modules
    cfg = getattr(model_obj, "config", None)
    alpha = float(getattr(cfg, "erc_loss_alpha", 0.5))
    coef = float(getattr(cfg, "erc_loss_coef", 0.0) or 0.0)
    if getattr(model_obj, "moe_group_owners", None) is None:
        return model_obj, None, coef, alpha, "no_groups"
    if coef == 0.0:
        return model_obj, None, coef, alpha, "coef0"
    return model_obj, shensi_erc_loss(model_obj, alpha=alpha), coef, alpha, "ok"


def attach_erc_to_loss(loss, num_tokens, report, *, output_tensor, loss_mask, model):
    """在已算好的 (loss, num_tokens, report) 上叠加 ERC 项；关掉或没有组时原样返回。"""
    global _ERC_DISABLED_LOGGED
    model_obj, erc, coef, alpha, reason = erc_of(model)
    if erc is None:
        if not _ERC_DISABLED_LOGGED:
            _ERC_DISABLED_LOGGED = True
            if reason == "no_groups":
                _print_erc(
                    "[shensi][erc] 本 stage 没有 MoE 组（moe_group_owners 为空）→ 不上报 "
                    "erc loss/total loss，也不参与反传。**原因不是 coef=0**：PP>1 时这只说明"
                    "本 rank 不是『持有 MoE 组』的那个 stage（ERC 在末 stage 算）。"
                )
            else:
                _print_erc(
                    f"[shensi][erc] ERC 关闭：erc_loss_coef={coef}（模型 config）→ "
                    "不上报 erc loss/total loss，也不参与反传"
                )
        return loss, num_tokens, report
    args = get_args()
    calc_per_token = bool(getattr(args, "calculate_per_token_loss", False))
    tokens = num_tokens.item() if torch.is_tensor(num_tokens) else num_tokens
    tokens = int(tokens)
    losses = output_tensor.view(-1).float()
    mask = loss_mask.view(-1).float()
    scaled_losses = attach_erc_loss(
        losses, erc, coef, calculate_per_token_loss=calc_per_token, num_tokens=tokens
    )
    loss_out = torch.sum(scaled_losses * mask)
    if not torch.allclose(loss_out.detach(), loss.detach(), rtol=0.0, atol=1e-4):
        _print_erc(
            f"[shensi][erc] [warn] 重算的 lm loss({float(loss_out):.6f}) 与上游"
            f"({float(loss):.6f}) 不一致 → 本步 ERC 只上报、不反传"
        )
        loss_out = loss
    denom = torch.tensor(float(tokens), dtype=torch.float, device=loss.device).view(1)
    erc_value = float(erc.detach())
    erc_sum = torch.tensor(coef * erc_value * tokens, dtype=torch.float, device=loss.device).view(1)
    report = dict(report)
    report["erc loss"] = torch.cat([erc_sum, denom])
    report["total loss"] = torch.cat([loss.detach().view(1) + erc_sum, denom])
    audit = check_erc_groups_complete(
        model_obj, getattr(model_obj, "moe_group_owners", None), verbose=False
    )
    _print_erc(
        f"[shensi][erc] coef={coef} alpha={alpha} "
        f"groups={audit['local_groups']}/{audit['global_groups']}（本 stage/全局）"
        f"status={audit['status']} erc={erc_value:.6e} tokens={tokens} "
        f"per_token_loss={calc_per_token} → 计入反传的 erc 项 = {coef * erc_value:.6e}"
    )
    return loss_out, num_tokens, report
