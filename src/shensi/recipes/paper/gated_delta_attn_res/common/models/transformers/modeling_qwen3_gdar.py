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

"""Qwen3 + GDAR（门控 delta 残差连接）的 HF 参考实现。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults

from .configuration_qwen3_gdar import GATE_CHANNELS, Qwen3GDARConfig

__all__ = ["Qwen3GDARConfig", "Qwen3GDARDecoderLayer", "Qwen3GDARModel", "Qwen3GDARForCausalLM"]







class UnweightedRMSNorm(nn.Module):

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _make_proj(in_features, out_features, rank, bias=False):
    if rank is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return nn.Sequential(
        nn.Linear(in_features, rank, bias=bias),
        nn.Linear(rank, out_features, bias=bias),
    )


def _apply_proj(x, proj):
    x = x.float()
    if isinstance(proj, torch.Tensor):
        return F.linear(x, proj.float())
    if isinstance(proj, nn.Sequential):
        layers = list(proj)
        for lin in layers:
            bias = None if lin.bias is None else lin.bias.float()
            x = F.linear(x, lin.weight.float(), bias)
        return x
    bias = None if proj.bias is None else proj.bias.float()
    return F.linear(x, proj.weight.float(), bias)


def _proj_last(proj):
    return proj if isinstance(proj, nn.Linear) else proj[-1]


def _proj_bias(proj):
    return proj.bias if isinstance(proj, nn.Linear) else proj[-1].bias


def _proj_weight(proj):
    if isinstance(proj, torch.Tensor):
        return proj
    return proj[1].weight @ proj[0].weight if isinstance(proj, nn.Sequential) else proj.weight


def _init_gate_proj(proj, init, init_bias):
    if init == "paper":
        return
    bias = proj.bias if isinstance(proj, nn.Linear) else proj[-1].bias
    hidden = bias.numel() // 3
    with torch.no_grad():
        if init == "zero":
            bias.zero_()
        elif init == "identity":
            target = torch.zeros_like(bias)
            for i, sign in enumerate((1.0, -1.0, 1.0)):
                target[i * hidden : (i + 1) * hidden] = sign * init_bias
            bias.copy_(target)
        elif init == "uniform":
            bias.fill_(init_bias)
        else:
            raise ValueError(f"unknown gate init: {init}")
        linears = [proj] if isinstance(proj, nn.Linear) else list(proj)
        scale = 1.0 / math.sqrt(hidden)
        for module in linears:
            nn.init.uniform_(module.weight, -scale, scale)


def _apply_gate_channels(decay, erase, write, channels):
    if channels == "dew":
        return decay, erase, write
    if channels == "none":
        ones = torch.ones_like(decay)
        return ones, torch.zeros_like(erase), ones
    if channels == "scalar":
        write = write.mean(dim=-1, keepdim=True)
        channels = "w"
    if "d" not in channels:
        decay = torch.ones_like(decay)
    if "e" not in channels:
        erase = torch.zeros_like(erase)
    if "w" not in channels:
        write = torch.ones_like(write)
    return decay, erase, write


def _whitening_transform(values, mode, ridge, return_inverse=False):






    with torch.no_grad():
        S = values.reshape(-1, values.shape[-1]).float().detach()
        if mode == "diag":
            var = S.pow(2).mean(dim=0)
            w = torch.rsqrt(var + ridge)
            return (w, 1.0 / w) if return_inverse else w
        cov = (S.transpose(0, 1) @ S) / S.shape[0]
        d = cov.shape[0]
        cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(cov)
        w = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
        if not return_inverse:
            return w
        w_inv = evecs @ torch.diag(torch.sqrt(evals.clamp_min(ridge))) @ evecs.transpose(0, 1)
        return w, w_inv


def _softmax1(logits, dim):
    s = torch.logsumexp(logits, dim=dim, keepdim=True)
    return torch.exp(logits - F.softplus(s))


def _depth_read(
    values,
    query,
    eps,
    heads=1,
    null=False,
    whiten="off",
    ridge=1e-3,
    return_scores=False,
    mix="raw",
):


    out_dtype = values.dtype
    values, query = values.float(), query.float()
    """Depth read: softmax (optionally Softmax_1) over sources, optionally whitened.

    Args:
        values: ``(num_tokens, num_sources, hidden)`` float32, raw retrieval values.
        query:  ``(num_tokens, hidden)`` float32.
    Returns:
        ``(routed, probs)``; ``probs`` is ``(num_tokens, num_sources)`` or
        ``(num_tokens, num_sources, heads)`` when ``heads > 1``.
    """
    if mix not in ("raw", "whitened"):
        raise ValueError(f"mix must be 'raw' or 'whitened', got {mix!r}")
    num_tokens, num_sources, hidden = values.shape
    w_inv = None
    if whiten == "per_head":





        values, query = values.float(), query.float()
        dh = hidden // heads
        flat_v = values.reshape(-1, heads, dh)
        flat_q = query.reshape(-1, heads, dh)
        with torch.no_grad():
            V = flat_v.detach()
            n = V.shape[0]
            cov = torch.einsum("nhd,nhe->hde", V, V) / n
            scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
            eps_d = torch.finfo(cov.dtype).eps
            ridge_h = max(dh, n) * eps_d * scale
            cov.diagonal(dim1=-2, dim2=-1).add_(ridge_h.unsqueeze(-1))
            evals, evecs = torch.linalg.eigh(cov)
            floor = (evals[..., -1:] * dh * eps_d).clamp_min(torch.finfo(cov.dtype).tiny)
            whiten_h = (
                evecs
                @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor)))
                @ evecs.transpose(-1, -2)
            )
        values_s = torch.einsum("nhd,hde->nhe", flat_v, whiten_h).reshape(
            num_tokens, num_sources, hidden
        )
        query_s = torch.einsum("nhd,hde->nhe", flat_q, whiten_h).reshape(num_tokens, hidden)
        mix_values = values
        if heads > 1:
            vs = values_s.view(num_tokens, num_sources, heads, dh)
            qs = query_s.view(num_tokens, heads, dh)
            recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
            logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip
            probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
            routed = (
                probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)
            ).sum(dim=1)
            routed = routed.reshape(num_tokens, hidden)
        else:
            recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
            logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip
            probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
            routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
        if return_scores:
            return routed.to(out_dtype), probs
        return routed.to(out_dtype)
    if whiten in ("diag", "full"):
        if mix == "whitened":
            w, w_inv = _whitening_transform(values, whiten, ridge, return_inverse=True)
        else:
            w = _whitening_transform(values, whiten, ridge)

        w = w.to(values.dtype)
        if w.dim() == 1:
            values_s = values * w
            query_s = query * w
        else:
            w = w.float()
            values_s = values @ w
            query_s = query @ w
    else:
        values_s, query_s = values, query


    mix_values = values_s if mix == "whitened" else values

    if heads > 1:
        dh = hidden // heads
        vs = values_s.view(num_tokens, num_sources, heads, dh)
        qs = query_s.view(num_tokens, heads, dh)
        recip = torch.rsqrt(vs.square().mean(dim=-1) + eps)
        logits = (vs * qs.unsqueeze(1)).sum(dim=-1) * recip
        probs = _softmax1(logits, dim=1) if null else logits.softmax(dim=1)
        routed = (probs.unsqueeze(-1) * mix_values.view(num_tokens, num_sources, heads, dh)).sum(
            dim=1
        )
        routed = routed.reshape(num_tokens, hidden)
    else:
        recip = torch.rsqrt(values_s.square().mean(dim=-1) + eps)
        logits = (values_s * query_s.unsqueeze(1)).sum(dim=-1) * recip
        probs = _softmax1(logits, dim=-1) if null else logits.softmax(dim=-1)
        routed = (probs.unsqueeze(-1) * mix_values).sum(dim=1)
    if mix == "whitened" and w_inv is not None:

        w_inv = w_inv.to(routed.dtype)
        routed = routed * w_inv if w_inv.dim() == 1 else routed @ w_inv
    if return_scores:
        return routed.to(out_dtype), probs
    return routed.to(out_dtype)


class AttentionResidual(nn.Module):

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__()
        self.norm = UnweightedRMSNorm(config.rms_norm_eps)



        self._eps = float(config.rms_norm_eps)
        hidden = config.hidden_size
        self.hidden = hidden

        self.gate_param = getattr(config, "attn_res_gate_param", "sigmoid")



        _rule = getattr(config, "attn_res_update", "shensi")
        self.update_rule = "shensi" if _rule == "reference" else _rule
        self.write_carrier_bias = getattr(config, "attn_res_write_carrier_bias", -4.0)
        self.read_heads = int(getattr(config, "attn_res_read_heads", 1) or 1)
        self.read_null = bool(getattr(config, "attn_res_read_null", False))
        self.read_whiten = getattr(config, "attn_res_read_whiten", "off")

        self.gate_source = getattr(config, "attn_res_gate_source", "state")
        self.decay_positivity = getattr(config, "attn_res_decay_positivity", "free")
        self.lambda_clamp = getattr(config, "attn_res_lambda_clamp", -0.5)
        self.read_mix = getattr(config, "attn_res_read_mix", "raw")
        if self.gate_source not in ("state", "prefix", "delta"):
            raise ValueError(
                f"attn_res_gate_source must be state|prefix|delta, got {self.gate_source!r}"
            )
        if self.decay_positivity not in ("free", "project"):
            raise ValueError(
                f"attn_res_decay_positivity must be free|project, got {self.decay_positivity!r}"
            )
        if self.read_mix not in ("raw", "whitened"):
            raise ValueError(f"attn_res_read_mix must be raw|whitened, got {self.read_mix!r}")
        self.read_ridge = float(getattr(config, "attn_res_read_ridge", 1e-3))
        self.address = getattr(config, "attn_res_address", "state")


        self.gate_channels = getattr(config, "attn_res_gate_channels", "dew")
        if self.gate_channels not in GATE_CHANNELS:
            raise ValueError(
                f"attn_res_gate_channels={self.gate_channels!r} not in {GATE_CHANNELS}"
            )
        if hidden % self.read_heads:
            raise ValueError(
                f"attn_res_read_heads={self.read_heads} must divide hidden_size={hidden}"
            )

        self.gate_proj = _make_proj(
            hidden, 3 * hidden, getattr(config, "attn_res_gate_rank", None), bias=True
        )


        self.q_rank = getattr(config, "attn_res_q_rank", None)
        self.k_rank = getattr(config, "attn_res_k_rank", None)
        self.q_proj = self._make_qk(hidden, self.q_rank)
        self.k_proj = self._make_qk(hidden, self.k_rank)













        self.decay_ladder = int(getattr(config, "attn_res_decay_ladder", 0) or 0)
        self.decay_tau_max = float(getattr(config, "attn_res_decay_tau_max", 100.0))
        self.decay_tau = nn.Parameter(torch.zeros(hidden)) if self.decay_ladder > 1 else None





        self.read_scale = nn.Parameter(torch.zeros(1))

        if self.gate_param == "deviation":



            for module in (
                [self.gate_proj] if isinstance(self.gate_proj, nn.Linear) else list(self.gate_proj)
            ):
                nn.init.zeros_(module.weight)
            self.decay_scale = nn.Parameter(torch.zeros(1))
            self.erase_scale = nn.Parameter(torch.zeros(1))
            self.write_scale = nn.Parameter(torch.zeros(1))

        self.init_name = getattr(config, "attn_res_gate_init", "paper")
        self.identity_bias = getattr(config, "attn_res_gate_init_bias", 4.0)
        self.reset_parameters()

    @staticmethod
    def _make_qk(hidden, rank):
        if rank is None:
            return nn.Parameter(torch.empty(hidden, hidden))
        return nn.Sequential(
            nn.Linear(hidden, rank, bias=False), nn.Linear(rank, hidden, bias=False)
        )

    def _gate_head(self, state):
        raw = _apply_proj(state, self.gate_proj)
        return raw.reshape(*state.shape[:-1], 3, -1)

    def _gate_input(self, state, prefix, delta):
        src = self.gate_source
        if src == "prefix" and prefix is not None:
            return self.norm(prefix.float())
        if src == "delta" and delta is not None:
            return self.norm(delta.float())
        return state

    def _gates(self, state):
        if getattr(self, "_force_identity", False):
            ones = torch.ones_like(state)
            zeros = torch.zeros_like(state)
            return ones, zeros, ones
        raw = self._gate_head(state)
        if self.gate_param == "sigmoid":
            decay, erase, write = torch.sigmoid(raw).unbind(-2)
        else:
            r_decay, r_erase, r_write = raw.unbind(-2)


            tau = self.decay_tau.exp() if self.decay_tau is not None else 1.0



            if self.decay_positivity == "project":





                decay_scale = (
                    self.decay_scale + (self.decay_scale.clamp(min=0.0) - self.decay_scale).detach()
                )
            else:
                decay_scale = self.decay_scale
            decay = torch.exp(-F.softplus(r_decay) * decay_scale * tau)
            erase = F.softplus(r_erase) * self.erase_scale
            write = 1.0 + torch.tanh(r_write) * self.write_scale
        return _apply_gate_channels(decay, erase, write, self.gate_channels)

    def reset_parameters(self):
        std = 0.02
        with torch.no_grad():
            for proj in (self.q_proj, self.k_proj):
                if isinstance(proj, nn.Parameter):
                    nn.init.normal_(proj, std=std)
            if self.gate_param == "deviation":



                bias = _proj_bias(self.gate_proj)
                bias.zero_()
                bias[2 * self.hidden : 3 * self.hidden] = self.write_carrier_bias
                for module in (
                    [self.gate_proj]
                    if isinstance(self.gate_proj, nn.Linear)
                    else list(self.gate_proj)
                ):
                    nn.init.zeros_(module.weight)


                    if module.bias is not None and module is not _proj_last(self.gate_proj):
                        nn.init.zeros_(module.bias)
                for scale in (self.decay_scale, self.erase_scale, self.write_scale):
                    scale.zero_()
            else:
                _init_gate_proj(self.gate_proj, self.init_name, self.identity_bias)
            if self.decay_tau is not None:


                log_tau = torch.linspace(0.0, 1.0, self.decay_ladder) * math.log(self.decay_tau_max)
                repeats = -(-self.hidden // self.decay_ladder)
                ladder = log_tau.repeat(repeats)[: self.hidden]
                self.decay_tau.copy_(ladder.to(self.decay_tau.device, self.decay_tau.dtype))
            self.read_scale.zero_()

    def _state(self, prefix, delta):
        return self.norm(prefix.float() + (delta.float() if delta is not None else 0.0))

    def read(self, prefix, blocks, state=None):
        prefix = prefix.float()
        stream_dtype = prefix.dtype if not hasattr(prefix, "dtype") else prefix.dtype
        if blocks is None or blocks.shape[-2] == 0:
            return prefix.to(stream_dtype), None
        if state is None:
            state = self._state(prefix, None)
        values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
        query = _apply_proj(state, self.q_proj)
        routed, scores = _depth_read(
            values,
            query,
            self._eps,
            heads=self.read_heads,
            null=self.read_null,
            whiten=self.read_whiten,
            mix=self.read_mix,
            ridge=self.read_ridge,
            return_scores=True,
        )
        return prefix + self.read_scale * routed, scores

    @torch.no_grad()
    def read_premises(self, prefix, blocks):
        if blocks is None or blocks.shape[-2] == 0:
            return {"n_sources": 0}
        prefix = prefix.float()
        values = torch.cat([blocks.float(), prefix.unsqueeze(-2)], dim=-2)
        query = _apply_proj(self._state(prefix, None), self.q_proj)
        _, probs = _depth_read(
            values,
            query,
            self._eps,
            heads=self.read_heads,
            null=self.read_null,
            whiten=self.read_whiten,
            ridge=self.read_ridge,
            return_scores=True,
            mix=self.read_mix,
        )
        out = {"n_sources": int(values.shape[-2])}

        k = min(values.shape[0], 8)
        gram = values[:k].transpose(1, 2) @ values[:k]
        out["key_cond"] = float(torch.linalg.cond(gram).mean())
        p_sum = probs.sum(dim=1).mean()
        out["null_mass"] = float(1.0 - p_sum)
        if probs.dim() == 3 and probs.shape[-1] > 1:
            w = probs.reshape(-1, probs.shape[-1]).float()
            w = w - w.mean(dim=0, keepdim=True)
            std = w.std(dim=0).clamp_min(1e-8)
            corr = (w / std).transpose(0, 1) @ (w / std) / w.shape[0]
            off = corr - torch.diag(torch.diagonal(corr))
            out["head_corr"] = float(off.abs().sum() / (corr.shape[0] * (corr.shape[0] - 1)))
        return out

    def update(self, prefix, delta, state=None):
        prefix_f = prefix.float()
        up_dtype = prefix.dtype
        delta_f = delta.float() if delta is not None else None
        if state is None:
            state = self._state(prefix_f, delta_f)


        decay, erase, write = self._gates(self._gate_input(state, prefix_f, delta_f))











        if self.address == "novelty":
            base = decay * prefix_f
            src = delta_f if delta_f is not None else state
            denom = base.square().sum(dim=-1, keepdim=True).clamp_min(self._eps)
            address_src = src - (src * base).sum(dim=-1, keepdim=True) / denom * base
        elif self.address == "delta" and delta_f is not None:
            address_src = delta_f
        else:
            address_src = state
        khat = F.normalize(_apply_proj(address_src, self.k_proj), dim=-1)
        delta_term = delta_f if delta_f is not None else 0.0
        if self.update_rule == "objective":
            m = decay * prefix_f + write * delta_term





            lam = erase.mean(dim=-1, keepdim=True)
            if self.lambda_clamp is not None:
                lam = lam.clamp(min=float(self.lambda_clamp))
            updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)
        else:
            forgotten = decay * prefix_f
            r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
            updated = forgotten - khat * r + write * delta_term
        return updated.to(up_dtype), (decay, erase, write)

    def forward(self, prefix, delta, blocks, output_norm_weight=None, num_blocks=0):
        out_dtype = prefix.dtype
        prefix_f = prefix.float()
        delta_f = delta.float() if delta is not None else None







        state = self._state(prefix_f, delta_f)
        routed, scores = self.read(prefix_f, blocks, state=state)
        updated, gates = self.update(prefix_f, delta_f, state=state)
        output = updated + (routed - prefix_f)
        if output_norm_weight is not None:
            reciprocal_std = torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self._eps)
            output = output * reciprocal_std * output_norm_weight.float()
        return output.to(out_dtype), updated.to(out_dtype), gates, scores


class DepthRead(nn.Module):

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__()
        self.q_proj = AttentionResidual._make_qk(
            config.hidden_size, getattr(config, "attn_res_q_rank", None)
        )
        if isinstance(self.q_proj, nn.Parameter):
            nn.init.normal_(self.q_proj, std=0.02)
        self.eps = config.rms_norm_eps
        self.read_heads = int(getattr(config, "attn_res_read_heads", 1) or 1)
        self.read_null = bool(getattr(config, "attn_res_read_null", False))
        self.read_whiten = getattr(config, "attn_res_read_whiten", "off")
        self.read_mix = getattr(config, "attn_res_read_mix", "raw")
        self.read_ridge = float(getattr(config, "attn_res_read_ridge", 1e-3))
        self.read_scale = nn.Parameter(torch.zeros(1))

    def forward(self, prefix_flat, blocks):
        if blocks is None or blocks.shape[-2] == 0:
            return prefix_flat
        values = torch.cat([blocks.float(), prefix_flat.float().unsqueeze(-2)], dim=-2)
        query = _apply_proj(prefix_flat.float(), self.q_proj)
        routed = _depth_read(
            values,
            query,
            self.eps,
            heads=self.read_heads,
            null=self.read_null,
            whiten=self.read_whiten,
            mix=self.read_mix,
            ridge=self.read_ridge,
        )
        return (prefix_flat.float() + self.read_scale * routed).to(prefix_flat.dtype)


def record_router_stats(
    stats, layer_idx, sublayer, scores=None, gates=None, n_sources=None, premises=None
):
    if stats is None:
        return
    entry = {"layer": layer_idx, "sublayer": sublayer}
    with torch.no_grad():
        if scores is not None:
            w = scores.detach().float()
            entry["n_sources"] = int(w.shape[-1])
            entry["sharpness"] = float(w.max(dim=-1).values.mean())
            entry["entropy"] = float(-(w * (w + 1e-8).log()).sum(dim=-1).mean())
        elif n_sources is not None:
            entry["n_sources"] = int(n_sources)
        if gates is not None:
            decay, erase, write = (g.detach().float().mean() for g in gates)
            entry["gate_decay"] = float(decay)
            entry["gate_erase"] = float(erase)
            entry["gate_write"] = float(write)
        if premises:
            entry.update(premises)
    stats.append(entry)


class Qwen3GDARDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3GDARConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_attn_res = AttentionResidual(config)
            self.mlp_attn_res = AttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
                delta_residual,
                attn_res_stats,
            )
        return self._forward_plain(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )

    def _attention(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings,
    ):
        output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        return output[0] if isinstance(output, tuple) else output

    def _forward_plain(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings,
    ):
        residual = hidden_states
        hidden_states = self._attention(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def _append_source(self, delta_residual, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1])
        return torch.cat([delta_residual, flat.unsqueeze(1)], dim=1)

    @staticmethod
    def _num_blocks(delta_residual):
        return 0 if delta_residual is None else delta_residual.shape[1]

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        delta_residual: torch.Tensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states.view(-1, hidden_size)
        per_sublayer_sources = self.attn_res_block_size == 1

        if not per_sublayer_sources and self.layer_idx % self.attn_res_block_size == 0:

            delta_residual = self._append_source(delta_residual, prefix_sum)
        elif per_sublayer_sources and (delta_residual is None or delta_residual.shape[1] == 0):
            delta_residual = self._append_source(delta_residual, prefix_sum)


        routed, scores = self.self_attention_attn_res.read(prefix_sum, delta_residual)

        routed = routed.to(hidden_states.dtype)
        attn_out = self._attention(
            self.input_layernorm(routed.view(batch_size, seq_len, hidden_size)),
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            position_embeddings,
        )
        prefix_sum, gates = self.self_attention_attn_res.update(
            prefix_sum, attn_out.reshape(-1, hidden_size)
        )
        record_router_stats(
            attn_res_stats,
            self.layer_idx,
            "attn",
            scores=scores,
            gates=gates,
            premises=self.self_attention_attn_res.read_premises(prefix_sum, delta_residual)
            if attn_res_stats is not None
            else None,
        )
        if per_sublayer_sources:
            delta_residual = self._append_source(delta_residual, attn_out)


        routed, scores = self.mlp_attn_res.read(prefix_sum, delta_residual)

        routed = routed.to(hidden_states.dtype)

        routed = routed.to(hidden_states.dtype)
        mlp_out = self.mlp(
            self.post_attention_layernorm(routed.view(batch_size, seq_len, hidden_size))
        )
        prefix_sum, gates = self.mlp_attn_res.update(prefix_sum, mlp_out.reshape(-1, hidden_size))
        record_router_stats(
            attn_res_stats,
            self.layer_idx,
            "mlp",
            scores=scores,
            gates=gates,
            premises=self.mlp_attn_res.read_premises(prefix_sum, delta_residual)
            if attn_res_stats is not None
            else None,
        )
        if per_sublayer_sources:
            delta_residual = self._append_source(delta_residual, mlp_out)

        return prefix_sum.view(batch_size, seq_len, hidden_size), delta_residual


class Qwen3GDARModel(Qwen3PreTrainedModel):

    config_class = Qwen3GDARConfig
    base_model_prefix = "model"

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3GDARDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.use_attn_residuals = config.attn_res_block_size is not None
        self.output_attn_res = self.use_attn_residuals and config.attn_res_output_route
        if self.output_attn_res:
            self.output_attn_res_module = DepthRead(config)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, AttentionResidual):

            module.reset_parameters()
            return
        super()._init_weights(module)

    def post_init(self):
        super().post_init()


        for module in self.modules():
            if isinstance(module, AttentionResidual):
                module.reset_parameters()

    @merge_with_config_defaults
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        attn_res_stats: list | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        delta_residual = None
        if self.use_attn_residuals:
            delta_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]
            )

        for decoder_layer in self.layers:
            if self.use_attn_residuals:
                hidden_states, delta_residual = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    delta_residual=delta_residual,
                    attn_res_stats=attn_res_stats,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

        if self.output_attn_res:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = self.output_attn_res_module(
                hidden_states.view(-1, hidden_size), delta_residual
            ).view(batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3GDARForCausalLM(Qwen3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen3GDARConfig

    def __init__(self, config: Qwen3GDARConfig):
        super().__init__(config)
        self.model = Qwen3GDARModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_attn_res_stats: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        stats = [] if return_attn_res_stats else None
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            attn_res_stats=stats,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        out = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values
        )
        if stats is not None:
            out.attn_res_stats = stats
        return out


for _register, _args in (
    (AutoConfig.register, ("qwen3_gdar", Qwen3GDARConfig)),
    (AutoModel.register, (Qwen3GDARConfig, Qwen3GDARModel)),
    (AutoModelForCausalLM.register, (Qwen3GDARConfig, Qwen3GDARForCausalLM)),
):
    try:
        _register(*_args)
    except ValueError:
        pass











for _cls, _auto in (
    (Qwen3GDARConfig, "AutoConfig"),
    (Qwen3GDARModel, "AutoModel"),
    (Qwen3GDARForCausalLM, "AutoModelForCausalLM"),
):
    try:
        _cls.register_for_auto_class(_auto)
    except (AttributeError, ValueError):
        pass
