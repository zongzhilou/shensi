"""RealFormer 算子的可执行参考：官方实现的逐行转写（PyTorch）。

上游是 TensorFlow（``google-research/google-research/realformer/realformer.py``，已随本目录
vendored，sha256 见 ``PROVENANCE.md``）。本文件把 ``residual_attention_layer``（820-836 行）
与模型层的调用方式（919-935 行）**逐行**转写过来，供 ``test_upstream_alignment.py`` 对拍：

    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_layer, query_layer)   # 820
    attention_scores = tf.multiply(attention_scores, 1/sqrt(size_per_head))   # 821-822
    cur_attention = attention_scores                                          # 824
    if prev_attention is not None: cur_attention += prev_attention            # 825-826
    attention_logits = cur_attention                                          # 828
    if use_running_mean: attention_logits /= (num_prev_layers + 1.0)          # 829-830
    attention_logits += (1 - mask) * -10000.0                                 # 838
    attention_probs = tf.nn.softmax(attention_logits)                         # 845
    context_layer = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_layer) # 849
    return context_layer, cur_attention                                       # 851

转写时只做形状适配（TF 的 ``[B, F, T, N, H]`` 布局 ↔ torch 的 ``[B, N, F, T, H]``）；
数学与加法顺序一字未改，包括"mask 用 ``(1 - mask) * -10000`` 的加性形式"这一点。
"""

from __future__ import annotations

import math

import torch


def upstream_attention_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """``QK^T / sqrt(d_head)``，入参出参都是 ``[B, N, F/T, H]``。"""
    scores = torch.matmul(query, key.transpose(-1, -2))
    return scores * (1.0 / math.sqrt(float(query.shape[-1])))


def upstream_residual_from_scores(
    attention_scores: torch.Tensor,
    prev_attention: torch.Tensor | None,
    num_prev_layers: int,
    *,
    use_running_mean: bool = False,
    attention_mask: torch.Tensor | None = None,
    mask_is_binary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 824-851 行的片段：**给定**已经算好的 ``attention_scores``。

    单独拆出来是为了对拍时两侧喂**同一个** ``attention_scores``（否则比的是两次不同的
    QK^T，不是同一件事）。
    """
    cur_attention = attention_scores  # 824
    if prev_attention is not None:
        cur_attention = cur_attention + prev_attention  # 825-826
    attention_logits = cur_attention  # 828
    if use_running_mean:
        attention_logits = attention_logits / (num_prev_layers + 1.0)  # 829-830
    if attention_mask is not None:
        if mask_is_binary:
            adder = (1.0 - attention_mask.to(attention_logits.dtype)) * -10000.0  # 836-838
        else:
            adder = attention_mask.to(attention_logits.dtype)
        attention_logits = attention_logits + adder  # 840
    attention_probs = torch.nn.functional.softmax(attention_logits, dim=-1)  # 845
    return attention_probs, cur_attention  # 845 / 851


def upstream_residual_from_scores(
    attention_scores: torch.Tensor,
    prev_attention: torch.Tensor | None,
    num_prev_layers: int,
    *,
    use_running_mean: bool = False,
    attention_mask: torch.Tensor | None = None,
    mask_is_binary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 824-851 行的片段：**给定**已经算好的 ``attention_scores``。

    单独拆出来是为了对拍时两侧喂**同一个** ``attention_scores``（否则比的是两次不同的
    QK^T，不是同一件事）。
    """
    cur_attention = attention_scores  # 824
    if prev_attention is not None:
        cur_attention = cur_attention + prev_attention  # 825-826
    attention_logits = cur_attention  # 828
    if use_running_mean:
        attention_logits = attention_logits / (num_prev_layers + 1.0)  # 829-830
    if attention_mask is not None:
        if mask_is_binary:
            adder = (1.0 - attention_mask.to(attention_logits.dtype)) * -10000.0  # 836-838
        else:
            adder = attention_mask.to(attention_logits.dtype)
        attention_logits = attention_logits + adder  # 840
    attention_probs = torch.nn.functional.softmax(attention_logits, dim=-1)  # 845
    return attention_probs, cur_attention  # 845 / 851


def upstream_residual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prev_attention: torch.Tensor | None,
    num_prev_layers: int,
    *,
    use_running_mean: bool = False,
    attention_mask: torch.Tensor | None = None,
    mask_is_binary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 ``residual_attention_layer`` 的完整转写。

    Args:
        query/key/value: ``[B, N, F, H]`` / ``[B, N, T, H]``（多头的，head 维已展开）。
        prev_attention: 上一层返回的 ``cur_attention``，``[B, N, F, T]``；layer 0 传 ``None``。
        num_prev_layers: 上游同名参数（0-based 层号）。
        use_running_mean: 上游同名开关。
        attention_mask: 形状 ``[B, 1 or N, F, T]``；``mask_is_binary=True`` 时按上游
            ``(1 - mask) * -10000.0`` 相加（1 = 保留），否则按"已是加性 float mask"直接相加。
        mask_is_binary: 见上；HF 侧的 mask 是加性的，测试里会显式声明。

    Returns:
        ``(context_layer, cur_attention)`` —— 与上游同名同义。
    """
    attention_scores = upstream_attention_scores(query, key)  # 820-822
    attention_probs, cur_attention = upstream_residual_from_scores(
        attention_scores,
        prev_attention,
        num_prev_layers,
        use_running_mean=use_running_mean,
        attention_mask=attention_mask,
        mask_is_binary=mask_is_binary,
    )
    context_layer = torch.matmul(attention_probs, value)  # 849
    return context_layer, cur_attention  # 851


def upstream_model_loop(
    layers_qkvo,
    hidden_states: torch.Tensor,
    *,
    use_running_mean: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """上游 ``realformer_model`` 的层循环（919-935 行的调用方式）。

    ``layers_qkvo`` 是 ``(query, key, value) -> (B,N,F,H)...`` 的生成器：给定第 i 层的
    q/k/v 张量序列；本函数只负责把 ``prev_attention`` 串起来，返回每层的 ``cur_attention``。
    """
    prev_attention: torch.Tensor | None = None
    carries: list[torch.Tensor] = []
    for layer_idx, (query, key, value) in enumerate(layers_qkvo):
        context, prev_attention = upstream_residual_attention(
            query,
            key,
            value,
            prev_attention,
            layer_idx,
            use_running_mean=use_running_mean,
        )
        carries.append(prev_attention)
        hidden_states = context + hidden_states
    return hidden_states, carries
