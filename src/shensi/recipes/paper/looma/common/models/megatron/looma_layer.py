"""Looma 的块层：把块内迭代解到不动点，主干全部走 Megatron-Core 原生算子。

深度状态（stream、前缀和、行银行）打包进 ``hidden_states`` 逐层传递，宽度随深度增长；由
``looma_spec`` 的预设装载。
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.attention import apply_rotary_pos_emb
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params

from .looma_connection import LoomaAttentionResidual, LoomaConfig, looma_knobs_from_kwargs, solve_block

__all__ = ["LoomaTransformerLayer", "build_looma_submodules", "looma_knobs_from_kwargs"]


def build_looma_submodules(config: TransformerConfig) -> TransformerLayerSubmodules:
    """返回与朴素稠密本地构建器逐字相同的子模块 spec。

    同一调用、同一参数顺序，使主干模块的构建与初始化 RNG 抽取完全一致。
    """
    return get_gpt_layer_local_submodules(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        None,  # 第 5 个参数是 fp8；本配方走稠密本地路径
        normalization=config.normalization,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
    )


@contextlib.contextmanager
def _isolated_rng(seed: int):
    """在不推进全局 RNG 流的前提下执行一段初始化。"""
    seed = int(seed) % (2**31 - 1)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _stand_in_config() -> TransformerConfig:
    """给占位 submodules 用的一份最小合法配置，只用于构造占位实例，不参与真实构建。"""
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
    )


# 占位 submodules：模块级 spec 拿不到真 config，只能带一份实例；层见到这个占位（或任何工厂、
# 或 None）就按 config 重建出原生子模块，因此这份实例的取值不影响构建结果。
PLACEHOLDER_SUBMODULES = build_looma_submodules(_stand_in_config())


class LoomaTransformerLayer(TransformerLayer):
    """残差路径换成 Looma 深度连接、并解到不动点的单层解码器。

    状态打包为 ``[stream | prefix | row_1 ... row_N]``，宽度 ``(2 + N) * H`` 并随深度加一；首层
    收到普通 ``[s, b, H]``（此时 stream 与 prefix 同为嵌入行），末层做输出读并回到宽度 ``H``。
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules | None = None,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection=None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        cfg = looma_knobs_from_kwargs(kwargs, config)
        if (
            submodules is None
            or callable(submodules)
            or submodules is PLACEHOLDER_SUBMODULES
        ):
            # spec 里的 submodules 可能是占位实例或工厂（mcore 原样传进来），在此解析；
            # MTP spec 校验那条路也不接受 None。
            submodules = build_looma_submodules(config)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        # MTP 层是额外的预测头而非块：没有块间深度记忆要维护，整条退化为 mcore 原生的
        # pre-norm 残差层（`forward` 交给父类，含它的 recompute 钩子与 MoE 路径），也不建深度连接。
        self.is_plain_layer = bool(is_mtp_layer)
        if config.pipeline_model_parallel_size > 1:
            # 打包状态的宽度随深度增长，阶段边界上的激活是 (2 + N) * H 而非 config.hidden_size：
            # 固定形状的 p2p 路径按 config.hidden_size 建接收缓冲，只有动态路径才从发送方张量取
            # 形状。该开关没有 CLI 入口，只能在共享的 TransformerConfig 上置位（调度器读它）。
            # 只有 MoE 的 allgather token dispatcher 会消费它，而它会拒绝——那里要用 alltoall。
            config.variable_seq_lengths = True
        # output_attn_res 只存在于末层，模型因此不是逐层同构的：torch_dist 的默认层映射会把各层
        # 张量放进同一 checkpoint 组并前置层轴，只由一层提供的张量会让该轴其余位置为空而被
        # dist_checkpointing 拒绝。hetereogenous_dist_checkpoint 是 mcore 对此的开关，只被
        # TransformerBlock.sharded_state_dict 读取，torch 格式不受影响。
        config.hetereogenous_dist_checkpoint = True

        self.hidden_size = config.hidden_size
        self.looma_cfg: LoomaConfig = cfg
        self.is_last_layer = self.layer_number == config.num_layers
        self.write_dropout = (
            float(self.hidden_dropout) if cfg.residual_dropout is None else float(cfg.residual_dropout)
        )
        # 注意力输出门：qkv 拆成 q/gate/k/v 四份，块循环里同样支持（见 ``_attention``）。
        self.attention_output_gate = bool(self.config.attention_output_gate)
        if self.config.fused_single_qkv_rope:
            # 融合 kernel 交出的是未拆分的 mixed_qkv，而块循环的后续迭代要求只移动 query，这一档
            # 表达不出来：降级走拆分路径并说明，不静默丢弃。
            print(
                "[depth-connection:looma] 注意：fused_single_qkv_rope 在块循环里不适用（后续迭代"
                "只移动 query），本层走拆分路径（数值等价，只是少了那次融合）。",
                flush=True,
            )

        # 激活重算：本层 forward 不是父类那条，mcore 的 recompute 钩子到不了块循环，改由本层自己
        # 在每次 ``_block_step`` 上做；粒度坍缩为整步（selective 与 full 在此是同一件事）。
        self.recompute = bool(config.recompute_granularity) and not self.is_plain_layer
        # fp32 残差流：整个打包状态走 fp32，sublayer 的输入再按模型精度转回（与 mcore 的
        # fp32_residual_connection 同口径）。
        self.fp32_residual = bool(config.fp32_residual_connection)

        # 在 fork 出的 RNG 里构建与初始化，主干保住自己的 RNG 抽取：同种子下逐位一致。
        base_seed = int(getattr(config, "seed", 0)) + 104729 * self.layer_number
        with _isolated_rng(base_seed):
            self.self_attention_attn_res = LoomaAttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            self.mlp_attn_res = LoomaAttentionResidual(
                self.hidden_size, cfg, eps=config.layernorm_epsilon
            )
            if self.is_last_layer and cfg.output_route:
                self.output_attn_res = LoomaAttentionResidual(
                    self.hidden_size, cfg, eps=config.layernorm_epsilon
                )

        if config.sequence_parallel:
            # 这些模块不是 TP 感知的，其梯度要在 TP 组内 all-reduce：与 mcore 对复制参数的
            # 一贯处理相同。
            for module in (self.self_attention_attn_res, self.mlp_attn_res):
                for param in module.parameters():
                    param.sequence_parallel = True

    def _unpack(self, hidden_states: Tensor):
        """``[s, b, W]`` -> ``(stream, prefix, rows | None)``。

        ``W == H`` 是首层的输入（此时 stream 与 prefix 同为嵌入）；否则 ``W == (2 + N) * H``，
        ``N`` 是已经贴出的行数。
        """
        h = self.hidden_size
        width = hidden_states.shape[-1]
        flat = hidden_states.reshape(-1, width)
        if self.fp32_residual and flat.dtype != torch.float32:
            flat = flat.float()
        if width == h:
            return flat, flat, None
        stream, prefix = flat[:, :h], flat[:, h : 2 * h]
        rows = flat[:, 2 * h :].reshape(flat.shape[0], (width - 2 * h) // h, h)
        return stream, prefix, rows

    def _pack(self, stream: Tensor, prefix: Tensor, rows: Tensor | None, shape) -> Tensor:
        parts = [stream, prefix] if rows is None else [stream, prefix, rows.reshape(stream.shape[0], -1)]
        flat = torch.cat(parts, dim=-1)
        return flat.reshape(shape[0], shape[1], flat.shape[-1])

    @staticmethod
    def _append(rows: Tensor | None, source: Tensor) -> Tensor:
        flat = source.reshape(-1, source.shape[-1])
        if rows is None:
            return flat.unsqueeze(1)
        return torch.cat([rows, flat.unsqueeze(1)], dim=1)

    def _owning_output(self, hidden_states: Tensor) -> Tensor:
        """让层输出在流水并行下自持存储。

        打包输出是 ``cat`` + ``reshape`` 的视图，而 ``schedules.deallocate_output_tensor`` 断言
        ``out._base is None``；仅在 pp > 1（会走释放那条路）且确实是视图时克隆。
        """
        if self.config.pipeline_model_parallel_size > 1 and hidden_states._base is not None:
            return hidden_states.clone()
        return hidden_states

    def _write_dropout(self, bda_fn, x: Tensor) -> Tensor:
        """施加 ``bias_dropout_add`` 本会作用在 ``x`` 上的那份 dropout。

        连接取代了残差相加，故一并接管子层输出的 dropout；复用同一个可调用对象是因其 philox
        消耗与 eager ``F.dropout`` 不同，传入零残差即可让它精确返回 ``dropout(x)``。
        """
        p = self.write_dropout
        if not self.training or p <= 0.0:
            return x
        if bda_fn is None:
            return F.dropout(x, p=p, training=True)
        bda = bda_fn(self.training, self.config.bias_dropout_fusion)
        return bda((x, None), torch.zeros_like(x), p)

    def _attention(
        self,
        hidden_states: Tensor,
        frozen_kv: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None,
        rotary_pos_emb,
        attention_bias: Tensor | None,
        packed_seq_params,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """原生注意力路径：query 始终移动，K/V 可选择冻结。

        各步都是 mcore 原生算子（QKV 投影与 GQA 头布局、RoPE、core attention、输出投影），只是允许
        query 与 K/V 取自不同张量，供块循环的后续迭代复用第一次的 K/V；仅训练/评测路径。
        """
        attn = self.self_attention
        gate = None
        if self.attention_output_gate:
            # output gate：qkv 拆成 (q, gate, k, v)，gate 与 q 一起每轮重算，K/V 照旧冻结
            query, gate, key, value = attn.get_query_key_value_tensors(
                hidden_states, split_qkv=True, output_gate=True
            )
        else:
            query, key, value = attn.get_query_key_value_tensors(hidden_states, split_qkv=True)

        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        cu_seqlens_q = cu_seqlens_kv = None
        rope_max_seqlen = None
        if rotary_pos_emb is not None and thd:
            q_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
            kv_padded = getattr(packed_seq_params, "cu_seqlens_kv_padded", None)
            cu_seqlens_q = q_padded if q_padded is not None else packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = kv_padded if kv_padded is not None else packed_seq_params.cu_seqlens_kv
            max_q, max_kv = packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv
            rope_max_seqlen = max(max_q, max_kv) if (max_q is not None and max_kv is not None) else None
        if thd:
            query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1)

        if frozen_kv is None and rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )
            frozen_kv = (key, value)
        elif frozen_kv is not None:
            # 后续迭代：只移动 query，历史用第一次贴出的那份 K/V
            key, value = frozen_kv
            if rotary_pos_emb is not None and rotary_pos_emb[0] is not None:
                query = apply_rotary_pos_emb(
                    query, rotary_pos_emb[0], config=self.config, cu_seqlens=cu_seqlens_q,
                    mscale=attn._yarn_concentration_factor, max_seqlen=rope_max_seqlen,
                )

        core_attn_out = apply_module(attn.core_attention)(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        if thd:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        if gate is not None:
            # 输出门交给 mcore 的原生实现
            core_attn_out = attn._apply_output_gate(core_attn_out, gate)
        output, bias = attn.forward_post_core_attn(core_attn_out)
        if bias is not None:
            output = output + bias
        return output, frozen_kv

    def _block_step(
        self,
        stream: Tensor,
        prefix: Tensor,
        rows: Tensor | None,
        out_dtype,
        frozen_kv,
        attention_mask,
        rotary_pos_emb,
        attention_bias,
        packed_seq_params,
        shape,
        padding_mask,
    ):
        """一次迭代：跑注意力与 MLP 两个 sublayer，返回 ``(stream, prefix, frozen_kv)``。"""
        # 注意力：连接给出 sublayer 输入（其收尾的带权 RMSNorm 即 input_layernorm），子层输出由
        # 下一次连接调用写进 stream。
        routed = self.self_attention_attn_res(
            prefix, stream - prefix, rows, output_norm_weight=self.input_layernorm.weight
        )
        attn_out, frozen_kv = self._attention(
            # sublayer 的输入按**模型** dtype：fp32 残差流下状态是 fp32，但注意力/MLP 仍按训练精度
            routed.to(self._sublayer_dtype()).reshape(shape[0], shape[1], self.hidden_size),
            frozen_kv,
            attention_mask,
            rotary_pos_emb,
            attention_bias,
            packed_seq_params,
        )
        written = self._write_dropout(self.self_attn_bda, attn_out)
        stream = routed + written.reshape(-1, self.hidden_size)
        prefix = stream

        routed = self.mlp_attn_res(
            prefix, prefix, rows, output_norm_weight=self.pre_mlp_layernorm.weight
        )
        mlp_out = apply_module(self.mlp)(
            routed.to(self._sublayer_dtype()).reshape(shape[0], shape[1], self.hidden_size),
            padding_mask=padding_mask,
        )
        if isinstance(mlp_out, tuple):
            output, bias = mlp_out
            mlp_out = output + bias if bias is not None else output
        written = self._write_dropout(self.mlp_bda, mlp_out)
        stream = routed + written.reshape(-1, self.hidden_size)
        prefix = prefix + stream
        return stream, prefix, frozen_kv

    def _sublayer_dtype(self) -> torch.dtype:
        """sublayer 的输入精度，取**参数** dtype 而非 ``config.params_dtype``。

        fp32 残差流下状态是 fp32，而注意力/MLP 仍与权重同精度；按参数取可保证整模型被 cast
        （fp32 校验、推理侧换精度）后依然一致。
        """
        return next(self.parameters()).dtype

    def _step(self, stream: Tensor, prefix: Tensor, **common):
        """一次迭代，按需带激活重算。

        重算的切分点是整步：``selective`` 与 ``full`` 在这里坍缩为同一件事（把一次 sublayer 对
        重算一遍）；非重算那条路一字不动，训练数值逐位相同。
        """
        if not (self.training and self.recompute):
            return self._block_step(stream, prefix, **common)
        return checkpoint(self._block_step, stream, prefix, use_reentrant=False, **common)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset: Tensor | None = None,
        padding_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        mhc_recompute_manager=None,
        **kwargs,
    ):
        """解块循环，并按其身份返回打包状态（中间层）或输出读结果（末层）。

        只实现训练/评测路径：推理侧（flash decode 的 RoPE、推理上下文）直接拒绝。
        """
        if self.is_plain_layer:
            # MTP 层：整条交给 mcore 原生残差层（含它的 recompute 钩子与 MoE 路径）
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                input_ids=input_ids,
                **kwargs,
            )

        inference_context = deprecate_inference_params(
            inference_context, kwargs.get("inference_params")
        )
        if inference_context is not None:
            raise NotImplementedError("Looma layers only implement the training/eval forward path.")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise NotImplementedError("Looma does not implement the flash-decode RoPE path.")

        shape = hidden_states.shape[:2]
        out_dtype = hidden_states.dtype
        stream, prefix, rows = self._unpack(hidden_states)

        # 块的"行"就是交给它的 stream（下块收敛到的和，首块即输入本身）：行银行每块一行、第 0 行
        # 是输入，于是任何块都至少有一行可读。
        rows = self._append(rows, prefix)

        common = dict(
            rows=rows,
            out_dtype=out_dtype,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            shape=shape,
            padding_mask=padding_mask,
        )

        # 第一次迭代会被循环复用：它的 K/V（后续迭代沿用这份历史，不再重算）与输出。
        stream, prefix, frozen_kv = self._step(stream, prefix, frozen_kv=None, **common)

        def block_map(next_stream, next_prefix):
            out = self._step(next_stream, next_prefix, frozen_kv=frozen_kv, **common)
            return out[0], out[1]

        cfg = self.looma_cfg
        stream, prefix = solve_block(
            block_map,
            [stream, prefix],
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            stop_mode=cfg.stop_mode,
            tau=cfg.tau,
            grad_steps=cfg.grad_steps,
        )

        if self.is_last_layer:
            # 输出读在模型收尾归一化之前，聚合行与末块收敛到的 stream
            out = self.output_attn_res(prefix, stream, rows) if cfg.output_route else stream
            return self._owning_output(out.reshape(shape[0], shape[1], self.hidden_size)), context

        return self._owning_output(self._pack(stream, prefix, rows, shape)), context
