# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import os
import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _get_backend_spec_provider,
)
from megatron.core.models.gpt.gpt_model import GPTModel as DeepSeekModel
from megatron_ext.core.transformer.shensi.attn_res import (
    ShensiAttentionResidual,
    bind_attn_res_state,
    bind_mtp_attn_res_state,
    num_attn_res_blocks,
)
from megatron_ext.core.transformer.shensi.hyper_connection import ShensiHyperHead
from megatron_ext.core.transformer.shensi.moe import tie_moe_groups
from megatron_ext.core.transformer.shensi.transformer_config import (
    SHENSI_PENDING_FIELDS,
    ShensiTransformerConfig,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.training import print_rank_0


class ShensiModel(DeepSeekModel):
    pending_shensi_overrides = tuple(SHENSI_PENDING_FIELDS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.config, ShensiTransformerConfig):
            raise TypeError(
                f"ShensiModel 需要 ShensiTransformerConfig，得到 {type(self.config).__name__}；"
                "请用 shensi_builder / build_shensi_transformer_config 构造。"
            )
        cfg = self.config
        self.hc_mult = int(cfg.num_residual_streams)
        self.n_hash_layers = int(cfg.moe_n_hash_layers)
        self.num_blocks = num_attn_res_blocks(
            cfg.num_layers, self.n_hash_layers, int(cfg.attn_res_block_size)
        )
        self.output_attn_res = ShensiAttentionResidual(cfg)
        self.hc_head = ShensiHyperHead(cfg)
        backend = _get_backend_spec_provider(config=cfg)
        self.output_norm = build_module(
            backend.layer_norm(rms_norm=cfg.normalization == "RMSNorm", for_qk=False),
            config=cfg,
            hidden_size=cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )
        if not hasattr(self.decoder, "layers"):
            raise RuntimeError(
                "ShensiModel 需要带 layers 的 decoder（TransformerBlock）；"
                f"得到 {type(self.decoder).__name__}"
            )
        self.attn_res_stage_plan = None
        self.attn_res_handoff = None
        pp_size = int(cfg.pipeline_model_parallel_size or 1)
        vpp_size = int(getattr(cfg, "virtual_pipeline_model_parallel_size", None) or 1)
        if pp_size > 1 or vpp_size > 1:
            from megatron_ext.core.transformer.shensi.attn_res import (
                AttnResStagePlan,
                ShensiAttnResPPHandoff,
                check_attn_res_pp_support,
            )

            report = check_attn_res_pp_support(cfg)
            plan = AttnResStagePlan.from_config(cfg)
            pp_rank, vp_stage = 0, 0
            try:
                from megatron.core import parallel_state as ps

                if ps.is_initialized():
                    pp_rank = int(ps.get_pipeline_model_parallel_rank())
                    vp_stage = int(ps.get_virtual_pipeline_model_parallel_rank() or 0)
            except Exception:  # noqa: BLE001
                pass
            plan.local_stage = (pp_rank, vp_stage)
            self.attn_res_stage_plan = plan
            verbose = (
                str(os.environ.get("SHENSI_ATTN_RES_PP_VERBOSE", "0")).strip().lower()
            )
            verbose_all = str(
                os.environ.get("SHENSI_ATTN_RES_PP_VERBOSE_ALL", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
            verbose_on = verbose in ("1", "true", "yes", "on") or verbose == "all"
            if verbose_on:
                logger = print if (verbose_all or verbose == "all") else print_rank_0
            else:
                logger = None
            self.attn_res_handoff = ShensiAttnResPPHandoff(
                plan, local_stage=(pp_rank, vp_stage), logger=logger
            )
            print_rank_0(
                f"[shensi][attn_res_pp] pp={pp_size} vpp={vpp_size} "
                f"本地层={list(plan.local_layers)} 导入层={list(plan.local_import_layers())} "
                f"导出层={list(plan.local_export_layers())} 跨 stage block="
                f"{[b.index for b, _ in plan.crossing_blocks()]} ｜ "
                f"布局校验={'PASS' if report.ok else 'FAIL'}"
            )
        self.attn_res_state = bind_attn_res_state(
            self.decoder,
            cfg,
            self.num_blocks,
            handoff=self.attn_res_handoff,
            plan=self.attn_res_stage_plan,
        )
        self.mtp_attn_res_states = {}
        mtp_block = getattr(self, "mtp", None)
        if mtp_block is not None:
            self.mtp_attn_res_states = bind_mtp_attn_res_state(mtp_block, cfg)
            print_rank_0(
                f"[shensi][mtp] AttnRes 状态已绑定：{len(self.mtp_attn_res_states)} 个 MTP stage"
                f"（每 stage 单槽 num_blocks=1）| is_hash=False / block_write=True / "
                "prev_valid_blocks=0（对齐服务侧 ShensiDecoderLayer 的 MTP 语义）"
            )
        self.moe_group_owners = tie_moe_groups(
            self.decoder,
            self.n_hash_layers,
            int(cfg.attn_res_block_size),
            num_layers=int(cfg.num_layers),
        )
        if not self.moe_group_owners:
            self.moe_group_owners = {}
        self._init_shensi_only_weights()
        from megatron_ext.core.transformer.shensi.fp32_keep import (
            apply_fp32_keep,
            dtype_counts,
            fp32_keep_groups,
        )

        groups = fp32_keep_groups(cfg)
        self.hc_fp32_keep_groups = tuple(groups)
        self.hc_fp32_keep = bool(groups)
        if groups:
            self.fp32_keep_params = apply_fp32_keep(self, groups=groups, verbose=True)
            counts = ", ".join(
                f"{k}={v[0]}个/{v[1]}元素"
                for k, v in sorted(dtype_counts(self).items())
            )
            print_rank_0(
                f"[shensi][fp32keep] ON：组={','.join(groups)}，{len(self.fp32_keep_params)} 个"
                f"参数保持 fp32（HF _keep_in_fp32_modules_strict 全量覆盖）| "
                f"全模型 dtype 分布：{counts}"
            )
        else:
            self.fp32_keep_params = []
            print_rank_0(
                "[shensi][fp32keep] OFF（hc_fp32_keep=False / SHENSI_HC_FP32_KEEP=0 / "
                "SHENSI_FP32_KEEP_GROUPS 为空）：命中的参数按模型 dtype 存储（与 HF 不一致，"
                "仅对照用）"
            )

    def _init_shensi_only_weights(self) -> None:
        std = float(self.config.init_method_std)
        hidden = int(self.config.hidden_size)
        for module in self.modules():
            init = getattr(module, "init_shensi_weights", None)
            if callable(init):
                init(std=std, hidden_size=hidden)

    def _postprocess(self, hidden_states: torch.Tensor, *args, **kwargs):
        from megatron.core.inference.utils import InferenceMode
        from megatron.core.utils import make_viewless_tensor

        if not self.post_process:
            return make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
        state = self.attn_res_state
        mtp_in_postprocess = bool(kwargs.get("mtp_in_postprocess"))
        in_inference_mode = InferenceMode.is_active()
        inference_context = kwargs.get("inference_context")
        is_spec_decode = (
            in_inference_mode
            and inference_context is not None
            and inference_context.is_dynamic_batching()
            and inference_context.num_speculative_tokens > 0
        )
        if mtp_in_postprocess and not (in_inference_mode or is_spec_decode):
            if state is None or state.prefix_sum is None:
                raise RuntimeError(
                    "AttnRes 状态为空：decoder 还没有跑过 forward，无法给 MTP 供多流输入"
                )
            s, b = hidden_states.shape[0], hidden_states.shape[1]
            multistream = state.prefix_sum.reshape(s, b, -1)
            contracted = self.shensi_output_contract(hidden_states)
            contracted = self.mtp(
                input_ids=kwargs.get("input_ids"),
                position_ids=kwargs.get("position_ids"),
                hidden_states=contracted,
                mhc_multistream=multistream,
                attention_mask=kwargs.get("attention_mask"),
                inference_params=None,
                rotary_pos_emb=kwargs.get("rotary_pos_emb"),
                rotary_pos_cos=kwargs.get("rotary_pos_cos"),
                rotary_pos_sin=kwargs.get("rotary_pos_sin"),
                packed_seq_params=kwargs.get("packed_seq_params"),
                sequence_len_offset=kwargs.get("sequence_len_offset"),
                padding_mask=kwargs.get("padding_mask"),
                embedding=self.embedding,
                **(kwargs.get("extra_block_kwargs") or {}),
            )
            kwargs["mtp_in_postprocess"] = False
            return super()._postprocess(contracted, *args, **kwargs)
        hidden_states = self.shensi_output_contract(hidden_states)
        return super()._postprocess(hidden_states, *args, **kwargs)

    def shensi_output_contract(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self.attn_res_state
        if state is None or state.prefix_sum is None:
            raise RuntimeError("AttnRes 状态为空：decoder 还没有跑过 forward")
        s, b = hidden_states.shape[0], hidden_states.shape[1]
        hidden = int(self.config.hidden_size)
        if hidden_states.dim() == 4:
            streams = hidden_states
        else:
            streams = hidden_states.view(s, b, self.hc_mult, hidden)
        collapsed, _ = self.output_attn_res(
            state.prefix_sum,
            streams,
            state.residual,
            output_norm_weight=None,
            num_blocks=self.num_blocks,
        )
        return self.output_norm(self.hc_head(collapsed))

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata=None
    ):
        ssd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        self.shensi_tied_shard_audit = self._audit_tied_shards(ssd)
        return ssd

    def _audit_tied_shards(self, ssd) -> dict:
        owners = getattr(self, "moe_group_owners", None) or {}
        audit = {"groups": len(owners), "tied_groups": 0, "expected": 0, "found": 0}
        if not owners:
            return audit
        seen: dict = {}
        for key, val in ssd.items():
            data = getattr(val, "data", val)
            if torch.is_tensor(data):
                seen.setdefault(id(data), []).append(key)
        by_id: dict = {}
        for name, param in self.named_parameters(remove_duplicate=False):
            by_id.setdefault(id(param), []).append(name)
        problems = []
        for block_id, owner in sorted(owners.items()):
            sharing = list(getattr(owner, "sharing_layers", []) or [])
            if len(sharing) <= 1:
                continue
            audit["tied_groups"] += 1
            tied = [
                p
                for mod in (
                    getattr(owner, "router", None),
                    getattr(owner, "experts", None),
                )
                if mod is not None
                for p in mod.parameters()
            ]
            for param in tied:
                names = by_id.get(id(param), [])
                if len(names) != len(sharing):
                    problems.append(
                        f"block {block_id}：被共享参数只在 {len(names)} 个名字下出现"
                        f"（组内 {len(sharing)} 层）-> 共享疑似未生效"
                    )
                    continue
                keys = seen.get(id(param), [])
                audit["expected"] += len(names)
                audit["found"] += len(keys)
                if len(keys) != len(names):
                    problems.append(
                        f"block {block_id}：{len(names)} 个别名（如 {names[0]}）在 "
                        f"sharded_state_dict 里只找到 {len(keys)} 个键"
                    )
        if problems:
            raise RuntimeError(
                "[shensi][dist_ckpt] tie_moe_groups 的共享参数在 sharded_state_dict 里"
                "**别名键不齐**：这在 dist checkpoint 下会静默丢权重（缺的键不在检查点里，"
                "断点续训时不报错、只保留当前值）。明细："
                + "；".join(problems[:8])
                + "。根因通常是上游 TransformerBlock.sharded_state_dict 不再逐层递归展开"
                "（依据与实测见 ShensiModel.sharded_state_dict 的 docstring / "
                "entrypoints/_probe_dist_ckpt.py）。"
            )
        return audit
