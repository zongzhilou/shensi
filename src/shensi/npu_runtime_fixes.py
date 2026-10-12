#!/usr/bin/env python3
"""NPU 运行时补丁段：机器侧逐项 debug 出来的适配（回退计数、NPU 安全 RoPE、attn_sink
初始化、cdist Gram 形式、torch.compile 解包、显存探针……）。

来源：shensi 8 卡迁移包的 `overwrite/src/shensi/runtime.py` 追加段（2026-10-12 并入本仓）。
`runtime` 导入本模块即生效；每段自带守卫，目标符号缺失时只打 skipped 行，不误打；
文件末尾打印 patch inventory selfcheck（生效/未生效/无法判定三类）。

风格：本段是机器上验证过的 debug 代码，保留其原始形状（%-格式化 print、宽泛 except、
模块级副作用）。整段豁免 lint 风格规则；**F821（未定义名）不豁免**——那类问题会变成
运行期 NameError（如 fix27 的 `_orig96` 兜底入口，已改为调用时惰性解析）。
"""

# ruff: noqa: UP031, SIM105, B010, F401, F841, I001

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# NPU 补丁段（机器侧已验证；2026-10-12 由 shensi-migration 包并入本仓）
# -----------------------------------------------------------------------------
# 基线：shensi@eeec1d6f 之上并入（随后叠加了 mindspeed 确定性适配与分块 shim 的接线）。
# 本段逐段自带守卫，目标符号缺失时只告警不误打；是否真生效看文件末尾的 selfcheck。
# 本段是机器上逐项 debug 出来的 NPU 适配（回退计数、NPU 安全 RoPE、attn_sink 初始化、
# cdist Gram、compile 解包、显存探针……），按"NPU 相关都在 shensi 里"的约定收进本仓。
# =============================================================================
# --- appended: opt-in host-fallback counter（SHENSI_FALLBACK_COUNT=1） ---
# 背景：mcore 的 DSA/CSA 稀疏注意力（shensi 每层都走）里有两类算子会离开 AiCore：
#   - dsa_masking.py:331 `scatter_reduce_(reduce="amax")` → VariableFallbackKernel，**落到主机 CPU**（最贵）
#   - csa_utils/fused_sparse_attention.py:433 `valid_mask.int().argsort(...)`、dsa_masking.py:121
#     `sort_key.argsort(...)` → ArgSortKernelNpuOpApi，跑在 AiCpu（不跨主机，但慢）
# torch_npu 的 warning 每进程只打一次，看不出频率。这个计数器把**每步**的 Python 层调用次数打出来，
# 用来判断"值不值得改"——没有量之前不去动注意力掩码逻辑（改错会静默污染注意力）。
# 只包 Python 层入口，不碰 C++ 内部调用；默认关闭，生产方式零影响。
try:
    import os as _os_fb

    if _os_fb.environ.get("SHENSI_FALLBACK_COUNT", "0") == "1":
        import torch as _t_fb

        _fb = {"scatter_reduce_": 0, "argsort": 0, "sort": 0, "topk": 0}
        _o_sr = _t_fb.Tensor.scatter_reduce_

        def _fb_sr(self, *a, **k):
            _fb["scatter_reduce_"] += 1
            return _o_sr(self, *a, **k)

        _t_fb.Tensor.scatter_reduce_ = _fb_sr

        _o_as = _t_fb.argsort

        def _fb_as(*a, **k):
            _fb["argsort"] += 1
            return _o_as(*a, **k)

        _t_fb.argsort = _fb_as

        _o_so = _t_fb.sort

        def _fb_so(*a, **k):
            _fb["sort"] += 1
            return _o_so(*a, **k)

        _t_fb.sort = _fb_so

        try:
            _o_tp = _t_fb.topk

            def _fb_tp(*a, **k):
                _fb["topk"] += 1
                return _o_tp(*a, **k)

            _t_fb.topk = _fb_tp
        except Exception:
            pass

        import megatron.training.training as _tr_fb

        if not getattr(_tr_fb.train_step, "_shensi_fb_count", False):
            _o_ts = _tr_fb.train_step

            def _fb_ts(*a, **k):
                _before = dict(_fb)
                _r = _o_ts(*a, **k)
                _d = {n: _fb[n] - _before[n] for n in _fb}
                print("[shensi][fb] per-step python-level op calls: %s" % _d, flush=True)
                return _r

            _fb_ts._shensi_fb_count = True
            _tr_fb.train_step = _fb_ts
        print("[shensi][fb] fallback counter installed (SHENSI_FALLBACK_COUNT=1)", flush=True)
except Exception as _e_fb:
    print("[shensi][fb] counter skipped:", _e_fb, flush=True)

# --- appended: MindSpeed 需要的已删 parallel_state 符号 → 响亮 stub ---
# 背景：MindSpeed 与本 mcore revision 有约 30 处引用漂移（见 MindSpeed-PORT-NOTES.md）。
# `mindspeed.megatron_adaptor` 的注册循环能跑完，但 `apply_patches()` 会在 filesystem_async 处失败
# → 补丁**部分应用**（这个状态本身就出现于我们已验证通过的 9 步运行里，可接受）。
# 致命的是它装上的 `TransformerConfig.__post_init__` 钩子会在 config 构建时调
# `MindSpeedFeaturesManager.validate_features_args`，其中 HCP / 多模态特征会
# `from megatron.core.parallel_state import ...` 四个本 mcore 已删的符号 —— 缺一个就整个起不来。
# 这里只保证**可导入**：真被调用时大声报错（绝不返回错误的默认值）；shensi 预训练不使用这些特征。
try:
    import megatron.core.parallel_state as _ps_stub

    for _name, _msg in (
        (
            "create_hierarchical_parallel_groups",
            "hierarchical context parallel（MindSpeed HCP/多模态特征）",
        ),
        (
            "is_pipeline_stage_before_split",
            "interleaved-pipeline split helper（MindSpeed 流水调度特征）",
        ),
        (
            "is_pipeline_stage_after_split",
            "interleaved-pipeline split helper（MindSpeed 流水调度特征）",
        ),
        (
            "get_moe_layer_wise_logging_tracker",
            "MoE layer-wise logging（MindSpeed 日志特征）",
        ),
    ):
        if not hasattr(_ps_stub, _name):
            def _make(_n=_name, _m=_msg):
                def _missing(*_a, **_k):
                    raise NotImplementedError(
                        "[shensi] megatron.core.parallel_state.%s 在本 mcore revision 已删除，"
                        "而 MindSpeed 的 %s 仍会 import 它。shensi 预训练不使用该特征；"
                        "若真的走到这里，说明它被启用了，需要按 MindSpeed-PORT-NOTES.md 移植。" % (_n, _m)
                    )

                return _missing

            setattr(_ps_stub, _name, _make())
            print("[shensi] shimmed parallel_state.%s (loud stub)" % _name, flush=True)
except Exception as _e_ps_stub:
    print("[shensi] parallel_state stubs skipped:", _e_ps_stub, flush=True)

# --- appended: robust accumulate_allreduce_grads_in_fp32 fix for bf16 main grads ---
try:
    from megatron.training import arguments as _margs

    if not getattr(_margs, "_shensi_accum_fix_v2", False):
        _va = _margs.validate_args

        def _va2(args, args_defaults=None, *a, **k):
            try:
                s = str(getattr(args, "main_grads_dtype", ""))
                if ("bf16" in s or "bfloat" in s) and getattr(
                    args, "accumulate_allreduce_grads_in_fp32", False
                ):
                    args.accumulate_allreduce_grads_in_fp32 = False
                    print(
                        "[shensi] accum_allreduce_fp32 -> False (main_grads_dtype=%s)" % s,
                        flush=True,
                    )
            except Exception:
                pass
            if args_defaults is None:
                return _va(args, *a, **k)
            return _va(args, args_defaults, *a, **k)

        _margs.validate_args = _va2
        _margs._shensi_accum_fix_v2 = True
        print("[shensi] accum fix hook v2 installed", flush=True)
except Exception as _e:
    print("[shensi] accum fix hook v2 skipped:", _e, flush=True)

# --- appended: Ascend TE FusedAdam rejects store_param_remainders with mixed param dtypes ---
try:
    from megatron.core.optimizer.optimizer_config import OptimizerConfig as _OC

    if not getattr(_OC, "_shensi_sprem_off", False):
        _post = _OC.__post_init__

        def _post2(self, *a, **k):
            _post(self, *a, **k)
            try:
                self.store_param_remainders = False
            except Exception:
                pass

        _OC.__post_init__ = _post2
        _OC._shensi_sprem_off = True
        print("[shensi] OptimizerConfig.store_param_remainders -> False", flush=True)
except Exception as _e:
    print("[shensi] store_param_remainders config patch skipped:", _e, flush=True)

# --- appended: guarantee dsa.hadamard_transform on NPU (dsa.py:266 assert) ---
try:
    import sys as _sys

    import torch as _t6

    def _hm(n, device):
        m = _t6.ones(1, 1, device=device, dtype=_t6.float32)
        while m.size(0) < n:
            m = _t6.cat([_t6.cat([m, m], 1), _t6.cat([m, -m], 1)], 0)
        if m.size(0) != n:
            raise ValueError("hadamard: need power-of-two dim, got %s" % n)
        return m

    def _had(x, scale=1.0):
        return ((x.float() @ _hm(x.size(-1), x.device)) * float(scale)).to(x.dtype)

    _had._shensi_npu_hadamard = True
    _fixed = []
    _mods = []
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa0
        _mods.append(("canonical", _dsa0))
    except Exception as _e:
        print("[shensi] dsa import failed:", _e, flush=True)
    for _n, _m in list(_sys.modules.items()):
        if _n.endswith("experimental_attention_variant.dsa") and _m is not None:
            _mods.append((_n, _m))
    for _n, _m in _mods:
        try:
            _m.hadamard_transform = _had
            _fixed.append(_n)
        except Exception as _e:
            print("[shensi] hadamard set failed on", _n, _e, flush=True)
    print("[shensi] dsa hadamard installed on:", _fixed, flush=True)
except Exception as _e:
    print("[shensi] hadamard patch skipped:", _e, flush=True)

# --- appended: isolate MTP k=1 recompute window with a private config copy (supersedes the in-place version) ---
try:
    import copy as _copy

    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer as _MPL3,
    )

    if not getattr(_MPL3, "_shensi_mtp_cfgcopy", False):
        _ocpf3 = _MPL3._checkpointed_forward

        def _cpf3(self, *a, **k):
            cfg = getattr(self, "config", None)
            if cfg is None:
                return _ocpf3(self, *a, **k)
            c2 = _copy.copy(cfg)
            c2.recompute_num_layers = 1
            self.config = c2
            try:
                return _ocpf3(self, *a, **k)
            finally:
                self.config = cfg

        _MPL3._shensi_mtp_cfgcopy = True
        _MPL3._checkpointed_forward = _cpf3
        print("[shensi] MTP recompute uses a private config copy (k=1)", flush=True)
except Exception as _e:
    print("[shensi] MTP cfgcopy patch skipped:", _e, flush=True)

# --- appended: autograd anomaly detection (name the backward that makes NaN) ---
try:
    import torch as _t34

    import os as _os34

    if _os34.environ.get("SHENSI_AUTOGRAD_ANOMALY", "0") == "1":
        try:
            _t34.autograd.set_detect_anomaly(True, check_nan=True)
        except TypeError:
            _t34.autograd.set_detect_anomaly(True)
        print("[shensi] autograd anomaly detection ON (check_nan=True)", flush=True)
    else:
        print("[shensi] autograd anomaly detection OFF", flush=True)
except Exception as _e:
    print("[shensi] anomaly detection skipped:", _e, flush=True)

# --- appended: safe fp32 F.rms_norm (covers functional RMSNorm calls) ---
try:
    import torch as _t39

    import torch.nn.functional as _F39

    _orig39 = _F39.rms_norm
    if not getattr(_orig39, "_shensi_safe39", False):

        def _safe39(input, normalized_shape, weight=None, eps=None):
            try:
                x = input.float()
                d = x.shape[-1]
                z = x * _t39.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + (float(eps) if eps else 1e-6))
                if weight is not None:
                    z = z * weight.float()
                return z.to(input.dtype)
            except Exception:
                return _orig39(input, normalized_shape, weight=weight, eps=eps)

        _safe39._shensi_safe39 = True
        _F39.rms_norm = _safe39
        try:
            _t39.rms_norm = _safe39
        except Exception:
            pass
        print("[shensi] safe F.rms_norm installed (fp32 + eps)", flush=True)
except Exception as _e:
    print("[shensi] safe rms_norm skipped:", _e, flush=True)

# --- appended: REAL FIX(6) - NPU-safe MLA interleaved rotary (materialized layouts) ---
try:
    import sys as _s58

    import torch as _t58

    from megatron.core.models.common.embeddings import rope_utils as _ru58

    _orig58 = _ru58._apply_rotary_pos_emb_bshd
    if not getattr(_orig58, "_shensi_safe58", False):
        _rh58 = _ru58._rotate_half

        def _safe58(
            t,
            freqs,
            rotary_interleaved=False,
            mla_rotary_interleaved=False,
            mscale=1.0,
            inverse=False,
            mla_output_remove_interleaving=False,
            multi_latent_attention=None,
        ):
            if not mla_rotary_interleaved:
                return _orig58(
                    t,
                    freqs,
                    rotary_interleaved=rotary_interleaved,
                    mla_rotary_interleaved=mla_rotary_interleaved,
                    mscale=mscale,
                    inverse=inverse,
                    mla_output_remove_interleaving=mla_output_remove_interleaving,
                    multi_latent_attention=multi_latent_attention,
                )
            if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
                freqs = freqs.squeeze(-2)
            rot_dim = freqs.shape[-1]
            core = t[..., :rot_dim].contiguous()
            t_pass = t[..., rot_dim:]
            x1 = core[..., 0::2].contiguous()
            x2 = core[..., 1::2].contiguous()
            tt = torch.cat((x1, x2), dim=-1).contiguous()
            cos_ = (torch.cos(freqs) * mscale).to(tt.dtype).contiguous()
            sin_ = (torch.sin(freqs) * mscale).to(tt.dtype).contiguous()
            if inverse:
                sin_ = -sin_
            out = (tt * cos_) + (_rh58(tt, rotary_interleaved).contiguous() * sin_)
            if mla_output_remove_interleaving:
                a, b = torch.chunk(out.contiguous(), 2, dim=-1)
                out = torch.stack((a.contiguous(), b.contiguous()), dim=-1).flatten(start_dim=-2)
            if t_pass.numel():
                out = torch.cat((out.contiguous(), t_pass), dim=-1)
            return out

        _safe58._shensi_safe58 = True
        _ru58._apply_rotary_pos_emb_bshd = _safe58
        try:
            from megatron.core.transformer.experimental_attention_variant import csa as _csa58

            if hasattr(_csa58, "_apply_rotary_pos_emb_bshd"):
                _csa58._apply_rotary_pos_emb_bshd = _safe58
        except Exception:
            pass
        print("[shensi] safe58 MLA interleaved rope installed", flush=True)
except Exception as _e:
    print("[shensi] safe58 skipped:", _e, flush=True)

# --- appended: crash visibility (faulthandler + excepthook) ---
try:
    import faulthandler as _fh62
    import sys as _sys62
    import traceback as _tb62

    _fh62.enable(all_threads=True)

    def _hook62(exc_type, exc, tbv):
        try:
            _sys62.stderr.write("\n[shensi][exc] %s: %s\n" % (exc_type.__name__, exc))
            _tb62.print_exception(exc_type, exc, tbv)
            _sys62.stderr.flush()
        except Exception:
            pass

    _sys62.excepthook = _hook62
    print("[shensi] crash visibility on", flush=True)
except Exception as _e:
    print("[shensi] crash visibility skipped:", _e, flush=True)

# --- appended: module-level torch for previously appended helpers ---
try:
    import torch  # noqa: F401
except Exception:
    pass

# --- appended: REAL FIX(26) - attn_sink left at -inf by mcore init -> finite init at model build ---
try:
    import torch as _t95

    from megatron.bridge.models.shensi import modeling_shensi as _ms95

    if not getattr(_ms95.ShensiModel, "_shensi_sink95", False):
        _i95 = _ms95.ShensiModel.__init__

        def _i95b(self, *a, **k):
            _i95(self, *a, **k)
            fixed = []
            for nm, p in self.named_parameters():
                if "attn_sink" not in nm:
                    continue
                with _t95.no_grad():
                    if not bool(_t95.isfinite(p).all()):
                        p.fill_(0.0)
                        fixed.append(nm)
            print("[shensi] fix26 attn_sink finite init, fixed %d: %s" % (len(fixed), fixed[:3]), flush=True)

        _ms95.ShensiModel.__init__ = _i95b
        _ms95.ShensiModel._shensi_sink95 = True
        print("[shensi] fix26 hook installed", flush=True)
except Exception as _e:
    print("[shensi] fix26 skipped:", _e, flush=True)

# --- appended: REAL FIX(27) - NPU-safe interleaved RoPE on EVERY module holding the symbol ---
try:
    import sys as _s96

    import torch as _t96

    def _orig96_resolved():
        """非 MLA 路径的兜底原实现：调用时再解析（runtime 导入早于 mcore 时，
        导入期扫描可能一个都遇不到——原写法只在 globals() 里留缓存，会 NameError）。"""
        f = globals().get("_orig96")
        if f is not None:
            return f
        for _m in list(_s96.modules.values()):
            g = getattr(_m, "apply_rotary_pos_emb", None) if _m is not None else None
            if g is not None and not getattr(g, "_shensi_safe96", False):
                globals()["_orig96"] = g
                return g
        raise RuntimeError("fix27: 找不到非 MLA 路径的原 apply_rotary_pos_emb 实现")

    def _rh96(x):
        d = x.shape[-1]
        return _t96.cat((-x[..., d // 2 :].contiguous(), x[..., : d // 2].contiguous()), dim=-1).contiguous()

    def _safe96(t, freqs, config=None, cu_seqlens=None, mscale=1.0, cp_group=None,
                rotary_interleaved=False, mla_rotary_interleaved=False, inverse=False,
                mla_output_remove_interleaving=False, multi_latent_attention=None,
                max_seqlen=None, position_ids=None):
        if not mla_rotary_interleaved:
            import inspect as _insp96

            _kw96 = dict(
                config=config,
                cu_seqlens=cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
                rotary_interleaved=rotary_interleaved,
                mla_rotary_interleaved=mla_rotary_interleaved,
                inverse=inverse,
                mla_output_remove_interleaving=mla_output_remove_interleaving,
                multi_latent_attention=multi_latent_attention,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
            )
            _f96 = _orig96_resolved()
            try:
                return _f96(t, freqs, **_kw96)
            except TypeError:
                _ok96 = set(_insp96.signature(_f96).parameters)
                return _f96(t, freqs, **{k: v for k, v in _kw96.items() if k in _ok96})
        if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
            freqs = freqs.squeeze(-2)
        rd = freqs.shape[-1]
        core = t[..., :rd].contiguous()
        t_pass = t[..., rd:]
        tt = _t96.cat((core[..., 0::2].contiguous(), core[..., 1::2].contiguous()), dim=-1).contiguous()
        cos_ = (_t96.cos(freqs) * mscale).to(tt.dtype).contiguous()
        sin_ = (_t96.sin(freqs) * mscale).to(tt.dtype).contiguous()
        if inverse:
            sin_ = -sin_
        out = (tt * cos_) + (_rh96(tt) * sin_)
        if mla_output_remove_interleaving:
            a, b = _t96.chunk(out.contiguous(), 2, dim=-1)
            out = _t96.stack((a.contiguous(), b.contiguous()), dim=-1).flatten(start_dim=-2)
        if t_pass.numel():
            out = _t96.cat((out.contiguous(), t_pass), dim=-1)
        return out

    _done96 = []
    for _n, _m in list(_s96.modules.items()):
        if _m is None:
            continue
        f = getattr(_m, "apply_rotary_pos_emb", None)
        if f is None or getattr(f, "_shensi_safe96", False):
            continue
        if "_orig96" not in globals():
            globals()["_orig96"] = f          # 用最先遇到的原实现做兜底入口
        _safe96._shensi_safe96 = True
        setattr(_m, "apply_rotary_pos_emb", _safe96)
        _done96.append(_n)
    print("[shensi] fix27 safe interleaved RoPE on:", _done96, flush=True)
except Exception as _e:
    print("[shensi] fix27 skipped:", _e, flush=True)

# --- appended: FIX(27b) - lazily re-apply safe interleaved RoPE at model init ---
try:
    import sys as _s97

    from megatron.bridge.models.shensi import modeling_shensi as _ms97

    if not getattr(_ms97.ShensiModel, "_shensi_rope97", False):

        def _apply97():
            done = []
            for _n, _m in list(_s97.modules.items()):
                if _m is None:
                    continue
                f = getattr(_m, "apply_rotary_pos_emb", None)
                if f is None or getattr(f, "_shensi_safe96", False):
                    continue
                try:
                    setattr(_m, "apply_rotary_pos_emb", _safe96)
                    done.append(_n)
                except Exception:
                    pass
            print("[shensi] fix27b applied on:", done, flush=True)

        _i97 = _ms97.ShensiModel.__init__

        def _i97b(self, *a, **k):
            _i97(self, *a, **k)
            try:
                _apply97()
            except Exception as _e:
                print("[shensi] fix27b failed:", _e, flush=True)

        _ms97.ShensiModel.__init__ = _i97b
        _ms97.ShensiModel._shensi_rope97 = True
        print("[shensi] fix27b hook installed", flush=True)
except Exception as _e:
    print("[shensi] fix27b skipped:", _e, flush=True)

# --- appended: FIX(28) - point-patch the by-value imported rope in the DSv4 attention module ---
try:
    from megatron.bridge.models.shensi import modeling_shensi as _ms98

    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as _d4_98,
    )

    if not getattr(_ms98.ShensiModel, "_shensi_rope98", False):
        _d4_98.apply_rotary_pos_emb = _safe96
        _i98 = _ms98.ShensiModel.__init__

        def _i98b(self, *a, **k):
            _i98(self, *a, **k)
            _d4_98.apply_rotary_pos_emb = _safe96
            print("[shensi] fix28 DSv4 attention rope -> safe96", flush=True)

        _ms98.ShensiModel.__init__ = _i98b
        _ms98.ShensiModel._shensi_rope98 = True
        print("[shensi] fix28 hook installed", flush=True)
except Exception as _e:
    print("[shensi] fix28 skipped:", _e, flush=True)

# --- appended: ROOT FIX - strided-free equivalent of MLA interleaved RoPE (queue-safe) ---
try:
    import torch as _tF
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as _d4F,
    )

    def _rope_safeF(x, freqs, mscale=1.0, inverse=False, remove_interleaving=True):
        if freqs.dim() == x.dim() + 1 and freqs.size(-2) == 1:
            freqs = freqs.squeeze(-2)
        rd = freqs.shape[-1]
        core = x[..., :rd].contiguous()
        t_pass = x[..., rd:]
        # interleave -> (..., rd/2, 2) -> flip the pair -> contiguous (no strided slicing)
        pair = core.reshape(*core.shape[:-1], rd // 2, 2).contiguous()
        a = pair[..., 0].contiguous()
        b = pair[..., 1].contiguous()
        tt = _tF.cat((a, b), dim=-1).contiguous()
        cos_ = (_tF.cos(freqs) * mscale).to(tt.dtype).contiguous()
        sin_ = (_tF.sin(freqs) * mscale).to(tt.dtype).contiguous()
        if inverse:
            sin_ = -sin_
        d = tt.shape[-1]
        x1 = tt[..., : d // 2].contiguous()
        x2 = tt[..., d // 2 :].contiguous()
        rot = _tF.cat((-x2, x1), dim=-1).contiguous()
        out = (tt * cos_) + (rot * sin_)
        if remove_interleaving:
            p2 = out.reshape(*out.shape[:-1], d // 2, 2).contiguous()
            out = _tF.stack((p2[..., 0], p2[..., 1]), dim=-1).flatten(start_dim=-2).contiguous()
        if t_pass.numel():
            out = _tF.cat((out, t_pass.contiguous()), dim=-1)
        return out

    _nF = [0]
    def _fF2(t, freqs, *a, **k):
        _nF[0] += 1
        return _rope_safeF(t, freqs, *a, **k)
    _d4F.apply_rotary_pos_emb = _fF2
    print("[shensi] ROOT FIX strided-free rope installed", flush=True)
except Exception as _e:
    print("[shensi] root-fix rope skipped:", _e, flush=True)

# --- appended: ROOT FIX v2 - wrap ALL rope symbols on ALL modules (lazy, by-value safe) ---
try:
    import sys as _sG
    import torch as _tG
    from megatron.bridge.models.shensi import modeling_shensi as _msG

    _nG = [0]

    def _safeG(t, freqs, *a, **k):
        if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
            freqs = freqs.squeeze(-2)
        rd = freqs.shape[-1]
        core = t[..., :rd].contiguous()
        t_pass = t[..., rd:]
        pair = core.reshape(*core.shape[:-1], rd // 2, 2).contiguous()
        tt = _tG.cat((pair[..., 0].contiguous(), pair[..., 1].contiguous()), dim=-1).contiguous()
        inv = k.get("inverse", False)
        ms = float(k.get("mscale", 1.0) or 1.0)
        cos_ = (_tG.cos(freqs) * ms).to(tt.dtype).contiguous()
        sin_ = (_tG.sin(freqs) * ms).to(tt.dtype).contiguous()
        if inv:
            sin_ = -sin_
        d = tt.shape[-1]
        rot = _tG.cat((-tt[..., d // 2 :].contiguous(), tt[..., : d // 2].contiguous()), dim=-1).contiguous()
        out = (tt * cos_) + (rot * sin_)
        if k.get("mla_output_remove_interleaving", True):
            p2 = out.reshape(*out.shape[:-1], d // 2, 2).contiguous()
            out = _tG.stack((p2[..., 0], p2[..., 1]), dim=-1).flatten(start_dim=-2).contiguous()
        if t_pass.numel():
            out = _tG.cat((out, t_pass.contiguous()), dim=-1)
        return out

    def _wrapG(f, tag):
        if f is None or getattr(f, "_shensi_safeG", False):
            return None

        def _g(*a, **kk):
            if __import__("os").environ.get("SHENSI_HANG_PROBE", "0") != "1":
                return _safeG(*a, **kk)
            _nG[0] += 1
            i = _nG[0]
            if i <= 40:
                print("[asG] rope %s enter #%d" % (tag, i), flush=True)
            r = _safeG(*a, **kk)
            try:
                _tG.npu.synchronize()
            except Exception:
                pass
            if i <= 40:
                print("[asG] rope %s exit  #%d" % (tag, i), flush=True)
            return r

        _g._shensi_safeG = True
        return _g

    _NAMES = ("apply_rotary_pos_emb", "_apply_rotary_pos_emb_bshd", "_apply_rotary_pos_emb_thd")

    def _applyG():
        done = []
        for _n, _m in list(_sG.modules.items()):
            if _m is None:
                continue
            for _nm in _NAMES:
                w = _wrapG(getattr(_m, _nm, None), "%s@%s" % (_nm.split("_")[-1], _n.split(".")[-1]))
                if w is not None:
                    setattr(_m, _nm, w)
                    done.append("%s@%s" % (_nm, _n))
        print("[shensi] rootG wrapped:", done, flush=True)

    if not getattr(_msG.ShensiModel, "_shensi_rootG", False):
        _iG = _msG.ShensiModel.__init__

        def _iGb(self, *a, **k):
            _iG(self, *a, **k)
            try:
                _applyG()
            except Exception as _e:
                print("[shensi] rootG apply failed:", _e, flush=True)

        _msG.ShensiModel.__init__ = _iGb
        _msG.ShensiModel._shensi_rootG = True
        print("[shensi] rootG hook installed", flush=True)
except Exception as _e:
    print("[shensi] rootG skipped:", _e, flush=True)


def _force_unfused_rope_provider_dict() -> None:
    """杠杆 1：把 shensi provider 的 MLA RoPE 融合硬默认压成 False。
    刻意不做设备探测：transfer_to_npu 会把 torch.cuda.is_available() 重写成 True。"""
    import os
    import torch
    print("[shensi-npu] probe cuda_avail=%s cuda_ver=%s has_npu=%s" % (
        torch.cuda.is_available(), getattr(torch.version, "cuda", None),
        hasattr(torch, "npu")), flush=True)
    if os.environ.get("SHENSI_MLA_ROPE_FUSION", "0") == "1":
        print("[shensi-npu] SHENSI_MLA_ROPE_FUSION=1 -> keep fused kernel", flush=True)
        return
    try:
        import megatron.bridge.models.shensi.shensi_provider as _sp
        d = getattr(_sp, "SHENSI_HARD_DEFAULTS", None)
        if isinstance(d, dict):
            d["apply_rope_fusion"] = False
            print("[shensi-npu] SHENSI_HARD_DEFAULTS['apply_rope_fusion']=False", flush=True)
    except Exception as exc:
        logger.warning("shensi runtime: rope-fusion defaults patch skipped (%s)", exc)


try:
    _force_unfused_rope_provider_dict()
except Exception as _e_rf1:  # noqa: BLE001 - import-time 安全：单段失败不能拖垮整个 runtime 导入
    print("[shensi] rope-fusion provider patch call skipped:", _e_rf1, flush=True)


def _force_unfused_rope_post_init_v2() -> None:
    """杠杆 2：任何路径建出的 MLA 配置，post_init 后一律压成 False，并打印一次生效值。"""
    import os
    if os.environ.get("SHENSI_MLA_ROPE_FUSION", "0") == "1":
        return
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        _o = TransformerConfig.__post_init__
        if not getattr(_o, "_shensi_unfused_rope_v2", False):
            _seen = {"printed": False}
            def _p(self, *a, **k):
                _o(self, *a, **k)
                if getattr(self, "multi_latent_attention", False):
                    self.apply_rope_fusion = False
                    if not _seen["printed"]:
                        _seen["printed"] = True
                        print("[shensi-npu] effective MLA apply_rope_fusion=%s"
                              % self.apply_rope_fusion, flush=True)
            _p._shensi_unfused_rope_v2 = True
            TransformerConfig.__post_init__ = _p
    except Exception as exc:
        logger.warning("shensi runtime: rope-fusion post_init patch skipped (%s)", exc)


try:
    _force_unfused_rope_post_init_v2()
except Exception as _e_rf2:  # noqa: BLE001
    print("[shensi] rope-fusion post_init call skipped:", _e_rf2, flush=True)


def _allow_checkpoint_globals() -> None:
    """Allow the non-tensor globals our configs pickle into checkpoints.

    torch.load defaults to weights_only=True (PyTorch >= 2.6) and rejects any
    global that is not whitelisted. Our model config stores activation_func =
    torch.nn.functional.silu, so loading a checkpoint fails with
    "Unsupported global: GLOBAL torch.nn.functional.silu". Register the usual
    activations here; set SHENSI_CKPT_WEIGHTS_ONLY=0 to fall back to
    weights_only=False for our own (trusted) checkpoints.
    """
    import os

    import torch

    try:
        import torch.nn.functional as _F

        _fn = [
            _F.silu, _F.gelu, _F.relu, _F.sigmoid, _F.tanh, _F.softplus,
            getattr(_F, "squared_relu", _F.relu),
        ]
        _cls = [torch.nn.SiLU, torch.nn.GELU, torch.nn.ReLU, torch.nn.Sigmoid,
                torch.nn.Tanh, torch.nn.Softplus]
        torch.serialization.add_safe_globals(_fn + _cls)
        print("[shensi-npu] checkpoint safe globals registered", flush=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shensi runtime: safe globals registration skipped (%s)", exc)

    if os.environ.get("SHENSI_CKPT_WEIGHTS_ONLY", "1") != "1":
        _orig_load = torch.load
        if not getattr(_orig_load, "_shensi_w_only", False):

            def _load(*a, **k):
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)

            _load._shensi_w_only = True
            torch.load = _load
            print("[shensi-npu] torch.load -> weights_only=False", flush=True)


try:
    _allow_checkpoint_globals()
except Exception as _e_ck:  # noqa: BLE001
    print("[shensi] checkpoint safe-globals call skipped:", _e_ck, flush=True)


# --- appended: stable torch.cdist (Gram form, clamp 1e-12) for NPU ---
# 承重补丁：配方 erc.py 的 erc_loss_func() 用 torch.cdist(R, R, p=2) 算 ERC 距离，
# 而 NPU 上 cdist 的**反向**在距离为 0 处返回 NaN（CdistBackward0 returned nan）→
# 第一步就 grad norm=nan、第三步 loss=nan。Gram 形式 + clamp_min(1e-12) 让梯度有限。
# 注：重建包时这段一度被误删（当时只 grep 了建模代码、没查配方），症状是冒烟在 iteration 1
# 就 grad norm=nan。
try:
    import torch as _t42

    _orig42 = _t42.cdist
    if not getattr(_orig42, "_shensi_cdist42", False):

        def _cdist42(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
            if float(p) != 2.0:
                return _orig42(x1, x2, p=p, compute_mode=compute_mode)
            a = x1.float()
            b = x2.float()
            a2 = (a * a).sum(dim=-1, keepdim=True)
            b2 = (b * b).sum(dim=-1).unsqueeze(-2)
            d2 = a2 + b2 - 2.0 * _t42.matmul(a, b.transpose(-1, -2))
            return d2.clamp_min(1e-12).sqrt().to(x1.dtype)

        _cdist42._shensi_cdist42 = True
        _t42.cdist = _cdist42
        print("[shensi] stable torch.cdist (Gram, clamp 1e-12) installed", flush=True)
except Exception as _e:
    print("[shensi] cdist patch skipped:", _e, flush=True)

# ---------------------------------------------------------------------------
# 启动自检：本段**不做任何补丁**，只把"上面各段是否真的生效"探一遍。
# 之所以放在文件**末尾**调用：前面的段（含 run-config / attn-res / jit-fuser /
# ERC / hash-ids / torch.compile）都在这行之前执行完，这里读到的才是真实状态。
# 探测法是"看补丁后的对象上有没有留标记"——探测不到就是那段打印过 skipped。
# ---------------------------------------------------------------------------
def _print_active_patch_inventory() -> None:
    import os
    import sys as _sys_inv

    def probe(module, attr_path, flag):
        """True=已生效 / False=未生效 / None=模块没加载或属性不存在（无从判断）

        只看 sys.modules：**绝不触发新导入**。补丁生效的前提就是它已经 import 过目标模块，
        所以“没在 sys.modules 里”本身就等于“这条补丁没装”；同时也避免在非 NPU 机器上
        因为这个自检去把配方包整条导入链拉起来（那条链会绕回 shensi.runtime 自身）。
        """
        try:
            obj = _sys_inv.modules.get(module)
            if obj is None:
                return None
            if attr_path:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
            return bool(getattr(obj, flag, False))
        except Exception:
            return None

    # (显示名, 模块, 属性路径, 标记)
    probes = [
        ("accum-allreduce-guard", "megatron.training.arguments", "", "_shensi_accum_fix_v2"),
        ("store-param-remainders=false", "megatron.core.optimizer.optimizer_config", "OptimizerConfig", "_shensi_sprem_off"),
        ("mtp-k1-window", "megatron.core.transformer.multi_token_prediction", "MultiTokenPredictionLayer", "_shensi_mtp_cfgcopy"),
        ("rms-norm-safe", "torch.nn.functional", "rms_norm", "_shensi_safe39"),
        ("attn-sink-finite-init", "megatron.bridge.models.shensi.modeling_shensi", "ShensiModel", "_shensi_sink95"),
        ("rope-symbol-wrappers", "megatron.bridge.models.shensi.modeling_shensi", "ShensiModel", "_shensi_rope97"),
        ("rope-rootG-override", "megatron.bridge.models.shensi.modeling_shensi", "ShensiModel", "_shensi_rootG"),
        ("set-run-config-idempotent", "megatron.training.global_vars", "_ensure_var_is_not_initialized", "_shensi_run_config_tolerant"),
        ("attn-res-fast-off-on-npu", "shensi.recipes.shensi.common.train.attn_res_fast", "apply_from_args", "_shensi_npu_safe"),
        ("erc-geom-mirror", "shensi.recipes.shensi.common.train.args", "postprocess_args", "_shensi_erc_mirror"),
        ("hash-ids-persist", "megatron.bridge.models.shensi.modeling_shensi", "HASH_INPUT_IDS", "_shensi_persist"),
        ("torch-compile-unwrapped", "torch", "compile", "_shensi_identity"),
        ("cdist-gram-safe", "torch", "cdist", "_shensi_cdist42"),
    ]
    live, dead, unknown = [], [], []
    for name, module, attr_path, flag in probes:
        r = probe(module, attr_path, flag)
        (live if r else (unknown if r is None else dead)).append(name)

    # MLA RoPE 融合：不是函数标记，而是 provider 的硬默认字典被改成 False
    try:
        from megatron.bridge.models.shensi import shensi_provider as _sp
        if _sp.SHENSI_HARD_DEFAULTS.get("apply_rope_fusion") is False:
            live.append("mla-rope-fusion-off")
        else:
            dead.append("mla-rope-fusion-off")
    except Exception:
        unknown.append("mla-rope-fusion-off")

    declared = [
        "cumulative(fp32 allreduce)", "autograd-anomaly(opt-in)", "crash-visibility",
        "checkpoint-safe-globals", "rope-strided-free", "mem-probe(opt-in)",
    ]
    print("[shensi-runtime] selfcheck 生效 %d 项: %s" % (len(live), ", ".join(live)), flush=True)
    if dead:
        print("[shensi-runtime] selfcheck 未生效 %d 项: %s  <-- 上面应有对应的 skipped 行，"
              "按需改用 CLI --set 兜底" % (len(dead), ", ".join(dead)), flush=True)
    if unknown:
        print("[shensi-runtime] selfcheck 无法判定 %d 项: %s（模块尚未导入/结构不同）"
              % (len(unknown), ", ".join(unknown)), flush=True)
    print("[shensi-runtime] 其余（无标记可探，以各自 [shensi] 行为准）: %s"
          % ", ".join(declared), flush=True)

    switches = {
        "SHENSI_MLA_ROPE_FUSION": "1 keeps the fused MLA RoPE (default off on NPU)",
        "SHENSI_ATTN_RES_FAST": "1 keeps the AttnRes fast path (default off on NPU)",
        "SHENSI_AUTOGRAD_ANOMALY": "1 enables autograd anomaly detection",
        "SHENSI_STORE_REMAINDERS": "0 keeps store_param_remainders forced off (default)",
        "SHENSI_CKPT_WEIGHTS_ONLY": "0 loads trusted checkpoints with weights_only=False",
        "SHENSI_MEM_PROBE": "1 totals+snapshot / 3 per-layer memory map (opt-in)",
        "SHENSI_HANG_PROBE": "1 traces rope calls to locate a hang",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True is required on this box",
    }
    for key, note in switches.items():
        if key in os.environ:
            print("[shensi-runtime] env %s=%s (%s)" % (key, os.environ[key], note), flush=True)


# --- appended: tolerate double registration of the run config (recipe registers it, mcore's pretrain() registers it again) ---
# The recipe entry (train_shensi.py) calls set_run_config() before pretrain(), and this mcore revision
# also calls it inside pretrain(); both pass the same container, so the second registration is harmless
# but the "already initialized" assert fires. We relax that single check instead of editing upstream,
# and do it on the shared helper (looked up at call time) so the recipe's by-value import is covered too.
try:
    from megatron.training import global_vars as _gv_run_cfg
    _orig_ensure_run_cfg = _gv_run_cfg._ensure_var_is_not_initialized
    if not getattr(_orig_ensure_run_cfg, "_shensi_run_config_tolerant", False):
        def _ensure_var_is_not_initialized_tolerant(var, name):
            if name == "run config":
                return
            return _orig_ensure_run_cfg(var, name)
        _ensure_var_is_not_initialized_tolerant._shensi_run_config_tolerant = True
        _gv_run_cfg._ensure_var_is_not_initialized = _ensure_var_is_not_initialized_tolerant
        print("[shensi] run-config double registration tolerated (recipe + pretrain)", flush=True)
except Exception as _e_run_cfg:
    print("[shensi] run-config idempotency patch skipped:", _e_run_cfg, flush=True)


# --- appended: force the AttnRes read-path torch.compile off on NPU ---
# attn_res_fast.enable_attn_res_fast_path() gates on torch.cuda.is_available(), which
# torch_npu.contrib.transfer_to_npu rewrites to True on Ascend. So the intended "no CUDA -> skip"
# guard never fires, the CLI default ("compile") takes effect, and torch.compile runs on NPU where
# torch_npu's inductor codegen dies (KeyError in torch_npu/_inductor/codegen/ir.py detect_flattened_axis).
# We force the flag to "none" on NPU here instead of editing the recipe; SHENSI_ATTN_RES_FAST=1 opts out
# (e.g. when trying CANN graph mode / a newer torch_npu).
try:
    import os as _os_arf

    def _shensi_has_npu():
        try:
            import torch as _t_arf
        except Exception:
            return False
        if not hasattr(_t_arf, "npu"):
            return False
        try:
            return bool(_t_arf.npu.is_available())
        except Exception:
            return True

    if _shensi_has_npu() and _os_arf.environ.get("SHENSI_ATTN_RES_FAST", "0") != "1":
        from shensi.recipes.shensi.common.train import attn_res_fast as _arf
        _orig_arf_apply = _arf.apply_from_args
        if not getattr(_orig_arf_apply, "_shensi_npu_safe", False):
            def _apply_from_args_npu_safe(model, args):
                cur = getattr(args, "shensi_attn_res_fast_path", "none") or "none"
                if cur != "none":
                    try:
                        setattr(args, "shensi_attn_res_fast_path", "none")
                    except Exception:
                        pass
                    print(
                        "[shensi] attn-res read path: torch.compile forced OFF (was %r) -- "
                        "torch_npu inductor codegen crashes on this graph; CANN graph mode is the "
                        "NPU equivalent (set SHENSI_ATTN_RES_FAST=1 to try anyway)" % cur,
                        flush=True,
                    )
                return _orig_arf_apply(model, args)

            _apply_from_args_npu_safe._shensi_npu_safe = True
            _arf.apply_from_args = _apply_from_args_npu_safe
            print("[shensi] attn_res_fast.apply_from_args wrapped (NPU-safe)", flush=True)
except Exception as _e_arf:
    print("[shensi] attn-res fast path NPU guard skipped:", _e_arf, flush=True)


# --- appended: unwrap mcore's jit_fuser (== torch.compile on torch>=2.2) on NPU ---
# megatron/core/jit.py does `jit_fuser = torch.jit.script` and then enable_jit_fuser() sets it to
# torch.compile for torch>=2.2, decorating swiglu/clamped_swiglu/situ_glu/... at *import* time.
# torch_npu's inductor codegen cannot compile those graphs (KeyError inside
# torch_npu/_inductor/codegen/ir.py detect_flattened_axis), which kills the first forward.
# mcore has the official switch (--disable-jit-fuser -> jit.disable_jit_fuser()), but it only affects
# modules imported afterwards, so here we also rebind the functions that were already wrapped
# (torch.compile keeps the original on `_torchdynamo_orig_callable`, older wrappers on `__wrapped__`)
# and rebind by-value imported copies too. SHENSI_JIT_FUSER=1 keeps the compiled versions.
try:
    _jf_probe = globals().get("_shensi_has_npu")
    _jf_is_npu = bool(_jf_probe()) if callable(_jf_probe) else False
    import os as _os_jf
    if _jf_is_npu and _os_jf.environ.get("SHENSI_JIT_FUSER", "0") != "1":
        import importlib as _il_jf
        import sys as _sys_jf
        import megatron.core.jit as _mjit_jf

        _mjit_jf.disable_jit_fuser()  # modules imported from now on get the no-op decorator
        _JF_MODULES = (
            "megatron.core.activations",
            "megatron.core.fusions.fused_bias_swiglu",
            "megatron.core.fusions.fused_bias_geglu",
            "megatron.core.fusions.fused_bias_gelu",
            "megatron.core.fusions.fused_bias_dropout",
            "megatron.core.fusions.fused_cross_entropy",
            "megatron.core.fusions.fused_pad_routing_map",
            "megatron.core.fusions.fused_weighted_squared_relu",
            "megatron.core.transformer.attention",
            "megatron.core.transformer.torch_norm",
            "megatron.core.transformer.utils",
            "megatron.core.transformer.moe.router",
            "megatron.core.transformer.moe.token_dispatcher",
            "megatron.core.ssm.gated_delta_net.common",
            "megatron.core.ssm.gated_delta_net.gdn",
            "megatron.core.ssm.gated_delta_net.gdn2",
        )
        _jf_pairs = []
        _jf_dynamo = _jf_wrapped = 0
        for _jf_modname in _JF_MODULES:
            try:
                _jf_mod = _il_jf.import_module(_jf_modname)
            except Exception:
                continue
            for _jf_attr in list(vars(_jf_mod)):
                _jf_obj = getattr(_jf_mod, _jf_attr, None)
                if not callable(_jf_obj):
                    continue
                _jf_orig = getattr(_jf_obj, "_torchdynamo_orig_callable", None)
                if callable(_jf_orig) and _jf_orig is not _jf_obj:
                    _jf_dynamo += 1
                else:
                    _jf_orig = None
                    _w = getattr(_jf_obj, "__wrapped__", None)
                    if callable(_w) and getattr(_w, "__module__", None) == getattr(_jf_obj, "__module__", None):
                        _jf_orig, _jf_wrapped = _w, _jf_wrapped + 1
                if _jf_orig is None:
                    continue
                try:
                    setattr(_jf_mod, _jf_attr, _jf_orig)
                    _jf_pairs.append((_jf_obj, _jf_orig))
                except Exception:
                    pass
        if _jf_pairs:
            # by-value imports keep the old (compiled) object, so rebind those copies as well
            for _jf_mod in list(_sys_jf.modules.values()):
                _d = getattr(_jf_mod, "__dict__", None)
                if not _d:
                    continue
                for _jf_attr, _jf_obj in list(_d.items()):
                    for _jf_w, _jf_o in _jf_pairs:
                        if _jf_obj is _jf_w:
                            try:
                                setattr(_jf_mod, _jf_attr, _jf_o)
                            except Exception:
                                pass
        print(
            "[shensi] jit_fuser unwrapped on NPU: %d functions (dynamo=%d, wrapped=%d) -- "
            "torch.compile is unusable on this box; SHENSI_JIT_FUSER=1 keeps it"
            % (len(_jf_pairs), _jf_dynamo, _jf_wrapped),
            flush=True,
        )
except Exception as _e_jf:
    print("[shensi] jit_fuser unwrap skipped:", _e_jf, flush=True)


# --- appended: let the geometry parity gate see the ERC knobs ---
# args.SHENSI_HF_PARITY_FIELDS pairs the HF field "erc_loss_alpha"/"erc_loss_coef" with an args attribute
# of the same name, but the CLI flag (args.py:272-273) and every profile (default/tiny/...) set
# `shensi_erc_loss_alpha/coef` instead. So getattr(args, "erc_loss_alpha") is always None, the gate reads
# 0 and aborts against the HF json (0.5 / 1.0) on every run that passes --shensi-hf-config -- including the
# official default.yaml/scale8/thru. Mirror the values just before postprocess_args so the gate compares the
# intended numbers instead of being skipped. (train_shensi calls shensi_args.postprocess_args by attribute,
# so patching the module attribute is enough.)
try:
    from shensi.recipes.shensi.common.train import args as _sargs_erc

    _orig_postprocess_erc = _sargs_erc.postprocess_args
    if not getattr(_orig_postprocess_erc, "_shensi_erc_mirror", False):
        def _postprocess_args_erc_mirror(args):
            for _erc_src, _erc_dst in (
                ("shensi_erc_loss_alpha", "erc_loss_alpha"),
                ("shensi_erc_loss_coef", "erc_loss_coef"),
            ):
                _erc_val = getattr(args, _erc_src, None)
                if _erc_val is not None and getattr(args, _erc_dst, None) is None:
                    try:
                        setattr(args, _erc_dst, _erc_val)
                    except Exception:
                        pass
            return _orig_postprocess_erc(args)

        _postprocess_args_erc_mirror._shensi_erc_mirror = True
        _sargs_erc.postprocess_args = _postprocess_args_erc_mirror
        print("[shensi] erc_loss_alpha/coef mirrored for the geometry gate", flush=True)
except Exception as _e_erc:
    print("[shensi] erc mirror patch skipped:", _e_erc, flush=True)


# --- appended: keep the published hash-MoE input_ids alive for the recompute replay ---
# ShensiTransformerLayer reads the token ids from the module-level contextvar HASH_INPUT_IDS
# (modeling_shensi.py:2343) because ShensiHashMLP is not mcore's hash-routed MoE (it has no
# mlp.router), so mcore's recompute kwargs path (recompute.py:128, gated on router.is_hash_layer)
# never fires. ShensiDecoderStack.forward publishes the ids and resets them in a finally, so during
# the backward recompute replay the contextvar is already None and ShensiHashMLP raises
# "needs input_ids for its embedding gate".
# Fix: swap HASH_INPUT_IDS for a shim that also remembers the last published value. The read site is
# a module-global lookup, so replacing the module attribute is enough, and the replay then sees the
# same micro-batch's ids. Caveat: under PP>1 / interleaved schedules a later micro-batch could
# refresh the value before an earlier one is replayed -- fine for the PP=1 layouts used here.
try:
    from megatron.bridge.models.shensi import modeling_shensi as _ms_hids

    _hids_real = _ms_hids.HASH_INPUT_IDS
    if not getattr(_hids_real, "_shensi_persist", False):
        class _PersistentHashInputIds:
            """ContextVar-compatible shim that keeps the last published value across reset()."""

            _shensi_persist = True

            def __init__(self, real):
                self._real = real
                self._last = None

            def set(self, value):
                self._last = value
                return self._real.set(value)

            def reset(self, token):
                try:
                    self._real.reset(token)
                except Exception:
                    pass

            def get(self):
                value = self._real.get()
                return value if value is not None else self._last

        _ms_hids.HASH_INPUT_IDS = _PersistentHashInputIds(_hids_real)
        print("[shensi] hash-MoE input_ids persisted across reset (recompute replay)", flush=True)
except Exception as _e_hids:
    print("[shensi] hash input_ids persistence patch skipped:", _e_hids, flush=True)


# --- appended: unwrap every mcore torch.compile wrapper on NPU ---
# Running tally of torch.compile in this mcore (besides the jit_fuser family handled above):
#   transformer/hyper_connection.py x9, fusions/fused_mhc_kernels.py x2,
#   transformer/moe/router.py x1, experimental_attention_variant/deepseek_v4_hybrid_attention.py x1.
# On Ascend they make torch_npu's inductor compile the mHC/AttnRes elementwise graphs every step
# ("Try to run debug mode for kernel triton_unk_fused_*" ~1300 lines/step) and keep the compiled
# artifacts alive: mcore's own memory estimator reports only ~13 GiB of weights+optimizer+activations
# while the device sits at ~59.5 GiB. Unwrapping removes both the per-step compile and that cache.
# SHENSI_ALLOW_TORCH_COMPILE=1 keeps the compiled versions.
try:
    import os as _os_tc

    def _tc_is_npu():
        """Self-contained NPU probe (do not depend on helpers defined by earlier blocks)."""
        try:
            import torch as _t_tc
        except Exception:
            return False
        if not hasattr(_t_tc, "npu"):
            return False
        try:
            return bool(_t_tc.npu.is_available())
        except Exception:
            return True

    _tc_env = _os_tc.environ.get("SHENSI_ALLOW_TORCH_COMPILE", "0")
    print(
        "[shensi] torch.compile unwrap check: npu=%s SHENSI_ALLOW_TORCH_COMPILE=%s"
        % (_tc_is_npu(), _tc_env),
        flush=True,
    )
    if _tc_env == "1":
        print("[shensi] torch.compile unwrap skipped (escape hatch set)", flush=True)
    elif not _tc_is_npu():
        print("[shensi] torch.compile unwrap skipped (no NPU detected)", flush=True)
    else:
        import importlib as _il_tc
        import sys as _sys_tc

        # Belt: make torch.compile a no-op so nothing can be compiled no matter when it is imported.
        # This is the robust half -- the attribute sweep below only covers modules already imported.
        try:
            import torch as _t_dc

            _orig_compile = _t_dc.compile
            # [NPU 补丁] MindSpeed 已加载时**让位**：它自己的 RequirementsBasicFeature 会 patch
            # torch.compile，且要求 compile 未被改动（否则抛 "the patch of compile exist !"），
            # 而它那套才是昇腾上的正解。MindSpeed 没起来时（本文件下面会打印）才用身份替换兜底。
            # 只有"MindSpeed 真的在打补丁"才让位。两个条件缺一不可：
            #  - 子模块真的导入成功（不能用 `"mindspeed" in sys.modules`：导入失败时父包也会留下）；
            #  - 特征补丁没被关掉（本包默认用 patches-mindspeed 把 MindSpeed 特征闸掉了，
            #    此时它不会 patch torch.compile，我们必须自己兜底）。
            _ms_ok = (
                _os_tc.environ.get("SHENSI_ENABLE_MINDSPEED", "0") == "1"
                and getattr(_sys_tc.modules.get("mindspeed"), "megatron_adaptor", None) is not None
            )
            if _ms_ok:
                print(
                    "[shensi] mindspeed present -> keep MindSpeed's torch.compile patch "
                    "(skip shensi identity replacement)",
                    flush=True,
                )
            elif not getattr(_orig_compile, "_shensi_identity", False):
                def _identity_compile(model=None, **kwargs):
                    if model is None:
                        return lambda m, **kw: m
                    return model

                _identity_compile._shensi_identity = True
                _t_dc.compile = _identity_compile
                try:
                    import torch._dynamo as _dyn_dc

                    _dyn_dc.config.disable = True
                except Exception:
                    pass
                print("[shensi] torch.compile -> identity on NPU (dynamo disabled)", flush=True)
        except Exception as _e_dc:
            print("[shensi] torch.compile -> identity skipped:", _e_dc, flush=True)

        # Import the known compile users first: this block runs at entry-import time, when some of
        # them have not been imported yet, and only already-imported modules can be rewritten.
        for _tc_name in (
            "megatron.core.transformer.hyper_connection",
            "megatron.core.fusions.fused_mhc_kernels",
            "megatron.core.transformer.moe.router",
            "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
        ):
            try:
                _il_tc.import_module(_tc_name)
            except Exception:
                pass

        _tc_pairs = []
        _tc_scanned = 0
        _tc_probe_failed = []
        for _tc_mod in list(_sys_tc.modules.values()):
            # Guard every step: probing attributes on lazy-import shims (e.g. apex/TE stand-ins)
            # raises on first access, and one raise used to abort the whole block.
            try:
                _tc_mname = getattr(_tc_mod, "__name__", "") or ""
                if not _tc_mname.startswith("megatron"):
                    continue
                _tc_scanned += 1
                _tc_dict = getattr(_tc_mod, "__dict__", None)
                if not _tc_dict:
                    continue
                for _tc_attr, _tc_obj in list(_tc_dict.items()):
                    try:
                        _tc_orig = getattr(_tc_obj, "_torchdynamo_orig_callable", None)
                    except Exception as _tc_probe_exc:
                        _tc_probe_failed.append("%s.%s (%s)" % (_tc_mname, _tc_attr, _tc_probe_exc))
                        continue
                    if callable(_tc_orig) and _tc_orig is not _tc_obj:
                        try:
                            setattr(_tc_mod, _tc_attr, _tc_orig)
                            _tc_pairs.append((_tc_obj, _tc_orig))
                        except Exception:
                            pass
            except Exception as _tc_mod_exc:
                _tc_probe_failed.append("%s (%s)" % (getattr(_tc_mod, "__name__", "?"), _tc_mod_exc))
                continue
        if _tc_probe_failed:
            print(
                "[shensi] torch.compile unwrap: skipped %d unprobeable attribute(s), e.g. %s"
                % (len(_tc_probe_failed), _tc_probe_failed[0]),
                flush=True,
            )

        if _tc_pairs:
            # by-value imports keep the old (compiled) object, so rebind those copies too
            for _tc_mod in list(_sys_tc.modules.values()):
                try:
                    _tc_dict = getattr(_tc_mod, "__dict__", None)
                    if not _tc_dict:
                        continue
                    for _tc_attr, _tc_obj in list(_tc_dict.items()):
                        for _tc_wrapped, _tc_unwrapped in _tc_pairs:
                            if _tc_obj is _tc_wrapped:
                                try:
                                    setattr(_tc_mod, _tc_attr, _tc_unwrapped)
                                except Exception:
                                    pass
                except Exception:
                    continue

        print(
            "[shensi] torch.compile unwrapped on NPU: %d wrappers (scanned %d megatron modules) -- "
            "SHENSI_ALLOW_TORCH_COMPILE=1 keeps them"
            % (len(_tc_pairs), _tc_scanned),
            flush=True,
        )
except Exception as _e_tc:
    print("[shensi] torch.compile unwrap skipped:", _e_tc, flush=True)


# --- appended: opt-in allocation snapshot probe (SHENSI_MEM_PROBE=1) ---
# Prints torch_npu allocator totals plus the largest segments (size + first allocation frame) before
# and after the first two train steps, so an unexplained footprint can be attributed to a call site
# instead of guessed at. Inert unless SHENSI_MEM_PROBE=1.
try:
    import os as _os_mp

    if _os_mp.environ.get("SHENSI_MEM_PROBE", "0") in ("1", "2", "3"):

        def _shensi_mem_report(tag):
            try:
                import torch as _t_mp

                if not hasattr(_t_mp, "npu"):
                    print("[shensi][mem] %s: no torch.npu" % tag, flush=True)
                    return
                gib = float(2 ** 30)
                mib = float(2 ** 20)
                alloc = _t_mp.npu.memory_allocated() / gib
                peak = _t_mp.npu.max_memory_allocated() / gib
                resv = _t_mp.npu.memory_reserved() / gib
                print(
                    "[shensi][mem] %s: allocated=%.2f GiB peak=%.2f GiB reserved=%.2f GiB"
                    % (tag, alloc, peak, resv),
                    flush=True,
                )
                try:
                    snap = _t_mp.npu.memory_snapshot()
                except Exception as _e_snap:
                    print("[shensi][mem] snapshot unavailable: %s" % _e_snap, flush=True)
                    return
                agg = {}
                for seg in snap:
                    for blk in (seg.get("blocks") or []):
                        size = int(blk.get("size", 0) or 0)
                        frames = blk.get("frames") or []
                        if frames:
                            key = " <- ".join(
                                "%s:%s" % (fr.get("filename", "?")[-70:], fr.get("line", "?"))
                                for fr in frames[:4]
                            )
                        else:
                            key = "<no frames recorded: run with SHENSI_MEM_PROBE=2>"
                        agg[key] = agg.get(key, 0) + size
                for key, size in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
                    print(
                        "[shensi][mem]   %8.1f MiB  %s" % (size / mib, key),
                        flush=True,
                    )
            except Exception as _e_mp:
                print("[shensi][mem] probe error: %s" % _e_mp, flush=True)

        _mp_mode = _os_mp.environ.get("SHENSI_MEM_PROBE", "0")

        if _mp_mode == "2":
            try:
                import torch as _t_hist

                _t_hist.npu.memory._record_memory_history(max_entries=200000)
                print("[shensi][mem] allocation history recording ON (mode 2)", flush=True)
            except Exception as _e_hist:
                print("[shensi][mem] history recording unavailable: %s" % _e_hist, flush=True)

        _mp_state_shared = {"step": 0}

        if _mp_mode == "3":
            # Per-layer memory打点: shows where inside a step the footprint grows.
            try:
                from megatron.bridge.models.shensi import modeling_shensi as _ms3

                _orig_layer_fwd_mp3 = _ms3.ShensiTransformerLayer.forward
                if not getattr(_orig_layer_fwd_mp3, "_shensi_mem_probe3", False):
                    _mp3_report = {"n": 0}

                    def _layer_fwd_mp3(self, *a, **k):
                        try:
                            import torch as _t3

                            if _mp_state_shared["step"] == 1 and _mp3_report["n"] < 60:
                                _mp3_report["n"] += 1
                                ln = getattr(self, "layer_number", "?")
                                before = _t3.npu.memory_allocated() / float(2 ** 30)
                                out = _orig_layer_fwd_mp3(self, *a, **k)
                                after = _t3.npu.memory_allocated() / float(2 ** 30)
                                print(
                                    "[shensi][mem] layer %-3s allocated %.2f -> %.2f GiB (d=%+.2f)"
                                    % (ln, before, after, after - before),
                                    flush=True,
                                )
                                return out
                        except Exception:
                            pass
                        return _orig_layer_fwd_mp3(self, *a, **k)

                    _layer_fwd_mp3._shensi_mem_probe3 = True
                    _ms3.ShensiTransformerLayer.forward = _layer_fwd_mp3
                    print("[shensi][mem] per-layer打点 installed (mode 3)", flush=True)
            except Exception as _e_p3:
                print("[shensi][mem] per-layer打点 skipped: %s" % _e_p3, flush=True)

        try:
            from megatron.training import training as _tr_mp

            _orig_train_step_mp = _tr_mp.train_step
            if not getattr(_orig_train_step_mp, "_shensi_mem_probe", False):
                _mp_state = {"n": 0}

                def _train_step_mp(*a, **k):
                    if _mp_state["n"] < 2:
                        _mp_state["n"] += 1
                        _mp_state_shared["step"] = _mp_state["n"]
                        _shensi_mem_report("before train_step #%d" % _mp_state["n"])
                        try:
                            out = _orig_train_step_mp(*a, **k)
                        except Exception as _e_step:
                            if "out of memory" in str(_e_step).lower():
                                print(
                                    "[shensi][mem] OOM inside train_step #%d -> dumping stacks"
                                    % _mp_state["n"],
                                    flush=True,
                                )
                                _shensi_mem_report("at OOM")
                                if _mp_mode == "2":
                                    try:
                                        import torch as _t_dump

                                        rank = 0
                                        try:
                                            rank = int(_os_mp.environ.get("RANK", "0"))
                                        except Exception:
                                            pass
                                        path = "/tmp/shensi_mem_snapshot_rank%d.pickle" % rank
                                        _t_dump.npu.memory._dump_snapshot(path)
                                        print("[shensi][mem] snapshot dumped -> %s" % path, flush=True)
                                    except Exception as _e_dump:
                                        print("[shensi][mem] dump failed: %s" % _e_dump, flush=True)
                            raise
                        _shensi_mem_report("after  train_step #%d" % _mp_state["n"])
                        return out
                    return _orig_train_step_mp(*a, **k)

                _train_step_mp._shensi_mem_probe = True
                _tr_mp.train_step = _train_step_mp
                print("[shensi][mem] probe installed on train_step (SHENSI_MEM_PROBE=1)", flush=True)
                _shensi_mem_report("at install time")
        except Exception as _e_mp2:
            print("[shensi][mem] probe install skipped: %s" % _e_mp2, flush=True)
except Exception as _e_mp3:
    print("[shensi][mem] probe block skipped: %s" % _e_mp3, flush=True)


# --- appended: per-layer memory map (SHENSI_MEM_PROBE=3) ---
# Prints live allocated GiB before/after every ShensiTransformerLayer (and the MTP layer) for the
# first training step only, so an unattributed in-step footprint can be pinned to a layer.
# Inert unless SHENSI_MEM_PROBE=3.
try:
    import os as _os_pl

    if _os_pl.environ.get("SHENSI_MEM_PROBE", "0") == "3":

        def _pl_alloc():
            try:
                import torch as _t_pl

                return _t_pl.npu.memory_allocated() / float(2 ** 30)
            except Exception:
                return -1.0

        _pl_state = {"step": 0}

        def _pl_wrap(mod, tag):
            try:
                _orig_fwd = mod.forward
                if getattr(_orig_fwd, "_shensi_layer_probe", False):
                    return
                if getattr(mod, "_shensi_layer_probe_owner", False):
                    return

                def _fwd_pl(self, *a, **k):
                    if _pl_state["step"] > 1:
                        return _orig_fwd(self, *a, **k)
                    ln = getattr(self, "layer_number", "?")
                    before = _pl_alloc()
                    out = _orig_fwd(self, *a, **k)
                    after = _pl_alloc()
                    print(
                        "[shensi][mem][layer] %s:%s in=%.2f out=%.2f d=%+.2f GiB"
                        % (tag, ln, before, after, after - before),
                        flush=True,
                    )
                    return out

                _fwd_pl._shensi_layer_probe = True
                mod.forward = _fwd_pl
                mod._shensi_layer_probe_owner = True
            except Exception as _e_pl:
                print("[shensi][mem][layer] wrap failed for %s: %s" % (tag, _e_pl), flush=True)

        try:
            from megatron.training import training as _tr_pl

            _orig_ts_pl = _tr_pl.train_step
            if not getattr(_orig_ts_pl, "_shensi_step_counter", False):

                def _ts_pl(*a, **k):
                    _pl_state["step"] += 1
                    return _orig_ts_pl(*a, **k)

                _ts_pl._shensi_step_counter = True
                _tr_pl.train_step = _ts_pl
        except Exception as _e_sc:
            print("[shensi][mem][layer] step counter unavailable: %s" % _e_sc, flush=True)

        try:
            from megatron.bridge.models.shensi import modeling_shensi as _ms_pl

            _pl_wrap(_ms_pl.ShensiTransformerLayer, "dec")
        except Exception as _e1:
            print("[shensi][mem][layer] ShensiTransformerLayer unavailable: %s" % _e1, flush=True)

        try:
            from megatron.core.transformer.multi_token_prediction import (
                MultiTokenPredictionLayer as _MPL_pl,
            )

            _pl_wrap(_MPL_pl, "mtp")
        except Exception as _e2:
            print("[shensi][mem][layer] MTP layer unavailable: %s" % _e2, flush=True)

        print("[shensi][mem][layer] per-layer memory probe installed (SHENSI_MEM_PROBE=3)", flush=True)
except Exception as _e_pl3:
    print("[shensi][mem][layer] probe skipped: %s" % _e_pl3, flush=True)


# --- 自检放在最后：以上所有补丁段都已执行，这里读到的才是真实生效状态 ---
try:
    _print_active_patch_inventory()
except Exception as _e_inv:  # noqa: BLE001
    print("[shensi] patch inventory selfcheck skipped:", _e_inv, flush=True)
