# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


from typing import List, Optional, Tuple
import torch

FP32_KEEP_ATTR = "_shensi_keep_fp32"
FP32_KEEP_GROUP_ATTR = "SHENSI_FP32_KEEP_GROUP"
GROUP_MHC = "mhc"
GROUP_ATTN_RES = "attn_res"
GROUP_UPSTREAM = "upstream_strict"
DEFAULT_GROUPS: Tuple[str, ...] = (GROUP_MHC, GROUP_UPSTREAM)
UPSTREAM_STRICT_SUFFIXES: Tuple[str, ...] = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_mlp_layernorm",
    "q_a_norm",
    "kv_norm",
    "norm",
    "sinks",
    "position_bias",
)
UPSTREAM_EXTRA_SUBSTRINGS: Tuple[str, ...] = (
    ".core_attention.attn_sink",
    ".core_attention.compressor.ape",
    ".core_attention.indexer.compressor.ape",
)
CAST_ORIG_FORWARD_ATTR = "_shensi_cast_orig_forward"
CAST_DTYPE_ATTR = "_shensi_cast_model_dtype"
APPLY_GUARD_ATTR = "_shensi_orig_apply"
ENV_SWITCH = "SHENSI_HC_FP32_KEEP"
ENV_GROUPS = "SHENSI_FP32_KEEP_GROUPS"
ENV_NO_CAST = "SHENSI_FP32_KEEP_NO_CAST"


class ShensiFp32KeepMixin:
    def _apply(self, fn, recurse=True):  # noqa: ANN001
        saved: List[Tuple[torch.nn.Module, str, torch.Tensor]] = []
        for mod in self.modules():
            for name, p in mod._parameters.items():
                if (
                    p is not None
                    and getattr(p, FP32_KEEP_ATTR, False)
                    and p.dtype == torch.float32
                ):
                    saved.append((mod, name, p.detach().clone()))
        out = super()._apply(fn, recurse=recurse)
        for mod, name, fp32 in saved:
            cur = mod._parameters.get(name)
            if cur is None:
                continue
            if cur.dtype != torch.float32 or cur.device != fp32.device:
                cur.data = fp32.to(cur.device)
        return out


def fp32_keep_enabled(config=None, env: Optional[str] = None) -> bool:
    return bool(fp32_keep_groups(config, env))


def fp32_keep_groups(config=None, env: Optional[str] = None) -> Tuple[str, ...]:
    import os

    raw_groups = os.environ.get(ENV_GROUPS, env)
    if raw_groups is not None and str(raw_groups).strip() != "":
        return tuple(
            g.strip() for g in str(raw_groups).replace(",", " ").split() if g.strip()
        )
    raw = os.environ.get(ENV_SWITCH, None)
    if raw is not None and str(raw).strip() != "":
        return (
            DEFAULT_GROUPS
            if str(raw).strip().lower() in ("1", "true", "yes", "on")
            else ()
        )
    if config is not None and hasattr(config, "hc_fp32_keep"):
        return DEFAULT_GROUPS if bool(getattr(config, "hc_fp32_keep")) else ()
    return DEFAULT_GROUPS


def _keep_targets(module: torch.nn.Module, groups) -> List[torch.nn.Module]:
    want = tuple(groups)
    return [
        m for m in module.modules() if getattr(m, FP32_KEEP_GROUP_ATTR, None) in want
    ]


def apply_fp32_keep(
    model: torch.nn.Module, *, groups=None, verbose: bool = False
) -> List[str]:
    groups = DEFAULT_GROUPS if groups is None else tuple(groups)
    if not groups:
        return []
    names: List[str] = []
    targets = _keep_targets(model, groups)
    for mod in targets:
        for pname, p in list(mod.named_parameters(recurse=True)):
            if p is None:
                continue
            if p.dtype != torch.float32:
                p.data = p.data.float()
            setattr(p, FP32_KEEP_ATTR, True)
            names.append(f"{type(mod).__name__}:{pname}")
    if GROUP_UPSTREAM in groups:
        names.extend(apply_upstream_fp32_keep(model))
    if verbose and names:
        from megatron.training import print_rank_0

        print_rank_0(
            f"[shensi][fp32keep] 已把 {len(targets)} 个家族模块 + 上游 strict 模块的参数转为 fp32"
            f"（共 {len(names)} 个参数；组={','.join(groups)}；HF _keep_in_fp32_modules_strict 对齐）"
        )
    return sorted(names)


def install_apply_guard(mod: torch.nn.Module) -> bool:
    if getattr(mod, APPLY_GUARD_ATTR, None) is not None:
        return False
    if not any(
        p is not None and getattr(p, FP32_KEEP_ATTR, False)
        for p in mod._parameters.values()
    ):
        return False
    orig_apply = mod._apply

    def guarded_apply(fn, recurse=True):
        saved = [
            (n, p, p.detach().clone())
            for n, p in mod._parameters.items()
            if p is not None
            and getattr(p, FP32_KEEP_ATTR, False)
            and p.dtype == torch.float32
        ]
        out = orig_apply(fn, recurse=recurse)
        for n, p, fp32 in saved:
            cur = mod._parameters.get(n)
            if cur is None or cur is fp32:
                continue
            if cur.dtype != torch.float32 or cur.device != fp32.device:
                cur.data = fp32.to(cur.device)
        return out

    setattr(mod, APPLY_GUARD_ATTR, orig_apply)
    mod._apply = guarded_apply
    return True


def compute_dtype_for(inputs, kwargs, fallback: torch.dtype) -> torch.dtype:
    try:
        if torch.is_autocast_enabled():
            return torch.get_autocast_dtype(torch.get_autocast_device_type())
    except Exception:  # noqa: BLE001
        pass
    for t in list(inputs) + list(kwargs.values()):
        if torch.is_tensor(t) and t.is_floating_point() and t.dtype != torch.float32:
            return t.dtype
    for t in list(inputs) + list(kwargs.values()):
        if torch.is_tensor(t) and t.is_floating_point():
            return t.dtype
    return fallback


class _ShensiFp32CastToComputeDtype(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dtype: torch.dtype):
        ctx.input_dtype = tensor.dtype
        return tensor.to(dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.to(ctx.input_dtype), None


def compute_dtype_weight(p: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if p.dtype == dtype:
        return p
    if not p.requires_grad:
        return p.detach().to(dtype)
    return _ShensiFp32CastToComputeDtype.apply(p, dtype)


def install_forward_cast(mod: torch.nn.Module) -> bool:
    if getattr(mod, CAST_ORIG_FORWARD_ATTR, None) is not None:
        return False
    import os

    if str(os.environ.get(ENV_NO_CAST, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    if not any(
        p is not None and getattr(p, FP32_KEEP_ATTR, False)
        for p in mod._parameters.values()
    ):
        return False
    orig = mod.forward
    model_dtype = None
    for p in mod._parameters.values():
        if p is not None and p.dtype != torch.float32:
            model_dtype = p.dtype
    if model_dtype is None:
        model_dtype = None

    def casted_forward(*args, **kwargs):
        fallback = getattr(mod, CAST_DTYPE_ATTR, None)
        if fallback is None or fallback == torch.float32:
            for q in mod.parameters():
                if q.is_floating_point() and q.dtype != torch.float32:
                    fallback = q.dtype
                    break
        dtype = compute_dtype_for(args, kwargs, fallback)
        if dtype is None or dtype == torch.float32:
            return orig(*args, **kwargs)
        saved = {}
        for name, p in list(mod._parameters.items()):
            if (
                p is None
                or not getattr(p, FP32_KEEP_ATTR, False)
                or p.dtype != torch.float32
            ):
                continue
            saved[name] = p
            mod._parameters[name] = compute_dtype_weight(p, dtype)
        try:
            return orig(*args, **kwargs)
        finally:
            for name, p in saved.items():
                mod._parameters[name] = p

    casted_forward.__name__ = "shensi_fp32_keep_cast_forward"
    setattr(mod, CAST_ORIG_FORWARD_ATTR, orig)
    mod.forward = casted_forward
    return True


def set_cast_model_dtype(model: torch.nn.Module) -> None:
    dtypes = {p.dtype for p in model.parameters() if p.dtype != torch.float32}
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
    for mod in model.modules():
        if getattr(mod, CAST_ORIG_FORWARD_ATTR, None) is not None:
            setattr(mod, CAST_DTYPE_ATTR, dtype)


def _is_family_keep_module(mod: torch.nn.Module) -> bool:
    return getattr(mod, FP32_KEEP_GROUP_ATTR, None) in (GROUP_MHC, GROUP_ATTN_RES)


def _ancestor_has_family_group(model: torch.nn.Module, mod_name: str) -> bool:
    mods = dict(model.named_modules())
    parts = mod_name.split(".") if mod_name else []
    for i in range(len(parts), -1, -1):
        m = mods.get(".".join(parts[:i]))
        if m is not None and _is_family_keep_module(m):
            return True
    return False


def upstream_keep_targets(model: torch.nn.Module, suffixes=UPSTREAM_STRICT_SUFFIXES):
    pats = tuple(suffixes) + tuple(UPSTREAM_EXTRA_SUBSTRINGS)
    mods = dict(model.named_modules())
    out = []
    for key, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if not any(s in key for s in pats):
            continue
        head = key.rsplit(".", 1)[0] if "." in key else ""
        if _ancestor_has_family_group(model, head):
            continue
        mod = mods.get(head, model)
        out.append((key, head, mod))
    return out


def apply_upstream_fp32_keep(
    model: torch.nn.Module, *, suffixes=None, verbose=False
) -> List[str]:
    suffixes = UPSTREAM_STRICT_SUFFIXES if suffixes is None else tuple(suffixes)
    names: List[str] = []
    touched = {}
    for key, mod_name, mod in upstream_keep_targets(model, suffixes):
        p = dict(mod.named_parameters(recurse=False)).get(key.rsplit(".", 1)[-1])
        if p is None:
            p = model.get_parameter(key)
        if p.dtype != torch.float32:
            p.data = p.data.float()
        setattr(p, FP32_KEEP_ATTR, True)
        names.append(key)
        touched[id(mod)] = (mod_name, mod)
    for mod_name, mod in touched.values():
        install_forward_cast(mod)
        install_apply_guard(mod)
    set_cast_model_dtype(model)
    if verbose and names:
        print(
            f"[shensi][fp32keep] upstream_strict：{len(names)} 个参数 / {len(touched)} 个模块"
        )
    return sorted(names)


def upstream_keep_param_names(model: torch.nn.Module) -> List[str]:
    return sorted(k for k, _m, _mod in upstream_keep_targets(model))


def keep_param_names(model: torch.nn.Module) -> List[str]:
    return sorted(
        f"{mod_name}.{pname}" if mod_name else pname
        for mod_name, mod in model.named_modules()
        for pname, p in mod._parameters.items()
        if p is not None and getattr(p, FP32_KEEP_ATTR, False)
    )


def dtype_counts(model: torch.nn.Module) -> dict:
    out: dict = {}
    for _, p in model.named_parameters():
        key = str(p.dtype).replace("torch.", "")
        n, numel = out.get(key, (0, 0))
        out[key] = (n + 1, numel + p.numel())
    return out
