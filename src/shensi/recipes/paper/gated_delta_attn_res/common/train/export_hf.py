"""把 mcore 检查点导出为 HF 目录（几何以检查点自带的配置为准）。"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.paper.gated_delta_attn_res.stage2_rl import variants as gdar_variants

_GEOMETRY: dict[str, tuple[str, str, object]] = {
    "num_hidden_layers": ("model.transformer.num_layers", "train.model.num_layers", None),
    "hidden_size": ("model.transformer.hidden_size", "train.model.hidden_size", None),
    "intermediate_size": ("model.transformer.ffn_hidden_size", "train.model.ffn_hidden_size", None),
    "num_attention_heads": (
        "model.transformer.num_attention_heads",
        "train.model.num_attention_heads",
        None,
    ),
    "num_key_value_heads": (
        "model.transformer.num_query_groups",
        "train.model.num_query_groups",
        None,
    ),
    "head_dim": ("model.transformer.kv_channels", "train.model.kv_channels", None),
    "max_position_embeddings": (
        "model.seq_length",
        "train.model.max_position_embeddings",
        None,
    ),
    "rms_norm_eps": ("model.transformer.layernorm_epsilon", "train.model.norm_epsilon", 1e-6),
    "rope_theta": ("model.transformer.rotary_base", "train.model.rotary_base", 1000000.0),
}

_VOCAB = ("model.vocab_size", "train.data.tokenizer.vocab_size", None)
_VOCAB_DIVISIBLE = (
    "model.make_vocab_size_divisible_by",
    "train.data.tokenizer.make_vocab_size_divisible_by",
    64,
)

_GATE_CHANNELS_BY_SPEC = {
    spec_object: channels
    for channels, spec_object in gdar_variants.GDAR_GATE_CHANNEL_SPECS.items()
    if spec_object is not None
}


def _dig(obj, dotted: str):
    node = obj
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


_NORM_MODEL_TO_CKPT = {
    "self_attention.input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
    "mlp.pre_mlp_layernorm.weight": "mlp.linear_fc1.layer_norm_weight",
}
_NORM_CKPT_TO_MODEL = {v: k for k, v in _NORM_MODEL_TO_CKPT.items()}


def _remap(d: dict, table: dict) -> dict:
    out = {}
    for name, value in d.items():
        for suffix, replacement in table.items():
            if name.endswith(suffix):
                name = name[: -len(suffix)] + replacement
                break
        out[name] = value
    return out


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
)


def _latest_iter(ckpt_dir: Path) -> str | None:
    p = ckpt_dir / "latest_checkpointed_iteration.txt"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def _resolve_iter_dir(ckpt_dir: Path, load_iter: str | None) -> Path:
    if (ckpt_dir / "metadata.json").is_file() or (ckpt_dir / "common.pt").is_file():
        return ckpt_dir
    iteration = load_iter or _latest_iter(ckpt_dir)
    if not iteration:
        raise SystemExit(
            f"{ckpt_dir} 里既没有检查点元数据，也没有 latest_checkpointed_iteration.txt"
        )
    d = ckpt_dir / f"iter_{int(iteration):07d}"
    if not d.is_dir():
        raise SystemExit(f"找不到迭代目录：{d}")
    return d


def _run_config(args, ckpt_dir: Path) -> tuple[dict | None, str]:
    if args.run_config:
        p = Path(args.run_config)
        if not p.is_file():
            raise SystemExit(f"--run-config 指向的文件不存在：{p}")
        return common.base.load_yaml(p), str(p)
    for p in (ckpt_dir / "config.yaml", ckpt_dir.parent / "config.yaml"):
        if p.is_file():
            return common.base.load_yaml(p), str(p)
    return None, f"（{ckpt_dir} 与 {ckpt_dir.parent} 下都没有 config.yaml）"


def _ckpt_root_from_run_cfg(run_cfg: dict | None) -> Path | None:
    if not run_cfg:
        return None
    save = (((run_cfg.get("train") or {}).get("system") or {}).get("checkpoint") or {}).get("save")
    return Path(save) if save else None


def _ckpt_geometry(iter_dir: Path) -> dict | None:
    p = iter_dir / "run_config.yaml"
    if not p.is_file():
        return None
    return common.base.load_yaml(p)


def _spec_object(run_cfg: dict | None, model_algo: str | None):
    spec = None
    if run_cfg is not None:
        spec = (run_cfg.get("train") or {}).get("model", {}).get("spec")
    if model_algo:
        algo_spec = common.MODEL_ALGOS.get(model_algo)
        if algo_spec is None:
            raise SystemExit(
                f"未知 --model-algo {model_algo!r}；可用：{sorted(common.MODEL_ALGOS)}"
            )
        spec = algo_spec.split() if isinstance(algo_spec, str) else list(algo_spec)
    if not spec:
        raise SystemExit(
            "没有变体/旋钮来源：给 --model-algo <名字>，或给 --run-config <那次训练的 "
            "config.yaml>（每个 stage 的 exp_dir 里都有一份，里面有 train.model.spec），"
            "或给 --hf-config <HF 目录/config.json> 直接指定几何与旋钮。"
        )
    if len(spec) != 2:
        raise SystemExit(f"train.model.spec 只支持 [模块, 对象] 两项，拿到 {spec!r}")
    module_name, object_name = spec
    module = importlib.import_module(module_name)
    spec_obj = getattr(module, object_name, None)
    if spec_obj is None:
        raise SystemExit(f"{module_name}:{object_name} 不存在")
    params = dict(getattr(spec_obj, "params", {}) or {})
    where = f"{module_name}:{object_name}"

    if module_name.endswith((".gdar_spec", ".ablation_spec")):
        variant = "gdar"
        knobs = {
            "attn_res_" + k[len("gdar_") :]: v for k, v in params.items() if k.startswith("gdar_")
        }
        if module_name.endswith(".ablation_spec"):
            channels = _GATE_CHANNELS_BY_SPEC.get(object_name)
            if channels is None:
                raise SystemExit(
                    f"{object_name} 不在 gates 反查表里（七个子集 + scalar 才有对应字段）；"
                    f"用 --hf-config 显式给配置"
                )
            knobs["attn_res_gate_channels"] = channels
        return variant, knobs, where, params

    variant = params.get("depth_variant") or params.get("hc_family")
    resolved = gdar_variants.VARIANTS.get(variant) if variant else None
    if resolved is None or resolved.knob_map is None:
        raise SystemExit(
            f"{where} 的连接旋钮不能从规格反写回 HF（{variant!r} 两侧名字不是一对一）。"
            f"给 --hf-config <那份 HF 目录/config.json>，几何与旋钮都读它。"
        )
    inverse = {mg: hf for hf, mg in resolved.knob_map.items()}
    knobs = {hf_name: params[mg] for mg, hf_name in inverse.items() if mg in params}
    return variant, knobs, where, params


def _build_hf_config(
    args, run_cfg: dict | None, ckpt_rc: dict | None, vocab_size: int | None = None
):
    if run_cfg is None and ckpt_rc is None:
        raise SystemExit("没有 run config，也没有检查点记录")

    variant, knobs, where, spec_params = _spec_object(run_cfg, args.model_algo)
    resolved = gdar_variants.VARIANTS[variant]

    geometry: dict[str, object] = {}
    for hf_field, (ckpt_path, stage_path, default) in _GEOMETRY.items():
        value = _dig(ckpt_rc, ckpt_path) if ckpt_rc else None
        source = "ckpt" if value is not None else None
        if value is None and run_cfg is not None:
            value = _dig(run_cfg, stage_path)
            source = "stage" if value is not None else None
        if value is None:
            value = default
            source = "default"
        if value is None:
            raise SystemExit(
                f"几何拿不到 {hf_field}：检查点里没有 run_config.yaml（{ckpt_path}），"
                f"stage config 里也没有（{stage_path}）"
            )
        if source == "default" and hf_field not in ("rms_norm_eps", "rope_theta"):
            print(f"[export] ⚠️ {hf_field} 用了缺省值 {value}（两侧来源都没有）")
        geometry[hf_field] = value

    for hf_field, (ckpt_path, stage_path, _) in _GEOMETRY.items():
        if ckpt_rc is None or run_cfg is None:
            break
        a, b = _dig(ckpt_rc, ckpt_path), _dig(run_cfg, stage_path)
        if a is not None and b is not None and a != b:
            print(
                f"[export] ⚠️ {hf_field}：检查点说 {a}，stage config 说 {b} —— 以检查点为准"
                f"（这两个文件不是同一次 run 的产物；换 --run-config 指对那份）"
            )

    if vocab_size is None:
        vocab_size = int(_dig(ckpt_rc, _VOCAB[0]) or _dig(run_cfg or {}, _VOCAB[1]) or 151936)
    divisible = int(
        _dig(ckpt_rc, _VOCAB_DIVISIBLE[0])
        or _dig(run_cfg or {}, _VOCAB_DIVISIBLE[1])
        or _VOCAB_DIVISIBLE[2]
    )
    tie = not bool(_dig(ckpt_rc, "model.untie_embeddings_and_output_weights"))
    if run_cfg is not None:
        tie = tie and not bool(_dig(run_cfg, "train.model.untie_embeddings_and_output_weights"))

    config_module = importlib.import_module(
        "shensi.recipes.paper.gated_delta_attn_res.common.models.transformers."
        + resolved.config_module
    )
    config_cls = getattr(config_module, resolved.config_class)
    cfg = config_cls(
        vocab_size=vocab_size,
        tie_word_embeddings=tie,
        attention_dropout=0.0,
        **geometry,
        **knobs,
    )
    cfg._attn_implementation = "eager"
    cfg.architectures = [resolved.lm_class]
    return cfg, (variant, knobs, where, spec_params), vocab_size, divisible


def _init_distributed(device: str) -> None:
    import os

    import torch
    from megatron.core import parallel_state as mpu

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    if device == "cuda":
        torch.cuda.set_device(0)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)


def _bind_pg_collection(provider) -> None:
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()  # noqa: SLF001


def _ckpt_tensor_shapes(iter_dir: Path) -> dict[str, tuple[int, ...]]:
    from torch.distributed.checkpoint import FileSystemReader

    meta = FileSystemReader(str(iter_dir)).read_metadata().state_dict_metadata
    shapes: dict[str, tuple[int, ...]] = {}
    for key, entry in meta.items():
        size = getattr(entry, "size", None)
        if size is not None:
            shapes[key] = tuple(int(s) for s in size)
    return shapes


def _check_shapes(model, iter_dir: Path) -> None:
    shapes = _ckpt_tensor_shapes(iter_dir)
    if not shapes:
        return
    qkv_key = "decoder.layers.0.self_attention.linear_qkv.weight"
    emb_key = "embedding.word_embeddings.weight"
    expected = model.sharded_state_dict()

    def _flat(d, key):
        v = d.get(key)
        if v is None:
            return None
        gs = getattr(v, "global_shape", None)
        if gs is None:
            gs = getattr(v, "shape", None) or getattr(v, "local_shape", None)
        return tuple(gs) if gs is not None else None

    hints = []
    for key in (qkv_key, emb_key):
        want, got = _flat(expected, key), shapes.get(key)
        if want is not None and got is not None and want != got:
            hints.append(f"  {key}：检查点 {got} vs 配置推出的模型 {want}")
    if hints:
        raise SystemExit(
            "[export] 检查点与几何对不上（几何来自检查点的 run_config.yaml，说明它和 stage "
            "config 不是同一次 run 的产物）：\n"
            + "\n".join(hints)
            + "\n-> 换 --run-config 指对那次 run 的 config.yaml，或把 --ckpt 指到那个 exp 目录上。"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="mcore torch_dist ckpt → HF 目录（RL / 评测 / 发布用）"
    )
    ap.add_argument(
        "--ckpt", required=True, help="mcore 检查点目录、它的父目录，或那次 run 的 exp 目录"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出目录（HF 格式：config.json + safetensors + 随权重走的 .py + tokenizer）",
    )
    ap.add_argument(
        "--run-config", default=None, help="那次训练的 config.yaml（默认在 --ckpt 附近找）"
    )
    ap.add_argument("--hf-config", default=None, help="几何/旋钮来源：HF 目录或 config.json")
    ap.add_argument(
        "--model-algo", default=None, help="覆盖 run config 里的规格（如 qwen3_gdar_paper）"
    )
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer 目录（默认用 run config 里的 tokenizer_model）"
    )
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument(
        "--load-iter",
        default=None,
        help="载入哪个 iteration（默认取 latest_checkpointed_iteration.txt）",
    )
    ap.add_argument("--max-position-embeddings", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true", help="--hf-config 需要时打开")
    args = ap.parse_args(argv)

    import torch

    ckpt_dir = Path(args.ckpt)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"检查点目录不存在：{ckpt_dir}")
    out = Path(args.out)

    run_cfg, run_cfg_where = (
        (None, "<--hf-config>") if args.hf_config else _run_config(args, ckpt_dir)
    )
    if run_cfg is not None and (ckpt_dir / "config.yaml").is_file():
        root = _ckpt_root_from_run_cfg(run_cfg)
        if root is not None and root.is_dir():
            ckpt_dir = root
            print(f"[export] 检查点位置取自 run config：{ckpt_dir}")

    iter_dir = _resolve_iter_dir(ckpt_dir, args.load_iter)
    ckpt_rc = _ckpt_geometry(iter_dir)
    ckpt_keys = _ckpt_tensor_shapes(iter_dir)
    ckpt_qk = any(k.endswith("self_attention.q_layernorm.weight") for k in ckpt_keys)
    print(
        f"[export] 检查点：{iter_dir}"
        + (
            "（run_config.yaml 在位，几何以它为准）"
            if ckpt_rc
            else "（没有 run_config.yaml，几何退回 stage config）"
        )
    )

    tok_dir = Path(args.tokenizer) if args.tokenizer else None
    if tok_dir is None:
        spec_tok = _dig(run_cfg or {}, "train.data.tokenizer.tokenizer_model")
        tok_dir = Path(spec_tok) if spec_tok else None
    if tok_dir is None:
        cand = iter_dir / "tokenizer"
        tok_dir = cand if cand.is_dir() and any(cand.iterdir()) else None
    tok = None
    if tok_dir is not None and tok_dir.is_dir():
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(tok_dir))
        except Exception as exc:  # noqa: BLE001 - tokenizer 目录可能是 mock 的
            print(f"[export] ⚠️ 读不出 tokenizer（{tok_dir}）：{type(exc).__name__}: {exc}")
            tok = None
    vocab_size = len(tok) if tok is not None else None
    if tok is None:
        padded = _dig(ckpt_rc, _VOCAB) or _dig(run_cfg or {}, _VOCAB[1])
        if padded is not None:
            print(
                f"[export] ⚠️ 没有可用的 tokenizer，HF 的 vocab_size 用检查点的补齐值 {padded}"
                "（补齐前的真实长度未知；先补一个 tokenizer 再导更稳）"
            )

    if args.hf_config:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(
            args.hf_config, trust_remote_code=args.trust_remote_code
        )
        spec_info, divisible = None, 64
        where = f"<--hf-config {args.hf_config}>"
        print(f"[export] 几何来源：{where}")
    else:
        hf_cfg, spec_info, _, divisible = _build_hf_config(args, run_cfg, ckpt_rc, vocab_size)
        where = f"{spec_info[2]}（run config：{run_cfg_where}）"
        print(
            f"[export] 变体：{spec_info[0]}（规格 {spec_info[2]}），连接旋钮 {len(spec_info[1])} 项"
        )
        print(f"[export] 几何来源：检查点 run_config.yaml + {run_cfg_where}")
    print(f"[export] qk_layernorm：{ckpt_qk}（跟随检查点）")
    if args.max_position_embeddings:
        hf_cfg.max_position_embeddings = args.max_position_embeddings

    dtype = args.dtype
    if dtype == "auto":
        system = ((run_cfg or {}).get("train") or {}).get("system") or {}
        bf16 = bool((system.get("precision") or {}).get("bf16", system.get("bf16", False)))
        if ckpt_rc is not None and _dig(ckpt_rc, "model.transformer.bf16") is not None:
            bf16 = bool(_dig(ckpt_rc, "model.transformer.bf16"))
        dtype = "bf16" if bf16 else "fp32"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    print(
        f"[export] 几何：vocab={hf_cfg.vocab_size}（补齐 {divisible}）layers={hf_cfg.num_hidden_layers} "
        f"hidden={hf_cfg.hidden_size} heads={hf_cfg.num_attention_heads} kv={hf_cfg.num_key_value_heads} "
        f"head_dim={hf_cfg.head_dim} dtype={dtype}"
    )

    _init_distributed(args.device)

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    from shensi.recipes.paper.gated_delta_attn_res.stage2_rl import (
        gdar_bridge,  # noqa: F401  导入即注册
    )

    bridge = AutoBridge.from_hf_config(hf_cfg)
    provider = bridge.to_megatron_provider(load_weights=False)
    overrides = {
        "qk_layernorm": ckpt_qk,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
        "seq_length": int(getattr(hf_cfg, "max_position_embeddings", 4096)),
        "should_pad_vocab": True,
        "make_vocab_size_divisible_by": divisible,
        "gradient_accumulation_fusion": False,
    }
    if hasattr(provider, "cuda_graph_modules"):
        provider.cuda_graph_modules = None
    if hasattr(provider, "apply_overrides_and_finalize"):
        provider.apply_overrides_and_finalize(dtype=torch_dtype, overrides=overrides)
    else:  # pragma: no cover - 只在更老的 Bridge 上走
        provider.params_dtype = torch_dtype
        provider.fp16, provider.bf16 = dtype == "fp16", dtype == "bf16"
        for name, value in overrides.items():
            setattr(provider, name, value)
        provider.finalize()

    if spec_info is not None and spec_info[0] == "gdar":
        from shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_layer import (
            gdar_knobs_from_kwargs,
        )

        trained = gdar_knobs_from_kwargs(dict(spec_info[3]))
        bridged = gdar_knobs_from_kwargs(dict(provider.transformer_layer_spec.params))
        if trained != bridged:
            diff = {
                k: (getattr(bridged, k), getattr(trained, k))
                for k in trained.__dataclass_fields__
                if getattr(bridged, k) != getattr(trained, k)
            }
            raise SystemExit(f"规格不一致（桥上 vs 训练）：{diff}")

    _bind_pg_collection(provider)
    model = provider.provide()
    if args.device == "cuda":
        model = model.cuda()
    _check_shapes(model, iter_dir)

    print(
        f"[export] 建好 mcore 模型（{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M 参数，"
        f"device={args.device}）；载入 {iter_dir}"
    )

    from megatron.core import dist_checkpointing

    loaded = dist_checkpointing.load(
        _remap(model.sharded_state_dict(), _NORM_MODEL_TO_CKPT), str(iter_dir)
    )
    loaded = _remap(loaded, _NORM_CKPT_TO_MODEL)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    real_missing = [k for k in missing if not k.endswith("_extra_state")]
    if real_missing:
        raise SystemExit(
            f"[export] 有 {len(real_missing)} 个参数没在检查点里找到（前几个）：{real_missing[:5]}"
        )
    print(
        f"[export] 权重已载入（missing={len(missing)}（应为 _extra_state）"
        f"unexpected={len(unexpected)}{'：' + str(unexpected[:6]) if unexpected else ''}）"
    )

    out.mkdir(parents=True, exist_ok=True)

    tok_where = "未复制"
    if tok is not None and tok_dir is not None:
        if len(tok) != int(hf_cfg.vocab_size):
            tok_where = (
                f"跳过：{tok_dir} 的词表 {len(tok)} != 导出的词表 {hf_cfg.vocab_size}"
                "（这个 ckpt 是别的 tokenizer 训的）"
            )
        else:
            copied = [f for f in tok_dir.iterdir() if f.is_file() and f.name in _TOKENIZER_FILES]
            for f in copied:
                shutil.copy2(f, out / f.name)
            tok_where = f"复制自 {tok_dir}（{len(copied)} 个文件）"
    print(f"[export] tokenizer：{tok_where}")

    try:
        bridge.save_hf_pretrained([model], str(out), show_progress=True)
    except ModuleNotFoundError as exc:
        if "modelopt" not in str(exc):
            raise
        print("[export] 没装 nvidia-modelopt，改用 export_hf_weights + safetensors 直接写（等价）")
        from safetensors.torch import save_file

        hf_cfg.save_pretrained(str(out))
        state = {name: tensor for name, tensor in bridge.export_hf_weights([model])}
        save_file(
            {k: v.contiguous() for k, v in state.items()},
            str(out / "model.safetensors"),
            metadata={"format": "pt"},
        )
        print(f"[export] 写了 {len(state)} 个张量到 model.safetensors")

    if spec_info is not None:
        variant = gdar_variants.VARIANTS[spec_info[0]]
        models_dir = Path("shensi/recipes/paper/gated_delta_attn_res/models/transformers")
        src_dir = Path(__file__).resolve().parents[1] / "models/transformers"
        del models_dir
        for name in (f"{variant.config_module}.py", f"{variant.modeling_module}.py"):
            src = src_dir / name
            if src.is_file():
                shutil.copy2(src, out / name)
                print(f"[export] 随权重复制的模块：{name}")

    if tok is not None and not (out / "generation_config.json").is_file():
        (out / "generation_config.json").write_text(
            json.dumps(
                {
                    "bos_token_id": tok.bos_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "pad_token_id": tok.pad_token_id or tok.eos_token_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    files = sorted(p.name for p in out.iterdir())
    print(f"[export] 完成：{out}\n        文件：{files}")
    (out / "SHENSI_EXPORT.json").write_text(
        json.dumps(
            {
                "ckpt": str(iter_dir),
                "spec": spec_info[2] if spec_info else where,
                "dtype": dtype,
                "tokenizer": tok_where,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        "[export] 下一步：stage4_eval/run_depth_retrieval.py --model "
        f"{out} --data <dr.jsonl> --out-json <score.json>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
