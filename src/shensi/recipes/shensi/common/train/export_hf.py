"""把 mcore 的 `torch_dist` 检查点导成一个 HF 目录（RL 的 rollout 与 stage3_eval 的 vLLM 都读 HF）。

链路里每个 stage 的产物都是 mcore ckpt，而 RL / 评测要 HF 目录（`model.path` /
`serving.model_path`）——这一步就是两者之间的桥。实现全部复用 Megatron-Bridge 已有的能力：

    provider = AutoBridge.from_hf_config(hf_cfg).to_megatron_provider(load_weights=False)
    model    = provider.provide()
    sd       = dist_checkpointing.load(model.sharded_state_dict(), ckpt)   # mcore 的 torch_dist
    AutoBridge.save_hf_pretrained([model], out)                            # mcore → HF

用法（极小链）：

    python -m shensi.recipes.shensi.common.train.export_hf \
        --ckpt $SHENSI_FS/shensi/ckpt/stage1_sft_debug --out $SHENSI_FS/shensi/models/sft-hf --tiny
    # 之后：stage2_rl / stage3_eval 用 --set model.path=<out> / serving.model_path=<out>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common, tiny_model


def _latest_iter(ckpt_dir: Path) -> str | None:
    p = ckpt_dir / "latest_checkpointed_iteration.txt"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def _resolve_iter_dir(ckpt_dir: Path, load_iter: str | None) -> Path:
    """`--ckpt` 给的是父目录（里面有 `latest_checkpointed_iteration.txt`）时，落到 `iter_XXXXXXX/`。

    mcore 的 `dist_checkpointing.load` 要的就是那个迭代目录（直接给父目录会报
    "is not a distributed checkpoint"）。
    """
    if (ckpt_dir / "metadata.json").is_file() or (ckpt_dir / "common.pt").is_file():
        return ckpt_dir
    it = load_iter or _latest_iter(ckpt_dir)
    if not it:
        raise SystemExit(
            f"{ckpt_dir} 里既没有 checkpoint 元数据，也没有 latest_checkpointed_iteration.txt"
        )
    d = ckpt_dir / f"iter_{int(it):07d}"
    if not d.is_dir():
        raise SystemExit(f"找不到迭代目录：{d}")
    return d


def _init_distributed(device: str) -> None:
    """单进程把分布式与 mcore 的并行组点起来：`sharded_state_dict()` / 建模型都要它们在场。"""
    import os

    import torch
    from megatron.core import parallel_state as mpu

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
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
        # 建模型会在 GPU 上初始化参数，需要 CUDA 的 model-parallel RNG（上游 initialize_megatron 也做这一步）
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)


def _bind_pg_collection(provider) -> None:
    """Bridge 的 provider 建模型时要 `self._pg_collection`（训练路径由 provide_distributed_model 塞）。

    这里不套 DDP（导出只要一份模型权重），所以按 Bridge 自己的方式把它绑上。
    """
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()  # noqa: SLF001


def _hf_config(args, paths: dict, mtp_layers: int = 0):
    """几何来源：`--hf-config <HF 目录/config.json>`（生产）或 `--tiny`（极小链）。"""
    if args.hf_config:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.hf_config)
        declared = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
        if mtp_layers and declared != mtp_layers:
            raise SystemExit(
                f"几何来源说 MTP={declared}，但检查点的元数据里有 {mtp_layers} 层 MTP："
                "两边必须一致（否则整棵 mtp.* 子树会被静默丢掉）。换对得上的 config，"
                "或去掉 --hf-config 让导出按检查点自动探测。"
            )
        return cfg
    if args.tiny:
        from transformers import AutoTokenizer

        tok_dir = Path(args.tokenizer or (Path(paths["models"]) / "tiny-tok"))
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
        return tiny_model.tiny_shensi_config(
            # 与训练侧同一个词表补齐口径（见 tiny_artifacts.build_model 的注释）
            vocab_size=tiny_model.aligned_vocab_size(len(tok)),
            eos_token_id=tok.eos_token_id,
            max_position_embeddings=args.max_position_embeddings,
            num_nextn_predict_layers=mtp_layers,
        )
    raise SystemExit("给一个几何来源：--hf-config <HF 目录或 config.json>，或 --tiny")


def _ckpt_mtp_layers(ckpt_dir: Path) -> int:
    """从 torch_dist 检查点的元数据里数 MTP 层数（键形如 `mtp.layers.{k}.*`）。

    极小链的几何来自 `--tiny`，而 MTP 层数是**检查点自己的属性**：不带 `--mtp` 时按这里实测，
    免得把带 MTP 的 ckpt 导成不带 MTP 的 HF 产物（那会静默丢掉整棵 `mtp.*` 子树）。
    """
    import re

    latest = ckpt_dir / "latest_checkpointed_iteration.txt"
    if latest.is_file():
        iteration = latest.read_text(encoding="utf-8").strip()
        iter_dir = ckpt_dir / f"iter_{int(iteration):07d}"
    else:
        cands = sorted(p for p in ckpt_dir.glob("iter_*") if p.is_dir())
        if not cands:
            return 0
        iter_dir = cands[-1]
    try:
        from torch.distributed.checkpoint import FileSystemReader

        keys = list(FileSystemReader(str(iter_dir)).read_metadata().state_dict_metadata.keys())
    except Exception as exc:  # noqa: BLE001 - 读不出来就当没有 MTP，并在调用处提示
        print(f"[export] 读不出检查点元数据（{type(exc).__name__}: {exc}），按 MTP=0 处理")
        return 0
    depths = {int(m.group(1)) for k in keys if (m := re.match(r"mtp\.layers\.(\d+)\.", k))}
    return (max(depths) + 1) if depths else 0


def _tokenizer_dir(args, paths: dict) -> Path | None:
    if args.tokenizer:
        return Path(args.tokenizer)
    if args.tiny:
        d = Path(paths["models"]) / "tiny-tok"
        return d if d.is_dir() else None
    if args.hf_config:
        d = Path(args.hf_config)
        return d if d.is_dir() else None
    return None


def main(argv: list[str] | None = None) -> int:
    paths = common.env_paths()
    ap = argparse.ArgumentParser(description="mcore torch_dist ckpt → HF 目录（给 RL / vLLM 用）")
    ap.add_argument("--ckpt", required=True, help="mcore 检查点目录（torch_dist）")
    ap.add_argument(
        "--out", required=True, help="输出目录（HF 格式：config.json + safetensors + tokenizer）"
    )
    ap.add_argument("--hf-config", default=None, help="几何来源：HF 目录或 config.json")
    ap.add_argument("--tiny", action="store_true", help="用 tiny_model.TINY 的几何（极小链）")
    ap.add_argument(
        "--mtp",
        type=int,
        default=None,
        help="MTP 层数（默认按检查点元数据自动探测；带 MTP 的 ckpt 不要手写成 0）",
    )
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer 目录（默认按 --tiny / --hf-config 推）"
    )
    ap.add_argument("--max-position-embeddings", type=int, default=16384)
    ap.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda"), help="导出时模型放哪（默认 CPU）"
    )
    ap.add_argument(
        "--load-iter",
        default=None,
        help="载入哪个 iteration（默认取 latest_checkpointed_iteration.txt）",
    )
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.ckpt)
    out = Path(args.out)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"检查点目录不存在：{ckpt_dir}")

    _init_distributed(args.device)

    mtp_layers = args.mtp if args.mtp is not None else _ckpt_mtp_layers(ckpt_dir)
    hf_cfg = _hf_config(args, paths, mtp_layers=mtp_layers)
    print(
        f"[export] MTP 层数：{mtp_layers}（{'命令行指定' if args.mtp is not None else '按检查点元数据探测'}）"
    )
    print(
        f"[export] 几何：vocab={getattr(hf_cfg, 'vocab_size', '?')} "
        f"layers={getattr(hf_cfg, 'num_hidden_layers', '?')} "
        f"hidden={getattr(hf_cfg, 'hidden_size', '?')}"
    )

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    bridge = AutoBridge.from_hf_config(hf_cfg)
    provider = bridge.to_megatron_provider(load_weights=False)
    # 导出是单进程活：并行度全给 1（ckpt 也要是同样并行度存的）
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.bf16 = True
    provider.fp16 = False
    provider.seq_length = getattr(hf_cfg, "max_position_embeddings", 4096)
    # 词表要和训练时一样补齐：训练侧把 tokenizer 的真实词表（如 614）按 128 对齐到 640，
    # 哈希嵌入表就是 640 行——不补就会撞 "Global shape mismatch ... deepemb.weight"
    provider.should_pad_vocab = True
    provider.make_vocab_size_divisible_by = 128
    # hash-MoE 的 deepemb 表按 `config.actual_vocab_size` 建（`ShensiHashMLP` 的实现），
    # 训练侧的训练入口显式把它设成补齐后的词表（builders.py），导出侧也要一样
    from megatron.training.vocab_utils import calculate_padded_vocab_size

    provider.actual_vocab_size = int(
        calculate_padded_vocab_size(
            provider.vocab_size,
            provider.make_vocab_size_divisible_by,
            provider.tensor_model_parallel_size,
        )
    )
    # 只导权重，不走前向：DSA 内核用哪个无所谓，但必须是合法值（本机没有 cudnn/tilelang 那套）
    if hasattr(provider, "dsa_kernel_backend"):
        provider.dsa_kernel_backend = "none"
    # 导出不需要 CUDA graph；而且 Bridge 这条 HF→mcore 的路会把它设成老式的字符串，
    # 上游 transformer_layer 拿 `CudaGraphModule.attn in <str>` 会 TypeError（踩过）
    if hasattr(provider, "cuda_graph_modules"):
        provider.cuda_graph_modules = None

    _bind_pg_collection(provider)
    model = provider.provide()
    if args.device == "cuda":
        model = model.cuda()

    iter_dir = _resolve_iter_dir(ckpt_dir, args.load_iter)
    print(
        f"[export] 建好 mcore 模型（{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M 参数，"
        f"device={args.device}）；载入 {iter_dir}"
    )

    from megatron.core import dist_checkpointing

    model_sharded = model.sharded_state_dict()
    loaded = dist_checkpointing.load(model_sharded, str(iter_dir))
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    real_missing = [k for k in missing if not k.endswith("_extra_state")]
    if real_missing:
        raise SystemExit(
            f"[export] 有 {len(real_missing)} 个参数没在检查点里找到（前几个）：{real_missing[:5]}"
        )
    print(
        f"[export] 权重已载入（missing={len(missing)}（都是 _extra_state）unexpected={len(unexpected)}）"
    )

    out.mkdir(parents=True, exist_ok=True)
    try:
        bridge.save_hf_pretrained([model], str(out), show_progress=True)
    except ModuleNotFoundError as exc:  # 只可能是 modelopt（Bridge 的量化分支无条件 import 它）
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

    tok_dir = _tokenizer_dir(args, paths)
    if tok_dir and tok_dir.is_dir():
        for f in tok_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
        print(f"[export] tokenizer 复制自 {tok_dir}")
    else:
        print("[export] 没找到 tokenizer 目录，HF 目录里只有权重与 config（vLLM 起服务前记得补上）")

    files = sorted(p.name for p in out.iterdir())
    print(f"[export] 完成：{out}\n        文件：{files}")
    (out / "SHENSI_EXPORT.json").write_text(
        json.dumps(
            {"ckpt": str(iter_dir), "iter": args.load_iter or _latest_iter(ckpt_dir)},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
