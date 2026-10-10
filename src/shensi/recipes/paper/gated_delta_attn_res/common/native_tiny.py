"""Prepare/run actual Megatron PT -> midtrain -> SFT with tiny synthetic data.

Use --dry-run to inspect commands without a device. This is a separate hardware
gate from tiny_reference; it does not claim to test verl RL or vLLM serving.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def prepare_data(out: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        word: idx
        for idx, word in enumerate(
            ["<pad>", "<bos>", "<eos>", "<unk>"] + [f"w{i}" for i in range(4, 64)]
        )
    }
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=64,
        chat_template="<bos> {% for m in messages %}{{ m['content'] }} <eos> {% endfor %}",
    )
    token_dir = out / "tokenizer"
    tokenizer.save_pretrained(token_dir)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"w{4 + i % 8} w12 w13"},
                {"role": "assistant", "content": "w14 w15 w16"},
            ]
        }
        for i in range(32)
    ]
    data = out / "sft.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return token_dir, data


def configurations(out: Path, backend: str, steps: int, optimizer: str):
    from shensi.recipes.paper.gated_delta_attn_res.common.config import smoke_config

    token_dir, data = prepare_data(out)
    result = []
    previous = None
    for stage in ("stage1_pretrain", "stage2_midtrain", "stage1_sft"):
        cfg = smoke_config(stage)
        cfg["experiment"]["exp_dir"] = str(out / stage)
        cfg["experiment"]["runner"]["nproc_per_node"] = 1
        cfg["experiment"]["envs"]["SHENSI_BACKEND"] = backend
        if backend == "npu":
            cfg["experiment"]["envs"].pop("CUDA_VISIBLE_DEVICES", None)
            cfg["experiment"]["envs"].pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
        train = cfg["train"]
        model = train["model"]
        model.update(
            num_layers=3,
            hidden_size=32,
            ffn_hidden_size=64,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=16,
            seq_length=16,
            max_position_embeddings=64,
            micro_batch_size=1,
            global_batch_size=2,
            train_iters=steps,
            eval_iters=0,
            finetune=previous is not None,
            spec=[
                "shensi.recipes.paper.gated_delta_attn_res.common.models.megatron.gdar_spec",
                "gdar_layer_spec_tiny",
            ],
        )
        model.pop("tokenizer_model", None)
        if optimizer == "adam":
            model["optimizer"] = {
                "optimizer": "adam",
                "lr": 1e-4,
                "min_lr": 1e-5,
                "lr_decay_style": "constant",
                "lr_warmup_iters": 0,
                "weight_decay": 0.0,
                "clip_grad": 1.0,
            }
        checkpoint = train["system"]["checkpoint"]
        checkpoint.update(
            save=str(out / stage / "ckpt"),
            load=previous,
            save_interval=steps,
            no_save_optim=False,
            no_save_rng=False,
            no_load_optim=previous is not None,
            no_load_rng=previous is not None,
        )
        train["data"].pop("tokenizer_model", None)
        train["data"].pop("data_path", None)
        train["data"]["tokenizer"] = {
            "tokenizer_type": "SFTTokenizer" if stage == "stage1_sft" else "NullTokenizer",
            "vocab_size": 64,
            "make_vocab_size_divisible_by": 64,
        }
        if stage == "stage1_sft":
            train["data"]["tokenizer"]["tokenizer_model"] = str(token_dir)
            train["data"]["data_path"] = str(data)
            train["data"]["mock_data"] = False
        else:
            train["data"]["mock_data"] = True
        result.append((stage, cfg))
        previous = checkpoint["save"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--optimizer", choices=("adam", "adaptive_muon"), default="adam")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        os.environ["SHENSI_BACKEND"] = args.backend
        if args.backend == "npu":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shensi.utils.ascend_env",
                    "--scope",
                    "train",
                    "--json",
                    str(out / "preflight.json"),
                ],
                check=True,
            )
    from shensi.recipes.paper.gated_delta_attn_res.common.train import launcher

    configs = configurations(out, args.backend, args.steps, args.optimizer)
    report = {
        "status": "planned",
        "backend": args.backend,
        "stages": [],
        "scope": "actual Megatron pretrain/midtrain/SFT; excludes RL/OPD/serving",
    }
    for stage, cfg in configs:
        run_dir = Path(cfg["experiment"]["exp_dir"])
        launcher.write_run_dir(cfg, run_dir)
        item = {"stage": stage, "command": launcher.build_command(cfg), "status": "planned"}
        report["stages"].append(item)
        (out / "native_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rc = launcher.launch(cfg, run_dir, dry_run=args.dry_run)
        if args.dry_run:
            continue
        log = (run_dir / "logs/host_0_localhost.output").read_text(encoding="utf-8")
        iterations = re.findall(r"iteration\s+(\d+)\s*/", log)
        ckpt = Path(cfg["train"]["system"]["checkpoint"]["save"])
        item["returncode"] = rc
        item["status"] = (
            "passed"
            if (
                rc == 0
                and iterations
                and int(iterations[-1]) == args.steps
                and "after training is done" in log
                and (ckpt / "latest_checkpointed_iteration.txt").is_file()
            )
            else "failed"
        )
        if item["status"] == "failed":
            report["status"] = "failed"
            (out / "native_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise RuntimeError(f"{stage} failed; inspect {run_dir / 'logs'}")
    report["status"] = "planned" if args.dry_run else "passed"
    (out / "native_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
