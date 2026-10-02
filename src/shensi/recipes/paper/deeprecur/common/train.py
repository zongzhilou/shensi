"""训练公共核：Trainer 组装、分组学习率、早停、dry-run 计划与产物落盘。"""

from __future__ import annotations

import math
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.config import (
    add_common_train_args,
    build_config,
    profile_from_args,
    smoke_config,
)
from shensi.recipes.paper.deeprecur.common.data import (
    MockVLDataset,
    VLChatDataset,
    VLCollator,
    load_rows,
)
from shensi.recipes.paper.deeprecur.common.model import (
    apply_freeze,
    load_model,
    load_processor,
    split_param_groups,
)
from shensi.recipes.paper.deeprecur.common.paths import env_paths


def _training_args(cfg: dict, n_train: int):
    from transformers import TrainingArguments

    train = cfg["train"]
    world = max(1, _world_size())
    micro = int(train.get("micro_batch_size", 1))
    accum = max(1, int(train["global_batch_size"]) // (micro * world))
    epochs = train.get("epochs", 1)
    max_steps = train.get("max_steps")
    total = max_steps or max(1, math.ceil(n_train / (micro * accum * world)) * int(epochs))
    warmup = round(float(train.get("warmup_ratio", 0.03)) * total)
    early = cfg["experiment"].get("early_stop")
    eval_interval = int(cfg["experiment"].get("eval_interval") or 0)
    args = TrainingArguments(
        output_dir=cfg["experiment"]["exp_dir"],
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,  # max_steps 给出时它被忽略，但这版 Trainer 必须传数字
        max_steps=max_steps,
        learning_rate=float(train["lr"]),
        lr_scheduler_type=train.get("lr_scheduler", "cosine"),
        warmup_steps=warmup,
        weight_decay=float(train.get("weight_decay", 0.0)),
        max_grad_norm=float(train.get("max_grad_norm", 1.0)),
        adam_beta1=float(train.get("adam_beta1", 0.9)),
        adam_beta2=float(train.get("adam_beta2", 0.999)),
        bf16=bool(train.get("bf16", True)),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", False)),
        logging_steps=int(train.get("logging_steps", 10)),
        save_steps=int(cfg["experiment"].get("save_steps", 500)),
        save_strategy="steps",
        eval_strategy="steps" if eval_interval > 0 else "no",
        eval_steps=eval_interval or 500,
        eval_accumulation_steps=1,
        seed=int(cfg["experiment"].get("seed", 42)),
        dataloader_num_workers=int(cfg["data"].get("num_workers", 4)),
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        metric_for_best_model="eval_loss" if (early and not cfg["experiment"].get("no_early_stop")) else None,
        greater_is_better=False,
    )
    return args, accum, world, total, warmup


def _world_size() -> int:
    import torch

    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def _build_optimizer(model, cfg: dict):
    """分组学习率：DeepStack-V/HD 的视觉编码器（``vision_lr``），否则单组。"""
    from torch.optim.adamw import AdamW

    train = cfg["train"]
    groups = split_param_groups(model, float(train["lr"]), train.get("vision_lr"))
    optimizer = AdamW(
        groups,
        lr=float(train["lr"]),
        weight_decay=float(train.get("weight_decay", 0.0)),
        betas=(float(train.get("adam_beta1", 0.9)), float(train.get("adam_beta2", 0.999))),
    )
    return optimizer, None  # scheduler 交给 Trainer


def _datasets(cfg: dict, stage: str, smoke: bool, allow_missing: bool = False):
    data = cfg["data"]
    if smoke:
        rows = None
        train_ds, val_ds = MockVLDataset(n=8), MockVLDataset(n=2)
    else:
        jsonl = data.get("jsonl")
        if not jsonl:
            if allow_missing:
                return None, None, None, True
            raise SystemExit(
                f"[deeprecur] 没有训练数据：先跑 data_prep.py --prepare（找 {data.get('jsonl')}）"
            )
        all_rows = load_rows(jsonl)
        cut = max(1, int(len(all_rows) * 0.02))
        rows = all_rows
        train_ds = VLChatDataset(all_rows[cut:], Path(jsonl).parent)
        val_jsonl = data.get("val_jsonl")
        if val_jsonl and Path(val_jsonl).is_file():
            val_ds = VLChatDataset(load_rows(val_jsonl), Path(val_jsonl).parent)
        else:
            val_ds = VLChatDataset(all_rows[:cut], Path(jsonl).parent)
    return train_ds, val_ds, rows, False


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """训练入口（stage 的 train.py 调它）。"""
    import argparse

    from transformers import EarlyStoppingCallback, Trainer

    parser = argparse.ArgumentParser(description=f"{stage} 训练（Qwen3-VL 占位）")
    add_common_train_args(parser)
    args = parser.parse_args(argv)

    if args.smoke:
        cfg = smoke_config(stage, args.override)
        cfg["experiment"]["no_early_stop"] = True  # 冒烟几步不看早停
    else:
        profile = profile_from_args(args.config, args.profile)
        data_dir = Path(args.data_dir) if args.data_dir else Path(env_paths()["data"]) / stage
        cfg = build_config(
            stage, profile, args.override, data_dir, tokens=args.tokens, load_ckpt=args.load
        )
        if args.no_early_stop:
            cfg["experiment"]["no_early_stop"] = True
        elif args.early_stop:
            cfg["experiment"]["early_stop"] = args.early_stop

    train_ds, val_ds, _rows, pending_data = _datasets(cfg, stage, bool(args.smoke), allow_missing=args.dry_run)
    args_tf, accum, world, total, warmup = _training_args(cfg, len(train_ds) if train_ds is not None else 1)
    if args.dry_run:
        # 不装模型不拉权重：接口检查只看合并后的配置与计划（数据可以还没准备）
        _print_plan(cfg, stage, args, train_ds, val_ds, total, accum, world, warmup)
        if pending_data:
            print(f"[deeprecur] dry-run：数据未准备（{cfg['data'].get('jsonl')}）——正式跑前先 data_prep.py --prepare")
        return 0
    model = load_model(cfg)
    freeze = cfg["train"].get("freeze", [])
    apply_freeze(model, freeze)
    processor = load_processor(cfg)

    _print_plan(cfg, stage, args, train_ds, val_ds, total, accum, world, warmup)

    collator = VLCollator(processor, Path(cfg["data"]["jsonl"]).parent if cfg["data"].get("jsonl") else None, int(cfg["model"].get("max_length", 2048)))
    callbacks = []
    if args_tf.metric_for_best_model:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg["experiment"].get("early_stop", 5))
            )
        )
    kwargs = {}
    if cfg["train"].get("vision_lr") is not None:
        kwargs["optimizers"] = _build_optimizer(model, cfg)
    trainer = Trainer(
        model=model,
        args=args_tf,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        callbacks=callbacks,
        **kwargs,
    )
    trainer.train()
    trainer.save_model(str(Path(cfg["train"]["save_dir"]) / "final"))
    processor.save_pretrained(str(Path(cfg["train"]["save_dir"]) / "final"))
    _dump_run(cfg, stage, args)
    print(f"[deeprecur] 完成：检查点在 {cfg['train']['save_dir']}")
    return 0


def _print_plan(cfg, stage, args, train_ds, val_ds, total, accum, world, warmup) -> None:
    train = cfg["train"]
    print(
        f"[deeprecur] {stage}/{args.profile}: 模型="
        f"{'tiny-random' if cfg['model'].get('tiny') else cfg['model']['placeholder']['name']}"
        f" | freeze={train.get('freeze')}"
        f" | batch={train['global_batch_size']}（micro×accum×world={train.get('micro_batch_size', 1)}×{accum}×{world}）"
        f" | lr={train['lr']}" + (f" | vision_lr={train['vision_lr']}" if train.get("vision_lr") else "")
        + f" | {train.get('lr_scheduler', 'cosine')} warmup {warmup}/{total} 步"
        f" | 数据 train={len(train_ds) if train_ds is not None else '未准备'}"
        f" val={len(val_ds) if val_ds is not None else '未准备'}"
    )


def _dump_run(cfg, stage, args) -> None:
    """最终配置与命令落 <exp_dir>，可照抄复跑。"""
    from omegaconf import OmegaConf

    exp_dir = Path(cfg["experiment"]["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), exp_dir / "config.yaml")
    (exp_dir / "run.sh").write_text(
        "#!/usr/bin/env bash\n# 复跑命令\npython train.py --profile "
        f"{getattr(args, 'profile', 'default')} "
        + " ".join(f"--set {item}" for item in getattr(args, "override", [])),
        encoding="utf-8",
    )
