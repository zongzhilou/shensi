"""共享训练循环：预训练 / SFT / RFT / OPD 全走这里。

- 数据：VLDataset（jsonl schema 见 vl_data.py）；
- 模型：ShensiVLModel（DSV4F + ViT + projector，modeling_vl.py）；
- OPD（论文 §2.5.4）：学生基于**自己生成的轨迹**训练，损失 = CE + Σ wᵢ·D_KL(π_θ ‖ π_Eᵢ)
  （反向 KL、全词表 logits；轨迹由 stage4_opd/data_prep.py 先用学生 ckpt 自采样好）；
- 单机多卡用 torchrun 起（DDP），单卡直接 python；bf16 autocast + 梯度累积；
- checkpoint = HF save_pretrained（tokenizer 一并落，供 RL/vLLM 直接吃 HF 目录）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch


def build_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["train"]["model"]["tokenizer_dir"])


def pad_token_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def cosine_lr(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def make_model(cfg: dict, image_token_id: int):
    from shensi.recipes.shensi_vl.common.train.modeling_vl import ShensiVLModel

    m = cfg["train"]["model"]
    model = ShensiVLModel(
        llm_path=m["llm_path"],
        vision_path=m["vision_path"],
        image_token_id=image_token_id,
        allow_random_vision=bool(m.get("allow_random_vision", False)),
        freeze_llm=bool(m.get("freeze_llm", False)),
        freeze_vision=bool(m.get("freeze_vision", False)),
        dtype=getattr(torch, m.get("dtype", "float32")),
    )
    # 词表扩展（DSV4F + 原语/图像 token）：行数对齐 128 倍数与 HF 侧一致
    vocab, aligned = m.get("vocab_size") or 0, 0
    if vocab:
        aligned = ((vocab + 127) // 128) * 128
        if aligned != model.llm.get_input_embeddings().weight.shape[0]:
            model.resize_embeddings(aligned)
    return model


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """D_KL(π_θ ‖ π_E)：逐位置全词表，mask 掉不参训的位置。"""
    s_logp = torch.log_softmax(student_logits.float(), dim=-1)
    t_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


def run(cfg: dict, *, mode: str = "sft") -> int:
    """mode: pretrain | sft | rft | opd（差异只在数据与损失，见 stage4_opd）。"""
    from shensi.recipes.shensi_vl.common import processors as proc_mod
    from shensi.recipes.shensi_vl.common import vl_tokens
    from shensi.recipes.shensi_vl.common.train import vl_data

    t = cfg["train"]
    tok = build_tokenizer(cfg)
    img_id = vl_tokens.token_ids(tok)[vl_tokens.IMAGE_TOKEN]
    model = make_model(cfg, img_id)
    iproc = proc_mod.load_image_processor(t["model"]["processor_path"])

    data_files = t["data"]["train_jsonl"]
    if isinstance(data_files, str):
        data_files = [data_files]
    train_ds = vl_data.VLDataset(
        data_files, tok, iproc,
        max_len=int(t["model"].get("max_seq_length", 8192)),
        limit=t["data"].get("limit"),
    )
    val_files = t["data"].get("val_jsonl")
    val_ds = (
        vl_data.VLDataset([val_files], tok, iproc, max_len=t["model"].get("max_seq_length", 8192))
        if val_files
        else None
    )
    coll = lambda b: vl_data.collate(b, pad_token_id(tok))  # noqa: E731
    gbs, mbs = int(t["batch"]["global_batch_size"]), int(t["batch"]["micro_batch_size"])
    world = max(1, int(__import__("os").environ.get("WORLD_SIZE", "1")))
    accum = max(1, gbs // (mbs * world))
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=mbs, shuffle=True, collate_fn=coll,
        num_workers=int(t["batch"].get("num_workers", 2)), drop_last=True,
    )
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(__import__("os").environ.get("LOCAL_RANK", "0"))]
        )
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=mbs, sampler=sampler, collate_fn=coll,
            num_workers=int(t["batch"].get("num_workers", 2)), drop_last=True,
        )
    raw = model.module if world > 1 else model

    total = int(t["batch"]["train_iters"])
    o = t["optim"]
    params = [p for p in raw.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.95))),
        weight_decay=float(o.get("weight_decay", 0.0)),
    )
    warmup = int(total * float(o.get("warmup_ratio", 0.01)))
    amp_dtype = getattr(torch, t["model"].get("autocast_dtype", "bfloat16"))
    device = raw.device

    # OPD 教师（§2.5.4：{E_TwG, E_TwP}，wᵢ 加权；本配方默认等权两教师）
    teachers = []
    if mode == "opd":
        weights = t.get("opd", {}).get("weights", [])
        for i, path in enumerate(t.get("opd", {}).get("teachers", [])):
            tc = dict(t["model"])
            tc["llm_path"] = path
            tc["freeze_llm"] = tc["freeze_vision"] = True
            tcfg = {"train": {"model": tc}}
            w = float(weights[i]) if i < len(weights) else 1.0
            teachers.append((w, make_model(tcfg, img_id).eval()))
        if not teachers:
            raise SystemExit("[opd] teachers 为空：stage4_opd 的配置要给专家目录列表")

    save_dir = Path(cfg["experiment"]["exp_dir"])
    ckpt_dir = Path(t["model"].get("save") or save_dir)
    micro, opt_step, t0, running = 0, 0, time.time(), 0.0
    log_every = int(t["batch"].get("log_steps", 10))
    opt.zero_grad(set_to_none=True)
    model.train()
    done = False
    keys = ("input_ids", "attention_mask", "pixel_values", "image_positions", "labels")
    while not done:
        if world > 1:
            sampler.set_epoch(opt_step)
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, student_logits = forward_with_logits(raw, batch, keys)
                if mode == "opd" and teachers:
                    # 反向 KL 在响应位上算（全词表 logits）；教师 fp32 前向、无梯度
                    resp_mask = (batch["labels"][:, 1:] != -100).float()
                    kl_total = student_logits.new_zeros(())
                    for w, teacher in teachers:
                        tl = teacher.teacher_logits(
                            batch["input_ids"], batch["attention_mask"],
                            batch["pixel_values"], batch["image_positions"],
                        )
                        kl_total = kl_total + w * reverse_kl(
                            student_logits[:, :-1, :], tl[:, :-1, :], resp_mask
                        )
                    loss = loss + float(t.get("opd", {}).get("kl_coef", 1.0)) * kl_total
            (loss / accum).backward()
            running += float(loss.detach())
            micro += 1
            if micro % accum == 0:
                opt_step += 1
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(opt_step, total, warmup, float(o["lr"]), float(o.get("min_lr", o["lr"] * 0.1)))
                torch.nn.utils.clip_grad_norm_(params, float(o.get("clip_grad", 1.0)))
                opt.step()
                opt.zero_grad(set_to_none=True)
            if micro % log_every == 0:
                sps = (micro * mbs * world) / (time.time() - t0)
                print(f"[train] step {opt_step}/{total} loss={running / log_every:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} samples/s={sps:.1f}", flush=True)
                running = 0.0
            if val_ds is not None and micro % (accum * int(t["batch"].get("eval_steps", 200))) == 0:
                vl = eval_loss(raw, val_ds, coll, device, amp_dtype)
                print(f"[train] val loss value={vl:.4f} @ step {opt_step}", flush=True)
            if micro % (accum * int(t["batch"].get("save_steps", 500))) == 0:
                save(raw, tok, ckpt_dir / f"iter_{opt_step:07d}")
            if opt_step >= total:
                done = True
                break
    save(raw, tok, ckpt_dir / "final")
    print(f"[train] 完成：{total} 步，ckpt 在 {ckpt_dir / 'final'}")
    return 0


def forward_with_logits(model_wrapper, batch: dict, keys: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """前向：返回 (loss, logits)。DDP 与裸模块都走 __call__。"""
    return model_wrapper(**{k: batch[k] for k in keys if k in batch})


def eval_loss(model, ds, coll, device, amp_dtype) -> float:
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=coll)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, _ = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_positions", "labels") if k in batch})
            tot += float(loss)
            n += 1
    model.train()
    return tot / max(1, n)


def save(model, tok, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (model.module if hasattr(model, "module") else model).llm.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[train] ckpt 已存 {out}", flush=True)
