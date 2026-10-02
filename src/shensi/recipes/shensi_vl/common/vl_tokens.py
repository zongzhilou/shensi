"""原语/图像特殊 token：在 DSV4F tokenizer 之上做**增量**扩展（基座词表一字不动）。

论文把 <|ref|>/<|box|>/<|point|> 等定义为"vocabulary 内的特殊 token"（§2.3.4）；DSV4F 的
tokenizer 没有这组 token，所以在加载后追加 7 个（6 原语 + 1 图像占位），落一份副本目录，
词表只增 7 行（模型侧 resize_token_embeddings，行数补齐到 128 的倍数由训练循环负责）。

`--profile tiny` / 不想动词表时，可用"文本模式"：原语按普通文本切分（render 不注入 id），
配置 `train.model.extend_tokenizer: false`。
"""

from __future__ import annotations

import shutil
from pathlib import Path

PRIMITIVE_TOKENS = (
    "<|ref|>",
    "<|/ref|>",
    "<|box|>",
    "<|/box|>",
    "<|point|>",
    "<|/point|>",
)
IMAGE_TOKEN = "<|image_pad|>"  # 图像槽位：渲染时按 grid 展开成 N 个
EXTENDED_TOKENS = (*PRIMITIVE_TOKENS, IMAGE_TOKEN)


def extend_tokenizer(src_dir: str | Path, out_dir: str | Path) -> Path:
    """DSV4F tokenizer → 追加扩展 token 的副本（幂等：out 已有且 token 数一致就复用）。"""
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    if (out_dir / "tokenizer_config.json").is_file():
        try:
            tok = AutoTokenizer.from_pretrained(str(out_dir))
            if all(t in tok.get_vocab() for t in EXTENDED_TOKENS):
                return out_dir
        except Exception:  # noqa: BLE001  坏副本就重建
            shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(src_dir))
    n_before = len(tok)
    tok.add_tokens(list(EXTENDED_TOKENS), special_tokens=True)
    tok.save_pretrained(str(out_dir))
    print(f"[vl_tokens] {src_dir} ({n_before}) → {out_dir} ({len(tok)})：+{len(tok) - n_before} 特殊 token")
    return out_dir


def token_ids(tok) -> dict:
    """扩展 token 的 id 表（缺哪个就报哪个——说明 tokenizer 没扩展）。"""
    vocab = tok.get_vocab()
    missing = [t for t in EXTENDED_TOKENS if t not in vocab]
    if missing:
        raise SystemExit(
            f"[vl_tokens] tokenizer 缺扩展 token：{missing} —— 先跑 extend_tokenizer 或换 "
            f"SHENSI_VL_TOKENIZER 指到扩展副本（{len(missing)} 个）"
        )
    return {t: vocab[t] for t in EXTENDED_TOKENS}
