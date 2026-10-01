"""极小链要用的两个本地产物：小 tokenizer 与 HF 格式的小模型。

为什么需要它们：真正的 DSv4 产物（tokenizer、HF 权重）在云端 filestorage 上，本机只做
"链路跑通"的调试。两类产物：

1. `$SHENSI_FS/shensi/models/tiny-tok`：在本地语料上现训的 BPE（vocab 2048），**带
   chat template**——verl 的 rollout 数据要 `apply_chat_template`，没有模板会直接
   `num_samples=0` 崩掉（踩过）；
2. `$SHENSI_FS/shensi/models/tiny-rl`：HF 格式的小模型（几何取 `tiny_model.TINY`），
   RL 的 rollout 侧与 stage3_eval 的 vLLM 都读它。

两处的几何/词表口径必须一致：模型 vocab 取 tokenizer 的真实大小，tokenizer 文件也复制进
模型目录，做成自包含的 HF 目录。PT / SFT 的数据打包用同一个 tokenizer（`--tokenizer-model`）。

用法：
    python -m shensi.recipes.shensi.tiny_artifacts            # 两个都生成
    python -m shensi.recipes.shensi.tiny_artifacts --tokenizer-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common, tiny_model

# 极小档的 chat 模板：roles + 内容顺次拼接（`add_generation_prompt` 时补 assistant 开头）。
# 生产档不用它——正式跑的 tokenizer 自带官方 DSv4 模板。
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)

SPECIAL_TOKENS = ("<unk>", "<|endoftext|>", "</s>", "<s>", "<pad>")


def build_tokenizer(corpus: Path, out: Path, vocab_size: int) -> Path:
    """本地语料上训一个小 BPE，并写进带 chat template 的 `tokenizer_config.json`。"""
    import json as _json

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    texts: list[str] = []
    for f in sorted(corpus.rglob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = _json.loads(line)
                    text = row.get("text") or row.get("content")
                    if isinstance(text, str):
                        texts.append(text)
    if not texts:
        raise SystemExit(f"[tiny] {corpus} 下没找到可用的 jsonl 文本（列名要 text/content）")
    print(f"[tiny] 语料 {len(texts)} 条（来自 {corpus}）")

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        texts,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        bos_token="<s>",
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(str(out))
    print(f"[tiny] tokenizer 写出 {out}（vocab={fast.vocab_size}，带 chat template）")
    return out


def build_model(
    tok_dir: Path, out: Path, seed: int = 0, max_position_embeddings: int = 2048
) -> Path:
    """按 `tiny_model.TINY` 建一个 HF 小模型（vocab 取 tokenizer 的真实大小）并存盘。

    `max_position_embeddings` 给得比 PT 的极小档（128）大：RL 的 rollout 与 stage3_eval 的 vLLM
    会按它校验 `max_model_len`（prompt 512 + response 128 起），128 会被直接拒。
    """
    import torch
    from transformers import AutoTokenizer
    from transformers.models.shensi import ShensiForCausalLM

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    # 与 mcore 侧同一个补齐口径：小 tokenizer 练出来的词表（如 614）不整除 128，而 mcore 会把
    # 词表补齐、哈希嵌入表按补齐值建——HF 侧不补齐就会在加载导出的 ckpt 时撞
    # "deepemb.weight: ckpt(640,128) vs model(614,128)"
    vocab = tiny_model.aligned_vocab_size(len(tok))
    cfg = tiny_model.tiny_shensi_config(
        vocab_size=vocab,
        eos_token_id=tok.eos_token_id,
        max_position_embeddings=max_position_embeddings,
    )

    torch.manual_seed(seed)
    model = ShensiForCausalLM(cfg).to(torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for f in tok_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)

    n = sum(p.numel() for p in model.parameters())
    print(
        f"[tiny] HF 模型写出 {out}（{n / 1e6:.3f}M 参数，vocab={vocab}（tokenizer {len(tok)} 补齐到 {vocab}））"
    )
    return out


def main() -> int:
    paths = common.env_paths()
    models_dir = Path(paths["models"])
    ap = argparse.ArgumentParser(description="Shensi 极小链的本地产物（tokenizer + HF 模型）")
    ap.add_argument("--corpus", default=None, help=f"训 BPE 的语料根，默认 {paths['pre']}")
    ap.add_argument(
        "--out-dir", default=str(models_dir), help="产物目录（默认 $SHENSI_FS/shensi/models）"
    )
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--tokenizer-only", action="store_true")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-position-embeddings", type=int, default=2048)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tok_dir = out_dir / "tiny-tok"
    model_dir = out_dir / "tiny-rl"

    if not args.model_only:
        build_tokenizer(Path(args.corpus or paths["pre"]), tok_dir, args.vocab_size)
    if not args.tokenizer_only:
        if not (tok_dir / "tokenizer.json").is_file():
            raise SystemExit(f"[tiny] 先建 tokenizer：{tok_dir} 里没有 tokenizer.json")
        build_model(
            tok_dir,
            model_dir,
            seed=args.seed,
            max_position_embeddings=args.max_position_embeddings,
        )
        # 记一笔口径，便于对拍（不是 HF 的必需文件）
        (model_dir / "SHENSI_TINY.json").write_text(
            json.dumps(
                {"TINY": tiny_model.TINY, "chat_template": CHAT_TEMPLATE},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
