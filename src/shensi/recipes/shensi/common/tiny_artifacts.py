"""本地产物：小 BPE tokenizer 与 HF 格式的 tiny 模型。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common, tiny_model

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
    """按 `tiny_model.TINY` 建一个 HF 小模型（vocab 取 tokenizer 的真实大小）并存盘。"""
    import torch
    from transformers import AutoTokenizer
    from transformers.models.shensi import ShensiForCausalLM

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    # HF 侧要用与 mcore 相同的词表补齐值（见 tiny_model.VOCAB_ALIGN）
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
