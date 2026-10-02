"""LLaVA 式 VL 模型：DSV4F（语言骨干，HF 权重）+ ViT + projector。

口径对齐论文 §2.2：
  - 语言骨干 = DeepSeek-V4-Flash（HF 目录，与 shensi 的 RL model.path 同一份权重）；
  - 视觉塔 = 任意 HF ViT。论文的 DeepSeek-ViT 是 in-house 未开源，开源替代推荐与 Kimi K3
    image_processor 同 patch 口径的 ViT 权重（`$SHENSI_FS/models/Kimi-K3-ViT`）；路径不存在时
    随机初始化一个小 ViT（仅冒烟用，配置里显式写 `allow_random_vision: true` 才放行）；
  - projector = LayerNorm → Linear → GELU → Linear（视觉特征 → LLM hidden）；
  - 图像特征按 ``<|image_pad|>`` 的位置**拼接**进 token embedding 序列（替换该位的文本 embedding），
    loss mask 已在渲染时把图像位屏蔽。

 forward(images, input_ids, image_positions, labels) → (loss, logits)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


def _load_vision(path: str, allow_random: bool, device) -> tuple[nn.Module, int]:
    """HF ViT 目录 → (vision tower, feature_dim)；没有权重且放行时随机建一个小 ViT。"""
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
        cfg = model.config
        dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "vision_hidden_size", None)
        if dim is None:
            dim = model.config.__dict__.get("hidden_size", 768)
        return model.eval(), int(dim)
    except Exception as exc:  # noqa: BLE001
        if not allow_random:
            raise SystemExit(
                f"[modeling_vl] 视觉塔加载失败（{path}: {type(exc).__name__}: {exc}）。"
                "生产档请给 HF ViT 权重目录（SHENSI_VL_VISION）；冒烟档在配置里开 allow_random_vision。"
            ) from exc
    from transformers import ViTConfig, ViTModel

    cfg = ViTConfig(
        image_size=224,
        patch_size=14,
        hidden_size=192,
        num_hidden_layers=4,
        num_attention_heads=3,
        intermediate_size=768,
    )
    return ViTModel(cfg).to(device).eval(), cfg.hidden_size


class ShensiVLModel(nn.Module):
    def __init__(self, llm_path: str, vision_path: str, image_token_id: int,
                 *, allow_random_vision: bool = False, freeze_llm: bool = False,
                 freeze_vision: bool = False, device: str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=dtype or torch.float32, trust_remote_code=True
        ).to(self.device)
        self.vision, vdim = _load_vision(vision_path, allow_random_vision, self.device)
        self.projector = Projector(vdim, self.llm.config.hidden_size).to(self.device)
        self.image_token_id = int(image_token_id)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
            self.vision.eval()

    def resize_embeddings(self, new_vocab: int) -> None:
        """词表加了原语/图像特殊 token 后调用一次（行数补齐交给调用方对齐 128 倍数）。"""
        self.llm.resize_token_embeddings(new_vocab)

    def _embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None,
               image_positions: torch.Tensor | None) -> torch.Tensor:
        embeds = self.llm.get_input_embeddings()(input_ids)
        if pixel_values is None or image_positions is None or int(image_positions.sum()) == 0:
            return embeds
        feats = self.vision(pixel_values.to(self.vision.device)).last_hidden_state
        feats = self.projector(feats.to(embeds.dtype))
        # grid 拉平后逐样本回填：<|image_pad|> 的位置按顺序消费视觉 token
        flat = feats.reshape(-1, feats.shape[-1])
        mask = image_positions.to(embeds.device)
        idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if idx.numel() != flat.shape[0]:
            raise SystemExit(
                f"[modeling_vl] 图像 token 数不匹配：视觉特征 {flat.shape[0]} vs 序列占位 {idx.numel()}"
                "（检查 image_processor 的 merge 口径与 render 的 image_counts 是否一致）"
            )
        embeds = embeds.reshape(-1, embeds.shape[-1]).clone()
        embeds[idx] = flat.to(embeds.dtype)
        return embeds.reshape(*input_ids.shape, -1)

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                image_positions=None, labels=None):
        embeds = self._embed(input_ids, pixel_values, image_positions)
        out = self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        logits = out.logits
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
                shift_labels.reshape(-1).to(shift_logits.device),
                ignore_index=-100,
            )
        return loss, logits

    @torch.no_grad()
    def teacher_logits(self, input_ids, attention_mask, pixel_values, image_positions):
        """OPD 教师前向（eval 模式；无梯度）。"""
        was_training = self.training
        self.eval()
        embeds = self._embed(input_ids, pixel_values, image_positions)
        logits = self.llm(inputs_embeds=embeds, attention_mask=attention_mask).logits
        if was_training:
            self.train()
        return logits

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, image_positions,
                 max_new_tokens: int = 1024, temperature: float = 1.0, do_sample: bool = True):
        """Rollout 生成（RL / 评测）：prompt 的 embedding 里已拼好视觉特征。"""
        was_training = self.training
        self.eval()
        embeds = self._embed(input_ids, pixel_values, image_positions)
        out = self.llm.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=0.95 if do_sample else None,
            pad_token_id=self.llm.config.eos_token_id,
        )
        if was_training:
            self.train()
        # generate(inputs_embeds=...) 返回的是纯新生成序列（无 prompt 前缀），token id 直接可解码
        return out
