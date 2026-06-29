"""
RealCLIP — dual encoders on real images + templated text.

**Model anchor** — implements CLIP Steps 2–5 (Step 1 is dataset_cifar.py, Step 6 is train_real.py):

  Step 2  ENCODE IMAGE  — ImageEncoder  (ResNet-18 → projection → L2 norm)
  Step 3  ENCODE TEXT   — TextEncoder   (Embedding → mean pool → projection → L2 norm)
  Steps 4–5             — clip_core.py  (score matrix + symmetric loss)

Compare with: `external/openai-clip/clip/model.py`

Smoke test:  cd ~/ml_learning && pixi run clip-smoke
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from clip_core import clip_contrastive_loss


class TextEncoder(nn.Module):
  """
  CLIP Step 3 — ENCODE TEXT.

  token_ids + mask  →  unit-norm text embedding [batch, latent_dim]

  Real CLIP (external/openai-clip): BPE → Transformer → EOT token → project.
  Our shortcut: mean-pool word embeddings (works with fixed caption template).
  """

  def __init__(self, vocab_size: int, embed_dim: int = 64, latent_dim: int = 128):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.projection = nn.Linear(embed_dim, latent_dim)

  def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # token_ids: [batch, seq_len]   mask: [batch, seq_len]  (1=real, 0=pad)

    # Step 3a — lookup: each token ID → word vector
    word_vectors = self.embedding(token_ids)       # [batch, seq_len, embed_dim]
    mask = mask.unsqueeze(-1)                      # [batch, seq_len, 1]

    # Step 3b — pool: collapse sequence into one vector (ignore padding)
    summed = (word_vectors * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / lengths

    # Step 3c — project into shared space + L2 normalize (cosine-sim ready)
    return F.normalize(self.projection(pooled), p=2, dim=-1)


class ImageEncoder(nn.Module):
  """
  CLIP Step 2 — ENCODE IMAGE.

  images [batch, 3, 224, 224]  →  unit-norm image embedding [batch, latent_dim]

  Real CLIP (external/openai-clip): ViT or ModifiedResNet → project.
  Ours: torchvision ResNet-18 (pretrained) → linear projection.
  """

  def __init__(self, latent_dim: int = 128):
    super().__init__()
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Drop final fc layer — keep 512-d global average pool features only.
    self.backbone = nn.Sequential(*list(backbone.children())[:-1])
    self.projection = nn.Linear(512, latent_dim)

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    # Step 2a — backbone: extract visual features from upscaled CIFAR images
    features = self.backbone(images).flatten(1)  # [batch, 512]

    # Step 2b — project into shared space + L2 normalize (cosine-sim ready)
    return F.normalize(self.projection(features), p=2, dim=-1)


class RealCLIP(nn.Module):
  """
  Wraps both encoders and runs CLIP Steps 2–5 in one call.

  Step 1 (DATA) and Step 6 (UPDATE) live outside this file — see train_real.py.
  """

  def __init__(self, vocab_size: int, latent_dim: int = 128):
    super().__init__()
    self.image_encoder = ImageEncoder(latent_dim)   # Step 2
    self.text_encoder = TextEncoder(vocab_size, embed_dim=64, latent_dim=latent_dim)  # Step 3
    # Used in Step 4 (score) — learnable temperature, same init as OpenAI CLIP
    self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

  def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    """CLIP Step 2 — image batch → unit-norm embeddings."""
    return self.image_encoder(images)

  def encode_text(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CLIP Step 3 — token batch → unit-norm embeddings."""
    return self.text_encoder(token_ids, mask)

  def forward(
    self,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CLIP Steps 2–5 in one forward pass.

    Returns (loss, logits) for training; Step 6 (backward/step) is in train_real.py.
    """
    # Step 2 — ENCODE IMAGE
    image_emb = self.encode_image(images)

    # Step 3 — ENCODE TEXT
    text_emb = self.encode_text(token_ids, mask)

    # Steps 4–5 — SCORE + LOSS (clip_core.py)
    #   Step 4: logits = exp(logit_scale) × image_emb @ text_emb.T
    #   Step 5: symmetric cross-entropy (diagonal = matched pairs)
    return clip_contrastive_loss(image_emb, text_emb, self.logit_scale)


if __name__ == "__main__":
  # Smoke test: Steps 2–5 on random tensors (Step 1 skipped — no real data needed)
  model = RealCLIP(vocab_size=50)
  images = torch.randn(4, 3, 224, 224)       # fake Step-1 images
  tokens = torch.randint(1, 50, (4, 12))     # fake Step-1 captions (tokenized)
  mask = torch.ones(4, 12)
  loss, logits = model(images, tokens, mask)
  print(f"RealCLIP logits shape: {list(logits.shape)}")
  print(f"loss (random init): {loss.item():.4f}")
