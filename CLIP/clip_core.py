"""
Shared CLIP contrastive loss — the heart of training.

This file is the **mathematical core** shared by our educational implementation and
OpenAI CLIP. Compare with `external/openai-clip/clip/model.py` (production code).

Training objective (InfoNCE with in-batch negatives):

  1. L2-normalize image and text embeddings  →  unit vectors on a hypersphere
  2. Compute pairwise cosine similarity      →  dot product of normalized vectors
  3. Scale by exp(logit_scale)               →  learnable temperature τ
  4. Treat in-batch pairs as classification:
       - diagonal (i, i) = positive (matched pair)
       - off-diagonal     = negatives (other images/texts in the batch)
  5. Symmetric cross-entropy: image→text AND text→image
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(
  image_embeddings: torch.Tensor,
  text_embeddings: torch.Tensor,
  logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Compute symmetric InfoNCE loss (CLIP loss).

  Args:
    image_embeddings: [batch, dim] — must already be L2-normalized
    text_embeddings:  [batch, dim] — must already be L2-normalized
    logit_scale:      scalar nn.Parameter storing log(1/τ)

  Returns:
    loss:   scalar tensor with gradients (pass to loss.backward())
    logits: [batch, batch] similarity matrix (for metrics / heatmaps)

  OpenAI CLIP `forward()` returns logits only (inference-style).
  Training adds symmetric cross-entropy on those logits.
  Logits formula matches external/openai-clip/clip/model.py lines 358–369.

  Intuition for a batch of 3:
    logits[0] = [sim(I0,T0), sim(I0,T1), sim(I0,T2)]  ← image 0 vs all texts
    labels = [0, 1, 2]  →  we want the diagonal to be largest in each row/column
  """
  # Step 1: scaled cosine similarity matrix.
  # Because embeddings are unit vectors, (I @ Tᵀ) IS cosine similarity.
  # Multiplying by exp(logit_scale) is equivalent to dividing by a learnable τ.
  # OpenAI CLIP initializes logit_scale = log(1/0.07) so exp ≈ 14.3.
  logits = logit_scale.exp() * image_embeddings @ text_embeddings.t()

  batch_size = image_embeddings.shape[0]

  # Step 2: ground truth — sample i in the batch is paired with sample i.
  # No external label file needed: other items in the same batch are negatives.
  labels = torch.arange(batch_size, device=image_embeddings.device)

  # Step 3: image→text — each ROW is a multi-class problem over captions.
  loss_image_to_text = F.cross_entropy(logits, labels)

  # Step 4: text→image — each COLUMN is a multi-class problem over images.
  # Transpose swaps the role of rows and columns.
  loss_text_to_image = F.cross_entropy(logits.t(), labels)

  # Step 5: average both directions (symmetric CLIP loss).
  loss = (loss_image_to_text + loss_text_to_image) / 2.0
  return loss, logits
