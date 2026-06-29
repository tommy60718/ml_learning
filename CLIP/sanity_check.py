"""
Automated sanity checks for CLIP — learn by building.

Verifies our implementation against patterns in external/openai-clip/clip/model.py.

Run:  cd ~/ml_learning && pixi run clip-sanity
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CLIP_DIR = Path(__file__).resolve().parent
OPENAI_CLIP = CLIP_DIR / "external" / "openai-clip"
OPENAI_MODEL = OPENAI_CLIP / "clip" / "model.py"


def check_submodule() -> None:
  """Submodule must be initialized so learners can diff against upstream."""
  assert OPENAI_MODEL.is_file(), (
    f"Missing {OPENAI_MODEL}\n"
    "Run: git submodule update --init --recursive"
  )
  print("✓ submodule  external/openai-clip/clip/model.py")


def check_loss_matches_openai_pattern() -> None:
  """
  OpenAI CLIP forward (model.py ~358-369):
    logits = exp(logit_scale) * (I_norm @ T_norm.T)
  Our clip_core.py adds symmetric cross-entropy — same logits, training extension.
  """
  from clip_core import clip_contrastive_loss

  torch.manual_seed(0)
  batch, dim = 8, 64
  image = F.normalize(torch.randn(batch, dim), dim=-1)
  text = F.normalize(torch.randn(batch, dim), dim=-1)
  logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))

  loss, logits = clip_contrastive_loss(image, text, logit_scale)

  # Replicate OpenAI logits formula exactly.
  expected_logits = logit_scale.exp() * image @ text.t()
  assert torch.allclose(logits, expected_logits, atol=1e-5), "logits differ from OpenAI formula"

  # logit_scale init matches OpenAI: np.log(1 / 0.07)
  assert abs(logit_scale.exp().item() - (1 / 0.07)) < 1e-4

  assert loss.shape == ()
  assert logits.shape == (batch, batch)
  print("✓ loss       clip_core logits match openai-clip forward pattern")


def check_model_smoke() -> None:
  from model_real import RealCLIP

  model = RealCLIP(vocab_size=50)
  images = torch.randn(4, 3, 224, 224)
  tokens = torch.randint(1, 50, (4, 12))
  mask = torch.ones(4, 12)
  loss, logits = model(images, tokens, mask)
  assert logits.shape == (4, 4)
  assert loss.ndim == 0
  print("✓ model      RealCLIP forward (random tensors)")


def check_data_pipeline() -> None:
  from dataset_cifar import CIFAR100CLIPDataset, WordTokenizer, CAPTION_TEMPLATE
  from torchvision.datasets import CIFAR100

  meta = CIFAR100(root=str(CLIP_DIR / "data"), train=True, download=False)
  tokenizer = WordTokenizer(meta.classes)
  ds = CIFAR100CLIPDataset(
    str(CLIP_DIR / "data"), train=True, tokenizer=tokenizer,
    class_names=meta.classes, max_samples=4,
  )

  image, token_ids, mask, label = ds[0]
  caption = tokenizer.caption_for_label(label, meta.classes)

  assert image.shape == (3, 224, 224)
  assert token_ids.shape[0] == tokenizer.max_len
  assert caption.startswith("a photo of a")
  assert CAPTION_TEMPLATE.format("dog") == "a photo of a dog"
  print(f"✓ data       CIFAR sample caption: {caption!r}")


def check_train_step() -> None:
  """One optimizer step on a tiny real batch — end-to-end differentiable."""
  from dataset_cifar import build_cifar_loaders
  from model_real import RealCLIP

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  loader, _, tokenizer, _ = build_cifar_loaders(
    str(CLIP_DIR / "data"), batch_size=4, train_max=4, val_max=4, num_workers=0,
  )
  model = RealCLIP(vocab_size=tokenizer.vocab_size).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-4)

  images, token_ids, mask, _ = next(iter(loader))
  images, token_ids, mask = images.to(device), token_ids.to(device), mask.to(device)

  opt.zero_grad()
  loss_before, _ = model(images, token_ids, mask)
  loss_before.backward()
  opt.step()
  loss_after, _ = model(images, token_ids, mask)

  assert torch.isfinite(loss_before)
  assert torch.isfinite(loss_after)
  print(f"✓ train      one step on device={device}  loss {loss_before.item():.3f} → {loss_after.item():.3f}")


def main() -> int:
  print("=== CLIP sanity check ===\n")
  checks = [
    check_submodule,
    check_loss_matches_openai_pattern,
    check_model_smoke,
    check_data_pipeline,
    check_train_step,
  ]
  failed = 0
  for fn in checks:
    try:
      fn()
    except Exception as exc:
      print(f"✗ {fn.__name__}: {exc}", file=sys.stderr)
      failed += 1

  print()
  if failed:
    print(f"FAILED ({failed} check(s))")
    return 1
  print("All checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
