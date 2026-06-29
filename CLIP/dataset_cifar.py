"""
CIFAR-100 → image–text pairs for CLIP training.

**Data anchor** — how raw photos become (image, caption) batches:
  - Images:  torchvision CIFAR-100 (auto-downloads ~170 MB to CLIP/data/)
  - Text:    templated captions "a photo of a {class}" (same idea as OpenAI zero-shot eval)
  - Splits:  official train set for training, test set for validation

Real CLIP uses 400M web (image, caption) pairs. CIFAR is a stand-in: real pixels,
synthetic but valid text — enough to see the model learn alignment.

Used by: train_real.py via build_cifar_loaders()
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

# ResNet was pretrained on ImageNet with these stats — we must match at inference.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Template from OpenAI CLIP zero-shot examples (see external/openai-clip README).
CAPTION_TEMPLATE = "a photo of a {}"


@dataclass(frozen=True)
class TokenBatch:
  """Helper type if you batch tokenize outside the Dataset."""

  token_ids: torch.Tensor  # [batch, max_len]
  mask: torch.Tensor       # [batch, max_len] — 1 = real token, 0 = padding


class WordTokenizer:
  """
  Minimal word-level tokenizer for templated captions.

  Real CLIP uses BPE (49k vocab, 77 tokens) — see external/openai-clip/clip/simple_tokenizer.py.
  We only need ~100 words: template words + CIFAR class names.
  """

  PAD_ID = 0  # index 0 reserved; nn.Embedding(padding_idx=0) ignores it in loss

  def __init__(self, class_names: list[str]):
    words: set[str] = set()
    # Collect vocabulary from template + all class names.
    for word in CAPTION_TEMPLATE.replace("{", "").replace("}", "").split():
      words.add(word.lower())
    for name in class_names:
      for word in name.replace("_", " ").lower().split():
        words.add(word)

    # Stable ordering → reproducible token IDs across runs.
    self.word_to_id = {word: idx + 1 for idx, word in enumerate(sorted(words))}
    self.max_len = 12  # longest CIFAR caption fits easily

  def caption_for_label(self, label: int, class_names: list[str]) -> str:
    """Class index 42 → caption string 'a photo of a mountain'."""
    name = class_names[label].replace("_", " ")
    return CAPTION_TEMPLATE.format(name)

  def encode(self, caption: str) -> tuple[list[int], list[int]]:
    """Caption string → (token_ids, mask) with right-padding."""
    tokens = [self.word_to_id.get(word, self.PAD_ID) for word in caption.lower().split()]
    tokens = tokens[: self.max_len]
    mask = [1] * len(tokens)
    # Pad to fixed length so batches can be stacked into a tensor.
    pad_count = self.max_len - len(tokens)
    tokens.extend([self.PAD_ID] * pad_count)
    mask.extend([0] * pad_count)
    return tokens, mask

  @property
  def vocab_size(self) -> int:
    return len(self.word_to_id) + 1  # +1 for PAD_ID


class CIFAR100CLIPDataset(Dataset):
  """
  PyTorch Dataset: one item = (image, token_ids, mask, label).

  label is kept for debugging ("which class is this image?").
  Training ignores label — CLIP learns from batch alignment, not class IDs.
  """

  def __init__(
    self,
    root: str,
    train: bool,
    tokenizer: WordTokenizer,
    class_names: list[str],
    max_samples: int | None = None,
  ):
    self.tokenizer = tokenizer
    self.class_names = class_names

    # CIFAR is 32×32; ResNet expects 224×224 — upscale + ImageNet normalize.
    self.transform = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    self.cifar = CIFAR100(root=root, train=train, download=True, transform=self.transform)

    # Subsample for faster demos (full train = 50k, test = 10k).
    self.indices = list(range(len(self.cifar)))
    if max_samples is not None:
      self.indices = self.indices[:max_samples]

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, idx: int):
    real_idx = self.indices[idx]
    image, label = self.cifar[real_idx]

    # Build the matching caption for this image's class.
    caption = self.tokenizer.caption_for_label(label, self.class_names)
    token_ids, mask = self.tokenizer.encode(caption)

    return (
      image,                                          # [3, 224, 224]
      torch.tensor(token_ids, dtype=torch.long),      # [max_len]
      torch.tensor(mask, dtype=torch.float32),        # [max_len]
      label,                                          # int, for debugging only
    )


def build_cifar_loaders(
  data_dir: str,
  batch_size: int,
  train_max: int | None = 5000,
  val_max: int | None = 1000,
  num_workers: int = 0,
):
  """
  Factory used by train_real.py — returns everything needed to train:

    train_loader, val_loader, tokenizer, class_names

  train=True  → CIFAR train split (50k images, we cap at train_max)
  train=False → CIFAR test split  (10k images, we cap at val_max) — our validation set
  """
  meta = CIFAR100(root=data_dir, train=True, download=True)
  class_names = meta.classes
  tokenizer = WordTokenizer(class_names)

  train_ds = CIFAR100CLIPDataset(
    data_dir, train=True, tokenizer=tokenizer, class_names=class_names, max_samples=train_max
  )
  val_ds = CIFAR100CLIPDataset(
    data_dir, train=False, tokenizer=tokenizer, class_names=class_names, max_samples=val_max
  )

  train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
    num_workers=num_workers, pin_memory=num_workers > 0,
  )
  val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
    num_workers=num_workers, pin_memory=num_workers > 0,
  )
  return train_loader, val_loader, tokenizer, class_names
