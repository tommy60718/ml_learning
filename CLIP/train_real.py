"""
Train and validate RealCLIP on CIFAR-100.

**Training anchor** — this file owns the full train loop:
  train()           → download data, build model, run epochs, save plots
  RealTrainConfig   → hyperparameters (edit here to experiment)

**Testing / validation anchor** — same file:
  evaluate()        → run val set, compute loss + retrieval accuracy
  plot_*()          → save heatmaps and curves under outputs/

Run:  cd ~/ml_learning && pixi run clip-train
Guide: CLIP/README.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset_cifar import build_cifar_loaders
from model_real import RealCLIP

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class RealTrainConfig:
  """
  Hyperparameters. Defaults work on CPU; GPU gets larger batch via resolve_config().

  Try: train_max=20000, epochs=15 for better accuracy (slower).
  """

  batch_size: int = 16       # → 64 automatically on GPU
  epochs: int = 8
  lr: float = 1e-4           # small LR because ResNet backbone is pretrained
  train_max: int = 5000      # cap train images (full set = 50000)
  val_max: int = 1000        # cap val images   (full set = 10000)
  latent_dim: int = 128      # shared embedding size for image + text towers
  seed: int = 42
  num_workers: int = 0       # → 4 automatically on GPU


def resolve_config(cfg: RealTrainConfig) -> RealTrainConfig:
  """GPU: bigger batches + parallel data loading. CPU: keep defaults."""
  if not torch.cuda.is_available():
    return cfg
  return RealTrainConfig(
    batch_size=64,
    epochs=cfg.epochs,
    lr=cfg.lr,
    train_max=cfg.train_max,
    val_max=cfg.val_max,
    latent_dim=cfg.latent_dim,
    seed=cfg.seed,
    num_workers=4,
  )


def pick_device() -> torch.device:
  """Use CUDA when pixi installed the GPU PyTorch build (see pyproject.toml)."""
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")


@torch.no_grad()
def evaluate(model: RealCLIP, loader: DataLoader, device: torch.device) -> dict:
  """
  **Validation / testing anchor** — runs CLIP Steps 1–5 on the val set (no Step 6).

  Per batch the CLIP pipeline is:
    Step 1  DATA          — load (image, tokens, mask) from DataLoader
    Steps 2–5             — model.forward() → encode → score → loss (clip_core.py)
    (metrics)             — retrieval accuracy from logits (not part of training loss)

  Metrics averaged over all val batches:
    loss, acc_i2t, acc_t2i, acc, logits (last batch for heatmap)
  """
  model.eval()  # inference mode — no dropout / frozen batchnorm stats
  total_loss = 0.0
  total_batches = 0
  correct_i2t = 0
  correct_t2i = 0
  total_samples = 0
  last_logits = None

  for images, token_ids, mask, _labels in loader:
    # ── CLIP Step 1 — DATA ──────────────────────────────────────────────
    # dataset_cifar.py already built matched (image, caption) pairs.
    # Move tensors to GPU/CPU for this forward pass.
    images = images.to(device)
    token_ids = token_ids.to(device)
    mask = mask.to(device)

    # ── CLIP Steps 2–5 — ENCODE → SCORE → LOSS ───────────────────────
    # model_real.RealCLIP.forward():
    #   Step 2  encode_image  — ResNet-18 → projection → L2 norm
    #   Step 3  encode_text   — Embedding → mean pool → projection → L2 norm
    #   Step 4  score         — logits = exp(logit_scale) × I @ Tᵀ
    #   Step 5  loss          — symmetric cross-entropy (diagonal = positives)
    # No Step 6 here — evaluate does not call backward() or optimizer.step().
    loss, logits = model(images, token_ids, mask)
    total_loss += loss.item()
    total_batches += 1
    last_logits = logits.cpu()  # keep one batch for similarity heatmap

    # ── Retrieval metrics (testing only, not CLIP training loss) ────────
    # Given logits[i, j] = similarity(image_i, text_j), check if diagonal wins.
    n = logits.shape[0]
    targets = torch.arange(n, device=device)  # correct index for pair i is i
    correct_i2t += (logits.argmax(dim=1) == targets).sum().item()  # each row → best col
    correct_t2i += (logits.argmax(dim=0) == targets).sum().item()  # each col → best row
    total_samples += n

  return {
    "loss": total_loss / max(total_batches, 1),
    "acc_i2t": correct_i2t / max(total_samples, 1),
    "acc_t2i": correct_t2i / max(total_samples, 1),
    "acc": (correct_i2t + correct_t2i) / (2 * max(total_samples, 1)),
    "logits": last_logits,
  }


def plot_similarity_heatmap(logits: torch.Tensor, path: Path, title: str):
  """
  Visualize the N×N similarity matrix as a heatmap.
  Black boxes on the diagonal = correct pairs we want to brighten during training.
  """
  if logits is None:
    return
  n = min(logits.shape[0], 16)
  logits = logits[:n, :n]
  fig, ax = plt.subplots(figsize=(5, 4))
  im = ax.imshow(logits.numpy(), cmap="RdYlGn", vmin=-2, vmax=20)
  ax.set_title(title, fontsize=10)
  ax.set_xlabel("text index in batch")
  ax.set_ylabel("image index in batch")
  for i in range(n):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.5))
  fig.colorbar(im, ax=ax, fraction=0.046)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_training_curves(history: dict, path: Path):
  """Save train/val loss and val accuracy over epochs."""
  epochs = range(1, len(history["train_loss"]) + 1)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  axes[0].plot(epochs, history["train_loss"], label="train", color="#2563eb", marker="o", ms=4)
  axes[0].plot(epochs, history["val_loss"], label="val", color="#dc2626", marker="o", ms=4)
  axes[0].set_title("Contrastive loss")
  axes[0].set_xlabel("epoch")
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(epochs, history["val_acc"], color="#16a34a", marker="o", ms=4)
  axes[1].set_ylim(0, 1.05)
  axes[1].set_title("Validation retrieval accuracy")
  axes[1].set_xlabel("epoch")
  axes[1].grid(True, alpha=0.3)

  fig.suptitle("RealCLIP on CIFAR-100", fontsize=12)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def train(cfg: RealTrainConfig = RealTrainConfig()):
  """
  **Main training entry point.**

  CLIP pipeline across this function:
    Step 1  DATA     — build CIFAR train/val loaders (dataset_cifar.py)
    (setup)          — build RealCLIP + Adam
    Steps 2–5        — evaluate() before training (baseline, no Step 6)
    per epoch:
      Steps 1–5      — forward on train batches (encode → score → loss)
      Step 6         — backward + optimizer.step (weight update)
      Steps 1–5      — evaluate() on val (measure, no Step 6)
    Steps 1–5        — evaluate() after training + save plots
  """
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  DATA_DIR.mkdir(parents=True, exist_ok=True)

  cfg = resolve_config(cfg)
  torch.manual_seed(cfg.seed)  # reproducible shuffling / init

  device = pick_device()
  print("=== RealCLIP training on CIFAR-100 ===", flush=True)
  if device.type == "cuda":
    print(f"device={device} ({torch.cuda.get_device_name(0)})  batch={cfg.batch_size}", flush=True)
  else:
    print(f"device={device}  batch={cfg.batch_size}", flush=True)
    print("  hint: see AGENTS.md § GPU and CLIP/README.md", flush=True)
  print(f"epochs={cfg.epochs}  num_workers={cfg.num_workers}", flush=True)
  print(f"train images (max): {cfg.train_max}  |  val images (max): {cfg.val_max}", flush=True)
  print("First run downloads CIFAR-100 (~170 MB)…\n", flush=True)

  # ── CLIP Step 1 — DATA ────────────────────────────────────────────────
  # dataset_cifar.py: CIFAR image + templated caption → (image, tokens, mask)
  train_loader, val_loader, tokenizer, class_names = build_cifar_loaders(
    str(DATA_DIR),
    batch_size=cfg.batch_size,
    train_max=cfg.train_max,
    val_max=cfg.val_max,
    num_workers=cfg.num_workers,
  )
  print(f"Classes: {len(class_names)}  |  Vocab size: {tokenizer.vocab_size}")
  print(f"Example caption: {tokenizer.caption_for_label(0, class_names)!r}\n")

  # ── Setup — model + optimizer (Steps 2–5 need this; Step 6 needs optimizer) ─
  model = RealCLIP(vocab_size=tokenizer.vocab_size, latent_dim=cfg.latent_dim).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

  # ── Baseline: CLIP Steps 1–5 on val (no Step 6 yet) ─────────────────────
  before = evaluate(model, val_loader, device)
  plot_similarity_heatmap(
    before["logits"],
    OUTPUT_DIR / "similarity_before.png",
    "Before training (val batch)",
  )
  print(
    f"BEFORE  val_loss={before['loss']:.3f}  "
    f"val_acc={before['acc']:.1%} (i2t={before['acc_i2t']:.1%}, t2i={before['acc_t2i']:.1%})",
    flush=True,
  )

  history = {"train_loss": [], "val_loss": [], "val_acc": []}

  # ── Epoch loop: repeat Steps 1–6 on train, then Steps 1–5 on val ───────
  for epoch in range(1, cfg.epochs + 1):
    model.train()  # enable training mode (dropout / batchnorm update)
    epoch_loss = 0.0
    n_batches = 0

    for images, token_ids, mask, _labels in train_loader:
      # CLIP Step 1 — DATA: one batch of matched (image, caption) pairs
      images = images.to(device)
      token_ids = token_ids.to(device)
      mask = mask.to(device)

      # CLIP Steps 2–5 — ENCODE → SCORE → LOSS (inside model.forward)
      optimizer.zero_grad()
      loss, _logits = model(images, token_ids, mask)

      # CLIP Step 6 — UPDATE: backprop loss into encoders + logit_scale, then step
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      n_batches += 1

    # CLIP Steps 1–5 on val — measure alignment without updating weights
    val_metrics = evaluate(model, val_loader, device)
    history["train_loss"].append(epoch_loss / n_batches)
    history["val_loss"].append(val_metrics["loss"])
    history["val_acc"].append(val_metrics["acc"])

    print(
      f"epoch {epoch}/{cfg.epochs}  "
      f"train_loss={history['train_loss'][-1]:.3f}  "
      f"val_loss={val_metrics['loss']:.3f}  "
      f"val_acc={val_metrics['acc']:.1%}",
      flush=True,
    )

  # ── Final: CLIP Steps 1–5 on val after all Step-6 updates ───────────────
  after = evaluate(model, val_loader, device)
  plot_similarity_heatmap(
    after["logits"],
    OUTPUT_DIR / "similarity_after.png",
    "After training (diagonal should brighten)",
  )
  plot_training_curves(history, OUTPUT_DIR / "training_curves.png")

  print(
    f"\nAFTER   val_loss={after['loss']:.3f}  "
    f"val_acc={after['acc']:.1%} (i2t={after['acc_i2t']:.1%}, t2i={after['acc_t2i']:.1%})",
    flush=True,
  )
  print(f"\nOutputs in {OUTPUT_DIR}/")
  print("  similarity_before.png / similarity_after.png")
  print("  training_curves.png")


if __name__ == "__main__":
  train()
