# ml_learning

Hands-on ML concept lab. Each subfolder is one topic: minimal code, runnable demos, and validation plots. Goal is to understand the **core idea** of each paper or architecture—not to reproduce production-scale systems.

## Setup

This repo uses [pixi](https://pixi.sh) for a single shared Python environment (PyTorch, matplotlib, …).

```bash
cd ~/ml_learning
pixi install                              # first time
git submodule update --init --recursive   # after clone (CLIP reference repo)
```

## Folder structure

```text
ml_learning/
├── README.md
├── pyproject.toml            # pixi workspace, deps, tasks
│
├── CLIP/                     # contrastive language–image pre-training
│   ├── README.md             # teaching guide + file anchors
│   ├── clip_core.py          # contrastive loss
│   ├── model_real.py         # ResNet-18 + text encoder
│   ├── dataset_cifar.py      # CIFAR-100 image–caption pairs
│   ├── train_real.py         # train + validate
│   ├── external/openai-clip/ # git submodule → github.com/openai/CLIP
│   ├── data/                 # auto-downloaded CIFAR (gitignored)
│   └── outputs/              # plots (gitignored)
│
├── Transformer/              # (planned)
└── _shared/                  # (optional) cross-topic helpers
```

### Conventions for new topics

| Item | Convention |
|------|------------|
| One folder per concept | `CLIP/`, `Transformer/`, … |
| Model + loss | `model_*.py`, `*_core.py` |
| Data pipeline | `dataset_*.py` |
| Train + eval | `train_*.py` with clear anchors (see CLIP/README) |
| Upstream reference | `external/<repo>/` as git submodule |
| Pixi tasks | `topic-train`, `topic-smoke` in `pyproject.toml` |

## Topics

### CLIP

See [`CLIP/README.md`](CLIP/README.md) for the full learning path.

```bash
pixi run clip-smoke    # model smoke test
pixi run clip-train    # CIFAR-100 training + validation
```

Official reference (submodule): [openai/CLIP](https://github.com/openai/CLIP)

### Transformer — (planned)

Self-attention, positional encoding, mini GPT-style LM.

## To Learn List

Planned topics and upgrades — in rough learning order:

| # | Topic | Goal | Builds on |
|---|-------|------|-----------|
| 1 | **Transformer** | `Transformer/` — self-attention, positional encoding, mini language model | Standalone foundation for modern NLP/VLM |
| 2 | **Text & Image Encoder** | Upgrade CLIP encoders: Transformer text tower + ViT/stronger image backbone (closer to `external/openai-clip`) | Current `CLIP/model_real.py` (ResNet-18 + word mean-pool) |

**Suggested path:** Transformer first (understand attention in isolation), then swap CLIP's simplified encoders for the real architecture pieces.

## Design principles

1. **Minimal but correct** — core math matches the real method; data/architecture scaled down.
2. **See the learning** — save plots under `<topic>/outputs/`.
3. **One env, many topics** — shared `pyproject.toml`.
4. **Submodules for upstream** — link to real repos, don't vendor them into our tree.

## Quick reference

```bash
pixi run clip-smoke
pixi run clip-sanity
pixi run clip-train
pixi shell
```

GPU check: see `~/AGENTS.md` § GPU.
