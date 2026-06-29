# ml_learning

Hands-on ML concept lab. Each subfolder is one topic with runnable code, validation, and a learning guide.

Goal: understand the **core idea** of each ML concept by building the smallest useful version, then comparing it with a real reference implementation.

For humans: start from the topic README and follow the guided testing/training path.

For AI agents: preserve the anchor structure described below when adding new topics.

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

## Learning Structure Skill

Use this structure whenever adding a new concept folder. The current reference implementation is [`CLIP/README.md`](CLIP/README.md).

### 1. Pick one highest-level anchor file

Each concept should guide the learner to one main file first.

That file should show the full user journey: setup, testing, training, metrics, and outputs.

For CLIP:

```text
CLIP/train_real.py
├── train()      ← training process
├── evaluate()   ← testing / validation process
└── plot_*()     ← visual proof
```

For future topics, use the same idea:

```text
<Topic>/train_<topic>.py
├── train()      ← learning/update loop
├── evaluate()   ← testing / validation path
└── plot_*()     ← visual proof, if useful
```

### 2. Separate testing from training

Every topic README should explain two flows:

| Flow | Purpose | Updates weights? |
|------|---------|------------------|
| **Testing / validation** | Measure what the current model does | No |
| **Training** | Improve the model from data | Yes |

For CLIP:

```text
Testing:  Steps 1-5  DATA → ENCODE → SCORE → LOSS/METRICS
Training: Steps 1-6  DATA → ENCODE → SCORE → LOSS → UPDATE
```

### 3. Define numbered learning steps

Each topic should name the stages of the algorithm and reuse those names in comments.

For CLIP:

```text
Step 1  DATA
Step 2  ENCODE IMAGE
Step 3  ENCODE TEXT
Step 4  SCORE
Step 5  LOSS
Step 6  UPDATE
```

For Transformer, this may become:

```text
Step 1  TOKENS
Step 2  EMBEDDING
Step 3  ATTENTION
Step 4  MLP
Step 5  LOGITS / LOSS
Step 6  UPDATE
```

### 4. Make every file a component anchor

Each concept should have a component map in its README:

| Component | Preferred file pattern | Purpose |
|-----------|------------------------|---------|
| Top-level guide | `train_<topic>.py` | full testing + training journey |
| Data | `dataset_<topic>.py` | how raw data becomes tensors |
| Model | `model_<topic>.py` | architecture and forward pass |
| Loss / core math | `<topic>_core.py` | objective or central algorithm |
| Sanity check | `sanity_check.py` | fast checks that code and references agree |
| Official reference | `external/<repo>/` | upstream repo as submodule |

### 5. Keep README human-readable and AI-readable

Human-readable:

- Start with “where do I begin?”
- Show commands before deep implementation details.
- Explain expected outputs and what “better” looks like.
- Use diagrams or text pipelines.

AI-readable:

- Use stable headings: `Start Here`, `Big Map`, `Component Anchors`, `Testing Process`, `Training Process`.
- Name exact files and functions.
- Keep numbered algorithm steps consistent between README and code comments.
- Make validation commands explicit (`pixi run <topic>-sanity`).

### 6. Required README template for new topics

```markdown
# <Topic> — learn by building

## Start Here
Highest-level anchor: `train_<topic>.py`

## Big Map
Testing flow vs training flow.

## Component Anchors
Table mapping data/model/loss/train/evaluate/sanity/reference.

## Pipeline
Mermaid or text diagram.

## Testing Process
Steps that measure behavior without updates.

## Training Process
Steps that include the update.

## Commands
pixi run <topic>-smoke
pixi run <topic>-sanity
pixi run <topic>-train

## What This Teaches vs Reference
Compare this implementation with the upstream repo or paper.
```

## Topic conventions

| Item | Convention |
|------|------------|
| One folder per concept | `CLIP/`, `Transformer/`, … |
| Highest-level anchor | `train_<topic>.py` |
| Model + loss | `model_*.py`, `*_core.py` |
| Data pipeline | `dataset_*.py` |
| Train + eval | `train_*.py` with `train()` and `evaluate()` |
| Sanity check | `sanity_check.py` with a `pixi run <topic>-sanity` task |
| Upstream reference | `external/<repo>/` as git submodule |
| Pixi tasks | `<topic>-smoke`, `<topic>-sanity`, `<topic>-train` |

## Topics

### CLIP

Current reference for the learning structure skill.

Highest-level anchor: [`CLIP/train_real.py`](CLIP/train_real.py)

Guide: [`CLIP/README.md`](CLIP/README.md)

```bash
pixi run clip-smoke    # model smoke test
pixi run clip-sanity   # automated sanity checks
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
