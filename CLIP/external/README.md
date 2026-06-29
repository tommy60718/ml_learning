# External references

Git submodules point at upstream repos. They are **not imported into our training code** — use them to read and compare the real implementation.

| Submodule | Upstream | Purpose |
|-----------|----------|---------|
| [`openai-clip/`](openai-clip/) | [github.com/openai/CLIP](https://github.com/openai/CLIP) | Official inference code + pretrained weights |

After cloning `ml_learning`:

```bash
git submodule update --init --recursive
```

Browse the real model in `openai-clip/clip/model.py` and compare with our `model_real.py`.

Verify everything matches:

```bash
cd ~/ml_learning && pixi run clip-sanity
```
