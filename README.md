## Deep Learning Tutorial Framework

A modular deep learning tutorial framework driven by YAML configs. Create models, backbones, heads, losses, datasets, and training hyperparameters from configuration. Uses UV for environment management and TensorBoard for monitoring.

### Quickstart

1. Install UV: see `https://docs.astral.sh/uv/`.
2. Create and sync the environment:
```bash
uv venv
uv sync
```
3. Activate the environment (PowerShell):
```bash
. .\.venv\Scripts\Activate.ps1
```
4. Download a dataset (e.g., CIFAR-10):
```bash
uv run python scripts/download_datasets.py --dataset cifar10 --root data
```
5. Train using a YAML config:
```bash
uv run python train.py --config configs/classification/resnet18_cifar10.yaml
```
6. Launch TensorBoard:
```bash
uv run tensorboard --logdir runs
```

### Structure
```
configs/
src/dl_tutorial/
  config.py
  registry.py
  builders.py
  models/
    __init__.py
    backbones/
    heads/
  losses/
  data/
    datasets/
    transforms.py
  engine/
    train.py
  utils/
    logging.py
    checkpoint.py
    seed.py
scripts/
  download_datasets.py
train.py
```

- Extend by registering new modules.
- See `configs/` for examples.
