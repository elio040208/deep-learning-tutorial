from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from . import registry


def build_from_registry(cfg: Dict[str, Any], reg: registry.Registry, extra_kwargs: Dict[str, Any] | None = None):
    if cfg is None:
        return None
    if "name" not in cfg:
        raise KeyError("Config must contain 'name'")
    name = cfg["name"]
    params = {k: v for k, v in cfg.items() if k != "name"}
    if extra_kwargs:
        params.update(extra_kwargs)
    cls_or_fn = reg.get(name)
    return cls_or_fn(**params)


def build_backbone(cfg: Dict[str, Any]):
    return build_from_registry(cfg, registry.BACKBONES)


def build_head(cfg: Dict[str, Any]):
    return build_from_registry(cfg, registry.HEADS)


def build_model(cfg: Dict[str, Any]):
    return build_from_registry(cfg, registry.MODELS)


def build_loss(cfg: Dict[str, Any]):
    return build_from_registry(cfg, registry.LOSSES)


def build_optimizer(cfg: Dict[str, Any], params: Iterable[torch.nn.Parameter]) -> Optimizer:
    return build_from_registry({**cfg, "params": params}, registry.OPTIMIZERS)


def build_scheduler(cfg: Dict[str, Any], optimizer: Optimizer) -> _LRScheduler | None:
    if cfg is None:
        return None
    return build_from_registry({**cfg, "optimizer": optimizer}, registry.SCHEDULERS)


def build_transforms(cfg_list: List[Dict[str, Any]]):
    transforms = []
    for t_cfg in cfg_list or []:
        transforms.append(build_from_registry(t_cfg, registry.TRANSFORMS))
    if not transforms:
        return None
    # Lazy import to avoid hard dependency in registries
    from torchvision import transforms as T
    return T.Compose(transforms)


def build_dataset(cfg: Dict[str, Any]):
    transforms = None
    if cfg.get("transforms"):
        transforms = build_transforms(cfg["transforms"])  # type: ignore[assignment]
    return build_from_registry({**cfg, "transforms": transforms}, registry.DATASETS)


def build_dataloader(cfg: Dict[str, Any], dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=cfg.get("shuffle", True),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        drop_last=cfg.get("drop_last", False),
    )
