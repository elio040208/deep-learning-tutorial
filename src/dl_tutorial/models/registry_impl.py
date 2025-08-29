from __future__ import annotations

import torch
import torch.nn as nn

from dl_tutorial.registry import MODELS, BACKBONES, HEADS


@MODELS.register("classification_model")
class ClassificationModel(nn.Module):
    def __init__(self, backbone: dict, head: dict):
        super().__init__()
        self.backbone = BACKBONES.get(backbone["name"])(**{k: v for k, v in backbone.items() if k != "name"})
        # infer feature dim for common backbones
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            in_dim = self.backbone.fc.in_features  # type: ignore[attr-defined]
        elif isinstance(self.backbone, nn.Sequential):
            in_dim = 512  # resnet18 last layer features
        else:
            # fallback, allow explicit in_features in head config
            in_dim = head.get("in_features", 512)
        head_cfg = {**head}
        head_cfg.setdefault("in_features", in_dim)
        self.head = HEADS.get(head_cfg["name"])(**{k: v for k, v in head_cfg.items() if k != "name"})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        return self.head(feats)
