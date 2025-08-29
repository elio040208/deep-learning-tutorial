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
        in_dim = head.get("in_features", None)
        if in_dim is None:
            # resnet-like: sequential of conv blocks -> AdaptiveAvgPool2d(1) expected 512/2048
            if isinstance(self.backbone, nn.Sequential):
                # try a dummy forward to infer
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 224, 224)
                        feats = self.backbone(dummy)
                        feats = torch.flatten(feats, 1) if feats.ndim == 4 else feats
                        in_dim = feats.shape[1]
                except Exception:
                    in_dim = 512
            else:
                # try generic probing
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 224, 224)
                        feats = self.backbone(dummy)
                        feats = torch.flatten(feats, 1) if feats.ndim == 4 else feats
                        in_dim = feats.shape[1]
                except Exception:
                    in_dim = 512
        head_cfg = {**head}
        head_cfg.setdefault("in_features", in_dim)
        self.head = HEADS.get(head_cfg["name"])(**{k: v for k, v in head_cfg.items() if k != "name"})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        return self.head(feats)
