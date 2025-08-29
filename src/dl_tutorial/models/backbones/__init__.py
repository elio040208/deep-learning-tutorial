from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from dl_tutorial.registry import BACKBONES


@BACKBONES.register("resnet18")
def resnet18(pretrained: bool = False, num_freeze_stages: int = 0, **kwargs) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    # Optionally freeze
    if num_freeze_stages > 0:
        frozen = 0
        for name, p in m.named_parameters():
            p.requires_grad = False
            frozen += 1
            if frozen >= num_freeze_stages:
                break
    # Remove fc, return feature extractor with output dim 512
    modules = list(m.children())[:-1]
    return nn.Sequential(*modules)


@BACKBONES.register("small_cnn")
class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Conv2d(128, feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.proj(x)
        x = torch.flatten(x, 1)
        return x
