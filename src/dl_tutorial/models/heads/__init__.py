from __future__ import annotations

import torch.nn as nn

from dl_tutorial.registry import HEADS


@HEADS.register("linear_classifier")
class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
