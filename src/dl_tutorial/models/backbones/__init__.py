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


@BACKBONES.register("resnet34")
def resnet34(pretrained: bool = False, num_freeze_stages: int = 0, **kwargs) -> nn.Module:
    m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    if num_freeze_stages > 0:
        frozen = 0
        for name, p in m.named_parameters():
            p.requires_grad = False
            frozen += 1
            if frozen >= num_freeze_stages:
                break
    modules = list(m.children())[:-1]
    return nn.Sequential(*modules)


@BACKBONES.register("resnet50")
def resnet50(pretrained: bool = False, num_freeze_stages: int = 0, **kwargs) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    if num_freeze_stages > 0:
        frozen = 0
        for name, p in m.named_parameters():
            p.requires_grad = False
            frozen += 1
            if frozen >= num_freeze_stages:
                break
    modules = list(m.children())[:-1]
    return nn.Sequential(*modules)


@BACKBONES.register("mobilenet_v2")
def mobilenet_v2(pretrained: bool = False, **kwargs) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    # feature extractor returns (B, 1280, 1, 1)
    return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))


@BACKBONES.register("efficientnet_b0")
def efficientnet_b0(pretrained: bool = False, **kwargs) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))


@BACKBONES.register("vit_b_16")
def vit_b_16(pretrained: bool = False, image_size: int = 224, **kwargs) -> nn.Module:
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
    # vit forward returns (B, D) after encoder+head; we strip head by exposing encoder+cls token output
    # Use torchvision implementation: take encoder and return token embedding
    class ViTFeature(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # emulate forward without classification head
            n = x.shape[0]
            x = self.vit._process_input(x)
            x = self.vit._forward_impl(x)
            # vit returns (B, D) already; ensure 2D
            return x
    return ViTFeature(m)
    
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


@BACKBONES.register("text_bilstm")
class TextBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_idx: int = 0,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, feature_dim or out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # Pooling: take last timestep
        feats = out[:, -1, :]
        return self.proj(feats)