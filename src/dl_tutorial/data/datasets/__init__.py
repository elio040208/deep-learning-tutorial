from __future__ import annotations

from typing import Any

from torchvision import datasets

from dl_tutorial.registry import DATASETS


@DATASETS.register("CIFAR10")
def cifar10(root: str, train: bool, download: bool = True, transforms=None, **kwargs: Any):
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transforms)


@DATASETS.register("MNIST")
def mnist(root: str, train: bool, download: bool = True, transforms=None, **kwargs: Any):
    return datasets.MNIST(root=root, train=train, download=download, transform=transforms)
