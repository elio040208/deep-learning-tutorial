from __future__ import annotations

from torchvision import transforms as T

from dl_tutorial.registry import TRANSFORMS


@TRANSFORMS.register("ToTensor")
def to_tensor():
    return T.ToTensor()


@TRANSFORMS.register("Normalize")
def normalize(mean: list[float], std: list[float]):
    return T.Normalize(mean=mean, std=std)


@TRANSFORMS.register("Resize")
def resize(size: int | tuple[int, int]):
    return T.Resize(size)


@TRANSFORMS.register("RandomHorizontalFlip")
def random_horizontal_flip(p: float = 0.5):
    return T.RandomHorizontalFlip(p=p)


@TRANSFORMS.register("RandomCrop")
def random_crop(size: int, padding: int | None = None):
    return T.RandomCrop(size, padding=padding)
