from __future__ import annotations

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from dl_tutorial.registry import OPTIMIZERS, SCHEDULERS


@OPTIMIZERS.register("SGD")
def build_sgd(params, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, nesterov: bool = False):
    return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


@OPTIMIZERS.register("Adam")
def build_adam(params, lr: float = 1e-3, weight_decay: float = 0.0, betas: tuple[float, float] = (0.9, 0.999)):
    return Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)


@SCHEDULERS.register("StepLR")
def build_step_lr(optimizer, step_size: int = 30, gamma: float = 0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@SCHEDULERS.register("CosineAnnealingLR")
def build_cosine_annealing_lr(optimizer, T_max: int, eta_min: float = 0.0):
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
