from __future__ import annotations

import os
from pathlib import Path

import torch

from dl_tutorial.config import build_argparser, load_config, parse_cli_overrides
from dl_tutorial.builders import (
    build_dataset,
    build_dataloader,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)
from dl_tutorial.utils.seed import set_seed
from dl_tutorial.engine.train import Trainer

# Ensure registries are populated by importing side-effect modules
from dl_tutorial import models as _models  # noqa: F401
from dl_tutorial.data import datasets as _datasets  # noqa: F401
from dl_tutorial.data import transforms as _transforms  # noqa: F401
from dl_tutorial import optim as _optim  # noqa: F401
from dl_tutorial import losses as _losses  # noqa: F401


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    overrides = parse_cli_overrides(args.override)
    cfg = load_config(args.config, overrides)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = build_model(cfg["model"]).to(device)

    # Loss
    criterion = build_loss(cfg["loss"]).to(device)  # type: ignore[attr-defined]

    # Data
    train_dataset = build_dataset(cfg["data"]["train"])  # type: ignore[index]
    val_dataset = build_dataset(cfg["data"]["val"])  # type: ignore[index]
    train_loader = build_dataloader(cfg["data"]["train_loader"], train_dataset)  # type: ignore[index]
    val_loader = build_dataloader(cfg["data"]["val_loader"], val_dataset)  # type: ignore[index]

    # Optimizer/Scheduler
    optimizer = build_optimizer(cfg["optimizer"], model.parameters())
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        work_dir=cfg.get("work_dir", "runs"),
        mixed_precision=cfg.get("mixed_precision", False),
        grad_clip_norm=cfg.get("grad_clip_norm"),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.get("epochs", 20),
        log_interval=cfg.get("log_interval", 50),
        ckpt_interval=cfg.get("ckpt_interval", 1),
    )


if __name__ == "__main__":
    main()
