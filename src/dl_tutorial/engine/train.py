from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dl_tutorial.utils.logging import create_logger
from dl_tutorial.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        work_dir: str = "runs",
        mixed_precision: bool = False,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger("trainer", str(self.work_dir))
        self.writer = SummaryWriter(log_dir=str(self.work_dir))
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision
        self.grad_clip_norm = grad_clip_norm

    def _run_epoch(self, loader, epoch: int, train: bool = True) -> tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for step, batch in enumerate(pbar):
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).coalesce().backward() if hasattr(self.scaler, "coalesce") else self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Accuracy for classification
            if outputs.ndim >= 2 and targets.ndim == 1:
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = total_correct / max(1, total_samples)
        self.logger.info(f"{'Train' if train else 'Val'} epoch {epoch} - loss: {avg_loss:.4f} acc: {avg_acc:.4f}")
        return avg_loss, avg_acc

    def fit(self, train_loader, val_loader, epochs: int, log_interval: int = 50, ckpt_interval: int = 1) -> None:
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, epoch, train=True)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(val_loader, epoch, train=False)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)

            if epoch % ckpt_interval == 0 or val_acc > best_acc:
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                        "best_acc": best_acc,
                    },
                    str(self.work_dir / (f"best.pt" if is_best else f"epoch_{epoch}.pt")),
                )

        self.writer.close()
