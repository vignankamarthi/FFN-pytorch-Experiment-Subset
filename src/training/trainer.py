"""Trainer class for vanilla TSM training.

Handles the complete training loop including:
- Forward/backward passes
- Optimization steps
- Learning rate scheduling (step decay matching original paper)
- Automatic mixed precision (AMP) training
- Gradient clipping
- Logging and checkpointing
"""

from typing import Optional, Dict, List
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler

from .utils import get_device, AverageMeter, accuracy
from .checkpoint import save_checkpoint, load_checkpoint


def create_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> SGD:
    """
    Create SGD optimizer with specified hyperparameters.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters to optimize.
    lr : float
        Learning rate. Default 0.01.
    momentum : float
        SGD momentum. Default 0.9.
    weight_decay : float
        L2 regularization. Default 5e-4.

    Returns
    -------
    SGD
        Configured optimizer.
    """
    return SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer: Optimizer,
    lr_steps: Optional[List[int]] = None,
    lr_gamma: float = 0.1,
) -> MultiStepLR:
    """
    Create step decay learning rate scheduler.

    Matches original FFN paper: LR multiplied by gamma at each milestone.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    lr_steps : list of int, optional
        Epoch milestones to decay LR. Default [20, 40].
    lr_gamma : float
        Multiplicative factor at each milestone. Default 0.1.

    Returns
    -------
    MultiStepLR
        Configured scheduler.
    """
    if lr_steps is None:
        lr_steps = [20, 40]
    return MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)


class Trainer:
    """
    Training manager for vanilla TSM.

    Handles complete training loop with logging, validation,
    and checkpointing. Supports AMP and gradient clipping
    to match original paper settings.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : Optimizer
        Optimizer for weight updates.
    scheduler : LRScheduler
        Learning rate scheduler.
    device : torch.device, optional
        Device to train on. Auto-detected if None.
    checkpoint_dir : str, optional
        Directory to save checkpoints. Default "checkpoints".
    use_amp : bool, optional
        Enable automatic mixed precision. Default False.
    max_grad_norm : float or None, optional
        Max gradient norm for clipping. None disables clipping. Default None.

    Attributes
    ----------
    model : nn.Module
        The model being trained.
    criterion : nn.CrossEntropyLoss
        Loss function.
    best_acc : float
        Best validation accuracy achieved.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AMP setup (only on CUDA - MPS/CPU don't support GradScaler)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Uses AMP autocast and GradScaler when enabled.
        Applies gradient clipping when max_grad_norm is set.

        Returns
        -------
        dict
            Training metrics: loss, top1_acc, top5_acc.
        """
        self.model.train()

        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")

        start_time = time.time()
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            # Move to device
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with optional AMP
            self.optimizer.zero_grad()
            with torch.autocast(device_type=amp_device, enabled=self.use_amp):
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

            # Backward pass with optional AMP scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

            # Compute accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            # Update meters
            batch_size = videos.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} "
                    f"Acc@1: {top1.avg:.2f}% "
                    f"Acc@5: {top5.avg:.2f}% "
                    f"Time: {elapsed:.1f}s"
                )

        return {
            "loss": losses.avg,
            "top1_acc": top1.avg,
            "top5_acc": top5.avg,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Uses autocast when AMP is enabled for faster inference.

        Returns
        -------
        dict
            Validation metrics: loss, top1_acc, top5_acc.
        """
        self.model.eval()

        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        for videos, labels in self.val_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(device_type=amp_device, enabled=self.use_amp):
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            batch_size = videos.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

        return {
            "loss": losses.avg,
            "top1_acc": top1.avg,
            "top5_acc": top5.avg,
        }

    def train(
        self,
        epochs: int,
        start_epoch: int = 0,
        validate_every: int = 1,
        save_every: int = 5,
    ) -> None:
        """
        Run full training loop.

        Parameters
        ----------
        epochs : int
            Total epochs to train.
        start_epoch : int, optional
            Epoch to start from (for resuming). Default 0.
        validate_every : int, optional
            Validate every N epochs. Default 1.
        save_every : int, optional
            Save checkpoint every N epochs. Default 5.
        """
        print(f"Training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print("-" * 50)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            print(f"\nEpoch [{epoch + 1}/{epochs}] LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Train
            train_metrics = self.train_epoch()
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} "
                f"Acc@1: {train_metrics['top1_acc']:.2f}% "
                f"Acc@5: {train_metrics['top5_acc']:.2f}%"
            )

            # Step scheduler
            self.scheduler.step()

            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                print(
                    f"  Val   - Loss: {val_metrics['loss']:.4f} "
                    f"Acc@1: {val_metrics['top1_acc']:.2f}% "
                    f"Acc@5: {val_metrics['top5_acc']:.2f}%"
                )

                # Save best model
                if val_metrics["top1_acc"] > self.best_acc:
                    self.best_acc = val_metrics["top1_acc"]
                    self._save_checkpoint("best.pth")
                    print(f"  New best accuracy: {self.best_acc:.2f}%")

            # Regular checkpoint every N epochs
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pth")

            # Always save latest.pth for resume capability
            self._save_checkpoint("latest.pth")

        # Save final checkpoint
        self._save_checkpoint("final.pth")
        print(f"\nTraining complete. Best accuracy: {self.best_acc:.2f}%")

    def _save_checkpoint(self, filename: str) -> None:
        """Save checkpoint to checkpoint directory."""
        filepath = self.checkpoint_dir / filename
        extra = {}
        if self.scaler is not None:
            extra["scaler_state_dict"] = self.scaler.state_dict()
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            best_acc=self.best_acc,
            filepath=str(filepath),
            extra=extra if extra else None,
        )

    def resume(self, checkpoint_path: str) -> int:
        """
        Resume training from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.

        Returns
        -------
        int
            Epoch to resume from.
        """
        info = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.best_acc = info["best_acc"]

        # Restore scaler state if available
        if self.scaler is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return info["epoch"] + 1  # Resume from next epoch
