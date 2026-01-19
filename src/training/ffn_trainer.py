"""FFN Trainer class for Frame Flexible Network training.

Handles the complete FFN training loop including:
- Multi-frame forward passes (4F, 8F, 16F simultaneously)
- Temporal distillation loss computation
- Per-frame-count validation
- Comprehensive logging of all loss components
"""

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from .utils import get_device, AverageMeter, accuracy
from .checkpoint import save_checkpoint, load_checkpoint


def create_ffn_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> SGD:
    """
    Create SGD optimizer for FFN training.

    Parameters
    ----------
    model : nn.Module
        FFN model whose parameters to optimize.
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


def create_ffn_scheduler(
    optimizer: Optimizer,
    epochs: int = 50,
) -> CosineAnnealingLR:
    """
    Create cosine annealing scheduler for FFN training.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    epochs : int
        Total training epochs. Default 50.

    Returns
    -------
    CosineAnnealingLR
        Configured scheduler.
    """
    return CosineAnnealingLR(optimizer, T_max=epochs)


class FFNTrainer:
    """
    Training manager for Frame Flexible Network.

    Handles FFN-specific training with multi-frame inputs and
    temporal distillation loss.

    Parameters
    ----------
    model : nn.Module
        TSMFFN model to train.
    train_loader : DataLoader
        Training data loader (returns v_4, v_8, v_16, label).
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        TemporalDistillationLoss or similar.
    optimizer : Optimizer
        Optimizer for weight updates.
    scheduler : LRScheduler
        Learning rate scheduler.
    device : torch.device, optional
        Device to train on. Auto-detected if None.
    checkpoint_dir : str, optional
        Directory to save checkpoints. Default "checkpoints".
    lambda_kl : float, optional
        Weight for KL divergence loss. Default 1.0.

    Attributes
    ----------
    model : nn.Module
        The FFN model being trained.
    criterion : nn.Module
        Loss function with temporal distillation.
    best_acc : float
        Best validation accuracy achieved (on 16F).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        lambda_kl: float = 1.0,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.lambda_kl = lambda_kl

        # Tracking
        self.best_acc = 0.0
        self.best_acc_4f = 0.0
        self.best_acc_8f = 0.0
        self.current_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with multi-frame inputs.

        Returns
        -------
        dict
            Training metrics including per-frame-count accuracies.
        """
        self.model.train()

        # Meters for overall loss
        losses = AverageMeter("Loss")
        losses_ce = AverageMeter("CE Loss")
        losses_kl = AverageMeter("KL Loss")

        # Meters for per-frame-count accuracy
        top1_4f = AverageMeter("Acc@1 (4F)")
        top1_8f = AverageMeter("Acc@1 (8F)")
        top1_16f = AverageMeter("Acc@1 (16F)")

        start_time = time.time()

        for batch_idx, (v_4, v_8, v_16, labels) in enumerate(self.train_loader):
            # Move to device
            v_4 = v_4.to(self.device)
            v_8 = v_8.to(self.device)
            v_16 = v_16.to(self.device)
            labels = labels.to(self.device)

            # Forward pass through all three frame counts
            out_4, out_8, out_16 = self.model(v_4, v_8, v_16, training=True)

            # Compute loss with temporal distillation
            loss, loss_dict = self.criterion(out_4, out_8, out_16, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute per-frame-count accuracy
            acc1_4f, _ = accuracy(out_4, labels, topk=(1, 5))
            acc1_8f, _ = accuracy(out_8, labels, topk=(1, 5))
            acc1_16f, _ = accuracy(out_16, labels, topk=(1, 5))

            # Update meters
            batch_size = v_16.size(0)
            losses.update(loss.item(), batch_size)
            losses_ce.update(loss_dict.get("loss_ce", 0), batch_size)
            losses_kl.update(
                loss_dict.get("loss_kl_total", loss_dict.get("loss_kl", 0)),
                batch_size,
            )

            top1_4f.update(acc1_4f.item(), batch_size)
            top1_8f.update(acc1_8f.item(), batch_size)
            top1_16f.update(acc1_16f.item(), batch_size)

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} (CE: {losses_ce.avg:.4f}, KL: {losses_kl.avg:.4f}) "
                    f"Time: {elapsed:.1f}s"
                )
                print(
                    f"    Acc@1 - 4F: {top1_4f.avg:.2f}% | "
                    f"8F: {top1_8f.avg:.2f}% | "
                    f"16F: {top1_16f.avg:.2f}%"
                )

        return {
            "loss": losses.avg,
            "loss_ce": losses_ce.avg,
            "loss_kl": losses_kl.avg,
            "top1_4f": top1_4f.avg,
            "top1_8f": top1_8f.avg,
            "top1_16f": top1_16f.avg,
        }

    @torch.no_grad()
    def validate(self, frame_count: Optional[int] = None) -> Dict[str, float]:
        """
        Run validation at specified frame count(s).

        Parameters
        ----------
        frame_count : int, optional
            If specified, validate only at this frame count.
            If None, validate at all three frame counts.

        Returns
        -------
        dict
            Validation metrics.
        """
        self.model.eval()

        if frame_count is not None:
            return self._validate_single_frame_count(frame_count)
        else:
            return self._validate_all_frame_counts()

    def _validate_single_frame_count(self, frame_count: int) -> Dict[str, float]:
        """Validate at a single frame count."""
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")

        ce_criterion = nn.CrossEntropyLoss()

        for v_4, v_8, v_16, labels in self.val_loader:
            labels = labels.to(self.device)

            # Select input based on frame count
            if frame_count == 4:
                v = v_4.to(self.device)
                outputs = self.model(x_4=v, training=False)
            elif frame_count == 8:
                v = v_8.to(self.device)
                outputs = self.model(x_8=v, training=False)
            else:  # 16
                v = v_16.to(self.device)
                outputs = self.model(x_16=v, training=False)

            loss = ce_criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            batch_size = v.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

        return {
            "loss": losses.avg,
            f"top1_{frame_count}f": top1.avg,
            f"top5_{frame_count}f": top5.avg,
        }

    def _validate_all_frame_counts(self) -> Dict[str, float]:
        """Validate at all three frame counts."""
        metrics = {}

        for fc in [4, 8, 16]:
            fc_metrics = self._validate_single_frame_count(fc)
            metrics.update(fc_metrics)

        return metrics

    def train(
        self,
        epochs: int,
        start_epoch: int = 0,
        validate_every: int = 1,
        save_every: int = 5,
    ) -> None:
        """
        Run full FFN training loop.

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
        print(f"FFN Training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Lambda KL: {self.lambda_kl}")
        print("-" * 60)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            print(f"\nEpoch [{epoch + 1}/{epochs}] LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Train
            train_metrics = self.train_epoch()
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} "
                f"(CE: {train_metrics['loss_ce']:.4f}, KL: {train_metrics['loss_kl']:.4f})"
            )
            print(
                f"    Acc@1 - 4F: {train_metrics['top1_4f']:.2f}% | "
                f"8F: {train_metrics['top1_8f']:.2f}% | "
                f"16F: {train_metrics['top1_16f']:.2f}%"
            )

            # Step scheduler
            self.scheduler.step()

            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                print("  Validation:")
                print(
                    f"    4F:  Acc@1: {val_metrics.get('top1_4f', 0):.2f}%"
                )
                print(
                    f"    8F:  Acc@1: {val_metrics.get('top1_8f', 0):.2f}%"
                )
                print(
                    f"    16F: Acc@1: {val_metrics.get('top1_16f', 0):.2f}%"
                )

                # Save best model (based on 16F accuracy)
                acc_16f = val_metrics.get("top1_16f", 0)
                if acc_16f > self.best_acc:
                    self.best_acc = acc_16f
                    self._save_checkpoint("best.pth")
                    print(f"  New best 16F accuracy: {self.best_acc:.2f}%")

                # Track best for other frame counts too
                acc_4f = val_metrics.get("top1_4f", 0)
                acc_8f = val_metrics.get("top1_8f", 0)
                if acc_4f > self.best_acc_4f:
                    self.best_acc_4f = acc_4f
                if acc_8f > self.best_acc_8f:
                    self.best_acc_8f = acc_8f

            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pth")

        # Save final checkpoint
        self._save_checkpoint("final.pth")
        print(f"\nFFN Training complete!")
        print(f"  Best 4F accuracy:  {self.best_acc_4f:.2f}%")
        print(f"  Best 8F accuracy:  {self.best_acc_8f:.2f}%")
        print(f"  Best 16F accuracy: {self.best_acc:.2f}%")

    def _save_checkpoint(self, filename: str) -> None:
        """Save checkpoint to checkpoint directory."""
        filepath = self.checkpoint_dir / filename
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            best_acc=self.best_acc,
            filepath=str(filepath),
            extra={
                "best_acc_4f": self.best_acc_4f,
                "best_acc_8f": self.best_acc_8f,
                "lambda_kl": self.lambda_kl,
            },
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
        if "extra" in info:
            self.best_acc_4f = info["extra"].get("best_acc_4f", 0.0)
            self.best_acc_8f = info["extra"].get("best_acc_8f", 0.0)
        return info["epoch"] + 1


def evaluate_tfd(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate Temporal Frequency Deviation (TFD).

    Computes accuracy at each frame count to measure TFD gap.

    Parameters
    ----------
    model : nn.Module
        Trained FFN model.
    val_loader : DataLoader
        Validation loader (returns v_4, v_8, v_16, label).
    device : torch.device
        Device to evaluate on.

    Returns
    -------
    dict
        Accuracy at each frame count and TFD gap.
    """
    model.eval()
    model = model.to(device)

    results = {}

    for frame_count in [4, 8, 16]:
        top1 = AverageMeter(f"Acc@1 ({frame_count}F)")

        with torch.no_grad():
            for v_4, v_8, v_16, labels in val_loader:
                labels = labels.to(device)

                if frame_count == 4:
                    v = v_4.to(device)
                    outputs = model(x_4=v, training=False)
                elif frame_count == 8:
                    v = v_8.to(device)
                    outputs = model(x_8=v, training=False)
                else:
                    v = v_16.to(device)
                    outputs = model(x_16=v, training=False)

                acc1, _ = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1.item(), v.size(0))

        results[f"acc_{frame_count}f"] = top1.avg

    # Compute TFD gap (16F - 4F)
    results["tfd_gap"] = results["acc_16f"] - results["acc_4f"]

    return results
