"""Checkpoint save/load utilities for training.

Handles model state, optimizer state, and training progress.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    best_acc: float,
    filepath: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint to disk.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    optimizer : Optimizer
        Optimizer state to save.
    scheduler : LRScheduler, optional
        Learning rate scheduler state.
    epoch : int
        Current epoch number.
    best_acc : float
        Best validation accuracy so far.
    filepath : str
        Path to save checkpoint.
    extra : dict, optional
        Additional data to save (e.g., config, metrics).
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint from disk.

    Parameters
    ----------
    filepath : str
        Path to checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : Optimizer, optional
        Optimizer to load state into. If None, skipped.
    scheduler : LRScheduler, optional
        Scheduler to load state into. If None, skipped.
    device : torch.device, optional
        Device to map checkpoint to. If None, uses checkpoint's device.

    Returns
    -------
    dict
        Checkpoint data including epoch, best_acc, and any extra data.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(filepath, weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_acc": checkpoint.get("best_acc", 0.0),
    }
