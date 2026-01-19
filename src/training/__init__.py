"""Training infrastructure for FFN reproduction.

Contains:
- Trainer class for vanilla TSM training
- FFNTrainer class for FFN multi-frame training
- Checkpoint save/load utilities
- Training utilities (device selection, metrics)
"""

from .utils import get_device, AverageMeter, accuracy
from .checkpoint import save_checkpoint, load_checkpoint
from .trainer import Trainer, create_optimizer, create_scheduler
from .ffn_trainer import (
    FFNTrainer,
    create_ffn_optimizer,
    create_ffn_scheduler,
    evaluate_tfd,
)

__all__ = [
    # Utilities
    "get_device",
    "AverageMeter",
    "accuracy",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    # Vanilla TSM Trainer
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    # FFN Trainer
    "FFNTrainer",
    "create_ffn_optimizer",
    "create_ffn_scheduler",
    "evaluate_tfd",
]
