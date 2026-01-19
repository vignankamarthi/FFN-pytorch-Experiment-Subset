"""Model implementations for FFN reproduction.

Contains:
- TSM (Temporal Shift Module) with ResNet-50 backbone (vanilla)
- FFN components: TSM-FFN model with Specialized BatchNorm and Weight Alteration
- Temporal Distillation Loss for FFN training
"""

from .temporal_shift import TemporalShift, temporal_shift
from .tsm import TSMResNet50, create_tsm_model, get_device
from .tsm_ffn import (
    TSMFFN,
    FFNBottleneck,
    FFNResNet,
    resnet50_ffn,
    create_ffn_model,
)
from .temporal_distillation import (
    TemporalDistillationLoss,
    FFNLoss,
    compute_ffn_loss,
)

__all__ = [
    # Vanilla TSM
    "TemporalShift",
    "temporal_shift",
    "TSMResNet50",
    "create_tsm_model",
    "get_device",
    # FFN Model
    "TSMFFN",
    "FFNBottleneck",
    "FFNResNet",
    "resnet50_ffn",
    "create_ffn_model",
    # FFN Loss
    "TemporalDistillationLoss",
    "FFNLoss",
    "compute_ffn_loss",
]
