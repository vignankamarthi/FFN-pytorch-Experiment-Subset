"""TSM-ResNet50 model for video action recognition.

Combines Temporal Shift Module with ResNet-50 backbone for
efficient video understanding.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .temporal_shift import TemporalShift


class TSMResNet50(nn.Module):
    """
    ResNet-50 with Temporal Shift Module for video classification.

    Architecture:
    1. ResNet-50 backbone with TSM inserted after conv1 of each residual block
    2. Global average pooling
    3. Temporal consensus (average predictions across frames)
    4. Final classification layer

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 174 for Something-Something V2.
    num_frames : int
        Number of input frames T. Default 16.
    pretrained : bool
        Whether to use ImageNet pretrained weights. Default True.
    dropout : float
        Dropout rate before final FC layer. Default 0.5.
    shift_div : int
        Channel division for TSM. Default 8 (1/8 forward, 1/8 backward).

    Attributes
    ----------
    backbone : nn.Module
        ResNet-50 backbone with TSM inserted.
    dropout : nn.Dropout
        Dropout layer.
    fc : nn.Linear
        Final classification layer.
    """

    def __init__(
        self,
        num_classes: int = 174,
        num_frames: int = 16,
        pretrained: bool = True,
        dropout: float = 0.5,
        shift_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.shift_div = shift_div

        # Load pretrained ResNet-50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # Insert TSM into residual blocks
        self._insert_tsm()

        # Get feature dimension from backbone
        feature_dim = self.backbone.fc.in_features  # 2048 for ResNet-50

        # Replace backbone's FC with identity (we'll add our own)
        self.backbone.fc = nn.Identity()

        # Add dropout and new classification head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

        # Initialize new FC layer
        nn.init.normal_(self.fc.weight, 0, 0.001)
        nn.init.constant_(self.fc.bias, 0)

    def _insert_tsm(self) -> None:
        """
        Insert Temporal Shift Module into ResNet backbone.

        TSM is inserted after conv1 of each residual block in all 4 stages.
        This is the 'blockres' placement strategy from the TSM paper.
        """
        def make_block_temporal(stage: nn.Sequential) -> nn.Sequential:
            """Wrap conv1 of each block with TSM."""
            blocks = list(stage.children())
            for block in blocks:
                # Wrap conv1 with TemporalShift
                block.conv1 = TemporalShift(
                    block.conv1,
                    num_frames=self.num_frames,
                    shift_div=self.shift_div,
                )
            return nn.Sequential(*blocks)

        self.backbone.layer1 = make_block_temporal(self.backbone.layer1)
        self.backbone.layer2 = make_block_temporal(self.backbone.layer2)
        self.backbone.layer3 = make_block_temporal(self.backbone.layer3)
        self.backbone.layer4 = make_block_temporal(self.backbone.layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TSM-ResNet50.

        Parameters
        ----------
        x : torch.Tensor
            Input video tensor of shape (B, C, T, H, W) where:
            - B: batch size
            - C: channels (3 for RGB)
            - T: number of frames
            - H, W: spatial dimensions (224)

        Returns
        -------
        torch.Tensor
            Class predictions of shape (B, num_classes).
        """
        b, c, t, h, w = x.size()

        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        # ResNet expects 2D images, so we treat each frame independently
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(b * t, c, h, w)  # (B*T, C, H, W)

        # Forward through backbone
        # TSM inside handles temporal interaction via channel shifting
        features = self.backbone(x)  # (B*T, 2048)

        # Apply dropout
        features = self.dropout(features)

        # Get class predictions for each frame
        out = self.fc(features)  # (B*T, num_classes)

        # Temporal consensus: average predictions across frames
        # Reshape to (B, T, num_classes) then mean over T
        out = out.view(b, t, -1)  # (B, T, num_classes)
        out = out.mean(dim=1)  # (B, num_classes)

        return out


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns
    -------
    torch.device
        CUDA if available, else MPS if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_tsm_model(
    num_classes: int = 174,
    num_frames: int = 16,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> TSMResNet50:
    """
    Factory function to create TSM-ResNet50 model.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 174.
    num_frames : int
        Number of input frames. Default 16.
    pretrained : bool
        Use ImageNet pretrained weights. Default True.
    device : torch.device, optional
        Device to place model on. If None, auto-selects best device.

    Returns
    -------
    TSMResNet50
        Model ready for training or inference.
    """
    if device is None:
        device = get_device()

    model = TSMResNet50(
        num_classes=num_classes,
        num_frames=num_frames,
        pretrained=pretrained,
    )

    return model.to(device)
