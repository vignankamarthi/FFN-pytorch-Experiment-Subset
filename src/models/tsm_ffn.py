"""TSM-FFN model for Frame Flexible Network.

Implements FFN's key components:
1. Specialized BatchNorm: Private BN layers per frame count (4F, 8F, 16F)
2. Weight Alteration: Depthwise adapters with near-zero initialization
3. Temporal Shift Module: Channel shifting for temporal modeling

Reference: Frame Flexible Network (CVPR 2023)
Paper: https://arxiv.org/abs/2303.14817
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


# Pretrained weights URL
RESNET50_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TSM(nn.Module):
    """
    Temporal Shift Module for FFN.

    Shifts 1/8 of channels forward in time and 1/8 backward,
    allowing temporal information exchange.

    Parameters
    ----------
    num_segments : int
        Number of temporal segments (frames).
    fold_div : int
        Channel division factor. Default 8.
    """

    def __init__(self, num_segments: int = 16, fold_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(
        self, x: torch.Tensor, num_segments: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply temporal shift.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N*T, C, H, W).
        num_segments : int, optional
            Number of segments to use. If None, uses self.num_segments.

        Returns
        -------
        torch.Tensor
            Shifted tensor of same shape.
        """
        n_seg = num_segments if num_segments is not None else self.num_segments
        return self._shift(x, n_seg, fold_div=self.fold_div)

    @staticmethod
    def _shift(x: torch.Tensor, num_segments: int, fold_div: int = 8) -> torch.Tensor:
        """Static shift implementation."""
        nt, c, h, w = x.size()
        n_batch = nt // num_segments
        x = x.view(n_batch, num_segments, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left (future -> current)
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # unchanged

        return out.view(nt, c, h, w)


class FFNBottleneck(nn.Module):
    """
    ResNet Bottleneck block with FFN components.

    Key FFN modifications:
    1. Specialized BatchNorm: bn{1,2,3}_{4,8,16} - private per frame count
    2. Weight Alteration: adaconv_{4,8,16} - depthwise adapters with residual
    3. TSM: Applied before first conv

    The architecture uses SHARED convolutions and PRIVATE BatchNorms.
    This allows the network to learn frame-count-specific statistics
    while reusing the learned filters.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Intermediate channels (output = planes * 4).
    num_segments_h : int
        Number of segments for high frame count (16F).
    stride : int
        Stride for 3x3 conv.
    downsample_conv : nn.Conv2d, optional
        Downsample convolution (shared).
    downsample_bn_{4,8,16} : nn.BatchNorm2d, optional
        Downsample BatchNorm (per frame count).
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        num_segments_h: int = 16,
        num_segments_m: int = 8,
        num_segments_l: int = 4,
        stride: int = 1,
        downsample_conv: Optional[nn.Conv2d] = None,
        downsample_bn_4: Optional[nn.BatchNorm2d] = None,
        downsample_bn_8: Optional[nn.BatchNorm2d] = None,
        downsample_bn_16: Optional[nn.BatchNorm2d] = None,
    ) -> None:
        super().__init__()

        # Store segment counts for each frame count
        self.num_segments_h = num_segments_h  # 16F
        self.num_segments_m = num_segments_m  # 8F
        self.num_segments_l = num_segments_l  # 4F

        # Shared convolutions
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)

        # Specialized BatchNorm: private per frame count
        # conv1 output
        self.bn1_4 = nn.BatchNorm2d(planes)
        self.bn1_8 = nn.BatchNorm2d(planes)
        self.bn1_16 = nn.BatchNorm2d(planes)
        # conv2 output
        self.bn2_4 = nn.BatchNorm2d(planes)
        self.bn2_8 = nn.BatchNorm2d(planes)
        self.bn2_16 = nn.BatchNorm2d(planes)
        # conv3 output
        self.bn3_4 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_8 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_16 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Temporal Shift Module (default segments, but can be overridden)
        self.tsm = TSM(num_segments_h)

        # Downsample path (shared conv, private BN)
        self.downsample_conv = downsample_conv
        self.downsample_bn_4 = downsample_bn_4
        self.downsample_bn_8 = downsample_bn_8
        self.downsample_bn_16 = downsample_bn_16

        self.stride = stride

        # Weight Alteration adapters: depthwise 1x1 convs
        # groups=inplanes makes this depthwise (C parameters, not C^2)
        # Near-zero init (sigma=1e-3) means y = x + adaconv(x) ≈ x at start
        self.adaconv_4 = nn.Conv2d(
            inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False
        )
        self.adaconv_8 = nn.Conv2d(
            inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False
        )
        self.adaconv_16 = nn.Conv2d(
            inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False
        )
        # Near-zero initialization preserves pretrained behavior
        self.adaconv_4.weight.data.normal_(0, 1e-3)
        self.adaconv_8.weight.data.normal_(0, 1e-3)
        self.adaconv_16.weight.data.normal_(0, 1e-3)

    def forward(
        self,
        x_4: Optional[torch.Tensor],
        x_8: Optional[torch.Tensor],
        x_16: Optional[torch.Tensor],
        training: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through FFN Bottleneck.

        In training mode, processes all three frame counts in parallel.
        In inference mode, processes only the non-None input.

        Parameters
        ----------
        x_4 : torch.Tensor or None
            4-frame input of shape (B*4, C, H, W).
        x_8 : torch.Tensor or None
            8-frame input of shape (B*8, C, H, W).
        x_16 : torch.Tensor or None
            16-frame input of shape (B*16, C, H, W).
        training : bool
            If True, process all three inputs. If False, process non-None input.

        Returns
        -------
        Tuple or Tensor
            Training: (out_4, out_8, out_16)
            Inference: single output tensor
        """
        if training:
            # In training mode, all inputs must be provided
            assert x_4 is not None and x_8 is not None and x_16 is not None
            return self._forward_train(x_4, x_8, x_16)
        else:
            return self._forward_inference(x_4, x_8, x_16)

    def _forward_train(
        self, x_4: torch.Tensor, x_8: torch.Tensor, x_16: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process all three frame counts during training."""
        identity_4 = x_4
        identity_8 = x_8
        identity_16 = x_16

        # 4-frame path (use num_segments_l for 4-frame input)
        out_4 = self.tsm(x_4, num_segments=self.num_segments_l)
        iden_4 = out_4
        out_4 = self.adaconv_4(out_4)
        out_4 = out_4 + iden_4  # Residual connection (Eq. 8)
        out_4 = self.conv1(out_4)
        out_4 = self.bn1_4(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv2(out_4)
        out_4 = self.bn2_4(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv3(out_4)
        out_4 = self.bn3_4(out_4)

        # 8-frame path (use num_segments_m for 8-frame input)
        out_8 = self.tsm(x_8, num_segments=self.num_segments_m)
        iden_8 = out_8
        out_8 = self.adaconv_8(out_8)
        out_8 = out_8 + iden_8
        out_8 = self.conv1(out_8)
        out_8 = self.bn1_8(out_8)
        out_8 = self.relu(out_8)
        out_8 = self.conv2(out_8)
        out_8 = self.bn2_8(out_8)
        out_8 = self.relu(out_8)
        out_8 = self.conv3(out_8)
        out_8 = self.bn3_8(out_8)

        # 16-frame path (use num_segments_h for 16-frame input)
        out_16 = self.tsm(x_16, num_segments=self.num_segments_h)
        iden_16 = out_16
        out_16 = self.adaconv_16(out_16)
        out_16 = out_16 + iden_16
        out_16 = self.conv1(out_16)
        out_16 = self.bn1_16(out_16)
        out_16 = self.relu(out_16)
        out_16 = self.conv2(out_16)
        out_16 = self.bn2_16(out_16)
        out_16 = self.relu(out_16)
        out_16 = self.conv3(out_16)
        out_16 = self.bn3_16(out_16)

        # Downsample if needed
        if self.downsample_conv is not None:
            assert self.downsample_bn_4 is not None
            assert self.downsample_bn_8 is not None
            assert self.downsample_bn_16 is not None
            identity_4 = self.downsample_bn_4(self.downsample_conv(x_4))
            identity_8 = self.downsample_bn_8(self.downsample_conv(x_8))
            identity_16 = self.downsample_bn_16(self.downsample_conv(x_16))

        out_4 = out_4 + identity_4
        out_4 = self.relu(out_4)
        out_8 = out_8 + identity_8
        out_8 = self.relu(out_8)
        out_16 = out_16 + identity_16
        out_16 = self.relu(out_16)

        return out_4, out_8, out_16

    def _forward_inference(
        self,
        x_4: Optional[torch.Tensor],
        x_8: Optional[torch.Tensor],
        x_16: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Process single frame count during inference."""
        if x_4 is not None:
            x = x_4
            num_segments = self.num_segments_l
            bn1, bn2, bn3, adaconv, downsample_bn = (
                self.bn1_4,
                self.bn2_4,
                self.bn3_4,
                self.adaconv_4,
                self.downsample_bn_4,
            )
        elif x_8 is not None:
            x = x_8
            num_segments = self.num_segments_m
            bn1, bn2, bn3, adaconv, downsample_bn = (
                self.bn1_8,
                self.bn2_8,
                self.bn3_8,
                self.adaconv_8,
                self.downsample_bn_8,
            )
        else:
            # x_16 must be provided
            assert x_16 is not None, "At least one input must be provided"
            x = x_16
            num_segments = self.num_segments_h
            bn1, bn2, bn3, adaconv, downsample_bn = (
                self.bn1_16,
                self.bn2_16,
                self.bn3_16,
                self.adaconv_16,
                self.downsample_bn_16,
            )

        identity = x
        out = self.tsm(x, num_segments=num_segments)
        iden = out
        out = adaconv(out)
        out = out + iden
        out = self.conv1(out)
        out = bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = bn3(out)

        if self.downsample_conv is not None and downsample_bn is not None:
            identity = downsample_bn(self.downsample_conv(x))

        out = out + identity
        out = self.relu(out)

        return out


class FFNResNet(nn.Module):
    """
    ResNet backbone with FFN components for frame-flexible video recognition.

    Architecture follows standard ResNet-50 but with:
    1. Specialized BatchNorm per frame count (4F, 8F, 16F)
    2. Weight Alteration adapters in each block
    3. TSM for temporal modeling

    Parameters
    ----------
    block : type
        Block class (FFNBottleneck).
    layers : list
        Number of blocks per stage [3, 4, 6, 3] for ResNet-50.
    num_segments_h : int
        High frame count. Default 16.
    num_segments_m : int
        Medium frame count. Default 8.
    num_segments_l : int
        Low frame count. Default 4.
    num_classes : int
        Number of output classes. Default 174 for SSv2.
    """

    def __init__(
        self,
        block: type,
        layers: list,
        num_segments_h: int = 16,
        num_segments_m: int = 8,
        num_segments_l: int = 4,
        num_classes: int = 174,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.num_segments_h = num_segments_h
        self.num_segments_m = num_segments_m
        self.num_segments_l = num_segments_l

        # Initial convolution (shared)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initial BatchNorm (specialized per frame count)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.bn1_8 = nn.BatchNorm2d(64)
        self.bn1_16 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.ModuleList:
        """Build a residual stage with specialized downsampling."""
        downsample_conv = None
        downsample_bn_4 = None
        downsample_bn_8 = None
        downsample_bn_16 = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample_bn_4 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_8 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_16 = nn.BatchNorm2d(planes * block.expansion)

        layers = nn.ModuleList()
        layers.append(
            block(
                self.inplanes,
                planes,
                num_segments_h=self.num_segments_h,
                num_segments_m=self.num_segments_m,
                num_segments_l=self.num_segments_l,
                stride=stride,
                downsample_conv=downsample_conv,
                downsample_bn_4=downsample_bn_4,
                downsample_bn_8=downsample_bn_8,
                downsample_bn_16=downsample_bn_16,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    num_segments_h=self.num_segments_h,
                    num_segments_m=self.num_segments_m,
                    num_segments_l=self.num_segments_l,
                )
            )

        return layers

    def forward(
        self,
        x_4: Optional[torch.Tensor],
        x_8: Optional[torch.Tensor],
        x_16: Optional[torch.Tensor],
        training: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through FFN ResNet.

        Parameters
        ----------
        x_4, x_8, x_16 : torch.Tensor or None
            Inputs at different frame counts, shape (B*T, C, H, W).
        training : bool
            If True, process all inputs. If False, process non-None input.

        Returns
        -------
        Tuple or Tensor
            Training: (logits_4, logits_8, logits_16) each (B, num_classes)
            Inference: single logits tensor (B, num_classes)
        """
        if training:
            # In training mode, all inputs must be provided
            assert x_4 is not None and x_8 is not None and x_16 is not None
            return self._forward_train(x_4, x_8, x_16)
        else:
            return self._forward_inference(x_4, x_8, x_16)

    def _forward_train(
        self, x_4: torch.Tensor, x_8: torch.Tensor, x_16: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training mode: process all three frame counts."""
        # Initial convolution (shared conv, specialized BN)
        x_4 = self.conv1(x_4)
        x_4 = self.bn1_4(x_4)
        x_4 = self.relu(x_4)
        x_4 = self.maxpool(x_4)

        x_8 = self.conv1(x_8)
        x_8 = self.bn1_8(x_8)
        x_8 = self.relu(x_8)
        x_8 = self.maxpool(x_8)

        x_16 = self.conv1(x_16)
        x_16 = self.bn1_16(x_16)
        x_16 = self.relu(x_16)
        x_16 = self.maxpool(x_16)

        # Residual stages
        for block in self.layer1:
            x_4, x_8, x_16 = block(x_4, x_8, x_16, True)
        for block in self.layer2:
            x_4, x_8, x_16 = block(x_4, x_8, x_16, True)
        for block in self.layer3:
            x_4, x_8, x_16 = block(x_4, x_8, x_16, True)
        for block in self.layer4:
            x_4, x_8, x_16 = block(x_4, x_8, x_16, True)

        # Pool and classify each
        def classify(x: torch.Tensor, num_segments: int) -> torch.Tensor:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # Temporal consensus: average across frames
            batch_size = x.size(0) // num_segments
            x = x.view(batch_size, num_segments, -1).mean(dim=1)
            return x

        out_4 = classify(x_4, self.num_segments_l)
        out_8 = classify(x_8, self.num_segments_m)
        out_16 = classify(x_16, self.num_segments_h)

        return out_4, out_8, out_16

    def _forward_inference(
        self,
        x_4: Optional[torch.Tensor],
        x_8: Optional[torch.Tensor],
        x_16: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Inference mode: process single frame count."""
        if x_4 is not None:
            x, bn1, num_seg = x_4, self.bn1_4, self.num_segments_l
            which_input = 0  # Track which position
        elif x_8 is not None:
            x, bn1, num_seg = x_8, self.bn1_8, self.num_segments_m
            which_input = 1
        else:
            assert x_16 is not None
            x, bn1, num_seg = x_16, self.bn1_16, self.num_segments_h
            which_input = 2

        # Initial layers
        x = self.conv1(x)
        x = bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Helper to create forward args tuple based on which input is active
        def make_forward_args(x_val: torch.Tensor) -> tuple:
            if which_input == 0:
                return (x_val, None, None)
            elif which_input == 1:
                return (None, x_val, None)
            else:
                return (None, None, x_val)

        # Residual stages
        for block in self.layer1:
            x = block(*make_forward_args(x), False)

        for block in self.layer2:
            x = block(*make_forward_args(x), False)

        for block in self.layer3:
            x = block(*make_forward_args(x), False)

        for block in self.layer4:
            x = block(*make_forward_args(x), False)

        # Classify
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Temporal consensus
        batch_size = x.size(0) // num_seg
        x = x.view(batch_size, num_seg, -1).mean(dim=1)

        return x


def resnet50_ffn(
    pretrained: bool = True,
    num_segments_h: int = 16,
    num_segments_m: int = 8,
    num_segments_l: int = 4,
    num_classes: int = 174,
) -> FFNResNet:
    """
    Construct ResNet-50 with FFN components.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet pretrained weights, mapped to specialized BN.
    num_segments_h, num_segments_m, num_segments_l : int
        Frame counts for high/medium/low.
    num_classes : int
        Number of output classes.

    Returns
    -------
    FFNResNet
        Model ready for FFN training.
    """
    model = FFNResNet(
        FFNBottleneck,
        [3, 4, 6, 3],  # ResNet-50 configuration
        num_segments_h=num_segments_h,
        num_segments_m=num_segments_m,
        num_segments_l=num_segments_l,
        num_classes=num_classes,
    )

    if pretrained:
        pretrained_dict = load_state_dict_from_url(RESNET50_URL)
        new_state_dict = model.state_dict()

        # Map pretrained weights to specialized BN layers
        for k in new_state_dict.keys():
            # Skip fc layer if num_classes != 1000 (ImageNet)
            if "fc" in k and num_classes != 1000:
                continue
            if "downsample_conv" in k:
                # downsample_conv -> downsample.0
                src_key = k.replace("_conv", ".0")
                if src_key in pretrained_dict:
                    new_state_dict[k] = pretrained_dict[src_key]
            elif "downsample_bn_4" in k and "num_batches_tracked" not in k:
                src_key = k.replace("_bn_4", ".1")
                if src_key in pretrained_dict:
                    new_state_dict[k] = pretrained_dict[src_key]
            elif "downsample_bn_8" in k and "num_batches_tracked" not in k:
                src_key = k.replace("_bn_8", ".1")
                if src_key in pretrained_dict:
                    new_state_dict[k] = pretrained_dict[src_key]
            elif "downsample_bn_16" in k and "num_batches_tracked" not in k:
                src_key = k.replace("_bn_16", ".1")
                if src_key in pretrained_dict:
                    new_state_dict[k] = pretrained_dict[src_key]
            elif k.replace("_4", "") in pretrained_dict:
                new_state_dict[k] = pretrained_dict[k.replace("_4", "")]
            elif k.replace("_8", "") in pretrained_dict:
                new_state_dict[k] = pretrained_dict[k.replace("_8", "")]
            elif k.replace("_16", "") in pretrained_dict:
                new_state_dict[k] = pretrained_dict[k.replace("_16", "")]
            elif k in pretrained_dict:
                new_state_dict[k] = pretrained_dict[k]
            # Skip adaconv weights - they use near-zero init

        model.load_state_dict(new_state_dict)

    return model


class TSMFFN(nn.Module):
    """
    High-level wrapper for TSM-FFN model.

    Provides a clean interface for training and inference,
    handling input reshaping automatically.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 174 for SSv2.
    num_frames_h : int
        High frame count. Default 16.
    num_frames_m : int
        Medium frame count. Default 8.
    num_frames_l : int
        Low frame count. Default 4.
    pretrained : bool
        Load ImageNet pretrained weights. Default True.
    dropout : float
        Dropout rate (applied during training). Default 0.5.

    Examples
    --------
    >>> model = TSMFFN(num_classes=174)
    >>> # Training: pass all three frame counts
    >>> v_4 = torch.randn(2, 3, 4, 224, 224)
    >>> v_8 = torch.randn(2, 3, 8, 224, 224)
    >>> v_16 = torch.randn(2, 3, 16, 224, 224)
    >>> out_4, out_8, out_16 = model(v_4, v_8, v_16, training=True)
    >>> # Inference: pass single frame count
    >>> out = model(v_8, num_frames=8)
    """

    def __init__(
        self,
        num_classes: int = 174,
        num_frames_h: int = 16,
        num_frames_m: int = 8,
        num_frames_l: int = 4,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_frames_h = num_frames_h
        self.num_frames_m = num_frames_m
        self.num_frames_l = num_frames_l
        self.dropout = nn.Dropout(p=dropout)

        self.backbone = resnet50_ffn(
            pretrained=pretrained,
            num_segments_h=num_frames_h,
            num_segments_m=num_frames_m,
            num_segments_l=num_frames_l,
            num_classes=num_classes,
        )

    def forward(
        self,
        x_4: Optional[torch.Tensor] = None,
        x_8: Optional[torch.Tensor] = None,
        x_16: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x_4, x_8, x_16 : torch.Tensor or None
            Video tensors of shape (B, C, T, H, W) for training mode.
        training : bool
            If True, use all three inputs for temporal distillation.

        Returns
        -------
        Tuple or Tensor
            Training: (logits_4, logits_8, logits_16)
            Inference: single logits tensor
        """
        if training:
            # Training mode: process all three
            # All inputs must be provided in training mode
            assert x_4 is not None, "x_4 required in training mode"
            assert x_8 is not None, "x_8 required in training mode"
            assert x_16 is not None, "x_16 required in training mode"
            # Reshape from (B, C, T, H, W) to (B*T, C, H, W)
            x_4_flat = self._reshape_input(x_4)
            x_8_flat = self._reshape_input(x_8)
            x_16_flat = self._reshape_input(x_16)

            return self.backbone(x_4_flat, x_8_flat, x_16_flat, training=True)
        else:
            # Inference mode: use single input
            if x_4 is not None:
                x_flat = self._reshape_input(x_4)
                return self.backbone(x_flat, None, None, training=False)
            elif x_8 is not None:
                x_flat = self._reshape_input(x_8)
                return self.backbone(None, x_flat, None, training=False)
            elif x_16 is not None:
                x_flat = self._reshape_input(x_16)
                return self.backbone(None, None, x_flat, training=False)
            else:
                raise ValueError("At least one input must be provided")

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C, T, H, W) to (B*T, C, H, W)."""
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(b * t, c, h, w)  # (B*T, C, H, W)
        return x


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


def create_ffn_model(
    num_classes: int = 174,
    num_frames_h: int = 16,
    num_frames_m: int = 8,
    num_frames_l: int = 4,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> TSMFFN:
    """
    Factory function to create TSM-FFN model.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 174.
    num_frames_h, num_frames_m, num_frames_l : int
        Frame counts for high/medium/low.
    pretrained : bool
        Use ImageNet pretrained weights. Default True.
    device : torch.device, optional
        Device to place model on. If None, auto-selects best device.

    Returns
    -------
    TSMFFN
        Model ready for FFN training.
    """
    if device is None:
        device = get_device()

    model = TSMFFN(
        num_classes=num_classes,
        num_frames_h=num_frames_h,
        num_frames_m=num_frames_m,
        num_frames_l=num_frames_l,
        pretrained=pretrained,
    )

    return model.to(device)
