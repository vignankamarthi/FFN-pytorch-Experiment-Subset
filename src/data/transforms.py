"""Video augmentation transforms for SSv2.

IMPORTANT: No horizontal flip for Something-Something V2!
The dataset contains directional actions like "pushing left to right"
which would be semantically changed by horizontal flipping.
"""

from typing import Callable, Tuple

import numpy as np
import torch
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoTransform:
    """
    Apply transforms to video frames.

    Handles the conversion from numpy array (T, H, W, C) to
    torch tensor (C, T, H, W) with appropriate augmentations.

    Parameters
    ----------
    spatial_transform : Callable
        Transform to apply to each frame spatially.
    normalize : bool
        Whether to apply ImageNet normalization.
    """

    def __init__(self, spatial_transform: Callable, normalize: bool = True) -> None:
        self.spatial_transform = spatial_transform
        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to video frames.

        Parameters
        ----------
        frames : np.ndarray
            Video frames of shape (T, H, W, C) with values in [0, 255].

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape (C, T, H, W) with values normalized.
        """
        # Apply spatial transform to each frame
        transformed_frames = []
        for i in range(frames.shape[0]):
            # Convert to PIL for torchvision transforms
            frame = frames[i]  # (H, W, C)
            # torchvision transforms expect PIL or tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = self.spatial_transform(frame_tensor)
            transformed_frames.append(frame_tensor)

        # Stack frames: list of (C, H, W) -> (T, C, H, W)
        video_tensor = torch.stack(transformed_frames, dim=0)

        # Rearrange to (C, T, H, W) for model input
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        # Normalize each frame
        if self.normalize:
            # video_tensor is (C, T, H, W), normalize expects (C, H, W)
            # Apply per-frame
            C, T, H, W = video_tensor.shape
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
            normalized = []
            for t in range(T):
                normalized.append(self.normalizer(video_tensor[t]))
            video_tensor = torch.stack(normalized, dim=0)  # (T, C, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

        return video_tensor


def get_train_transforms(crop_size: int = 224) -> VideoTransform:
    """
    Get training transforms for SSv2.

    Includes random resized crop for data augmentation.
    NO horizontal flip (directional actions in SSv2).

    Parameters
    ----------
    crop_size : int
        Output spatial size (default 224 for ResNet).

    Returns
    -------
    VideoTransform
        Transform pipeline for training.
    """
    spatial_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            crop_size,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.333),
            antialias=True,
        ),
        # NO HORIZONTAL FLIP - SSv2 has directional actions
    ])

    return VideoTransform(spatial_transform, normalize=True)


def get_val_transforms(crop_size: int = 224) -> VideoTransform:
    """
    Get validation/test transforms for SSv2.

    Uses center crop for deterministic evaluation.

    Parameters
    ----------
    crop_size : int
        Output spatial size (default 224 for ResNet).

    Returns
    -------
    VideoTransform
        Transform pipeline for validation.
    """
    spatial_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(crop_size),
    ])

    return VideoTransform(spatial_transform, normalize=True)


class GroupRandomCrop:
    """
    Random crop that applies the same crop to all frames.

    This ensures temporal consistency in the crop location.

    Parameters
    ----------
    size : int or Tuple[int, int]
        Desired output size.
    """

    def __init__(self, size: int) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply same random crop to all frames.

        Parameters
        ----------
        frames : np.ndarray
            Video frames of shape (T, H, W, C).

        Returns
        -------
        np.ndarray
            Cropped frames of shape (T, crop_h, crop_w, C).
        """
        T, H, W, C = frames.shape
        crop_h, crop_w = self.size

        # Random crop position (same for all frames)
        top = np.random.randint(0, H - crop_h + 1) if H > crop_h else 0
        left = np.random.randint(0, W - crop_w + 1) if W > crop_w else 0

        return frames[:, top:top + crop_h, left:left + crop_w, :]
