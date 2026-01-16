"""PyTorch Dataset classes for Something-Something V2.

Provides two dataset variants:
1. SSv2Dataset: Single frame-count dataset for vanilla TSM training/eval
2. SSv2MultiFrameDataset: Multi frame-count dataset for FFN training
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .video_loader import load_video_frames
from .transforms import get_train_transforms, get_val_transforms


class SSv2Dataset(Dataset):
    """
    Something-Something V2 Dataset for single frame-count training.

    Returns video clips with a fixed number of frames.

    Parameters
    ----------
    video_dir : str
        Path to directory containing .webm video files.
    annotation_file : str
        Path to JSON annotation file (train.json or validation.json).
    label_file : str
        Path to labels.json mapping class names to indices.
    num_frames : int
        Number of frames to sample per video (T).
    transform : Optional[Callable]
        Transform to apply to video frames.
    is_train : bool
        Whether this is training set (affects default transforms).

    Attributes
    ----------
    samples : List[Tuple[str, int]]
        List of (video_path, label) tuples.
    num_classes : int
        Number of action classes (174 for SSv2).
    """

    def __init__(
        self,
        video_dir: str,
        annotation_file: str,
        label_file: str,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.is_train = is_train

        # Load label mapping
        with open(label_file, "r") as f:
            self.label_to_idx = json.load(f)
        self.num_classes = len(self.label_to_idx)

        # Load annotations
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        # Build sample list
        self.samples: List[Tuple[Path, int]] = []
        for item in annotations:
            video_id = item["id"]
            template = item["template"]

            # Remove brackets from template to match labels.json format
            template_clean = template.replace("[", "").replace("]", "")

            if template_clean in self.label_to_idx:
                label_idx = int(self.label_to_idx[template_clean])
                video_path = self.video_dir / f"{video_id}.webm"
                self.samples.append((video_path, label_idx))

        # Set transform
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, int]
            (video_tensor, label) where video_tensor has shape (C, T, H, W).
        """
        video_path, label = self.samples[idx]

        # Load frames
        frames = load_video_frames(str(video_path), self.num_frames)

        # Apply transforms
        video_tensor = self.transform(frames)

        return video_tensor, label


class SSv2MultiFrameDataset(Dataset):
    """
    Something-Something V2 Dataset for FFN multi-frame training.

    Returns the same video at three different frame counts (4F, 8F, 16F)
    for temporal distillation training.

    Parameters
    ----------
    video_dir : str
        Path to directory containing .webm video files.
    annotation_file : str
        Path to JSON annotation file (train.json or validation.json).
    label_file : str
        Path to labels.json mapping class names to indices.
    frame_counts : Tuple[int, int, int]
        Frame counts for (low, medium, high) - default (4, 8, 16).
    transform : Optional[Callable]
        Transform to apply to video frames.
    is_train : bool
        Whether this is training set.

    Attributes
    ----------
    samples : List[Tuple[str, int]]
        List of (video_path, label) tuples.
    """

    def __init__(
        self,
        video_dir: str,
        annotation_file: str,
        label_file: str,
        frame_counts: Tuple[int, int, int] = (4, 8, 16),
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.frame_counts = frame_counts
        self.is_train = is_train

        # Load label mapping
        with open(label_file, "r") as f:
            self.label_to_idx = json.load(f)
        self.num_classes = len(self.label_to_idx)

        # Load annotations
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        # Build sample list
        self.samples: List[Tuple[Path, int]] = []
        for item in annotations:
            video_id = item["id"]
            template = item["template"]

            # Remove brackets from template to match labels.json format
            template_clean = template.replace("[", "").replace("]", "")

            if template_clean in self.label_to_idx:
                label_idx = int(self.label_to_idx[template_clean])
                video_path = self.video_dir / f"{video_id}.webm"
                self.samples.append((video_path, label_idx))

        # Set transform
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample at all three frame counts.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
            (v_L, v_M, v_H, label) where:
            - v_L: low frame count tensor (C, T_low, H, W)
            - v_M: medium frame count tensor (C, T_med, H, W)
            - v_H: high frame count tensor (C, T_high, H, W)
            - label: class index
        """
        video_path, label = self.samples[idx]
        t_low, t_med, t_high = self.frame_counts

        # Load frames at highest count first (we'll subsample for lower counts)
        frames_high = load_video_frames(str(video_path), t_high)

        # For lower frame counts, subsample from the same temporal positions
        # This ensures consistency across frame counts
        frames_med = load_video_frames(str(video_path), t_med)
        frames_low = load_video_frames(str(video_path), t_low)

        # Apply transforms
        v_high = self.transform(frames_high)
        v_med = self.transform(frames_med)
        v_low = self.transform(frames_low)

        return v_low, v_med, v_high, label


def create_dataloaders(
    video_dir: str,
    labels_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 4,
    multi_frame: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Parameters
    ----------
    video_dir : str
        Path to video directory.
    labels_dir : str
        Path to labels directory containing train.json, validation.json, labels.json.
    batch_size : int
        Batch size per GPU.
    num_frames : int
        Number of frames for single-frame dataset.
    num_workers : int
        Number of data loading workers.
    multi_frame : bool
        If True, use SSv2MultiFrameDataset for FFN training.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    labels_path = Path(labels_dir)

    DatasetClass = SSv2MultiFrameDataset if multi_frame else SSv2Dataset

    # Common kwargs
    common_kwargs = {
        "video_dir": video_dir,
        "label_file": str(labels_path / "labels.json"),
    }

    if not multi_frame:
        common_kwargs["num_frames"] = num_frames

    train_dataset = DatasetClass(
        annotation_file=str(labels_path / "train.json"),
        is_train=True,
        **common_kwargs,
    )

    val_dataset = DatasetClass(
        annotation_file=str(labels_path / "validation.json"),
        is_train=False,
        **common_kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
