"""Test script for Phase 3: Data Loading.

Run with: python -m tests.test_data_loading (from project root)
      or: pytest tests/test_data_loading.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_video_frames,
    sample_frame_indices,
    get_train_transforms,
    get_val_transforms,
    SSv2Dataset,
    SSv2MultiFrameDataset,
)


def test_frame_sampling():
    """Test uniform frame sampling."""
    print("\n=== Test: Frame Sampling ===")

    # Test normal case
    indices = sample_frame_indices(48, 8)
    print(f"48 frames, sample 8: {indices}")
    assert len(indices) == 8, "Should return 8 indices"

    # Test video shorter than segments
    indices = sample_frame_indices(4, 8)
    print(f"4 frames, sample 8: {indices}")
    assert len(indices) == 8, "Should still return 8 indices (with repeats)"

    print("Frame sampling: PASSED")


def test_video_loading():
    """Test video loading with PyAV."""
    print("\n=== Test: Video Loading ===")

    video_dir = Path("database/data/20bn-something-something-v2")
    video_path = video_dir / "1.webm"

    if not video_path.exists():
        print(f"SKIP: Video not found at {video_path}")
        return

    # Load 16 frames
    frames = load_video_frames(str(video_path), num_frames=16)
    print(f"Loaded frames shape: {frames.shape}")
    assert frames.shape[0] == 16, "Should have 16 frames"
    assert frames.shape[3] == 3, "Should have 3 color channels (RGB)"

    # Load different frame counts
    for T in [4, 8, 16]:
        frames = load_video_frames(str(video_path), num_frames=T)
        print(f"  T={T}: shape {frames.shape}")
        assert frames.shape[0] == T

    print("Video loading: PASSED")


def test_transforms():
    """Test video transforms."""
    print("\n=== Test: Transforms ===")

    import numpy as np

    # Create dummy video (T, H, W, C)
    dummy_video = np.random.randint(0, 255, (8, 240, 320, 3), dtype=np.uint8)

    # Test train transforms
    train_transform = get_train_transforms(crop_size=224)
    output = train_transform(dummy_video)
    print(f"Train transform: {dummy_video.shape} -> {output.shape}")
    assert output.shape == (3, 8, 224, 224), f"Expected (3, 8, 224, 224), got {output.shape}"

    # Test val transforms
    val_transform = get_val_transforms(crop_size=224)
    output = val_transform(dummy_video)
    print(f"Val transform: {dummy_video.shape} -> {output.shape}")
    assert output.shape == (3, 8, 224, 224), f"Expected (3, 8, 224, 224), got {output.shape}"

    # Check normalization (values should be roughly in [-2, 2] range)
    print(f"Output value range: [{output.min():.2f}, {output.max():.2f}]")

    print("Transforms: PASSED")


def test_single_frame_dataset():
    """Test SSv2Dataset."""
    print("\n=== Test: SSv2Dataset ===")

    video_dir = "database/data/20bn-something-something-v2"
    labels_dir = "database/labels"

    if not Path(video_dir).exists():
        print(f"SKIP: Video dir not found at {video_dir}")
        return

    dataset = SSv2Dataset(
        video_dir=video_dir,
        annotation_file=f"{labels_dir}/train.json",
        label_file=f"{labels_dir}/labels.json",
        num_frames=16,
        is_train=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")

    # Load one sample
    video, label = dataset[0]
    print(f"Sample shape: {video.shape}, label: {label}")
    assert video.shape == (3, 16, 224, 224), f"Expected (3, 16, 224, 224), got {video.shape}"
    assert 0 <= label < 174, f"Label {label} out of range"

    print("SSv2Dataset: PASSED")


def test_multi_frame_dataset():
    """Test SSv2MultiFrameDataset."""
    print("\n=== Test: SSv2MultiFrameDataset ===")

    video_dir = "database/data/20bn-something-something-v2"
    labels_dir = "database/labels"

    if not Path(video_dir).exists():
        print(f"SKIP: Video dir not found at {video_dir}")
        return

    dataset = SSv2MultiFrameDataset(
        video_dir=video_dir,
        annotation_file=f"{labels_dir}/train.json",
        label_file=f"{labels_dir}/labels.json",
        frame_counts=(4, 8, 16),
        is_train=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Load one sample
    v_low, v_med, v_high, label = dataset[0]
    print(f"v_L (4F) shape: {v_low.shape}")
    print(f"v_M (8F) shape: {v_med.shape}")
    print(f"v_H (16F) shape: {v_high.shape}")
    print(f"Label: {label}")

    assert v_low.shape == (3, 4, 224, 224), f"Expected (3, 4, 224, 224), got {v_low.shape}"
    assert v_med.shape == (3, 8, 224, 224), f"Expected (3, 8, 224, 224), got {v_med.shape}"
    assert v_high.shape == (3, 16, 224, 224), f"Expected (3, 16, 224, 224), got {v_high.shape}"

    print("SSv2MultiFrameDataset: PASSED")


def test_dataloader():
    """Test DataLoader with batching."""
    print("\n=== Test: DataLoader ===")

    import torch

    video_dir = "database/data/20bn-something-something-v2"
    labels_dir = "database/labels"

    if not Path(video_dir).exists():
        print(f"SKIP: Video dir not found at {video_dir}")
        return

    dataset = SSv2Dataset(
        video_dir=video_dir,
        annotation_file=f"{labels_dir}/train.json",
        label_file=f"{labels_dir}/labels.json",
        num_frames=16,
        is_train=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
    )

    # Get one batch
    batch = next(iter(loader))
    videos, labels = batch

    print(f"Batch videos shape: {videos.shape}")
    print(f"Batch labels shape: {labels.shape}")

    assert videos.shape == (2, 3, 16, 224, 224), f"Expected (2, 3, 16, 224, 224), got {videos.shape}"

    print("DataLoader: PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 3: Data Loading Tests")
    print("=" * 50)

    test_frame_sampling()
    test_video_loading()
    test_transforms()
    test_single_frame_dataset()
    test_multi_frame_dataset()
    test_dataloader()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
