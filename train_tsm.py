#!/usr/bin/env python
"""
Train vanilla TSM on Something-Something V2.

Usage:
    python train_tsm.py --data_dir database --epochs 50 --batch_size 8
    python train_tsm.py --resume checkpoints/epoch_10.pth

For cluster:
    python train_tsm.py --data_dir /path/to/data --epochs 50 --batch_size 8 --num_workers 4
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import SSv2Dataset, get_train_transforms, get_val_transforms
from src.models import TSMResNet50
from src.training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    get_device,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train vanilla TSM on SSv2")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="database",
        help="Path to database directory containing data/ and labels/",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample per video (default: 16)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )

    # DataLoader
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Validation
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    video_dir = data_dir / "data" / "20bn-something-something-v2"
    labels_dir = data_dir / "labels"

    # Verify paths exist
    assert video_dir.exists(), f"Video directory not found: {video_dir}"
    assert labels_dir.exists(), f"Labels directory not found: {labels_dir}"

    print("=" * 60)
    print("Vanilla TSM Training on Something-Something V2")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SSv2Dataset(
        video_dir=str(video_dir),
        annotation_file=str(labels_dir / "train.json"),
        label_file=str(labels_dir / "labels.json"),
        num_frames=args.num_frames,
        transform=get_train_transforms(),
    )

    val_dataset = SSv2Dataset(
        video_dir=str(video_dir),
        annotation_file=str(labels_dir / "validation.json"),
        label_file=str(labels_dir / "labels.json"),
        num_frames=args.num_frames,
        transform=get_val_transforms(),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    device = get_device()
    model = TSMResNet50(
        num_classes=174,
        num_frames=args.num_frames,
        pretrained=True,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(optimizer, epochs=args.epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.resume(args.resume)
        print(f"Resuming from epoch {start_epoch}")

    # Train
    print("\nStarting training...")
    trainer.train(
        epochs=args.epochs,
        start_epoch=start_epoch,
        validate_every=args.validate_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
