#!/usr/bin/env python3
"""FFN Training Script.

Train Frame Flexible Network (FFN) with temporal distillation on SSv2.

Usage:
    python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                        --labels_dir database/labels \
                        --epochs 50 \
                        --batch_size 8

For local testing (2 batches):
    python train_ffn.py --video_dir database/data/20bn-something-something-v2 \
                        --labels_dir database/labels \
                        --test_run

Reference: Frame Flexible Network (CVPR 2023)
Paper: https://arxiv.org/abs/2303.14817
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from models import TSMFFN, TemporalDistillationLoss, create_ffn_model
from data.dataset import SSv2MultiFrameDataset
from training import (
    FFNTrainer,
    create_ffn_optimizer,
    create_ffn_scheduler,
    get_device,
    evaluate_tfd,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FFN on Something-Something V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to SSv2 video directory",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Path to labels directory (contains train.json, validation.json, labels.json)",
    )

    # Frame counts
    parser.add_argument(
        "--num_frames_h",
        type=int,
        default=16,
        help="High frame count (teacher)",
    )
    parser.add_argument(
        "--num_frames_m",
        type=int,
        default=8,
        help="Medium frame count",
    )
    parser.add_argument(
        "--num_frames_l",
        type=int,
        default=4,
        help="Low frame count",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 penalty)",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=1.0,
        help="Weight for KL divergence loss",
    )

    # Model
    parser.add_argument(
        "--num_classes",
        type=int,
        default=174,
        help="Number of classes (174 for SSv2)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Do not use pretrained weights",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/ffn",
        help="Directory to save checkpoints",
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
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    # Workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Testing/Debug
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run with 2 batches for testing",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate, no training",
    )

    args = parser.parse_args()

    # Handle pretrained flag
    if args.no_pretrained:
        args.pretrained = False

    return args


def create_dataloaders(
    video_dir: str,
    labels_dir: str,
    batch_size: int,
    num_workers: int,
    test_run: bool = False,
) -> tuple:
    """
    Create train and validation dataloaders for FFN training.

    Parameters
    ----------
    video_dir : str
        Path to video directory.
    labels_dir : str
        Path to labels directory.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    test_run : bool
        If True, use tiny subset for testing.

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    labels_path = Path(labels_dir)

    # Create datasets
    train_dataset = SSv2MultiFrameDataset(
        video_dir=video_dir,
        annotation_file=str(labels_path / "train.json"),
        label_file=str(labels_path / "labels.json"),
        is_train=True,
    )

    val_dataset = SSv2MultiFrameDataset(
        video_dir=video_dir,
        annotation_file=str(labels_path / "validation.json"),
        label_file=str(labels_path / "labels.json"),
        is_train=False,
    )

    # For test runs, use tiny subsets
    if test_run:
        train_dataset = Subset(train_dataset, range(min(16, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(8, len(val_dataset))))
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()

    # Device selection
    device = get_device()
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        video_dir=args.video_dir,
        labels_dir=args.labels_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_run=args.test_run,
    )

    # Create model
    print("Creating FFN model...")
    model = create_ffn_model(
        num_classes=args.num_classes,
        num_frames_h=args.num_frames_h,
        num_frames_m=args.num_frames_m,
        num_frames_l=args.num_frames_l,
        pretrained=args.pretrained,
        device=device,
    )

    # Create loss function
    criterion = TemporalDistillationLoss(lambda_kl=args.lambda_kl)

    # Eval only mode
    if args.eval_only:
        if args.resume is None:
            print("Error: --resume required for --eval_only")
            sys.exit(1)

        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("\nEvaluating TFD...")
        tfd_results = evaluate_tfd(model, val_loader, device)
        print(f"  4F Accuracy:  {tfd_results['acc_4f']:.2f}%")
        print(f"  8F Accuracy:  {tfd_results['acc_8f']:.2f}%")
        print(f"  16F Accuracy: {tfd_results['acc_16f']:.2f}%")
        print(f"  TFD Gap (16F - 4F): {tfd_results['tfd_gap']:.2f} points")
        return

    # Create optimizer and scheduler
    optimizer = create_ffn_optimizer(
        model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = create_ffn_scheduler(optimizer, epochs=args.epochs)

    # Create trainer
    trainer = FFNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        lambda_kl=args.lambda_kl,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.resume(args.resume)
        print(f"Resuming from epoch {start_epoch}")

    # Train
    print("\nStarting FFN training...")
    trainer.train(
        epochs=args.epochs,
        start_epoch=start_epoch,
        validate_every=args.validate_every,
        save_every=args.save_every,
    )

    # Final TFD evaluation
    print("\n" + "=" * 60)
    print("Final TFD Evaluation")
    print("=" * 60)
    tfd_results = evaluate_tfd(model, val_loader, device)
    print(f"  4F Accuracy:  {tfd_results['acc_4f']:.2f}%")
    print(f"  8F Accuracy:  {tfd_results['acc_8f']:.2f}%")
    print(f"  16F Accuracy: {tfd_results['acc_16f']:.2f}%")
    print(f"  TFD Gap (16F - 4F): {tfd_results['tfd_gap']:.2f} points")


if __name__ == "__main__":
    main()
