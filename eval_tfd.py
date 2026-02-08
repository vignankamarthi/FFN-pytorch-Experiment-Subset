#!/usr/bin/env python3
"""Unified TFD Evaluation Script.

Evaluates a checkpoint at multiple frame counts (4F, 8F, 16F) to demonstrate
Temporal Frequency Deviation (TFD) for vanilla TSM or TFD recovery for FFN.

Usage:
    # Evaluate vanilla TSM checkpoint (TFD collapse)
    python eval_tfd.py --model_type tsm \
                       --checkpoint checkpoints/tsm/best.pth \
                       --video_dir database/data/20bn-something-something-v2 \
                       --labels_dir database/labels

    # Evaluate FFN checkpoint (TFD recovery)
    python eval_tfd.py --model_type ffn \
                       --checkpoint checkpoints/ffn/best.pth \
                       --video_dir database/data/20bn-something-something-v2 \
                       --labels_dir database/labels

    # Quick test (2 batches, verifies script works)
    python eval_tfd.py --model_type tsm \
                       --checkpoint checkpoints/tsm/best.pth \
                       --video_dir database/data/20bn-something-something-v2 \
                       --labels_dir database/labels \
                       --test_run

Reference: Frame Flexible Network (CVPR 2023)
Paper: https://arxiv.org/abs/2303.14817
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import SSv2Dataset, SSv2MultiFrameDataset
from data.transforms import get_val_transforms
from models import TSMResNet50, create_ffn_model
from training import get_device, AverageMeter, accuracy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TFD at multiple frame counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["tsm", "ffn"],
        help="Model type: 'tsm' for vanilla TSM, 'ffn' for FFN",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth)",
    )
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
        help="Path to labels directory (train.json, validation.json, labels.json)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=174,
        help="Number of classes (174 for SSv2)",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Quick test with 2 batches to verify script works",
    )

    return parser.parse_args()


def evaluate_tsm_tfd(
    checkpoint_path: str,
    video_dir: str,
    labels_dir: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    test_run: bool = False,
) -> dict:
    """
    Evaluate vanilla TSM checkpoint at 4F, 8F, 16F to demonstrate TFD.

    Loads the same checkpoint into models created with different num_frames.
    The conv/FC weights are frame-count-agnostic, but BatchNorm running
    statistics were computed at 16F during training. Evaluating at 4F/8F
    with these mismatched BN stats causes the TFD accuracy collapse.

    Parameters
    ----------
    checkpoint_path : str
        Path to vanilla TSM checkpoint.
    video_dir : str
        Path to SSv2 video directory.
    labels_dir : str
        Path to labels directory.
    num_classes : int
        Number of output classes.
    batch_size : int
        Evaluation batch size.
    num_workers : int
        DataLoader workers.
    device : torch.device
        Evaluation device.
    test_run : bool
        If True, evaluate on only 2 batches.

    Returns
    -------
    dict
        Keys: acc_4f, acc_8f, acc_16f, tfd_gap, top5_4f, top5_8f, top5_16f
    """
    labels_path = Path(labels_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint best_acc: {checkpoint.get('best_acc', 'unknown')}")

    results = {}

    for num_frames in [4, 8, 16]:
        print(f"\n  Evaluating at {num_frames}F...")

        # Create dataset with this frame count
        val_dataset = SSv2Dataset(
            video_dir=video_dir,
            annotation_file=str(labels_path / "validation.json"),
            label_file=str(labels_path / "labels.json"),
            num_frames=num_frames,
            transform=get_val_transforms(),
        )

        if test_run:
            val_dataset = Subset(val_dataset, range(min(batch_size * 2, len(val_dataset))))
            workers = 0
        else:
            workers = num_workers

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

        # Create model with this frame count and load checkpoint weights
        model = TSMResNet50(
            num_classes=num_classes,
            num_frames=num_frames,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        top1 = AverageMeter(f"Acc@1 ({num_frames}F)")
        top5 = AverageMeter(f"Acc@5 ({num_frames}F)")
        num_batches = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1.item(), videos.size(0))
                top5.update(acc5.item(), videos.size(0))

                num_batches += 1
                if test_run and num_batches >= 2:
                    break

        results[f"acc_{num_frames}f"] = top1.avg
        results[f"top5_{num_frames}f"] = top5.avg
        print(f"    Acc@1: {top1.avg:.2f}%  |  Acc@5: {top5.avg:.2f}%")

    results["tfd_gap"] = results["acc_16f"] - results["acc_4f"]
    return results


def evaluate_ffn_tfd(
    checkpoint_path: str,
    video_dir: str,
    labels_dir: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    test_run: bool = False,
) -> dict:
    """
    Evaluate FFN checkpoint at 4F, 8F, 16F to demonstrate TFD recovery.

    FFN uses specialized BatchNorm per frame count, so each frame count
    gets properly calibrated BN statistics. This eliminates the TFD collapse.

    Parameters
    ----------
    checkpoint_path : str
        Path to FFN checkpoint.
    video_dir : str
        Path to SSv2 video directory.
    labels_dir : str
        Path to labels directory.
    num_classes : int
        Number of output classes.
    batch_size : int
        Evaluation batch size.
    num_workers : int
        DataLoader workers.
    device : torch.device
        Evaluation device.
    test_run : bool
        If True, evaluate on only 2 batches.

    Returns
    -------
    dict
        Keys: acc_4f, acc_8f, acc_16f, tfd_gap, top5_4f, top5_8f, top5_16f
    """
    labels_path = Path(labels_dir)

    # Create multi-frame validation dataset
    val_dataset = SSv2MultiFrameDataset(
        video_dir=video_dir,
        annotation_file=str(labels_path / "validation.json"),
        label_file=str(labels_path / "labels.json"),
        is_train=False,
    )

    if test_run:
        val_dataset = Subset(val_dataset, range(min(batch_size * 2, len(val_dataset))))
        num_workers = 0

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create FFN model and load checkpoint
    model = create_ffn_model(
        num_classes=num_classes,
        pretrained=False,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint best_acc: {checkpoint.get('best_acc', 'unknown')}")

    # Use existing evaluate_tfd for FFN (handles x_4/x_8/x_16 interface)
    # But we need per-frame top5 too, so do it manually
    model.eval()
    results = {}

    for frame_count in [4, 8, 16]:
        print(f"\n  Evaluating at {frame_count}F...")

        top1 = AverageMeter(f"Acc@1 ({frame_count}F)")
        top5 = AverageMeter(f"Acc@5 ({frame_count}F)")
        num_batches = 0

        with torch.no_grad():
            for v_4, v_8, v_16, labels in val_loader:
                labels = labels.to(device)

                if frame_count == 4:
                    v = v_4.to(device)
                    outputs = model(x_4=v, training=False)
                elif frame_count == 8:
                    v = v_8.to(device)
                    outputs = model(x_8=v, training=False)
                else:
                    v = v_16.to(device)
                    outputs = model(x_16=v, training=False)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1.item(), v.size(0))
                top5.update(acc5.item(), v.size(0))

                num_batches += 1
                if test_run and num_batches >= 2:
                    break

        results[f"acc_{frame_count}f"] = top1.avg
        results[f"top5_{frame_count}f"] = top5.avg
        print(f"    Acc@1: {top1.avg:.2f}%  |  Acc@5: {top5.avg:.2f}%")

    results["tfd_gap"] = results["acc_16f"] - results["acc_4f"]
    return results


def print_results_table(model_type: str, results: dict) -> None:
    """
    Print formatted results table.

    Parameters
    ----------
    model_type : str
        'tsm' or 'ffn'.
    results : dict
        Evaluation results with acc and tfd_gap keys.
    """
    label = "Vanilla TSM (TFD Collapse)" if model_type == "tsm" else "FFN (TFD Recovery)"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Frames':<10} {'Acc@1':>10} {'Acc@5':>10}")
    print(f"  {'-' * 30}")
    print(f"  {'4F':<10} {results['acc_4f']:>9.2f}% {results['top5_4f']:>9.2f}%")
    print(f"  {'8F':<10} {results['acc_8f']:>9.2f}% {results['top5_8f']:>9.2f}%")
    print(f"  {'16F':<10} {results['acc_16f']:>9.2f}% {results['top5_16f']:>9.2f}%")
    print(f"  {'-' * 30}")
    print(f"  TFD Gap (16F - 4F): {results['tfd_gap']:.2f} points")
    print(f"{'=' * 60}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Device selection
    device = get_device()

    print("=" * 60)
    print("TFD Evaluation")
    print("=" * 60)
    print(f"  Model type:  {args.model_type}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {device}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Test run:    {args.test_run}")
    print("=" * 60)

    if args.model_type == "tsm":
        results = evaluate_tsm_tfd(
            checkpoint_path=args.checkpoint,
            video_dir=args.video_dir,
            labels_dir=args.labels_dir,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            test_run=args.test_run,
        )
    else:
        results = evaluate_ffn_tfd(
            checkpoint_path=args.checkpoint,
            video_dir=args.video_dir,
            labels_dir=args.labels_dir,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            test_run=args.test_run,
        )

    print_results_table(args.model_type, results)


if __name__ == "__main__":
    main()
