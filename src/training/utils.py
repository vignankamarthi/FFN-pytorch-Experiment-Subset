"""Training utilities for FFN reproduction.

Contains device selection, metric tracking, and accuracy computation.
"""

from typing import Tuple

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Get the best available device.

    Priority: CUDA > MPS > CPU

    Parameters
    ----------
    verbose : bool, optional
        If True, print detected device info (default: True).

    Returns
    -------
    torch.device
        Best available device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print("=" * 60)
            print("DETECTED: CUDA (NVIDIA GPU)")
            print(f"  GPU: {gpu_name}")
            print(f"  Available GPUs: {gpu_count}")
            print("=" * 60)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("=" * 60)
            print("DETECTED: MPS (Apple Silicon GPU)")
            print("  Running on Mac Metal Performance Shaders")
            print("=" * 60)
    else:
        device = torch.device("cpu")
        if verbose:
            print("=" * 60)
            print("DETECTED: CPU (No GPU available)")
            print("  Warning: Training will be slow!")
            print("=" * 60)
    return device


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking loss and accuracy during training.

    Parameters
    ----------
    name : str
        Name of the metric being tracked.

    Attributes
    ----------
    val : float
        Most recent value.
    avg : float
        Running average.
    sum : float
        Running sum.
    count : int
        Number of updates.

    Examples
    --------
    >>> meter = AverageMeter("loss")
    >>> meter.update(0.5)
    >>> meter.update(0.3)
    >>> meter.avg
    0.4
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.

        Parameters
        ----------
        val : float
            New value to add.
        n : int, optional
            Number of samples this value represents (for weighted avg).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list:
    """
    Compute top-k accuracy for the specified values of k.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions of shape (B, num_classes).
    target : torch.Tensor
        Ground truth labels of shape (B,).
    topk : tuple of int, optional
        Which top-k accuracies to compute. Default (1,) for top-1 only.

    Returns
    -------
    list of torch.Tensor
        Top-k accuracies as percentages (0-100).

    Examples
    --------
    >>> output = torch.randn(32, 174)
    >>> target = torch.randint(0, 174, (32,))
    >>> top1, top5 = accuracy(output, target, topk=(1, 5))
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, B)

        # Compare with target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
