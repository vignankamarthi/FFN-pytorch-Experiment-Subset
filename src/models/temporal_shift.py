"""Temporal Shift Module (TSM) implementation.

Reference: "TSM: Temporal Shift Module for Efficient Video Understanding"
Paper: https://arxiv.org/abs/1811.08383
"""

import torch
import torch.nn as nn


def temporal_shift(x: torch.Tensor, num_frames: int, shift_div: int = 8) -> torch.Tensor:
    """
    Perform temporal shift operation on input tensor.

    Shifts 1/shift_div channels forward in time, 1/shift_div backward,
    and leaves the rest unchanged. This allows information to flow
    between adjacent frames with zero extra parameters.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B*T, C, H, W) where B is batch size
        and T is number of frames.
    num_frames : int
        Number of frames T in each video.
    shift_div : int, optional
        Fraction of channels to shift. Default 8 means 1/8 forward,
        1/8 backward, 6/8 unchanged.

    Returns
    -------
    torch.Tensor
        Shifted tensor of same shape (B*T, C, H, W).

    Examples
    --------
    >>> x = torch.randn(16, 64, 56, 56)  # B=2, T=8
    >>> out = temporal_shift(x, num_frames=8)
    >>> out.shape
    torch.Size([16, 64, 56, 56])
    """
    bt, c, h, w = x.size()
    b = bt // num_frames
    t = num_frames

    # Reshape to (B, T, C, H, W)
    x = x.view(b, t, c, h, w)

    # Calculate shift amount
    fold = c // shift_div  # Number of channels to shift

    # Create output tensor
    out = torch.zeros_like(x)

    # Shift left (forward in time): frame t gets frame t+1's channels
    # This lets current frame "see" future information
    out[:, :-1, :fold] = x[:, 1:, :fold]

    # Shift right (backward in time): frame t gets frame t-1's channels
    # This lets current frame "see" past information
    out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]

    # Keep remaining channels unchanged
    out[:, :, 2*fold:] = x[:, :, 2*fold:]

    # Reshape back to (B*T, C, H, W)
    return out.view(bt, c, h, w)


class TemporalShift(nn.Module):
    """
    Temporal Shift Module wrapper for insertion into residual blocks.

    Wraps a convolutional layer and applies temporal shift before it.
    This is the standard way to inject TSM into existing architectures.

    Parameters
    ----------
    net : nn.Module
        The module to wrap (typically conv1 of a residual block).
    num_frames : int
        Number of frames T in each video.
    shift_div : int, optional
        Fraction of channels to shift. Default 8.

    Attributes
    ----------
    net : nn.Module
        The wrapped module.
    num_frames : int
        Number of temporal frames.
    shift_div : int
        Channel division factor for shifting.
    """

    def __init__(self, net: nn.Module, num_frames: int, shift_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal shift then pass through wrapped module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B*T, C, H, W).

        Returns
        -------
        torch.Tensor
            Output from wrapped module after temporal shift.
        """
        x = temporal_shift(x, self.num_frames, self.shift_div)
        return self.net(x)
