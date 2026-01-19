"""Temporal Distillation Loss for FFN training.

Implements the knowledge distillation loss from FFN paper (Equation 4-5):
- High frame count (16F) acts as teacher
- Medium (8F) and low (4F) frame counts are students
- KL divergence transfers knowledge from teacher to students

Reference: Frame Flexible Network (CVPR 2023)
Paper: https://arxiv.org/abs/2303.14817
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDistillationLoss(nn.Module):
    """
    Combined loss for FFN training with temporal distillation.

    Total loss (Equation 5):
        L = L_CE(p_H, y) + lambda * (L_KL(p_L || p_H) + L_KL(p_M || p_H))

    The high-frame prediction (p_H) acts as the teacher, distilling
    knowledge to medium (p_M) and low (p_L) frame predictions.

    CRITICAL: p_H is detached before computing KL divergence to prevent
    gradients flowing through the teacher. This is essential - without
    detach, the high-frame branch would receive conflicting gradients.

    Parameters
    ----------
    lambda_kl : float
        Weight for KL divergence loss. Default 1.0 (from paper).
    temperature : float
        Softmax temperature for softer probability distributions.
        Default 1.0 (no temperature scaling).

    Attributes
    ----------
    ce_loss : nn.CrossEntropyLoss
        Cross-entropy loss for classification.
    kl_loss : nn.KLDivLoss
        KL divergence loss for distillation.

    Examples
    --------
    >>> criterion = TemporalDistillationLoss(lambda_kl=1.0)
    >>> out_4 = torch.randn(8, 174)  # logits from 4-frame
    >>> out_8 = torch.randn(8, 174)  # logits from 8-frame
    >>> out_16 = torch.randn(8, 174)  # logits from 16-frame
    >>> labels = torch.randint(0, 174, (8,))
    >>> loss, loss_dict = criterion(out_4, out_8, out_16, labels)
    """

    def __init__(
        self,
        lambda_kl: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_kl = lambda_kl
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        output_l: torch.Tensor,
        output_m: torch.Tensor,
        output_h: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FFN loss.

        Parameters
        ----------
        output_l : torch.Tensor
            Low frame count logits, shape (B, num_classes).
        output_m : torch.Tensor
            Medium frame count logits, shape (B, num_classes).
        output_h : torch.Tensor
            High frame count logits (teacher), shape (B, num_classes).
        target : torch.Tensor
            Ground truth labels, shape (B,).

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            - Total loss (scalar)
            - Dictionary with individual loss components for logging
        """
        # Cross-entropy loss on high-frame prediction only
        loss_ce = self.ce_loss(output_h, target)

        # Get soft targets from teacher (high-frame)
        # CRITICAL: detach() prevents gradients through teacher
        teacher_probs = F.softmax(output_h.detach() / self.temperature, dim=1)

        # KL divergence for low-frame student
        # KLDivLoss expects log-probabilities as input
        student_l_log_probs = F.log_softmax(output_l / self.temperature, dim=1)
        loss_kl_l = self.kl_loss(student_l_log_probs, teacher_probs)

        # KL divergence for medium-frame student
        student_m_log_probs = F.log_softmax(output_m / self.temperature, dim=1)
        loss_kl_m = self.kl_loss(student_m_log_probs, teacher_probs)

        # Scale KL loss by temperature^2 (standard in distillation)
        if self.temperature != 1.0:
            loss_kl_l = loss_kl_l * (self.temperature ** 2)
            loss_kl_m = loss_kl_m * (self.temperature ** 2)

        # Total loss (Equation 5)
        loss_kl_total = self.lambda_kl * (loss_kl_l + loss_kl_m)
        total_loss = loss_ce + loss_kl_total

        # Loss components for logging
        loss_dict = {
            "loss_ce": loss_ce.item(),
            "loss_kl_l": loss_kl_l.item(),
            "loss_kl_m": loss_kl_m.item(),
            "loss_kl_total": loss_kl_total.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_dict


class FFNLoss(nn.Module):
    """
    Alternative interface for FFN loss computation.

    This class provides the same functionality as TemporalDistillationLoss
    but with a more flexible API for different use cases.

    Parameters
    ----------
    lambda_kl : float
        Weight for KL divergence loss. Default 1.0.
    reduction : str
        Reduction for CE loss. Default 'mean'.

    Notes
    -----
    The FFN paper uses lambda=1.0 for all experiments. Increasing lambda
    gives more weight to frame-count consistency, while decreasing it
    emphasizes classification accuracy on high-frame inputs.
    """

    def __init__(
        self,
        lambda_kl: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_kl = lambda_kl
        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss from tuple of outputs.

        Parameters
        ----------
        outputs : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (output_4, output_8, output_16) logits.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            Total loss and component dictionary.
        """
        output_l, output_m, output_h = outputs

        # CE on high-frame only
        loss_ce = self.ce_criterion(output_h, target)

        # Soft targets from teacher (detached!)
        with torch.no_grad():
            teacher_probs = F.softmax(output_h, dim=1)

        # KL losses
        loss_kl_l = F.kl_div(
            F.log_softmax(output_l, dim=1),
            teacher_probs,
            reduction="batchmean",
        )
        loss_kl_m = F.kl_div(
            F.log_softmax(output_m, dim=1),
            teacher_probs,
            reduction="batchmean",
        )

        # Total
        loss_kl = self.lambda_kl * (loss_kl_l + loss_kl_m)
        total_loss = loss_ce + loss_kl

        loss_dict = {
            "loss_ce": loss_ce.item(),
            "loss_kl_4f": loss_kl_l.item(),
            "loss_kl_8f": loss_kl_m.item(),
            "loss_kl": loss_kl.item(),
            "loss": total_loss.item(),
        }

        return total_loss, loss_dict


def compute_ffn_loss(
    output_l: torch.Tensor,
    output_m: torch.Tensor,
    output_h: torch.Tensor,
    target: torch.Tensor,
    lambda_kl: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Functional interface for FFN loss computation.

    This matches the reference implementation's loss computation pattern:
    ```python
    loss = criterion(output_H, target)  # CE on high-frame
    loss_L_kl = lambda * KLDivLoss(LogSoftmax(output_L), Softmax(output_H.detach()))
    loss_M_kl = lambda * KLDivLoss(LogSoftmax(output_M), Softmax(output_H.detach()))
    total_loss = loss + loss_L_kl + loss_M_kl
    ```

    Parameters
    ----------
    output_l : torch.Tensor
        Low frame (4F) logits, shape (B, C).
    output_m : torch.Tensor
        Medium frame (8F) logits, shape (B, C).
    output_h : torch.Tensor
        High frame (16F) logits, shape (B, C).
    target : torch.Tensor
        Ground truth labels, shape (B,).
    lambda_kl : float
        KL divergence weight. Default 1.0.

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, float]]
        Total loss and loss components.

    Examples
    --------
    >>> output_l = torch.randn(8, 174)
    >>> output_m = torch.randn(8, 174)
    >>> output_h = torch.randn(8, 174)
    >>> target = torch.randint(0, 174, (8,))
    >>> loss, info = compute_ffn_loss(output_l, output_m, output_h, target)
    >>> print(f"Total: {info['loss']:.4f}, CE: {info['loss_ce']:.4f}")
    """
    # Cross-entropy on high-frame prediction
    loss_ce = F.cross_entropy(output_h, target)

    # Teacher probabilities (detached)
    teacher_probs = F.softmax(output_h.detach(), dim=1)

    # KL divergence for students
    loss_kl_l = lambda_kl * F.kl_div(
        F.log_softmax(output_l, dim=1),
        teacher_probs,
        reduction="batchmean",
    )
    loss_kl_m = lambda_kl * F.kl_div(
        F.log_softmax(output_m, dim=1),
        teacher_probs,
        reduction="batchmean",
    )

    total_loss = loss_ce + loss_kl_l + loss_kl_m

    loss_dict = {
        "loss_ce": loss_ce.item(),
        "loss_kl_4f": loss_kl_l.item(),
        "loss_kl_8f": loss_kl_m.item(),
        "loss": total_loss.item(),
    }

    return total_loss, loss_dict
