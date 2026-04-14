# -*- coding: utf-8 -*-
"""Classification loss (cross-entropy) for few-shot query set."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss for few-shot classification on query set.

    Args:
        label_smoothing: Label smoothing factor (0 = no smoothing).
        reduction: Reduction method ("mean", "sum", "none").
    """

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions [B, num_classes].
            labels: Ground truth labels [B].
            weight: Optional per-class weights [num_classes].

        Returns:
            Loss scalar.
        """
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
            weight=weight,
            reduction=self.reduction,
        )
        return loss
