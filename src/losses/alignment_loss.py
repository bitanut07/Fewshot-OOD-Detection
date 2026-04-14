# -*- coding: utf-8 -*-
"""Global and local alignment losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAlignmentLoss(nn.Module):
    """
    Global alignment loss: CLIP-style contrastive loss between global image and text features.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_global: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global alignment loss (InfoNCE-style).

        Args:
            image_global: Global image features [B, D].
            text_embeddings: Text embeddings [B, num_classes, D] or [num_classes, D].
            labels: Ground truth labels [B].

        Returns:
            Loss scalar.
        """
        image_norm = F.normalize(image_global, p=2, dim=-1)

        if text_embeddings.dim() == 3:
            text_norm = F.normalize(text_embeddings.mean(dim=1), p=2, dim=-1)
        else:
            text_norm = F.normalize(text_embeddings, p=2, dim=-1)

        logits = torch.matmul(image_norm, text_norm.T) / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss


class LocalAlignmentLoss(nn.Module):
    """
    Local alignment loss: align local patch features with class text embeddings.
    """

    def __init__(self, temperature: float = 0.07, aggregation: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.aggregation = aggregation

    def forward(
        self,
        local_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local alignment loss.

        Args:
            local_features: Local patch features [B, num_patches, D].
            text_embeddings: Text embeddings [B, num_classes, D].
            labels: Ground truth labels [B].

        Returns:
            Loss scalar.
        """
        B, P, D = local_features.shape
        local_norm = F.normalize(local_features, p=2, dim=-1)
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)

        sim = torch.einsum("bpd,bcd->bpc", local_norm, text_norm) / self.temperature

        if self.aggregation == "mean":
            sim = sim.mean(dim=1)
        elif self.aggregation == "max":
            sim = sim.max(dim=1).values

        loss = F.cross_entropy(sim, labels)
        return loss
