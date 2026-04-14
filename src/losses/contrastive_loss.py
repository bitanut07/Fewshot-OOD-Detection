# -*- coding: utf-8 -*-
"""Local contrastive loss for disease-relevant region alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalContrastiveLoss(nn.Module):
    """
    Supervised local contrastive loss.

    Pulls disease-relevant (top-k) patches closer to class prototypes.
    Pushes disease-irrelevant (bottom-k) patches away from prototypes.
    """

    def __init__(self, temperature: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        relevant_features: torch.Tensor,
        irrelevant_features: torch.Tensor,
        class_prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised local contrastive loss.

        Args:
            relevant_features: Disease-relevant patches [B, top_k, D].
            irrelevant_features: Disease-irrelevant patches [B, bottom_k, D].
            class_prototypes: Class prototypes [num_classes, D].
            labels: Class labels [B].

        Returns:
            Loss scalar.
        """
        B, K, D = relevant_features.shape
        rel_flat = F.normalize(relevant_features, p=2, dim=-1).view(-1, D)
        irr_flat = F.normalize(irrelevant_features, p=2, dim=-1).view(-1, D)
        protos = F.normalize(class_prototypes, p=2, dim=-1)

        pos_sim = torch.matmul(rel_flat, protos.T) / self.temperature
        neg_sim = torch.matmul(rel_flat, irr_flat.T) / self.temperature

        logits = torch.cat([pos_sim.mean(dim=1, keepdim=True),
                            neg_sim.mean(dim=1, keepdim=True)], dim=1)
        pos_labels = torch.zeros(B * K, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, pos_labels, reduction=self.reduction)
        return loss
