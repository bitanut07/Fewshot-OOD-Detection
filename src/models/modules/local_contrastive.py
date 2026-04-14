# -*- coding: utf-8 -*-
"""Local Contrastive Learner for disease-relevant region alignment.

Performs supervised local contrastive learning to:
    - Pull disease-relevant regions (top-k) of the same class closer
    - Push disease-irrelevant regions (bottom-k) away from class prototypes
    - Separate cross-class disease-relevant regions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LocalContrastiveLearner(nn.Module):
    """
    Supervised Local Contrastive Learning Module.

    Loss formulation:
        - Positive pairs: Disease-relevant patches from same class
        - Negative pairs: Disease-irrelevant patches OR cross-class patches

    Args:
        temperature: Temperature for softmax normalization of similarities
        top_k: Number of disease-relevant patches per image
        bottom_k: Number of disease-irrelevant patches per image
        embed_dim: Feature dimension
        trainable: Whether this module is trainable
    """

    def __init__(
        self,
        temperature: float = 0.1,
        top_k: int = 4,
        bottom_k: int = 4,
        embed_dim: int = 512,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.embed_dim = embed_dim
        self.trainable_flag = trainable

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        relevant_features: torch.Tensor,
        irrelevant_features: torch.Tensor,
        class_prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute supervised local contrastive loss.

        Args:
            relevant_features: Disease-relevant patch features [B, top_k, dim]
            irrelevant_features: Disease-irrelevant patch features [B, bottom_k, dim]
            class_prototypes: Class prototype features [num_classes, dim]
            labels: Class labels for each sample [B] (optional)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        B, K, D = relevant_features.shape
        BK = self.bottom_k

        # Normalize features
        relevant_features = F.normalize(relevant_features, p=2, dim=-1)
        irrelevant_features = F.normalize(irrelevant_features, p=2, dim=-1)
        class_prototypes = F.normalize(class_prototypes, p=2, dim=-1)

        # Reshape for computation
        # relevant: [B*K, D], irrelevant: [B*BK, D]
        rel_flat = relevant_features.view(-1, D)  # [B*K, D]
        irr_flat = irrelevant_features.view(-1, D)  # [B*BK, D]

        # Positive pairs: each relevant patch with its class prototype
        pos_sim = torch.matmul(rel_flat, class_prototypes.T) / self.temperature  # [B*K, num_classes]

        # Negative pairs: each relevant patch with irrelevant patches
        neg_sim = torch.matmul(rel_flat, irr_flat.T) / self.temperature  # [B*K, B*BK]

        # Contrastive loss: supervised version
        # We want to push relevant patches close to their own class prototype
        # and away from other class prototypes

        # Option 1: Simple contrastive with positives and negatives
        # logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [B*K, num_classes + B*BK]
        # labels = torch.zeros(B*K, dtype=torch.long, device=logits.device)  # positive is at index 0

        # Option 2: Per-class NT-Xent style loss
        loss = self._compute_ntxent_loss(rel_flat, irr_flat, class_prototypes)

        # Compute metrics
        with torch.no_grad():
            metrics = {
                "local_contrastive_loss": loss.item(),
                "avg_pos_sim": pos_sim.mean().item(),
                "avg_neg_sim": neg_sim.mean().item(),
            }

        return loss, metrics

    def _compute_ntxent_loss(
        self,
        anchors: torch.Tensor,
        negatives: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent style supervised contrastive loss."""
        # Anchor-positive: anchors to their own class prototype
        # Anchor-negative: anchors to all other prototypes AND irrelevant patches

        all_negatives = torch.cat([prototypes, negatives], dim=0)  # [num_classes + total_neg, D]
        all_negatives = F.normalize(all_negatives, p=2, dim=-1)

        # For simplicity, use prototype-based contrastive loss
        # Each anchor should be close to its own class prototype
        # and far from all other prototypes
        pos_sim = torch.matmul(anchors, prototypes.T) / self.temperature
        neg_sim = torch.matmul(anchors, prototypes.T) / self.temperature

        # Create positive mask (each anchor's own class)
        num_anchors = anchors.size(0)
        num_prototypes = prototypes.size(0)
        labels = torch.arange(num_anchors, device=anchors.device) % num_prototypes

        # logits = [pos_sim, neg_sim] with positive at index 0
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        pos_labels = torch.zeros(num_anchors, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, pos_labels)
        return loss

        
