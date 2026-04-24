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

        relevant_features = F.normalize(relevant_features, p=2, dim=-1)
        irrelevant_features = F.normalize(irrelevant_features, p=2, dim=-1)
        class_prototypes = F.normalize(class_prototypes, p=2, dim=-1)

        rel_flat = relevant_features.view(-1, D)   # [B*K, D]
        irr_flat = irrelevant_features.view(-1, D)  # [B*BK, D]

        loss = self._compute_ntxent_loss(rel_flat, irr_flat, class_prototypes, labels, K)

        with torch.no_grad():
            pos_sim = torch.matmul(rel_flat, class_prototypes.T) / self.temperature
            neg_sim = torch.matmul(rel_flat, irr_flat.T) / self.temperature
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
        labels: Optional[torch.Tensor] = None,
        patches_per_sample: int = 1,
    ) -> torch.Tensor:
        """Supervised NT-Xent: each anchor → own class proto (pos) vs all
        other protos + irrelevant patches (neg)."""
        num_classes = prototypes.size(0)
        N = anchors.size(0)  # B * K

        # Similarity of anchors to all class prototypes  [N, C]
        proto_sim = torch.matmul(anchors, prototypes.T) / self.temperature
        # Similarity of anchors to all irrelevant patches [N, B*BK]
        neg_sim = torch.matmul(anchors, negatives.T) / self.temperature

        # Build per-anchor positive index (the GT class column in proto_sim)
        if labels is not None:
            # labels: [B] → repeat each K times to get [B*K]
            anchor_labels = labels.repeat_interleave(patches_per_sample)
        else:
            anchor_labels = torch.arange(N, device=anchors.device) % num_classes

        # Positive score: similarity to the GT class prototype  [N, 1]
        pos_score = proto_sim.gather(1, anchor_labels.unsqueeze(1))

        # Negative logits: all proto columns (incl. pos — will be masked)
        # + irrelevant patch similarities
        logits = torch.cat([proto_sim, neg_sim], dim=1)  # [N, C + B*BK]

        # Mask out the positive prototype column so it is not double-counted
        mask = torch.ones_like(logits)
        mask.scatter_(1, anchor_labels.unsqueeze(1), 0.0)

        # log-sum-exp over negatives only
        neg_logits = logits + (mask.log())  # -inf for masked positions
        log_sum_exp_neg = torch.logsumexp(neg_logits, dim=1, keepdim=True)

        # InfoNCE: -log( exp(pos) / (exp(pos) + sum_neg exp(neg)) )
        loss = -pos_score + torch.logaddexp(pos_score, log_sum_exp_neg)
        return loss.mean()

