# -*- coding: utf-8 -*-
"""Lesion Region Selector: Select disease-relevant vs disease-irrelevant regions.

Selects top-k disease-relevant regions (highest similarity to class prototype)
and bottom-k disease-irrelevant regions (lowest similarity) from patch embeddings.
Used by local contrastive learning to create positive/negative pairs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LesionRegionSelector(nn.Module):
    """
    Selects disease-relevant (top-k) and disease-irrelevant (bottom-k) regions.

    For each image, computes similarity between local patch embeddings and
    the class prototype, then selects:
        - Top-k patches: Disease-relevant regions (high similarity)
        - Bottom-k patches: Disease-irrelevant regions (low similarity)

    Args:
        top_k: Number of disease-relevant regions to select
        bottom_k: Number of disease-irrelevant regions to select
        similarity_metric: Metric for computing patch-prototype similarity
        normalize_before_sim: Whether to normalize embeddings before similarity
        aggregate_method: How to aggregate selected patches (mean, cls_token, attn_weighted)
    """

    def __init__(
        self,
        top_k: int = 4,
        bottom_k: int = 4,
        similarity_metric: str = "cosine",
        normalize_before_sim: bool = True,
        aggregate_method: str = "mean",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.similarity_metric = similarity_metric
        self.normalize_before_sim = normalize_before_sim
        self.aggregate_method = aggregate_method

    def select_regions(
        self,
        local_features: torch.Tensor,
        prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select disease-relevant and disease-irrelevant regions.

        Args:
            local_features: Local patch features [B, num_patches, dim]
            prototypes: Class prototypes [B, num_classes, dim] or [B, 1, dim]
            labels: GT class indices [B]. When provided, similarity is
                    computed against the GT class prototype only (matching
                    glali's per-sample region selection logic).

        Returns:
            Tuple of:
                top_k_features: Disease-relevant features [B, top_k, dim]
                bottom_k_features: Disease-irrelevant features [B, bottom_k, dim]
                top_k_indices: Indices of selected top-k patches [B, top_k]
                bottom_k_indices: Indices of selected bottom-k patches [B, bottom_k]
        """
        if self.normalize_before_sim:
            local_norm = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
            proto_norm = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            local_norm = local_features
            proto_norm = prototypes

        if self.similarity_metric not in ("cosine", "dot_product"):
            raise ValueError(f"Unknown similarity_metric: {self.similarity_metric}")

        # [B, num_patches, num_classes]
        sim = torch.bmm(local_norm, proto_norm.transpose(-2, -1))

        if labels is not None:
            # Use only the GT class prototype for each sample → [B, num_patches]
            gt_idx = labels.view(-1, 1, 1).expand(-1, sim.size(1), 1)
            sim = sim.gather(2, gt_idx).squeeze(-1)
        else:
            sim = sim.mean(dim=-1)  # fallback: average over all prototypes

        top_k = min(self.top_k, sim.shape[-1])
        bottom_k = min(self.bottom_k, sim.shape[-1])

        top_k_indices = torch.topk(sim, top_k, dim=-1).indices
        bottom_k_indices = torch.topk(sim, bottom_k, dim=-1, largest=False).indices

        B = local_features.size(0)
        batch_top = torch.arange(B, device=local_features.device).unsqueeze(1).expand(-1, top_k)
        batch_bot = torch.arange(B, device=local_features.device).unsqueeze(1).expand(-1, bottom_k)

        top_k_features = local_features[batch_top, top_k_indices]
        bottom_k_features = local_features[batch_bot, bottom_k_indices]

        return top_k_features, bottom_k_features, top_k_indices, bottom_k_indices

    def forward(
        self,
        local_features: torch.Tensor,
        prototypes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: select regions and aggregate.

        Args:
            local_features: Local patch features [B, num_patches, dim]
            prototypes: Class prototypes [B, num_classes, dim]
            labels: GT class indices [B] (optional but recommended).

        Returns:
            Dictionary with selected features and indices.
        """
        top_k_feat, bottom_k_feat, top_k_idx, bottom_k_idx = self.select_regions(
            local_features, prototypes, labels=labels,
        )

        return {
            "relevant_features": top_k_feat,
            "irrelevant_features": bottom_k_feat,
            "relevant_indices": top_k_idx,
            "irrelevant_indices": bottom_k_idx,
        }
