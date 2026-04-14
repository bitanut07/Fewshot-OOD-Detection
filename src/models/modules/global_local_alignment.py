# -*- coding: utf-8 -*-
"""Global-Local Aligner: Combine global and local alignment for classification.

Computes final classification logits by combining:
    - Global alignment: image global feature vs class text features
    - Local alignment: image local features vs class text features (per-patch)

Final logits = alpha_global * global_logits + alpha_local * local_logits
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GlobalLocalAligner(nn.Module):
    """
    Global-Local Alignment Module for few-shot classification.

    Computes alignment scores between image features and class text features
    at both global and local levels, then combines them for classification.

    Architecture:
        Global Image Feature + Class Text Features -> Global Alignment Score
        Local Image Features + Class Text Features -> Local Alignment Score
        Global Score * alpha_global + Local Score * alpha_local -> Final Logits

    Args:
        embed_dim: Feature dimension for global alignment
        local_dim: Feature dimension for local alignment
        alpha_global: Weight for global alignment
        alpha_local: Weight for local alignment
        logit_temperature: Temperature for softmax over logits
        learnable_weights: If True, make alpha_global/alpha_local learnable
        trainable: Whether this module is trainable
    """

    def __init__(
        self,
        embed_dim: int = 512,
        local_dim: int = 768,
        alpha_global: float = 0.5,
        alpha_local: float = 0.5,
        logit_temperature: float = 1.0,
        learnable_weights: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.logit_temperature = logit_temperature
        self.learnable_weights = learnable_weights
        self.trainable_flag = trainable

        # Project local features to embed_dim if needed
        if local_dim != embed_dim:
            self.local_proj = nn.Linear(local_dim, embed_dim)
        else:
            self.local_proj = nn.Identity()

        if learnable_weights:
            self.log_alpha_global = nn.Parameter(torch.tensor(0.0))
            self.log_alpha_local = nn.Parameter(torch.tensor(0.0))

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def compute_global_alignment(
        self,
        image_global: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global alignment scores.

        Args:
            image_global: Global image feature [B, embed_dim]
            text_embeddings: Class text embeddings [num_classes, embed_dim] or [B, num_classes, embed_dim]

        Returns:
            Global alignment logits [B, num_classes]
        """
        # Normalize
        image_global = F.normalize(image_global, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Cosine similarity
        if text_embeddings.dim() == 3:
            # [B, num_classes, embed_dim]
            sim = (image_global.unsqueeze(1) * text_embeddings).sum(dim=-1)
        else:
            # [num_classes, embed_dim]
            sim = torch.matmul(image_global, text_embeddings.T)

        return sim / self.logit_temperature

    def compute_local_alignment(
        self,
        local_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Compute local alignment scores by averaging per-patch similarities.

        Args:
            local_features: Local patch features [B, num_patches, local_dim]
            text_embeddings: Class text embeddings [num_classes, embed_dim] or [B, num_classes, embed_dim]
            aggregation: How to aggregate per-patch scores (mean, max)

        Returns:
            Local alignment logits [B, num_classes]
        """
        # Project to embed_dim
        local_proj = self.local_proj(local_features)  # [B, num_patches, embed_dim]
        local_proj = F.normalize(local_proj, p=2, dim=-1)
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)

        # Per-patch cosine similarity
        if text_norm.dim() == 3:
            # [B, num_patches, embed_dim] * [B, num_classes, embed_dim] -> [B, num_patches, num_classes]
            sim = torch.einsum("bpd,bcd->bpc", local_proj, text_norm)
        else:
            # [B, num_patches, embed_dim] * [num_classes, embed_dim] -> [B, num_patches, num_classes]
            sim = torch.einsum("bpd,cd->bpc", local_proj, text_norm)

        # Aggregate across patches
        if aggregation == "mean":
            sim = sim.mean(dim=1)  # [B, num_classes]
        elif aggregation == "max":
            sim = sim.max(dim=1).values  # [B, num_classes]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return sim / self.logit_temperature

    def forward(
        self,
        image_global: torch.Tensor,
        local_features: Optional[torch.Tensor],
        text_embeddings: torch.Tensor,
        use_local: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute final alignment logits.

        Args:
            image_global: Global image feature [B, embed_dim]
            local_features: Local patch features [B, num_patches, local_dim] or None
            text_embeddings: Class text embeddings
            use_local: Whether to include local alignment

        Returns:
            Tuple of (logits [B, num_classes], metrics_dict)
        """
        # Global alignment
        global_logits = self.compute_global_alignment(image_global, text_embeddings)

        if not use_local or local_features is None:
            logits = global_logits
            metrics = {"global_logits_mean": global_logits.mean().item()}
        else:
            # Local alignment
            local_logits = self.compute_local_alignment(local_features, text_embeddings)

            # Combine
            if self.learnable_weights:
                a_g = torch.sigmoid(self.log_alpha_global)
                a_l = torch.sigmoid(self.log_alpha_local)
                total = a_g + a_l + 1e-8
                a_g = a_g / total
                a_l = a_l / total
            else:
                a_g = self.alpha_global
                a_l = self.alpha_local

            logits = a_g * global_logits + a_l * local_logits

            metrics = {
                "global_logits_mean": global_logits.mean().item(),
                "local_logits_mean": local_logits.mean().item(),
                "alpha_global": a_g if isinstance(a_g, float) else a_g.item(),
                "alpha_local": a_l if isinstance(a_l, float) else a_l.item(),
            }

        return logits, metrics
