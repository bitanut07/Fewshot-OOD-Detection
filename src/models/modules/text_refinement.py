# -*- coding: utf-8 -*-
"""Disease Text Refiner Module.

Refines raw CLIP text embeddings using disease-relevant visual embeddings.
Uses self-attention, cross-attention, and feed-forward layers to produce
refined text embeddings that are more aligned with visual features.
Architecture follows the paper's Text Refinement (c) component.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseTextRefiner(nn.Module):
    """
    Text Refinement Module that refines text embeddings using visual features.

    Architecture:
        Text Embeddings + Visual Embeddings -> Self-Attention
                                          -> Cross-Attention (text attends to visual)
                                          -> FFN
                                          -> Refined Text Embeddings

    The refinement interpolates between original text and refined output:
        refined = alpha * text + (1 - alpha) * transformed

    Args:
        text_dim: Dimension of text embeddings from CLIP
        visual_dim: Dimension of visual embeddings from CLIP
        hidden_dim: Hidden dimension for transformer layers
        num_heads: Number of attention heads
        num_layers: Number of refinement layers
        dropout: Dropout probability
        alpha: Interpolation weight (0=pure-text, 1=pure-refined)
        trainable: Whether this module is trainable
    """

    def __init__(
        self,
        text_dim: int = 512,
        visual_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.5,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.trainable_flag = trainable

        # Project visual features to hidden_dim if needed
        if visual_dim != hidden_dim:
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        else:
            self.visual_proj = nn.Identity()

        # Project text features to hidden_dim
        if text_dim != hidden_dim:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()

        # Refinement transformer layers
        self.layers = nn.ModuleList([
            RefinementLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, text_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        text_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
        return_refined_only: bool = True,
    ) -> torch.Tensor:
        """
        Refine text embeddings using disease-relevant visual embeddings.

        Args:
            text_embeddings: Text embeddings [B, T, text_dim] or [B, text_dim]
            visual_embeddings: Visual embeddings [B, V, visual_dim] or [B, visual_dim]
            return_refined_only: If True, return only refined embeddings

        Returns:
            Refined text embeddings [B, T, text_dim] or [B, text_dim]
        """
        # Handle 2D vs 3D tensors
        squeeze = False
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(1)
            squeeze = True
        if visual_embeddings.dim() == 2:
            visual_embeddings = visual_embeddings.unsqueeze(1)

        # Project to hidden_dim
        x = self.text_proj(text_embeddings)  # [B, T, hidden_dim]
        visual = self.visual_proj(visual_embeddings)  # [B, V, hidden_dim]

        # Pass through refinement layers
        for layer in self.layers:
            x = layer(x, visual)

        # Project back to text_dim and apply interpolation
        refined = self.output_proj(x)

        if return_refined_only:
            result = refined
        else:
            # Also return original text for comparison
            result = torch.cat([text_embeddings, refined], dim=1)

        if squeeze:
            result = result.squeeze(1)

        return result


class RefinementLayer(nn.Module):
    """Single refinement layer with self-attention, cross-attention, and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Self-attention on text
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention: text attends to visual
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention with residual: text attends to visual
        cross_out, _ = self.cross_attn(x, visual, visual)
        x = self.norm2(x + self.dropout(cross_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x
