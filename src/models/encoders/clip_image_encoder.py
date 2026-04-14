# -*- coding: utf-8 -*-
"""CLIP Image Encoder with global and local feature extraction.

Extracts both global (pooled) and local (per-patch) features from images
using a frozen CLIP vision encoder. Local features are used for
disease-relevant region selection and local contrastive learning.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import open_clip


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder that returns both global and local features.

    Architecture:
        Input Image -> CLIP Vision Encoder -> Global Feature (pooled)
                                       └── Local Features (per-patch, sequence of patch embeddings)

    Args:
        backbone: CLIP model name (e.g., "ViT-B/16")
        pretrained: Pretrained weights source (e.g., "openai")
        freeze: Whether to freeze the encoder weights
        device: Device to load model on

    Attributes:
        model: The underlying CLIP vision model
        embed_dim: Dimension of global embedding
        local_dim: Dimension of per-patch local features
        num_patches: Number of patches (196 for ViT-B/16 @ 224px)
    """

    def __init__(
        self,
        backbone: str = "ViT-B/16",
        pretrained: str = "openai",
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained, device=self.device
        )
        self.model.eval() if freeze else None

        # Extract dimensions from model
        # ViT-B/16: embed_dim=512, local_dim=768, num_patches=196
        self.embed_dim = self.model.visual.embed_dim  # type: ignore
        self.local_dim = self.model.visual.width  # type: ignore
        self.num_patches = (self.model.visual.image_size[0] // self.model.visual.patch_size[0]) ** 2  # type: ignore

        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():  # type: ignore
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def encode_image(
        self,
        images: torch.Tensor,
        return_local: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images to global and local features.

        Args:
            images: Input image tensor [B, C, H, W]
            return_local: Whether to return per-patch local features

        Returns:
            global_feat: Global pooled feature [B, embed_dim]
            local_feat: Per-patch features [B, num_patches, local_dim] or None
        """
        # Get CLS token + all patch tokens from vision transformer
        x = self.model.visual.conv1(images)  # [B, width, h, w]
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [B, h*w, width]

        # Add class token and positional embedding
        cls_token = self.model.visual.class_token  # type: ignore
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.model.visual.positional_embedding  # type: ignore

        # Pass through transformer blocks
        x = self.model.visual.transformer(x)  # type: ignore

        # Split into global (CLS) and local (patch) features
        global_feat = x[:, 0]  # [B, embed_dim]

        if not return_local:
            return global_feat, None

        # Project to embed_dim if needed
        local_feat = x[:, 1:]  # [B, num_patches, width]
        return global_feat, local_feat

    def forward(
        self,
        images: torch.Tensor,
        return_local: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: encode images to global and local features."""
        return self.encode_image(images, return_local)

    def get_global_only(self, images: torch.Tensor) -> torch.Tensor:
        """Get only global features (faster, no local features computed)."""
        global_feat, _ = self.encode_image(images, return_local=False)
        return global_feat
