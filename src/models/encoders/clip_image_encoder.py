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

from .openai_clip_weights import resolve_open_clip_load_args


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder that returns both global and local features.

    Architecture:
        Input Image -> CLIP Vision Encoder -> Global Feature (pooled)
                                       └── Local Features (per-patch, sequence of patch embeddings)

    Args:
        backbone: CLIP model name (e.g., ``"ViT-B/16"``). For OpenAI-public weights,
                  any alias matching ``ViT-B/16`` / ``ViT-B-16`` is accepted.
        pretrained: Weight source: ``"openai"`` (open_clip hub tag), a local ``.pt``
                    path, or ``"openai_public"`` / ``"openai-azure"`` to download the
                    same Azure ``.pt`` files as glali and load via ``open_clip``.
        freeze: Whether to freeze the encoder weights
        device: Device to load model on
        weight_cache_dir: Cache directory for ``openai_public`` downloads
            (defaults to ``CLIP_OPENAI_CACHE`` or ``~/.cache/clip``).

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
        weight_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone.replace("/", "-")
        self.pretrained = pretrained
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        oc_name, pretrained_arg = resolve_open_clip_load_args(
            backbone, pretrained, weight_cache_dir=weight_cache_dir,
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            oc_name,
            pretrained=pretrained_arg,
            device=self.device,
        )
        self.model.eval() if freeze else None

        # Extract dimensions from model.
        # For ViT-B-16: global projected dim=512; token dim=768; num_patches=196 (224/16)^2.
        self.embed_dim = int(getattr(self.model.visual, "output_dim", 512))  # type: ignore
        image_size = getattr(self.model.visual, "image_size", 224)  # type: ignore
        patch_size = getattr(self.model.visual, "patch_size", 16)  # type: ignore
        if isinstance(image_size, (tuple, list)):
            image_size_0 = int(image_size[0])
        else:
            image_size_0 = int(image_size)
        if isinstance(patch_size, (tuple, list)):
            patch_size_0 = int(patch_size[0])
        else:
            patch_size_0 = int(patch_size)
        self.num_patches = (image_size_0 // patch_size_0) ** 2

        # We return local features projected to the same embedding space as text (embed_dim).
        self.local_dim = self.embed_dim

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
            local_feat: Per-patch features [B, num_patches, embed_dim] or None
        """
        # Global feature (projected to embed_dim by open_clip)
        global_feat = self.model.encode_image(images)

        if not return_local:
            return global_feat, None

        # Local patch tokens from the last ViT block, then project them to embed_dim.
        # forward_intermediates returns:
        #   - image_intermediates[-1]        : [B, num_patches, token_dim]
        #   - image_intermediates_prefix[-1] : [B, 1, token_dim]
        out = self.model.visual.forward_intermediates(  # type: ignore[attr-defined]
            images,
            indices=None,
            output_fmt="NLC",
            output_extra_tokens=True,
        )
        patch_tokens = out["image_intermediates"][-1]  # [B, num_patches, token_dim]
        # Project patch tokens to embed_dim using the same ln_post + proj as the CLS token.
        ln_post = getattr(self.model.visual, "ln_post", None)  # type: ignore
        proj = getattr(self.model.visual, "proj", None)  # type: ignore
        if ln_post is None or proj is None:
            raise RuntimeError("open_clip visual model does not expose ln_post/proj for token projection")

        patch_tokens = ln_post(patch_tokens)
        local_feat = patch_tokens @ proj  # [B, num_patches, embed_dim]

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
