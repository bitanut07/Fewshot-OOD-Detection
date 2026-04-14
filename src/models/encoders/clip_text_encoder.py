# -*- coding: utf-8 -*-
"""CLIP Text Encoder for encoding class names and disease descriptions.

Encodes text prompts (class names, disease descriptions) into embeddings
using a frozen CLIP text encoder. These embeddings are used for
global-local alignment with image features.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import open_clip


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder that encodes text prompts into embeddings.

    Architecture:
        Text Prompts -> Tokenizer -> CLIP Text Encoder -> Text Embeddings

    Args:
        backbone: CLIP model name (must match image encoder)
        pretrained: Pretrained weights source (must match image encoder)
        freeze: Whether to freeze the encoder weights
        device: Device to load model on
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

        # Load CLIP model (only need text encoder)
        self.model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained, device=self.device
        )
        self.model.eval() if freeze else None

        self.embed_dim = self.model.text.embed_dim  # type: ignore
        self.context_length = self.model.text.context_length  # type: ignore
        self.vocab_size = self.model.text.vocab_size  # type: ignore

        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():  # type: ignore
                param.requires_grad = False
            self.eval()
        else:
            self.train()

    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text prompts into embeddings.

        Args:
            texts: List of text prompts (e.g., ["A photo of Fracture", ...])
            normalize: Whether to L2-normalize embeddings

        Returns:
            Text embeddings [num_texts, embed_dim]
        """
        with torch.no_grad() if self.freeze else torch.enable_grad():
            token_ids = open_clip.tokenize(texts).to(self.device)
            embeddings = self.model.encode_text(token_ids)
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings

    def encode_descriptions(
        self,
        class_names: List[str],
        descriptions: List[List[str]],
        use_template: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode class names and their descriptions.

        Args:
            class_names: List of class names (e.g., ["Fracture", "Arthritis"])
            descriptions: List of description lists per class
            use_template: Prepend "A photo of {class_name}" as first description
            normalize: Whether to L2-normalize embeddings

        Returns:
            All embeddings [total_num_texts, embed_dim]
        """
        all_texts = []
        for cls_name, descs in zip(class_names, descriptions):
            if use_template:
                all_texts.append(f"A photo of {cls_name}")
            all_texts.extend(descs)
        return self.encode_text(all_texts, normalize=normalize)

    def forward(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass: encode texts to embeddings."""
        return self.encode_text(texts, normalize=normalize)
