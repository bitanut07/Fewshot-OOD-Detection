# -*- coding: utf-8 -*-
"""GLocalFSLOODModel: Full model integration for Global-Local FSL-OOD.

Integrates all components:
  1. CLIP Image Encoder (frozen)
  2. CLIP Text Encoder (frozen)
  3. Disease Text Refiner (trainable)
  4. Lesion Region Selector (trainable)
  5. Local Contrastive Learner (trainable)
  6. Global-Local Aligner (trainable)

Supports config-driven instantiation via Registry.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..encoders.clip_image_encoder import CLIPImageEncoder
from ..encoders.clip_text_encoder import CLIPTextEncoder
from ..modules.text_refinement import DiseaseTextRefiner
from ..modules.local_region_selector import LesionRegionSelector
from ..modules.local_contrastive import LocalContrastiveLearner
from ..modules.global_local_alignment import GlobalLocalAligner


class GLocalFSLOODModel(nn.Module):
    """
    Full GLOCAL-FSL-OOD model integrating all components.

    The model processes images through the following pipeline:
        Image -> CLIPImageEncoder -> global_feat + local_feats
                                    |
                                    v
                         LesionRegionSelector
                                    |
                         top_k + bottom_k patches
                                    |
                         LocalContrastiveLearner (loss only)
                                    |
                         GlobalLocalAligner <-- refined text embeddings
                                    |
                               logits

    Args:
        config: Config object with all model parameters
        class_names: List of class names for text encoding
        descriptions: Dict mapping class_name -> list of descriptions
        device: Device to run model on
    """

    def __init__(
        self,
        config: Any,
        class_names: List[str],
        descriptions: Dict[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. CLIP Image Encoder (frozen)
        clip_cfg = config.model.clip
        self.image_encoder = CLIPImageEncoder(
            backbone=clip_cfg.backbone,
            pretrained=clip_cfg.pretrained,
            freeze=clip_cfg.freeze,
            device=self.device,
        )
        self.embed_dim = self.image_encoder.embed_dim
        self.local_dim = self.image_encoder.local_dim

        # 2. CLIP Text Encoder (frozen)
        self.text_encoder = CLIPTextEncoder(
            backbone=clip_cfg.backbone,
            pretrained=clip_cfg.pretrained,
            freeze=True,
            device=self.device,
        )

        # 3. Encode all class descriptions
        self.class_names = class_names
        self.descriptions = descriptions
        self._build_text_embeddings()

        # 4. Disease Text Refiner (trainable, optional)
        if config.model.text_refiner.trainable:
            tr_cfg = config.model.text_refiner
            self.text_refiner = DiseaseTextRefiner(
                text_dim=self.embed_dim,
                visual_dim=self.embed_dim,
                hidden_dim=tr_cfg.hidden_dim,
                num_heads=tr_cfg.num_heads,
                num_layers=tr_cfg.num_layers,
                dropout=tr_cfg.dropout,
                alpha=tr_cfg.alpha,
                trainable=True,
            )
        else:
            self.text_refiner = None

        # 5. Lesion Region Selector (trainable, optional)
        if config.model.local_contrastive.trainable:
            lr_cfg = config.model.local_region_selector
            self.region_selector = LesionRegionSelector(
                top_k=lr_cfg.top_k,
                bottom_k=lr_cfg.bottom_k,
                similarity_metric=lr_cfg.similarity_metric,
                normalize_before_sim=lr_cfg.normalize_before_sim,
            )
        else:
            self.region_selector = None

        # 6. Local Contrastive Learner (trainable, optional)
        if config.model.local_contrastive.trainable:
            lc_cfg = config.model.local_contrastive
            self.local_contrastive = LocalContrastiveLearner(
                temperature=lc_cfg.temperature,
                top_k=lc_cfg.top_k,
                bottom_k=lc_cfg.bottom_k,
                embed_dim=self.embed_dim,
                trainable=True,
            )
        else:
            self.local_contrastive = None

        # 7. Global-Local Aligner (trainable)
        al_cfg = config.model.alignment
        self.aligner = GlobalLocalAligner(
            embed_dim=self.embed_dim,
            local_dim=self.local_dim,
            alpha_global=al_cfg.alpha_global,
            alpha_local=al_cfg.alpha_local,
            logit_temperature=al_cfg.logit_temperature,
            learnable_weights=al_cfg.learnable_weights,
            trainable=True,
        )

        # Register trainable vs frozen for clarity
        self.trainable_modules = nn.ModuleDict()
        if self.text_refiner is not None:
            self.trainable_modules["text_refiner"] = self.text_refiner
        if self.region_selector is not None:
            self.trainable_modules["region_selector"] = self.region_selector
        if self.local_contrastive is not None:
            self.trainable_modules["local_contrastive"] = self.local_contrastive
        self.trainable_modules["aligner"] = self.aligner

    def _build_text_embeddings(self) -> None:
        """Build and cache text embeddings for all classes and descriptions."""
        all_texts = []
        self.text_per_class = []  # number of texts per class (for pooling)
        for cls_name in self.class_names:
            texts = [f"A photo of {cls_name}"]
            if cls_name in self.descriptions:
                descs = self.descriptions.get(cls_name) or []
                if not isinstance(descs, list):
                    raise TypeError(
                        f"Descriptions for class '{cls_name}' must be a list of strings, got {type(descs)}"
                    )
                texts.extend([d for d in descs if isinstance(d, str) and d.strip()])
            self.text_per_class.append(len(texts))
            all_texts.extend(texts)

        self.all_text_embeddings = self.text_encoder.encode_text(all_texts, normalize=True)
        # Compute per-class averaged embedding (prototype)
        self.class_prototypes = self._compute_class_prototypes()

    def _compute_class_prototypes(self) -> torch.Tensor:
        """Compute averaged text embedding per class."""
        prototypes = []
        offset = 0
        for num_texts in self.text_per_class:
            proto = self.all_text_embeddings[offset:offset + num_texts].mean(dim=0)
            prototypes.append(proto)
            offset += num_texts
        return torch.stack(prototypes, dim=0)  # [num_classes, embed_dim]

    def get_text_embeddings(self) -> torch.Tensor:
        """Return all text embeddings."""
        return self.all_text_embeddings

    def get_class_prototypes(self) -> torch.Tensor:
        """Return per-class averaged text prototypes."""
        return self.class_prototypes

    def forward(
        self,
        images: torch.Tensor,
        return_loss: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            images: Image tensor [B, 3, H, W]
            return_loss: Whether to compute losses
            labels: Class labels [B] (needed for loss computation)

        Returns:
            Dictionary with logits, features, and optionally losses
        """
        B = images.size(0)

        # Encode image
        global_feat, local_feat = self.image_encoder(images, return_local=True)
        # global_feat: [B, embed_dim], local_feat: [B, num_patches, local_dim]

        # Refine per-class text prototypes (optional)
        if self.text_refiner is not None:
            base_text = self.class_prototypes.to(self.device).unsqueeze(0).expand(B, -1, -1)
            refined_text = self.text_refiner(base_text, global_feat.unsqueeze(1))
            text_for_align = refined_text  # [B, num_classes, embed_dim]
        else:
            text_for_align = self.class_prototypes.to(self.device)  # [num_classes, embed_dim]

        # Select regions (optional)
        if self.region_selector is not None and local_feat is not None:
            # Project local_feat to embed_dim if needed
            if local_feat.shape[-1] != self.embed_dim:
                # TODO: project local features
                pass
            proto_for_region = self.class_prototypes.to(self.device).unsqueeze(0).expand(B, -1, -1)
            region_result = self.region_selector(local_feat, proto_for_region)
            relevant_feats = region_result["relevant_features"]
            irrelevant_feats = region_result["irrelevant_features"]
        else:
            relevant_feats = None
            irrelevant_feats = None

        # Local contrastive loss (computed but not used for forward logits)
        loss_contrastive = None
        metrics = {}
        if return_loss and self.local_contrastive is not None and relevant_feats is not None:
            loss_contrastive, lc_metrics = self.local_contrastive(
                relevant_feats, irrelevant_feats, self.class_prototypes.to(self.device), labels
            )
            metrics.update(lc_metrics)

        # Global-Local Alignment for final logits
        use_local = local_feat is not None and self.region_selector is not None
        logits, align_metrics = self.aligner(global_feat, local_feat, text_for_align, use_local=use_local)
        metrics.update(align_metrics)

        return {
            "logits": logits,  # [B, num_classes]
            "global_feat": global_feat,
            "local_feat": local_feat,
            "loss_contrastive": loss_contrastive,
            "metrics": metrics,
        }

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return list of trainable parameters."""
        return list(self.trainable_modules.parameters())

    def freeze_encoders(self) -> None:
        """Explicitly freeze encoder modules."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
        self.text_encoder.eval()
