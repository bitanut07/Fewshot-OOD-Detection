# -*- coding: utf-8 -*-
"""GLocalFSLOODModel: Full model integration for Global-Local FSL-OOD.

Integrates all components, inspired by the ``LocProto`` architecture
from ``glali/trainers/locproto_supc.py``:

  1. CLIP Image Encoder (student, partially trainable in last attn block)
  2. CLIP Image Encoder (teacher, frozen copy — for distillation)
  3. CLIP Text Encoder (frozen)
  4. Disease Text Refiner (trainable)  — attends text prototypes to visual
  5. Lesion Region Selector (top-k / bottom-k patches w.r.t. prototype)
  6. Local Contrastive Learner (SupCon-like on relevant / irrelevant)
  7. Global-Local Aligner (global logits + local logits + combined logits)

The forward pass returns — besides ``logits`` — everything needed for:
  - Classification on combined logits
  - Classification on local logits (per-patch aggregated)
  - Entropy regularization on non-topK local patches (glali-style OOD reg)
  - Teacher / student feature distillation (global image features)
  - Teacher / student text-prototype distillation (refined vs raw text)
  - Local contrastive loss over selected regions

All extra outputs are ``None`` when the corresponding component is disabled.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.clip_image_encoder import CLIPImageEncoder
from ..encoders.clip_text_encoder import CLIPTextEncoder
from ..modules.global_local_alignment import GlobalLocalAligner
from ..modules.local_contrastive import LocalContrastiveLearner
from ..modules.local_region_selector import LesionRegionSelector
from ..modules.text_refinement import DiseaseTextRefiner


def _get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested getter for dict / Config-like objects."""
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


class GLocalFSLOODModel(nn.Module):
    """Full GLOCAL-FSL-OOD model.

    Args:
        config: Config object (supports both ``config.model.clip`` and
                top-level ``config.clip`` style configs).
        class_names: List of class names for text encoding.
        descriptions: Dict mapping ``class_name → list[str]`` of LLM
                      descriptions.
        device: Device to run model on.
        build_teacher: Whether to build a frozen teacher image encoder
                       for distillation (matches glali's ``zs_img_encoder``).
    """

    def __init__(
        self,
        config: Any,
        class_names: List[str],
        descriptions: Dict[str, List[str]],
        device: Optional[torch.device] = None,
        build_teacher: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- resolve sub-configs (support nested model.* or top-level) ---
        clip_cfg = _get(config, "model", "clip") or _get(config, "clip") or {}
        tr_cfg = _get(config, "model", "text_refiner") or _get(config, "text_refiner") or {}
        lrs_cfg = _get(config, "model", "local_region_selector") or _get(config, "local_region_selector") or {}
        lc_cfg = _get(config, "model", "local_contrastive") or _get(config, "local_contrastive") or {}
        al_cfg = _get(config, "model", "alignment") or _get(config, "alignment") or {}

        backbone = _get(clip_cfg, "backbone", default="ViT-B/16")
        pretrained = _get(clip_cfg, "pretrained", default="openai")
        freeze_clip = _get(clip_cfg, "freeze", default=True)

        # 1. CLIP Image Encoder (student)
        self.image_encoder = CLIPImageEncoder(
            backbone=backbone, pretrained=pretrained, freeze=freeze_clip, device=self.device,
        )
        self.embed_dim = self.image_encoder.embed_dim
        self.local_dim = self.image_encoder.local_dim

        # 2. Teacher image encoder (frozen copy) — glali zs_img_encoder analogue
        self.build_teacher = bool(build_teacher)
        if self.build_teacher:
            self.teacher_image_encoder = copy.deepcopy(self.image_encoder)
            for p in self.teacher_image_encoder.parameters():
                p.requires_grad = False
            self.teacher_image_encoder.eval()
        else:
            self.teacher_image_encoder = None

        # 3. CLIP Text Encoder (frozen)
        self.text_encoder = CLIPTextEncoder(
            backbone=backbone, pretrained=pretrained, freeze=True, device=self.device,
        )

        # 4. Build per-class text embeddings + prototypes
        self.class_names = class_names
        self.descriptions = descriptions or {}
        self._build_text_embeddings()

        # 5. Disease Text Refiner (optional)
        if bool(_get(tr_cfg, "trainable", default=False)):
            self.text_refiner = DiseaseTextRefiner(
                text_dim=self.embed_dim,
                visual_dim=self.embed_dim,
                hidden_dim=int(_get(tr_cfg, "hidden_dim", default=self.embed_dim)),
                num_heads=int(_get(tr_cfg, "num_heads", default=8)),
                num_layers=int(_get(tr_cfg, "num_layers", default=2)),
                dropout=float(_get(tr_cfg, "dropout", default=0.1)),
                alpha=float(_get(tr_cfg, "alpha", default=0.5)),
                trainable=True,
            )
        else:
            self.text_refiner = None

        # 6. Region selector + local contrastive (jointly enabled)
        use_local_contrast = bool(_get(lc_cfg, "trainable", default=False))
        if use_local_contrast:
            self.region_selector = LesionRegionSelector(
                top_k=int(_get(lrs_cfg, "top_k", default=4)),
                bottom_k=int(_get(lrs_cfg, "bottom_k", default=4)),
                similarity_metric=str(_get(lrs_cfg, "similarity_metric", default="cosine")),
                normalize_before_sim=bool(_get(lrs_cfg, "normalize_before_sim", default=True)),
            )
            self.local_contrastive = LocalContrastiveLearner(
                temperature=float(_get(lc_cfg, "temperature", default=0.1)),
                top_k=int(_get(lc_cfg, "top_k", default=4)),
                bottom_k=int(_get(lc_cfg, "bottom_k", default=4)),
                embed_dim=self.embed_dim,
                trainable=True,
            )
        else:
            self.region_selector = None
            self.local_contrastive = None

        # 7. Global-Local Aligner
        self.aligner = GlobalLocalAligner(
            embed_dim=self.embed_dim,
            local_dim=self.local_dim,
            alpha_global=float(_get(al_cfg, "alpha_global", default=0.5)),
            alpha_local=float(_get(al_cfg, "alpha_local", default=0.5)),
            logit_temperature=float(_get(al_cfg, "logit_temperature", default=1.0)),
            learnable_weights=bool(_get(al_cfg, "learnable_weights", default=False)),
            trainable=True,
        )

        # Bookkeeping: which modules are trainable
        self.trainable_modules = nn.ModuleDict()
        if self.text_refiner is not None:
            self.trainable_modules["text_refiner"] = self.text_refiner
        if self.local_contrastive is not None:
            self.trainable_modules["local_contrastive"] = self.local_contrastive
        if self.region_selector is not None:
            self.trainable_modules["region_selector"] = self.region_selector
        self.trainable_modules["aligner"] = self.aligner

    # ------------------------------------------------------------------
    # Text embeddings
    # ------------------------------------------------------------------
    def _build_text_embeddings(self) -> None:
        all_texts: List[str] = []
        self.text_per_class: List[int] = []
        template = "A photo of {}"
        for cls_name in self.class_names:
            texts = [template.format(cls_name.replace("_", " "))]
            descs = self.descriptions.get(cls_name) or self.descriptions.get(cls_name.replace("_", " ")) or []
            if isinstance(descs, list):
                texts.extend([d for d in descs if isinstance(d, str) and d.strip()])
            self.text_per_class.append(len(texts))
            all_texts.extend(texts)

        with torch.no_grad():
            all_embeds = self.text_encoder.encode_text(all_texts, normalize=True)
        # Register as buffer (not a parameter) so checkpointing still works
        self.register_buffer("all_text_embeddings", all_embeds, persistent=False)
        self.register_buffer("class_prototypes", self._compute_prototypes(all_embeds), persistent=False)

    def _compute_prototypes(self, all_embeds: torch.Tensor) -> torch.Tensor:
        protos = []
        off = 0
        for n in self.text_per_class:
            protos.append(all_embeds[off:off + n].mean(dim=0))
            off += n
        protos = torch.stack(protos, dim=0)
        return F.normalize(protos, p=2, dim=-1)

    def get_class_prototypes(self) -> torch.Tensor:
        return self.class_prototypes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        return_loss: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full forward pass.

        Returns dict with keys:
            logits, global_logits, local_logits, global_feat, local_feat,
            global_feat_teacher, text_for_align, class_prototypes,
            relevant_features, irrelevant_features,
            loss_contrastive, metrics.
        """
        images = images.to(self.device)
        B = images.size(0)

        # --- student encode
        global_feat, local_feat = self.image_encoder(images, return_local=True)

        # --- teacher encode (no grad, frozen)
        global_feat_teacher: Optional[torch.Tensor] = None
        if self.teacher_image_encoder is not None:
            with torch.no_grad():
                global_feat_teacher, _ = self.teacher_image_encoder(images, return_local=False)

        # --- text / prototypes (optionally refined via visual)
        protos = self.class_prototypes.to(self.device)  # [C, D]
        if self.text_refiner is not None:
            base_text = protos.unsqueeze(0).expand(B, -1, -1)     # [B, C, D]
            refined = self.text_refiner(base_text, global_feat.unsqueeze(1))
            text_for_align = refined                                # [B, C, D]
        else:
            text_for_align = protos                                 # [C, D]

        # --- region selection
        relevant_feats = irrelevant_feats = None
        if self.region_selector is not None and local_feat is not None:
            proto_for_region = protos.unsqueeze(0).expand(B, -1, -1)
            region_out = self.region_selector(local_feat, proto_for_region)
            relevant_feats = region_out["relevant_features"]
            irrelevant_feats = region_out["irrelevant_features"]

        # --- local contrastive loss
        loss_contrastive: Optional[torch.Tensor] = None
        metrics: Dict[str, Any] = {}
        if return_loss and self.local_contrastive is not None and relevant_feats is not None:
            loss_contrastive, lc_metrics = self.local_contrastive(
                relevant_feats, irrelevant_feats, protos, labels,
            )
            metrics.update(lc_metrics)

        # --- alignment → logits
        use_local = (local_feat is not None)
        combined_logits, align_metrics = self.aligner(
            global_feat, local_feat, text_for_align, use_local=use_local,
        )
        metrics.update(align_metrics)

        # Also expose the individual global / local logits separately, so
        # the trainer can compute loss on each branch (mirrors glali).
        global_logits = self.aligner.compute_global_alignment(global_feat, text_for_align)
        if local_feat is not None:
            local_logits = self._compute_local_logits(local_feat, text_for_align)
        else:
            local_logits = None

        return {
            "logits": combined_logits,                # [B, C] — main classification logits
            "global_logits": global_logits,           # [B, C]
            "local_logits": local_logits,             # [B, P, C] or None
            "global_feat": global_feat,               # [B, D]
            "local_feat": local_feat,                 # [B, P, D] or None
            "global_feat_teacher": global_feat_teacher,  # [B, D] or None
            "text_for_align": text_for_align,
            "class_prototypes": protos,
            "relevant_features": relevant_feats,
            "irrelevant_features": irrelevant_feats,
            "loss_contrastive": loss_contrastive,
            "metrics": metrics,
        }

    def _compute_local_logits(
        self,
        local_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-patch logits [B, P, C] (no aggregation).

        Used for OOD entropy regularization — we want the *distribution*
        over classes for every patch so we can push non-topK patches
        toward a uniform distribution (glali entropy_select_topk style).
        """
        local_proj = self.aligner.local_proj(local_features)
        local_proj = F.normalize(local_proj, p=2, dim=-1)
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)
        if text_norm.dim() == 3:  # [B, C, D]
            sim = torch.einsum("bpd,bcd->bpc", local_proj, text_norm)
        else:                       # [C, D]
            sim = torch.einsum("bpd,cd->bpc", local_proj, text_norm)
        return sim / self.aligner.logit_temperature

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_encoders(self) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()
        self.text_encoder.eval()
        if self.teacher_image_encoder is not None:
            for p in self.teacher_image_encoder.parameters():
                p.requires_grad = False
            self.teacher_image_encoder.eval()
