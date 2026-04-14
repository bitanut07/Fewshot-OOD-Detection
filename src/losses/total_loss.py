# -*- coding: utf-8 -*-
"""Total loss: weighted combination of all loss terms."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Optional


class TotalLoss(nn.Module):
    """
    Weighted combination of all loss terms for GLOCAL-FSL-OOD training.

    Total loss = w_cls * L_cls + w_ga * L_ga + w_la * L_la + w_lc * L_lc + w_tr * L_tr
    """

    def __init__(
        self,
        weight_cls: float = 1.0,
        weight_global_alignment: float = 1.0,
        weight_local_alignment: float = 1.0,
        weight_local_contrastive: float = 0.5,
        weight_text_refinement: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight_cls = weight_cls
        self.weight_global_alignment = weight_global_alignment
        self.weight_local_alignment = weight_local_alignment
        self.weight_local_contrastive = weight_local_contrastive
        self.weight_text_refinement = weight_text_refinement

    def forward(
        self,
        loss_cls: Optional[torch.Tensor] = None,
        loss_global_alignment: Optional[torch.Tensor] = None,
        loss_local_alignment: Optional[torch.Tensor] = None,
        loss_local_contrastive: Optional[torch.Tensor] = None,
        loss_text_refinement: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute weighted total loss.

        Returns:
            Dict with "total_loss" and individual loss values.
        """
        total = 0.0
        result = {"total_loss": None, "loss_cls": 0.0, "loss_global_alignment": 0.0,
                  "loss_local_alignment": 0.0, "loss_local_contrastive": 0.0,
                  "loss_text_refinement": 0.0}

        if loss_cls is not None:
            total = total + self.weight_cls * loss_cls
            result["loss_cls"] = loss_cls.item()
        if loss_global_alignment is not None:
            total = total + self.weight_global_alignment * loss_global_alignment
            result["loss_global_alignment"] = loss_global_alignment.item()
        if loss_local_alignment is not None:
            total = total + self.weight_local_alignment * loss_local_alignment
            result["loss_local_alignment"] = loss_local_alignment.item()
        if loss_local_contrastive is not None:
            total = total + self.weight_local_contrastive * loss_local_contrastive
            result["loss_local_contrastive"] = loss_local_contrastive.item()
        if loss_text_refinement is not None:
            total = total + self.weight_text_refinement * loss_text_refinement
            result["loss_text_refinement"] = loss_text_refinement.item()

        result["total_loss"] = total
        return result
