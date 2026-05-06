# -*- coding: utf-8 -*-
"""Total loss: weighted combination of all loss terms.

Supports the following loss components (all optional, summed with
per-term weights). Names mirror the glali training recipe so the
trainer can plug in additional regularization terms without touching
the model framework:

    L_total = w_cls              * L_cls
            + w_global_alignment * L_ga
            + w_local_alignment  * L_la
            + w_local_contrastive* L_lc
            + w_text_refinement  * L_tr
            + w_ood_reg          * L_ood_reg    (entropy on non-topK regions)
            + w_distill_img      * L_distill_img (teacher/student image)
            + w_distill_text     * L_distill_text (teacher/student text)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(
        self,
        weight_cls: float = 1.0,
        weight_global_alignment: float = 1.0,
        weight_local_alignment: float = 1.0,
        weight_local_contrastive: float = 0.5,
        weight_text_refinement: float = 0.1,
        weight_ood_reg: float = 0.25,
        weight_distill_img: float = 0.0,
        weight_distill_text: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight_cls = weight_cls
        self.weight_global_alignment = weight_global_alignment
        self.weight_local_alignment = weight_local_alignment
        self.weight_local_contrastive = weight_local_contrastive
        self.weight_text_refinement = weight_text_refinement
        self.weight_ood_reg = weight_ood_reg
        self.weight_distill_img = weight_distill_img
        self.weight_distill_text = weight_distill_text

    def forward(
        self,
        loss_cls: Optional[torch.Tensor] = None,
        loss_global_alignment: Optional[torch.Tensor] = None,
        loss_local_alignment: Optional[torch.Tensor] = None,
        loss_local_contrastive: Optional[torch.Tensor] = None,
        loss_text_refinement: Optional[torch.Tensor] = None,
        loss_ood_reg: Optional[torch.Tensor] = None,
        loss_distill_img: Optional[torch.Tensor] = None,
        loss_distill_text: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute weighted total loss and return a breakdown dict."""
        device = None
        for t in (loss_cls, loss_global_alignment, loss_local_alignment,
                  loss_local_contrastive, loss_text_refinement, loss_ood_reg,
                  loss_distill_img, loss_distill_text):
            if isinstance(t, torch.Tensor):
                device = t.device
                break

        total = torch.zeros((), device=device) if device is not None else torch.zeros(())
        breakdown: Dict[str, float] = {
            "loss_cls": 0.0,
            "loss_global_alignment": 0.0,
            "loss_local_alignment": 0.0,
            "loss_local_contrastive": 0.0,
            "loss_text_refinement": 0.0,
            "loss_ood_reg": 0.0,
            "loss_distill_img": 0.0,
            "loss_distill_text": 0.0,
        }

        def add(key: str, val: Optional[torch.Tensor], w: float) -> None:
            nonlocal total
            if val is None or w == 0.0:
                return
            total = total + w * val
            breakdown[key] = float(val.detach().item())

        add("loss_cls", loss_cls, self.weight_cls)
        add("loss_global_alignment", loss_global_alignment, self.weight_global_alignment)
        add("loss_local_alignment", loss_local_alignment, self.weight_local_alignment)
        add("loss_local_contrastive", loss_local_contrastive, self.weight_local_contrastive)
        add("loss_text_refinement", loss_text_refinement, self.weight_text_refinement)
        add("loss_ood_reg", loss_ood_reg, self.weight_ood_reg)
        add("loss_distill_img", loss_distill_img, self.weight_distill_img)
        add("loss_distill_text", loss_distill_text, self.weight_distill_text)

        return {"total_loss": total, **breakdown}


def entropy_select_topk(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Glali-style OOD entropy regularization on non-topK local patches.

    Flattens ``local_logits`` (shape [B, P, C]) to [B*P, C], computes
    per-patch softmax, and for each patch checks whether its top-``k``
    predictions contain the ground-truth class. Patches that do NOT
    contain the GT class in their top-k are considered "OOD / irrelevant"
    regions — maximizing their entropy pushes those patches toward a
    uniform (agnostic) distribution, which empirically improves OOD
    discriminability at test time (LoCoOp / GL-MCM family).

    Args:
        local_logits: [B, P, C] per-patch logits.
        labels: [B] ground-truth class indices.
        top_k: number of top classes to consider as "belongs to GT".

    Returns:
        Scalar loss (mean entropy of selected patches, negated so
        *minimizing* the returned value *maximizes* entropy).
    """
    if local_logits.dim() != 3:
        raise ValueError(f"local_logits must be [B,P,C], got {tuple(local_logits.shape)}")
    B, P, C = local_logits.shape
    if labels.min().item() < 0 or labels.max().item() >= C:
        raise ValueError(
            f"entropy_select_topk: labels out of range [0,{C}) "
            f"min={labels.min().item()} max={labels.max().item()}. "
            "Ensure labels are class indices into local_logits last dim."
        )
    logits_flat = local_logits.reshape(B * P, C)
    probs = torch.softmax(logits_flat, dim=-1)
    label_repeat = labels.repeat_interleave(P)                       # [B*P]
    top_k = max(1, min(int(top_k), C))
    topk_idx = torch.topk(probs, k=top_k, dim=1).indices              # [B*P, k]
    contains_label = topk_idx.eq(label_repeat.unsqueeze(1)).any(dim=1)  # [B*P]
    # Keep patches that do NOT contain GT → non-topK / OOD-like regions
    selected = probs[~contains_label]
    if selected.numel() == 0:
        return local_logits.new_zeros(())
    # Maximize entropy → minimize negative entropy
    entropy = -(selected * torch.log(selected + 1e-5)).sum(dim=-1).mean()
    # Returning *negative* entropy so that the optimizer, which minimizes
    # the total loss, effectively maximizes the entropy of non-top-k
    # patches (matches glali sign convention).
    return -entropy
