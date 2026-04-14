# -*- coding: utf-8 -*-
"""OOD detection metrics."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class OODMetrics:
    """
    Compute OOD detection metrics: AUROC, AUPR-In, AUPR-Out, FPR@95.

    Args:
        method: OOD scoring method (msp, energy, cosine, maha).
    """

    def __init__(self, method: str = "msp") -> None:
        self.method = method
        self.reset()

    def reset(self) -> None:
        self.ood_scores_id: List[float] = []   # Scores for in-distribution samples
        self.ood_scores_ood: List[float] = []  # Scores for OOD samples

    def update(
        self,
        id_scores: torch.Tensor,
        ood_scores: torch.Tensor,
    ) -> None:
        self.ood_scores_id.extend(id_scores.cpu().tolist())
        self.ood_scores_ood.extend(ood_scores.cpu().tolist())

    def compute(self) -> Dict[str, float]:
        id_scores = np.array(self.ood_scores_id)
        ood_scores = np.array(self.ood_scores_ood)
        labels = np.concatenate([
            np.ones(len(id_scores)),
            np.zeros(len(ood_scores)),
        ])
        scores = np.concatenate([id_scores, ood_scores])

        # AUROC
        auroc = 0.0
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.0

        # AUPR-In (ID as positive)
        aupr_in = 0.0
        try:
            prec, rec, _ = precision_recall_curve(labels, scores)
            aupr_in = auc(rec, prec)
        except ValueError:
            aupr_in = 0.0

        # AUPR-Out (OOD as positive)
        aupr_out = 0.0
        try:
            prec, rec, _ = precision_recall_curve(1 - labels, -scores)
            aupr_out = auc(rec, prec)
        except ValueError:
            aupr_out = 0.0

        # FPR@95 (FPR when TPR=95%)
        fpr95 = 1.0
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels, scores)
            idx = np.searchsorted(tpr, 0.95)
            if idx < len(fpr):
                fpr95 = fpr[idx]
        except ValueError:
            fpr95 = 1.0

        return {
            "auroc": auroc,
            "aupr_in": aupr_in,
            "aupr_out": aupr_out,
            "fpr95": fpr95,
        }
