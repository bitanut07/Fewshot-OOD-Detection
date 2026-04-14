# -*- coding: utf-8 -*-
"""Classification metrics for few-shot learning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class ClassificationMetrics:
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1, AUROC.

    Args:
        num_classes: Number of classes.
    """

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.predictions: List[int] = []
        self.labels: List[int] = []
        self.scores: List[np.ndarray] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        preds = logits.argmax(dim=1).cpu().tolist()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        self.predictions.extend(preds)
        self.labels.extend(labels.tolist())
        self.scores.append(probs)

    def compute(self) -> Dict[str, float]:
        scores = np.concatenate(self.scores, axis=0)
        labels_arr = np.array(self.labels)
        preds_arr = np.array(self.predictions)

        acc = accuracy_score(labels_arr, preds_arr)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_arr, preds_arr, average="macro", zero_division=0
        )

        # AUROC (one-vs-rest)
        auroc = 0.0
        try:
            if scores.shape[1] > 1:
                auroc = roc_auc_score(
                    labels_arr, scores, multi_class="ovr", average="macro"
                )
        except ValueError:
            auroc = 0.0

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auroc": auroc,
        }
