# -*- coding: utf-8 -*-
"""Test loop for few-shot learning."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Any,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Test the model.

    Args:
        model: GLOCAL-FSL-OOD model.
        test_loader: Test data loader.
        device: Device.
        config: Config object.
        logger: Optional logger.

    Returns:
        Dict of test metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)

            outputs = model(images, return_loss=False)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_scores.extend(probs.cpu().tolist())

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    if logger:
        logger.info(f"Test - Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
        "scores": all_scores,
    }
