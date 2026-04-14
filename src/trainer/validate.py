# -*- coding: utf-8 -*-
"""Validation loop for few-shot learning."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Any,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: GLOCAL-FSL-OOD model.
        val_loader: Validation data loader.
        loss_fn: Loss function.
        device: Device.
        epoch: Current epoch number.
        config: Config object.
        logger: Optional logger.

    Returns:
        Dict of validation metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, return_loss=False)
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    if logger:
        logger.info(f"Val Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    return {"val_loss": avg_loss, "val_accuracy": accuracy}
