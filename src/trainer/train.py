# -*- coding: utf-8 -*-
"""Training loop for few-shot learning."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Any,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: GLOCAL-FSL-OOD model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device to train on.
        epoch: Current epoch number.
        config: Config object.
        logger: Optional logger.

    Returns:
        Dict of training metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, return_loss=True, labels=labels)
        logits = outputs["logits"]
        loss_contrastive = outputs.get("loss_contrastive")

        # Classification loss
        loss_cls = loss_fn(logits, labels)

        # Total loss with contrastive term
        if loss_contrastive is not None:
            loss = loss_cls + 0.5 * loss_contrastive
        else:
            loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    if logger:
        logger.info(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    return {"loss": avg_loss, "accuracy": accuracy}
