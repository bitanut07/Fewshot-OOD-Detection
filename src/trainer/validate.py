# -*- coding: utf-8 -*-
"""Validation loop for GLOCAL-FSL-OOD.

Computes the primary classification metrics on an ID validation loader
and — when an OOD loader is provided — also reports OOD detection
scores (AUROC / AUPR / FPR@95) using the MSP baseline. This mirrors
how glali evaluates mid-training (val accuracy) while also providing a
lightweight OOD sanity check useful for early stopping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics_cls import ClassificationMetrics
from src.evaluation.metrics_ood import OODMetrics
from src.losses.classification_loss import ClassificationLoss


def _get(cfg: Any, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: Optional[nn.Module],
    device: torch.device,
    epoch: int,
    config: Any,
    logger: Optional[Any] = None,
    ood_loader: Optional[DataLoader] = None,
    tb_logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Validate the model on ID and (optionally) OOD loaders."""
    model.eval()

    loss_fn = loss_fn or ClassificationLoss()
    num_id_classes = len(_get(config, "data", "id_classes", default=[0, 1, 2, 3, 4, 5]))
    metrics_cls = ClassificationMetrics(num_classes=num_id_classes)

    total_loss = 0.0
    num_batches = 0
    id_scores = []

    if val_loader is None:
        if logger:
            logger.warning(f"Epoch {epoch} [val] skipped (no val_loader)")
        return {"val_loss": float("nan"), "val_accuracy": 0.0}

    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
        images, labels = _unpack(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images, return_loss=False)
        logits = out["logits"]

        # Restrict to ID class indices for classification metric
        id_indices = torch.as_tensor(
            _get(config, "data", "id_classes", default=list(range(num_id_classes))),
            device=logits.device, dtype=torch.long,
        )
        id_logits = logits.index_select(1, id_indices)

        # Relabel (labels are already 0..C-1 with C = total classes;
        # re-index to 0..num_id_classes-1 for metric computation).
        remap = {int(c.item()): i for i, c in enumerate(id_indices)}
        id_labels = torch.tensor([remap.get(int(l.item()), 0) for l in labels], device=labels.device)

        loss = loss_fn(id_logits, id_labels)
        total_loss += float(loss.item())
        num_batches += 1
        metrics_cls.update(id_logits, id_labels)

        # MSP score over FULL logits (includes ood class indices) → ID samples
        probs = F.softmax(logits, dim=1)
        id_scores.append(probs.max(dim=1).values.detach().cpu().numpy())

    cls_results = metrics_cls.compute()
    avg_loss = total_loss / max(1, num_batches)

    result: Dict[str, float] = {
        "val_loss": avg_loss,
        "val_accuracy": float(cls_results.get("accuracy", 0.0)),
        "val_f1": float(cls_results.get("f1_score", 0.0)),
        "val_precision": float(cls_results.get("precision", 0.0)),
        "val_recall": float(cls_results.get("recall", 0.0)),
        "val_auroc": float(cls_results.get("auroc", 0.0)),
    }

    # Optional quick OOD metrics
    if ood_loader is not None and len(id_scores):
        id_scores_np = np.concatenate(id_scores, axis=0)
        ood_scores = []
        for batch in tqdm(ood_loader, desc=f"Epoch {epoch} [val-ood]", leave=False):
            images, _ = _unpack(batch)
            images = images.to(device, non_blocking=True)
            out = model(images, return_loss=False)
            probs = F.softmax(out["logits"], dim=1)
            ood_scores.append(probs.max(dim=1).values.detach().cpu().numpy())
        ood_scores_np = np.concatenate(ood_scores, axis=0)

        metrics_ood = OODMetrics(method=_get(config, "ood", "method", default="msp"))
        metrics_ood.update(
            torch.as_tensor(id_scores_np), torch.as_tensor(ood_scores_np),
        )
        ood_res = metrics_ood.compute()
        for k, v in ood_res.items():
            result[f"val_ood_{k}"] = float(v)

    if logger:
        msg = " | ".join(f"{k}={v:.4f}" for k, v in result.items())
        logger.info(f"Epoch {epoch} [val] {msg}")

    if tb_logger is not None:
        for k, v in result.items():
            tb_logger.log_scalar(f"val/{k}", float(v), epoch)

    return result


def _unpack(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        return batch["img"], batch["label"]
    raise TypeError(f"Unsupported batch type: {type(batch)}")
