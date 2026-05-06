# -*- coding: utf-8 -*-
"""Test loop for GLOCAL-FSL-OOD.

Runs classification on the ID test loader and, when available, OOD
detection on the OOD test loader. Supports multiple OOD scoring
methods (MSP, GL-MCM, local-MCM) matching what glali reports at
eval time:

    - MSP (global): -max(softmax(global_logits))
    - GL-MCM (combined): -max(softmax(combined_logits))
    - local-MCM: -max(softmax(local_logits_per_patch))
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
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Any,
    logger: Optional[Any] = None,
    ood_loader: Optional[DataLoader] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run test-time evaluation on ID (and optionally OOD) loaders."""
    model.eval()

    num_id_classes = len(_get(config, "data", "id_classes", default=[0, 1, 2, 3, 4, 5]))
    id_indices = torch.as_tensor(
        _get(config, "data", "id_classes", default=list(range(num_id_classes))),
        dtype=torch.long, device=device,
    )

    metrics_cls = ClassificationMetrics(num_classes=num_id_classes)

    all_preds, all_labels = [], []
    id_scores_msp, id_scores_glmcm, id_scores_loc = [], [], []

    for batch in tqdm(test_loader, desc="Test [id]", leave=False):
        images, labels = _unpack(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images, return_loss=False)
        combined = out["logits"]            # [B, C]
        global_ = out.get("global_logits")   # [B, C] or None
        local_ = out.get("local_logits")     # [B, P, C] or None

        # Classification on restricted ID logits
        id_logits = combined.index_select(1, id_indices)
        remap = {int(c.item()): i for i, c in enumerate(id_indices)}
        id_labels = torch.tensor(
            [remap.get(int(l.item()), 0) for l in labels], device=labels.device,
        )
        metrics_cls.update(id_logits, id_labels)

        preds = id_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(id_labels.cpu().tolist())

        # OOD scores (on full logits)
        probs_g = F.softmax((global_ if global_ is not None else combined) / temperature, dim=1)
        probs_c = F.softmax(combined / temperature, dim=1)
        id_scores_msp.append(probs_g.max(dim=1).values.cpu().numpy())
        id_scores_glmcm.append(probs_c.max(dim=1).values.cpu().numpy())
        if local_ is not None:
            probs_l = F.softmax(local_ / temperature, dim=-1)
            id_scores_loc.append(probs_l.amax(dim=(1, 2)).cpu().numpy())

    cls_results = metrics_cls.compute()

    result: Dict[str, Any] = {
        "accuracy": float(cls_results.get("accuracy", 0.0)),
        "precision": float(cls_results.get("precision", 0.0)),
        "recall": float(cls_results.get("recall", 0.0)),
        "f1_score": float(cls_results.get("f1_score", 0.0)),
        "auroc": float(cls_results.get("auroc", 0.0)),
        "predictions": all_preds,
        "labels": all_labels,
    }

    # OOD evaluation
    if ood_loader is not None:
        id_msp = np.concatenate(id_scores_msp, axis=0) if id_scores_msp else np.array([])
        id_glmcm = np.concatenate(id_scores_glmcm, axis=0) if id_scores_glmcm else np.array([])
        id_loc = np.concatenate(id_scores_loc, axis=0) if id_scores_loc else np.array([])

        ood_msp, ood_glmcm, ood_loc = [], [], []
        for batch in tqdm(ood_loader, desc="Test [ood]", leave=False):
            images, _ = _unpack(batch)
            images = images.to(device, non_blocking=True)
            out = model(images, return_loss=False)
            combined = out["logits"]
            global_ = out.get("global_logits")
            local_ = out.get("local_logits")

            probs_g = F.softmax((global_ if global_ is not None else combined) / temperature, dim=1)
            probs_c = F.softmax(combined / temperature, dim=1)
            ood_msp.append(probs_g.max(dim=1).values.cpu().numpy())
            ood_glmcm.append(probs_c.max(dim=1).values.cpu().numpy())
            if local_ is not None:
                probs_l = F.softmax(local_ / temperature, dim=-1)
                ood_loc.append(probs_l.amax(dim=(1, 2)).cpu().numpy())

        ood_msp = np.concatenate(ood_msp, axis=0) if ood_msp else np.array([])
        ood_glmcm = np.concatenate(ood_glmcm, axis=0) if ood_glmcm else np.array([])
        ood_loc = np.concatenate(ood_loc, axis=0) if ood_loc else np.array([])

        result["ood"] = {}
        for tag, id_s, ood_s in (
            ("msp", id_msp, ood_msp),
            ("glmcm", id_glmcm, ood_glmcm),
            ("local_mcm", id_loc, ood_loc),
        ):
            if id_s.size == 0 or ood_s.size == 0:
                continue
            m = OODMetrics(method=tag)
            m.update(torch.as_tensor(id_s), torch.as_tensor(ood_s))
            result["ood"][tag] = {k: float(v) for k, v in m.compute().items()}

    if logger:
        logger.info(
            f"Test [id] acc={result['accuracy']:.4f} f1={result['f1_score']:.4f} "
            f"prec={result['precision']:.4f} rec={result['recall']:.4f}"
        )
        if "ood" in result:
            for tag, d in result["ood"].items():
                logger.info(f"Test [ood/{tag}] " + " ".join(f"{k}={v:.4f}" for k, v in d.items()))

    return result


def _unpack(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        return batch["img"], batch["label"]
    raise TypeError(f"Unsupported batch type: {type(batch)}")
