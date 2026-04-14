# -*- coding: utf-8 -*-
"""Unified evaluator for classification and OOD detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics_cls import ClassificationMetrics
from .metrics_ood import OODMetrics


class Evaluator:
    """
    Unified evaluation interface for classification and OOD detection.

    Args:
        model: Model to evaluate.
        device: Device.
        config: Config object.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.mode = config.eval.mode  # "fsl", "ood", or "both"

    def evaluate_cls(
        self,
        data_loader: DataLoader,
        num_classes: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate classification performance.

        Args:
            data_loader: Data loader for evaluation.
            num_classes: Number of classes.

        Returns:
            Dict of classification metrics.
        """
        self.model.eval()
        metrics = ClassificationMetrics(num_classes=num_classes)

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Classification Eval"):
                images = images.to(self.device)
                outputs = self.model(images, return_loss=False)
                logits = outputs["logits"]
                metrics.update(logits, labels)

        return metrics.compute()

    def evaluate_ood(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        method: str = "msp",
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance.

        Args:
            id_loader: Data loader for in-distribution samples.
            ood_loader: Data loader for OOD samples.
            method: OOD scoring method.

        Returns:
            Dict of OOD metrics.
        """
        self.model.eval()
        metrics = OODMetrics(method=method)

        with torch.no_grad():
            for images, _ in tqdm(id_loader, desc="OOD Eval (ID)"):
                images = images.to(self.device)
                outputs = self.model(images, return_loss=False)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                scores = probs.max(dim=1).values
                metrics.update(scores, torch.zeros_like(scores))

            for images, _ in tqdm(ood_loader, desc="OOD Eval (OOD)"):
                images = images.to(self.device)
                outputs = self.model(images, return_loss=False)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                scores = probs.max(dim=1).values
                metrics.update(torch.zeros_like(scores), scores)

        return metrics.compute()

    def evaluate(
        self,
        id_loader: Optional[DataLoader] = None,
        ood_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        num_classes: int = 5,
    ) -> Dict[str, Any]:
        """
        Run evaluation based on config mode.

        Returns:
            Dict with all computed metrics.
        """
        results = {}
        mode = self.mode

        if mode in ("fsl", "both") and test_loader is not None:
            results["classification"] = self.evaluate_cls(test_loader, num_classes)

        if mode in ("ood", "both") and id_loader is not None and ood_loader is not None:
            method = self.config.eval.ood.method
            results["ood"] = self.evaluate_ood(id_loader, ood_loader, method)

        return results
