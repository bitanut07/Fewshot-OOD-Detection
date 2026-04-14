# -*- coding: utf-8 -*-
"""Bone X-Ray dataset implementation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import torchvision.transforms as T
from .base_dataset import BaseDataset


class BoneXRayDataset(BaseDataset):
    """
    Bone X-Ray (RSNA) dataset for few-shot medical image classification.

    Supports known classes (in-distribution) and OOD classes (out-of-distribution).
    """

    CLASS_NAMES = [
        "No Finding", "Fracture", "Osteoarthritis", "Bone Lesion",
        "Osteoporosis", "Degenerative Joint Disease", "Spondylosis",
        "Disc Degeneration", "Spinal Stenosis", "Scoliosis",
        "Osteomyelitis", "Bone Tumor", "Metastatic Bone Disease",
        "Avascular Necrosis", "Paget Disease",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        known_classes: Optional[List[int]] = None,
        ood_classes: Optional[List[int]] = None,
        transform: Optional[T.Compose] = None,
        cache_images: bool = False,
    ) -> None:
        self.split = split
        self.known_classes = known_classes or list(range(10))
        self.ood_classes = ood_classes or list(range(10, 15))

        split_file = Path(data_root).parent / "splits" / f"bone_xray_{split}.txt"

        super().__init__(
            data_root=data_root,
            split_file=str(split_file) if split_file.exists() else None,
            class_names=self.CLASS_NAMES,
            transform=transform,
            cache_images=cache_images,
        )

    def filter_known(self) -> "BoneXRayDataset":
        self.samples = [(p, l) for p, l in self.samples if l in self.known_classes]
        return self

    def filter_ood(self) -> "BoneXRayDataset":
        self.samples = [(p, l) for p, l in self.samples if l in self.ood_classes]
        return self
