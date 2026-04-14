# -*- coding: utf-8 -*-
"""Base dataset class with image loading and augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BaseDataset(Dataset):
    """
    Base dataset class for medical image classification.

    Handles image loading, transforms, and basic dataset operations.
    """

    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.cache_images = cache_images
        self.cache: dict[str, Image.Image] = {}
        self.samples: List[Tuple[str, int]] = []

        if split_file and Path(split_file).exists():
            self._load_split_file(split_file)
        else:
            self._discover_samples()

        self.class_names = class_names or []
        self.num_classes = len(self.class_names) if self.class_names else 0

    def _load_split_file(self, split_file: str) -> None:
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_path, label = parts[0], int(parts[1])
                else:
                    img_path = parts[0]
                    label = 0
                self.samples.append((str(self.data_root / img_path), label))

    def _discover_samples(self) -> None:
        for img_path in sorted(self.data_root.rglob("*.jpg")):
            self.samples.append((str(img_path), 0))

    def _load_image(self, path: str) -> Image.Image:
        if self.cache_images and path in self.cache:
            return self.cache[path]
        img = Image.open(path).convert("RGB")
        if self.cache_images:
            self.cache[path] = img
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_samples(self, class_idx: int) -> List[Tuple[str, int]]:
        return [(p, l) for p, l in self.samples if l == class_idx]
