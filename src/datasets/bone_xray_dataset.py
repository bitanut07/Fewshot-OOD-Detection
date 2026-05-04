# -*- coding: utf-8 -*-
"""Bone X-Ray dataset implementation (BTXRD + FracAtlas).

Loads samples directly from the processed manifest CSV produced by
``src/scripts/splits_dataset.py``:

    name_id, data, path, label, class

``class`` is either ``"id"`` (in-distribution) or ``"ood"``
(out-of-distribution). ``label`` is the raw tumor/condition name
(e.g. ``"osteosarcoma"``, ``"fractured"``, ``"no_tumor"``).

Each sample is mapped to an integer class index using the ordered
``class_names`` list. Known / OOD classes are derived from the
configured ``id_classes`` / ``ood_classes`` index lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


DEFAULT_CLASS_NAMES = [
    "giant cell tumor",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "simple bone cyst",
    "synovial osteochondroma",
    "no_tumor",
    "fractured",
]
DEFAULT_ID_CLASSES = list(range(6))
DEFAULT_OOD_CLASSES = [6, 7]

# ImageNet stats (legacy / non-CLIP pipelines)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI CLIP / open_clip preprocessing (matches reference glali INPUT.PIXEL_*)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BoneXRayDataset(Dataset):
    """
    Bone X-Ray dataset (BTXRD + FracAtlas) loaded from manifest CSV.

    Args:
        manifest_file: Path to manifest CSV produced by ``splits_dataset.py``.
        split: One of ``"train" | "val" | "test"``. When a ``split_file``
               is provided, only ``name_id`` listed in it are kept.
        split_file: Optional text file with one ``name_id`` per line.
        class_names: Ordered list of all class names (index order).
        id_classes: Indices considered in-distribution.
        ood_classes: Indices considered OOD.
        mode: ``"id"`` → keep only id samples, ``"ood"`` → keep only ood,
              ``"all"`` → keep both.
        transform: torchvision transforms applied to each image.
        image_size: Image size (for default transform).
        cache_images: If True, cache decoded PIL images in memory.
    """

    def __init__(
        self,
        manifest_file: str | Path,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        class_names: Optional[List[str]] = None,
        id_classes: Optional[List[int]] = None,
        ood_classes: Optional[List[int]] = None,
        mode: str = "id",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        cache_images: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_file = Path(manifest_file)
        self.split = split
        self.mode = mode.lower()
        self.class_names: List[str] = class_names or DEFAULT_CLASS_NAMES
        self.id_classes: List[int] = list(id_classes) if id_classes is not None else DEFAULT_ID_CLASSES
        self.ood_classes: List[int] = list(ood_classes) if ood_classes is not None else DEFAULT_OOD_CLASSES
        self.num_classes = len(self.class_names)
        self.cache_images = cache_images
        self._cache: dict[str, Image.Image] = {}

        # Transform
        self.transform = transform or self.get_default_transform(split, image_size)

        # Label → index mapping
        self._label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Load manifest
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")
        df = pd.read_csv(self.manifest_file)
        required = {"name_id", "path", "label", "class"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["class"] = df["class"].astype(str).str.strip().str.lower()

        # Filter by split_file if provided
        if split_file is not None:
            sf = Path(split_file)
            if sf.exists():
                with open(sf) as f:
                    keep = {ln.strip() for ln in f if ln.strip()}
                df = df[df["name_id"].astype(str).isin(keep)].reset_index(drop=True)

        # Map label → index; drop unknowns
        df["label_idx"] = df["label"].map(lambda l: self._label_to_idx.get(l, -1))
        df = df[df["label_idx"] >= 0].reset_index(drop=True)

        # Filter by mode (id / ood / all)
        if self.mode == "id":
            df = df[df["label_idx"].isin(self.id_classes)].reset_index(drop=True)
        elif self.mode == "ood":
            df = df[df["label_idx"].isin(self.ood_classes)].reset_index(drop=True)
        elif self.mode == "all":
            pass
        else:
            raise ValueError(f"Unknown mode: {self.mode} (use id|ood|all)")

        self.df = df
        self.samples: List[Tuple[str, int, int]] = [
            (str(r["path"]), int(r["label_idx"]), int(r["label_idx"] in self.id_classes))
            for _, r in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        if self.cache_images and path in self._cache:
            return self._cache[path]
        img = Image.open(path).convert("RGB")
        if self.cache_images:
            self._cache[path] = img
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label, _is_id = self.samples[idx]
        img = self._load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ---- helpers ----
    def filter_known(self) -> "BoneXRayDataset":
        self.samples = [s for s in self.samples if s[1] in self.id_classes]
        return self

    def filter_ood(self) -> "BoneXRayDataset":
        self.samples = [s for s in self.samples if s[1] in self.ood_classes]
        return self

    def get_class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, lbl, _ in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_id_class_names(self) -> List[str]:
        return [self.class_names[i] for i in self.id_classes]

    def get_ood_class_names(self) -> List[str]:
        return [self.class_names[i] for i in self.ood_classes]

    @staticmethod
    def _to_norm_tuples(
        mean: Optional[Union[Sequence[float], Tuple[float, ...]]],
        std: Optional[Union[Sequence[float], Tuple[float, ...]]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Resolve normalization; default to OpenAI CLIP stats for open_clip models."""
        if mean is not None and std is not None:
            m = tuple(float(x) for x in mean)
            s = tuple(float(x) for x in std)
            if len(m) != 3 or len(s) != 3:
                raise ValueError("image_mean and image_std must each have length 3")
            return m, s
        return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    @staticmethod
    def get_default_transform(
        split: str = "train",
        image_size: int = 224,
        mean: Optional[Union[Sequence[float], Tuple[float, ...]]] = None,
        std: Optional[Union[Sequence[float], Tuple[float, ...]]] = None,
    ) -> T.Compose:
        """Build resize / augment / normalize pipeline.

        When ``mean`` and ``std`` are omitted, uses OpenAI CLIP normalization so
        inputs match ``open_clip.create_model_and_transforms`` (ViT-B/16, etc.).
        """
        m, s = BoneXRayDataset._to_norm_tuples(mean, std)
        normalize = T.Normalize(mean=m, std=s)
        if split == "train":
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                normalize,
            ])
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ])
