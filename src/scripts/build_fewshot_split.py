#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build few-shot train/val/test splits from full dataset.

Splits the dataset into known classes (for training) and OOD classes (for testing).
Ensures each class has enough samples for k_shot support + n_query query.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config


def build_splits(
    data_root: str,
    class_names: list,
    known_classes: list,
    ood_classes: list,
    n_way: int,
    k_shot: int,
    n_query: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = "data/splits",
):
    """
    Build few-shot splits for the dataset.

    Args:
        data_root: Root directory of images.
        class_names: List of class names.
        known_classes: List of known class indices.
        ood_classes: List of OOD class indices.
        n_way: Number of classes per episode.
        k_shot: Support samples per class.
        n_query: Query samples per class.
        train_ratio: Fraction of known samples for training.
        val_ratio: Fraction of known samples for validation.
        seed: Random seed.
        output_dir: Output directory for split files.
    """
    random.seed(seed)

    # TODO: Implement actual split building based on image directory structure
    # This is a placeholder that generates empty split files

    os.makedirs(output_dir, exist_ok=True)

    for split_name, classes in [("train", known_classes), ("val", known_classes), ("test", known_classes + ood_classes)]:
        output_file = Path(output_dir) / f"bone_xray_{split_name}.txt"
        with open(output_file, "w") as f:
            # TODO: Replace with actual image paths from data_root
            # Format: image_path label (one per line)
            pass

    print(f"Splits saved to: {output_dir}")
    print("TODO: Implement actual image path discovery from data_root")


def main():
    parser = argparse.ArgumentParser(description="Build few-shot dataset splits")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/splits")
    args = parser.parse_args()

    config = load_config(args.config)
    build_splits(
        data_root=config.paths.data_root,
        class_names=config.data.get("class_names", []),
        known_classes=config.data.get("known_classes", list(range(10))),
        ood_classes=config.data.get("ood_classes", list(range(10, 15))),
        n_way=config.fewshot.n_way,
        k_shot=config.fewshot.k_shot,
        n_query=config.fewshot.n_query,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=config.experiment.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
