#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end training entry point for GLOCAL-FSL-OOD.

Wires together:
    1. Config loading (``configs/default.yaml`` + experiment override)
    2. Dataset loading from ``manifest.csv`` (ID / OOD split, auto
       stratified train/val/test split if split files are missing)
    3. Model construction (``GLocalFSLOODModel`` with teacher encoder)
    4. Trainer orchestrator (optimizer, scheduler, AMP, checkpointing)
    5. Optional end-of-run test + OOD evaluation

Usage::

    python src/scripts/train_fsl.py \\
        --config configs/experiment/exp_full_model.yaml

    python src/scripts/train_fsl.py \\
        --config configs/experiment/exp_full_model.yaml \\
        --override configs/train/fewshot_4shot.yaml \\
        --do-test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.datasets.bone_xray_dataset import BoneXRayDataset
from src.models.framework.glocal_fsl_ood_model import GLocalFSLOODModel
from src.trainer.trainer import Trainer
from src.utils.config import Config, load_config
from src.utils.logger import TensorBoardLogger, setup_logging
from src.utils.seed import set_seed


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


def _merge_configs(base: Config, override: Config) -> Config:
    def _merge(a, b):
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out
    return Config(_merge(base.to_dict(), override.to_dict()))


def _load_descriptions(config: Any, logger) -> Dict[str, List[str]]:
    """Load LLM descriptions (YAML preferred, JSON fallback)."""
    y = _get(config, "llm_descriptions", "output_file",
             default="data/prompts/class_descriptions.yaml")
    if os.path.exists(y):
        with open(y) as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Loaded descriptions (YAML): {y}")
        return data
    j = _get(config, "llm_descriptions", "glali_output_file",
             default="data/prompts/class_descriptions.json")
    if os.path.exists(j):
        import json
        with open(j) as f:
            data = json.load(f) or {}
        logger.info(f"Loaded descriptions (JSON): {j}")
        return data
    logger.warning("No descriptions file found — proceeding with class names only.")
    return {}


def _resolve_manifest(config: Any) -> str:
    cand = [
        _get(config, "data", "manifest_file"),
        "data/processed/data_processing/manifest.csv",
        "data/processed/image_processing/manifest.csv",
    ]
    for c in cand:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Manifest CSV not found. Run src/scripts/splits_dataset.py first "
        "(or set data.manifest_file in your config)."
    )


def _stratified_split(
    dataset: BoneXRayDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Stratified split by class index."""
    by_class: Dict[int, List[int]] = {}
    for i, (_, lbl, _) in enumerate(dataset.samples):
        by_class.setdefault(lbl, []).append(i)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _lbl, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def _build_loaders(config: Any, logger) -> Dict[str, Optional[DataLoader]]:
    manifest = _resolve_manifest(config)
    class_names = _get(config, "data", "class_names", default=None)
    id_classes = _get(config, "data", "id_classes", default=None)
    ood_classes = _get(config, "data", "ood_classes", default=None)
    image_size = int(_get(config, "data", "image_size", default=224))
    batch_size = int(_get(config, "train", "batch_size", default=4))
    num_workers = int(_get(config, "data", "num_workers", default=4))
    pin = bool(_get(config, "data", "pin_memory", default=True))
    seed = int(_get(config, "experiment", "seed", default=42))

    # --- Load full ID dataset (one object, two transforms)
    id_train = BoneXRayDataset(
        manifest_file=manifest, split="train",
        class_names=class_names, id_classes=id_classes, ood_classes=ood_classes,
        mode="id", image_size=image_size,
        transform=BoneXRayDataset.get_default_transform("train", image_size),
    )
    id_eval = BoneXRayDataset(
        manifest_file=manifest, split="test",
        class_names=class_names, id_classes=id_classes, ood_classes=ood_classes,
        mode="id", image_size=image_size,
        transform=BoneXRayDataset.get_default_transform("test", image_size),
    )
    ood_eval = BoneXRayDataset(
        manifest_file=manifest, split="test",
        class_names=class_names, id_classes=id_classes, ood_classes=ood_classes,
        mode="ood", image_size=image_size,
        transform=BoneXRayDataset.get_default_transform("test", image_size),
    )

    # Stratified train/val/test split on the ID set (shared indices for
    # training transforms vs eval transforms → same indices, different
    # dataset objects per Subset).
    val_ratio = float(_get(config, "data", "val_ratio", default=0.15))
    test_ratio = float(_get(config, "data", "test_ratio", default=0.15))
    train_sub, _, _ = _stratified_split(id_train, val_ratio, test_ratio, seed=seed)
    _, val_sub, test_sub = _stratified_split(id_eval, val_ratio, test_ratio, seed=seed)

    # Few-shot k-shot capping for the training subset (lightweight)
    k_shot = int(_get(config, "fewshot", "k_shot", default=0))
    if k_shot > 0:
        train_sub = _cap_kshot(id_train, train_sub, k_shot, seed=seed)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_sub, batch_size=max(1, batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_sub, batch_size=max(1, batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    ood_loader = DataLoader(
        ood_eval, batch_size=max(1, batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    ) if len(ood_eval) > 0 else None

    logger.info(
        f"Dataset built | manifest={manifest} | "
        f"train={len(train_sub)} val={len(val_sub)} test={len(test_sub)} "
        f"ood={len(ood_eval) if ood_eval else 0}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader, "ood": ood_loader}


def _cap_kshot(full_ds: BoneXRayDataset, subset: Subset, k_shot: int, seed: int = 42) -> Subset:
    """Cap the training subset to k_shot samples per ID class."""
    rng = np.random.default_rng(seed)
    per_class: Dict[int, List[int]] = {}
    for i in subset.indices:
        _, lbl, _ = full_ds.samples[i]
        per_class.setdefault(lbl, []).append(int(i))
    capped: List[int] = []
    for lbl, idxs in per_class.items():
        rng.shuffle(idxs)
        capped.extend(idxs[:k_shot])
    return Subset(full_ds, capped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GLOCAL-FSL-OOD model")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--override", default=None, help="Optional override config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint file to resume from")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--do-test", action="store_true", help="Run test after training")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only run test")
    args = parser.parse_args()

    # ---- config ----
    config = load_config(args.config)
    if args.override:
        config = _merge_configs(config, load_config(args.override))

    seed = int(_get(config, "experiment", "seed", default=42))
    set_seed(seed, deterministic=bool(_get(config, "experiment", "deterministic", default=False)))

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # ---- logging ----
    exp_name = _get(config, "experiment_name",
                    default=_get(config, "experiment", "name", default="default"))
    log_root = _get(config, "paths", "log_dir", default="outputs/logs")
    log_dir = os.path.join(log_root, str(exp_name))
    logger = setup_logging(log_dir, level=str(_get(config, "logging", "level", default="INFO")))
    tb_logger = TensorBoardLogger(_get(config, "paths", "tb_dir", default="outputs/tensorboard"), str(exp_name))

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    if args.override:
        logger.info(f"Override: {args.override}")

    # ---- data ----
    loaders = _build_loaders(config, logger)

    # ---- descriptions ----
    descriptions = _load_descriptions(config, logger)

    # ---- model ----
    class_names = _get(config, "data", "class_names", default=[])
    model = GLocalFSLOODModel(
        config=config, class_names=class_names, descriptions=descriptions, device=device,
    ).to(device)
    model.freeze_encoders()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: total={total:,} trainable={trainable:,}")

    # ---- trainer ----
    trainer = Trainer(
        model=model, config=config, device=device,
        train_loader=loaders["train"], val_loader=loaders["val"],
        test_loader=loaders["test"], ood_loader=loaders["ood"],
        logger=logger, tb_logger=tb_logger,
    )

    if args.resume:
        trainer.resume(args.resume)

    # ---- run ----
    if not args.eval_only:
        trainer.train()

    if args.do_test or args.eval_only:
        results = trainer.test(use_best=True)
        logger.info(f"Test results: {results}")

    tb_logger.close()


if __name__ == "__main__":
    main()
