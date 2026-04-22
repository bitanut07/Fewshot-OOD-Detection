#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation entry point for GLOCAL-FSL-OOD.

Runs:
    - ID classification metrics (accuracy, precision, recall, F1, AUROC)
    - OOD detection metrics (AUROC, AUPR-In, AUPR-Out, FPR@95) using
      three scoring methods in parallel: ``msp``, ``glmcm``, ``local_mcm``.

Examples::

    # Standard eval using best checkpoint from training run
    python src/scripts/eval_ood.py \\
        --config configs/experiment/exp_full_model.yaml \\
        --checkpoint outputs/runs/full_model/checkpoints/best.pt

    # Eval at a different softmax temperature (tuning)
    python src/scripts/eval_ood.py \\
        --config configs/experiment/exp_full_model.yaml \\
        --checkpoint outputs/runs/full_model/checkpoints/best.pt \\
        --temperature 0.5

    # Skip OOD detection (ID classification only)
    python src/scripts/eval_ood.py \\
        --config configs/experiment/exp_full_model.yaml \\
        --checkpoint outputs/runs/full_model/checkpoints/best.pt \\
        --no-ood
"""

from __future__ import annotations

import argparse
import json
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
from src.trainer.test import test as test_fn
from src.utils.checkpoint import load_checkpoint
from src.utils.config import Config, load_config
from src.utils.logger import setup_logging
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


def _resolve_manifest(config: Any) -> str:
    candidates = [
        _get(config, "data", "manifest_file"),
        "data/processed/data_processing/manifest.csv",
        "data/processed/image_processing/manifest.csv",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Manifest CSV not found. Set data.manifest_file in config or run "
        "src/scripts/splits_dataset.py first."
    )


def _load_descriptions(config: Any) -> Dict[str, List[str]]:
    y = _get(config, "llm_descriptions", "output_file",
             default="data/prompts/class_descriptions.yaml")
    if os.path.exists(y):
        with open(y) as f:
            return yaml.safe_load(f) or {}
    j = _get(config, "llm_descriptions", "glali_output_file",
             default="data/prompts/class_descriptions.json")
    if os.path.exists(j):
        with open(j) as f:
            return json.load(f) or {}
    return {}


def _stratified_test_subset(
    dataset: BoneXRayDataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Subset:
    """Take the deterministic test partition matching train_fsl.py."""
    by_class: Dict[int, List[int]] = {}
    for i, (_, lbl, _) in enumerate(dataset.samples):
        by_class.setdefault(lbl, []).append(i)

    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for _lbl, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(n * test_ratio))
        test_idx.extend(idxs[:n_test])
    return Subset(dataset, test_idx)


def _build_eval_loaders(
    config: Any,
    no_ood: bool,
    batch_size_override: Optional[int],
) -> Tuple[DataLoader, Optional[DataLoader]]:
    manifest = _resolve_manifest(config)
    class_names = _get(config, "data", "class_names", default=None)
    id_classes = _get(config, "data", "id_classes", default=None)
    ood_classes = _get(config, "data", "ood_classes", default=None)
    image_size = int(_get(config, "data", "image_size", default=224))
    num_workers = int(_get(config, "data", "num_workers", default=4))
    pin = bool(_get(config, "data", "pin_memory", default=True))
    seed = int(_get(config, "experiment", "seed", default=42))
    batch_size = int(batch_size_override or _get(config, "eval", "batch_size",
                                                 default=_get(config, "train", "batch_size", default=16)))

    transform_test = BoneXRayDataset.get_default_transform("test", image_size)

    id_eval_ds = BoneXRayDataset(
        manifest_file=manifest, split="test",
        class_names=class_names, id_classes=id_classes, ood_classes=ood_classes,
        mode="id", image_size=image_size, transform=transform_test,
    )
    val_ratio = float(_get(config, "data", "val_ratio", default=0.15))
    test_ratio = float(_get(config, "data", "test_ratio", default=0.15))
    id_test_sub = _stratified_test_subset(id_eval_ds, val_ratio, test_ratio, seed=seed)

    id_loader = DataLoader(
        id_test_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    ood_loader: Optional[DataLoader] = None
    if not no_ood:
        ood_ds = BoneXRayDataset(
            manifest_file=manifest, split="test",
            class_names=class_names, id_classes=id_classes, ood_classes=ood_classes,
            mode="ood", image_size=image_size, transform=transform_test,
        )
        if len(ood_ds) > 0:
            ood_loader = DataLoader(
                ood_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
            )
    return id_loader, ood_loader


def _format_results(result: Dict[str, Any]) -> str:
    lines = ["=" * 60, "  ID classification"]
    for k in ("accuracy", "precision", "recall", "f1_score", "auroc"):
        if k in result:
            lines.append(f"    {k:<10}: {float(result[k]):.4f}")
    if result.get("ood"):
        lines.append("-" * 60)
        lines.append("  OOD detection")
        for method, metrics in result["ood"].items():
            lines.append(f"    [{method}]")
            for k, v in metrics.items():
                lines.append(f"      {k:<10}: {float(v):.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _dump_json(result: Dict[str, Any], path: Path) -> None:
    to_save = {k: v for k, v in result.items() if k not in ("predictions", "labels")}
    to_save["predictions"] = result.get("predictions", [])
    to_save["labels"] = result.get("labels", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GLOCAL-FSL-OOD model")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--override", default=None, help="Optional override YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Eval batch size (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Softmax temperature for OOD scoring (default: ood.temperature or 1.0)")
    parser.add_argument("--no-ood", action="store_true",
                        help="Skip OOD detection (ID classification only)")
    parser.add_argument("--save-json", default=None,
                        help="Path to save results JSON (default: outputs/eval/<exp_name>.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = _merge_configs(config, load_config(args.override))

    seed = int(_get(config, "experiment", "seed", default=42))
    set_seed(seed, deterministic=bool(_get(config, "experiment", "deterministic", default=False)))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    exp_name = _get(config, "experiment_name",
                    default=_get(config, "experiment", "name", default="default"))
    log_root = _get(config, "paths", "log_dir", default="outputs/logs")
    log_dir = os.path.join(log_root, f"{exp_name}_eval")
    logger = setup_logging(log_dir, level=str(_get(config, "logging", "level", default="INFO")))

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config} | Checkpoint: {args.checkpoint}")

    id_loader, ood_loader = _build_eval_loaders(
        config, no_ood=args.no_ood, batch_size_override=args.batch_size,
    )
    logger.info(
        f"Loaders built | id_test={len(id_loader.dataset)} "
        f"ood_test={len(ood_loader.dataset) if ood_loader is not None else 0}"
    )

    descriptions = _load_descriptions(config)
    class_names = _get(config, "data", "class_names", default=[])
    model = GLocalFSLOODModel(
        config=config, class_names=class_names, descriptions=descriptions, device=device,
    ).to(device)
    model.freeze_encoders() if hasattr(model, "freeze_encoders") else None

    ckpt = load_checkpoint(args.checkpoint, model=model, device=device)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        logger.info(f"Loaded checkpoint epoch={ckpt.get('epoch')} best={ckpt.get('best_metric')}")

    temperature = float(args.temperature if args.temperature is not None
                        else _get(config, "ood", "temperature", default=1.0))

    result = test_fn(
        model=model,
        test_loader=id_loader,
        device=device,
        config=config,
        logger=logger,
        ood_loader=ood_loader,
        temperature=temperature,
    )

    print(_format_results(result))

    out_path = Path(args.save_json) if args.save_json else Path("outputs/eval") / f"{exp_name}.json"
    _dump_json(result, out_path)
    logger.info(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
