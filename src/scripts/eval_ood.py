#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OOD detection evaluation script.

Usage:
    python src/scripts/eval_ood.py --config configs/eval/ood.yaml --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.framework.glocal_fsl_ood_model import GLocalFSLOODModel
from src.datasets.bone_xray_dataset import BoneXRayDataset
from src.evaluation.evaluator import Evaluator
from src.utils.checkpoint import load_checkpoint
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.experiment.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load descriptions
    desc_path = config.llm_descriptions.output_file
    descriptions = {}
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            descriptions = yaml.safe_load(f) or {}

    # Load model
    model = GLocalFSLOODModel(
        config=config,
        class_names=config.data.get("class_names", []),
        descriptions=descriptions,
        device=device,
    )
    model = model.to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)

    # OOD datasets
    id_dataset = BoneXRayDataset(
        data_root=config.paths.data_root, split="test",
        known_classes=config.data.get("known_classes", list(range(10))),
        transform=BoneXRayDataset.get_default_transform("test"),
    ).filter_known()

    ood_dataset = BoneXRayDataset(
        data_root=config.paths.data_root, split="test",
        ood_classes=config.data.get("ood_classes", list(range(10, 15))),
        transform=BoneXRayDataset.get_default_transform("test"),
    ).filter_ood()

    id_loader = torch.utils.data.DataLoader(id_dataset, batch_size=16, shuffle=False, num_workers=4)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Evaluate
    evaluator = Evaluator(model, device, config)
    results = evaluator.evaluate_ood(id_loader, ood_loader, method=config.eval.ood.method)

    print("OOD Detection Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
