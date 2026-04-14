#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main training script for few-shot learning.

Usage:
    python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml
    python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml --override configs/train/fewshot_4shot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logging
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.models.framework.glocal_fsl_ood_model import GLocalFSLOODModel
from src.datasets.bone_xray_dataset import BoneXRayDataset
from src.datasets.sampler_fewshot import FewShotSampler
from src.losses.total_loss import TotalLoss
from src.losses.classification_loss import ClassificationLoss
from src.trainer.train import train
from src.trainer.validate import validate
import torch
from torch.utils.data import DataLoader
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train GLOCAL-FSL-OOD model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--override", type=str, default=None, help="Override config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.override:
        override_cfg = load_config(args.override)
        config = {**config.to_dict(), **override_cfg.to_dict()}
        from src.utils.config import Config
        config = Config(config)

    # Setup
    set_seed(config.experiment.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(config.paths.log_dir, config.experiment.name)
    logger = setup_logging(log_dir, level=config.logging.level)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # Load descriptions
    desc_path = config.llm_descriptions.output_file
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            descriptions = yaml.safe_load(f) or {}
    else:
        descriptions = {}
        logger.warning(f"Descriptions file not found: {desc_path}. Using empty descriptions.")

    # Load dataset
    class_names = config.data.get("class_names", [])
    train_dataset = BoneXRayDataset(
        data_root=config.paths.data_root,
        split="train",
        known_classes=config.data.get("known_classes", list(range(10))),
        transform=BoneXRayDataset.get_default_transform("train"),
    ).filter_known()

    # Build model
    model = GLocalFSLOODModel(
        config=config,
        class_names=class_names,
        descriptions=descriptions,
        device=device,
    )
    model = model.to(device)
    model.freeze_encoders()
    logger.info(f"Model created. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    loss_fn = ClassificationLoss()
    total_loss_fn = TotalLoss(
        weight_cls=config.loss.classification.weight,
        weight_global_alignment=config.loss.global_alignment.weight,
        weight_local_alignment=config.loss.local_alignment.weight,
        weight_local_contrastive=config.loss.local_contrastive.weight,
        weight_text_refinement=config.loss.text_refinement.weight,
    )

    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.scheduler.T_max, eta_min=config.train.scheduler.eta_min
    )

    # Training loop
    best_acc = 0.0
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train(
            model, None, optimizer, loss_fn, device, epoch, config, logger
        )
        scheduler.step()

        if epoch % config.train.eval_every_n_epochs == 0:
            val_metrics = validate(model, None, loss_fn, device, epoch, config, logger)
            is_best = val_metrics["val_accuracy"] > best_acc
            if is_best:
                best_acc = val_metrics["val_accuracy"]

        if epoch % config.checkpoint.save_freq == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                },
                config.paths.checkpoint_dir,
                f"epoch_{epoch}.pt",
                is_best=is_best if config.train.save_best else False,
            )

    logger.info(f"Training complete. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
