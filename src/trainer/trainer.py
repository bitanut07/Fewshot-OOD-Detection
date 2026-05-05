# -*- coding: utf-8 -*-
"""Trainer orchestrator for GLOCAL-FSL-OOD.

Acts like glali's ``TrainerX``: wraps model, optimizer, scheduler, and
data loaders; exposes ``train()`` / ``test()`` entry points; persists
checkpoints; and tracks the best model.

Features:
    - Automatic optimizer / scheduler construction (AdamW + Cosine LR).
    - Linear warmup for the first ``warmup_epochs`` epochs.
    - Mixed precision (AMP) via ``torch.amp.GradScaler``.
    - Periodic validation + best-model checkpointing.
    - Optional end-of-training test evaluation on ID + OOD loaders.
    - Resume-from-checkpoint support.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.losses.classification_loss import ClassificationLoss
from src.trainer.test import test as test_epoch
from src.trainer.train import _build_total_loss_from_cfg, train as train_epoch
from src.trainer.validate import validate as validate_epoch
from src.utils.checkpoint import (
    load_checkpoint,
    prune_checkpoint_dir,
    save_checkpoint,
)


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


def _build_optimizer(params, cfg) -> torch.optim.Optimizer:
    opt_cfg = _get(cfg, "train", "optimizer") or {}
    opt_name = str(_get(opt_cfg, "type", default="AdamW")).lower()
    lr = float(_get(opt_cfg, "lr", default=2e-4))
    wd = float(_get(opt_cfg, "weight_decay", default=0.01))
    betas = tuple(_get(opt_cfg, "betas", default=(0.9, 0.999)))
    if opt_name == "sgd":
        momentum = float(_get(opt_cfg, "momentum", default=0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)


def _build_scheduler(optimizer, cfg, num_epochs: int) -> Optional[Any]:
    sch_cfg = _get(cfg, "train", "scheduler") or {}
    name = str(_get(sch_cfg, "type", default="CosineAnnealingLR")).lower()
    warmup = int(_get(cfg, "train", "warmup_epochs", default=0))
    T_max = int(_get(sch_cfg, "T_max", default=num_epochs))
    eta_min = float(_get(sch_cfg, "eta_min", default=1e-6))

    if name in ("cosine", "cosineannealinglr"):
        base_lr = optimizer.param_groups[0]["lr"]
        warmup_cons_lr = float(_get(cfg, "train", "warmup_cons_lr", default=0.0))

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                if warmup_cons_lr > 0:
                    return warmup_cons_lr / base_lr
                return (epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, (T_max - warmup))
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            factor = (eta_min + (base_lr - eta_min) * cos) / base_lr
            return float(factor)

        return LambdaLR(optimizer, lr_lambda)
    if name == "step":
        step = int(_get(sch_cfg, "step_size", default=max(1, num_epochs // 3)))
        gamma = float(_get(sch_cfg, "gamma", default=0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    return None


class Trainer:
    """End-to-end orchestrator: train → validate → test."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        ood_loader: Optional[DataLoader] = None,
        logger: Optional[Any] = None,
        tb_logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.ood_loader = ood_loader
        self.logger = logger
        self.tb_logger = tb_logger

        # Output dirs
        if output_dir is None:
            out_root = _get(config, "paths", "output_dir", default="outputs")
            exp_name = _get(config, "experiment_name", default=_get(config, "experiment", "name", default="default"))
            output_dir = str(Path(out_root) / exp_name)
        self.output_dir = Path(output_dir)
        self.ckpt_dir = Path(_get(config, "paths", "checkpoint_dir", default=self.output_dir / "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Disk-efficient checkpointing: by default we only keep best.pt + last.pt
        # on disk (overwriting each epoch). Turn off with keep_only_best=false.
        self.keep_only_best = bool(_get(config, "train", "keep_only_best", default=True))
        # On startup, wipe any stale per-epoch checkpoints left by previous runs.
        if self.keep_only_best:
            pruned = prune_checkpoint_dir(self.ckpt_dir, keep=("best.pt", "last.pt"))
            if pruned and logger:
                logger.info(f"Removed {pruned} old checkpoint file(s) from {self.ckpt_dir}")

        # Optimizer / scheduler
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable) == 0:
            raise RuntimeError("No trainable parameters. Did you freeze everything by mistake?")
        self.optimizer = _build_optimizer(trainable, config)
        self.scheduler = _build_scheduler(
            self.optimizer, config,
            int(_get(config, "train", "epochs", default=40)),
        )

        # Losses
        self.loss_fn = ClassificationLoss(
            label_smoothing=float(_get(config, "train", "label_smoothing", default=0.0))
        )
        self.total_loss_fn = _build_total_loss_from_cfg(config)

        # AMP
        self.use_amp = bool(_get(config, "train", "mixed_precision", default=False)) and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.grad_clip = float(_get(config, "train", "grad_clip", default=1.0))
        self.start_epoch = 1
        self.best_val = -float("inf")
        self.best_epoch = 0
        self.global_step = 0

        # Early stopping
        self.es_patience = int(_get(config, "train", "early_stopping_patience", default=0))
        self.es_min_delta = float(_get(config, "train", "early_stopping_min_delta", default=0.0))
        self._es_counter = 0

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def resume(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            if self.logger:
                self.logger.warning(f"Checkpoint not found, starting fresh: {ckpt_path}")
            return
        ckpt = load_checkpoint(ckpt_path, model=self.model, optimizer=self.optimizer, device=self.device)
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val = float(ckpt.get("best_val", -float("inf")))
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        if self.logger:
            self.logger.info(
                f"Resumed from {ckpt_path} (epoch={self.start_epoch-1}, "
                f"best_val={self.best_val:.4f}, best_epoch={self.best_epoch})"
            )

    def save(self, epoch: int, val_metric: float, is_best: bool = False, tag: Optional[str] = None) -> None:
        # Only persist trainable module weights (not frozen CLIP encoders).
        # Frozen encoders are rebuilt from pretrained weights at model construction.
        if hasattr(self.model, "trainable_modules"):
            model_sd = self.model.trainable_modules.state_dict()
        else:
            model_sd = self.model.state_dict()

        state = {
            "epoch": epoch,
            "model_state_dict": model_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "global_step": self.global_step,
            "val_metric": val_metric,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else None,
            "checkpoint_format": "trainable_only",
        }
        # Default: always overwrite 'last.pt'. 'best.pt' is written by
        # save_checkpoint when is_best=True. No per-epoch files hit disk.
        if self.keep_only_best:
            fname = tag or "last.pt"
            save_checkpoint(
                state, self.ckpt_dir, fname,
                is_best=is_best,
                keep_only_best=True,
                keep_names=("best.pt", "last.pt"),
            )
        else:
            fname = tag or f"epoch_{epoch}.pt"
            save_checkpoint(state, self.ckpt_dir, fname, is_best=is_best)

        if self.logger:
            flag = " (best)" if is_best else ""
            self.logger.info(f"Saved checkpoint → {self.ckpt_dir / fname}{flag}")

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        epochs = int(_get(self.config, "train", "epochs", default=40))
        eval_every = int(_get(self.config, "train", "eval_every_n_epochs", default=1))
        save_every = int(_get(self.config, "train", "save_every_n_epochs",
                              default=_get(self.config, "checkpoint", "save_freq", default=5)))
        save_best = bool(_get(self.config, "train", "save_best", default=True))

        if self.train_loader is None:
            raise RuntimeError("train() called without a train_loader")

        history = {"train": [], "val": []}
        stopped_early = False
        for epoch in range(self.start_epoch, epochs + 1):
            # ---- train epoch ----
            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, self.loss_fn,
                self.device, epoch, self.config, self.logger,
                total_loss_fn=self.total_loss_fn, scaler=self.scaler,
                scheduler=None,  # stepped here for consistency
                grad_clip=self.grad_clip, tb_logger=self.tb_logger,
                global_step_start=self.global_step,
            )
            self.global_step = int(train_metrics.get("global_step", self.global_step))
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:  # noqa: BLE001
                    pass
            history["train"].append(train_metrics)

            # ---- validate ----
            is_best = False
            val_metrics: Dict[str, Any] = {}
            if self.val_loader is not None and (epoch % eval_every == 0 or epoch == epochs):
                val_metrics = validate_epoch(
                    self.model, self.val_loader, self.loss_fn, self.device,
                    epoch, self.config, self.logger,
                    ood_loader=self.ood_loader, tb_logger=self.tb_logger,
                )
                history["val"].append(val_metrics)
                val_metric = float(val_metrics.get("val_accuracy", 0.0))
                if val_metric > self.best_val + self.es_min_delta:
                    self.best_val = val_metric
                    self.best_epoch = epoch
                    is_best = True
                    self._es_counter = 0
                else:
                    self._es_counter += 1

            # ---- checkpoint ----
            if epoch % save_every == 0 or epoch == epochs or is_best:
                self.save(epoch, float(val_metrics.get("val_accuracy", 0.0)),
                          is_best=(is_best and save_best))

            # ---- early stopping ----
            if self.es_patience > 0 and self._es_counter >= self.es_patience:
                if self.logger:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}: "
                        f"no improvement for {self.es_patience} eval cycles "
                        f"(best_val={self.best_val:.4f} @ epoch {self.best_epoch})"
                    )
                stopped_early = True
                break

        if self.logger:
            tag = " (early stopped)" if stopped_early else ""
            self.logger.info(
                f"Training complete{tag}. Best val acc={self.best_val:.4f} @ epoch {self.best_epoch}"
            )
        return {"history": history, "best_val": self.best_val, "best_epoch": self.best_epoch}

    def test(self, use_best: bool = True) -> Dict[str, Any]:
        """Run test-time evaluation (optionally loading the best checkpoint)."""
        if use_best:
            best_path = self.ckpt_dir / "best.pt"
            if best_path.exists():
                load_checkpoint(str(best_path), model=self.model, device=self.device)
                if self.logger:
                    self.logger.info(f"Loaded best checkpoint from {best_path}")
        if self.test_loader is None:
            if self.logger:
                self.logger.warning("test() called without a test_loader")
            return {}
        return test_epoch(
            self.model, self.test_loader, self.device, self.config, self.logger,
            ood_loader=self.ood_loader,
            temperature=float(_get(self.config, "ood", "temperature", default=1.0)),
        )
