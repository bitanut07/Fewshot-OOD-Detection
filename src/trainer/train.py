# -*- coding: utf-8 -*-
"""Training epoch for GLOCAL-FSL-OOD.

Architecture / logic inspired by ``glali/trainers/locproto_supc.py`` —
in particular the ``forward_backward`` method — but refactored to the
project's modular loss / model abstractions:

    - Combined classification loss on the fused global+local logits
    - Optional classification loss on local-only logits (per-patch
      aggregated via mean) — glali ``loss_id2``
    - Optional SupCon-style local contrastive loss over top-k / bottom-k
      lesion regions — glali ``loss_supc``
    - Optional teacher → student feature distillation (L1) on global
      image features — glali ``loss_distil_img``
    - Optional text distillation (L1) between refined prototypes and
      frozen CLIP text prototypes — glali ``loss_distil_text``
    - OOD entropy regularization over non-top-K local patches — glali
      ``entropy_select_topk`` / LoCoOp style

Supports fp32 and AMP (mixed precision) training, per-batch gradient
clipping, cosine LR with optional linear warmup (stepped *per epoch*),
and per-batch tqdm metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.alignment_loss import GlobalAlignmentLoss, LocalAlignmentLoss
from src.losses.classification_loss import ClassificationLoss
from src.losses.total_loss import TotalLoss, entropy_select_topk


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


def _ensure_logits(logits: torch.Tensor) -> torch.Tensor:
    """Aggregate [B,P,C] patch logits to [B,C] via mean."""
    return logits.mean(dim=1) if logits.dim() == 3 else logits


def _remap_train_labels(labels: torch.Tensor, config: Any) -> torch.Tensor:
    """Map original class indices to contiguous ID indices when needed."""
    id_classes = _get(config, "data", "id_classes", default=None)
    if not id_classes:
        return labels

    id_classes = [int(x) for x in id_classes]
    remap = {orig: i for i, orig in enumerate(id_classes)}
    mapped = torch.full_like(labels, -1)
    for orig, new in remap.items():
        mapped[labels == orig] = new

    if torch.any(mapped < 0):
        bad = torch.unique(labels[mapped < 0]).detach().cpu().tolist()
        raise ValueError(
            f"Train labels contain indices not in data.id_classes: {bad}. "
            "This causes CUDA device-side assert in CrossEntropy."
        )
    return mapped


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module],
    device: torch.device,
    epoch: int,
    config: Any,
    logger: Optional[Any] = None,
    total_loss_fn: Optional[TotalLoss] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
    tb_logger: Optional[Any] = None,
    global_step_start: int = 0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: A ``GLocalFSLOODModel`` (or compatible) instance.
        train_loader: Iterable yielding ``(images, labels)``.
        optimizer: Optimizer.
        loss_fn: Primary classification loss (cross-entropy by default).
                 If ``None``, defaults to ``ClassificationLoss()``.
        device: CUDA device.
        epoch: Current epoch (1-indexed).
        config: Full config.
        logger: Optional Python logger.
        total_loss_fn: ``TotalLoss`` aggregator. Created from config
                       weights if not supplied.
        scaler: AMP grad scaler. Auto-created when mixed precision is on.
        scheduler: Step scheduler (stepped at end of epoch by caller or here
                   if ``step_per_epoch`` is True).
        grad_clip: Gradient clipping norm (defaults to config).
        tb_logger: Optional TensorBoard logger (``log_scalar(tag, v, step)``).
        global_step_start: Starting global step (for tb logging).
    """
    model.train()
    if hasattr(model, "freeze_encoders"):
        model.freeze_encoders()

    # Build defaults from config
    loss_fn = loss_fn or ClassificationLoss()
    total_loss_fn = total_loss_fn or _build_total_loss_from_cfg(config)

    use_amp = bool(_get(config, "train", "mixed_precision", default=False)) and device.type == "cuda"
    if use_amp and scaler is None:
        scaler = GradScaler("cuda")

    if grad_clip is None:
        grad_clip = float(_get(config, "train", "grad_clip", default=1.0))

    ood_topk = int(_get(config, "ood", "topk", default=_get(config, "local_contrastive", "top_k", default=50)))

    ga_fn = GlobalAlignmentLoss(temperature=float(_get(config, "alignment", "logit_temperature", default=0.07)))
    la_fn = LocalAlignmentLoss(temperature=float(_get(config, "alignment", "logit_temperature", default=0.07)))

    running: Dict[str, float] = {"loss": 0.0, "loss_cls": 0.0, "correct": 0.0, "total": 0.0}
    num_batches = 0
    global_step = global_step_start

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            images, labels = batch["img"], batch["label"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        train_labels = _remap_train_labels(labels, config)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(images, return_loss=True, labels=train_labels)
            logits = out["logits"]                           # [B, C]
            local_logits = out.get("local_logits")            # [B, P, C] or None
            global_feat = out.get("global_feat")
            global_feat_teacher = out.get("global_feat_teacher")
            text_for_align = out.get("text_for_align")
            class_protos = out.get("class_prototypes")
            loss_contrastive = out.get("loss_contrastive")

            # (1) classification loss on combined logits
            loss_cls = loss_fn(logits, train_labels)

            # (2) local-only classification loss (glali loss_id2)
            loss_la: Optional[torch.Tensor] = None
            if local_logits is not None:
                loss_la = F.cross_entropy(local_logits.mean(dim=1), train_labels)

            # (3) global alignment (image <-> text embeddings)
            loss_ga: Optional[torch.Tensor] = None
            if global_feat is not None and text_for_align is not None:
                loss_ga = ga_fn(global_feat, text_for_align, train_labels)

            # (4) OOD entropy regularization (glali entropy_select_topk)
            loss_ood_reg: Optional[torch.Tensor] = None
            if local_logits is not None:
                loss_ood_reg = entropy_select_topk(local_logits, train_labels, top_k=ood_topk)

            # (5) teacher/student distillation on global image features
            loss_distil_img: Optional[torch.Tensor] = None
            if global_feat_teacher is not None and global_feat is not None:
                loss_distil_img = F.l1_loss(global_feat_teacher, global_feat, reduction="mean")

            # (6) text distillation: refined prototypes vs raw class prototypes
            loss_distil_text: Optional[torch.Tensor] = None
            if (text_for_align is not None and class_protos is not None
                    and text_for_align.dim() == 3):
                with torch.no_grad():
                    raw = class_protos.unsqueeze(0).expand_as(text_for_align)
                loss_distil_text = F.l1_loss(text_for_align, raw, reduction="mean")

            breakdown = total_loss_fn(
                loss_cls=loss_cls,
                loss_global_alignment=loss_ga,
                loss_local_alignment=loss_la,
                loss_local_contrastive=loss_contrastive,
                loss_text_refinement=None,
                loss_ood_reg=loss_ood_reg,
                loss_distill_img=loss_distil_img,
                loss_distill_text=loss_distil_text,
            )
            loss = breakdown["total_loss"]

        # Backward + step
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
            optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == train_labels).sum().item()
            total = train_labels.size(0)
        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_cls.detach().item())
        running["correct"] += correct
        running["total"] += total
        num_batches += 1
        global_step += 1

        pbar.set_postfix({
            "loss": f"{running['loss']/num_batches:.4f}",
            "cls": f"{running['loss_cls']/num_batches:.4f}",
            "acc": f"{running['correct']/max(1, running['total']):.3f}",
        })

        if tb_logger is not None:
            tb_logger.log_scalar("train/loss_step", float(loss.detach().item()), global_step)
            for k, v in breakdown.items():
                if k == "total_loss":
                    continue
                tb_logger.log_scalar(f"train/{k}", float(v), global_step)

    avg_loss = running["loss"] / max(1, num_batches)
    avg_cls = running["loss_cls"] / max(1, num_batches)
    accuracy = running["correct"] / max(1, running["total"])

    # Optionally step scheduler at end of epoch
    if scheduler is not None and _get(config, "train", "scheduler_step", default="epoch") == "epoch":
        try:
            scheduler.step()
        except Exception:  # noqa: BLE001
            pass

    if logger:
        logger.info(
            f"Epoch {epoch} [train] loss={avg_loss:.4f} cls={avg_cls:.4f} acc={accuracy:.4f}"
        )

    return {
        "loss": avg_loss,
        "loss_cls": avg_cls,
        "accuracy": accuracy,
        "global_step": global_step,
    }


def _build_total_loss_from_cfg(config: Any) -> TotalLoss:
    return TotalLoss(
        weight_cls=float(_get(config, "loss", "classification", "weight", default=1.0)),
        weight_global_alignment=float(_get(config, "loss", "global_alignment", "weight", default=1.0)),
        weight_local_alignment=float(_get(config, "loss", "local_alignment", "weight", default=1.0)),
        weight_local_contrastive=float(_get(config, "loss", "local_contrastive", "weight", default=0.5)),
        weight_text_refinement=float(_get(config, "loss", "text_refinement", "weight", default=0.1)),
        weight_ood_reg=float(_get(config, "loss", "ood_reg", "weight", default=0.25)),
        weight_distill_img=float(_get(config, "loss", "distill_img", "weight", default=0.0)),
        weight_distill_text=float(_get(config, "loss", "distill_text", "weight", default=0.0)),
    )
