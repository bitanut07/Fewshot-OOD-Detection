# -*- coding: utf-8 -*-
"""Checkpoint utilities."""
import os
from pathlib import Path
from typing import Optional
import torch

def save_checkpoint(state, checkpoint_dir, filename, is_best=False):
    cd = Path(checkpoint_dir)
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / filename
    torch.save(state, path)
    if is_best:
        torch.save(state, cd / "best.pt")
    return str(path)

def load_checkpoint(checkpoint_path, model=None, optimizer=None, device=None, load_finetuned_only=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device or torch.device("cpu"))
    if model is not None and not load_finetuned_only:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
