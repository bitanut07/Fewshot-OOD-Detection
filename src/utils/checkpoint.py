# -*- coding: utf-8 -*-
"""Checkpoint utilities — atomic save + prune helpers.

Atomic save: ``torch.save`` first writes to ``<path>.tmp`` and only renames
to the final path on success. If the disk fills up partway through, the
final file never appears (and the half-written ``.tmp`` is removed), so
we never end up with a corrupted checkpoint that the next run tries to
load and crashes on.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Union

import torch

log = logging.getLogger(__name__)


def _atomic_torch_save(state, path: Path) -> None:
    """Write ``state`` to ``path`` atomically (tmp → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        # clean up partial write so next run doesn't load a corrupt ckpt
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def save_checkpoint(
    state,
    checkpoint_dir: Union[str, Path],
    filename: str,
    is_best: bool = False,
    keep_only_best: bool = False,
    keep_names: Optional[Iterable[str]] = None,
) -> str:
    """Save ``state`` atomically, optionally pruning old checkpoints.

    Parameters
    ----------
    state, checkpoint_dir, filename:
        As usual.
    is_best:
        If True, also writes ``best.pt`` in the same dir.
    keep_only_best:
        If True, delete every ``*.pt`` in the directory except the file(s)
        listed in ``keep_names`` (default: ``{"best.pt", "last.pt"}``).
        This keeps disk usage flat across a long run.
    """
    cd = Path(checkpoint_dir)
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / filename
    _atomic_torch_save(state, path)

    if is_best:
        _atomic_torch_save(state, cd / "best.pt")

    if keep_only_best:
        keep = set(keep_names or {"best.pt", "last.pt"})
        keep.add(filename)  # the file we just wrote is always kept
        prune_checkpoint_dir(cd, keep=keep)

    return str(path)


def prune_checkpoint_dir(
    checkpoint_dir: Union[str, Path],
    keep: Iterable[str] = ("best.pt", "last.pt"),
) -> int:
    """Remove every ``*.pt`` / ``*.pt.tmp`` in *checkpoint_dir* not in *keep*.

    Returns the number of files deleted.
    """
    cd = Path(checkpoint_dir)
    if not cd.is_dir():
        return 0
    keep_set = set(keep)
    removed = 0
    for f in cd.iterdir():
        if not f.is_file():
            continue
        if f.suffix not in (".pt", ".tmp") and not f.name.endswith(".pt.tmp"):
            continue
        if f.name in keep_set:
            continue
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            log.warning("Could not remove %s: %s", f, e)
    if removed:
        log.info("Pruned %d old checkpoint file(s) in %s", removed, cd)
    return removed


def load_checkpoint(
    checkpoint_path,
    model=None,
    optimizer=None,
    device=None,
    load_finetuned_only=False,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device or torch.device("cpu"), weights_only=False)

    if model is not None and not load_finetuned_only:
        sd = ckpt.get("model_state_dict", ckpt)
        fmt = ckpt.get("checkpoint_format", "full")
        if fmt == "trainable_only" and hasattr(model, "trainable_modules"):
            model.trainable_modules.load_state_dict(sd, strict=True)
        else:
            model.load_state_dict(sd, strict=False)

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
