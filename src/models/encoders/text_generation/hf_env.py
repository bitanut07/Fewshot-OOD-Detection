# -*- coding: utf-8 -*-
"""HuggingFace cache environment helper.

This module exists to solve one very specific problem:

    We run training on GPU cloud nodes with ~32GB of disk. A 7B LLM weighs
    ~15GB. If HuggingFace caches the model under ``~/.cache/huggingface``
    (same disk as the training checkpoints), the disk fills up and training
    crashes mid-epoch.

The fix is to redirect *all* HF / transformers / hub cache locations to a
transient directory (default: ``/tmp/hf-cache-fewshot-ood``). ``/tmp`` is
usually on a separate tmpfs / scratch volume on cloud instances, and we can
blow it away after LLM generation finishes (before real training starts).

IMPORTANT: this must run **before** ``transformers`` / ``huggingface_hub``
are imported. That is why every module that imports those libraries
(``hf_local_generator.py``, the factory, and the generation script) calls
``setup_hf_cache()`` as its *first* line.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/tmp/hf-cache-fewshot-ood"

# The env vars that transformers + huggingface_hub consult, in priority order.
# We set all of them so newer/older lib versions agree.
_HF_ENV_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)

_DID_SETUP = False


def setup_hf_cache(
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Redirect HF/transformers caches to *cache_dir* (default: /tmp).

    Idempotent — safe to call many times. Resolution order:
        1. explicit *cache_dir* argument
        2. ``$HF_CACHE_DIR`` environment variable
        3. ``DEFAULT_CACHE_DIR`` (``/tmp/hf-cache-fewshot-ood``)

    Returns the resolved absolute path actually used.
    """
    global _DID_SETUP

    resolved = (
        cache_dir
        or os.environ.get("HF_CACHE_DIR")
        or DEFAULT_CACHE_DIR
    )
    resolved = str(Path(resolved).expanduser().resolve())

    os.makedirs(resolved, exist_ok=True)
    for var in _HF_ENV_VARS:
        os.environ[var] = resolved
    os.environ["HF_CACHE_DIR"] = resolved

    if verbose and not _DID_SETUP:
        log.info("HF cache redirected → %s", resolved)
    _DID_SETUP = True
    return resolved


def get_hf_cache_dir() -> str:
    """Return the currently configured HF cache directory."""
    return (
        os.environ.get("HF_CACHE_DIR")
        or os.environ.get("HF_HOME")
        or DEFAULT_CACHE_DIR
    )


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def cleanup_hf_cache(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """Delete the HF cache directory. Returns MB freed (0.0 if nothing)."""
    path = Path(cache_dir or get_hf_cache_dir())
    if not path.is_dir():
        if verbose:
            log.info("HF cache dir not present: %s", path)
        return 0.0

    size_mb = _dir_size_mb(path)
    shutil.rmtree(path, ignore_errors=True)
    if verbose:
        log.info("Removed HF cache %s (~%.1f MB freed)", path, size_mb)
    return size_mb


def cleanup_model_cache(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """Delete cache of a *single* model only (keeps other models intact)."""
    root = Path(cache_dir or get_hf_cache_dir())
    if not root.is_dir():
        return 0.0

    # HF Hub lays out as: <cache_root>/models--<org>--<name>/...
    slug = "models--" + model_name_or_path.replace("/", "--")
    target = root / slug
    if not target.is_dir():
        # also try transformers layout (hash dirs) — just fall back to full scan
        hits = [p for p in root.rglob(slug) if p.is_dir()]
        if not hits:
            if verbose:
                log.info("No cache found for model: %s", model_name_or_path)
            return 0.0
        target = hits[0]

    size_mb = _dir_size_mb(target)
    shutil.rmtree(target, ignore_errors=True)
    if verbose:
        log.info(
            "Removed model cache %s (~%.1f MB freed)",
            target, size_mb,
        )
    return size_mb
