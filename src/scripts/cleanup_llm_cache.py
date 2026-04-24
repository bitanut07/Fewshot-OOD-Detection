#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delete HuggingFace / LLM on-disk cache to reclaim space.

Use this when:
  * The GPU cloud disk is full and the previous script run reported 0 MB
    freed (usually because the old cache lives under ``~/.cache/huggingface``
    instead of the ``/tmp`` path this project now uses).
  * You are done generating descriptions and want the 15GB LLM weights
    *off* the GPU node before real training starts saving checkpoints.

The script scans every well-known HF cache location and prints what it
finds before touching anything (unless ``--yes`` is given).

Usage
-----
    # Just SCAN — show every cache that exists, no deletion
    python src/scripts/cleanup_llm_cache.py --scan

    # Wipe the default /tmp cache (original behaviour)
    python src/scripts/cleanup_llm_cache.py

    # Wipe EVERY known HF cache location (recommended on cloud)
    python src/scripts/cleanup_llm_cache.py --all

    # Delete only a specific model's weights, everywhere
    python src/scripts/cleanup_llm_cache.py --all --model "Qwen/Qwen2.5-7B-Instruct"

    # Also remove stale training checkpoints (keeps best.pt + last.pt)
    python src/scripts/cleanup_llm_cache.py --all --prune-checkpoints outputs/runs

    # Skip confirmation prompt
    python src/scripts/cleanup_llm_cache.py --all --yes
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.encoders.text_generation.hf_env import get_hf_cache_dir

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cleanup_llm_cache")


# ── Utilities ───────────────────────────────────────────────────────────

def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                total += p.lstat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def _fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def _disk_free_gb(path: str = "/") -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except OSError:
        return float("nan")


def _candidate_cache_paths() -> List[Path]:
    """Every place HF / transformers might have stashed model weights."""
    paths: List[Path] = []
    seen = set()

    def add(p) -> None:
        if not p:
            return
        abs_p = Path(p).expanduser().resolve()
        if str(abs_p) in seen:
            return
        seen.add(str(abs_p))
        paths.append(abs_p)

    # Project-specific (this repo's default)
    add(get_hf_cache_dir())
    add("/tmp/hf-cache-fewshot-ood")

    # Env-var driven (whatever the user / system set)
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        add(os.environ.get(var))

    # Default HF locations
    add("~/.cache/huggingface")
    add("/root/.cache/huggingface")  # cloud boxes often run as root

    # Common cloud-provider pre-set locations (vast.ai, runpod, lambda, ...)
    add("/workspace/.hf_home")
    add("/workspace/.cache/huggingface")
    add("/workspace/huggingface")
    add("/data/.cache/huggingface")

    # KaggleHub cache (can be 1-2 GB after dataset downloads)
    add("~/.cache/kagglehub")
    add("/root/.cache/kagglehub")

    # Accelerate / torch / datasets side-caches
    add("~/.cache/torch/hub")
    add("/root/.cache/torch/hub")

    return paths


def _scan(paths: List[Path]) -> List[Tuple[Path, float]]:
    """Return list of (path, size_mb) for dirs that actually exist."""
    found: List[Tuple[Path, float]] = []
    for p in paths:
        if p.is_dir():
            found.append((p, _dir_size_mb(p)))
    return found


def _confirm(msg: str) -> bool:
    try:
        ans = input(f"{msg} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


# ── Actions ─────────────────────────────────────────────────────────────

def _remove_dir(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    size = _dir_size_mb(path)
    shutil.rmtree(path, ignore_errors=True)
    log.info("Removed %s (~%s)", path, _fmt_mb(size))
    return size


def _remove_model_everywhere(model: str, paths: List[Path]) -> float:
    slug = "models--" + model.replace("/", "--")
    total = 0.0
    for root in paths:
        if not root.is_dir():
            continue
        # HF layout: <root>/hub/models--org--name OR <root>/models--org--name
        for candidate in (root / slug, root / "hub" / slug):
            if candidate.is_dir():
                total += _remove_dir(candidate)
    if total == 0.0:
        log.info("No cache found for model: %s", model)
    return total


def _prune_checkpoints(root: Path, keep=("best.pt", "last.pt")) -> float:
    """Remove every *.pt / *.pt.tmp under *root* except files in *keep*."""
    if not root.is_dir():
        log.info("Checkpoint root not present: %s", root)
        return 0.0
    total = 0.0
    removed = 0
    keep_set = set(keep)
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        name = f.name
        if not (name.endswith(".pt") or name.endswith(".pt.tmp")):
            continue
        if name in keep_set:
            continue
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            f.unlink()
            total += size_mb
            removed += 1
        except OSError as e:
            log.warning("Could not remove %s: %s", f, e)
    log.info(
        "Pruned %d stale checkpoint file(s) under %s (~%s)",
        removed, root, _fmt_mb(total),
    )
    return total


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Delete LLM / HF caches on disk.")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Specific cache dir to wipe. Defaults to $HF_CACHE_DIR or /tmp/hf-cache-fewshot-ood.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan AND wipe every known HF cache location (recommended on cloud).",
    )
    parser.add_argument(
        "--also-home", action="store_true",
        help="Also remove ~/.cache/huggingface (subset of --all).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Delete only this model's cache (e.g. 'Qwen/Qwen2.5-7B-Instruct').",
    )
    parser.add_argument(
        "--prune-checkpoints", default=None,
        metavar="DIR",
        help="Also remove stale *.pt checkpoints under DIR, keeping best.pt + last.pt.",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Just list what exists; do not delete anything.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Alias of --scan.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    before_free = _disk_free_gb("/")
    log.info("Disk free before cleanup: %.2f GB (root partition)", before_free)

    candidates = _candidate_cache_paths()
    found = _scan(candidates)

    if found:
        log.info("── HF cache locations found ──")
        for p, mb in found:
            log.info("  %s  (%s)", p, _fmt_mb(mb))
    else:
        log.info("No HF cache directories found in any standard location.")

    if args.scan or args.dry_run:
        total = sum(mb for _, mb in found)
        log.info("Total cached weights on disk: %s (nothing deleted)", _fmt_mb(total))
        return 0

    # Build the target list
    if args.all:
        targets = [p for p, _ in found]
    elif args.also_home:
        home = Path(os.path.expanduser("~/.cache/huggingface")).resolve()
        targets = [p for p, _ in found if p == home or p == Path(args.cache_dir or get_hf_cache_dir()).resolve()]
        if not targets:
            targets = [home, Path(args.cache_dir or get_hf_cache_dir()).resolve()]
    else:
        # Original conservative behaviour: only the configured cache dir
        targets = [Path(args.cache_dir or get_hf_cache_dir()).resolve()]

    # Remove duplicates while preserving order
    seen = set()
    targets = [p for p in targets if not (str(p) in seen or seen.add(str(p)))]

    # Confirm when deleting large blobs
    to_touch_mb = sum(mb for p, mb in found if p in targets)
    if to_touch_mb > 1024 and not args.yes:
        log.warning(
            "About to delete ~%s across %d location(s).",
            _fmt_mb(to_touch_mb), len([t for t in targets if t.is_dir()]),
        )
        if not _confirm("Proceed?"):
            log.info("Aborted.")
            return 1

    total_freed = 0.0

    if args.model:
        total_freed += _remove_model_everywhere(args.model, targets)
    else:
        for p in targets:
            total_freed += _remove_dir(p)

    if args.prune_checkpoints:
        total_freed += _prune_checkpoints(Path(args.prune_checkpoints).expanduser())

    after_free = _disk_free_gb("/")
    log.info("Disk free after cleanup: %.2f GB (root partition)", after_free)
    log.info("Freed total: %s (delta on /: %.2f GB)",
             _fmt_mb(total_freed), max(0.0, after_free - before_free))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
