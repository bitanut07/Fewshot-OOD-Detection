#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delete HuggingFace / LLM on-disk cache to reclaim space.

Use this when you are done generating descriptions and want the 15GB LLM
weights *off* the GPU node before the main training stage saves checkpoints.

Usage
-----
    # Wipe the default /tmp cache
    python src/scripts/cleanup_llm_cache.py

    # Wipe a specific model only (keep others)
    python src/scripts/cleanup_llm_cache.py --model "Qwen/Qwen2.5-7B-Instruct"

    # Wipe a custom cache directory
    python src/scripts/cleanup_llm_cache.py --cache-dir /tmp/hf-cache-fewshot-ood

    # Also wipe the default HF cache under $HOME (~/.cache/huggingface)
    python src/scripts/cleanup_llm_cache.py --also-home

    # Dry run — show what would be deleted
    python src/scripts/cleanup_llm_cache.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.encoders.text_generation.hf_env import (
    cleanup_hf_cache,
    cleanup_model_cache,
    get_hf_cache_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cleanup_llm_cache")


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def _dry_run_report(path: Path, label: str) -> float:
    if not path.is_dir():
        log.info("[dry-run] %s not present: %s", label, path)
        return 0.0
    mb = _dir_size_mb(path)
    log.info("[dry-run] would remove %s: %s (~%.1f MB)", label, path, mb)
    return mb


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete LLM / HF caches on disk.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override cache dir. Defaults to $HF_CACHE_DIR or /tmp/hf-cache-fewshot-ood.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Delete only this model's cache (e.g. 'Qwen/Qwen2.5-7B-Instruct').",
    )
    parser.add_argument(
        "--also-home",
        action="store_true",
        help="Also remove ~/.cache/huggingface (the default HF location).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_hf_cache_dir()
    home_cache = Path(os.path.expanduser("~/.cache/huggingface"))

    total_freed = 0.0

    if args.model:
        target = Path(cache_dir) / ("models--" + args.model.replace("/", "--"))
        if args.dry_run:
            total_freed += _dry_run_report(target, f"model '{args.model}'")
            if args.also_home:
                home_target = home_cache / "hub" / ("models--" + args.model.replace("/", "--"))
                total_freed += _dry_run_report(home_target, f"model '{args.model}' (home)")
        else:
            total_freed += cleanup_model_cache(args.model, cache_dir=cache_dir, verbose=True)
            if args.also_home:
                home_target = home_cache / "hub" / ("models--" + args.model.replace("/", "--"))
                if home_target.is_dir():
                    size = _dir_size_mb(home_target)
                    shutil.rmtree(home_target, ignore_errors=True)
                    log.info("Removed %s (~%.1f MB)", home_target, size)
                    total_freed += size
    else:
        if args.dry_run:
            total_freed += _dry_run_report(Path(cache_dir), "project cache")
            if args.also_home:
                total_freed += _dry_run_report(home_cache, "home cache (~/.cache/huggingface)")
        else:
            total_freed += cleanup_hf_cache(cache_dir, verbose=True)
            if args.also_home and home_cache.is_dir():
                size = _dir_size_mb(home_cache)
                shutil.rmtree(home_cache, ignore_errors=True)
                log.info("Removed %s (~%.1f MB)", home_cache, size)
                total_freed += size

    action = "Would free" if args.dry_run else "Freed"
    log.info("%s total: %.1f MB", action, total_freed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
