#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-stage LLM description generation for known ID classes.

Implements a two-stage generation pipeline:
    Stage A: Extract discriminative attributes per class
    Stage B: Generate CLIP-friendly descriptions from attributes

All generation is cross-class aware to maximize discriminability.

**Only known in-distribution (ID) classes receive generated text.**
OOD detection relies on poor alignment with known-class embeddings.

Usage:
    python src/scripts/generate_llm_descriptions.py \\
        --config configs/experiment/exp_full_model.yaml

    # Force regeneration:
    python src/scripts/generate_llm_descriptions.py \\
        --config configs/experiment/exp_full_model.yaml --force

    # Skip question generation (reuse existing):
    python src/scripts/generate_llm_descriptions.py \\
        --config configs/experiment/exp_full_model.yaml --skip_questions
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Redirect HF cache to /tmp *before* transformers is imported anywhere.
from src.models.encoders.text_generation.hf_env import setup_hf_cache  # noqa: E402
setup_hf_cache(verbose=True)

from src.utils.config import load_config  # noqa: E402
from src.models.encoders.text_generation import (  # noqa: E402
    LLMWrapper,
    GenerationConfig,
    PromptBuilder,
    OutputCleaner,
    DescriptionScorer,
    CacheManager,
    build_generator,
    release_generator,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("generate_descriptions")


def _resolve_id_classes(cfg, override_class_names):
    """Resolve **known ID classes** only — never OOD classes."""
    if override_class_names:
        return override_class_names

    class_names = cfg.data.get("class_names", [])
    if not class_names:
        return []

    id_classes = cfg.data.get("id_classes")
    if not id_classes:
        return class_names

    resolved = []
    for idx in id_classes:
        if isinstance(idx, int) and 0 <= idx < len(class_names):
            resolved.append(class_names[idx])
    return resolved


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Multi-stage LLM description generation for known ID classes"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Override known class names")
    parser.add_argument("--skip_questions", action="store_true",
                        help="Reuse existing questions file")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if valid cache exists")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--num_attributes", type=int, default=None,
                        help="Number of discriminative attributes per class (Stage A)")
    parser.add_argument("--num_descriptions", type=int, default=None)
    parser.add_argument("--min_final_descriptions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--quality_report",
        type=str,
        default=None,
        help="Optional JSON path to export candidate-level quality report",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete the HF cache after generation (overrides config).",
    )
    parser.add_argument(
        "--force-cleanup",
        action="store_true",
        help="Always delete the HF cache after generation (overrides config).",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Force API backend (sets use_local_llm=false for this run).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    log.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    # ── Resolve ID classes ───────────────────────────────────────────────
    class_names = _resolve_id_classes(cfg, args.class_names)
    if not class_names:
        log.error("No class names resolved. Set --class_names or config.data.class_names.")
        sys.exit(1)
    log.info("Known ID classes (%d): %s", len(class_names), class_names)

    # ── Parameters ───────────────────────────────────────────────────────
    llm_cfg = cfg.model.llm
    desc_cfg = cfg.llm_descriptions

    num_q = args.num_questions if args.num_questions is not None else desc_cfg.get("num_questions", 10)
    num_a = args.num_attributes if args.num_attributes is not None else desc_cfg.get("num_attributes_per_class", 8)
    num_d = args.num_descriptions if args.num_descriptions is not None else desc_cfg.get("num_descriptions_per_class", 30)
    seed = args.seed if args.seed is not None else llm_cfg.get("seed", 42)
    deterministic = args.deterministic or llm_cfg.get("deterministic", False)

    log.info("Generation targets: questions=%d, attributes=%d, descriptions=%d", num_q, num_a, num_d)
    log.info("Seed=%s, deterministic=%s", seed, deterministic)

    # ── Output paths ─────────────────────────────────────────────────────
    questions_out = desc_cfg.get("questions_output_file", "data/prompts/class_questions.yaml")
    descriptions_out = desc_cfg.get("output_file", "data/prompts/class_descriptions.yaml")
    json_out = desc_cfg.get("glali_output_file") or str(Path(descriptions_out).with_suffix(".json"))

    # ── Dataset description ──────────────────────────────────────────────
    dataset_desc = desc_cfg.get(
        "dataset_description",
        "A bone X-ray image dataset for few-shot classification of bone tumors."
    )
    try:
        dataset_desc = dataset_desc.format(num_classes=len(class_names))
    except Exception:
        pass

    # ── Build components ─────────────────────────────────────────────────
    gen_config = GenerationConfig(
        max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        temperature=llm_cfg.get("temperature", 0.3),
        top_p=llm_cfg.get("top_p", 0.85),
        repetition_penalty=llm_cfg.get("repetition_penalty", 1.15),
        seed=seed,
        deterministic=deterministic,
    )

    # Resolve the effective LLM config, applying CLI overrides.
    effective_llm_cfg = dict(llm_cfg)
    effective_llm_cfg["cache_dir"] = (
        effective_llm_cfg.get("cache_dir") or os.getenv("HF_CACHE_DIR")
    )
    if args.use_api:
        effective_llm_cfg["use_local_llm"] = False
    if args.force_cleanup:
        effective_llm_cfg["cleanup_cache"] = True
    if args.no_cleanup:
        effective_llm_cfg["cleanup_cache"] = False

    generator = build_generator(effective_llm_cfg)

    prompt_builder = PromptBuilder(dataset_description=dataset_desc)
    cleaner = OutputCleaner()
    scorer = DescriptionScorer()
    cache = CacheManager(
        descriptions_path=descriptions_out,
        questions_path=questions_out,
        json_path=json_out,
    )

    llm = LLMWrapper(
        generator=generator,
        generation_config=gen_config,
        prompt_builder=prompt_builder,
        output_cleaner=cleaner,
        description_scorer=scorer,
        cache_manager=cache,
        max_retries=3,
    )
    if args.quality_report:
        llm.enable_quality_report(args.quality_report)

    # ── Cache-first check ────────────────────────────────────────────────
    if not args.force and cache.cache_valid(class_names, num_d):
        log.info("Valid cache found — skipping generation.")
        log.info("  Questions: %s", questions_out)
        log.info("  Descriptions: %s", descriptions_out)
        log.info("  JSON: %s", json_out)
        log.info("Use --force to regenerate.")
        release_generator(generator, delete_cache=effective_llm_cfg.get("cleanup_cache"))
        return

    # ── Run multi-stage generation ───────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting multi-stage generation pipeline")
    log.info("=" * 60)

    try:
        _run_generation(
            args=args,
            cfg=cfg,
            llm=llm,
            generator=generator,
            cache=cache,
            cleaner=cleaner,
            prompt_builder=prompt_builder,
            gen_config=gen_config,
            class_names=class_names,
            num_q=num_q,
            num_a=num_a,
            num_d=num_d,
            seed=seed,
            questions_out=questions_out,
        )
    finally:
        # Always reclaim VRAM and (optionally) disk, even on failure.
        release_generator(generator, delete_cache=effective_llm_cfg.get("cleanup_cache"))
        if effective_llm_cfg.get("cleanup_cache"):
            log.info("LLM cache cleanup requested → disk freed.")

    log.info("=" * 60)
    log.info("Generation complete in %.1fs", time.time() - t0)
    log.info("=" * 60)
    log.info("  Questions:    %s", questions_out)
    log.info("  Descriptions: %s", descriptions_out)
    log.info("  Flat JSON:    %s", json_out)
    if args.quality_report:
        llm.export_quality_report({
            "dataset_name": cfg.data.get("dataset_name", "bone_xray"),
            "class_names": class_names,
            "num_questions": num_q,
            "num_attributes": num_a,
            "num_descriptions": num_d,
        })


def _run_generation(
    *,
    args,
    cfg,
    llm,
    generator,
    cache,
    cleaner,
    prompt_builder,
    gen_config,
    class_names,
    num_q,
    num_a,
    num_d,
    seed,
    questions_out,
) -> None:
    # Handle skip_questions
    if args.skip_questions and cache.questions_exist():
        questions = cache.load_questions()
        log.info("Loaded %d existing questions from %s", len(questions), questions_out)

        # Manual generation without question step
        from src.models.encoders.text_generation.cache_manager import (
            build_class_entry,
            build_output_payload,
        )

        classes_data = {}
        all_descriptions = {}

        for idx, cls_name in enumerate(class_names, 1):
            log.info("[%d/%d] Processing '%s' …", idx, len(class_names), cls_name)
            other_classes = [c for c in class_names if c != cls_name]

            # Stage A
            attributes = llm.generate_attributes(cls_name, other_classes, num_a)
            log.info("  → %d attributes", len(attributes))

            # Stage B
            descriptions = llm.generate_descriptions(
                cls_name, num_d, other_classes, attributes, questions
            )
            if len(descriptions) < args.min_final_descriptions:
                need = args.min_final_descriptions - len(descriptions)
                descriptions = llm._targeted_retry(  # noqa: SLF001 - intentional fallback
                    class_name=cls_name,
                    existing=descriptions,
                    num_needed=need,
                    other_classes=other_classes,
                )
            log.info("  → %d descriptions", len(descriptions))

            all_descriptions[cls_name] = descriptions
            classes_data[cls_name] = build_class_entry(
                default_prompt=prompt_builder.default_prompt(cls_name),
                attributes=attributes,
                descriptions=descriptions,
            )

        # Cross-class dedup
        log.info("Cross-class deduplication …")
        deduped = cleaner.remove_cross_class_duplicates(all_descriptions)
        for cls_name, descs in deduped.items():
            classes_data[cls_name]["descriptions"] = descs
            classes_data[cls_name]["metadata"]["num_descriptions"] = len(descs)

        # Save
        payload = build_output_payload(
            dataset_name=cfg.data.get("dataset_name", "bone_xray"),
            model_name=generator.model_name(),
            seed=seed,
            generation_config=gen_config.as_dict(),
            class_names=class_names,
            questions=questions,
            classes=classes_data,
        )
        cache.save_descriptions(payload)
        class_centric = {
            cls: {
                "default_prompt": data["default_prompt"],
                "attributes": data["attributes"],
                "descriptions": data["descriptions"],
            }
            for cls, data in classes_data.items()
        }
        cache.save_flat_json(class_centric)

    else:
        # Full pipeline
        llm.generate_all(
            class_names=class_names,
            dataset_name=cfg.data.get("dataset_name", "bone_xray"),
            num_questions=num_q,
            num_attributes=num_a,
            num_descriptions=num_d,
            min_descriptions_per_class=args.min_final_descriptions,
            force_regenerate=True,  # Already checked cache above
        )


if __name__ == "__main__":
    main()
