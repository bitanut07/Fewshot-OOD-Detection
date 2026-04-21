#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LLM-based class descriptions for known ID classes.

Offline preprocessing step that produces:
  1. Discriminative questions (dataset-level)
  2. Visual descriptions per known class (guided by questions)

Outputs are saved in a structured schema (v2) consumed by the CLIP text
encoder and text-refinement module during training.

**Only known in-distribution (ID) classes receive generated text.**
OOD detection relies on poor alignment with known-class embeddings.

Usage:
    python src/scripts/generate_llm_descriptions.py \\
        --config configs/experiment/exp_full_model.yaml

    # Force regeneration even if cache exists:
    python src/scripts/generate_llm_descriptions.py \\
        --config configs/experiment/exp_full_model.yaml --force

    # Skip question step (reuse existing):
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

from src.utils.config import load_config
from src.models.encoders.text_generation import (
    LLMWrapper,
    GenerationConfig,
    HFLocalGenerator,
    PromptBuilder,
    OutputCleaner,
    CacheManager,
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
        description="Generate LLM descriptions for known ID classes"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Override known class names")
    parser.add_argument("--skip_questions", action="store_true",
                        help="Reuse existing questions file")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if valid cache exists")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--num_descriptions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    log.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    # ── Resolve ID classes ───────────────────────────────────────────────
    class_names = _resolve_id_classes(cfg, args.class_names)
    if not class_names:
        log.error("No class names resolved.  Set --class_names or config.data.class_names.")
        sys.exit(1)
    log.info("Known ID classes (%d): %s", len(class_names), class_names)

    # ── Parameters ───────────────────────────────────────────────────────
    llm_cfg = cfg.model.llm
    desc_cfg = cfg.llm_descriptions

    num_q = args.num_questions or desc_cfg.get("num_questions", 10)
    num_d = args.num_descriptions or desc_cfg.get("num_descriptions_per_class", 8)
    seed = args.seed if args.seed is not None else llm_cfg.get("seed", 42)
    deterministic = args.deterministic or llm_cfg.get("deterministic", False)

    log.info("Generation targets: questions=%d, descriptions_per_class=%d", num_q, num_d)
    log.info("Seed=%s, deterministic=%s", seed, deterministic)

    # ── Output paths ─────────────────────────────────────────────────────
    questions_out = desc_cfg.get("questions_output_file", "data/prompts/class_questions.yaml")
    descriptions_out = desc_cfg.get("output_file", "data/prompts/class_descriptions.yaml")
    json_out = (
        desc_cfg.get("glali_output_file")
        or str(Path(descriptions_out).with_suffix(".json"))
    )

    # ── Dataset description (no OOD class names) ─────────────────────────
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

    effective_cache_dir = llm_cfg.get("cache_dir") or os.getenv("HF_CACHE_DIR")

    generator = HFLocalGenerator(
        model_name_or_path=llm_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device_map=llm_cfg.get("device_map", "auto"),
        torch_dtype=llm_cfg.get("torch_dtype", "float16"),
        cache_dir=effective_cache_dir,
        trust_remote_code=llm_cfg.get("trust_remote_code", True),
    )

    prompt_builder = PromptBuilder(
        dataset_description=dataset_desc,
        question_prompt_template=desc_cfg.get("question_prompt_template"),
        description_prompt_template=desc_cfg.get("description_prompt_template"),
    )

    cache = CacheManager(
        descriptions_path=descriptions_out,
        questions_path=questions_out,
        json_path=json_out,
    )

    llm = LLMWrapper(
        generator=generator,
        generation_config=gen_config,
        prompt_builder=prompt_builder,
        cache_manager=cache,
        max_retries=3,
    )

    # ── Cache-first check ────────────────────────────────────────────────
    if not args.force and cache.cache_valid(class_names, num_d):
        log.info("Valid cache found — skipping generation.")
        log.info("  Questions: %s", questions_out)
        log.info("  Descriptions: %s", descriptions_out)
        log.info("  JSON: %s", json_out)
        log.info("Use --force to regenerate.")
        return

    # ── Step 1: Questions ────────────────────────────────────────────────
    step1_t = time.time()
    if args.skip_questions and cache.questions_exist():
        questions = cache.load_questions()
        log.info("Loaded %d existing questions from %s", len(questions), questions_out)
    else:
        log.info("Generating %d discriminative questions …", num_q)
        questions = llm.generate_questions(num_q)
        cache.save_questions(questions)
        log.info("Generated %d questions in %.1fs", len(questions), time.time() - step1_t)
        for q in questions:
            log.info("  Q: %s", q)

    # ── Step 2: Per-class descriptions (ID only) ─────────────────────────
    step2_t = time.time()
    log.info("Generating %d descriptions × %d known classes …", num_d, len(class_names))

    from src.models.encoders.text_generation.cache_manager import (
        build_class_entry,
        build_output_payload,
    )

    classes_data = {}
    flat = {}
    for idx, cls_name in enumerate(class_names, 1):
        cls_t = time.time()
        log.info("[%d/%d] '%s' …", idx, len(class_names), cls_name)
        descs = llm.generate_descriptions(cls_name, num_d, questions)
        default_prompt = prompt_builder.default_prompt(cls_name)
        classes_data[cls_name] = build_class_entry(
            default_prompts=[default_prompt],
            generated_descriptions=descs,
        )
        flat[cls_name] = descs
        log.info(
            "[%d/%d] '%s' → %d descriptions (%.1fs)",
            idx, len(class_names), cls_name, len(descs), time.time() - cls_t,
        )
        for d in descs:
            log.info("    • %s", d)

    log.info("Step 2 completed in %.1fs", time.time() - step2_t)

    # ── Step 3: Save ─────────────────────────────────────────────────────
    dataset_name = cfg.data.get("dataset_name", "bone_xray")
    payload = build_output_payload(
        dataset_name=dataset_name,
        model_name=generator.model_name(),
        seed=seed,
        generation_config=gen_config.as_dict(),
        class_names=class_names,
        questions=questions,
        classes=classes_data,
    )
    cache.save_descriptions(payload)
    cache.save_flat_json(flat)

    log.info("Done in %.1fs", time.time() - t0)
    log.info("  Questions:    %s", questions_out)
    log.info("  Descriptions: %s", descriptions_out)
    log.info("  Flat JSON:    %s", json_out)


if __name__ == "__main__":
    main()
