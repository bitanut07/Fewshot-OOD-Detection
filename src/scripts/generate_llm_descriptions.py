#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LLM-based disease descriptions and questions for all classes.

Run this script OFFLINE before training to generate:
  1. Discriminative questions (shared across dataset)
  2. Disease descriptions per class (guided by questions)

Output is saved to:
  - data/prompts/class_questions.yaml
  - data/prompts/class_descriptions.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.models.encoders.llm_wrapper import LLMWrapper


def _log(msg: str) -> None:
    """Lightweight timestamped logger for CLI progress tracking."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def _resolve_target_classes(cfg, override_class_names):
    """Resolve classes to generate for, prioritizing CLI override."""
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
        if not isinstance(idx, int):
            continue
        if 0 <= idx < len(class_names):
            resolved.append(class_names[idx])
    return resolved


def _save_glali_like_json(descriptions: dict, output_path: str) -> None:
    """Save flat class->descriptions JSON like glali/description/*.json."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False)
    print(f"[main] Glali-like JSON saved to: {out}")


def main():
    start_time = time.time()
    _log("Starting LLM description generation script.")

    parser = argparse.ArgumentParser(description="Generate LLM disease descriptions and questions")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument(
        "--class_names", type=str, nargs="+", default=None,
        help="Override class names (default: from config)"
    )
    parser.add_argument(
        "--skip_questions", action="store_true",
        help="Skip question generation (use existing class_questions.yaml)"
    )
    parser.add_argument(
        "--num_questions", type=int, default=None,
        help="Override number of questions"
    )
    parser.add_argument(
        "--num_descriptions", type=int, default=None,
        help="Override number of descriptions per class"
    )
    parser.add_argument(
        "--glali_json_out", type=str, default=None,
        help="Optional output JSON path (flat {class_name: [descriptions]} format like glali)"
    )
    args = parser.parse_args()
    _log(f"Arguments parsed. config={args.config}")

    # Load merged config
    _log("Loading merged configuration...")
    cfg = load_config(args.config)
    _log("Configuration loaded.")

    # Class names (default to ID classes from config if available)
    class_names = _resolve_target_classes(cfg, args.class_names)
    if not class_names:
        print("Error: No class names. Set --class_names or config.data.class_names.")
        return
    _log(f"Resolved target classes: {len(class_names)}")
    _log(f"Target class list: {class_names}")

    # Number of outputs
    num_q = args.num_questions or cfg.llm_descriptions.get("num_questions", 10)
    num_d = args.num_descriptions or cfg.llm_descriptions.get("num_descriptions_per_class", 5)
    _log(f"Generation targets: num_questions={num_q}, num_descriptions_per_class={num_d}")

    # Output paths
    questions_out = cfg.llm_descriptions.get("questions_output_file", "data/prompts/class_questions.yaml")
    descriptions_out = cfg.llm_descriptions.get("output_file", "data/prompts/class_descriptions.yaml")
    glali_json_out = (
        args.glali_json_out
        or cfg.llm_descriptions.get("glali_output_file")
        or cfg.llm_descriptions.get("descriptions_json_file")
        or str(Path(descriptions_out).with_suffix(".json"))
    )
    _log(f"Output files: questions={questions_out}, yaml={descriptions_out}, json={glali_json_out}")

    # Dataset description (from config, or fallback to inline)
    dataset_desc = cfg.llm_descriptions.get(
        "dataset_description",
        "A bone X-ray dataset with multiple bone condition classes."
    )
    # Allow lightweight templating in config strings, e.g. "{num_classes}".
    try:
        dataset_desc = dataset_desc.format(num_classes=len(class_names))
    except Exception:
        # Keep original string if it contains other braces/placeholders.
        pass

    # Prompt templates from config (if provided)
    question_prompt = cfg.llm_descriptions.get("question_prompt_template")
    desc_prompt = cfg.llm_descriptions.get("description_prompt_template")
    _log(
        "Prompt templates: "
        f"question={'custom' if question_prompt else 'default'}, "
        f"description={'custom' if desc_prompt else 'default'}"
    )

    # Build LLM wrapper from config
    llm_cfg = cfg.model.llm
    effective_cache_dir = llm_cfg.get("cache_dir") or os.getenv("HF_CACHE_DIR")
    _log(f"Initializing LLM wrapper with model={llm_cfg.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')} ...")
    _log(f"LLM cache_dir: {effective_cache_dir or 'default_hf_cache'}")
    llm = LLMWrapper(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device_map=llm_cfg.get("device_map", "auto"),
        max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        temperature=llm_cfg.get("temperature", 0.7),
        top_p=llm_cfg.get("top_p", 0.9),
        repetition_penalty=llm_cfg.get("repetition_penalty", 1.1),
        cache_dir=effective_cache_dir,
        torch_dtype=getattr(
            __import__("torch"),
            llm_cfg.get("torch_dtype", "float16"),
        ),
        trust_remote_code=llm_cfg.get("trust_remote_code", True),
    )
    _log("LLM wrapper initialized. Model weights will be loaded lazily on first generation call.")

    # -------------------------------------------------------------------
    # Step 1: Generate questions
    # -------------------------------------------------------------------
    step1_start = time.time()
    if args.skip_questions:
        import yaml
        questions_path = Path(questions_out)
        if questions_path.exists():
            _log(f"Step 1/3: Loading existing questions from {questions_out} ...")
            with open(questions_path) as f:
                data = yaml.safe_load(f) or {}
            questions = data.get("questions", [])
            _log(f"Loaded {len(questions)} existing questions from {questions_out}")
        else:
            _log(f"Warning: --skip_questions but {questions_out} not found. Generating new questions...")
            questions = llm.generate_questions(
                dataset_description=dataset_desc,
                num_questions=num_q,
                prompt_template=question_prompt,
            )
    else:
        _log(f"Step 1/3: Generating {num_q} discriminative questions...")
        questions = llm.generate_questions(
            dataset_description=dataset_desc,
            num_questions=num_q,
            prompt_template=question_prompt,
        )
        _log(f"Generated {len(questions)} questions:")
        for q in questions:
            print(f"  - {q}")
        llm.save_questions(questions, questions_out)
        _log(f"Saved questions to {questions_out}")
    _log(f"Step 1/3 completed in {time.time() - step1_start:.1f}s.")

    # -------------------------------------------------------------------
    # Step 2: Generate descriptions per class
    # -------------------------------------------------------------------
    step2_start = time.time()
    _log(f"Step 2/3: Generating {num_d} descriptions per class for {len(class_names)} classes...")
    descriptions: dict = {}
    for idx, cls_name in enumerate(class_names, start=1):
        class_start = time.time()
        _log(f"[Class {idx}/{len(class_names)}] Generating descriptions for '{cls_name}' ...")
        descs = llm.generate_descriptions(
            class_name=cls_name,
            num_descriptions=num_d,
            dataset_description=dataset_desc,
            questions=questions,
            prompt_template=desc_prompt,
        )
        descriptions[cls_name] = descs
        _log(
            f"[Class {idx}/{len(class_names)}] Done '{cls_name}': "
            f"{len(descs)} descriptions in {time.time() - class_start:.1f}s."
        )
        print(f"[main]   -> {len(descs)} descriptions:")
        for d in descs:
            print(f"      {d}")
    _log(f"Step 2/3 completed in {time.time() - step2_start:.1f}s.")

    # -------------------------------------------------------------------
    # Step 3: Save descriptions
    # -------------------------------------------------------------------
    step3_start = time.time()
    _log("Step 3/3: Saving generated description files...")
    llm.save_descriptions(descriptions, descriptions_out)
    _save_glali_like_json(descriptions, glali_json_out)
    _log(f"Step 3/3 completed in {time.time() - step3_start:.1f}s.")

    _log(f"Done! Total runtime: {time.time() - start_time:.1f}s.")
    print(f"[main] Questions:       {questions_out}")
    print(f"[main] Descriptions:    {descriptions_out}")
    print(f"[main] Glali JSON:      {glali_json_out}")


if __name__ == "__main__":
    main()
