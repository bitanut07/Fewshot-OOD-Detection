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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.models.encoders.llm_wrapper import LLMWrapper


def main():
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
    args = parser.parse_args()

    # Load merged config
    cfg = load_config(args.config)

    # Class names
    class_names = args.class_names or cfg.data.get("class_names", [])
    if not class_names:
        print("Error: No class names. Set --class_names or config.data.class_names.")
        return

    # Number of outputs
    num_q = args.num_questions or cfg.llm_descriptions.get("num_questions", 10)
    num_d = args.num_descriptions or cfg.llm_descriptions.get("num_descriptions_per_class", 5)

    # Output paths
    questions_out = cfg.llm_descriptions.get("questions_output_file", "data/prompts/class_questions.yaml")
    descriptions_out = cfg.llm_descriptions.get("output_file", "data/prompts/class_descriptions.yaml")

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

    # Build LLM wrapper from config
    llm_cfg = cfg.model.llm
    llm = LLMWrapper(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device_map=llm_cfg.get("device_map", "auto"),
        max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        temperature=llm_cfg.get("temperature", 0.7),
        top_p=llm_cfg.get("top_p", 0.9),
        repetition_penalty=llm_cfg.get("repetition_penalty", 1.1),
        cache_dir=llm_cfg.get("cache_dir"),
        torch_dtype=getattr(
            __import__("torch"),
            llm_cfg.get("torch_dtype", "float16"),
        ),
        trust_remote_code=llm_cfg.get("trust_remote_code", True),
    )

    # -------------------------------------------------------------------
    # Step 1: Generate questions
    # -------------------------------------------------------------------
    if args.skip_questions:
        import yaml
        questions_path = Path(questions_out)
        if questions_path.exists():
            with open(questions_path) as f:
                data = yaml.safe_load(f) or {}
            questions = data.get("questions", [])
            print(f"[main] Loaded {len(questions)} existing questions from {questions_out}")
        else:
            print(f"[main] Warning: --skip_questions but {questions_out} not found. Generating...")
            questions = llm.generate_questions(
                dataset_description=dataset_desc,
                num_questions=num_q,
                prompt_template=question_prompt,
            )
    else:
        print(f"[main] Generating {num_q} discriminative questions...")
        questions = llm.generate_questions(
            dataset_description=dataset_desc,
            num_questions=num_q,
            prompt_template=question_prompt,
        )
        print(f"[main] Generated {len(questions)} questions:")
        for q in questions:
            print(f"  - {q}")
        llm.save_questions(questions, questions_out)

    # -------------------------------------------------------------------
    # Step 2: Generate descriptions per class
    # -------------------------------------------------------------------
    print(f"\n[main] Generating {num_d} descriptions per class for {len(class_names)} classes...")
    descriptions: dict = {}
    for cls_name in class_names:
        print(f"\n[main] Class: {cls_name}")
        descs = llm.generate_descriptions(
            class_name=cls_name,
            num_descriptions=num_d,
            dataset_description=dataset_desc,
            questions=questions,
            prompt_template=desc_prompt,
        )
        descriptions[cls_name] = descs
        print(f"[main]   -> {len(descs)} descriptions:")
        for d in descs:
            print(f"      {d}")

    # -------------------------------------------------------------------
    # Step 3: Save descriptions
    # -------------------------------------------------------------------
    llm.save_descriptions(descriptions, descriptions_out)

    print(f"\n[main] Done!")
    print(f"  Questions:       {questions_out}")
    print(f"  Descriptions:    {descriptions_out}")


if __name__ == "__main__":
    main()
