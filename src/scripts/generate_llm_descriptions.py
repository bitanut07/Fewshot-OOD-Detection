#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LLM-based disease descriptions for all classes.

Run this script OFFLINE before training to generate disease descriptions.
Output is saved to data/prompts/class_descriptions.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.models.encoders.llm_wrapper import LLMWrapper
import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate LLM disease descriptions")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--class_names", type=str, nargs="+", default=None, help="Class names")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = args.class_names or config.data.get("class_names", [])

    if not class_names:
        print("Error: No class names provided. Please specify --class_names or set in config.")
        return

    llm = LLMWrapper(
        model_name=config.model.llm.model_name,
        max_new_tokens=config.model.llm.max_new_tokens,
        temperature=config.model.llm.temperature,
        top_p=config.model.llm.top_p,
    )

    results = {}
    num_desc = config.llm_descriptions.num_descriptions_per_class

    for cls_name in class_names:
        print(f"Generating descriptions for: {cls_name}")
        descs = llm.generate_descriptions(cls_name, num_descriptions=num_desc)
        results[cls_name] = descs
        print(f"  Generated {len(descs)} descriptions")

    output_path = args.output or config.llm_descriptions.output_file
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"Descriptions saved to: {output_path}")


if __name__ == "__main__":
    main()
