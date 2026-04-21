# -*- coding: utf-8 -*-
"""LLMWrapper — Multi-stage discriminative description generation.

Implements a two-stage generation pipeline for CLIP-friendly bone X-ray
classification descriptions:

    Stage A: Extract discriminative attributes for each class
    Stage B: Generate final descriptions from attributes with cross-class awareness

Generates text **only for known in-distribution (ID) classes**.
OOD detection relies on poor alignment with known-class embeddings;
no OOD text is ever generated.

Key improvements over single-stage generation:
    - Cross-class awareness in all prompts
    - Attribute-guided description generation
    - Quality scoring and filtering
    - Targeted retries with context
    - Cross-class deduplication
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_generator import BaseTextGenerator, GenerationConfig
from .cache_manager import CacheManager, build_class_entry, build_output_payload
from .description_scorer import DescriptionScorer
from .hf_local_generator import HFLocalGenerator
from .output_cleaner import OutputCleaner
from .prompt_builder import PromptBuilder

log = logging.getLogger(__name__)


class LLMWrapper:
    """Multi-stage discriminative description generator.

    Generation workflow:
        1. Generate dataset-level discriminative questions
        2. For each known ID class:
           a. Stage A: Generate discriminative attributes
           b. Stage B: Generate descriptions guided by attributes
           c. Score, filter, and validate descriptions
        3. Cross-class deduplication pass
        4. Save structured output with metadata

    All generation is cross-class aware to maximize discriminability.
    """

    def __init__(
        self,
        generator: Optional[BaseTextGenerator] = None,
        generation_config: Optional[GenerationConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        output_cleaner: Optional[OutputCleaner] = None,
        description_scorer: Optional[DescriptionScorer] = None,
        cache_manager: Optional[CacheManager] = None,
        max_retries: int = 3,
        # HFLocalGenerator kwargs
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "float16",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        # GenerationConfig kwargs
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = 42,
        deterministic: bool = False,
        # PromptBuilder kwargs
        dataset_description: str = "",
    ) -> None:
        # Generator backend
        self.generator: BaseTextGenerator = generator or HFLocalGenerator(
            model_name_or_path=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # Generation config
        self.gen_config = generation_config or GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
        )

        # Prompt builder
        self.prompt_builder = prompt_builder or PromptBuilder(
            dataset_description=dataset_description,
        )

        # Output cleaner
        self.cleaner = output_cleaner or OutputCleaner()

        # Description scorer
        self.scorer = description_scorer or DescriptionScorer()

        # Cache manager
        self.cache = cache_manager or CacheManager()

        self.max_retries = max_retries

    # =========================================================================
    # Step 1: Question generation
    # =========================================================================

    def generate_questions(
        self,
        num_questions: int = 10,
        class_names: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate discriminative questions with cross-class awareness."""
        log.info("Generating %d discriminative questions …", num_questions)

        prompt = self.prompt_builder.build_question_prompt(
            num_questions=num_questions,
            class_names=class_names,
        )

        all_lines: List[str] = []
        for attempt in range(1, self.max_retries + 1):
            remaining = num_questions - len(all_lines)
            if remaining <= 0:
                break

            cfg = self._make_config(extra_tokens=128 * (attempt - 1))
            raw = self.generator.generate(prompt, config=cfg)
            new = self.cleaner.clean(raw, existing=all_lines)
            all_lines.extend(new)

            if len(all_lines) >= num_questions:
                break

            log.warning(
                "Question attempt %d/%d: got %d/%d",
                attempt, self.max_retries, len(all_lines), num_questions,
            )

        if len(all_lines) < num_questions:
            log.warning(
                "Only %d/%d questions after %d retries",
                len(all_lines), num_questions, self.max_retries,
            )

        return all_lines[:num_questions]

    # =========================================================================
    # Step 2a: Stage A — Attribute extraction
    # =========================================================================

    def generate_attributes(
        self,
        class_name: str,
        other_classes: List[str],
        num_attributes: int = 6,
    ) -> List[str]:
        """Stage A: Extract discriminative attributes for a class."""
        log.info("Stage A: Extracting %d attributes for '%s' …", num_attributes, class_name)

        prompt = self.prompt_builder.build_attribute_prompt(
            class_name=class_name,
            other_classes=other_classes,
            num_attributes=num_attributes,
        )

        all_attrs: List[str] = []
        for attempt in range(1, self.max_retries + 1):
            remaining = num_attributes - len(all_attrs)
            if remaining <= 0:
                break

            cfg = self._make_config(extra_tokens=64 * (attempt - 1))
            raw = self.generator.generate(prompt, config=cfg)
            new = self.cleaner.clean_attributes(raw)

            for attr in new:
                if attr not in all_attrs:
                    all_attrs.append(attr)
                if len(all_attrs) >= num_attributes:
                    break

            if len(all_attrs) >= num_attributes:
                break

            log.warning(
                "Attribute attempt %d/%d for '%s': got %d/%d",
                attempt, self.max_retries, class_name, len(all_attrs), num_attributes,
            )

        return all_attrs[:num_attributes]

    # =========================================================================
    # Step 2b: Stage B — Description generation from attributes
    # =========================================================================

    def generate_descriptions(
        self,
        class_name: str,
        num_descriptions: int = 8,
        other_classes: Optional[List[str]] = None,
        attributes: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
    ) -> List[str]:
        """Stage B: Generate descriptions with cross-class awareness and optional attributes."""
        log.info(
            "Stage B: Generating %d descriptions for '%s' (attrs=%d) …",
            num_descriptions, class_name, len(attributes) if attributes else 0,
        )

        prompt = self.prompt_builder.build_description_prompt(
            class_name=class_name,
            num_descriptions=num_descriptions,
            other_classes=other_classes,
            attributes=attributes,
            questions=questions,
        )

        all_lines: List[str] = []

        # First pass: generate with main prompt
        for attempt in range(1, self.max_retries + 1):
            remaining = num_descriptions - len(all_lines)
            if remaining <= 0:
                break

            cfg = self._make_config(extra_tokens=128 * (attempt - 1))
            raw = self.generator.generate(prompt, config=cfg)
            new = self.cleaner.clean(raw, existing=all_lines)

            # Score and filter
            scored_new = self.scorer.filter_and_rank(
                new, class_name=class_name, min_score=0.0,
            )

            for desc in scored_new:
                if desc not in all_lines:
                    all_lines.append(desc)
                if len(all_lines) >= num_descriptions:
                    break

            if len(all_lines) >= num_descriptions:
                break

            log.warning(
                "Description attempt %d/%d for '%s': got %d/%d",
                attempt, self.max_retries, class_name, len(all_lines), num_descriptions,
            )

        # Second pass: targeted retry for missing descriptions
        if len(all_lines) < num_descriptions:
            all_lines = self._targeted_retry(
                class_name=class_name,
                existing=all_lines,
                num_needed=num_descriptions - len(all_lines),
                other_classes=other_classes,
            )

        if len(all_lines) < num_descriptions:
            log.warning(
                "Only %d/%d descriptions for '%s' after all retries",
                len(all_lines), num_descriptions, class_name,
            )

        return all_lines[:num_descriptions]

    def _targeted_retry(
        self,
        class_name: str,
        existing: List[str],
        num_needed: int,
        other_classes: Optional[List[str]] = None,
    ) -> List[str]:
        """Targeted retry: generate more descriptions avoiding existing ones."""
        log.info("Targeted retry for '%s': need %d more", class_name, num_needed)

        prompt = self.prompt_builder.build_retry_prompt(
            class_name=class_name,
            num_needed=num_needed + 2,  # Request extra to account for filtering
            existing_descriptions=existing,
            other_classes=other_classes,
        )

        result = list(existing)

        for attempt in range(1, 3):  # 2 retry attempts
            remaining = num_needed - (len(result) - len(existing))
            if remaining <= 0:
                break

            cfg = self._make_config(extra_tokens=64 * attempt)
            raw = self.generator.generate(prompt, config=cfg)
            new = self.cleaner.clean(raw, existing=result)

            scored = self.scorer.filter_and_rank(new, class_name=class_name, min_score=0.0)

            for desc in scored:
                if desc not in result:
                    result.append(desc)
                if len(result) >= len(existing) + num_needed:
                    break

            if len(result) >= len(existing) + num_needed:
                break

        return result

    # =========================================================================
    # Full pipeline: questions + attributes + descriptions for all ID classes
    # =========================================================================

    def generate_all(
        self,
        class_names: List[str],
        dataset_name: str = "bone_xray",
        num_questions: int = 10,
        num_attributes: int = 6,
        num_descriptions: int = 8,
        force_regenerate: bool = False,
    ) -> dict:
        """Full multi-stage generation pipeline for all known ID classes.

        Workflow:
            1. Generate discriminative questions (cross-class aware)
            2. For each class:
               a. Stage A: Extract discriminative attributes
               b. Stage B: Generate descriptions from attributes
            3. Cross-class deduplication
            4. Save structured output
        """
        # Cache check
        if not force_regenerate and self.cache.cache_valid(class_names, num_descriptions):
            log.info("Valid cache found — loading from disk.")
            return self.cache.load_descriptions()

        log.info(
            "Starting multi-stage generation for %d classes (q=%d, a=%d, d=%d)",
            len(class_names), num_questions, num_attributes, num_descriptions,
        )

        # Step 1: Questions
        questions = self.generate_questions(num_questions, class_names)
        self.cache.save_questions(questions)
        log.info("Generated %d questions", len(questions))

        # Step 2: Per-class generation (attributes + descriptions)
        classes_data: Dict[str, dict] = {}
        all_descriptions: Dict[str, List[str]] = {}

        for idx, cls_name in enumerate(class_names, 1):
            log.info("[%d/%d] Processing '%s' …", idx, len(class_names), cls_name)

            other_classes = [c for c in class_names if c != cls_name]

            # Stage A: Attributes
            attributes = self.generate_attributes(
                class_name=cls_name,
                other_classes=other_classes,
                num_attributes=num_attributes,
            )
            log.info("  → %d attributes extracted", len(attributes))

            # Stage B: Descriptions
            descriptions = self.generate_descriptions(
                class_name=cls_name,
                num_descriptions=num_descriptions,
                other_classes=other_classes,
                attributes=attributes,
                questions=questions,
            )
            log.info("  → %d descriptions generated", len(descriptions))

            all_descriptions[cls_name] = descriptions

            default_prompt = self.prompt_builder.default_prompt(cls_name)
            classes_data[cls_name] = build_class_entry(
                default_prompt=default_prompt,
                attributes=attributes,
                descriptions=descriptions,
            )

        # Step 3: Cross-class deduplication
        log.info("Running cross-class deduplication …")
        deduped = self.cleaner.remove_cross_class_duplicates(all_descriptions, threshold=0.75)

        # Update classes_data with deduped descriptions
        for cls_name, descs in deduped.items():
            removed = len(all_descriptions[cls_name]) - len(descs)
            if removed > 0:
                log.info("  '%s': removed %d cross-class duplicates", cls_name, removed)
            classes_data[cls_name]["descriptions"] = descs
            classes_data[cls_name]["metadata"]["num_descriptions"] = len(descs)

        # Step 4: Build and save payload
        payload = build_output_payload(
            dataset_name=dataset_name,
            model_name=self.generator.model_name(),
            seed=self.gen_config.seed,
            generation_config=self.gen_config.as_dict(),
            class_names=class_names,
            questions=questions,
            classes=classes_data,
        )

        self.cache.save_descriptions(payload)

        # Save flat JSON for compatibility
        flat = {cls: data["descriptions"] for cls, data in classes_data.items()}
        self.cache.save_flat_json(flat)

        log.info("Generation complete!")
        return payload

    # =========================================================================
    # Convenience loaders
    # =========================================================================

    def load_questions(self) -> List[str]:
        return self.cache.load_questions()

    def load_descriptions(self) -> dict:
        return self.cache.load_descriptions()

    def load_flat_descriptions(self) -> Dict[str, List[str]]:
        return self.cache.load_flat_descriptions()

    def load_with_defaults(self) -> Dict[str, Dict[str, Any]]:
        return self.cache.load_with_defaults()

    def load_all(self) -> dict:
        return self.cache.load_all()

    def questions_exist(self) -> bool:
        return self.cache.questions_exist()

    def descriptions_exist(self) -> bool:
        return self.cache.descriptions_exist()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _make_config(self, extra_tokens: int = 0) -> GenerationConfig:
        """Create a GenerationConfig with optional extra tokens."""
        return GenerationConfig(
            max_new_tokens=self.gen_config.max_new_tokens + extra_tokens,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            repetition_penalty=self.gen_config.repetition_penalty,
            seed=self.gen_config.seed,
            deterministic=self.gen_config.deterministic,
        )

    # =========================================================================
    # Legacy compatibility
    # =========================================================================

    def save_questions(self, questions: List[str], output_path: str) -> None:
        cm = CacheManager(questions_path=output_path)
        cm.save_questions(questions)

    def save_descriptions(self, descriptions: Dict[str, List[str]], output_path: str) -> None:
        from pathlib import Path
        import yaml as _yaml
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _yaml.dump(descriptions, f, default_flow_style=False,
                       allow_unicode=True, sort_keys=False)
        log.info("Legacy descriptions saved → %s", p)
