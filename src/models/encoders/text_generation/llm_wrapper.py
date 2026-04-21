# -*- coding: utf-8 -*-
"""LLMWrapper — top-level facade for offline text generation.

Composes:
    - ``HFLocalGenerator``  (or any ``BaseTextGenerator``)
    - ``PromptBuilder``
    - ``OutputCleaner``
    - ``CacheManager``

Generates text **only for known in-distribution (ID) classes**.
OOD detection relies on poor alignment with known-class embeddings;
no OOD text is ever generated or stored.

Workflow:
    1. Generate dataset-level discriminative questions
    2. For each known class → generate M visual descriptions
    3. Save structured output with metadata
    4. Reload from cache for training / evaluation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .base_generator import BaseTextGenerator, GenerationConfig
from .cache_manager import CacheManager, build_class_entry, build_output_payload
from .hf_local_generator import HFLocalGenerator
from .output_cleaner import OutputCleaner
from .prompt_builder import PromptBuilder

log = logging.getLogger(__name__)


class LLMWrapper:
    """Facade for offline LLM-based description generation.

    Parameters
    ----------
    generator:
        A ``BaseTextGenerator`` backend.  If ``None`` a default
        ``HFLocalGenerator`` is created from the keyword arguments.
    generation_config:
        Default generation parameters (temperature, top_p, …).
    prompt_builder:
        Builds prompts for questions and descriptions.
    output_cleaner:
        Post-processes raw LLM output into clean lines.
    cache_manager:
        Handles load / save / validation of cached results.
    max_retries:
        How many times to re-call the LLM when fewer than the requested
        number of valid lines are returned.

    Convenience keyword arguments (forwarded to ``HFLocalGenerator`` when
    *generator* is ``None``):
        model_name, device_map, torch_dtype, cache_dir, trust_remote_code
    """

    def __init__(
        self,
        generator: Optional[BaseTextGenerator] = None,
        generation_config: Optional[GenerationConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        output_cleaner: Optional[OutputCleaner] = None,
        cache_manager: Optional[CacheManager] = None,
        max_retries: int = 3,
        # HFLocalGenerator convenience kwargs
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "float16",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        # GenerationConfig convenience kwargs
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = 42,
        deterministic: bool = False,
        # PromptBuilder convenience kwargs
        dataset_description: str = "",
        default_prompt_template: Optional[str] = None,
        question_prompt_template: Optional[str] = None,
        description_prompt_template: Optional[str] = None,
    ) -> None:
        # -- Generator ---------------------------------------------------------
        self.generator: BaseTextGenerator = generator or HFLocalGenerator(
            model_name_or_path=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # -- Generation config -------------------------------------------------
        self.gen_config = generation_config or GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
        )

        # -- Prompt builder ----------------------------------------------------
        self.prompt_builder = prompt_builder or PromptBuilder(
            dataset_description=dataset_description,
            default_prompt_template=default_prompt_template,
            question_prompt_template=question_prompt_template,
            description_prompt_template=description_prompt_template,
        )

        # -- Output cleaner ----------------------------------------------------
        self.cleaner = output_cleaner or OutputCleaner()

        # -- Cache manager -----------------------------------------------------
        self.cache = cache_manager or CacheManager()

        self.max_retries = max_retries

    # =====================================================================
    # Question generation
    # =====================================================================

    def generate_questions(self, num_questions: int = 10) -> List[str]:
        """Generate *num_questions* discriminative questions."""
        log.info("Generating %d discriminative questions …", num_questions)
        prompt = self.prompt_builder.build_question_prompt(num_questions)

        all_lines: List[str] = []
        for attempt in range(1, self.max_retries + 1):
            remaining = num_questions - len(all_lines)
            if remaining <= 0:
                break

            cfg = GenerationConfig(
                max_new_tokens=self.gen_config.max_new_tokens + 128 * (attempt - 1),
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                repetition_penalty=self.gen_config.repetition_penalty,
                seed=self.gen_config.seed,
                deterministic=self.gen_config.deterministic,
            )
            raw = self.generator.generate(prompt, config=cfg)
            new = self.cleaner.clean(raw, existing=all_lines)
            all_lines.extend(new)

            if len(all_lines) >= num_questions:
                break
            log.warning(
                "Attempt %d/%d: got %d/%d questions",
                attempt, self.max_retries, len(all_lines), num_questions,
            )

        if len(all_lines) < num_questions:
            log.warning(
                "Only %d/%d questions after %d retries",
                len(all_lines), num_questions, self.max_retries,
            )
        return all_lines[:num_questions]

    # =====================================================================
    # Description generation (ID classes only)
    # =====================================================================

    def generate_descriptions(
        self,
        class_name: str,
        num_descriptions: int = 8,
        questions: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate *num_descriptions* visual descriptions for one **known** class."""
        log.info("Generating %d descriptions for '%s' …", num_descriptions, class_name)

        prompt = self.prompt_builder.build_description_prompt(
            class_name=class_name,
            num_descriptions=num_descriptions,
            questions=questions,
        )

        all_lines: List[str] = []
        for attempt in range(1, self.max_retries + 1):
            remaining = num_descriptions - len(all_lines)
            if remaining <= 0:
                break

            request_n = remaining + 2 if attempt > 1 else remaining
            retry_prompt = prompt.replace(
                f"exactly {num_descriptions} diverse",
                f"exactly {request_n} diverse",
            ) if attempt > 1 else prompt

            cfg = GenerationConfig(
                max_new_tokens=self.gen_config.max_new_tokens + 128 * (attempt - 1),
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                repetition_penalty=self.gen_config.repetition_penalty,
                seed=self.gen_config.seed,
                deterministic=self.gen_config.deterministic,
            )
            raw = self.generator.generate(retry_prompt, config=cfg)
            new = self.cleaner.clean(raw, existing=all_lines)
            all_lines.extend(new)

            if len(all_lines) >= num_descriptions:
                break
            log.warning(
                "Attempt %d/%d for '%s': got %d/%d",
                attempt, self.max_retries, class_name,
                len(all_lines), num_descriptions,
            )

        if len(all_lines) < num_descriptions:
            log.warning(
                "Only %d/%d descriptions for '%s' after %d retries",
                len(all_lines), num_descriptions, class_name, self.max_retries,
            )
        return all_lines[:num_descriptions]

    # =====================================================================
    # Full pipeline: questions + descriptions for all known classes
    # =====================================================================

    def generate_all(
        self,
        class_names: List[str],
        dataset_name: str = "bone_xray",
        num_questions: int = 10,
        num_descriptions: int = 8,
        force_regenerate: bool = False,
    ) -> dict:
        """Generate questions + descriptions for all **known ID classes**.

        Cache-first: skips generation if valid cache exists, unless
        *force_regenerate* is ``True``.

        Returns the full structured payload (schema v2).
        """
        if not force_regenerate and self.cache.cache_valid(class_names, num_descriptions):
            log.info("Valid cache found — loading from disk.")
            return self.cache.load_descriptions()

        # Step 1: questions
        questions = self.generate_questions(num_questions)
        self.cache.save_questions(questions)

        # Step 2: per-class descriptions
        classes_data: Dict[str, dict] = {}
        flat: Dict[str, List[str]] = {}
        for idx, cls_name in enumerate(class_names, 1):
            log.info("[%d/%d] Generating for '%s'", idx, len(class_names), cls_name)
            descs = self.generate_descriptions(cls_name, num_descriptions, questions)
            default_prompt = self.prompt_builder.default_prompt(cls_name)
            classes_data[cls_name] = build_class_entry(
                default_prompts=[default_prompt],
                generated_descriptions=descs,
            )
            flat[cls_name] = descs
            log.info(
                "[%d/%d] '%s' done — %d descriptions",
                idx, len(class_names), cls_name, len(descs),
            )

        # Step 3: build & save structured payload
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
        self.cache.save_flat_json(flat)

        return payload

    # =====================================================================
    # Convenience loaders (delegate to cache)
    # =====================================================================

    def load_questions(self) -> List[str]:
        return self.cache.load_questions()

    def load_descriptions(self) -> dict:
        return self.cache.load_descriptions()

    def load_flat_descriptions(self) -> Dict[str, List[str]]:
        return self.cache.load_flat_descriptions()

    def load_all(self) -> dict:
        return self.cache.load_all()

    def questions_exist(self) -> bool:
        return self.cache.questions_exist()

    def descriptions_exist(self) -> bool:
        return self.cache.descriptions_exist()

    # =====================================================================
    # Legacy save helpers (backward compatibility)
    # =====================================================================

    def save_questions(self, questions: List[str], output_path: str) -> None:
        cm = CacheManager(questions_path=output_path)
        cm.save_questions(questions)

    def save_descriptions(self, descriptions: Dict[str, List[str]], output_path: str) -> None:
        from pathlib import Path
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import yaml as _yaml
        with open(p, "w", encoding="utf-8") as f:
            _yaml.dump(descriptions, f, default_flow_style=False,
                       allow_unicode=True, sort_keys=False)
        log.info("Legacy descriptions saved → %s", p)
