# -*- coding: utf-8 -*-
"""LLM Wrapper for generating disease descriptions and discriminative questions.

Wraps an LLM (e.g., Qwen, LLaMA) to generate:
  1. Discriminative questions that highlight key visual differences between disease classes
  2. Diverse, clinically accurate visual descriptions per disease class

These are used OFFLINE before training. The generated outputs are consumed by
the CLIP text encoder and the text refinement module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------------------------------
# Generic-line filters
# -------------------------------------------------------------------------
_GENERIC_PATTERNS = [
    re.compile(r"^this (x-?ray|image|photo|scan) (shows?|displays?|reveals?|indicates?)", re.IGNORECASE),
    re.compile(r"^the (x-?ray|image|photo|scan|bone|patient) (may|might|could)", re.IGNORECASE),
    re.compile(r"^visible (in this|on the) (x-?ray|image|photo|scan)", re.IGNORECASE),
    re.compile(r"^a (radiograph|x-?ray|image|photo|scan) of", re.IGNORECASE),
    re.compile(r"^the (bone|joint|bone\s+(?:x-?ray|image|scan|film)) (shows?|displays?|reveals?)", re.IGNORECASE),
    re.compile(r"^(may|possibly|might|can|could) (be|show|appear|indicate|suggest)", re.IGNORECASE),
]


def _is_generic_line(line: str) -> bool:
    """Return True if a line is too generic to be useful for CLIP embeddings."""
    line_lower = line.lower().strip()
    if len(line_lower.split()) < 4:
        return True
    for pat in _GENERIC_PATTERNS:
        if pat.match(line_lower):
            return True
    return False


class LLMWrapper:
    """
    LLM Wrapper for disease description and question generation.

    Workflow:
        1. Generate discriminative questions from dataset description
        2. For each class, generate descriptions guided by those questions
        3. Save results to YAML for later use by CLIP text encoder

    The module is used OFFLINE only (frozen during training).

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
        device_map: Device mapping strategy (e.g., "auto", "cuda")
        max_new_tokens: Max tokens to generate per call
        temperature: Sampling temperature (0 = greedy, >0 = sampling)
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
        cache_dir: Cache directory for model weights
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        cache_dir: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._device_map = device_map
        self._cache_dir = cache_dir
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code

        # Lazy-loaded attributes
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the LLM and tokenizer on first use."""
        if self.model is not None:
            return
        print(f"[LLMWrapper] Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )
        self.model.eval()
        print(f"[LLMWrapper] Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Internal generation
    # -------------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Internal generation method. Wraps prompt in chat template and calls the model.

        Args:
            prompt: The user prompt content.
            max_new_tokens, temperature, top_p, repetition_penalty: Override defaults if provided.

        Returns:
            Raw assistant response string.
        """
        self._load_model()

        mt = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        tp = temperature if temperature is not None else self.temperature
        pp = top_p if top_p is not None else self.top_p
        rp = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

        # Only use sampling when temperature > 0
        do_sample = tp > 0.0

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=mt,
                temperature=tp,
                top_p=pp,
                do_sample=do_sample,
                repetition_penalty=rp,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant portion — split on last "assistant" marker
        marker_pos = full_output.rfind("assistant")
        if marker_pos != -1:
            response = full_output[marker_pos + len("assistant"):].strip()
        else:
            # Fallback: strip the prompt portion
            response = full_output[len(text):].strip()

        return response

    # -------------------------------------------------------------------------
    # Output parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _strip_prefix(line: str) -> str:
        """Remove common numbering and bullet prefixes from a line."""
        line = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[\-\*\•]\s+", "", line)
        return line.strip()

    @staticmethod
    def _parse_lines(
        raw_text: str,
        min_len: int = 15,
        max_len: int = 250,
        max_lines: Optional[int] = None,
    ) -> List[str]:
        """
        Parse model output into clean lines suitable for CLIP embeddings.

        Steps:
          1. Split on newlines
          2. Strip each line
          3. Remove numbering (1., 1), bullets (-, *, •)
          4. Skip lines that are too short, too long, or too generic

        Args:
            raw_text: Raw model output string
            min_len: Minimum character length to accept a line
            max_len: Maximum character length to accept a line
            max_lines: Stop once this many clean lines are collected

        Returns:
            List of clean, non-generic lines
        """
        lines: List[str] = []
        for raw_line in raw_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            line = LLMWrapper._strip_prefix(line)

            if len(line) < min_len or len(line) > max_len:
                continue
            if _is_generic_line(line):
                continue

            lines.append(line)

            if max_lines is not None and len(lines) >= max_lines:
                break

        return lines

    # -------------------------------------------------------------------------
    # Question generation
    # -------------------------------------------------------------------------

    def generate_questions(
        self,
        dataset_description: str,
        num_questions: int = 10,
        prompt_template: Optional[str] = None,
    ) -> List[str]:
        """
        Generate discriminative questions based on dataset description.

        These questions highlight what visual features distinguish disease classes,
        serving as context for subsequent description generation.

        Args:
            dataset_description: Text describing the dataset domain and classes
            num_questions: Number of questions to generate
            prompt_template: Custom prompt. If None, uses the default prompt.

        Returns:
            List of question strings (one per line, no numbering)
        """
        if prompt_template is None:
            prompt_template = (
                "You are a medical imaging expert specializing in bone X-ray analysis.\n\n"
                "Dataset description:\n{dataset_description}\n\n"
                "Task:\n"
                "Generate {num_questions} diagnostic questions that help distinguish "
                "between different bone diseases in X-ray images.\n\n"
                "Requirements:\n"
                "- Focus ONLY on visible radiographic features\n"
                "- Do NOT include diagnosis, treatment, or patient history\n"
                "- Questions must help differentiate diseases visually\n"
                "- Avoid referring to specific absolute positions "
                "(e.g., left/right/top/bottom)\n\n"
                "Focus on:\n"
                "- bone structure and continuity\n"
                "- fracture patterns and cortical disruption\n"
                "- lesion characteristics (e.g., lytic or sclerotic changes)\n"
                "- bone density variations\n"
                "- structural deformities and alignment abnormalities\n"
                "- presence of abnormal growths or irregular bone shapes\n\n"
                "Output:\n"
                "- One question per line\n"
                "- No numbering\n"
                "- No explanations"
            )

        prompt = prompt_template.format(
            num_questions=num_questions,
            dataset_description=dataset_description,
        )

        raw = self._generate(prompt)
        lines = self._parse_lines(raw, min_len=15, max_len=200, max_lines=num_questions)

        if len(lines) < num_questions:
            print(
                f"[LLMWrapper] Warning: got {len(lines)} questions, expected {num_questions}."
            )
        return lines[:num_questions]

    # -------------------------------------------------------------------------
    # Description generation
    # -------------------------------------------------------------------------

    def generate_descriptions(
        self,
        class_name: str,
        num_descriptions: int = 5,
        dataset_description: Optional[str] = None,
        questions: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ) -> List[str]:
        """
        Generate diverse visual descriptions for a single disease class.

        Descriptions focus exclusively on visible radiographic features, suitable for
        CLIP text encoder embeddings.

        Args:
            class_name: Name of the disease class
            num_descriptions: Number of descriptions to generate
            dataset_description: Optional dataset context for richer prompts
            questions: Optional discriminative questions to guide generation
            prompt_template: Custom prompt. If None, uses the default prompt.

        Returns:
            List of description strings (one per line, no numbering)
        """
        if prompt_template is None:
            prompt_template = (
                "You are a medical imaging expert specializing in bone X-ray interpretation.\n\n"
                "Target disease class: \"{class_name}\"\n\n"
                "{questions_block}"
                "Task:\n"
                "Generate {num_descriptions} diverse and clinically accurate visual "
                "descriptions of this disease as seen in bone X-ray images.\n\n"
                "Requirements:\n"
                "- Each description must be ONE sentence\n"
                "- Focus ONLY on visible radiographic features\n"
                "- Avoid diagnosis, treatment, or clinical symptoms outside the image\n"
                "- Avoid uncertainty words (e.g., \"may\", \"possibly\")\n"
                "- Avoid absolute spatial positions (e.g., left/right/top/bottom)\n"
                "- Each description must highlight a distinct visual feature\n"
                "- Use clear and professional medical language\n"
                "- Prefer concise descriptions suitable for a vision-language model\n\n"
                "Focus on:\n"
                "- bone structure and integrity\n"
                "- fracture lines and cortical disruption (if present)\n"
                "- lesion characteristics (e.g., radiolucent or radiopaque regions)\n"
                "- bone density variations\n"
                "- abnormal growths or masses\n"
                "- structural deformities and alignment changes\n"
                "- trabecular pattern changes\n"
                "- bone surface irregularities\n\n"
                "Output:\n"
                "- One description per line\n"
                "- No numbering\n"
                "- No explanations"
            )

        if questions:
            questions_block = (
                "Context:\n"
                "Below are diagnostic questions used to distinguish bone diseases:\n"
                + "\n".join(f"- {q}" for q in questions[:10])
                + "\n\n"
            )
        elif dataset_description:
            questions_block = f"Dataset context: {dataset_description}\n\n"
        else:
            questions_block = ""

        prompt = prompt_template.format(
            class_name=class_name,
            num_descriptions=num_descriptions,
            questions_block=questions_block,
        )

        raw = self._generate(prompt)
        lines = self._parse_lines(raw, min_len=15, max_len=250, max_lines=num_descriptions)

        if len(lines) < num_descriptions:
            print(
                f"[LLMWrapper] Warning: got {len(lines)} descriptions for "
                f"'{class_name}', expected {num_descriptions}."
            )

        return lines[:num_descriptions]

    def generate_all_descriptions(
        self,
        class_names: List[str],
        dataset_description: str,
        num_descriptions_per_class: int = 5,
        num_questions: int = 10,
        prompt_template: Optional[str] = None,
        description_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: generate questions then descriptions for all classes.

        Args:
            class_names: List of all class names to generate descriptions for
            dataset_description: Dataset context passed to both question and description prompts
            num_descriptions_per_class: How many descriptions per class
            num_questions: How many discriminative questions to generate
            prompt_template: Custom question prompt template (None = use default)
            description_prompt_template: Custom description prompt template (None = use default)

        Returns:
            Dict with keys:
              - "questions": List[str] of generated questions
              - "descriptions": Dict[class_name -> List[str]]
        """
        # Step 1: generate questions once for the whole dataset
        questions = self.generate_questions(
            dataset_description=dataset_description,
            num_questions=num_questions,
            prompt_template=prompt_template,
        )
        print(f"[LLMWrapper] Generated {len(questions)} questions.")

        # Step 2: generate descriptions per class, guided by questions
        descriptions: Dict[str, List[str]] = {}
        for cls_name in class_names:
            descs = self.generate_descriptions(
                class_name=cls_name,
                num_descriptions=num_descriptions_per_class,
                dataset_description=dataset_description,
                questions=questions,
                prompt_template=description_prompt_template,
            )
            descriptions[cls_name] = descs
            print(f"[LLMWrapper] '{cls_name}': {len(descs)} descriptions.")

        return {"questions": questions, "descriptions": descriptions}

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_questions(
        self,
        questions: List[str],
        output_path: Union[str, Path],
        format: str = "yaml",
    ) -> None:
        """
        Save generated questions to a file.

        Args:
            questions: List of question strings
            output_path: Path to save the file
            format: "yaml" or "json"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"questions": questions}

        if format == "yaml":
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: '{format}'. Use 'yaml' or 'json'.")

        print(f"[LLMWrapper] Questions saved to: {output_path}")

    def save_descriptions(
        self,
        descriptions: Dict[str, List[str]],
        output_path: Union[str, Path],
        format: str = "yaml",
    ) -> None:
        """
        Save class descriptions to a file.

        Args:
            descriptions: Dict mapping class_name -> list of description strings
            output_path: Path to save the file
            format: "yaml" or "json"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(descriptions, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: '{format}'. Use 'yaml' or 'json'.")

        print(f"[LLMWrapper] Descriptions saved to: {output_path}")

    def save_all(
        self,
        questions: List[str],
        descriptions: Dict[str, List[str]],
        questions_output_path: Union[str, Path],
        descriptions_output_path: Union[str, Path],
        questions_format: str = "yaml",
        descriptions_format: str = "yaml",
    ) -> None:
        """Save both questions and descriptions to their respective files."""
        self.save_questions(questions, questions_output_path, format=questions_format)
        self.save_descriptions(descriptions, descriptions_output_path, format=descriptions_format)
