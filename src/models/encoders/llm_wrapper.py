# -*- coding: utf-8 -*-
"""LLM Wrapper for generating disease descriptions.

Wraps an LLM (e.g., Qwen, LLaMA) to generate diverse, clinically
accurate visual descriptions for each disease class. Descriptions are
used by the text refinement module to improve text embeddings.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper:
    """
    LLM Wrapper for disease description generation.

    This module is used OFFLINE to generate descriptions per class.
    It is NOT part of the training pipeline (frozen).

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
        device_map: Device mapping strategy (e.g., "auto")
        max_new_tokens: Maximum tokens to generate per description
        temperature: Sampling temperature (0 = greedy, >0 = sampling)
        top_p: Nucleus sampling parameter
        cache_dir: Cache directory for model weights
        torch_dtype: Data type for model weights
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        cache_dir: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None

        # Lazy loading: load only when needed
        self._device_map = device_map
        self._cache_dir = cache_dir
        self._torch_dtype = torch_dtype

    def _load_model(self) -> None:
        """Lazy-load the LLM and tokenizer."""
        if self.model is not None:
            return
        print(f"Loading LLM: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
        )
        self.model.eval()

    def generate_descriptions(
        self,
        class_name: str,
        num_descriptions: int = 5,
        prompt_template: Optional[str] = None,
    ) -> List[str]:
        """
        Generate diverse descriptions for a single disease class.

        Args:
            class_name: Name of the disease class
            num_descriptions: Number of descriptions to generate
            prompt_template: Custom prompt template with {num} and {class_name} placeholders

        Returns:
            List of generated description strings
        """
        self._load_model()

        if prompt_template is None:
            prompt_template = (
                "You are a medical imaging expert. "
                "Generate {num} diverse, clinically accurate visual descriptions "
                "for the disease class: '{class_name}'. "
                "Each description should focus on distinct radiological or visual features. "
                "Output only the descriptions, one per line, without numbering."
            )

        prompt = prompt_template.format(num=num_descriptions, class_name=class_name)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("assistant")[-1].strip() if "assistant" in response else response

        # Parse line-by-line descriptions
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        descriptions = []
        for line in lines:
            # Skip lines that look like numbers or bullets
            if line[0].isdigit() or line.startswith("-") or line.startswith("*"):
                line = line.lstrip("0123456789.-* ")
            if line and len(line) > 5:
                descriptions.append(line)
            if len(descriptions) >= num_descriptions:
                break

        return descriptions[:num_descriptions]

    def generate_all_descriptions(
        self,
        class_names: List[str],
        num_per_class: int = 5,
        prompt_template: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate descriptions for all classes.

        Args:
            class_names: List of all class names
            num_per_class: Number of descriptions per class
            prompt_template: Custom prompt template

        Returns:
            Dictionary mapping class_name -> list of descriptions
        """
        results = {}
        for cls_name in class_names:
            print(f"Generating descriptions for: {cls_name}")
            results[cls_name] = self.generate_descriptions(
                cls_name, num_per_class, prompt_template
            )
        return results
