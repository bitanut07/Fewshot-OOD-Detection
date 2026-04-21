# -*- coding: utf-8 -*-
"""Abstract base class for text generators.

Defines the interface that any text generation backend (local HF model,
remote API, etc.) must implement.  The LLMWrapper facade delegates all
raw text generation to a concrete subclass of this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    """Parameters that control text generation behaviour."""

    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    seed: Optional[int] = 42
    deterministic: bool = False

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }


class BaseTextGenerator(ABC):
    """Interface every text-generation backend must satisfy."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Return raw text completion for *prompt*."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """True once model weights / connection are ready."""

    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model."""
