# -*- coding: utf-8 -*-
"""Backward-compatible re-export of the refactored LLMWrapper.

All logic has moved to ``src.models.encoders.text_generation``.
This module exists so that existing imports continue to work:

    from src.models.encoders.llm_wrapper import LLMWrapper
"""
from src.models.encoders.text_generation.llm_wrapper import LLMWrapper  # noqa: F401
from src.models.encoders.text_generation.base_generator import (  # noqa: F401
    BaseTextGenerator,
    GenerationConfig,
)
from src.models.encoders.text_generation.hf_local_generator import HFLocalGenerator  # noqa: F401
from src.models.encoders.text_generation.api_generator import APITextGenerator  # noqa: F401
from src.models.encoders.text_generation.factory import (  # noqa: F401
    build_generator,
    release_generator,
)
from src.models.encoders.text_generation.prompt_builder import PromptBuilder  # noqa: F401
from src.models.encoders.text_generation.output_cleaner import OutputCleaner  # noqa: F401
from src.models.encoders.text_generation.cache_manager import CacheManager  # noqa: F401
from src.models.encoders.text_generation.hf_env import (  # noqa: F401
    setup_hf_cache,
    cleanup_hf_cache,
    cleanup_model_cache,
    get_hf_cache_dir,
)

__all__ = [
    "LLMWrapper",
    "BaseTextGenerator",
    "GenerationConfig",
    "HFLocalGenerator",
    "APITextGenerator",
    "PromptBuilder",
    "OutputCleaner",
    "CacheManager",
    "build_generator",
    "release_generator",
    "setup_hf_cache",
    "cleanup_hf_cache",
    "cleanup_model_cache",
    "get_hf_cache_dir",
]
