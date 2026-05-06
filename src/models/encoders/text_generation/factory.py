# -*- coding: utf-8 -*-
"""Factory that builds the right text-generation backend from config.

Config shape (examples)::

    # 1) Local LLM (4-bit on GPU, cache in /tmp, auto-delete after use)
    llm:
      use_local_llm: true
      cleanup_cache: true
      model_name: "Qwen/Qwen2.5-7B-Instruct"
      device_map: "auto"
      torch_dtype: "float16"
      quantization: "4bit"
      low_cpu_mem_usage: true
      cache_dir: "/tmp/hf-cache-fewshot-ood"
      trust_remote_code: true

    # 2) Remote API (zero disk, zero VRAM)
    llm:
      use_local_llm: false
      api:
        provider: "openai"
        model_name: "gpt-4o-mini"
        api_key_env: "OPENAI_API_KEY"
        base_url: null
"""
from __future__ import annotations

import logging
from typing import Optional

from .api_generator import APITextGenerator
from .base_generator import BaseTextGenerator
from .hf_env import setup_hf_cache
from .hf_local_generator import HFLocalGenerator

log = logging.getLogger(__name__)


def build_generator(llm_cfg: dict) -> BaseTextGenerator:
    """Return a concrete :class:`BaseTextGenerator` based on *llm_cfg*.

    Falls back to a local HF generator when ``use_local_llm`` is unset,
    preserving backward compatibility with older configs.
    """
    use_local = llm_cfg.get("use_local_llm", True)

    if not use_local:
        api_cfg = llm_cfg.get("api") or {}
        provider = api_cfg.get("provider", "openai")
        gen = APITextGenerator(
            provider=provider,
            model_name=api_cfg.get("model_name", "gpt-4o-mini"),
            api_key_env=api_cfg.get("api_key_env"),
            base_url=api_cfg.get("base_url"),
            timeout=float(api_cfg.get("timeout", 60.0)),
            extra_headers=api_cfg.get("extra_headers"),
        )
        log.info("Built API generator: %s", gen.model_name())
        return gen

    # Local path — make sure HF cache is redirected before any HF call.
    cache_dir = setup_hf_cache(llm_cfg.get("cache_dir"), verbose=True)

    gen = HFLocalGenerator(
        model_name_or_path=llm_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device_map=llm_cfg.get("device_map", "auto"),
        torch_dtype=llm_cfg.get("torch_dtype", "float16"),
        cache_dir=cache_dir,
        trust_remote_code=llm_cfg.get("trust_remote_code", True),
        quantization=llm_cfg.get("quantization"),
        low_cpu_mem_usage=llm_cfg.get("low_cpu_mem_usage", True),
        cleanup_cache=llm_cfg.get("cleanup_cache", False),
    )
    log.info(
        "Built local HF generator (%s, quant=%s, cleanup_cache=%s)",
        gen.model_name(),
        llm_cfg.get("quantization") or "none",
        llm_cfg.get("cleanup_cache", False),
    )
    return gen


def release_generator(
    gen: Optional[BaseTextGenerator],
    delete_cache: Optional[bool] = None,
) -> None:
    """Free resources owned by *gen* (GPU mem + optional disk cache)."""
    if gen is None:
        return
    unload = getattr(gen, "unload", None)
    if callable(unload):
        try:
            unload(delete_cache=delete_cache)
        except TypeError:
            unload()
