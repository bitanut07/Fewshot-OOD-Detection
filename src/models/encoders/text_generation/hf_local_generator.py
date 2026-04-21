# -*- coding: utf-8 -*-
"""HuggingFace local text generator.

Loads a causal-LM from the HF Hub (or local cache) and runs generation
on the local GPU / CPU.  Implements the ``BaseTextGenerator`` interface.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_generator import BaseTextGenerator, GenerationConfig

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(spec: str | torch.dtype) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    return _DTYPE_MAP.get(spec.lower().strip(), torch.float16)


class HFLocalGenerator(BaseTextGenerator):
    """Local HuggingFace causal-LM text generator.

    Parameters
    ----------
    model_name_or_path:
        HF Hub model id or local path.
    device_map:
        Accelerate device-map strategy (``"auto"`` recommended).
    torch_dtype:
        Weight dtype — accepts string (``"float16"``) or ``torch.dtype``.
    cache_dir:
        Override for HF cache directory.  Falls back to ``$HF_CACHE_DIR``.
    trust_remote_code:
        Forwarded to ``AutoModelForCausalLM.from_pretrained``.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "float16",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name_or_path
        self._device_map = device_map
        self._torch_dtype = _resolve_dtype(torch_dtype)
        self._cache_dir = cache_dir or os.getenv("HF_CACHE_DIR")
        self._trust_remote_code = trust_remote_code

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # -- BaseTextGenerator interface ------------------------------------------

    def model_name(self) -> str:
        return self._model_name

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        cfg = config or GenerationConfig()
        self._ensure_loaded()

        do_sample = cfg.temperature > 0.0 and not cfg.deterministic
        temperature = cfg.temperature if do_sample else 1.0
        top_p = cfg.top_p if do_sample else 1.0

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=cfg.repetition_penalty,
            )
        gen_tokens = int(outputs.shape[-1]) - int(input_len)
        log.info("Generated %d tokens in %.1fs", gen_tokens, time.time() - t0)

        full = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        marker = full.rfind("assistant")
        if marker != -1:
            return full[marker + len("assistant"):].strip()
        return full[len(text):].strip()

    # -- Internal -------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading model: %s (dtype=%s)", self._model_name, self._torch_dtype)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )
        self._model.eval()
        log.info("Model loaded in %.1fs", time.time() - t0)
