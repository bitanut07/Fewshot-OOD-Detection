# -*- coding: utf-8 -*-
"""HuggingFace local text generator (disk-efficient, cloud-friendly).

Optimisations vs. the previous version:

* HF cache is redirected to ``/tmp`` via :mod:`hf_env` *before* ``transformers``
  is imported, so large weight downloads never land on the same disk as the
  training checkpoints.
* Supports **4-bit / 8-bit quantization** via ``bitsandbytes`` when
  ``quantization="4bit"`` or ``"8bit"`` — cuts a 15GB model down to ~4GB
  (4-bit) or ~8GB (8-bit) VRAM *and* skips the full FP16 download is still
  required but the on-disk footprint stays the same; the win is VRAM.
* ``low_cpu_mem_usage=True`` + ``device_map="auto"`` → Accelerate streams
  weights straight to GPU without a full CPU copy.
* :meth:`unload` frees GPU/CPU memory and optionally deletes the on-disk
  cache, so the training stage starts with a clean slate.
* Context-manager support (``with HFLocalGenerator(...) as gen:``) auto-unloads.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import Optional

# Redirect HF cache BEFORE transformers is imported. This is the whole
# point of hf_env — side-effect ordering matters here.
from .hf_env import setup_hf_cache, cleanup_hf_cache

setup_hf_cache(verbose=False)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .base_generator import BaseTextGenerator, GenerationConfig  # noqa: E402

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


def _build_quant_config(quantization: Optional[str], compute_dtype: torch.dtype):
    """Return a ``BitsAndBytesConfig`` or ``None``. Lazy-imports bitsandbytes."""
    if not quantization:
        return None
    quant = quantization.lower().strip()
    if quant not in {"4bit", "8bit"}:
        log.warning("Unknown quantization '%s' — skipping (use '4bit' or '8bit')", quantization)
        return None

    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:  # pragma: no cover
        log.warning("BitsAndBytesConfig unavailable (%s) — falling back to full precision", e)
        return None

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)


class HFLocalGenerator(BaseTextGenerator):
    """Local HuggingFace causal-LM generator with disk/VRAM safeguards.

    Parameters
    ----------
    model_name_or_path:
        HF Hub model id or local path.
    device_map:
        Accelerate device-map strategy (``"auto"`` recommended).
    torch_dtype:
        Weight / compute dtype. String (``"float16"``) or ``torch.dtype``.
    cache_dir:
        Override HF cache directory. Falls back to ``$HF_CACHE_DIR`` then
        ``/tmp/hf-cache-fewshot-ood`` (see :func:`setup_hf_cache`).
    trust_remote_code:
        Forwarded to ``from_pretrained``.
    quantization:
        ``"4bit"`` | ``"8bit"`` | ``None``. Requires ``bitsandbytes`` + CUDA.
    low_cpu_mem_usage:
        Forwarded to ``from_pretrained``; avoids a full CPU copy during load.
    cleanup_cache:
        If True, deleting this instance (``unload()`` or ``__exit__``) also
        wipes the on-disk HF cache. Use for one-shot description generation.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "float16",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        quantization: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        cleanup_cache: bool = False,
    ) -> None:
        self._model_name = model_name_or_path
        self._device_map = device_map
        self._torch_dtype = _resolve_dtype(torch_dtype)
        self._trust_remote_code = trust_remote_code
        self._quantization = (quantization or None)
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._cleanup_cache = cleanup_cache

        # Resolve cache dir — this also ensures env vars are pointed at /tmp.
        self._cache_dir = setup_hf_cache(cache_dir or os.getenv("HF_CACHE_DIR"))

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # ── BaseTextGenerator interface ──────────────────────────────────────

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

    # ── Memory / disk management ─────────────────────────────────────────

    def unload(self, delete_cache: Optional[bool] = None) -> None:
        """Free GPU/CPU memory. Optionally wipe the on-disk HF cache.

        ``delete_cache`` overrides the constructor-time ``cleanup_cache`` flag.
        """
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        should_delete = self._cleanup_cache if delete_cache is None else delete_cache
        if should_delete:
            cleanup_hf_cache(self._cache_dir, verbose=True)

    def __enter__(self) -> "HFLocalGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def __del__(self) -> None:
        try:
            self.unload(delete_cache=False)
        except Exception:
            pass

    # ── Internal ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        log.info(
            "Loading LLM: %s (dtype=%s, quant=%s, cache=%s)",
            self._model_name,
            self._torch_dtype,
            self._quantization or "none",
            self._cache_dir,
        )
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quant_config = _build_quant_config(self._quantization, self._torch_dtype)

        load_kwargs = dict(
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
            low_cpu_mem_usage=self._low_cpu_mem_usage,
            device_map=self._device_map,
        )
        if quant_config is not None:
            # dtype must NOT be passed alongside a bnb config
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = self._torch_dtype

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **load_kwargs,
        )
        self._model.eval()
        log.info("Model loaded in %.1fs", time.time() - t0)
