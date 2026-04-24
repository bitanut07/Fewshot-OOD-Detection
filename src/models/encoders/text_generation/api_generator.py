# -*- coding: utf-8 -*-
"""API-based text generator.

Zero-disk alternative to :class:`HFLocalGenerator`: instead of downloading
a ~15GB LLM onto the cloud instance, we call a hosted API (OpenAI,
Anthropic, or any OpenAI-compatible endpoint). The training pipeline is
completely decoupled from LLM weights.

Supported providers:

* ``provider="openai"`` — uses the ``openai`` SDK if installed, else falls
  back to a plain ``requests`` POST to ``/v1/chat/completions``.
* ``provider="anthropic"`` — uses the ``anthropic`` SDK if installed, else
  falls back to a ``requests`` POST to ``/v1/messages``.
* ``provider="openai-compatible"`` — any server speaking the OpenAI chat
  spec (e.g. vLLM, Ollama, Together, Groq). Set ``base_url`` accordingly.

API keys are read from environment variables (never committed), resolved
at generate-time so the object can be constructed before env setup.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from .base_generator import BaseTextGenerator, GenerationConfig

log = logging.getLogger(__name__)


class APITextGenerator(BaseTextGenerator):
    """Hosted-LLM generator. Does not touch local disk."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        extra_headers: Optional[dict] = None,
    ) -> None:
        self._provider = provider.lower().strip()
        self._model_name = model_name
        self._api_key_env = api_key_env or self._default_api_key_env()
        self._base_url = base_url or self._default_base_url()
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._ready = False

    # ── BaseTextGenerator interface ──────────────────────────────────────

    def model_name(self) -> str:
        return f"{self._provider}:{self._model_name}"

    def is_loaded(self) -> bool:
        return self._ready

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        cfg = config or GenerationConfig()
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key env var '{self._api_key_env}' is not set. "
                f"Export it or switch to use_local_llm=true."
            )

        t0 = time.time()
        if self._provider == "anthropic":
            text = self._call_anthropic(prompt, cfg, api_key)
        else:
            text = self._call_openai_compatible(prompt, cfg, api_key)
        self._ready = True
        log.info("API generate [%s] in %.1fs", self.model_name(), time.time() - t0)
        return text.strip()

    # ── OpenAI / OpenAI-compatible ───────────────────────────────────────

    def _call_openai_compatible(
        self,
        prompt: str,
        cfg: GenerationConfig,
        api_key: str,
    ) -> str:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key, base_url=self._base_url)
            resp = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                timeout=self._timeout,
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            return self._raw_http_openai(prompt, cfg, api_key)

    def _raw_http_openai(
        self,
        prompt: str,
        cfg: GenerationConfig,
        api_key: str,
    ) -> str:
        import requests

        url = (self._base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    # ── Anthropic ────────────────────────────────────────────────────────

    def _call_anthropic(
        self,
        prompt: str,
        cfg: GenerationConfig,
        api_key: str,
    ) -> str:
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=api_key, base_url=self._base_url)
            msg = client.messages.create(
                model=self._model_name,
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._timeout,
            )
            parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            return "".join(parts)
        except ImportError:
            return self._raw_http_anthropic(prompt, cfg, api_key)

    def _raw_http_anthropic(
        self,
        prompt: str,
        cfg: GenerationConfig,
        api_key: str,
    ) -> str:
        import requests

        url = (self._base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{url}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload = {
            "model": self._model_name,
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", []))

    # ── Defaults ─────────────────────────────────────────────────────────

    def _default_api_key_env(self) -> str:
        if self._provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        return "OPENAI_API_KEY"

    def _default_base_url(self) -> Optional[str]:
        if self._provider == "anthropic":
            return None  # SDK default
        if self._provider == "openai":
            return None  # SDK default
        # openai-compatible → caller must specify
        return None
