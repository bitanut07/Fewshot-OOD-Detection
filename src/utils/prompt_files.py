# -*- coding: utf-8 -*-
"""Load class descriptions produced by generate_llm_descriptions.

Supports both the new structured schema (v2) and the legacy flat format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

SCHEMA_VERSION = 2


def normalize_descriptions_payload(raw: Any) -> Dict[str, List[str]]:
    """Return flat ``{class_name: [str, …]}`` from any supported format.

    Supported formats:
        - Schema v2 structured payload (``schema_version: 2``)
        - Schema v1 with ``descriptions`` key
        - Legacy flat ``{class_name: [str, …]}``
    """
    if not raw or not isinstance(raw, dict):
        return {}

    # Schema v2 — structured with classes.*.generated_descriptions
    if raw.get("schema_version") == SCHEMA_VERSION and "classes" in raw:
        flat: Dict[str, List[str]] = {}
        for cls, entry in raw["classes"].items():
            if isinstance(entry, dict):
                flat[cls] = entry.get("generated_descriptions", [])
            elif isinstance(entry, list):
                flat[cls] = entry
        return flat

    # Schema v1 — simple wrapper
    if raw.get("schema_version") == 1 and "descriptions" in raw:
        inner = raw["descriptions"]
        return inner if isinstance(inner, dict) else {}

    # Legacy flat mapping
    return {k: v for k, v in raw.items() if isinstance(v, list)}


def load_class_descriptions(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load descriptions from ``.json`` or ``.yaml``/``.yml``."""
    path = Path(path)
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    return normalize_descriptions_payload(data)


def load_class_descriptions_with_defaults(
    path: Union[str, Path],
) -> Dict[str, Dict[str, List[str]]]:
    """Load structured payload and return per-class default + generated lists.

    Returns ``{class_name: {"default_prompts": [...], "generated": [...]}}``.
    Falls back to empty defaults for legacy formats.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)

    if not isinstance(data, dict):
        return {}

    if data.get("schema_version") == SCHEMA_VERSION and "classes" in data:
        result: Dict[str, Dict[str, List[str]]] = {}
        for cls, entry in data["classes"].items():
            if isinstance(entry, dict):
                result[cls] = {
                    "default_prompts": entry.get("default_prompts", []),
                    "generated": entry.get("generated_descriptions", []),
                }
        return result

    # Legacy: no default prompts stored
    flat = normalize_descriptions_payload(data)
    return {cls: {"default_prompts": [], "generated": descs} for cls, descs in flat.items()}
