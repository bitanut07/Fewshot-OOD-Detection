# -*- coding: utf-8 -*-
"""Load class descriptions produced by generate_llm_descriptions (YAML or JSON)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def normalize_descriptions_payload(raw: Any) -> Dict[str, List[str]]:
    """
    Accept either legacy flat mapping {class_name: [str, ...]} or structured JSON:

        {"schema_version": 1, "descriptions": {class_name: [str, ...]}, ...}
    """
    if not raw or not isinstance(raw, dict):
        return {}
    if raw.get("schema_version") == 1 and "descriptions" in raw:
        inner = raw["descriptions"]
        return inner if isinstance(inner, dict) else {}
    return raw  # type: ignore[return-value]


def load_class_descriptions(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load descriptions from .json (UTF-8) or .yaml / .yml."""
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
