# -*- coding: utf-8 -*-
"""Cache manager for LLM-generated descriptions and questions.

Handles persistence in the structured YAML/JSON schema required by the
pipeline, including metadata for reproducibility.  Implements cache-first
behaviour: load existing valid outputs unless ``force_regenerate`` is set.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


# ─── Public data helpers ────────────────────────────────────────────────

def build_output_payload(
    *,
    dataset_name: str,
    model_name: str,
    seed: Optional[int],
    generation_config: dict,
    class_names: List[str],
    questions: List[str],
    classes: Dict[str, Dict[str, Any]],
) -> dict:
    """Build the canonical output dict ready for serialisation."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_config": generation_config,
        "class_names": class_names,
        "questions": questions,
        "classes": classes,
    }


def build_class_entry(
    default_prompts: List[str],
    generated_descriptions: List[str],
) -> dict:
    return {
        "default_prompts": default_prompts,
        "generated_descriptions": generated_descriptions,
        "metadata": {
            "num_default": len(default_prompts),
            "num_generated": len(generated_descriptions),
        },
    }


# ─── Cache manager ──────────────────────────────────────────────────────

class CacheManager:
    """Read / write / validate the structured description cache.

    Parameters
    ----------
    descriptions_path:
        Primary output — structured YAML with full schema.
    questions_path:
        Standalone questions file (kept for backward compat).
    json_path:
        Flat ``{class_name: [descriptions]}`` JSON consumed by glali code.
    """

    def __init__(
        self,
        descriptions_path: Union[str, Path] = "data/prompts/class_descriptions.yaml",
        questions_path: Union[str, Path] = "data/prompts/class_questions.yaml",
        json_path: Optional[Union[str, Path]] = "data/prompts/class_descriptions.json",
    ) -> None:
        self.descriptions_path = Path(descriptions_path)
        self.questions_path = Path(questions_path)
        self.json_path = Path(json_path) if json_path else None

    # -- Existence checks -----------------------------------------------------

    def questions_exist(self) -> bool:
        return self.questions_path.is_file()

    def descriptions_exist(self) -> bool:
        return self.descriptions_path.is_file()

    def cache_valid(self, expected_classes: List[str], num_descriptions: int) -> bool:
        """Return True if cache has all *expected_classes* with enough descriptions."""
        if not self.descriptions_exist():
            return False
        try:
            data = self._load_yaml(self.descriptions_path)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False

        classes = data.get("classes", data)
        for cls in expected_classes:
            entry = classes.get(cls)
            if entry is None:
                return False
            descs = entry.get("generated_descriptions", entry) if isinstance(entry, dict) else entry
            if not isinstance(descs, list) or len(descs) < num_descriptions:
                return False
        return True

    # -- Load -----------------------------------------------------------------

    def load_questions(self) -> List[str]:
        if not self.questions_exist():
            return []
        data = self._load_yaml(self.questions_path)
        if isinstance(data, dict):
            return data.get("questions", [])
        return []

    def load_descriptions(self) -> dict:
        """Load full structured payload (schema v2) or legacy flat dict."""
        if not self.descriptions_exist():
            return {}
        return self._load_yaml(self.descriptions_path)

    def load_flat_descriptions(self) -> Dict[str, List[str]]:
        """Return flat ``{class_name: [str, …]}`` regardless of schema version."""
        data = self.load_descriptions()
        if not data:
            return {}
        if data.get("schema_version") == SCHEMA_VERSION and "classes" in data:
            flat: Dict[str, List[str]] = {}
            for cls, entry in data["classes"].items():
                descs = entry.get("generated_descriptions", [])
                flat[cls] = descs
            return flat
        # Legacy: the dict itself is flat
        return {k: v for k, v in data.items() if isinstance(v, list)}

    def load_all(self) -> dict:
        """Return dict with ``questions`` and ``descriptions`` keys."""
        return {
            "questions": self.load_questions(),
            "descriptions": self.load_flat_descriptions(),
        }

    # -- Save -----------------------------------------------------------------

    def save_questions(self, questions: List[str]) -> None:
        self._ensure_dir(self.questions_path)
        self._save_yaml(self.questions_path, {"questions": questions})
        log.info("Questions saved → %s (%d items)", self.questions_path, len(questions))

    def save_descriptions(self, payload: dict) -> None:
        """Save the full structured payload (schema v2)."""
        self._ensure_dir(self.descriptions_path)
        self._save_yaml(self.descriptions_path, payload)
        log.info("Descriptions saved → %s", self.descriptions_path)

    def save_flat_json(self, flat: Dict[str, List[str]]) -> None:
        """Save flat ``{class: [desc, …]}`` JSON for glali compatibility."""
        if self.json_path is None:
            return
        self._ensure_dir(self.json_path)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False)
        log.info("Flat JSON saved → %s", self.json_path)

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _save_yaml(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
