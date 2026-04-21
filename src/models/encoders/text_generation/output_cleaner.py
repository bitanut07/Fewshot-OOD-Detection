# -*- coding: utf-8 -*-
"""Post-processing and validation of LLM-generated text.

Responsible for:
  - stripping numbering / bullet prefixes
  - filtering generic, too-short, too-long, or irrelevant lines
  - near-duplicate removal
  - optional keyword relevance checks for medical / radiographic text
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Set

# Lines matching any of these patterns are considered too generic for CLIP.
_GENERIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^this (x-?ray|image|photo|scan|radiograph) (shows?|displays?|reveals?|indicates?)",
        r"^the (x-?ray|image|photo|scan|bone|patient|radiograph) (may|might|could|shows?|displays?|reveals?)",
        r"^visible (in this|on the) (x-?ray|image|photo|scan)",
        r"^a (radiograph|x-?ray|image|photo|scan) of",
        r"^(may|possibly|might|can|could) (be|show|appear|indicate|suggest)",
        r"^in this (x-?ray|image|scan|radiograph)",
    ]
]

# Optional: if a line contains none of these domain keywords it is likely off-topic.
_RADIOGRAPHIC_KEYWORDS: Set[str] = {
    "bone", "cortex", "cortical", "trabecul", "lytic", "sclerotic", "sclerosis",
    "lesion", "periosteal", "periosteum", "fracture", "osteophyte", "osteolytic",
    "radiolucen", "radiopaque", "mineralization", "calcif", "matrix", "margin",
    "expansion", "expansile", "destruction", "deformity", "density", "lucen",
    "cyst", "tumor", "mass", "outgrowth", "exostosis", "metaphys", "diaphys",
    "epiphys", "endplate", "joint", "articular", "erosion", "septation",
    "moth-eaten", "permeative", "sunburst", "codman", "ground-glass",
}

_BULLET_RE = re.compile(r"^\s*(?:[\d]+[.\)]\s*|[-*•]\s+)")
_UNCERTAINTY_RE = re.compile(
    r"\b(may|might|possibly|could|sometimes|occasionally)\b", re.IGNORECASE
)


class OutputCleaner:
    """Clean and validate raw LLM output into CLIP-ready description lines.

    Parameters
    ----------
    min_length:
        Minimum character length for an accepted line.
    max_length:
        Maximum character length for an accepted line.
    near_dup_threshold:
        SequenceMatcher ratio above which two lines are near-duplicates.
    require_domain_keyword:
        If True, reject lines that contain no radiographic keyword.
    strip_uncertainty:
        If True, reject lines with uncertainty hedging words.
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 300,
        near_dup_threshold: float = 0.85,
        require_domain_keyword: bool = True,
        strip_uncertainty: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.near_dup_threshold = near_dup_threshold
        self.require_domain_keyword = require_domain_keyword
        self.strip_uncertainty = strip_uncertainty

    def clean(
        self,
        raw_text: str,
        max_lines: Optional[int] = None,
        existing: Optional[List[str]] = None,
    ) -> List[str]:
        """Parse *raw_text* into cleaned description lines.

        Parameters
        ----------
        raw_text:
            Raw model output string.
        max_lines:
            Stop collecting once this many valid lines are found.
        existing:
            Already-accepted lines — new lines are de-duped against these.

        Returns
        -------
        List of cleaned, validated, de-duplicated lines.
        """
        accepted: List[str] = list(existing or [])
        new_lines: List[str] = []

        for raw_line in raw_text.split("\n"):
            line = _BULLET_RE.sub("", raw_line).strip()
            if not line:
                continue
            # Remove wrapping quotes
            if (line.startswith('"') and line.endswith('"')) or \
               (line.startswith("'") and line.endswith("'")):
                line = line[1:-1].strip()

            if not self._passes_filters(line):
                continue
            if self._is_near_duplicate(line, accepted):
                continue

            accepted.append(line)
            new_lines.append(line)
            if max_lines is not None and len(new_lines) >= max_lines:
                break

        return new_lines

    # -- Filters --------------------------------------------------------------

    def _passes_filters(self, line: str) -> bool:
        if len(line) < self.min_length or len(line) > self.max_length:
            return False
        if self._is_generic(line):
            return False
        if self.strip_uncertainty and _UNCERTAINTY_RE.search(line):
            return False
        if self.require_domain_keyword and not self._has_domain_keyword(line):
            return False
        return True

    @staticmethod
    def _is_generic(line: str) -> bool:
        for pat in _GENERIC_PATTERNS:
            if pat.match(line):
                return True
        return False

    @staticmethod
    def _has_domain_keyword(line: str) -> bool:
        lower = line.lower()
        return any(kw in lower for kw in _RADIOGRAPHIC_KEYWORDS)

    def _is_near_duplicate(self, candidate: str, existing: List[str]) -> bool:
        c_lower = candidate.lower()
        for prev in existing:
            ratio = SequenceMatcher(None, c_lower, prev.lower()).ratio()
            if ratio >= self.near_dup_threshold:
                return True
        return False
