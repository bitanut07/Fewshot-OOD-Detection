# -*- coding: utf-8 -*-
"""Enhanced post-processing and validation of LLM-generated text.

Responsible for:
  - Stripping numbering / bullet prefixes
  - Filtering generic, forbidden, and low-quality lines
  - Near-duplicate removal (within class and cross-class)
  - Keyword relevance checks for radiographic content
  - Length-based filtering
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Forbidden patterns (instant reject)
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_STARTS = [
    r"^this (x-?ray|image|photo|scan|radiograph) (shows?|displays?|reveals?|indicates?|demonstrates?)",
    r"^the (x-?ray|image|photo|scan|bone|patient|radiograph|lesion) (may|might|could|shows?|displays?|reveals?)",
    r"^visible (in this|on the) (x-?ray|image|photo|scan)",
    r"^a (radiograph|x-?ray|image|photo|scan) of",
    r"^(may|possibly|might|can|could) (be|show|appear|indicate|suggest)",
    r"^in this (x-?ray|image|scan|radiograph)",
    r"^(indicative|suggestive|suspicious|consistent) (of|with|for)",
    r"^there (is|are) (a |an |visible |apparent )",
    r"^(showing|revealing|demonstrating|displaying) (a |an |the )",
]

_FORBIDDEN_PHRASES = [
    "this x-ray shows", "the image reveals", "a radiograph of",
    "indicative of", "suggestive of", "suspicious for", "consistent with",
    "may be seen", "might be present", "could represent",
    "bone structure remains", "lesion is visible", "abnormal bone appearance",
    "bone abnormality", "visible abnormality", "pathological changes",
    "patient presents", "clinical presentation", "symptoms include",
    "pain", "tenderness", "swelling", "fever",
]

_GENERIC_PATTERNS = re.compile(
    "|".join(_FORBIDDEN_STARTS),
    re.IGNORECASE,
)

_FORBIDDEN_PHRASE_RE = re.compile(
    "|".join(re.escape(p) for p in _FORBIDDEN_PHRASES),
    re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"\b(may|might|possibly|could|sometimes|occasionally|perhaps|likely|unlikely|probably)\b",
    re.IGNORECASE,
)

_WEAK_GENERIC_RE = re.compile(
    r"\b(abnormal|abnormality|patholog|disorder|disease|condition|finding|appearance)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Required domain vocabulary
# ─────────────────────────────────────────────────────────────────────────────

RADIOGRAPHIC_KEYWORDS: Set[str] = {
    # Lesion patterns
    "lytic", "sclerotic", "sclerosis", "osteolytic", "osteoblastic",
    "geographic", "permeative", "moth-eaten", "mixed",
    # Cortical
    "cortex", "cortical", "thinning", "thinned", "expansion", "expansile",
    "breach", "breached", "destruction", "destroyed", "intact",
    # Periosteal
    "periosteal", "periosteum", "sunburst", "codman", "lamellated",
    "onion-skin", "spiculated",
    # Matrix
    "matrix", "mineralization", "calcification", "ground-glass",
    "chondroid", "osteoid", "ring-arc", "stippled", "cloud-like",
    # Margins
    "margin", "well-defined", "ill-defined", "sclerotic rim",
    "narrow zone", "transition",
    # Location
    "epiphysis", "epiphyseal", "metaphysis", "metaphyseal",
    "diaphysis", "diaphyseal", "medullary", "intramedullary",
    "subarticular", "articular",
    # Bone anatomy
    "trabecular", "trabeculae", "cancellous", "endosteal",
    # Shape
    "lobulated", "septation", "septated", "multiloculated",
    "unilocular", "eccentric", "central", "exophytic",
    # Density
    "radiolucent", "radiopaque", "lucent", "dense", "opacity",
    # Soft tissue
    "soft tissue", "extra-osseous",
    # Tumor descriptors
    "lesion", "tumor", "mass", "cyst", "nodule", "outgrowth",
    "exostosis", "projection", "pedunculated", "sessile",
}

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing patterns
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*(?:[\d]+[.\)]\s*|[-*•]\s+|[a-z]\)\s*)")
_QUOTE_STRIP_RE = re.compile(r'^["\'](.+)["\']$')
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


# ─────────────────────────────────────────────────────────────────────────────
# OutputCleaner
# ─────────────────────────────────────────────────────────────────────────────

class OutputCleaner:
    """Clean and validate raw LLM output into CLIP-ready description lines.

    Implements strict filtering to ensure only high-quality, visually grounded,
    discriminative descriptions pass through.
    """

    def __init__(
        self,
        min_length: int = 25,
        max_length: int = 200,
        near_dup_threshold: float = 0.80,
        require_domain_keyword: bool = True,
        min_domain_keywords: int = 2,
        strip_uncertainty: bool = True,
        strip_generic: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.near_dup_threshold = near_dup_threshold
        self.require_domain_keyword = require_domain_keyword
        self.min_domain_keywords = min_domain_keywords
        self.strip_uncertainty = strip_uncertainty
        self.strip_generic = strip_generic

    def clean(
        self,
        raw_text: str,
        max_lines: Optional[int] = None,
        existing: Optional[List[str]] = None,
    ) -> List[str]:
        """Parse and filter raw LLM output into clean description lines.

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
            line = self._preprocess(raw_line)
            if not line:
                continue

            if not self._passes_all_filters(line):
                continue

            if self._is_near_duplicate(line, accepted):
                continue

            accepted.append(line)
            new_lines.append(line)

            if max_lines is not None and len(new_lines) >= max_lines:
                break

        return new_lines

    def clean_attributes(
        self,
        raw_text: str,
        max_lines: Optional[int] = None,
    ) -> List[str]:
        """Parse raw LLM output into clean attribute phrases.

        Less strict than description cleaning — allows shorter phrases.
        """
        attributes: List[str] = []

        for raw_line in raw_text.split("\n"):
            line = self._preprocess(raw_line)
            if not line:
                continue

            # Attributes can be shorter (3-50 chars typically)
            if len(line) < 10 or len(line) > 80:
                continue

            # Skip forbidden starts
            if _GENERIC_PATTERNS.match(line):
                continue

            # Skip uncertainty
            if _UNCERTAINTY_RE.search(line):
                continue

            # Dedupe
            if self._is_near_duplicate(line, attributes):
                continue

            attributes.append(line)

            if max_lines and len(attributes) >= max_lines:
                break

        return attributes

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, raw_line: str) -> str:
        """Clean up a raw line: strip bullets, quotes, extra whitespace."""
        line = _BULLET_RE.sub("", raw_line).strip()
        if not line:
            return ""

        # Strip wrapping quotes
        match = _QUOTE_STRIP_RE.match(line)
        if match:
            line = match.group(1).strip()

        # Collapse multiple spaces
        line = _MULTI_SPACE_RE.sub(" ", line)

        return line

    # ─────────────────────────────────────────────────────────────────────────
    # Filters
    # ─────────────────────────────────────────────────────────────────────────

    def _passes_all_filters(self, line: str) -> bool:
        """Return True only if line passes all quality filters."""
        # Length check
        if len(line) < self.min_length or len(line) > self.max_length:
            return False

        # Forbidden pattern starts
        if _GENERIC_PATTERNS.match(line):
            return False

        # Forbidden phrases anywhere
        if _FORBIDDEN_PHRASE_RE.search(line):
            return False

        # Uncertainty words
        if self.strip_uncertainty and _UNCERTAINTY_RE.search(line):
            return False

        # Too many generic medical words without specifics
        if self.strip_generic:
            generic_count = len(_WEAK_GENERIC_RE.findall(line))
            domain_count = self._count_domain_keywords(line)
            if generic_count > domain_count:
                return False

        # Must have sufficient radiographic terminology
        if self.require_domain_keyword:
            if self._count_domain_keywords(line) < self.min_domain_keywords:
                return False

        return True

    def _count_domain_keywords(self, line: str) -> int:
        """Count how many radiographic keywords appear in the line."""
        lower = line.lower()
        return sum(1 for kw in RADIOGRAPHIC_KEYWORDS if kw in lower)

    def _is_near_duplicate(self, candidate: str, existing: List[str]) -> bool:
        """Check if candidate is too similar to any existing line."""
        c_lower = candidate.lower()
        for prev in existing:
            ratio = SequenceMatcher(None, c_lower, prev.lower()).ratio()
            if ratio >= self.near_dup_threshold:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-class deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def remove_cross_class_duplicates(
        self,
        class_descriptions: dict[str, List[str]],
        threshold: float = 0.75,
    ) -> dict[str, List[str]]:
        """Remove descriptions that are too similar across different classes.

        Descriptions that appear nearly identical in multiple classes are not
        discriminative and should be removed.
        """
        result: dict[str, List[str]] = {}

        for target_class, descs in class_descriptions.items():
            other_descs: List[str] = []
            for cls, d_list in class_descriptions.items():
                if cls != target_class:
                    other_descs.extend(d_list)

            kept: List[str] = []
            for desc in descs:
                is_cross_dup = False
                d_lower = desc.lower()
                for other in other_descs:
                    if SequenceMatcher(None, d_lower, other.lower()).ratio() >= threshold:
                        is_cross_dup = True
                        break
                if not is_cross_dup:
                    kept.append(desc)

            result[target_class] = kept

        return result
