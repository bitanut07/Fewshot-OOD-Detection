# -*- coding: utf-8 -*-
"""Quality scoring and ranking for generated descriptions.

Implements rule-based scoring to identify high-quality, discriminative
descriptions suitable for CLIP text embeddings.  Used to filter and rank
candidates after LLM generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary sets for scoring
# ─────────────────────────────────────────────────────────────────────────────

# High-value radiographic terms that indicate visual grounding
RADIOGRAPHIC_TERMS: Set[str] = {
    # Lesion patterns
    "lytic", "sclerotic", "sclerosis", "osteolytic", "osteoblastic",
    "geographic", "permeative", "moth-eaten", "mixed",
    # Cortical features
    "cortex", "cortical", "thinning", "thinned", "breach", "breached",
    "destruction", "destroyed", "intact", "expansion", "expansile",
    # Periosteal
    "periosteal", "periosteum", "sunburst", "codman", "lamellated",
    "onion-skin", "spiculated", "aggressive",
    # Matrix
    "matrix", "mineralization", "calcification", "ground-glass",
    "chondroid", "osteoid", "ring-arc", "stippled", "cloud-like",
    # Margins
    "margin", "well-defined", "ill-defined", "sharp", "sclerotic rim",
    "narrow zone", "wide zone", "transition",
    # Location
    "epiphysis", "epiphyseal", "metaphysis", "metaphyseal",
    "diaphysis", "diaphyseal", "medullary", "intramedullary",
    # Bone anatomy
    "trabecular", "trabeculae", "trabeculation", "cancellous",
    "endosteal", "subarticular", "articular",
    # Shape/contour
    "lobulated", "septation", "septated", "multiloculated",
    "unilocular", "eccentric", "central", "exophytic",
    # Density
    "radiolucent", "radiopaque", "lucent", "dense", "opacity",
    # Soft tissue
    "soft tissue", "extra-osseous", "extraosseous",
    # Common tumor descriptors
    "lesion", "tumor", "mass", "cyst", "nodule", "outgrowth",
    "exostosis", "osteochondroma", "projection",
}

# Class-distinctive terms for bone tumors (boost score if present)
CLASS_DISTINCTIVE_TERMS: Dict[str, Set[str]] = {
    "giant cell tumor": {
        "epiphyseal", "epimetaphyseal", "subarticular", "soap-bubble",
        "eccentric", "expansile", "non-sclerotic", "multiloculated",
    },
    "osteochondroma": {
        "exostosis", "outgrowth", "pedunculated", "sessile", "stalk",
        "cortical continuity", "medullary continuity", "cartilage cap",
        "metaphyseal", "pointing away",
    },
    "osteofibroma": {
        "intracortical", "cortical", "ground-glass", "fibrous",
        "tibial", "anterior bowing", "shepherd's crook",
    },
    "osteosarcoma": {
        "aggressive", "sunburst", "codman", "periosteal reaction",
        "moth-eaten", "permeative", "osteoid", "cloud-like",
        "soft tissue mass", "cortical destruction",
    },
    "simple bone cyst": {
        "unilocular", "central", "metaphyseal", "fallen fragment",
        "well-defined", "thin cortex", "fluid", "homogeneous",
    },
    "synovial osteochondroma": {
        "intra-articular", "periarticular", "loose bodies", "calcified",
        "ring-arc", "multiple", "joint", "synovial",
    },
}

CLASS_RULES: Dict[str, Dict[str, Set[str]]] = {
    "osteosarcoma": {
        "preferred": {"sunburst", "codman", "periosteal", "cortical destruction", "mixed lytic-sclerotic", "soft tissue"},
        "suspicious": {"smooth", "well-defined solitary", "benign", "pedunculated"},
        "forbidden": {"cartilage cap", "medullary continuity", "intra-articular loose bodies"},
    },
    "giant cell tumor": {
        "preferred": {"expansile", "lytic", "cortical thinning", "septation", "eccentric", "subarticular"},
        "suspicious": {"sunburst", "codman", "aggressive periosteal"},
        "forbidden": {"pedunculated exostosis", "medullary continuity"},
    },
    "osteochondroma": {
        "preferred": {"metaphyseal", "bony projection", "pedunculated", "sessile", "exostosis", "cortical continuity", "medullary continuity", "cartilage cap"},
        "suspicious": {"central lytic", "aggressive periosteal", "soft tissue mass"},
        "forbidden": {"sunburst", "codman triangle", "permeative"},
    },
    "simple bone cyst": {
        "preferred": {"unilocular", "central", "lytic", "well-defined", "no periosteal", "medullary", "diaphyseal"},
        "suspicious": {"soft tissue mass", "sunburst", "codman", "aggressive"},
        "forbidden": {"periosteal reaction", "moth-eaten", "permeative"},
    },
    "synovial osteochondroma": {
        "preferred": {"intra-articular", "periarticular", "calcified nodules", "loose bodies", "synovial"},
        "suspicious": {"ground-glass", "aggressive periosteal", "sunburst"},
        "forbidden": {"codman", "permeative destruction"},
    },
    "osteofibroma": {
        "preferred": {"cortical", "fibro-osseous", "well-defined", "expansile", "sclerotic border"},
        "suspicious": {"sunburst", "codman", "large soft tissue mass"},
        "forbidden": {"aggressive periosteal", "permeative", "moth-eaten"},
    },
}

# Forbidden phrases that indicate generic or low-quality output
FORBIDDEN_PHRASES: List[str] = [
    "this x-ray shows", "the image reveals", "the radiograph demonstrates",
    "a radiograph of", "indicative of", "suggesting", "consistent with",
    "may be", "might be", "could be", "possibly", "suspicious for",
    "bone structure remains", "lesion is visible", "abnormal bone appearance",
    "bone abnormality", "visible abnormality", "pathological changes",
]

# Compiled patterns
_FORBIDDEN_RE = re.compile(
    "|".join(re.escape(p) for p in FORBIDDEN_PHRASES),
    re.IGNORECASE,
)
_WEAK_STARTS_RE = re.compile(
    r"^(the |a |an |there is |there are |this |these |it |showing )",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredDescription:
    """A description with its quality score and breakdown."""
    text: str
    total_score: float
    radiographic_count: int
    distinctive_count: int
    length_score: float
    has_forbidden: bool
    weak_start: bool
    class_rule_hits: Dict[str, List[str]]

    def __lt__(self, other: "ScoredDescription") -> bool:
        return self.total_score < other.total_score


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class DescriptionScorer:
    """Score and rank generated descriptions for quality and discriminability.

    Scoring criteria:
        - Radiographic term count (more = better)
        - Class-distinctive term count (more = better)
        - Optimal length (not too short, not too long)
        - Absence of forbidden phrases
        - Strong opening (not weak/generic start)
        - Cross-class uniqueness (penalty for descriptions similar to other classes)
    """

    def __init__(
        self,
        ideal_min_length: int = 40,
        ideal_max_length: int = 260,
        radiographic_weight: float = 2.0,
        distinctive_weight: float = 3.0,
        length_weight: float = 1.0,
        forbidden_penalty: float = -10.0,
        weak_start_penalty: float = -2.0,
        cross_class_dup_threshold: float = 0.75,
        cross_class_penalty: float = -5.0,
    ) -> None:
        self.ideal_min = ideal_min_length
        self.ideal_max = ideal_max_length
        self.w_radio = radiographic_weight
        self.w_distinct = distinctive_weight
        self.w_length = length_weight
        self.penalty_forbidden = forbidden_penalty
        self.penalty_weak = weak_start_penalty
        self.cross_dup_thresh = cross_class_dup_threshold
        self.cross_penalty = cross_class_penalty

    def score(
        self,
        text: str,
        class_name: Optional[str] = None,
    ) -> ScoredDescription:
        """Compute quality score for a single description."""
        lower = text.lower()

        # Radiographic term count
        radio_count = sum(1 for t in RADIOGRAPHIC_TERMS if t in lower)

        # Class-distinctive term count
        distinct_count = 0
        if class_name and class_name.lower() in CLASS_DISTINCTIVE_TERMS:
            for t in CLASS_DISTINCTIVE_TERMS[class_name.lower()]:
                if t in lower:
                    distinct_count += 1

        # Length score (bell curve around ideal range)
        length = len(text)
        if self.ideal_min <= length <= self.ideal_max:
            length_score = 1.0
        elif length < self.ideal_min:
            length_score = max(0, length / self.ideal_min)
        else:
            overage = length - self.ideal_max
            length_score = max(0, 1.0 - overage / 100)

        # Forbidden phrase check
        has_forbidden = bool(_FORBIDDEN_RE.search(text))

        # Weak start check
        weak_start = bool(_WEAK_STARTS_RE.match(text))

        rule_hits = {"preferred": [], "suspicious": [], "forbidden": []}
        class_bonus = 0.0
        if class_name and class_name.lower() in CLASS_RULES:
            rules = CLASS_RULES[class_name.lower()]
            for t in rules["preferred"]:
                if t in lower:
                    rule_hits["preferred"].append(t)
                    class_bonus += 2.0
            for t in rules["suspicious"]:
                if t in lower:
                    rule_hits["suspicious"].append(t)
                    class_bonus -= 2.5
            for t in rules["forbidden"]:
                if t in lower:
                    rule_hits["forbidden"].append(t)
                    class_bonus -= 6.0

        # Total score
        total = (
            radio_count * self.w_radio
            + distinct_count * self.w_distinct
            + length_score * self.w_length
            + (self.penalty_forbidden if has_forbidden else 0)
            + (self.penalty_weak if weak_start else 0)
            + class_bonus
        )

        return ScoredDescription(
            text=text,
            total_score=total,
            radiographic_count=radio_count,
            distinctive_count=distinct_count,
            length_score=length_score,
            has_forbidden=has_forbidden,
            weak_start=weak_start,
            class_rule_hits=rule_hits,
        )

    def score_batch(
        self,
        descriptions: List[str],
        class_name: Optional[str] = None,
    ) -> List[ScoredDescription]:
        """Score a batch of descriptions and return sorted by score (descending)."""
        scored = [self.score(d, class_name) for d in descriptions]
        return sorted(scored, reverse=True)

    def filter_and_rank(
        self,
        descriptions: List[str],
        class_name: Optional[str] = None,
        min_score: float = 0.0,
        max_keep: Optional[int] = None,
    ) -> List[str]:
        """Score, filter by minimum score, and return top descriptions."""
        scored = self.score_batch(descriptions, class_name)
        filtered = [s for s in scored if s.total_score >= min_score and not s.has_forbidden]
        if max_keep:
            filtered = filtered[:max_keep]
        return [s.text for s in filtered]

    def dedupe_cross_class(
        self,
        class_descriptions: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Remove descriptions that are too similar across different classes.

        For each class, penalize descriptions that are near-duplicates of
        descriptions from other classes (since they're not discriminative).
        """
        result: Dict[str, List[str]] = {}

        # Collect all descriptions from all other classes
        for target_class, descs in class_descriptions.items():
            other_descs: List[str] = []
            for cls, d_list in class_descriptions.items():
                if cls != target_class:
                    other_descs.extend(d_list)

            # Keep only descriptions that are sufficiently different from others
            kept: List[str] = []
            for desc in descs:
                is_cross_dup = False
                d_lower = desc.lower()
                for other in other_descs:
                    ratio = SequenceMatcher(None, d_lower, other.lower()).ratio()
                    if ratio >= self.cross_dup_thresh:
                        is_cross_dup = True
                        break
                if not is_cross_dup:
                    kept.append(desc)
            result[target_class] = kept

        return result

    def select_diverse_topk(
        self,
        descriptions: List[str],
        class_name: Optional[str] = None,
        k: int = 8,
        min_score: float = 0.0,
    ) -> List[str]:
        """Select top-k descriptions while maximizing feature-category diversity."""
        scored = [s for s in self.score_batch(descriptions, class_name) if s.total_score >= min_score and not s.has_forbidden]
        if not scored:
            return []

        selected: List[str] = []
        used_categories: Set[str] = set()

        for item in scored:
            cat = self._feature_category(item.text)
            if cat not in used_categories:
                selected.append(item.text)
                used_categories.add(cat)
            if len(selected) >= k:
                return selected

        for item in scored:
            if item.text not in selected:
                selected.append(item.text)
            if len(selected) >= k:
                break
        return selected

    @staticmethod
    def _feature_category(text: str) -> str:
        """Map description to a coarse visual feature category."""
        t = text.lower()
        if any(x in t for x in ["cortex", "cortical", "breach", "thinning", "destruction"]):
            return "cortical"
        if any(x in t for x in ["trabec", "cancellous"]):
            return "trabecular"
        if any(x in t for x in ["periosteal", "sunburst", "codman", "lamellated", "spiculated"]):
            return "periosteal"
        if any(x in t for x in ["lytic", "sclerotic", "permeative", "moth-eaten", "geographic", "mixed"]):
            return "lesion_pattern"
        if any(x in t for x in ["matrix", "osteoid", "chondroid", "ground-glass", "ring-arc", "calcif"]):
            return "matrix"
        if any(x in t for x in ["epiph", "metaph", "diaph", "subarticular", "intramedullary"]):
            return "location"
        if any(x in t for x in ["expansile", "expansion", "eccentric", "unilocular", "multiloculated", "shape", "lobulated"]):
            return "shape_growth"
        if any(x in t for x in ["soft tissue", "extra-osseous", "extraosseous"]):
            return "soft_tissue"
        return "other"
