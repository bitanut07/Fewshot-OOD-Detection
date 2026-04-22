# -*- coding: utf-8 -*-
"""Prompt builder for multi-stage discriminative description generation.

Implements a two-stage generation strategy:
    Stage A: Generate discriminative attributes per class
    Stage B: Generate final CLIP-friendly descriptions from attributes

All prompts include cross-class awareness to maximize discriminability.
Only builds prompts for **known in-distribution classes**.
"""
from __future__ import annotations

from typing import List, Optional


class PromptBuilder:
    """Builds prompts for multi-stage discriminative text generation.

    The generation workflow:
        1. Generate dataset-level discriminative questions
        2. For each class:
           a. Generate discriminative attributes (Stage A)
           b. Generate final descriptions from attributes (Stage B)

    All class-level prompts include cross-class awareness.
    """

    _DEFAULT_PROMPT_TPL = "a radiograph of {class_name}"

    # ─────────────────────────────────────────────────────────────────────────
    # Question generation prompt
    # ─────────────────────────────────────────────────────────────────────────

    _QUESTION_PROMPT = """\
You are a musculoskeletal radiologist specializing in bone tumor X-ray analysis.

Dataset: Bone X-ray images for few-shot classification of bone tumors.
Known tumor classes: {class_list}

Task:
Generate exactly {num_questions} diagnostic questions that help visually distinguish
between these bone tumor types on plain X-ray radiographs.

Hard constraints:
- Focus ONLY on features visible in plain X-ray images
- Do NOT mention clinical symptoms (pain, fever, tenderness, swelling)
- Do NOT mention patient history, demographics, lab results, or MRI/CT findings
- Do NOT use uncertainty hedging (may, might, possibly, could)
- Each question must target a specific visual discriminator between tumor types

Focus on these radiographic discriminators:
- Lesion pattern: lytic vs sclerotic vs mixed
- Margin characteristics: well-defined vs ill-defined, sclerotic rim vs none
- Location within bone: epiphysis vs metaphysis vs diaphysis
- Cortical integrity: intact vs thinned vs breached vs destroyed
- Periosteal reaction: none vs smooth vs spiculated vs sunburst vs Codman triangle
- Matrix mineralization: absent vs ground-glass vs cloud-like vs ring-arc
- Growth pattern: central vs eccentric vs exophytic
- Multiplicity: solitary vs multiple lesions
- Soft tissue extension: present vs absent

Output format:
- Exactly {num_questions} questions, one per line
- No numbering, no bullets, no explanations
- Each question should help differentiate at least 2 tumor types"""

    # ─────────────────────────────────────────────────────────────────────────
    # Stage A: Attribute extraction prompt
    # ─────────────────────────────────────────────────────────────────────────

    _ATTRIBUTE_PROMPT = """\
You are a musculoskeletal radiologist. Your task is to identify the key VISUAL 
features that distinguish "{class_name}" from other bone tumors on plain X-ray.

Other known bone tumor classes (for comparison):
{other_classes}

Task:
List exactly {num_attributes} distinctive radiographic attributes of "{class_name}" 
that help differentiate it from the other tumor types listed above.

Hard constraints:
- Each attribute must be a short phrase (3-8 words)
- Focus ONLY on features visible on plain X-ray
- Do NOT mention symptoms, treatment, prognosis, or patient history
- Do NOT use uncertainty words (may, might, possibly, could)
- Each attribute should be a DISTINCTIVE feature, not generic

Focus on what makes "{class_name}" visually DIFFERENT from other tumors:
- Typical location within the bone
- Characteristic lesion pattern (lytic/sclerotic/mixed)
- Distinctive margin appearance
- Typical periosteal reaction pattern
- Characteristic matrix pattern if present
- Typical growth pattern or shape
- Cortical involvement pattern

Output format:
- Exactly {num_attributes} attributes, one per line
- Short phrases only (3-8 words each)
- No numbering, no bullets, no explanations
- No sentences - just attribute phrases"""

    # ─────────────────────────────────────────────────────────────────────────
    # Stage B: Description generation from attributes
    # ─────────────────────────────────────────────────────────────────────────

    _DESCRIPTION_FROM_ATTRIBUTES_PROMPT = """\
You are a musculoskeletal radiologist generating CLIP-friendly text descriptions.

Target class: "{class_name}"
Other known classes: {other_classes}

Key discriminative attributes of "{class_name}":
{attributes_block}

{questions_block}

Task:
Generate exactly {num_descriptions} diverse, concise visual descriptions of 
"{class_name}" as seen on plain X-ray. Each description should emphasize what 
makes this tumor type DIFFERENT from: {other_classes}

STRICT requirements:
- Each description is exactly ONE short sentence (10-35 words ideal)
- Describe ONLY features visible on plain X-ray
- Each description must highlight a DISTINCT visual pattern
- Descriptions must help a vision-language model distinguish "{class_name}" 
  from the other tumor types

FORBIDDEN (reject any output containing these):
- "This X-ray shows" / "The image reveals" / "A radiograph of"
- "indicative of" / "suggesting" / "consistent with"
- "may" / "might" / "possibly" / "could"
- Generic phrases like "bone abnormality" / "lesion is visible"
- Clinical symptoms (pain, swelling, tenderness)
- Patient history, demographics, lab results

REQUIRED features to incorporate:
- Radiographic terminology (lytic, sclerotic, cortical, periosteal, etc.)
- Specific anatomical location when relevant
- Distinctive morphological features

Output:
- Exactly {num_descriptions} descriptions, one per line
- No numbering, no bullets, no explanations
- Each line is a complete, self-contained visual description"""

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback: Direct description generation (without attributes)
    # ─────────────────────────────────────────────────────────────────────────

    _DIRECT_DESCRIPTION_PROMPT = """\
You are a musculoskeletal radiologist generating CLIP-friendly text descriptions.

Target class: "{class_name}"
Other known classes for comparison: {other_classes}

{questions_block}

Task:
Generate exactly {num_descriptions} diverse, concise visual descriptions of 
"{class_name}" as seen on plain X-ray that help distinguish it from the other 
tumor types listed above.

STRICT requirements:
- Each description is exactly ONE short sentence (10-35 words ideal)
- Describe ONLY features visible on plain X-ray
- Each description must highlight a DISTINCT visual pattern
- Focus on what makes "{class_name}" DIFFERENT from other tumor types

FORBIDDEN:
- "This X-ray shows" / "The image reveals" / "A radiograph of"
- "indicative of" / "suggesting" / "consistent with" / "suspicious for"
- "may" / "might" / "possibly" / "could" / "sometimes"
- Generic phrases: "bone abnormality" / "lesion is visible" / "abnormal appearance"
- Clinical symptoms, patient history, demographics

REQUIRED:
- Use precise radiographic terminology
- Include specific visual discriminators (pattern, location, margins, etc.)

Output:
- Exactly {num_descriptions} descriptions, one per line
- No numbering, no bullets, no explanations"""

    # ─────────────────────────────────────────────────────────────────────────
    # Targeted retry prompt (for missing descriptions)
    # ─────────────────────────────────────────────────────────────────────────

    _RETRY_PROMPT = """\
You are a musculoskeletal radiologist. Generate {num_needed} MORE visual 
descriptions for "{class_name}" that are DIFFERENT from the ones already accepted.

Already accepted descriptions (DO NOT repeat or paraphrase these):
{existing_descriptions}

Other known classes: {other_classes}

Preferred class-specific visual features:
{preferred_features}

Features to avoid (mismatched or unlikely for this class):
{avoid_features}

Requirements:
- Generate exactly {num_needed} NEW descriptions, one per line
- Each must be visually DISTINCT from the accepted ones above
- Focus on different radiographic features not yet covered
- Each description: ONE sentence, concise (about 6-35 words), plain X-ray features only
- Output must be English-only text

FORBIDDEN: "This X-ray shows", "may", "might", "possibly", symptoms, history

Output: {num_needed} new descriptions only, one per line, no numbering"""

    # ─────────────────────────────────────────────────────────────────────────
    # Constructor
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        dataset_description: str = "",
        default_prompt_template: Optional[str] = None,
        question_prompt_template: Optional[str] = None,
        attribute_prompt_template: Optional[str] = None,
        description_prompt_template: Optional[str] = None,
    ) -> None:
        self.dataset_description = dataset_description
        self._default_tpl = default_prompt_template or self._DEFAULT_PROMPT_TPL
        self._question_tpl = question_prompt_template or self._QUESTION_PROMPT
        self._attribute_tpl = attribute_prompt_template or self._ATTRIBUTE_PROMPT
        self._description_tpl = description_prompt_template or self._DESCRIPTION_FROM_ATTRIBUTES_PROMPT
        self._direct_desc_tpl = self._DIRECT_DESCRIPTION_PROMPT
        self._retry_tpl = self._RETRY_PROMPT

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def default_prompt(self, class_name: str) -> str:
        """Return the single default CLIP prompt for a class."""
        return self._default_tpl.format(class_name=class_name)

    def build_question_prompt(
        self,
        num_questions: int,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for generating discriminative questions."""
        class_list = ", ".join(class_names) if class_names else "various bone tumors"
        return self._question_tpl.format(
            num_questions=num_questions,
            class_list=class_list,
        )

    def build_attribute_prompt(
        self,
        class_name: str,
        other_classes: List[str],
        num_attributes: int = 6,
    ) -> str:
        """Build Stage A prompt for extracting discriminative attributes."""
        other_list = ", ".join(c for c in other_classes if c != class_name)
        return self._attribute_tpl.format(
            class_name=class_name,
            other_classes=other_list,
            num_attributes=num_attributes,
        )

    def build_description_prompt(
        self,
        class_name: str,
        num_descriptions: int,
        other_classes: Optional[List[str]] = None,
        attributes: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
    ) -> str:
        """Build Stage B prompt for generating descriptions from attributes.

        If attributes are provided, uses the attribute-guided template.
        Otherwise falls back to direct description generation.
        """
        other_list = ", ".join(c for c in (other_classes or []) if c != class_name)
        if not other_list:
            other_list = "other bone tumors"

        questions_block = ""
        if questions:
            questions_block = (
                "Diagnostic questions to guide description diversity:\n"
                + "\n".join(f"- {q}" for q in questions[:10])
                + "\n\n"
            )

        if attributes:
            attributes_block = "\n".join(f"- {a}" for a in attributes)
            return self._description_tpl.format(
                class_name=class_name,
                other_classes=other_list,
                attributes_block=attributes_block,
                questions_block=questions_block,
                num_descriptions=num_descriptions,
            )
        else:
            return self._direct_desc_tpl.format(
                class_name=class_name,
                other_classes=other_list,
                questions_block=questions_block,
                num_descriptions=num_descriptions,
            )

    def build_retry_prompt(
        self,
        class_name: str,
        num_needed: int,
        existing_descriptions: List[str],
        other_classes: Optional[List[str]] = None,
        preferred_features: Optional[List[str]] = None,
        avoid_features: Optional[List[str]] = None,
    ) -> str:
        """Build targeted retry prompt for generating additional descriptions."""
        other_list = ", ".join(c for c in (other_classes or []) if c != class_name)
        if not other_list:
            other_list = "other bone tumors"

        existing_block = "\n".join(f"- {d}" for d in existing_descriptions)
        preferred_block = "\n".join(f"- {d}" for d in (preferred_features or [])) or "- class-specific radiographic morphology"
        avoid_block = "\n".join(f"- {d}" for d in (avoid_features or [])) or "- non-specific generic findings"

        return self._retry_tpl.format(
            class_name=class_name,
            num_needed=num_needed,
            existing_descriptions=existing_block,
            other_classes=other_list,
            preferred_features=preferred_block,
            avoid_features=avoid_block,
        )
