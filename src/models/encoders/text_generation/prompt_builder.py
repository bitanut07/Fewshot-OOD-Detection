# -*- coding: utf-8 -*-
"""Prompt builder for discriminative questions and class descriptions.

Constructs well-structured prompts with hard constraints that ensure
generated text is visually grounded, medically precise, and suitable
for CLIP text embeddings.

Only builds prompts for **known in-distribution classes**.
"""
from __future__ import annotations

from typing import List, Optional


class PromptBuilder:
    """Builds prompts for question and description generation.

    Parameters
    ----------
    dataset_description:
        Domain context injected into every prompt.
    default_prompt_template:
        Template for the per-class default prompt.
        Must contain ``{class_name}``.
    question_prompt_template / description_prompt_template:
        Override templates from config.  ``None`` → use built-in defaults.
    """

    _DEFAULT_PROMPT_TPL = "a bone X-ray image showing {class_name}"

    _BUILTIN_QUESTION_PROMPT = """\
You are a musculoskeletal radiologist specializing in bone tumor X-ray analysis.

Dataset description:
{dataset_description}

Task:
Generate exactly {num_questions} diagnostic questions that help distinguish
between different bone tumor types visible on plain X-ray radiographs.

Hard constraints:
- Focus ONLY on features visible in plain X-ray images
- Do NOT mention clinical symptoms (pain, fever, tenderness, swelling)
- Do NOT mention patient history, demographics, or lab results
- Do NOT use uncertainty hedging (may, might, possibly, could)
- Do NOT start with "This X-ray shows" or similar generic openings
- Avoid absolute spatial terms (left/right/top/bottom)

Focus areas:
- lesion morphology (lytic, sclerotic, mixed, expansile)
- cortical integrity (intact, thinned, breached, destroyed)
- periosteal reaction (none, smooth, spiculated, Codman triangle)
- matrix mineralization (absent, ground-glass, cloud-like, ring-arc)
- bone contour changes (expansion, deformity, outgrowth)
- margin characteristics (well-defined, ill-defined, sclerotic rim)
- trabecular pattern (preserved, rarefied, destroyed)
- intra/extra-osseous extension

Output:
- Exactly {num_questions} questions, one per line
- No numbering, no bullets, no explanations"""

    _BUILTIN_DESCRIPTION_PROMPT = """\
You are a musculoskeletal radiologist specializing in bone tumor X-ray interpretation.

Target bone tumor class: "{class_name}"

{questions_block}

Task:
Generate exactly {num_descriptions} diverse, radiographically accurate visual
descriptions of "{class_name}" as seen in plain X-ray images.

Hard constraints:
- Each description is exactly ONE sentence
- Describe ONLY features visible on plain radiographs
- Do NOT mention clinical symptoms (pain, tenderness, swelling, fever)
- Do NOT mention patient history, age, gender, or lab results
- Do NOT use uncertainty hedging (may, might, possibly, could, sometimes)
- Do NOT start with "This X-ray shows", "The image reveals", "A radiograph of"
- Do NOT use absolute spatial terms (left/right/top/bottom)
- Each sentence MUST highlight a distinct radiographic feature
- Use precise radiological terminology
- Descriptions must be concise and suitable for a vision-language model

Focus areas:
- lesion location (epiphysis, metaphysis, diaphysis)
- lesion pattern (lytic, sclerotic, mixed, geographic, permeative, moth-eaten)
- cortical changes (thinning, expansion, breach, destruction)
- periosteal reaction (lamellated, sunburst, Codman triangle, none)
- matrix pattern (osteoid, chondroid, ground-glass, absent)
- margin definition (sharp, sclerotic rim, ill-defined)
- bone contour and trabecular architecture changes
- soft tissue component presence

Output:
- Exactly {num_descriptions} descriptions, one per line
- No numbering, no bullets, no explanations"""

    def __init__(
        self,
        dataset_description: str,
        default_prompt_template: Optional[str] = None,
        question_prompt_template: Optional[str] = None,
        description_prompt_template: Optional[str] = None,
    ) -> None:
        self.dataset_description = dataset_description
        self._default_tpl = default_prompt_template or self._DEFAULT_PROMPT_TPL
        self._question_tpl = question_prompt_template or self._BUILTIN_QUESTION_PROMPT
        self._description_tpl = description_prompt_template or self._BUILTIN_DESCRIPTION_PROMPT

    # -- Public API -----------------------------------------------------------

    def default_prompt(self, class_name: str) -> str:
        """Return the single default prompt for *class_name*."""
        return self._default_tpl.format(class_name=class_name)

    def build_question_prompt(self, num_questions: int) -> str:
        return self._question_tpl.format(
            num_questions=num_questions,
            dataset_description=self.dataset_description,
        )

    def build_description_prompt(
        self,
        class_name: str,
        num_descriptions: int,
        questions: Optional[List[str]] = None,
    ) -> str:
        if questions:
            block = (
                "Diagnostic context (use to guide description diversity):\n"
                + "\n".join(f"- {q}" for q in questions[:15])
                + "\n\n"
            )
        else:
            block = ""

        return self._description_tpl.format(
            class_name=class_name,
            num_descriptions=num_descriptions,
            questions_block=block,
            dataset_description=self.dataset_description,
        )
