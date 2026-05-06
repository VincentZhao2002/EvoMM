#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-correcting research agent for EvoMM evaluation.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from evomm import ResearchAgent
from evomm.schemas import Hit, Result


SELF_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "critique": {"type": "string", "description": "Critique of the draft answer based on evidence."},
        "is_correct": {
            "type": "boolean",
            "description": "Whether the draft answer is fully supported and correct.",
        },
        "corrected_answer": {
            "type": "string",
            "description": "The corrected answer. If is_correct is true, keep the draft answer.",
        },
    },
    "required": ["critique", "is_correct", "corrected_answer"],
}


def _coerce_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


class SelfCorrectingResearchAgent(ResearchAgent):
    """ResearchAgent with a post-integration self-correction pass."""

    def _integrate(
        self,
        hits: List[Hit],
        result: Result,
        question: str,
        integration_prompt: Optional[str] = None,
    ) -> Result:
        initial_result = super()._integrate(hits, result, question, integration_prompt)

        if not hits or not initial_result.content:
            return initial_result

        evidence_text = []
        for i, hit in enumerate(hits[: self.MAX_INTEGRATE_HITS], 1):
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            snippet = hit.snippet[: self.MAX_SNIPPET_CHARS]
            evidence_text.append(f"{i}. {source_info} {snippet}")
        evidence_context = "\n".join(evidence_text)

        prompt = f"""
You are a strict but pragmatic fact-checker and editor for long-conversation memory.

Question:
{question}

Retrieved Evidence (numbered):
{evidence_context}

Draft Answer (Memory Summary):
{initial_result.content}

EDITING GOALS:
- Produce a memory summary that is (1) NOT contradictory with Evidence, and (2) most useful for answering the Question.
- LoCoMo-style questions may require recalling specific facts across long sessions and different reasoning types (single-hop, multi-hop, temporal, commonsense/world knowledge, adversarial).
  Prioritize facts that directly support answering the Question.

CONSERVATIVE POLICY (follow exactly):
A) Do NOT delete or rewrite statements solely because they are not mentioned in the Evidence.
   (Lack of mention is not a reason to change.)
B) You MUST edit if there is a CONTRADICTION with Evidence.
C) You SHOULD edit if there is a CRITICAL OMISSION:
   Evidence contains information essential to answer the Question correctly, and omission would be misleading or materially incomplete.
D) You MAY make small "HELPFUL" edits even without (B) or (C), but ONLY if they improve answerability for the Question AND are fully evidence-grounded:
   - Reorder or condense the Draft Answer to highlight question-relevant facts.
   - Add an evidence-supported detail that makes the answer clearer (e.g., exact date/time, identity, key event), when it is clearly relevant.
   - Remove hedging or verbosity, but do not remove meaning.
E) Never add speculation. Never invent details. Use ONLY what is explicitly supported by Evidence when you add/alter content.

CHANGE JUSTIFICATION RULE:
- For every change you make (including additions or deletions), you must be able to cite one or more Evidence item numbers that justify that change.
- If you do not make changes, say so.

OUTPUT REQUIREMENTS:
Return ONLY valid JSON with exactly these keys:
- "critique": Briefly state what you did:
    - "No change needed" OR
    - which of (B) CONTRADICTION / (C) CRITICAL OMISSION / (D) HELPFUL EDITS applied,
    - and list the Evidence item numbers used.
- "is_correct": true ONLY if you made NO edits at all.
- "corrected_answer":
    - If "is_correct" is true, it MUST be EXACTLY identical to the Draft Answer (character-for-character).
    - Otherwise, provide the minimally edited improved summary.
    - It MUST NOT include critique text, evidence item numbers, citation notes, or change justifications.
    - It MUST NOT contain phrases like "Evidence item numbers used", "No change needed", or "I changed".

Output ONLY JSON. No extra text.
"""

        try:
            response = self.generator.generate_single(prompt=prompt, schema=SELF_CORRECTION_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            if hasattr(self, "_record_llm_call"):
                self._record_llm_call(
                    phase="self_correction",
                    prompt=prompt,
                    schema=SELF_CORRECTION_SCHEMA,
                    response=response,
                    parsed=data,
                    extra={"question": question, "initial_result": initial_result.content},
                )
            is_correct = _coerce_bool(data.get("is_correct", True), default=True)
            corrected_content = data.get("corrected_answer", initial_result.content)
            if isinstance(corrected_content, str):
                corrected_content = re.sub(
                    r"\s*Evidence item numbers used:.*$",
                    "",
                    corrected_content,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
            if not is_correct and corrected_content != initial_result.content:
                return Result(content=corrected_content, sources=initial_result.sources)
            return initial_result
        except Exception as e:
            if hasattr(self, "_record_llm_call"):
                self._record_llm_call(
                    phase="self_correction",
                    prompt=prompt,
                    schema=SELF_CORRECTION_SCHEMA,
                    error=str(e),
                    extra={"question": question, "initial_result": initial_result.content},
                )
            return initial_result
