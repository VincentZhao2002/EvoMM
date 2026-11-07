from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Experience(BaseModel):
    """A single distilled experience from a QA research trace."""

    question: str = Field(..., description="The original question")
    category: str = Field("", description="Question category (AR/CD/VS/TR/TTL)")
    tools_used: List[str] = Field(default_factory=list, description="Retrieval tools that were actually used")
    search_rounds: int = Field(0, description="Number of search iterations performed")
    effective_queries: List[str] = Field(default_factory=list, description="Queries that yielded useful results")
    key_insight: str = Field("", description="Concise summary of the integrated result")
    self_assessment: str = Field("", description="LLM self-assessment of answer quality")
    lessons: List[str] = Field(default_factory=list, description="Actionable lessons for future similar questions")
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


EXPERIENCE_EXTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "effective_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Queries from the trace that retrieved the most useful evidence.",
        },
        "key_insight": {
            "type": "string",
            "description": "One-sentence summary of the most important finding.",
        },
        "self_assessment": {
            "type": "string",
            "description": "Brief assessment: was the answer complete and accurate?",
        },
        "lessons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 actionable lessons for answering similar questions in the future.",
        },
    },
    "required": ["effective_queries", "key_insight", "self_assessment", "lessons"],
}


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization for keyword matching."""
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())


class ExperienceStore:
    """Persistent experience store backed by a JSON file.

    Retrieval uses lightweight keyword overlap + optional category filtering,
    which is sufficient for the typical scale of experiences (tens to hundreds).
    """

    def __init__(self, dir_path: Optional[str] = None) -> None:
        self._dir_path = Path(dir_path) if dir_path else None
        self._experiences: List[Experience] = []
        if self._dir_path:
            self._file = self._dir_path / "experiences.json"
            if self._file.exists():
                self._experiences = self._load_from_disk()
        else:
            self._file = None

    def _load_from_disk(self) -> List[Experience]:
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [Experience(**item) for item in raw]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: failed to load experiences from {self._file}: {e}")
            return []

    def _save_to_disk(self) -> None:
        if self._dir_path is None:
            return
        self._dir_path.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(
                    [exp.model_dump() for exp in self._experiences],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: failed to save experiences to {self._file}: {e}")

    def add(self, exp: Experience) -> None:
        self._experiences.append(exp)
        self._save_to_disk()

    def load(self) -> List[Experience]:
        return list(self._experiences)

    def save(self) -> None:
        self._save_to_disk()

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> List[Experience]:
        """Retrieve the most relevant experiences via keyword overlap scoring.

        Optionally filter by category first, then rank by token overlap between
        the query and each experience's question + lessons text.
        """
        if not self._experiences:
            return []

        candidates = self._experiences
        if category:
            cat_upper = category.upper()
            filtered = [e for e in candidates if e.category.upper() == cat_upper]
            if filtered:
                candidates = filtered

        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return candidates[:top_k]

        scored: List[tuple[float, int, Experience]] = []
        for idx, exp in enumerate(candidates):
            doc_text = f"{exp.question} {' '.join(exp.lessons)} {exp.key_insight}"
            doc_tokens = Counter(_tokenize(doc_text))
            overlap = sum((query_tokens & doc_tokens).values())
            if overlap > 0:
                scored.append((overlap, idx, exp))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [exp for _, _, exp in scored[:top_k]]
