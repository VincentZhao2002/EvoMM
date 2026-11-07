"""
SelfEvolvingResearchAgent

Extends the self-correcting research pipeline with a cross-task experience
loop: before each research run it retrieves relevant past experiences to
augment planning/integration prompts, and after each run it distils the
trace into a new experience entry for future reuse.

When the experience store is empty the behaviour is identical to
SelfCorrectingResearchAgent, so the upgrade is fully backward-compatible.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from evomm.prompts.experience_prompts import (
    EXPERIENCE_EXTRACT_PROMPT,
    format_experience_context,
)
from evomm.schemas.experience import (
    Experience,
    ExperienceStore,
    EXPERIENCE_EXTRACT_SCHEMA,
)
from evomm.schemas import ResearchOutput, PageStore, MemoryStore, Retriever, ToolRegistry
from evomm.generator import AbsGenerator


class SelfEvolvingResearchAgent:
    """Drop-in replacement that adds experience-driven self-evolution.

    Because SelfCorrectingResearchAgent lives in eval/ (outside the core
    package) and is imported dynamically at eval time, this class accepts
    a *base_agent_cls* parameter so it can wrap any ResearchAgent subclass
    (including SelfCorrectingResearchAgent) without creating a hard import
    dependency on eval code.

    Alternatively, call ``wrap()`` to retroactively attach experience
    evolution to an already-instantiated agent.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        *,
        base_agent_cls=None,
        evolution_enabled: bool = True,
        # pass-through kwargs for the underlying agent
        page_store: Optional[PageStore] = None,
        memory_store: Optional[MemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: Optional[AbsGenerator] = None,
        max_iters: int = 3,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if base_agent_cls is None:
            from evomm.agents.research_agent import ResearchAgent
            base_agent_cls = ResearchAgent

        self.experience_store = experience_store
        self.evolution_enabled = evolution_enabled

        self._base_agent = base_agent_cls(
            page_store=page_store,
            memory_store=memory_store,
            tool_registry=tool_registry,
            retrievers=retrievers,
            generator=generator,
            max_iters=max_iters,
            dir_path=dir_path,
            system_prompts=dict(system_prompts) if system_prompts else None,
        )

        self._original_system_prompts = dict(self._base_agent.system_prompts)

    @classmethod
    def wrap(
        cls,
        agent,
        experience_store: ExperienceStore,
        evolution_enabled: bool = True,
    ) -> "SelfEvolvingResearchAgent":
        """Wrap an already-constructed agent with experience evolution."""
        obj = cls.__new__(cls)
        obj.experience_store = experience_store
        obj.evolution_enabled = evolution_enabled
        obj._base_agent = agent
        obj._original_system_prompts = dict(agent.system_prompts)
        return obj

    def __getattr__(self, name: str):
        """Proxy attribute access to the underlying agent."""
        return getattr(self._base_agent, name)

    def research(self, request: str, category: str = "") -> ResearchOutput:
        experience_trace: Dict[str, Any] = {
            "enabled": self.evolution_enabled,
            "category": category,
            "retrieved_experiences": [],
            "injected_context": "",
            "extract_prompt": None,
            "extract_response": None,
            "extract_parsed": None,
            "extract_error": None,
            "stored_experience": None,
        }

        if self.evolution_enabled:
            self._inject_experience(request, category, experience_trace)

        output = self._base_agent.research(request)

        if self.evolution_enabled:
            self._extract_and_store(request, output, category, experience_trace)

        self._restore_system_prompts()
        if isinstance(output.raw_memory, dict):
            output.raw_memory["experience_evolution"] = experience_trace
        return output

    def _inject_experience(self, request: str, category: str, experience_trace: Dict[str, Any]) -> None:
        relevant = self.experience_store.search(
            request, top_k=3, category=category or None
        )
        experience_trace["retrieved_experiences"] = [exp.model_dump() for exp in relevant]
        if not relevant:
            return

        context_block = format_experience_context(relevant)
        experience_trace["injected_context"] = context_block
        if not context_block:
            return

        for key in ("planning", "integration"):
            current = self._base_agent.system_prompts.get(key, "")
            self._base_agent.system_prompts[key] = (
                f"{current}\n\n{context_block}" if current else context_block
            )

    def _restore_system_prompts(self) -> None:
        self._base_agent.system_prompts = dict(self._original_system_prompts)

    def _extract_and_store(
        self,
        request: str,
        output: ResearchOutput,
        category: str,
        experience_trace: Dict[str, Any],
    ) -> None:
        raw = output.raw_memory if output else {}
        iterations = raw.get("iterations", []) if isinstance(raw, dict) else []

        tools_used: List[str] = []
        all_queries: List[str] = []
        for it in iterations:
            plan = it.get("plan", {})
            tools_used.extend(plan.get("tools", []))
            all_queries.extend(plan.get("keyword_collection", []))
            all_queries.extend(plan.get("vector_queries", []))

        tools_used = list(dict.fromkeys(tools_used))
        search_rounds = len(iterations)
        integrated_abbrev = (output.integrated_memory or "")[:600]

        prompt = EXPERIENCE_EXTRACT_PROMPT.format(
            question=request,
            category=category,
            search_rounds=search_rounds,
            tools_used=", ".join(tools_used) if tools_used else "none",
            queries_issued="; ".join(all_queries[:10]) if all_queries else "none",
            integrated_result=integrated_abbrev,
        )
        experience_trace["extract_prompt"] = prompt

        try:
            response = self._base_agent.generator.generate_single(
                prompt=prompt, schema=EXPERIENCE_EXTRACT_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])
            experience_trace["extract_response"] = response
            experience_trace["extract_parsed"] = data

            exp = Experience(
                question=request,
                category=category,
                tools_used=tools_used,
                search_rounds=search_rounds,
                effective_queries=data.get("effective_queries", all_queries[:5]),
                key_insight=data.get("key_insight", integrated_abbrev[:200]),
                self_assessment=data.get("self_assessment", ""),
                lessons=data.get("lessons", []),
            )
            self.experience_store.add(exp)
            experience_trace["stored_experience"] = exp.model_dump()
        except Exception as e:
            print(f"[SelfEvolvingResearchAgent] experience extraction failed: {e}")
            experience_trace["extract_error"] = str(e)
            exp = Experience(
                question=request,
                category=category,
                tools_used=tools_used,
                search_rounds=search_rounds,
                effective_queries=all_queries[:5],
                key_insight=integrated_abbrev[:200],
                self_assessment="extraction_failed",
                lessons=[],
            )
            self.experience_store.add(exp)
            experience_trace["stored_experience"] = exp.model_dump()
