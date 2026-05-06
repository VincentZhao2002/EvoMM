# research_agent.py
# -*- coding: utf-8 -*-
"""
ResearchAgent Module

This module defines the ResearchAgent for the EvoMM (General-Agentic-Memory) framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from evomm.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from evomm.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from evomm.generator import AbsGenerator

class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,
        max_iters: int = 3,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        
        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            self.system_prompts = {**default_system_prompts, **system_prompts}
        self._llm_trace: List[Dict[str, Any]] = []
        self._current_step: Optional[int] = None

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        self._update_retrievers()
        self._llm_trace = []
        self._current_step = None
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request
        prev_content: Optional[str] = None

        for step in range(self.max_iters):
            self._current_step = step
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            # Early stop only after collecting non-empty content. Empty summaries
            # often mean retrieval missed the target and should get follow-ups.
            if prev_content is not None and temp.content.strip() and temp.content == prev_content:
                iterations.append({
                    "step": step,
                    "plan": plan.__dict__,
                    "temp_memory": temp.__dict__,
                    "decision": {"enough": True, "new_request": None, "early_stop": "content_unchanged"},
                })
                break
            prev_content = temp.content

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request

        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
            "llm_trace": self._llm_trace,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _to_jsonable(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except Exception:
            if isinstance(value, dict):
                return {str(k): self._to_jsonable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [self._to_jsonable(v) for v in value]
            if hasattr(value, "model_dump"):
                return self._to_jsonable(value.model_dump())
            return repr(value)

    def _record_llm_call(
        self,
        phase: str,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        parsed: Optional[Any] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "step": self._current_step,
            "phase": phase,
            "prompt": prompt,
            "schema": self._to_jsonable(schema),
            "response": self._to_jsonable(response),
            "parsed": self._to_jsonable(parsed),
            "error": error,
        }
        if extra:
            entry["extra"] = self._to_jsonable(extra)
        self._llm_trace.append(entry)

    def _update_retrievers(self):
        """Keep retriever indexes in sync with the page store."""
        current_page_count = len(self.page_store.load())

        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"Page count changed ({self._last_page_count} -> {current_page_count}); updating retriever indexes...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        self._last_page_count = current_page_count

    # ---- Internal ----
    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                truncated = abstract[:300] + "..." if len(abstract) > 300 else abstract
                memory_context_lines.append(f"Page {i}: {truncated}")
            memory_context = "\n".join(memory_context_lines)

        available_tools = self._available_retrieval_tools()
        available_tools_context = "\n".join(
            f'- "{tool}"' for tool in available_tools
        ) or "No retrieval tools are currently available."
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(
            request=request,
            memory=memory_context,
            available_tools=available_tools_context,
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            raw_page_index = data.get("page_index", [])
            safe_page_index = [idx for idx in raw_page_index if isinstance(idx, int)][:5]
            parsed_plan = {
                "info_needs": data.get("info_needs", []),
                "tools": [tool for tool in data.get("tools", []) if tool in available_tools],
                "keyword_collection": data.get("keyword_collection", []),
                "vector_queries": data.get("vector_queries", []),
                "page_index": safe_page_index,
            }
            if (
                parsed_plan["keyword_collection"]
                and "keyword" in available_tools
                and "keyword" not in parsed_plan["tools"]
            ):
                parsed_plan["tools"].append("keyword")
            if safe_page_index and "page_index" in available_tools and "page_index" not in parsed_plan["tools"]:
                parsed_plan["tools"].append("page_index")
            if not parsed_plan["vector_queries"] and "vector" in available_tools:
                parsed_plan["vector_queries"] = [request]
            if parsed_plan["vector_queries"] and "vector" in available_tools and "vector" not in parsed_plan["tools"]:
                parsed_plan["tools"].append("vector")
            self._record_llm_call(
                phase="planning",
                prompt=prompt,
                schema=PLANNING_SCHEMA,
                response=response,
                parsed=parsed_plan,
            )
            return SearchPlan(
                **parsed_plan,
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            self._record_llm_call(
                phase="planning",
                prompt=prompt,
                schema=PLANNING_SCHEMA,
                error=str(e),
            )
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )

    def _available_retrieval_tools(self) -> List[str]:
        # Keyword search has a built-in page-store substring fallback, so expose
        # it even when the BM25 retriever is disabled or unavailable.
        tools: List[str] = ["keyword"]
        if "vector" in self.retrievers:
            tools.append("vector")
        if "page_index" in self.retrievers:
            tools.append("page_index")
        return tools
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect all hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    keyword_results = self._search_by_keyword(plan.keyword_collection, top_k=10)
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=10)
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        if not all_hits:
            return result
        
        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)
        
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        return self._integrate(sorted_hits, result, question)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration — collect raw hits as text.
        """
        all_hits: List[Hit] = []

        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    keyword_results = self._search_by_keyword(plan.keyword_collection, top_k=10)
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=10)
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        if not all_hits:
            return result
        
        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        for i, hit in enumerate(sorted_hits[:self.MAX_INTEGRATE_HITS], 1):
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            snippet = hit.snippet[:self.MAX_SNIPPET_CHARS]
            evidence_text.append(f"{i}. {source_info} {snippet}")
            
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    MAX_INTEGRATE_HITS = 10
    MAX_SNIPPET_CHARS = 40000

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits[:self.MAX_INTEGRATE_HITS], 1):
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            snippet = hit.snippet[:self.MAX_SNIPPET_CHARS]
            evidence_text.append(f"{i}. {source_info} {snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "No search results"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            
            llm_sources = data.get("sources", sources)
            if llm_sources:
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            parsed_result = {
                "content": data.get("content", ""),
                "sources": sources,
            }
            self._record_llm_call(
                phase="integration",
                prompt=prompt,
                schema=INTEGRATE_SCHEMA,
                response=response,
                parsed=parsed_result,
                extra={"hit_count": len(hits), "question": question},
            )
            return Result(
                content=parsed_result["content"],
                sources=parsed_result["sources"],
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            self._record_llm_call(
                phase="integration",
                prompt=prompt,
                schema=INTEGRATE_SCHEMA,
                error=str(e),
                extra={"hit_count": len(hits), "question": question},
            )
            return result

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        check_prompt = ""
        generate_prompt = ""
        try:
            system_prompt = self.system_prompts.get("reflection")
            
            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            print(f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)")
            
            # Step 1: Check for completeness of information
            template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            self._record_llm_call(
                phase="reflection_info_check",
                prompt=check_prompt,
                schema=INFO_CHECK_SCHEMA,
                response=check_response,
                parsed=check_data,
                extra={"request": request},
            )
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate a list of new requests
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])
            self._record_llm_call(
                phase="reflection_generate_requests",
                prompt=generate_prompt,
                schema=GENERATE_REQUESTS_SCHEMA,
                response=generate_response,
                parsed=generate_data,
                extra={"request": request},
            )
            
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            self._record_llm_call(
                phase="reflection",
                prompt=generate_prompt or check_prompt,
                error=str(e),
                extra={"request": request},
            )
            return ReflectionDecision(enough=False, new_request=None)