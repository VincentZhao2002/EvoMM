"""
Schemas Module

This module exposes all core data models and protocol definitions for the EvoMM (General-Agentic-Memory) framework.
It organizes memory, page, search, tool, and result schemas for unified import and type safety across the system.
"""
from .memory import MemoryState, MemoryUpdate, MemoryStore, InMemoryMemoryStore
from .page import Page, PageStore, InMemoryPageStore
from .ttl_memory import TTLMemoryStore, TTLMemoryState, TTLMemoryEntry
from .ttl_page import TTLPageStore
from .search import SearchPlan, Retriever, Hit
from .tools import ToolResult, Tool, ToolRegistry
from .result import Result, EnoughDecision, ReflectionDecision, ResearchOutput, GenerateRequests
from .experience import Experience, ExperienceStore, EXPERIENCE_EXTRACT_SCHEMA

# =============================
# Model rebuilding for forward references
# =============================
# Explicit model rebuild so forward refs (e.g. 'Page') resolve under concurrency.
# Especially important in multi-threaded use.
MemoryUpdate.model_rebuild()
ResearchOutput.model_rebuild()

# JSON Schema constants for LLM and system validation
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = EnoughDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()

__all__ = [
    "MemoryState", "MemoryUpdate", "MemoryStore", "InMemoryMemoryStore",
    "Page", "PageStore", "InMemoryPageStore",
    "TTLMemoryStore", "TTLMemoryState", "TTLMemoryEntry",
    "TTLPageStore",
    "SearchPlan", "Retriever", "Hit",
    "ToolResult", "Tool", "ToolRegistry",
    "Result", "EnoughDecision", "ReflectionDecision", "ResearchOutput", "GenerateRequests",
    "Experience", "ExperienceStore", "EXPERIENCE_EXTRACT_SCHEMA",
    "PLANNING_SCHEMA", "INTEGRATE_SCHEMA", "INFO_CHECK_SCHEMA", "GENERATE_REQUESTS_SCHEMA",
]
