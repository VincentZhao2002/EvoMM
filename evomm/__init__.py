# -*- coding: utf-8 -*-
"""Top-level package entrypoint for EvoMM.

This module intentionally avoids eager imports to keep import side effects minimal.
Symbols are loaded lazily on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

__version__ = "0.1.0"

# name -> (module_path, attr_name)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core agents
    "MemoryAgent": ("evomm.agents", "MemoryAgent"),
    "ResearchAgent": ("evomm.agents", "ResearchAgent"),
    "SelfEvolvingResearchAgent": ("evomm.agents", "SelfEvolvingResearchAgent"),

    # Generators
    "AbsGenerator": ("evomm.generator", "AbsGenerator"),
    "OpenAIGenerator": ("evomm.generator", "OpenAIGenerator"),
    "VLLMGenerator": ("evomm.generator", "VLLMGenerator"),

    # Retrievers
    "AbsRetriever": ("evomm.retriever", "AbsRetriever"),
    "IndexRetriever": ("evomm.retriever", "IndexRetriever"),
    "BM25Retriever": ("evomm.retriever", "BM25Retriever"),
    "DenseRetriever": ("evomm.retriever", "DenseRetriever"),

    # Configurations
    "OpenAIGeneratorConfig": ("evomm.config", "OpenAIGeneratorConfig"),
    "VLLMGeneratorConfig": ("evomm.config", "VLLMGeneratorConfig"),
    "DenseRetrieverConfig": ("evomm.config", "DenseRetrieverConfig"),
    "BM25RetrieverConfig": ("evomm.config", "BM25RetrieverConfig"),
    "IndexRetrieverConfig": ("evomm.config", "IndexRetrieverConfig"),

    # Schemas
    "MemoryState": ("evomm.schemas", "MemoryState"),
    "Page": ("evomm.schemas", "Page"),
    "MemoryUpdate": ("evomm.schemas", "MemoryUpdate"),
    "SearchPlan": ("evomm.schemas", "SearchPlan"),
    "Hit": ("evomm.schemas", "Hit"),
    "Result": ("evomm.schemas", "Result"),
    "EnoughDecision": ("evomm.schemas", "EnoughDecision"),
    "ReflectionDecision": ("evomm.schemas", "ReflectionDecision"),
    "ResearchOutput": ("evomm.schemas", "ResearchOutput"),
    "InMemoryMemoryStore": ("evomm.schemas", "InMemoryMemoryStore"),
    "InMemoryPageStore": ("evomm.schemas", "InMemoryPageStore"),

    # Experience / self-evolution
    "Experience": ("evomm.schemas", "Experience"),
    "ExperienceStore": ("evomm.schemas", "ExperienceStore"),
}

__all__ = ["__version__", *_EXPORTS.keys()]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
