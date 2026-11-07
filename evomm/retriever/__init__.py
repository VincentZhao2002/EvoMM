# -*- coding: utf-8 -*-
"""
Retriever Module

This module contains all retrieval implementations for the EvoMM framework.
Retrievers provide different search strategies for finding relevant information.

Available Retrievers:
- DenseRetriever: Semantic search using dense vector embeddings
- BM25Retriever: Keyword-based search using BM25 algorithm
- IndexRetriever: Direct page access by index
"""

from __future__ import annotations

from .base import AbsRetriever
from .index_retriever import IndexRetriever

__all__ = ["AbsRetriever", "IndexRetriever", "BM25Retriever", "DenseRetriever"]


def __getattr__(name: str):
    """Lazily import optional retrievers to avoid hard dependency at import time."""
    if name == "BM25Retriever":
        from .bm25 import BM25Retriever

        return BM25Retriever
    if name == "DenseRetriever":
        from .dense_retriever import DenseRetriever

        return DenseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
