#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMMemory-style retriever adapter.

Uses Mem-Gallery's:
- LinearStorage
- MultiModalRetrieval

to provide a retriever interface compatible with EvoMM ResearchAgent,
AND a recall_for_qa() method that returns memories in the same format
as MMMemory for direct QA answer generation.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import threading
import types
from typing import Any, Dict, List, Optional, Tuple

import torch


def _pick_first_existing(candidates: List[str], fallback: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return fallback


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_MEMGALLERY_BENCHMARK = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "..", "Mem-Gallery", "benchmark")
)
BENCHMARK_ROOT = os.environ.get(
    "MEMGALLERY_BENCHMARK_ROOT",
    _pick_first_existing(
        [
            _DEFAULT_MEMGALLERY_BENCHMARK,
            os.path.expanduser("~/Mem-Gallery/benchmark"),
            "./Mem-Gallery/benchmark",
        ],
        _DEFAULT_MEMGALLERY_BENCHMARK,
    ),
)
# Mem-Gallery 的 memengine 必须与 PyPI 同名包区分：插入到 path 最前，否则会 import 到
# site-packages 里不完整的 memengine，导致 MultiModalRetrieval 缺失。
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)


def _install_memgallery_memengine_package() -> None:
    """注册 Mem-Gallery 的 memengine，且不执行 memengine/__init__.py。

    原因：1) PyPI 的同名包会抢先被 import；2) Mem-Gallery 的 __init__.py 会 import
    UniEvoMemory，依赖外部 Uni-Evo-Memory 的 src，本环境常不存在。
    用仅带 __path__ 的命名空间包即可正常加载 function/utils/config 子模块。
    """
    mg = os.path.join(BENCHMARK_ROOT, "memengine")
    if not os.path.isdir(mg):
        return
    pkg = types.ModuleType("memengine")
    pkg.__path__ = [mg]  # type: ignore[attr-defined]
    sys.modules["memengine"] = pkg


_install_memgallery_memengine_package()

from memengine.utils.Storage import LinearStorage
from memengine.function.MultiModalRetrieval import MultiModalRetrieval
from memengine.config.Config import MemoryConfig
from default_config.DefaultMMMemoryConfig import DEFAULT_MMMEMORY
from evomm.schemas import Hit


def _strip_image_markers(text: str) -> str:
    return re.sub(r"\[Image: .*?\]", "", text).strip()


def _extract_first_image_path(text: str) -> Optional[str]:
    m = re.search(r"\[Image: (.*?)\]", text)
    return m.group(1).strip() if m else None


_HEADER_RE = re.compile(r"^===\s.*===\s*$", re.MULTILINE)
_MARKER_RE = re.compile(r"^\[(Image|ImageID|ImageCaption|QuestionImageCaption):.*\]\s*$", re.MULTILINE)


def _clean_page_text(raw_text: str) -> str:
    """Strip session/turn headers and image markers, keep dialog text only."""
    text = _HEADER_RE.sub("", raw_text)
    text = _MARKER_RE.sub("", text)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


class MMMemoryStyleRetriever:
    """Retriever with MMMemory's MultiModalRetrieval mechanism.

    Build() 会复用一张全局 **页面级 embedding cache**（按 ``(image_path, text)``
    去重）。在 DocQA 这种"一个文档对应多条问题"的场景下，每一页只会被 GME
    encoder 编码一次，后续问题的 ``build`` 退化为纯张量拼接 + 字典 append，
    基本不再碰 GPU。需要完全重置时调用 ``clear_emb_cache()``。
    """

    def __init__(self, top_k: int = 10, use_emb_cache: bool = True) -> None:
        self.top_k = top_k
        cfg = MemoryConfig(DEFAULT_MMMEMORY)
        self.storage = LinearStorage(cfg.args.storage)
        self.mm_retrieval = MultiModalRetrieval(cfg.args.multimodal_retrieval)
        self._pages: List[Any] = []
        self.last_retrieved_ids: List[str] = []
        self._mid_to_page_idx: Dict[int, int] = {}

        # ---- page-level embedding cache ---------------------------------
        # key: (image_path, clean_text) -> normalized embedding tensor
        # 线程安全由调用方（外层 mm_retriever_lock）保证，这里的 lock 只是
        # 防止日后有人单独调用时出问题。
        self._use_emb_cache = use_emb_cache
        self._emb_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._emb_cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def reset(self) -> None:
        self.storage.reset()
        self.mm_retrieval.reset()
        self._pages = []
        self.last_retrieved_ids = []
        self._mid_to_page_idx = {}

    def clear_emb_cache(self) -> None:
        """Free page-embedding cache (e.g. when switching to a different run)."""
        with self._emb_cache_lock:
            self._emb_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def cache_stats(self) -> Dict[str, int]:
        with self._emb_cache_lock:
            return {
                "size": len(self._emb_cache),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
            }

    # ---------------------------------------------------------------------
    # Internal helpers used by build()
    # ---------------------------------------------------------------------

    def _embed_observation(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Return a (normalized) embedding for ``obs``, using the cache.

        Cache key is ``(image_path, clean_text)``. ``clean_text`` for pages of
        the same document is identical (we strip session headers upstream),
        so the same page hits cache across every question on that doc.
        """
        img = obs.get("image") or {}
        img_path = img.get("path", "") if isinstance(img, dict) else ""
        text = obs.get("text", "") or ""
        key = (img_path, text)

        if self._use_emb_cache:
            with self._emb_cache_lock:
                cached = self._emb_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        emb = self.mm_retrieval.encoder(obs, return_type="tensor")
        if getattr(self.mm_retrieval.config, "mode", "cosine") == "cosine":
            emb = torch.nn.functional.normalize(emb, dim=-1)

        if self._use_emb_cache:
            with self._emb_cache_lock:
                # Another thread may have populated it in the meantime.
                self._emb_cache.setdefault(key, emb)
            self._cache_misses += 1
        return emb

    def _insert_observation(self, obs: Dict[str, Any]) -> None:
        """Append ``obs`` into storage/tensorstore without re-running encoder."""
        emb = self._embed_observation(obs)
        ts = self.mm_retrieval.tensorstore
        self.mm_retrieval.tensorstore = emb if ts is None else torch.cat([ts, emb], dim=0)
        self.mm_retrieval.memory_metadata.append(
            {
                "has_text": bool(obs.get("text")),
                "has_image": bool(obs.get("image")),
            }
        )
        self.storage.add(obs)

    def build(self, page_store: Any) -> None:
        self.reset()
        self._pages = page_store.load()
        mid_counter = 0

        for page_idx, p in enumerate(self._pages):
            meta = p.meta or {}
            images = meta.get("images", []) or []
            image_ids = meta.get("image_ids", []) or []
            image_caps = meta.get("image_captions", []) or []

            clean_text = _clean_page_text(p.content)

            if len(images) <= 1:
                observation: Dict[str, Any] = {
                    "text": clean_text,
                    "image": None,
                    "timestamp": meta.get("date", ""),
                    "dialogue_id": meta.get("turn_id", ""),
                }
                if images:
                    observation["image"] = {
                        "path": images[0],
                        "caption": image_caps[0] if image_caps else "",
                        "img_id": image_ids[0] if image_ids else "",
                    }
                self._insert_observation(observation)
                self._mid_to_page_idx[mid_counter] = page_idx
                mid_counter += 1
            else:
                for img_i, img_path in enumerate(images):
                    img_id = image_ids[img_i] if img_i < len(image_ids) else ""
                    img_cap = image_caps[img_i] if img_i < len(image_caps) else ""
                    observation = {
                        "text": clean_text,
                        "image": {
                            "path": img_path,
                            "caption": img_cap,
                            "img_id": img_id,
                        },
                        "timestamp": meta.get("date", ""),
                        "dialogue_id": meta.get("turn_id", ""),
                    }
                    self._insert_observation(observation)
                    self._mid_to_page_idx[mid_counter] = page_idx
                    mid_counter += 1

    def update(self, page_store: Any) -> None:
        self.build(page_store)

    def snapshot_state(self) -> Dict[str, Any]:
        """Create a deep-copied retriever state for reuse."""
        return {
            "storage": copy.deepcopy(self.storage),
            "mm_retrieval": copy.deepcopy(self.mm_retrieval),
            "pages": copy.deepcopy(self._pages),
            "mid_to_page_idx": copy.deepcopy(self._mid_to_page_idx),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore retriever state from a snapshot."""
        self.storage = copy.deepcopy(state["storage"])
        self.mm_retrieval = copy.deepcopy(state["mm_retrieval"])
        self._pages = copy.deepcopy(state["pages"])
        self._mid_to_page_idx = copy.deepcopy(state["mid_to_page_idx"])
        self.last_retrieved_ids = []

    def build_memory_dict_list_from_page_indices(
        self,
        page_store: Any,
        page_indices: List[int],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        从 Research Agent 的 resolved_page_index 构建 MMMemory 格式的 memory_dict_list，
        用于与 recall_for_qa 结果融合，确保 Research 找到的关键页面进入 QA 上下文。

        Returns
        -------
        memory_dict_list : list[dict]
            格式与 recall_for_qa 一致: {text, image, timestamp, dialogue_id}
        retrieved_ids : list[str]
            dialogue_ids (turn_id)
        """
        pages = page_store.load()
        memory_list: List[Dict[str, Any]] = []
        retrieved_ids: List[str] = []
        mem_index = 0

        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(pages):
                continue
            p = pages[page_idx]
            meta = p.meta or {}
            images = meta.get("images", []) or []
            image_ids = meta.get("image_ids", []) or []
            image_caps = meta.get("image_captions", []) or []
            timestamp = meta.get("date", "")
            dialogue_id = meta.get("turn_id", "")

            clean_text = _clean_page_text(p.content)

            if len(images) <= 1:
                img_info = None
                if images:
                    img_info = {
                        "path": images[0],
                        "caption": image_caps[0] if image_caps else "",
                        "img_id": image_ids[0] if image_ids else "",
                    }
                formatted_text = f"[Memory {mem_index}] timestamp: {timestamp}\n{clean_text}"
                mem_index += 1
                memory_list.append({
                    "text": formatted_text,
                    "image": img_info,
                    "timestamp": timestamp,
                    "dialogue_id": dialogue_id,
                })
                if dialogue_id:
                    retrieved_ids.append(str(dialogue_id))
            else:
                for img_i, img_path in enumerate(images):
                    img_id = image_ids[img_i] if img_i < len(image_ids) else ""
                    img_cap = image_caps[img_i] if img_i < len(image_caps) else ""
                    formatted_text = f"[Memory {mem_index}] timestamp: {timestamp}\n{clean_text}"
                    mem_index += 1
                    memory_list.append({
                        "text": formatted_text,
                        "image": {
                            "path": img_path,
                            "caption": img_cap,
                            "img_id": img_id,
                        },
                        "timestamp": timestamp,
                        "dialogue_id": dialogue_id,
                    })
                    if dialogue_id and img_i == 0:
                        retrieved_ids.append(str(dialogue_id))

        seen: set = set()
        dedup_ids: List[str] = []
        for rid in retrieved_ids:
            if rid not in seen:
                seen.add(rid)
                dedup_ids.append(rid)

        return memory_list, dedup_ids

    # ------------------------------------------------------------------
    # EvoMM ResearchAgent compatible interface
    # ------------------------------------------------------------------
    def search(self, query_list: List[str], top_k: int = 5) -> List[List[Hit]]:
        self.last_retrieved_ids = []
        k = top_k if top_k is not None else self.top_k
        all_results: List[List[Hit]] = []

        for query in query_list:
            q_img = _extract_first_image_path(query)
            q_text = _strip_image_markers(query)
            qobj = {"text": q_text, "image": {"path": q_img} if q_img else None}

            mids = self.mm_retrieval(qobj, topk=k)
            hits: List[Hit] = []
            seen_page_idx: set = set()
            for mid in mids:
                idx = int(mid)
                mem = self.storage.get_memory_element_by_mid(idx)
                dialogue_id = mem.get("dialogue_id", "")
                if dialogue_id:
                    self.last_retrieved_ids.append(str(dialogue_id))

                page_idx = self._mid_to_page_idx.get(idx, idx)
                if page_idx in seen_page_idx:
                    continue
                seen_page_idx.add(page_idx)

                snippet = mem.get("text", "")
                image_info = mem.get("image")
                if isinstance(image_info, dict):
                    img_id = image_info.get("img_id", "")
                    cap = image_info.get("caption", "")
                    if img_id:
                        snippet += f"\n[ImageID: {img_id}]"
                    if cap:
                        snippet += f"\n[ImageCaption: {cap}]"

                hits.append(
                    Hit(
                        page_id=str(page_idx),
                        snippet=snippet,
                        source="mmmemory_retrieval",
                        meta={"dialogue_id": dialogue_id},
                    )
                )
            all_results.append(hits)

        return all_results

    # ------------------------------------------------------------------
    # MMMemory-compatible recall for direct QA (matching run_bench.py)
    # ------------------------------------------------------------------
    def recall_for_qa(
        self,
        question: str,
        question_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve memories in the same dict format as MMMemory's recall.

        Returns
        -------
        memory_dict_list : list[dict]
            Each dict: {text, image, timestamp, dialogue_id}
            ``text`` is formatted with ``[Memory i] timestamp: …`` prefix
            (same as MultiModalUtilization).
        retrieved_ids : list[str]
            De-duplicated dialogue_ids of retrieved memories.
        """
        k = top_k if top_k is not None else self.top_k
        qobj = {
            "text": _strip_image_markers(question),
            "image": {"path": question_image_path} if question_image_path else None,
        }
        mids = self.mm_retrieval(qobj, topk=k)

        memory_list: List[Dict[str, Any]] = []
        retrieved_ids: List[str] = []
        seen_page_idx: set = set()
        mem_index = 0

        for mid in mids:
            idx = int(mid)
            mem = self.storage.get_memory_element_by_mid(idx)
            dialogue_id = mem.get("dialogue_id", "")
            if dialogue_id:
                retrieved_ids.append(str(dialogue_id))

            page_idx = self._mid_to_page_idx.get(idx, idx)
            if page_idx in seen_page_idx:
                continue
            seen_page_idx.add(page_idx)

            timestamp = mem.get("timestamp", "")
            raw_text = mem.get("text", "")
            formatted_text = f"[Memory {mem_index}] timestamp: {timestamp}\n{raw_text}"
            mem_index += 1

            memory_list.append({
                "text": formatted_text,
                "image": mem.get("image"),
                "timestamp": timestamp,
                "dialogue_id": dialogue_id,
            })

        seen: set = set()
        dedup_ids: List[str] = []
        for rid in retrieved_ids:
            if rid not in seen:
                seen.add(rid)
                dedup_ids.append(rid)

        return memory_list, dedup_ids
