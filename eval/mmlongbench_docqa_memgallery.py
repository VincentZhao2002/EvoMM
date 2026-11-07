#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mem-Gallery memory evaluation on MMLongBench DocQA.

This evaluator keeps the same input/output contract as
``mmlongbench_docqa_graph.py``: it reads one documentQA jsonl file, writes a
resumable jsonl cache, a final json bundle, and a ``.score`` file.  The only
difference is page selection: pages are stored into a Mem-Gallery memory
(``GAMemory``, ``MMMemory``, ``WorldMMMemory``, or ``HippoRAGMemory``), recalled with the question, then passed to
the same MMLongBench DocQA VLM prompt for answer generation and scoring.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import openai
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mmlongbench_docqa_graph import (  # type: ignore
    _load_mmlb_utils,
    load_resume,
    load_samples,
    parse_output,
    score_sample,
)


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
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)


def _install_memgallery_memengine_package() -> None:
    """Prefer Mem-Gallery's local memengine over any PyPI package."""
    mg = os.path.join(BENCHMARK_ROOT, "memengine")
    if not os.path.isdir(mg):
        return
    pkg = types.ModuleType("memengine")
    pkg.__path__ = [mg]  # type: ignore[attr-defined]
    sys.modules["memengine"] = pkg


_install_memgallery_memengine_package()

from default_config.DefaultMemoryConfig import DEFAULT_GAMEMORY  # noqa: E402
from default_config.DefaultMemoryConfig import DEFAULT_HIPPORAGMEMORY  # noqa: E402
from default_config.DefaultMMMemoryConfig import DEFAULT_MMMEMORY  # noqa: E402
from default_config.DefaultWorldMMMemoryConfig import DEFAULT_WORLDMMMEMORY  # noqa: E402


SUPPORTED_MEMORY_NAMES = ("GAMemory", "MMMemory", "WorldMMMemory", "HippoRAGMemory")
_REUSABLE_MEMORY_CACHE: Dict[Tuple[Any, ...], Tuple[Any, threading.Lock]] = {}
_REUSABLE_MEMORY_CACHE_LOCK = threading.Lock()

USER_TEMPLATE = (
    "You are given a document with text and images, and a question. "
    "Answer the question as concisely as you can, using a single phrase or "
    "sentence if possible. If the question cannot be answered based on the "
    "information in the article, write 'Not answerable.' Write your answer "
    "in the following format:\nAnswer: [answer]\n\n"
    "{context}\n\n"
    "{retrieved_notes}"
    "Question: {question}"
)
ITEM_TEMPLATE = "Document {doc_id:.15}: <image>"
SYSTEM_TEMPLATE = "Answer:"
RETRIEVED_NOTES_TEMPLATE = (
    "Retrieved page text hints from the memory system:\n"
    "{notes}\n\n"
)


def _load_memory_class(memory_name: str) -> Any:
    # Import lazily so --no-retrieval / empty smoke tests do not require the
    # local CUDA/PyTorch stack until a Mem-Gallery memory is actually built.
    if memory_name == "GAMemory":
        from memengine.memory.GAMemory import GAMemory  # type: ignore

        return GAMemory
    if memory_name == "WorldMMMemory":
        from memengine.memory.WorldMMMemory import WorldMMMemory  # type: ignore

        return WorldMMMemory
    if memory_name == "MMMemory":
        from memengine.memory.MMMemory import MMMemory  # type: ignore

        return MMMemory
    if memory_name == "HippoRAGMemory":
        from memengine.memory.HippoRAGMemory import HippoRAGMemory  # type: ignore

        return HippoRAGMemory
    raise ValueError(f"Unsupported memory_name: {memory_name}")


def _load_memory_config_class() -> Any:
    from memengine.config.Config import MemoryConfig  # type: ignore

    return MemoryConfig


def override_apillm_config(config: Any, model_name: str, api_key: str, api_base: str) -> int:
    """Recursively align nested Mem-Gallery APILLM configs with runtime args."""
    replaced = 0

    def _walk(node: Any) -> None:
        nonlocal replaced
        if isinstance(node, dict):
            if node.get("method") == "APILLM":
                node["name"] = model_name
                node["api_key"] = api_key
                node["base_url"] = api_base
                replaced += 1
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(config)
    return replaced


def configure_memory_config(
    memory_name: str,
    top_k: int,
    model_name: str,
    api_key: str,
    api_base: str,
    disable_gam_reflection: bool,
) -> Dict[str, Any]:
    if memory_name == "GAMemory":
        cfg = copy.deepcopy(DEFAULT_GAMEMORY)
        cfg["recall"]["topk"] = top_k
        cfg["recall"]["text_retrieval"]["topk"] = top_k
        cfg["recall"]["importance_retrieval"]["topk"] = top_k
        if disable_gam_reflection:
            # Store still computes importance, but this prevents reflection LLM
            # calls from firing while ingesting every page of a long document.
            cfg["reflect"]["reflector"]["threshold"] = 1e9
    elif memory_name == "WorldMMMemory":
        cfg = copy.deepcopy(DEFAULT_WORLDMMMEMORY)
        # DocQA pages are image candidates.  The jsonl files usually do not
        # contain OCR text, so WorldMM's episodic/semantic memories mostly see
        # placeholder strings ("Document ... page N").  Keep a small text quota
        # for fallback, but retrieve enough visual candidates to drive page
        # selection.
        cfg["worldmm_config"]["episodic_top_k"] = min(2, top_k)
        cfg["worldmm_config"]["semantic_top_k"] = 0
        cfg["worldmm_config"]["visual_top_k"] = max(top_k, cfg["worldmm_config"].get("visual_top_k", top_k))
        cfg["worldmm_config"]["enable_semantic_extraction"] = False
    elif memory_name == "MMMemory":
        cfg = copy.deepcopy(DEFAULT_MMMEMORY)
        cfg["multimodal_retrieval"]["topk"] = top_k
        cfg["recall"]["multimodal_retrieval"]["topk"] = top_k
    elif memory_name == "HippoRAGMemory":
        cfg = copy.deepcopy(DEFAULT_HIPPORAGMEMORY)
        cfg["recall"]["top_k"] = top_k
        cfg["hipporag"]["retrieval_top_k"] = top_k
        cfg["hipporag"]["llm_name"] = os.environ.get("HIPPORAG_LLM_NAME", model_name)
        cfg["hipporag"]["llm_base_url"] = os.environ.get("HIPPORAG_LLM_BASE_URL", api_base)
    else:
        raise ValueError(f"Unsupported memory_name: {memory_name}")

    override_apillm_config(cfg, model_name=model_name, api_key=api_key, api_base=api_base)
    return cfg


def _page_text(sample: Dict[str, Any], page_idx: int) -> Optional[str]:
    texts = sample.get("page_text_list") or []
    if page_idx < len(texts) and texts[page_idx]:
        return str(texts[page_idx])
    return None


def _has_page_text(sample: Dict[str, Any]) -> bool:
    texts = sample.get("page_text_list") or []
    return len(texts) >= len(sample.get("page_list", [])) and all(
        str(t).strip() for t in texts
    )


def _build_observation(
    sample: Dict[str, Any],
    page_idx: int,
    image_root: str,
    include_image: bool,
    require_page_text: bool,
) -> Dict[str, Any]:
    rel = sample["page_list"][page_idx]
    abs_path = os.path.join(image_root, rel)
    page_text = _page_text(sample, page_idx)
    if page_text is None:
        if require_page_text:
            raise ValueError(
                f"{args.memory_name} requires page_text_list/OCR text; use text_mmlongdoc_* "
                "or provide OCR/caption cache instead of image-only jsonl."
            )
        page_text = f"Document {sample['doc_name']} page {page_idx}."
    obs: Dict[str, Any] = {
        "text": page_text,
        "timestamp": page_idx,
        "time": page_idx,
        "dialogue_id": str(page_idx),
    }
    if include_image:
        obs["image"] = {
            "path": abs_path,
            "caption": "",
            "img_id": f"PAGE{page_idx}",
        }
    else:
        obs["image"] = None
    return obs


def build_messages_with_memory_text(
    sample: Dict[str, Any],
    page_indices: List[int],
    image_root: str,
    memory_context: Any,
    max_note_chars: int,
) -> List[Dict[str, Any]]:
    doc = sample["doc_name"]
    context_block = "\n\n".join(ITEM_TEMPLATE.format(doc_id=doc) for _ in page_indices)
    question = (
        f"Based on Document {doc[:15]}, answer the following question. "
        + sample["question"]
    )
    notes = _format_retrieved_notes(memory_context, max_note_chars)
    retrieved_notes = RETRIEVED_NOTES_TEMPLATE.format(notes=notes) if notes else ""
    prompt_text = (
        USER_TEMPLATE.format(
            context=context_block,
            retrieved_notes=retrieved_notes,
            question=question,
        )
        + "\n"
        + SYSTEM_TEMPLATE
    )

    parts = prompt_text.split("<image>")
    if len(parts) - 1 != len(page_indices):
        raise RuntimeError(
            f"prompt split mismatch: {len(parts) - 1} slots vs {len(page_indices)} pages"
        )
    content: List[Dict[str, Any]] = []
    for i, segment in enumerate(parts):
        if segment:
            content.append({"type": "text", "text": segment})
        if i < len(page_indices):
            rel = sample["page_list"][page_indices[i]]
            abs_path = os.path.join(image_root, rel)
            content.append({"type": "image_url", "image_url": {"url": f"file://{abs_path}"}})
    return [{"role": "user", "content": content}]


def _format_retrieved_notes(memory_context: Any, max_note_chars: int) -> str:
    if max_note_chars <= 0 or not memory_context:
        return ""

    chunks: List[str] = []
    if isinstance(memory_context, str):
        chunks = [memory_context]
    elif isinstance(memory_context, list):
        for item in memory_context:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                dialogue_id = item.get("dialogue_id", "")
                if text:
                    prefix = f"Page {dialogue_id}: " if dialogue_id not in (None, "") else ""
                    chunks.append(prefix + text)
            elif item is not None:
                chunks.append(str(item))
    else:
        chunks = [str(memory_context)]

    notes = "\n\n".join(chunk for chunk in chunks if chunk.strip()).strip()
    if len(notes) > max_note_chars:
        notes = notes[:max_note_chars].rstrip() + "\n[truncated]"
    return notes


def _ids_to_page_indices(ids: List[Any], num_pages: int) -> List[int]:
    out: List[int] = []
    for value in ids:
        text = str(value)
        match = re.search(r"(\d+)$", text)
        if not match:
            continue
        idx = int(match.group(1))
        if 0 <= idx < num_pages and idx not in out:
            out.append(idx)
    return out


def _append_unique_page_indices(out: List[int], ids: List[Any], num_pages: int) -> None:
    for idx in _ids_to_page_indices(ids, num_pages):
        if idx not in out:
            out.append(idx)


def _worldmm_context_to_page_indices(memory_context: Any, num_pages: int) -> List[int]:
    """Prefer visual memories when mapping WorldMM recall results to pages."""
    if not isinstance(memory_context, list):
        return []

    visual_ids: List[Any] = []
    other_ids: List[Any] = []
    for item in memory_context:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        image = item.get("image")
        item_id = item.get("dialogue_id")
        is_visual = image is not None or "[Visual Memory]" in text
        if is_visual:
            if isinstance(image, dict):
                img_id = image.get("img_id")
                if img_id not in (None, ""):
                    visual_ids.append(img_id)
            visual_ids.append(item_id)
        elif item_id not in (None, ""):
            other_ids.append(item_id)

    selected: List[int] = []
    _append_unique_page_indices(selected, visual_ids, num_pages)
    _append_unique_page_indices(selected, other_ids, num_pages)
    return selected


def _get_reusable_memory(
    args: argparse.Namespace,
    memory_config: Dict[str, Any],
) -> Tuple[Any, threading.Lock]:
    """Return one reusable memory instance per process/config.

    The memory owns heavy encoders such as GME. Reusing it avoids loading
    checkpoint shards for every sample.
    """
    key = (
        args.memory_name,
        id(memory_config),
        args.top_k,
        args.model,
        args.api_base,
    )
    with _REUSABLE_MEMORY_CACHE_LOCK:
        cached = _REUSABLE_MEMORY_CACHE.get(key)
        if cached is None:
            memory_cls = _load_memory_class(args.memory_name)
            memory_config_cls = _load_memory_config_class()
            memory = memory_cls(memory_config_cls(copy.deepcopy(memory_config)))
            cached = (memory, threading.Lock())
            _REUSABLE_MEMORY_CACHE[key] = cached
            print(f"Initialized reusable {args.memory_name} memory for this process.")
        return cached


def select_pages_with_memory(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    memory_config: Dict[str, Any],
) -> Tuple[List[int], List[int], Any]:
    num_pages = len(sample["page_list"])
    if args.no_retrieval or num_pages <= args.top_k:
        if args.memory_name in ("GAMemory", "HippoRAGMemory") and args.require_page_text and not _has_page_text(sample):
            raise ValueError(
                f"{args.memory_name} requires page_text_list/OCR text even when all pages fit top-k."
            )
        full_context = []
        if args.memory_name in ("GAMemory", "HippoRAGMemory") and _has_page_text(sample):
            full_context = [
                {
                    "text": f"[Memory {idx}] page {idx}\n{_page_text(sample, idx)}",
                    "image": None,
                    "timestamp": idx,
                    "dialogue_id": str(idx),
                }
                for idx in range(num_pages)
            ]
        return list(range(num_pages)), [], full_context

    include_image = args.memory_name in ("MMMemory", "WorldMMMemory")
    require_page_text = args.memory_name in ("GAMemory", "HippoRAGMemory") and args.require_page_text
    memory, memory_lock = _get_reusable_memory(args, memory_config)
    with memory_lock:
        memory.reset()
        for page_idx in range(num_pages):
            memory.store(
                _build_observation(
                    sample,
                    page_idx,
                    args.image_root,
                    include_image,
                    require_page_text=require_page_text,
                )
            )

        query: Any
        if include_image:
            query = {"text": sample["question"], "image": None}
        else:
            query = sample["question"]

        memory_context = memory.recall(query)
        retrieved_ids: List[Any] = []

        recall_op = getattr(memory, "recall_op", None)
        if recall_op is not None:
            retrieved_ids = list(getattr(recall_op, "last_retrieved_ids", []) or [])
        if not retrieved_ids and isinstance(memory_context, list):
            retrieved_ids = [
                item.get("dialogue_id")
                for item in memory_context
                if isinstance(item, dict) and item.get("dialogue_id") not in (None, "")
            ]
        memory_context = copy.deepcopy(memory_context)

    if args.memory_name == "WorldMMMemory":
        selected = _worldmm_context_to_page_indices(memory_context, num_pages)
        seeds = selected
    else:
        selected = _ids_to_page_indices(retrieved_ids, num_pages)
        seeds = _ids_to_page_indices(retrieved_ids, num_pages)
    if not selected:
        selected = list(range(min(num_pages, args.top_k)))
    selected = selected[: args.top_k]
    selected.sort()
    return selected, seeds, memory_context


def run_one_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    memory_config: Dict[str, Any],
    vllm_client: openai.OpenAI,
) -> Dict[str, Any]:
    picked, seeds, memory_context = select_pages_with_memory(sample, args, memory_config)
    messages = build_messages_with_memory_text(
        sample,
        picked,
        args.image_root,
        memory_context if args.include_memory_text else None,
        args.max_memory_text_chars,
    )

    t0 = time.time()
    output_text = ""
    error: Optional[str] = None
    for attempt in range(args.max_retries + 1):
        try:
            resp = vllm_client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            output_text = resp.choices[0].message.content or ""
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if attempt >= args.max_retries:
                break
            time.sleep(1.0 * (attempt + 1))
    latency = time.time() - t0

    parsed = parse_output(output_text) if output_text else ""
    mets, judge_result = (
        score_sample(parsed, sample, args._judge_client, args.llm_judge_model)
        if parsed and not error
        else ({"doc_qa": 0.0}, None)
    )

    return {
        "id": sample.get("id", ""),
        "doc_name": sample["doc_name"],
        "question": sample["question"],
        "answer": sample["answer"],
        "answer_format": sample.get("answer_format", "String"),
        "ans_page_list": sample.get("ans_page_list", []),
        "num_pages": len(sample["page_list"]),
        "memory_name": args.memory_name,
        "seed_page_indices": seeds,
        "selected_page_indices": picked,
        "memory_context": memory_context,
        "output": output_text,
        "parsed_output": parsed,
        "metric_result": mets,
        "judge_result": judge_result,
        "latency_s": round(latency, 3),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Mem-Gallery memories on MMLongBench DocQA."
    )
    parser.add_argument("--data-file", required=True, help="documentQA/*.jsonl path")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--mmlongbench-root",
        default=os.environ.get("MMLB_ROOT", "./MMLongBench/MMLongBench"),
        help="MMLongBench python root (for parse_output / calculate_metrics).",
    )
    parser.add_argument("--memory-name", choices=SUPPORTED_MEMORY_NAMES, required=True)
    parser.add_argument("--model", default="qwen3-vl-4b-instruct-199")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--require-page-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require page_text_list/OCR text. Defaults to true for textual memories.",
    )
    parser.add_argument(
        "--include-memory-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include retrieved memory text hints in the VLM prompt.",
    )
    parser.add_argument(
        "--max-memory-text-chars",
        type=int,
        default=6000,
        help="Maximum characters of retrieved text hints injected into the VLM prompt.",
    )
    parser.add_argument("--docqa-llm-judge", action="store_true")
    parser.add_argument("--llm-judge-key", default="")
    parser.add_argument("--llm-judge-endpoint", default="")
    parser.add_argument("--llm-judge-model", default="Doubao-Seed-1.8")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--disable-gam-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable GAMemory reflection during page ingestion.",
    )
    args = parser.parse_args()
    if args.require_page_text is None:
        args.require_page_text = args.memory_name in ("GAMemory", "HippoRAGMemory")
    if args.include_memory_text is None:
        args.include_memory_text = args.memory_name != "WorldMMMemory"

    os.makedirs(args.outdir, exist_ok=True)
    _load_mmlb_utils(args.mmlongbench_root)

    args._judge_client = None  # type: ignore[attr-defined]
    if args.docqa_llm_judge:
        if not (args.llm_judge_key and args.llm_judge_endpoint):
            raise SystemExit(
                "--docqa-llm-judge requires --llm-judge-key and --llm-judge-endpoint"
            )
        args._judge_client = openai.OpenAI(  # type: ignore[attr-defined]
            api_key=args.llm_judge_key,
            base_url=args.llm_judge_endpoint,
        )

    vllm_client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)
    total_cap = args.limit if args.limit is not None else args.max_samples
    samples = load_samples(args.data_file, total_cap)
    print(f"Loaded {len(samples)} samples from {args.data_file}")
    print(f"Using {args.memory_name} for page selection (top_k={args.top_k})")
    if args.memory_name in ("GAMemory", "HippoRAGMemory") and args.require_page_text:
        missing = sum(1 for sample in samples if not _has_page_text(sample))
        if missing:
            raise SystemExit(
                f"{args.memory_name} requires page_text_list/OCR text, but {missing}/{len(samples)} "
                f"samples in {args.data_file} do not have complete page_text_list. "
                "Use text_mmlongdoc_K*.jsonl or provide an OCR/caption cache."
            )

    memory_config = configure_memory_config(
        memory_name=args.memory_name,
        top_k=args.top_k,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        disable_gam_reflection=args.disable_gam_reflection,
    )

    memory_tag = args.memory_name.lower()
    file_tag = os.path.splitext(os.path.basename(args.data_file))[0]
    out_jsonl = os.path.join(args.outdir, f"{file_tag}_{memory_tag}.jsonl")
    out_json = os.path.join(args.outdir, f"{file_tag}_{memory_tag}.json")
    score_file = out_json + ".score"
    if args.no_resume and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    existing, done_keys = load_resume(out_jsonl)
    if existing:
        print(f"Resuming: {len(existing)} samples already in cache.")

    remaining = [
        s
        for s in samples
        if f"{s.get('id', '')}_{s.get('question', '')}" not in done_keys
    ]
    remaining.sort(key=lambda s: (s.get("doc_name", ""), s.get("id", "")))
    print(f"Remaining to process: {len(remaining)} with {args.num_workers} workers")

    results: List[Dict[str, Any]] = list(existing)
    f_out = open(out_jsonl, "a", encoding="utf-8")
    write_lock = threading.Lock()

    def _task(sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            item = run_one_sample(sample, args, memory_config, vllm_client)
        except Exception as exc:  # noqa: BLE001
            item = {
                "id": sample.get("id", ""),
                "doc_name": sample.get("doc_name", ""),
                "question": sample.get("question", ""),
                "memory_name": args.memory_name,
                "error": repr(exc),
                "metric_result": {"doc_qa": 0.0},
            }
        with write_lock:
            f_out.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            f_out.flush()
        return item

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_task, s) for s in remaining]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=file_tag):
                results.append(fut.result())
    finally:
        f_out.close()

    metrics_agg: Dict[str, List[float]] = defaultdict(list)
    for it in results:
        mr = it.get("metric_result") or {}
        for key, value in mr.items():
            if isinstance(value, (int, float)):
                metrics_agg[key].append(float(value))
    averaged = {key: 100.0 * sum(vals) / len(vals) for key, vals in metrics_agg.items() if vals}

    serialisable_args = {
        key: value for key, value in vars(args).items() if not key.startswith("_")
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": serialisable_args,
                "data": results,
                "averaged_metrics": averaged,
                "num_samples": len(results),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(averaged, f, ensure_ascii=False, indent=2)

    print(f"\n== {file_tag} / {args.memory_name} ==")
    print(f"  results -> {out_json}")
    print(f"  scores  -> {score_file}")
    if averaged:
        for key in sorted(averaged):
            print(f"  {key:16s}: {averaged[key]:.2f}")
    else:
        print("  (no metrics produced)")


if __name__ == "__main__":
    main()
