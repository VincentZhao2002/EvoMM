#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-enhanced DocQA evaluation on MMLongBench.

Per-sample pipeline (for one jsonl row of longdocurl / mmlongdoc / slidevqa):
    1) Build a page store (one Page per image in ``page_list``).
    2) If ``num_pages <= top_k``: keep all pages in doc order (fast path,
       behaviour-equivalent to MMLongBench ``load_doc_qa``).
    3) Else: seed retrieval via ``MMMemoryStyleRetriever.recall_for_qa`` ->
       1-hop/2-hop temporal graph expansion -> top-k selection.
    4) Compose a docqa-compatible multimodal prompt and call vLLM.
    5) Parse with MMLongBench ``parse_output`` and score with
       ``calculate_metrics`` (metric ``doc_qa`` or ``doc_qa_llm``).

Outputs a JSON bundle + a ``.score`` file with averaged metrics, plus a
``.jsonl`` cache for resume.  Output schema mimics the layout that
``eval_api_batch.py`` produces so downstream aggregation is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evomm.schemas import Page  # type: ignore

import openai

# NOTE: MMMemoryStyleRetriever is imported lazily inside ``_lazy_import_retriever``
# because its import chain (Mem-Gallery memengine + langchain_core) may be absent
# in environments where the user only needs --no-retrieval / fast path.
MMMemoryStyleRetriever = None  # populated on demand


def _lazy_import_retriever():
    global MMMemoryStyleRetriever
    if MMMemoryStyleRetriever is None:
        from eval.mmmemory_retriever_adapter import (  # type: ignore
            MMMemoryStyleRetriever as _R,
        )
        MMMemoryStyleRetriever = _R
    return MMMemoryStyleRetriever


# ---------------------------------------------------------------------------
# MMLongBench docqa prompt (kept byte-identical to load_doc_qa in data.py)
# ---------------------------------------------------------------------------

USER_TEMPLATE = (
    "You are given a document with text and images, and a question. "
    "Answer the question as concisely as you can, using a single phrase or "
    "sentence if possible. If the question cannot be answered based on the "
    "information in the article, write 'Not answerable.' Write your answer "
    "in the following format:\nAnswer: [answer]\n\n"
    "{context}\n\n"
    "Question: {question}"
)
ITEM_TEMPLATE = "Document {doc_id:.15}: <image>"
SYSTEM_TEMPLATE = "Answer:"


# ---------------------------------------------------------------------------
# Lightweight page store (only implements ``load`` / ``save`` as required by
# MMMemoryStyleRetriever.build).
# ---------------------------------------------------------------------------

class _ListPageStore:
    def __init__(self, pages: List[Page]) -> None:
        self._pages = pages

    def load(self) -> List[Page]:
        return self._pages

    def save(self, pages: List[Page]) -> None:
        self._pages = pages


def build_page_store(sample: Dict[str, Any], image_root: str) -> _ListPageStore:
    pages: List[Page] = []
    for i, rel in enumerate(sample["page_list"]):
        abs_path = os.path.join(image_root, rel)
        pages.append(
            Page(
                header=f"[PAGE {i}] doc={sample['doc_name']}",
                content=f"[Image: {abs_path}]",
                meta={
                    "turn_id": str(i),
                    "session_id": sample["doc_name"],
                    "images": [abs_path],
                    "image_ids": [f"PAGE{i}"],
                    "page_idx": i,
                },
            )
        )
    return _ListPageStore(pages)


# ---------------------------------------------------------------------------
# Graph expansion (single-document -> temporal only + image-id look-up)
# ---------------------------------------------------------------------------

def _temporal_expand(seeds: List[int], num_pages: int, max_extra: int) -> List[int]:
    out: List[int] = []
    seen = set(seeds)
    for hop in (1, 2):
        for p in seeds:
            for nb in (p - hop, p + hop):
                if 0 <= nb < num_pages and nb not in seen:
                    out.append(nb)
                    seen.add(nb)
                    if len(out) >= max_extra:
                        return out
    return out


def select_pages(
    sample: Dict[str, Any],
    image_root: str,
    mm_retriever: Any,
    mm_retriever_lock: threading.Lock,
    top_k: int,
    graph_extra: int,
    enable_retrieval: bool,
) -> Tuple[List[int], List[int]]:
    """Return (final_indices_in_doc_order, seed_indices)."""
    num_pages = len(sample["page_list"])
    if (not enable_retrieval) or mm_retriever is None or num_pages <= top_k:
        return list(range(num_pages)), []

    # Seed retrieval: question-conditioned top-k over this doc's pages.
    seed_indices: List[int] = []
    with mm_retriever_lock:
        store = build_page_store(sample, image_root)
        mm_retriever.build(store)
        mem_list, _ = mm_retriever.recall_for_qa(sample["question"], top_k=top_k)
    for m in mem_list:
        img = m.get("image") or {}
        iid = str(img.get("img_id", ""))
        mat = re.match(r"^PAGE(\d+)$", iid)
        if mat:
            idx = int(mat.group(1))
            if 0 <= idx < num_pages:
                seed_indices.append(idx)
    seed_indices = list(dict.fromkeys(seed_indices))

    if not seed_indices:
        return list(range(min(num_pages, top_k))), []

    extra = _temporal_expand(seed_indices, num_pages, max_extra=graph_extra)
    picked = list(dict.fromkeys(seed_indices + extra))[:top_k]
    picked.sort()  # preserve document order for the VLM
    return picked, seed_indices


# ---------------------------------------------------------------------------
# Prompt assembly for vLLM (OpenAI-compatible multimodal messages)
# ---------------------------------------------------------------------------

def build_messages(
    sample: Dict[str, Any],
    page_indices: List[int],
    image_root: str,
) -> List[Dict[str, Any]]:
    doc = sample["doc_name"]
    context_block = "\n\n".join(ITEM_TEMPLATE.format(doc_id=doc) for _ in page_indices)
    question = (
        f"Based on Document {doc[:15]}, answer the following question. "
        + sample["question"]
    )
    prompt_text = (
        USER_TEMPLATE.format(context=context_block, question=question)
        + "\n"
        + SYSTEM_TEMPLATE
    )
    # Splice image placeholders back with absolute file:// urls.
    parts = prompt_text.split("<image>")
    if len(parts) - 1 != len(page_indices):
        raise RuntimeError(
            f"prompt split mismatch: {len(parts) - 1} slots vs {len(page_indices)} pages"
        )
    content: List[Dict[str, Any]] = []
    for i, seg in enumerate(parts):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(page_indices):
            rel = sample["page_list"][page_indices[i]]
            abs_path = os.path.join(image_root, rel)
            content.append(
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}}
            )
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Scoring: prefer MMLongBench utils.py; fall back to simple substring match
# ---------------------------------------------------------------------------

_SCORE_FN_LOCK = threading.Lock()
_SCORE_FNS: Dict[str, Any] = {}


def _load_mmlb_utils(mmlb_root: str) -> None:
    if "parse_output" in _SCORE_FNS:
        return
    with _SCORE_FN_LOCK:
        if "parse_output" in _SCORE_FNS:
            return
        if mmlb_root and mmlb_root not in sys.path:
            sys.path.append(mmlb_root)
        try:
            from utils import calculate_metrics, parse_output  # type: ignore
            _SCORE_FNS["calculate_metrics"] = calculate_metrics
            _SCORE_FNS["parse_output"] = parse_output
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] MMLongBench utils unavailable, fallback scoring: {exc}")
            _SCORE_FNS["calculate_metrics"] = None
            _SCORE_FNS["parse_output"] = None


def parse_output(text: str) -> str:
    fn = _SCORE_FNS.get("parse_output")
    if fn is not None:
        try:
            parsed = fn(text, prefix=SYSTEM_TEMPLATE)
            if parsed is not None:
                return parsed
        except Exception:
            pass
    idx = text.find(SYSTEM_TEMPLATE)
    if idx >= 0:
        return text[idx + len(SYSTEM_TEMPLATE):].strip()
    return text.strip()


def _fallback_doc_qa_score(pred: str, sample: Dict[str, Any]) -> float:
    ans = sample.get("answer", "")
    if isinstance(ans, list):
        ans = " ".join(str(a) for a in ans)
    return 1.0 if str(ans).strip() and str(ans).lower() in pred.lower() else 0.0


def score_sample(
    pred: str,
    sample: Dict[str, Any],
    judge_client: Any = None,
    judge_model: str = "",
) -> Tuple[Dict[str, float], Any]:
    calc = _SCORE_FNS.get("calculate_metrics")
    if calc is None:
        return {"doc_qa": _fallback_doc_qa_score(pred, sample)}, None
    extra: Dict[str, Any] = {"answer_format": sample.get("answer_format", "String")}
    answer = sample["answer"]
    try:
        if judge_client is not None:
            extra.update(
                {
                    "llm_judge_client": judge_client,
                    "llm_judge_model": judge_model,
                    "question": sample["question"],
                }
            )
            mets = calc(pred, answer, "doc_qa_llm", extra_info=extra)
            val = mets.get("doc_qa_llm")
            if isinstance(val, dict):
                mets["doc_qa_llm"] = float(val.get("final_score", 0.0))
                return mets, val.get("judge_result")
            return mets, None
        mets = calc(pred, answer, "doc_qa", extra_info=extra)
        return mets, None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] scoring failed ({exc}); falling back")
        return {"doc_qa": _fallback_doc_qa_score(pred, sample)}, None


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def run_one_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    mm_retriever: Any,
    mm_retriever_lock: threading.Lock,
    vllm_client: openai.OpenAI,
) -> Dict[str, Any]:
    picked, seeds = select_pages(
        sample=sample,
        image_root=args.image_root,
        mm_retriever=mm_retriever,
        mm_retriever_lock=mm_retriever_lock,
        top_k=args.top_k,
        graph_extra=args.graph_extra,
        enable_retrieval=(not args.no_retrieval),
    )
    messages = build_messages(sample, picked, args.image_root)

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
        "seed_page_indices": seeds,
        "selected_page_indices": picked,
        "output": output_text,
        "parsed_output": parsed,
        "metric_result": mets,
        "judge_result": judge_result,
        "latency_s": round(latency, 3),
        "error": error,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_samples(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_resume(out_jsonl: str) -> Tuple[List[Dict[str, Any]], set]:
    if not os.path.exists(out_jsonl):
        return [], set()
    done: List[Dict[str, Any]] = []
    keys: set = set()
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            key = f"{item.get('id', '')}_{item.get('question', '')}"
            if key in keys:
                continue
            keys.add(key)
            done.append(item)
    return done, keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph-enhanced DocQA evaluation on MMLongBench."
    )
    parser.add_argument("--data-file", required=True, help="documentQA/*.jsonl path")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--mmlongbench-root",
        default=os.environ.get("MMLB_ROOT", "./MMLongBench/MMLongBench"),
        help="MMLongBench python root (for parse_output / calculate_metrics).",
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct-199")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="max pages passed into the VLM prompt (<= num_pages).",
    )
    parser.add_argument(
        "--graph-extra",
        type=int,
        default=4,
        help="extra temporal-neighbour pages added on top of retrieval seeds.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Fallback cap matching docqa_all.yaml (max_test_samples=100).",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="disable retrieval/graph entirely; pass the whole page_list in order.",
    )
    parser.add_argument("--docqa-llm-judge", action="store_true")
    parser.add_argument("--llm-judge-key", default="")
    parser.add_argument("--llm-judge-endpoint", default="")
    parser.add_argument("--llm-judge-model", default="Doubao-Seed-1.8")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _load_mmlb_utils(args.mmlongbench_root)

    args._judge_client = None  # type: ignore[attr-defined]
    if args.docqa_llm_judge:
        if not (args.llm_judge_key and args.llm_judge_endpoint):
            raise SystemExit(
                "--docqa-llm-judge requires --llm-judge-key and --llm-judge-endpoint"
            )
        args._judge_client = openai.OpenAI(  # type: ignore[attr-defined]
            api_key=args.llm_judge_key, base_url=args.llm_judge_endpoint
        )

    vllm_client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)

    total_cap = args.limit if args.limit is not None else args.max_samples
    samples = load_samples(args.data_file, total_cap)
    print(f"Loaded {len(samples)} samples from {args.data_file}")

    mm_retriever: Any = None
    mm_retriever_lock = threading.Lock()
    need_retriever = (not args.no_retrieval) and any(
        len(s.get("page_list", [])) > args.top_k for s in samples
    )
    if need_retriever:
        print("Initialising MMMemoryStyleRetriever (GME model) …")
        retriever_cls = _lazy_import_retriever()
        mm_retriever = retriever_cls(top_k=args.top_k)
    else:
        print("All samples fit in top-k (or --no-retrieval); skipping retriever init.")

    file_tag = os.path.splitext(os.path.basename(args.data_file))[0]
    out_jsonl = os.path.join(args.outdir, f"{file_tag}_graph.jsonl")
    out_json = os.path.join(args.outdir, f"{file_tag}_graph.json")
    score_file = out_json + ".score"
    if args.no_resume and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    existing, done_keys = load_resume(out_jsonl)
    if existing:
        print(f"Resuming: {len(existing)} samples already in cache.")

    remaining = [
        s for s in samples
        if f"{s.get('id', '')}_{s.get('question', '')}" not in done_keys
    ]
    # Group same-doc questions together so that the retriever's page-level
    # embedding cache warms up on the first question of a doc and every
    # subsequent question hits the cache (build() then costs ~0 GPU time).
    remaining.sort(key=lambda s: (s.get("doc_name", ""), s.get("id", "")))
    print(f"Remaining to process: {len(remaining)} with {args.num_workers} workers")

    results: List[Dict[str, Any]] = list(existing)
    f_out = open(out_jsonl, "a", encoding="utf-8")
    write_lock = threading.Lock()

    def _task(sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            item = run_one_sample(
                sample, args, mm_retriever, mm_retriever_lock, vllm_client
            )
        except Exception as exc:  # noqa: BLE001
            item = {
                "id": sample.get("id", ""),
                "doc_name": sample.get("doc_name", ""),
                "question": sample.get("question", ""),
                "error": repr(exc),
                "metric_result": {"doc_qa": 0.0},
            }
        with write_lock:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()
        return item

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_task, s) for s in remaining]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=file_tag):
                results.append(fut.result())
    finally:
        f_out.close()

    if mm_retriever is not None and hasattr(mm_retriever, "cache_stats"):
        stats = mm_retriever.cache_stats()
        total = stats["hits"] + stats["misses"]
        hit_rate = (stats["hits"] / total * 100.0) if total else 0.0
        print(
            f"[retriever] emb_cache size={stats['size']} "
            f"hits={stats['hits']} misses={stats['misses']} "
            f"hit_rate={hit_rate:.1f}%"
        )

    # ---------------- Aggregate ----------------
    metrics_agg: Dict[str, List[float]] = defaultdict(list)
    for it in results:
        mr = it.get("metric_result") or {}
        for k, v in mr.items():
            if isinstance(v, (int, float)):
                metrics_agg[k].append(float(v))
    averaged = {k: 100.0 * sum(v) / len(v) for k, v in metrics_agg.items() if v}

    serialisable_args = {
        k: v for k, v in vars(args).items() if not k.startswith("_")
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

    print(f"\n== {file_tag} ==")
    print(f"  results -> {out_json}")
    print(f"  scores  -> {score_file}")
    if averaged:
        for k in sorted(averaged):
            print(f"  {k:16s}: {averaged[k]:.2f}")
    else:
        print("  (no metrics produced)")


if __name__ == "__main__":
    main()
