#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMM evaluation script for MMLongBench.

Core loop per sample:
1) Build raw memory pages (batch write).
2) Build summary memory (coarse abstracts over page chunks).
3) Multi-round QA:
   - Plan retrieval queries from current summaries (few-shot guided).
   - Retrieve from raw pages (incremental, avoid re-fetching same pages).
   - Revise summaries with retrieved evidence (add/delete/modify).
   - Force at least min_rounds before allowing early stop.
4) Generate final answer with strict short-answer formatting.
5) Built-in sub_em scoring (no external dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evomm import InMemoryMemoryStore, InMemoryPageStore, VLLMGenerator
from evomm.schemas import Page
from eval.mmmemory_retriever_adapter import MMMemoryStyleRetriever


SYSTEM_PROMPT = (
    "You are a precise multimodal long-context QA assistant.\n"
    "Answer only based on retrieved evidence.\n"
    "If evidence is insufficient, say so briefly instead of hallucinating."
)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_multimodal_content(text: str) -> List[Dict[str, Any]]:
    pattern = r"\[Image: (.*?)\]"
    parts = re.split(pattern, text)
    content: List[Dict[str, Any]] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part:
                content.append({"type": "text", "text": part})
            continue
        image_path = part.strip()
        if os.path.exists(image_path):
            content.append({"type": "image_url", "image_url": {"url": f"file://{image_path}"}})
        else:
            content.append({"type": "text", "text": f"[Image not found: {image_path}]"})
    return content


class MultimodalVLLMGenerator(VLLMGenerator):
    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if prompt is not None and messages is None:
            if "[Image: " in prompt:
                messages = [{"role": "user", "content": parse_multimodal_content(prompt)}]
            else:
                messages = [{"role": "user", "content": prompt}]
        elif messages is not None:
            converted: List[Dict[str, Any]] = []
            for m in messages:
                c = m.get("content")
                if isinstance(c, str) and "[Image: " in c:
                    converted.append({"role": m["role"], "content": parse_multimodal_content(c)})
                else:
                    converted.append({"role": m["role"], "content": c})
            messages = converted
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages


def _abs_join(root: str, rel_path: str) -> str:
    return os.path.abspath(os.path.join(root, rel_path))


def _load_jsonl(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
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


def _flatten_icl_json(path: str, image_root: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    out: List[Dict[str, Any]] = []
    for domain, data in blob.items():
        exemplars = data.get("exemplar_list", [])
        test_examples = data.get("test_example", [])
        for i, ex in enumerate(test_examples):
            if limit is not None and len(out) >= limit:
                return out
            ctx_items: List[Dict[str, Any]] = []
            if exemplars:
                chosen = exemplars[i % len(exemplars)]
                rounds: List[Dict[str, Any]] = []
                for rd in chosen:
                    rounds.extend(rd)
                for r in rounds[:16]:
                    img_rel = r.get("image", "")
                    if img_rel:
                        ctx_items.append(
                            {
                                "text": f"Training exemplar label={r.get('id', '')}",
                                "images": [_abs_join(image_root, img_rel)],
                            }
                        )
            q_img_rel = ex.get("image", "")
            question = "Classify the test image according to the exemplars.\n<image>"
            question_mm = question.replace("<image>", f"[Image: {_abs_join(image_root, q_img_rel)}]")
            out.append(
                {
                    "id": ex.get("id", f"{domain}_{i}"),
                    "dataset": "icl",
                    "question_plain": "Classify the test image according to the exemplars.",
                    "question_with_mm": question_mm,
                    "answer": ex.get("answer", ""),
                    "answer_format": "String",
                    "ctx_items": ctx_items,
                }
            )
    return out


def _normalize_sample(sample: Dict[str, Any], data_root: str, dataset_hint: str) -> Dict[str, Any]:
    sid = sample.get("id", "")
    answer = sample.get("answer", sample.get("summary", ""))
    answer_format = sample.get("answer_format", "String")
    ctx_items: List[Dict[str, Any]] = []
    question_plain = sample.get("question", "Summarize this document.")
    question_with_mm = question_plain

    is_vrag = "ctxs" in sample and any(k in dataset_hint for k in ["infoseek", "viquae"])

    if is_vrag:
        for c in sample.get("ctxs", []):
            if not isinstance(c, dict):
                continue
            text = f"Document (Title: {c.get('title', '')}): {c.get('text', '')}"
            ctx_items.append({"text": text, "images": []})
        q_img = sample.get("image")
        if q_img:
            q_abs = _abs_join(data_root, q_img)
            question_with_mm = f"[Image: {q_abs}]\nQuestion: {question_plain}"
    elif "ctxs" in sample:
        ctxs = sample.get("ctxs", []) or []
        image_pool = [_abs_join(data_root, p) for p in sample.get("image_list", [])]
        image_cursor = 0
        for c in ctxs:
            if isinstance(c, dict) and c.get("type") == "image":
                if image_cursor < len(image_pool):
                    ctx_items.append({"text": "", "images": [image_pool[image_cursor]]})
                    image_cursor += 1
                continue
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            if text.strip():
                ctx_items.append({"text": text, "images": []})

    if "page_list" in sample:
        for i, rel in enumerate(sample.get("page_list", [])):
            abs_path = _abs_join(data_root, rel)
            ctx_items.append({"text": f"Document page {i}", "images": [abs_path]})

    if "image_list" in sample and "ctxs" not in sample and "page_list" not in sample:
        for i, rel in enumerate(sample.get("image_list", [])):
            abs_path = _abs_join(data_root, rel)
            ctx_items.append({"text": f"Image page {i}", "images": [abs_path]})

    if "choices_image" in sample:
        choice_lines = [question_plain]
        for j, rel in enumerate(sample.get("choices_image", [])):
            choice_lines.append(f"{chr(ord('A') + j)}. [Image: {_abs_join(data_root, rel)}]")
        question_with_mm = "\n".join(choice_lines)
    elif "question" in sample and "<image>" in str(sample.get("question", "")):
        imgs = [_abs_join(data_root, p) for p in sample.get("needle_image_list", [])]
        text = str(sample.get("question", ""))
        for img in imgs:
            text = text.replace("<image>", f"[Image: {img}]", 1)
        question_with_mm = text

    if "summary" in sample and "question" not in sample:
        question_plain = "Please summarize the document."
        question_with_mm = question_plain
        answer_format = "Summary"

    if not ctx_items:
        ctx_items.append({"text": "No context extracted.", "images": []})

    return {
        "id": sid,
        "dataset": dataset_hint,
        "question_plain": question_plain,
        "question_with_mm": question_with_mm,
        "answer": answer,
        "answer_format": answer_format,
        "ctx_items": ctx_items,
        "raw_sample": sample,
    }


def load_mmlongbench_samples(
    file_path: str,
    image_root: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_hint = os.path.basename(file_path).lower()
    if file_path.endswith(".jsonl"):
        rows = _load_jsonl(file_path, limit)
        return [_normalize_sample(r, image_root, dataset_hint) for r in rows]
    if file_path.endswith(".json"):
        return _flatten_icl_json(file_path, image_root, limit)
    raise ValueError(f"Unsupported file format: {file_path}")


# ---------------------------------------------------------------------------
# Built-in scoring (no rouge_score dependency needed for sub_em)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def builtin_sub_em(prediction: str, answers: Any) -> float:
    pred_norm = _normalize_answer(prediction)
    if isinstance(answers, str):
        answers = [answers]
    if isinstance(answers, list) and answers and isinstance(answers[0], list):
        answers = [a for group in answers for a in group]
    for ans in answers:
        if _normalize_answer(str(ans)) in pred_norm:
            return 1.0
    return 0.0


def builtin_score(prediction: str, sample: Dict[str, Any], metric: str) -> Dict[str, Any]:
    answer = sample.get("answer", "")
    if metric == "sub_em":
        return {"sub_em": builtin_sub_em(prediction, answer)}
    if metric == "binary_acc":
        pred_lower = _normalize_answer(prediction)
        match = re.search(r"\b(yes|no)\b", pred_lower)
        pred_label = match.group(0) if match else ""
        gt_label = _normalize_answer(str(answer))
        gt_match = re.search(r"\b(yes|no)\b", gt_label)
        gt_label = gt_match.group(0) if gt_match else str(answer).strip().lower()
        return {"binary_acc": 1.0 if pred_label == gt_label else 0.0}
    if metric == "mc_acc":
        pred_match = re.search(r"\b([A-J])\b", prediction.upper())
        pred_choice = ord(pred_match.group(0)) - ord("A") if pred_match else -1
        return {"mc_acc": 1.0 if pred_choice == answer else 0.0}
    return {"sub_em": builtin_sub_em(prediction, answer)}


def try_score(prediction: str, sample: Dict[str, Any], metric: str, mmlongbench_root: str) -> Dict[str, Any]:
    util_file = os.path.join(mmlongbench_root, "utils.py")
    if os.path.exists(util_file):
        if mmlongbench_root not in sys.path:
            sys.path.append(mmlongbench_root)
        try:
            from utils import calculate_metrics  # type: ignore
            answer = sample.get("answer", "")
            extra_info = {"answer_format": sample.get("answer_format", "String")}
            return calculate_metrics(prediction, answer, metric, extra_info=extra_info)
        except Exception:
            pass
    return builtin_score(prediction, sample, metric)


# ---------------------------------------------------------------------------
# Memory construction (batch write to avoid per-page disk I/O)
# ---------------------------------------------------------------------------

def summarize_chunk(text_blob: str, generator: MultimodalVLLMGenerator) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        "Summarize the following raw memory into concise factual bullets (4-8 bullets).\n"
        "Keep key entities, numbers, temporal clues, and image-grounded details.\n"
        "Do not infer unseen facts.\n\n"
        f"{text_blob}\n\nSummary:"
    )
    try:
        out = generator.generate_single(prompt=prompt).get("text", "").strip()
        return out or text_blob[:500]
    except Exception:
        return text_blob[:500]


def build_two_level_memory(
    sample: Dict[str, Any],
    sample_outdir: str,
    generator: MultimodalVLLMGenerator,
    chunk_size: int,
) -> Tuple[InMemoryMemoryStore, InMemoryPageStore]:
    memory_store = InMemoryMemoryStore(dir_path=sample_outdir)
    page_store = InMemoryPageStore(dir_path=sample_outdir)

    memory_store.save(memory_store.load().__class__(abstracts=[]))
    page_store.save([])

    pages_to_add: List[Page] = []
    for i, item in enumerate(sample["ctx_items"]):
        lines = [f"=== RAW CHUNK {i} ==="]
        if item.get("text"):
            lines.append(item["text"])
        for img in item.get("images", []):
            lines.append(f"[Image: {img}]")
        content = "\n".join(lines)
        pages_to_add.append(
            Page(
                header=f"[RAW {i}] sample={sample['id']}",
                content=content,
                meta={"chunk_id": str(i), "images": item.get("images", [])},
            )
        )
    page_store.save(pages_to_add)

    for start in range(0, len(pages_to_add), chunk_size):
        part = pages_to_add[start : start + chunk_size]
        blob = "\n\n".join(p.content for p in part)
        summary = summarize_chunk(blob, generator)
        memory_store.add(f"[SUMMARY {start}-{start + len(part) - 1}] {summary}")

    return memory_store, page_store


# ---------------------------------------------------------------------------
# Retrieval planning (few-shot guided, gap-aware)
# ---------------------------------------------------------------------------

PLAN_FEW_SHOT = """Example:
Question: What country does this lake belong to?
Summaries: [SUMMARY 0-7] Sils is a municipality in Catalonia, Spain...
Queries:
1. "lake country location"
2. "lake image identification"
3. "Sils municipality lake"

Example:
Question: What is the highest point of this mountain?
Summaries: [SUMMARY 0-7] Mont Blanc massif spans France, Italy, Switzerland...
Queries:
1. "mountain highest point name"
2. "mountain peak summit elevation"
3. "mountain image identification"
"""


def plan_queries(
    question: str,
    summaries: List[str],
    generator: MultimodalVLLMGenerator,
    round_idx: int,
    max_queries: int = 4,
    prev_queries: Optional[List[str]] = None,
) -> List[str]:
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["queries"],
    }
    round_guidance = ""
    if round_idx == 0:
        round_guidance = "This is the first retrieval round. Generate broad queries covering the main question."
    else:
        round_guidance = (
            f"This is round {round_idx + 1}. Previous queries were: {prev_queries}.\n"
            "Generate DIFFERENT queries targeting information gaps in the summaries."
        )

    prompt = (
        "You are planning retrieval queries for long multimodal document QA.\n"
        f"{PLAN_FEW_SHOT}\n"
        f"Now plan queries for:\n"
        f"Question:\n{question}\n\n"
        f"Current summaries:\n{chr(10).join(summaries[:16])}\n\n"
        f"{round_guidance}\n\n"
        f"Return up to {max_queries} focused, diverse search queries as JSON."
    )
    try:
        resp = generator.generate_single(prompt=prompt, schema=schema)
        data = resp.get("json") or json.loads(resp.get("text", "{}"))
        queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
    except Exception:
        queries = []
    if not queries:
        queries = [question]
    return queries[:max_queries]


# ---------------------------------------------------------------------------
# Summary revision (with forced multi-round)
# ---------------------------------------------------------------------------

def revise_summaries(
    question: str,
    summaries: List[str],
    evidence_text: str,
    generator: MultimodalVLLMGenerator,
    force_need_more: bool = False,
) -> Tuple[List[str], bool]:
    schema = {
        "type": "object",
        "properties": {
            "updated_summaries": {"type": "array", "items": {"type": "string"}},
            "need_more_retrieval": {"type": "boolean"},
            "confidence": {
                "type": "number",
                "description": "0.0 to 1.0, how confident you are that the summaries now contain enough info to answer the question",
            },
        },
        "required": ["updated_summaries", "need_more_retrieval", "confidence"],
    }
    prompt = (
        "You update hierarchical summaries after retrieval for long document QA.\n"
        "Rules:\n"
        "- Keep evidence-grounded facts, remove contradicted facts, add missing key facts.\n"
        "- Set need_more_retrieval=true if the summaries still lack critical info for the question.\n"
        "- Set confidence (0-1) for how well you can answer the question from current summaries.\n\n"
        f"Question:\n{question}\n\n"
        f"Current summaries:\n{chr(10).join(summaries)}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Return revised summaries, need_more_retrieval, and confidence."
    )
    try:
        resp = generator.generate_single(prompt=prompt, schema=schema)
        data = resp.get("json") or json.loads(resp.get("text", "{}"))
        updated = [s.strip() for s in data.get("updated_summaries", []) if s and s.strip()]
        need_more = bool(data.get("need_more_retrieval", False))
        confidence = float(data.get("confidence", 0.5))
        if not updated:
            updated = summaries
        if force_need_more:
            need_more = True
        elif confidence < 0.7:
            need_more = True
        return updated, need_more
    except Exception:
        return summaries, force_need_more


def build_evidence_text(page_store: InMemoryPageStore, page_indices: List[int], max_chars: int = 8000) -> str:
    pages = page_store.load()
    chunks: List[str] = []
    total = 0
    for idx in page_indices:
        if idx < 0 or idx >= len(pages):
            continue
        text = f"[PAGE {idx}] {pages[idx].content}"
        total += len(text)
        if total > max_chars:
            break
        chunks.append(text)
    return "\n\n".join(chunks)


def infer_metric(sample: Dict[str, Any], file_path: str) -> str:
    fn = os.path.basename(file_path).lower()
    raw = sample.get("raw_sample", {})
    if "answer_format" in raw:
        return "doc_qa"
    if "choices_image" in raw:
        return "mc_acc"
    if "counting" in fn:
        return "soft_acc"
    if "vh_single" in fn or "vh_multi" in fn:
        return "binary_acc"
    if "summary" in raw and "question" not in raw:
        return "rouge"
    return "sub_em"


# ---------------------------------------------------------------------------
# Final answer generation (strict short-answer format)
# ---------------------------------------------------------------------------

ANSWER_FORMAT_HINT = {
    "sub_em": (
        "IMPORTANT: Output ONLY the answer itself in a few words. "
        "Do NOT include explanations, reasoning, analysis, or introductory phrases. "
        "Examples of good answers: 'Switzerland', 'Dent de Barme', '42', 'Yes'.\n"
    ),
    "binary_acc": "Answer with ONLY 'Yes' or 'No'. No explanation.\n",
    "mc_acc": "Answer with ONLY the letter (A, B, C, etc.). No explanation.\n",
    "rouge": "Write a concise summary (200-550 words) organized in paragraphs.\n",
    "doc_qa": (
        "Answer as concisely as possible using a single phrase or sentence. "
        "If not answerable, reply 'Not answerable.'\n"
    ),
}


def answer_with_memory(
    sample: Dict[str, Any],
    summaries: List[str],
    memory_dict_list: List[Dict[str, Any]],
    generator: MultimodalVLLMGenerator,
    metric: str = "sub_em",
) -> str:
    memory_lines: List[str] = []
    for i, m in enumerate(memory_dict_list):
        memory_lines.append(f"[Memory {i}]")
        memory_lines.append(m.get("text", ""))
        img = m.get("image")
        if isinstance(img, dict) and img.get("path"):
            memory_lines.append(f"[Image: {img['path']}]")

    fmt_hint = ANSWER_FORMAT_HINT.get(metric, ANSWER_FORMAT_HINT["sub_em"])
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Use revised summaries and retrieved raw memories to answer the question.\n"
        f"{fmt_hint}\n"
        f"Revised summaries:\n{chr(10).join(summaries)}\n\n"
        f"Retrieved raw memories:\n{chr(10).join(memory_lines)}\n\n"
        f"Question:\n{sample['question_with_mm']}\n\n"
        "Answer:"
    )
    try:
        return generator.generate_single(prompt=prompt).get("text", "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main sample processing loop
# ---------------------------------------------------------------------------

def run_one_sample(
    sample: Dict[str, Any],
    sample_outdir: str,
    generator: MultimodalVLLMGenerator,
    mm_retriever: MMMemoryStyleRetriever,
    max_rounds: int,
    min_rounds: int,
    chunk_size: int,
    top_k: int,
    metric: str,
    retriever_state_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    os.makedirs(sample_outdir, exist_ok=True)
    memory_store, page_store = build_two_level_memory(sample, sample_outdir, generator, chunk_size)

    # Reuse built retriever state for identical contexts to avoid repeated index construction.
    ctx_signature = hashlib.sha1(
        json.dumps(sample.get("ctx_items", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reused_retriever = False
    if retriever_state_cache is not None and ctx_signature in retriever_state_cache:
        try:
            mm_retriever.restore_state(retriever_state_cache[ctx_signature])
            reused_retriever = True
        except Exception:
            mm_retriever.build(page_store)
    else:
        mm_retriever.build(page_store)
    if retriever_state_cache is not None and not reused_retriever:
        try:
            retriever_state_cache[ctx_signature] = mm_retriever.snapshot_state()
        except Exception:
            pass

    retrieved_page_set: set = set()
    retrieved_page_indices: List[int] = []
    traces: List[Dict[str, Any]] = []
    all_prev_queries: List[str] = []

    for ridx in range(max_rounds):
        summaries = memory_store.load().abstracts
        queries = plan_queries(
            sample["question_with_mm"],
            summaries,
            generator,
            round_idx=ridx,
            max_queries=4,
            prev_queries=all_prev_queries[-8:] if all_prev_queries else None,
        )
        all_prev_queries.extend(queries)

        search_hits = mm_retriever.search(queries, top_k=top_k)

        cur_new_indices: List[int] = []
        for hits in search_hits:
            for h in hits:
                pid = str(h.page_id or "").strip()
                if pid.isdigit():
                    page_idx = int(pid)
                    if page_idx not in retrieved_page_set:
                        retrieved_page_set.add(page_idx)
                        retrieved_page_indices.append(page_idx)
                        cur_new_indices.append(page_idx)

        evidence_text = build_evidence_text(page_store, retrieved_page_indices)

        force_need_more = ridx < min_rounds - 1
        updated_summaries, need_more = revise_summaries(
            sample["question_plain"],
            summaries,
            evidence_text,
            generator,
            force_need_more=force_need_more,
        )
        state = memory_store.load()
        state.abstracts = updated_summaries
        memory_store.save(state)

        traces.append(
            {
                "round": ridx + 1,
                "queries": queries,
                "new_page_indices": cur_new_indices,
                "all_page_indices": list(retrieved_page_indices),
                "need_more_retrieval": need_more,
                "total_retrieved": len(retrieved_page_indices),
            }
        )
        if not need_more:
            break

    memory_dict_list, retrieved_ids = mm_retriever.build_memory_dict_list_from_page_indices(
        page_store, retrieved_page_indices
    )
    final_summaries = memory_store.load().abstracts
    pred = answer_with_memory(sample, final_summaries, memory_dict_list, generator, metric=metric)

    return {
        "id": sample["id"],
        "question": sample["question_plain"],
        "question_with_mm": sample["question_with_mm"],
        "prediction": pred,
        "answer": sample.get("answer", ""),
        "answer_format": sample.get("answer_format", "String"),
        "retrieved_page_indices": retrieved_page_indices,
        "retrieved_dialogue_ids": retrieved_ids,
        "final_summaries": final_summaries,
        "traces": traces,
        "num_rounds": len(traces),
    }


# ---------------------------------------------------------------------------
# Resume support: load partial results
# ---------------------------------------------------------------------------

def load_partial_results(out_file: str, out_jsonl_file: str) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}

    if os.path.exists(out_jsonl_file):
        try:
            with open(out_jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    key = f"{item.get('id', '')}_{item.get('question', '')}"
                    by_key[key] = item
            if by_key:
                return by_key
        except Exception:
            pass

    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                key = f"{item.get('id', '')}_{item.get('question', '')}"
                by_key[key] = item
        except Exception:
            pass
    return by_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EvoMM evaluation for MMLongBench")
    parser.add_argument("--data-file", type=str, required=True, help="MMLongBench test file (.jsonl/.json)")
    parser.add_argument(
        "--image-root",
        type=str,
        default=os.environ.get(
            "MMLONGBENCH_IMAGE_ROOT", "./datasets/mmlongbench/mmlb_image"
        ),
    )
    parser.add_argument(
        "--mmlongbench-root",
        type=str,
        default=os.environ.get("MMLB_ROOT", "./MMLongBench"),
    )
    parser.add_argument("--outdir", type=str, default="./results/mmlongbench_evomm")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-instruct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="empty")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=3, help="max summary-retrieval revision rounds")
    parser.add_argument("--min-rounds", type=int, default=2, help="minimum rounds before allowing early stop")
    parser.add_argument("--chunk-size", type=int, default=32, help="raw pages per summary chunk")
    parser.add_argument("--top-k", type=int, default=6, help="retrieval top-k per planned query")
    parser.add_argument("--no-score", action="store_true", help="disable metric scoring")
    parser.add_argument("--no-resume", action="store_true", help="disable resuming from partial results")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = {
        "model_name": args.model,
        "api_key": args.api_key,
        "base_url": args.api_base,
        "temperature": 0.0,
        "max_tokens": 1024,
        "system_prompt": SYSTEM_PROMPT,
    }
    generator = MultimodalVLLMGenerator(cfg)
    # Keep retrieval model/process alive across samples to avoid repeated gme loading.
    mm_retriever = MMMemoryStyleRetriever(top_k=max(args.top_k, 4))

    samples = load_mmlongbench_samples(args.data_file, args.image_root, args.limit)
    print(f"Loaded {len(samples)} samples from {args.data_file}")

    file_tag = os.path.splitext(os.path.basename(args.data_file))[0]
    run_outdir = os.path.join(args.outdir, file_tag)
    os.makedirs(run_outdir, exist_ok=True)

    out_file = os.path.join(args.outdir, f"{file_tag}_evomm_results.json")
    out_jsonl_file = os.path.join(args.outdir, f"{file_tag}_evomm_results.jsonl")
    if args.no_resume and os.path.exists(out_jsonl_file):
        os.remove(out_jsonl_file)
    if args.no_resume and os.path.exists(out_file):
        os.remove(out_file)
    existing = {} if args.no_resume else load_partial_results(out_file, out_jsonl_file)
    if existing:
        print(f"Resuming: found {len(existing)} completed samples")

    results: List[Dict[str, Any]] = []
    skipped = 0
    metrics_agg: Dict[str, List[float]] = defaultdict(list)
    retriever_state_cache: Dict[str, Dict[str, Any]] = {}

    for i, sample in enumerate(tqdm(samples, desc="MMLongBench-EvoMM"), 1):
        resume_key = f"{sample['id']}_{sample['question_plain']}"
        if resume_key in existing:
            mr = existing[resume_key].get("metric_result", {})
            for mk, mv in mr.items():
                if isinstance(mv, (int, float)):
                    metrics_agg[mk].append(float(mv))
            skipped += 1
            continue

        metric = infer_metric(sample, args.data_file)
        sample_outdir = os.path.join(run_outdir, f"sample_{i}_{sample['id']}")
        item = run_one_sample(
            sample=sample,
            sample_outdir=sample_outdir,
            generator=generator,
            mm_retriever=mm_retriever,
            max_rounds=args.max_rounds,
            min_rounds=args.min_rounds,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            metric=metric,
            retriever_state_cache=retriever_state_cache,
        )
        if not args.no_score:
            item["metric"] = metric
            item["metric_result"] = try_score(
                prediction=item["prediction"],
                sample=sample,
                metric=metric,
                mmlongbench_root=args.mmlongbench_root,
            )
            for mk, mv in item["metric_result"].items():
                if isinstance(mv, (int, float)):
                    metrics_agg[mk].append(float(mv))
        item["timestamp"] = now_ts()
        results.append(item)
        with open(out_jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if existing:
        # Keep one final record per resumed sample and avoid duplicates in final JSON output.
        results = list(existing.values()) + results

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --------------- Summary report ---------------
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {len(results)} samples ({skipped} resumed)")
    print(f"Results: {out_file}")
    if metrics_agg:
        print(f"\n--- Metrics Summary ---")
        for mk, vals in sorted(metrics_agg.items()):
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {mk}: {avg*100:.1f}% ({sum(1 for v in vals if v >= 1.0)}/{len(vals)} correct)")

    round_counts = [item.get("num_rounds", 1) for item in results]
    page_counts = [len(item.get("retrieved_page_indices", [])) for item in results]
    print(f"\n--- Retrieval Summary ---")
    print(f"  Rounds: avg={sum(round_counts)/len(round_counts):.1f} min={min(round_counts)} max={max(round_counts)}")
    print(f"  Pages retrieved: avg={sum(page_counts)/len(page_counts):.1f} min={min(page_counts)} max={max(page_counts)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
