#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoMM + LoCoMo evaluation with self-reflection.

This is the EvoMM-port of the GAM-based
``general-agentic-memory/eval/locomo_test_reflection.py`` script.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evomm import (  # noqa: E402
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
    MemoryAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)

try:  # noqa: E402
    from eval.self_correcting_research import SelfCorrectingResearchAgent
except Exception:  # pragma: no cover - allows running from eval/ directly.
    from self_correcting_research import SelfCorrectingResearchAgent  # type: ignore


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """Extract sessions as (zero_based_idx, timestamp, turns, optional_summary)."""
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for key, value in conv_obj.items():
        match = re.match(r"^session_(\d+)$", key)
        if not (match and isinstance(value, list)):
            continue
        original_idx = int(match.group(1))
        idx = original_idx - 1
        timestamp = conv_obj.get(f"session_{original_idx}_date_time", "")
        session_summary = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append(
            (
                idx,
                timestamp,
                value,
                session_summary if isinstance(session_summary, str) and session_summary.strip() else None,
            )
        )
    sessions.sort(key=lambda x: x[0])
    return sessions


def parse_dialogue_datetime(timestamp: str) -> Optional[datetime]:
    """Parse LoCoMo timestamps such as '1:56 pm on 8 May, 2023'."""
    timestamp = (timestamp or "").strip()
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"):
        try:
            return datetime.strptime(timestamp.upper(), fmt)
        except ValueError:
            continue
    return None


def format_date(value: datetime) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"


def format_month(value: datetime) -> str:
    return f"{value.strftime('%B')} {value.year}"


def previous_weekday(anchor: datetime, weekday: int) -> datetime:
    days_back = (anchor.weekday() - weekday) % 7
    if days_back == 0:
        days_back = 7
    return anchor - timedelta(days=days_back)


def add_months(anchor: datetime, months: int) -> datetime:
    month_index = anchor.month - 1 + months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return anchor.replace(year=year, month=month, day=day)


def relative_time_notes(text: str, timestamp: str) -> List[str]:
    """Resolve common LoCoMo relative time phrases against the session date."""
    anchor = parse_dialogue_datetime(timestamp)
    if anchor is None:
        return []

    lowered = text.lower()
    notes: List[str] = []

    phrase_to_date = {
        "yesterday": anchor - timedelta(days=1),
        "last night": anchor - timedelta(days=1),
        "the previous day": anchor - timedelta(days=1),
    }
    for phrase, resolved in phrase_to_date.items():
        if phrase in lowered:
            notes.append(f'"{phrase}" resolves to {format_date(resolved)}')

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, weekday in weekday_map.items():
        if re.search(rf"\blast\s+{name}\b", lowered):
            notes.append(f'"last {name}" resolves to {format_date(previous_weekday(anchor, weekday))}')

    if re.search(r"\blast\s+fri\b", lowered):
        notes.append(f'"last Fri" resolves to {format_date(previous_weekday(anchor, 4))}')
    if re.search(r"\blast\s+sat\b", lowered):
        notes.append(f'"last Sat" resolves to {format_date(previous_weekday(anchor, 5))}')
    if re.search(r"\blast\s+sun\b", lowered):
        notes.append(f'"last Sun" resolves to {format_date(previous_weekday(anchor, 6))}')

    if "last week" in lowered:
        start = anchor - timedelta(days=anchor.weekday() + 7)
        end = start + timedelta(days=6)
        notes.append(f'"last week" resolves to {format_date(start)} to {format_date(end)}')

    if "last weekend" in lowered:
        saturday = previous_weekday(anchor, 5)
        sunday = previous_weekday(anchor, 6)
        notes.append(f'"last weekend" resolves around {format_date(saturday)} to {format_date(sunday)}')

    if "last year" in lowered:
        notes.append(f'"last year" resolves to {anchor.year - 1}')
    if "this year" in lowered:
        notes.append(f'"this year" resolves to {anchor.year}')
    if "next month" in lowered:
        notes.append(f'"next month" resolves to {format_month(add_months(anchor, 1))}')
    if "this month" in lowered:
        notes.append(f'"this month" resolves to {format_month(anchor)}')

    return list(dict.fromkeys(notes))


def session_to_text(
    idx: int,
    timestamp: str,
    turns: List[Dict[str, Any]],
    session_summary: Optional[str],
) -> str:
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {timestamp} ===", ""]
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
        notes = relative_time_notes(text, timestamp)
        if notes:
            lines.append(f"  [Resolved relative time: {'; '.join(notes)}]")

    if session_summary:
        lines.extend(["", f"Session {idx} summary: {session_summary}"])

    return "\n".join(lines).strip()


def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    conv = sample.get("conversation", {})
    return [session_to_text(idx, ts, turns, summary) for idx, ts, turns, summary in extract_sessions(conv)]


def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    sample_id = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append(
            {
                "sample_id": sample_id,
                "question": q.get("question"),
                "answer": q.get("answer"),
                "category": q.get("category"),
                "evidence": q.get("evidence"),
            }
        )
    return qas


# ---------------------------------------------------------------------------
# QA prompts
# ---------------------------------------------------------------------------

LOCOMO_RESEARCH_SYSTEM_PROMPTS = {
    "integration": """LoCoMo time-handling instructions:
- For questions asking when something happened or will happen, always preserve the evidence session timestamp, e.g. "Dialogue Time: 1:14 pm on 25 May, 2023".
- If the evidence uses a relative time expression such as yesterday, last night, last Friday, last Saturday, last weekend, next month, this month, or last year, resolve it against the session timestamp whenever possible.
- Include both the original relative expression and the resolved absolute date/month/year in the integrated memory.
- For future plans like "next month", prefer a month and year (e.g. June 2023) over the literal phrase "next month".
- Do not drop date anchors from the integrated memory, even when the answer seems obvious.""",
    "reflection": """LoCoMo completeness instructions:
- For time/date questions, information is not complete if it only says a relative phrase like yesterday, last Saturday, or next month while the supporting evidence contains a session timestamp that can anchor it.
- Mark enough=true only when the result contains the date anchor and, where possible, the resolved absolute date/month/year.""",
}


def make_summary_prompt(summary: str, question: str) -> str:
    return f"""Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.

TIME ANSWER RULES:
- If the question asks "when", do NOT answer with relative phrases such as "yesterday", "last Saturday", "last week", "next month", or "last year" when the summary gives a dialogue/session date.
- Resolve relative dates against the dialogue/session date in the summary.
- Use "15 July 2023" for a specific date, "July 2023" for a month, and a single year for year-only answers.
- If the question is about duration, answer in several years, months, or days.
- Output only the final short answer, with no explanation.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def answer_with_summary(category: Optional[int], summary: str, question: str, generator: Any) -> str:
    if not summary.strip():
        return "Unknown"
    prompt = make_summary_prompt_category3(summary, question) if category == 3 else make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    text = str(s).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(s: Any) -> List[str]:
    text = normalize_text(s)
    return text.split() if text else []


def f1_score(pred: Any, gold: Any) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu1_score(pred: Any, gold: Any) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks)
    bp = 1.0 if gtoks and len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks)) if gtoks else 0.0
    return bp * precision


def compute_metrics_by_category(items: List[Dict[str, Any]], pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg: Dict[Any, List[Tuple[float, float]]] = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        val = ex.get(pred_key, "")
        pred = val.get(pred_field, "") if isinstance(val, dict) else val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append(
            {
                "q_idx": idx,
                "category": cat,
                "gold_answer": str(gold),
                "prediction": str(pred),
                "F1": f1,
                "BLEU1": b1,
            }
        )

    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        summary.append(
            {
                "category": cat,
                "count": len(scores),
                "F1_avg": sum(s[0] for s in scores) / len(scores),
                "BLEU1_avg": sum(s[1] for s in scores) / len(scores),
            }
        )
    return summary, rows


# ---------------------------------------------------------------------------
# EvoMM wiring
# ---------------------------------------------------------------------------

def make_generator(
    api_type: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    use_schema: bool = False,
) -> Any:
    config_cls = OpenAIGeneratorConfig if api_type == "openai" else VLLMGeneratorConfig
    generator_cls = OpenAIGenerator if api_type == "openai" else VLLMGenerator
    config = config_cls(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        use_schema=use_schema,
    )
    return generator_cls(config.__dict__)


def build_retrievers(
    sample_results_dir: str,
    dense_model_name: str,
    dense_api_url: Optional[str],
    enable_index: bool,
    enable_bm25: bool,
    enable_dense: bool,
) -> Dict[str, Any]:
    retrievers: Dict[str, Any] = {}

    if enable_index:
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            if os.path.exists(page_index_dir):
                shutil.rmtree(page_index_dir)
            retrievers["page_index"] = IndexRetriever(IndexRetrieverConfig(index_dir=page_index_dir).__dict__)
            print("[OK] Index retriever created")
        except Exception as exc:
            print(f"[WARN] Index retriever failed: {exc}")

    if enable_bm25:
        try:
            from evomm import BM25Retriever

            bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")
            if os.path.exists(bm25_index_dir):
                shutil.rmtree(bm25_index_dir)
            retrievers["keyword"] = BM25Retriever(BM25RetrieverConfig(index_dir=bm25_index_dir, threads=1).__dict__)
            print("[OK] BM25 retriever created")
        except Exception as exc:
            print(f"[WARN] BM25 retriever failed: {exc}")

    if enable_dense:
        try:
            from evomm import DenseRetriever

            dense_index_dir = os.path.join(sample_results_dir, "dense_index")
            if os.path.exists(dense_index_dir):
                shutil.rmtree(dense_index_dir)
            dense_kwargs: Dict[str, Any] = {"index_dir": dense_index_dir}
            if dense_api_url:
                dense_kwargs["api_url"] = dense_api_url
            else:
                dense_kwargs["model_name"] = dense_model_name
            retrievers["vector"] = DenseRetriever(DenseRetrieverConfig(**dense_kwargs).__dict__)
            print("[OK] Dense retriever created")
        except Exception as exc:
            print(f"[WARN] Dense retriever failed: {exc}")

    print(f"[INFO] Created {len(retrievers)} retrievers")
    return retrievers


def process_sample(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    use_schema: bool = False,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai",
    dense_model_name: str = os.environ.get("LOCAL_BGE_PATH", "BAAI/bge-large-en"),
    dense_api_url: Optional[str] = None,
    enable_index: bool = True,
    enable_bm25: bool = True,
    enable_dense: bool = True,
    max_iters: int = 3,
) -> List[Dict[str, Any]]:
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    print(f"\n{'=' * 60}")
    print(f"Processing sample #{sample_index}: {sample_id}")
    print(f"{'=' * 60}")

    try:
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"Session count: {len(session_chunks)}")
        if session_chunks:
            print(f"First session preview:\n{session_chunks[0][:400]}...")

        sample_results_dir = os.path.join(outdir, str(sample_id))
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"Output directory: {sample_results_dir}")

        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)

        print("\nStep 1: Create Memory Generator")
        memory_generator = make_generator(
            memory_api_type,
            memory_model,
            memory_api_key,
            memory_base_url,
            temperature=0.3,
            max_tokens=256,
        )
        memory_agent = MemoryAgent(memory_store=memory_store, page_store=page_store, generator=memory_generator)
        print("[OK] MemoryAgent created")

        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        if not os.path.exists(memory_state_file):
            print("\nStep 2: Build memory with MemoryAgent")
            for i, session_chunk in enumerate(session_chunks, 1):
                print(f"  Processing session {i}/{len(session_chunks)}...")
                memory_agent.memorize(session_chunk)
        else:
            print("[INFO] Found existing memory_state.json; skipping memory build")

        final_state = memory_store.load()
        print(f"[OK] Memory build done; {len(final_state.abstracts)} abstracts")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")

        print("\nStep 3: Create retrievers")
        retrievers = build_retrievers(
            sample_results_dir,
            dense_model_name=dense_model_name,
            dense_api_url=dense_api_url,
            enable_index=enable_index,
            enable_bm25=enable_bm25,
            enable_dense=enable_dense,
        )

        print("\nStep 4: Create Research Generator and Working Generator")
        research_generator = make_generator(
            research_api_type,
            research_model,
            research_api_key,
            research_base_url,
            temperature=0.3,
            max_tokens=2048,
            use_schema=use_schema,
        )
        working_generator = make_generator(
            working_api_type,
            working_model,
            working_api_key,
            working_base_url,
            temperature=0.3,
            max_tokens=256,
        )
        print("[OK] Research Generator and Working Generator created")

        print("\nStep 5: Create SelfCorrectingResearchAgent")
        research_agent = SelfCorrectingResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=max_iters,
            system_prompts=LOCOMO_RESEARCH_SYSTEM_PROMPTS,
        )
        print("[OK] SelfCorrectingResearchAgent created")

        print("\nStep 6: Run Q&A")
        qas = collect_qa_items_for_sample(sample)
        print(f"{len(qas)} questions to answer")

        qa_results: List[Dict[str, Any]] = []
        for i, qi in tqdm(list(enumerate(qas, 1)), desc="Processing questions"):
            question = qi.get("question") or ""
            gold = qi.get("answer")
            category = qi.get("category")

            print(f"\n--- Question {i}/{len(qas)} ---")
            print(f"Question: {question}")
            print(f"Gold answer: {gold}")
            print(f"Category: {category}")

            if category == 5:
                continue

            try:
                result = research_agent.research(question)
                research_summary = result.integrated_memory
                raw_memory = result.raw_memory
                print(f"[Q{i}] [OK] Research done; iterations: {len(raw_memory.get('iterations', []))}")
                print(f"[Q{i}] Research summary: {research_summary[:200]}...")

                research_trace = {
                    "question": question,
                    "raw_memory": raw_memory,
                    "integrated_memory": research_summary,
                    "iterations": raw_memory.get("iterations", []),
                    "search_plans": raw_memory.get("search_plans", []),
                    "reflections": raw_memory.get("reflections", []),
                    "llm_trace": raw_memory.get("llm_trace", []),
                }
                trace_file = os.path.join(sample_results_dir, f"research_trace_q{i}.json")
                with open(trace_file, "w", encoding="utf-8") as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[Q{i}] [INFO] Research trace saved: {trace_file}")

                summary_answer = answer_with_summary(category, research_summary, question, working_generator)
                print(f"[Q{i}] Predicted answer: {summary_answer}")

                qa_results.append(
                    {
                        "question": question,
                        "gold_answer": gold,
                        "category": category,
                        "research_summary": research_summary,
                        "summary_answer": summary_answer,
                        "iterations": len(raw_memory.get("iterations", [])),
                        "research_trace_file": trace_file,
                    }
                )
            except Exception as exc:
                print(f"[Q{i}] [ERROR] Failed to process question: {exc}")
                import traceback

                traceback.print_exc()
                qa_results.append(
                    {
                        "question": question,
                        "gold_answer": gold,
                        "category": category,
                        "error": str(exc),
                    }
                )

        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Results saved to: {results_file}")

        all_research_traces = []
        for idx, qa_result in enumerate(qa_results, 1):
            trace_file = qa_result.get("research_trace_file")
            if trace_file and os.path.exists(trace_file):
                with open(trace_file, "r", encoding="utf-8") as f:
                    all_research_traces.append(
                        {
                            "question_index": idx,
                            "question": qa_result["question"],
                            "category": qa_result["category"],
                            "research_trace": json.load(f),
                        }
                    )

        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, "w", encoding="utf-8") as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] All research traces saved to: {traces_summary_file}")

        print(f"\n{'=' * 60}")
        print("Processing summary")
        print(f"{'=' * 60}")
        print(f"Sample ID: {sample_id}")
        print(f"Sessions: {len(session_chunks)}")
        print(f"Memory abstracts: {len(final_state.abstracts)}")
        print(f"Questions processed: {len(qa_results)}")
        print(f"Research trace files: {len(all_research_traces)}")
        print(f"Results directory: {sample_results_dir}")
        return qa_results
    except Exception as exc:
        print(f"ERROR: Failed to process sample {sample_index}: {exc}")
        import traceback

        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoMM + LoCoMo evaluation with self-reflection")
    parser.add_argument("--data", type=str, required=True, help="Path to LoCoMo dataset")
    parser.add_argument("--outdir", type=str, default="./results/locomo_reflection", help="Output directory")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive)")

    parser.add_argument("--memory-api-key", type=str, default="empty", help="Memory model API key")
    parser.add_argument("--memory-base-url", type=str, default="https://api.openai.com/v1", help="Memory model base URL")
    parser.add_argument("--memory-model", type=str, default="gpt-4o-mini", help="Memory model name")
    parser.add_argument("--memory-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Memory API type")

    parser.add_argument("--research-api-key", type=str, default="empty", help="Research model API key")
    parser.add_argument("--research-base-url", type=str, default="https://api.openai.com/v1", help="Research model base URL")
    parser.add_argument("--research-model", type=str, default="gpt-4o-mini", help="Research model name")
    parser.add_argument("--research-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Research API type")
    parser.add_argument("--use-schema", type=parse_bool, default=False, help="Enable structured output")

    parser.add_argument("--working-api-key", type=str, default="empty", help="Working model API key")
    parser.add_argument("--working-base-url", type=str, default="https://api.openai.com/v1", help="Working model base URL")
    parser.add_argument("--working-model", type=str, default="gpt-4o-mini", help="Working model name")
    parser.add_argument("--working-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Working API type")

    parser.add_argument(
        "--dense-model-name",
        type=str,
        default=os.environ.get("LOCAL_BGE_PATH", "BAAI/bge-large-en"),
        help="Dense retriever model name or local path",
    )
    parser.add_argument("--dense-api-url", type=str, default=None, help="Dense encoding service URL, e.g. http://localhost:8001")
    parser.add_argument("--disable-index", action="store_true", help="Disable page_index retriever")
    parser.add_argument("--disable-bm25", action="store_true", help="Disable BM25 retriever")
    parser.add_argument("--disable-dense", action="store_true", help="Disable dense retriever")
    parser.add_argument("--max-iters", type=int, default=3, help="Max ResearchAgent iterations")

    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    print("=" * 60)
    print("EvoMM + LoCoMo dataset test (with self-reflection)")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Output directory: {args.outdir}")
    print(f"Sample range: {args.start_idx} to {args.end_idx - 1 if args.end_idx else 'all'}")
    print("=" * 60)

    samples = load_locomo(args.data)
    print(f"Loaded {len(samples)} samples")

    if args.end_idx is None:
        args.end_idx = len(samples)
    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"Error: start index {args.start_idx} out of range (total samples: {len(samples)})")
        return
    if args.end_idx > len(samples):
        print(f"Warning: end index {args.end_idx} out of range; clamped to {len(samples)}")
        args.end_idx = len(samples)
    if args.start_idx >= args.end_idx:
        print(f"Error: start {args.start_idx} must be less than end {args.end_idx}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    sample_indices = list(range(args.start_idx, args.end_idx))
    print(f"Will process {len(sample_indices)} samples sequentially...")

    all_results: List[Dict[str, Any]] = []
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        print(f"\n{'=' * 80}")
        print(f"Starting sample {sample_idx}/{len(samples) - 1} (range {args.start_idx}-{args.end_idx - 1})")
        print(f"{'=' * 80}")
        results = process_sample(
            samples[sample_idx],
            sample_idx,
            args.outdir,
            args.memory_api_key,
            args.memory_base_url,
            args.memory_model,
            args.research_api_key,
            args.research_base_url,
            args.research_model,
            args.working_api_key,
            args.working_base_url,
            args.working_model,
            args.use_schema,
            args.memory_api_type,
            args.research_api_type,
            args.working_api_type,
            args.dense_model_name,
            args.dense_api_url,
            not args.disable_index,
            not args.disable_bm25,
            not args.disable_dense,
            args.max_iters,
        )
        print(f"[OK] Sample {sample_idx} done")
        all_results.extend(results)

    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx - 1}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Batch summary saved: {summary_file}")

        print(f"\n{'=' * 60}")
        print("Computing metrics...")
        print(f"{'=' * 60}")
        summary, details = compute_metrics_by_category(all_results, pred_key="summary_answer", pred_field="answer")

        print("\nPer-category stats:")
        for row in summary:
            print(
                f"Category {row['category']}: n={row['count']}, "
                f"F1_avg={row['F1_avg']:.4f}, BLEU1_avg={row['BLEU1_avg']:.4f}"
            )

        all_f1_scores = [row["F1"] for row in details]
        all_bleu1_scores = [row["BLEU1"] for row in details]
        overall_f1_avg = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        overall_bleu1_avg = sum(all_bleu1_scores) / len(all_bleu1_scores) if all_bleu1_scores else 0.0

        print("\nOverall stats:")
        print(f"Total questions: {len(all_results)}")
        print(f"Overall mean F1: {overall_f1_avg:.4f}")
        print(f"Overall mean BLEU1: {overall_bleu1_avg:.4f}")

        statistics = {
            "total_samples": args.end_idx - args.start_idx,
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1_avg,
            "overall_bleu1_avg": overall_bleu1_avg,
            "by_category": summary,
            "details": details,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1,
        }
        stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx - 1}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {stats_file}")

    print(f"\n{'=' * 60}")
    print("[OK] Batch test finished.")
    print(f"Samples processed: {args.end_idx - args.start_idx}")
    print(f"Questions succeeded: {len(all_results)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
