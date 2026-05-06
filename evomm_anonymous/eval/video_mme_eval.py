#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMM evaluation script for Video-MME benchmark.

Pipeline (memory-centric):
1) Extract frames and build video sessions (chunk-level) + turn pages (frame-level).
2) Summarize each session into explicit long-term memory and persist it.
3) For each QA: memory-guided research retrieval.
4) Build evidence from retrieved pages.
5) Check/revise long-term memory summaries with evidence.
6) Answer with merged retrieval evidence + checked summary.
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress repeated fork-related tokenizer warnings during retrieval.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add parent directory to path to import evomm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eval.self_correcting_research import SelfCorrectingResearchAgent
    from eval.mmmemory_retriever_adapter import MMMemoryStyleRetriever
    from eval.memgallery_test_mm import MultimodalVLLMGenerator
except ImportError:
    # Fallback for when running directly from eval directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from self_correcting_research import SelfCorrectingResearchAgent
    from mmmemory_retriever_adapter import MMMemoryStyleRetriever
    from memgallery_test_mm import MultimodalVLLMGenerator

from evomm import (
    ExperienceStore,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
    SelfEvolvingResearchAgent,
)
from evomm.schemas import Page


def _pick_first_existing(candidates: List[str], fallback: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return fallback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_ROOT = _pick_first_existing(
    [
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "datasets", "video-mme")),
        os.path.expanduser("~/datasets/video-mme"),
        "./datasets/video-mme",
    ],
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "datasets", "video-mme")),
)

# Constants
DATASET_ROOT = os.environ.get("VIDEOMME_DATASET_ROOT", DEFAULT_DATASET_ROOT)
VIDEO_MAP_FILE = os.environ.get(
    "VIDEOMME_VIDEO_MAP_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_map.json"),
)
PARQUET_FILE = os.environ.get(
    "VIDEOMME_PARQUET_FILE",
    os.path.join(DATASET_ROOT, "videomme", "test-00000-of-00001.parquet"),
)
DEFAULT_OUTDIR = os.environ.get(
    "VIDEOMME_OUTDIR",
    os.path.join(PROJECT_ROOT, "results", "videomme_evomm"),
)
ALLOWED_MEDIA_ROOT = os.environ.get(
    "VIDEOMME_ALLOWED_MEDIA_ROOT",
    os.path.abspath(os.path.dirname(DEFAULT_OUTDIR)),
)

SYSTEM_PROMPT = (
    "You are an AI assistant evaluated on video understanding and reasoning.\n"
    "You are given retrieved memory from video frames and a multiple-choice question.\n"
    "Answer only based on provided memory evidence.\n"
    "If evidence is insufficient, output the most likely option conservatively."
)

MSG_START_PROMPT_WO_MEMORY = "The retrieved memory contents are as follows:\n\n"
MMMEMORY_DIALOGUE_PROMPT = "{textual_context}\nimage:\nimage_id: {image_id}\nimage_content:\n"

VIDEO_QA_PROMPT = (
    "Your task is to answer the multiple-choice question using the retrieved video memories.\n"
    "Return one concise answer in format: '<OptionLetter>. <OptionText>' (e.g., 'B. ...').\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n"
)

OPTION_VERIFY_PROMPT = (
    "Based on the video memory evidence provided above, evaluate support strength for the statement.\n\n"
    "Statement: \"{statement}\"\n\n"
    "Reply with ONLY one label: 'STRONG', 'WEAK', or 'NONE'."
)

VISUAL_TASK_TYPES = {
    "Counting Problem", "Object Recognition", "Object Reasoning",
    "Action Recognition", "Scene Transition",
}

NON_VISUAL_TASK_TYPES = {
    "Action Reasoning", "Information Synopsis", "Attribute Perception",
    "OCR Problems", "Spatial Reasoning",
}

VISUAL_QA_PROMPT = (
    "Look at the video frames provided and answer the question.\n"
    "You MUST choose exactly one option. Answer in format: '<Letter>. <OptionText>'.\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n"
)


def extract_frames(video_path: str, output_dir: str, max_frames: int = 600, target_fps: float = 1.0) -> List[str]:
    """Extract sampled frames and return frame paths."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0

    use_fps = target_fps
    if duration > 0 and duration * use_fps > max_frames:
        use_fps = max_frames / duration

    frame_interval = max(1, int(fps / use_fps)) if fps > 0 else 1

    frame_paths: List[str] = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_idx += 1
        if saved_count >= max_frames:
            break

    cap.release()
    return frame_paths


SUMMARY_SAMPLE_PER_SESSION = 8


def build_video_records(
    frame_paths: List[str],
    video_id: str,
    session_window: int = 32,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build pseudo-session records and turn records from frame sequence."""
    session_records: List[Dict[str, Any]] = []
    turn_records: List[Dict[str, Any]] = []

    for i, frame_path in enumerate(frame_paths):
        session_idx = i // session_window + 1
        session_id = f"S{session_idx:02d}"
        turn_id = f"{i:04d}"
        image_id = f"{video_id}:F{turn_id}"

        turn_text = (
            f"=== SESSION {session_id} - TURN {turn_id} ===\n"
            f"Frame index: {i}\n"
            f"[ImageID: {image_id}]\n"
            f"[Image: {frame_path}]"
        )

        turn_records.append(
            {
                "session_id": session_id,
                "date": f"frame_{i}",
                "turn_id": turn_id,
                "image_ids": [image_id],
                "image_captions": [""],
                "images": [frame_path],
                "raw_text": turn_text,
            }
        )

    session_map: Dict[str, List[Dict[str, Any]]] = {}
    for tr in turn_records:
        session_map.setdefault(tr["session_id"], []).append(tr)

    for sid in sorted(session_map.keys()):
        turns = session_map[sid]
        all_images: List[str] = []
        for t in turns:
            for img in t["images"]:
                all_images.append(img)

        n = len(turns)
        k = min(SUMMARY_SAMPLE_PER_SESSION, n)
        step = max(1, n // k)
        sampled = [turns[j] for j in range(0, n, step)][:k]

        lines = [f"=== SESSION {sid} ({n} frames, {k} representative shown) ===", ""]
        for t in sampled:
            lines.append(f"Frame index: {int(t['turn_id'])}")
            for img in t["images"]:
                lines.append(f"[Image: {img}]")
            lines.append("")

        session_records.append(
            {
                "session_id": sid,
                "date": f"frames_{turns[0]['turn_id']}_{turns[-1]['turn_id']}",
                "raw_text": "\n".join(lines).strip(),
                "images": all_images,
            }
        )

    return session_records, turn_records


SESSION_SUMMARY_PROMPT = (
    "You are building a long-term memory for a video.\n"
    "Given the frames from one session segment, write 4-8 concise factual bullet points.\n\n"
    "Requirements:\n"
    "- Describe what you SEE: objects, people, text overlays, scenes, actions.\n"
    "- Note counts precisely (e.g., 'three red stockings', 'two people at a desk').\n"
    "- Record temporal progression: what appears first, what changes, what appears last.\n"
    "- Include any on-screen text, titles, captions, logos verbatim.\n"
    "- Do NOT interpret or infer meaning beyond what is visible.\n"
    "- Do NOT generalize (e.g., say 'Christmas-themed' instead of listing specific items).\n\n"
    "{raw_text}\n\n"
    "Factual bullet-point summary:"
)


def summarize_session(record: Dict[str, Any], generator: MultimodalVLLMGenerator) -> str:
    prompt = SESSION_SUMMARY_PROMPT.format(raw_text=record["raw_text"])
    try:
        out = generator.generate_single(prompt=prompt).get("text", "").strip()
        return out or record["raw_text"][:300]
    except Exception:
        return record["raw_text"][:300]


def build_memory_and_pages(
    sample_outdir: str,
    session_records: List[Dict[str, Any]],
    turn_records: List[Dict[str, Any]],
    generator: MultimodalVLLMGenerator,
    rebuild: bool = True,
) -> Tuple[InMemoryMemoryStore, InMemoryPageStore]:
    memory_store = InMemoryMemoryStore(dir_path=sample_outdir)
    page_store = InMemoryPageStore(dir_path=sample_outdir)

    memory_file = os.path.join(sample_outdir, "memory_state.json")
    pages_file = os.path.join(sample_outdir, "pages.json")
    summaries_file = os.path.join(sample_outdir, "session_summaries.json")

    if (not rebuild) and os.path.exists(memory_file) and os.path.exists(pages_file):
        return memory_store, page_store

    memory_store.save(memory_store.load().__class__(abstracts=[]))
    page_store.save([])

    all_summaries: List[Dict[str, Any]] = []

    for rec in session_records:
        summary = summarize_session(rec, generator)
        memory_store.add(f"[SESSION {rec['session_id']}] {summary}")
        all_summaries.append(
            {
                "session_id": rec["session_id"],
                "date": rec["date"],
                "summary": summary,
            }
        )

    for turn in turn_records:
        page_store.add(
            Page(
                header=f"[TURN {turn['turn_id']}] Session {turn['session_id']}",
                content=turn["raw_text"],
                meta={
                    "session_id": turn["session_id"],
                    "date": turn["date"],
                    "turn_id": turn["turn_id"],
                    "image_ids": turn["image_ids"],
                    "image_captions": turn["image_captions"],
                    "images": turn["images"],
                },
            )
        )

    with open(summaries_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    return memory_store, page_store


def _dedup_adjacent_frames(page_indices: List[int], min_gap: int = 3) -> List[int]:
    """Drop page indices that are too close to an already-kept index."""
    if not page_indices:
        return []
    kept: List[int] = [page_indices[0]]
    for idx in page_indices[1:]:
        if all(abs(idx - k) >= min_gap for k in kept):
            kept.append(idx)
    return kept


def build_retrieval_evidence(
    research_out: Any,
    page_store: InMemoryPageStore,
    mm_retriever: Optional[Any] = None,
    top_k_pages: int = 8,
) -> Tuple[str, List[str], List[int]]:
    raw = research_out.raw_memory if research_out else {}
    temp_memory = raw.get("temp_memory", {}) if isinstance(raw, dict) else {}
    source_ids = temp_memory.get("sources", []) if isinstance(temp_memory, dict) else []

    pages = page_store.load()
    resolved_page_index: List[int] = []
    retrieved_ids: List[str] = []

    turn_to_page: Dict[str, int] = {}
    for idx, p in enumerate(pages):
        turn_id = str((p.meta or {}).get("turn_id", "") or "").strip()
        if turn_id:
            turn_to_page[turn_id] = idx

    for src in source_ids:
        src_s = str(src).strip()
        idx: Optional[int] = None
        if src_s.isdigit():
            idx = int(src_s)
        elif src_s in turn_to_page:
            idx = turn_to_page[src_s]
        if idx is None or idx < 0 or idx >= len(pages):
            continue
        resolved_page_index.append(idx)

    if mm_retriever and hasattr(mm_retriever, "last_retrieved_ids"):
        for rid in mm_retriever.last_retrieved_ids:
            rid_s = str(rid).strip()
            if rid_s in turn_to_page:
                resolved_page_index.append(turn_to_page[rid_s])

    seen = set()
    unique_idx: List[int] = []
    for idx in resolved_page_index:
        if idx in seen:
            continue
        seen.add(idx)
        unique_idx.append(idx)

    dedup_idx = _dedup_adjacent_frames(unique_idx, min_gap=3)[:top_k_pages]

    evidence_chunks: List[str] = []
    for idx in dedup_idx:
        p = pages[idx]
        turn_id = str((p.meta or {}).get("turn_id", "") or "")
        if turn_id:
            retrieved_ids.append(turn_id)
        evidence_chunks.append(f"[PAGE {idx}] {p.content[:1200]}")

    dedup_ids: List[str] = []
    seen_id = set()
    for rid in retrieved_ids:
        if rid in seen_id:
            continue
        seen_id.add(rid)
        dedup_ids.append(rid)

    return "\n\n".join(evidence_chunks), dedup_ids, dedup_idx


def _extract_session_summary_entry(abstract: str) -> Tuple[Optional[str], str]:
    m = re.match(r"^\[SESSION\s+([^\]]+)\]\s*(.*)$", abstract.strip(), re.DOTALL)
    if not m:
        return None, abstract
    return m.group(1).strip(), m.group(2).strip()


REVISION_GATE_PROMPT = (
    "You are a strict fact-checker for video memory.\n"
    "Given the previous session summary and newly retrieved frame evidence,\n"
    "decide the update type.\n\n"
    "Rules:\n"
    "- CONTRADICT: summary states facts opposite to evidence.\n"
    "- INCOMPLETE: no contradiction, but crucial detail for current question is missing.\n"
    "- CONSISTENT: no contradiction and no crucial omission.\n\n"
    "Previous Summary:\n{summary}\n\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Reply with ONLY one label: 'CONTRADICT', 'INCOMPLETE', or 'CONSISTENT'."
)

REVISION_REWRITE_PROMPT = (
    "You are correcting a video memory summary that contradicts frame evidence.\n"
    "Rewrite the summary to fix ONLY the contradicted facts.\n"
    "Keep all other information unchanged. Do not shorten or generalize.\n"
    "Output ONLY the revised summary text.\n\n"
    "Previous Summary:\n{summary}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Revised Summary:"
)

REVISION_APPEND_PROMPT = (
    "You are updating a video memory summary.\n"
    "There is no contradiction, but important detail for the current question is missing.\n"
    "Append 1 concise evidence-grounded sentence to the end.\n"
    "Do not rewrite existing statements. Do not invent details.\n"
    "Output ONLY the updated summary.\n\n"
    "Previous Summary:\n{summary}\n\n"
    "Question:\n{question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Updated Summary:"
)


def revise_long_term_summaries_with_evidence(
    memory_store: InMemoryMemoryStore,
    target_session_ids: List[str],
    question: str,
    evidence_text: str,
    generator: MultimodalVLLMGenerator,
) -> List[Dict[str, Any]]:
    revision_logs: List[Dict[str, Any]] = []
    if not target_session_ids or not evidence_text.strip():
        return revision_logs

    state = memory_store.load()
    if not state.abstracts:
        return revision_logs

    target_set = {str(sid).strip() for sid in target_session_ids if str(sid).strip()}
    if not target_set:
        return revision_logs

    updated = False
    new_abstracts: List[str] = []

    for abs_text in state.abstracts:
        sid, summary_text = _extract_session_summary_entry(abs_text)
        if sid is None or sid not in target_set:
            new_abstracts.append(abs_text)
            continue

        gate_prompt = REVISION_GATE_PROMPT.format(
            summary=summary_text, question=question, evidence=evidence_text
        )
        try:
            gate_response = generator.generate_single(prompt=gate_prompt).get("text", "").strip().upper()
        except Exception:
            gate_response = "CONSISTENT"

        gate_label = "CONSISTENT"
        if "CONTRADICT" in gate_response:
            gate_label = "CONTRADICT"
        elif "INCOMPLETE" in gate_response:
            gate_label = "INCOMPLETE"

        if gate_label == "CONSISTENT":
            new_abstracts.append(abs_text)
            continue

        if gate_label == "CONTRADICT":
            rewrite_prompt = REVISION_REWRITE_PROMPT.format(summary=summary_text, evidence=evidence_text)
            try:
                revised = generator.generate_single(prompt=rewrite_prompt).get("text", "").strip()
                if not revised or len(revised) < 20:
                    revised = summary_text
            except Exception:
                revised = summary_text
        else:
            append_prompt = REVISION_APPEND_PROMPT.format(
                summary=summary_text, question=question, evidence=evidence_text
            )
            try:
                revised = generator.generate_single(prompt=append_prompt).get("text", "").strip()
                if not revised or len(revised) < len(summary_text):
                    revised = summary_text
            except Exception:
                revised = summary_text

        if revised != summary_text:
            updated = True
            revision_logs.append(
                {
                    "session_id": sid,
                    "question": question,
                    "gate": gate_label,
                    "old_summary": summary_text,
                    "new_summary": revised,
                    "evidence_text": evidence_text,
                }
            )
        new_abstracts.append(f"[SESSION {sid}] {revised}")

    if updated:
        state.abstracts = new_abstracts
        memory_store.save(state)
    return revision_logs


def merge_research_and_recall_for_qa(
    research_memory_list: List[Dict[str, Any]],
    recall_memory_list: List[Dict[str, Any]],
    max_total: int = 12,
) -> List[Dict[str, Any]]:
    research_dialogue_ids = {
        mem.get("dialogue_id", "")
        for mem in research_memory_list
        if mem.get("dialogue_id", "")
    }

    merged: List[Dict[str, Any]] = []
    for mem in research_memory_list:
        merged.append(mem)
        if len(merged) >= max_total:
            return merged

    for mem in recall_memory_list:
        if len(merged) >= max_total:
            break
        did = mem.get("dialogue_id", "")
        if did and did in research_dialogue_ids:
            continue
        if did:
            research_dialogue_ids.add(did)
        merged.append(mem)

    return merged


def score_options_individually(
    memory_dict_list: List[Dict[str, Any]],
    question: str,
    options: List[str],
    checked_summary: str,
    generator: MultimodalVLLMGenerator,
) -> Tuple[Optional[str], Dict[str, float]]:
    """Score each option independently via TRUE/FALSE verification.
    Returns the best option string like 'A. ...' or None if fallback needed."""

    memory_text_parts: List[str] = []
    for mem in memory_dict_list:
        memory_text_parts.append(mem.get("text", ""))
    memory_context = "\n".join(memory_text_parts)

    if checked_summary.strip():
        memory_context += f"\n\nChecked Long-Term Summary:\n{checked_summary}"

    scores: Dict[str, float] = {}

    for opt in options:
        letter_match = re.match(r"^([A-D])\.\s*(.+)", opt.strip())
        if not letter_match:
            continue
        letter = letter_match.group(1)
        statement = f"For the question '{question}', the answer is: {opt.strip()}"

        verify_prompt = (
            f"Video memory evidence:\n{memory_context}\n\n"
            + OPTION_VERIFY_PROMPT.format(statement=statement)
        )

        try:
            resp = generator.generate_single(prompt=verify_prompt).get("text", "").strip().upper()
            if "STRONG" in resp:
                scores[letter] = scores.get(letter, 0) + 2.0
            elif "WEAK" in resp:
                scores[letter] = scores.get(letter, 0) + 1.0
            else:
                scores[letter] = scores.get(letter, 0) + 0.0
        except Exception:
            scores[letter] = scores.get(letter, 0) + 0.0

    if not scores:
        return None, {}

    # For "not discussed/mentioned" questions, prefer least supported option.
    q_lower = question.lower()
    if "not discussed" in q_lower or "not mentioned" in q_lower:
        best_letter = min(scores, key=scores.get)
    else:
        best_letter = max(scores, key=scores.get)

    for opt in options:
        if opt.strip().startswith(best_letter):
            return opt.strip(), scores

    return f"{best_letter}.", scores


def build_qa_user_content(
    memory_dict_list: List[Dict[str, Any]],
    question: str,
    options: List[str],
    checked_summary: str,
) -> List[Dict[str, Any]]:
    conversation_info_flow: List[Dict[str, Any]] = []

    for mem in memory_dict_list:
        img_info = mem.get("image")
        if img_info and isinstance(img_info, dict) and img_info.get("path"):
            img_id = img_info.get("img_id", "")
            textual_content = MMMEMORY_DIALOGUE_PROMPT.format(
                textual_context=mem.get("text", ""),
                image_id=img_id,
            )
            conversation_info_flow.append({"type": "text", "text": textual_content})
            img_path = img_info["path"]
            if os.path.exists(img_path):
                conversation_info_flow.append(
                    {"type": "image_url", "image_url": {"url": f"file://{img_path}"}}
                )
        else:
            conversation_info_flow.append({"type": "text", "text": mem.get("text", "")})

    qa_prompt = VIDEO_QA_PROMPT.format(question=question, options="\n".join(options))

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": MSG_START_PROMPT_WO_MEMORY},
        *conversation_info_flow,
    ]

    if checked_summary.strip():
        user_content.append({"type": "text", "text": f"\nChecked Long-Term Summary:\n{checked_summary}\n"})

    user_content.append({"type": "text", "text": qa_prompt})
    return user_content


def evaluate_video(
    video_id: str,
    video_path: str,
    questions: List[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    api_base: str,
    api_key: str,
    enable_evolution: bool = False,
    rebuild_memory: bool = True,
    session_window: int = 32,
    shared_mm_retriever: Optional[MMMemoryStyleRetriever] = None,
) -> List[Dict[str, Any]]:
    video_outdir = os.path.join(output_dir, f"video_{video_id}")
    frames_dir = os.path.join(video_outdir, "frames")

    if os.path.exists(video_outdir):
        shutil.rmtree(video_outdir)
    os.makedirs(video_outdir, exist_ok=True)

    print(f"Processing video {video_id} from {video_path}...")
    frame_paths = extract_frames(video_path, frames_dir)
    print(f"Extracted {len(frame_paths)} frames.")

    gen_cfg = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": api_base,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    mm_generator = MultimodalVLLMGenerator(gen_cfg)

    session_records, turn_records = build_video_records(frame_paths, video_id, session_window=session_window)
    memory_store, page_store = build_memory_and_pages(
        sample_outdir=video_outdir,
        session_records=session_records,
        turn_records=turn_records,
        generator=mm_generator,
        rebuild=rebuild_memory,
    )

    retrievers: Dict[str, Any] = {}
    index_dir = os.path.join(video_outdir, "page_index")
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    index_r = IndexRetriever(IndexRetrieverConfig(index_dir=index_dir).__dict__)
    index_r.build(page_store)
    retrievers["page_index"] = index_r

    # Reuse a single MMMemoryStyleRetriever across videos to avoid reloading GME each time.
    mm_retriever = shared_mm_retriever if shared_mm_retriever is not None else MMMemoryStyleRetriever(top_k=10)
    mm_retriever.build(page_store)
    retrievers["keyword"] = mm_retriever
    retrievers["vector"] = mm_retriever

    base_agent = SelfCorrectingResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=mm_generator,
        max_iters=3,
    )
    experience_store = ExperienceStore(dir_path=output_dir)
    research_agent = SelfEvolvingResearchAgent.wrap(
        base_agent,
        experience_store=experience_store,
        evolution_enabled=enable_evolution,
    )

    results: List[Dict[str, Any]] = []
    memory_revision_log: List[Dict[str, Any]] = []
    pages = page_store.load()

    for q in questions:
        question_text = str(q["question"])
        options_raw = q["options"]
        options = options_raw.tolist() if isinstance(options_raw, np.ndarray) else list(options_raw)

        full_question = f"{question_text}\nOptions:\n" + "\n".join(options) + "\nAnswer:"
        print(f"Q: {question_text}")

        try:
            research_out = research_agent.research(full_question, category=str(q.get("task_type", "VideoQA")))
            checked_summary = research_out.integrated_memory

            evidence_text, retrieved_ids, resolved_page_index = build_retrieval_evidence(
                research_out=research_out,
                page_store=page_store,
                mm_retriever=mm_retriever,
                top_k_pages=8,
            )

            target_session_ids: List[str] = []
            seen_sids = set()
            for idx in resolved_page_index:
                sid = str((pages[idx].meta or {}).get("session_id", "") or "").strip()
                if sid and sid not in seen_sids:
                    seen_sids.add(sid)
                    target_session_ids.append(sid)

            revision_entries = revise_long_term_summaries_with_evidence(
                memory_store=memory_store,
                target_session_ids=target_session_ids,
                question=question_text,
                evidence_text=evidence_text,
                generator=mm_generator,
            )
            if revision_entries:
                for entry in revision_entries:
                    entry["question_id"] = q["question_id"]
                memory_revision_log.extend(revision_entries)

            research_memory_list, _ = mm_retriever.build_memory_dict_list_from_page_indices(
                page_store, resolved_page_index
            )
            recall_memory_list, recall_ids = mm_retriever.recall_for_qa(
                question=question_text,
                question_image_path=None,
                top_k=10,
            )
            memory_dict_list = merge_research_and_recall_for_qa(
                research_memory_list,
                recall_memory_list,
                max_total=12,
            )

            if not retrieved_ids:
                retrieved_ids = recall_ids

            task_type = str(q.get("task_type", ""))
            is_visual_task = task_type in VISUAL_TASK_TYPES
            solver_type = "visual_mm" if is_visual_task else "option_verify"
            option_scores: Dict[str, float] = {}

            if is_visual_task:
                # Visual tasks: multimodal QA with retrieved frame images directly
                user_content = build_qa_user_content(
                    memory_dict_list=memory_dict_list,
                    question=question_text,
                    options=options,
                    checked_summary=checked_summary,
                )
                visual_qa = VISUAL_QA_PROMPT.format(
                    question=question_text, options="\n".join(options)
                )
                user_content.append({"type": "text", "text": visual_qa})
                qa_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                answer = mm_generator.generate_single(messages=qa_messages).get("text", "").strip()
            else:
                # Non-visual tasks: per-option verification with text evidence
                option_answer, option_scores = score_options_individually(
                    memory_dict_list=memory_dict_list,
                    question=question_text,
                    options=options,
                    checked_summary=checked_summary,
                    generator=mm_generator,
                )
                if option_answer:
                    answer = option_answer
                else:
                    solver_type = "fallback_generate"
                    user_content = build_qa_user_content(
                        memory_dict_list=memory_dict_list,
                        question=question_text,
                        options=options,
                        checked_summary=checked_summary,
                    )
                    qa_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                    answer = mm_generator.generate_single(messages=qa_messages).get("text", "").strip()

            print(f"A: {answer} [{'visual' if is_visual_task else 'verify'}]")

            results.append(
                {
                    "video_id": video_id,
                    "question_id": q["question_id"],
                    "task_type": q.get("task_type", ""),
                    "question": question_text,
                    "options": options,
                    "answer": q.get("answer", ""),
                    "model_answer": answer,
                    "retrieved_ids": retrieved_ids,
                    "resolved_page_index": resolved_page_index,
                    "checked_summary": checked_summary,
                    "solver_type": solver_type,
                    "option_scores": option_scores,
                    "research_trace": research_out.raw_memory,
                }
            )
        except Exception as e:
            print(f"Error processing question {q['question_id']}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "video_id": video_id,
                    "question_id": q["question_id"],
                    "task_type": q.get("task_type", ""),
                    "question": question_text,
                    "options": options,
                    "answer": q.get("answer", ""),
                    "error": str(e),
                }
            )

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    revision_log_file = os.path.join(video_outdir, "memory_revision_log.json")
    with open(revision_log_file, "w", encoding="utf-8") as f:
        json.dump(memory_revision_log, f, ensure_ascii=False, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-MME Evaluation (Memory-centric EvoMM)")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct", help="Model name")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--api-key", type=str, default="empty", help="API key")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--no-evolution", action="store_true", help="Disable evolution")
    parser.add_argument("--no-rebuild-memory", action="store_true", help="Reuse existing per-video memory/pages")
    parser.add_argument("--session-window", type=int, default=32, help="Frames per pseudo session")
    args = parser.parse_args()

    args.outdir = os.path.abspath(args.outdir)
    if not args.outdir.startswith(ALLOWED_MEDIA_ROOT):
        raise ValueError(
            f"--outdir must be under {ALLOWED_MEDIA_ROOT} for local image access, got: {args.outdir}"
        )

    os.makedirs(args.outdir, exist_ok=True)

    with open(VIDEO_MAP_FILE, "r", encoding="utf-8") as f:
        video_map = json.load(f)

    df = pd.read_parquet(PARQUET_FILE)
    grouped = df.groupby("video_id")

    all_results: List[Dict[str, Any]] = []
    structured_results: List[Dict[str, Any]] = []
    processed_count = 0
    # Global retriever: load GME once, then rebuild storage/index per video.
    shared_mm_retriever = MMMemoryStyleRetriever(top_k=10)

    for video_id, group in tqdm(grouped, desc="Videos"):
        if args.limit and processed_count >= args.limit:
            break

        youtube_id = group.iloc[0]["videoID"]
        if youtube_id not in video_map:
            print(f"Warning: Video {youtube_id} (ID: {video_id}) not found in map. Skipping.")
            continue

        video_info = video_map[youtube_id]
        zip_file = video_info["zip_file"]
        rel_path = video_info["rel_path"]

        zip_path = os.path.join(DATASET_ROOT, zip_file)
        temp_video_dir = os.path.join(args.outdir, "temp_videos")
        os.makedirs(temp_video_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract(rel_path, temp_video_dir)

            video_path = os.path.join(temp_video_dir, rel_path)
            questions = group.to_dict("records")

            results = evaluate_video(
                video_id=str(video_id),
                video_path=video_path,
                questions=questions,
                output_dir=args.outdir,
                model_name=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                enable_evolution=not args.no_evolution,
                rebuild_memory=not args.no_rebuild_memory,
                session_window=args.session_window,
                shared_mm_retriever=shared_mm_retriever,
            )

            all_results.extend(results)

            video_meta = group.iloc[0]
            structured_video = {
                "video_id": str(video_id),
                "duration": video_meta.get("duration", ""),
                "domain": video_meta.get("domain", ""),
                "sub_category": video_meta.get("sub_category", ""),
                "questions": [],
            }

            for r in results:
                structured_video["questions"].append(
                    {
                        "question_id": r.get("question_id", ""),
                        "task_type": r.get("task_type", ""),
                        "question": r.get("question", ""),
                        "options": r.get("options", []),
                        "answer": r.get("answer", ""),
                        "response": r.get("model_answer", ""),
                    }
                )

            structured_results.append(structured_video)

            if os.path.exists(video_path):
                os.remove(video_path)

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            import traceback

            traceback.print_exc()

        processed_count += 1

        with open(os.path.join(args.outdir, "results_intermediate.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.outdir, "output_test.json"), "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.outdir, "results_final.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "output_test.json"), "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {args.outdir}")


if __name__ == "__main__":
    main()
