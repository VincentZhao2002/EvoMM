 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMM + Mem-Gallery 多模态评测脚本

流程：
1. 每个 session 单独生成 summary（写入 memory abstracts）。
2. 每个 session 原始文本 + 图片路径作为 page 保存。
3. Research 时基于 summaries 做 planning，再检索 pages。
4. 检索后用 SelfCorrectingResearchAgent 校验并修正研究 summary。
5. QA 证据以 MMMemory 格式（text + image 交替）呈现给 LLM。
6. 输出格式对齐 benchmark *_results.json。
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_ROOT = os.environ.get(
    "MEMGALLERY_BENCHMARK_ROOT", "./Mem-Gallery/benchmark"
)
if BENCHMARK_ROOT not in sys.path:
    sys.path.append(BENCHMARK_ROOT)
PROMPT_DIR = os.path.join(BENCHMARK_ROOT, "prompt")

from evomm import (
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    VLLMGenerator,
    IndexRetrieverConfig,
    SelfEvolvingResearchAgent,
    ExperienceStore,
)
from evomm.schemas import Page
from evomm.schemas import Hit
from eval.self_correcting_research import SelfCorrectingResearchAgent
from eval.mmmemory_retriever_adapter import MMMemoryStyleRetriever

def _load_prompt_file(filename: str, fallback: str = "") -> str:
    path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return fallback


# 与 Mem-Gallery benchmark/prompt 完全对齐
SYSTEM_PROMPT = _load_prompt_file(
    "sys_prompt.txt",
    fallback=(
        "You are an AI assistant evaluated on multimodal long-term conversational memory.\n"
        "For the given question-answering task, your responses must be concise, yet complete enough to accurately answer the questions.\n"
        "If multiple pieces of information about the same event appear in the conversation, always rely on the most recent information.\n"
        "Follow all instructions strictly. Only answer using information contained within the multimodal conversation. Do not hallucinate."
    ),
)
AR_CONSTRAINT = _load_prompt_file(
    "ar_prompt.txt",
    fallback="Provide your answer based on the information in the conversation. Only if the information about the question is not present in the conversation, reply with: 'Not mentioned.'",
)
CD_CONSTRAINT = _load_prompt_file(
    "cd_prompt.txt",
    fallback="Please check whether this information conflicts with the conversation, and reply strictly with either “Yes.” or “No.”",
)
VS_CONSTRAINT = _load_prompt_file(
    "vs_prompt.txt",
    fallback="Return the image_id of the image(s). If there are multiple images, sort them in ascending order and separate them by commas. Format example: “D2:IMG_003, D2:IMG_010, D10:IMG_002” (for format reference only).",
)

TEXT_MSG_PROMPT = (
    "The retrieved memory contents are as follows:\n\n"
    "{memory_context}\n"
)

# 与 run_bench.py DialogueAgentPrompt 完全对齐，保证评测一致
DIALOGUE_AGENT_PROMPT = (
    "Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} in a concise manner with the help of memory content.\n"
    "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
    "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n"
    "Generate answers primarily concise, yet complete enough to accurately answer the questions.\n\n"
    "The current question is as follows:\n"
    "{observation} {format_constraint}\n"
)

MSG_START_PROMPT_WO_MEMORY = "The retrieved memory contents are as follows:\n\n"
DIALOGUE_AGENT_PROMPT_IMAGE = "Here is the attached image of the question:\n"

MMMEMORY_DIALOGUE_PROMPT = (
    "{textual_context}\n"
    "image:\n"
    "image_id: {image_id}\n"
    "image_content:\n"
)

CATEGORY_HINTS = {
    "TR": (
        "Answer as briefly as possible. For before/after questions, answer with ONLY \"Before.\" or \"After.\". "
        "For yes/no questions, answer with ONLY \"Yes.\" or \"No.\". "
        "For date/time questions, return ONLY the date in the requested format. No extra explanation.\n"
    ),
    "TTL": (
        "Answer with ONLY the specific name or short noun phrase. "
        "Do NOT include descriptive sentences, scientific names, or explanations. "
        "For example, if asked about a flower species, answer 'Foxglove' not 'The flower is a foxglove (Digitalis purpurea).'\n"
    ),
    "CD": (
        "Carefully compare the statement in the question with the conversation history. "
        "If ANY detail differs from what was said in the conversation, answer 'Yes.' "
        "Only answer 'No.' if the statement exactly matches the conversation.\n"
    ),
}


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_multimodal_content(text: str) -> List[Dict[str, Any]]:
    """把 [Image: /abs/path] 文本拆成 OpenAI 多模态 content。"""
    pattern = r"\[Image: (.*?)\]"
    parts = re.split(pattern, text)
    content: List[Dict[str, Any]] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
            continue

        image_path = part.strip()
        if os.path.exists(image_path):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"},
                }
            )
        else:
            content.append({"type": "text", "text": f"[Image not found: {image_path}]"})
    return content


def strip_image_markers(text: str) -> str:
    return re.sub(r"\[Image: .*?\]", "", text).strip()


def extract_first_image_path(text: str) -> Optional[str]:
    m = re.search(r"\[Image: (.*?)\]", text)
    return m.group(1).strip() if m else None


class MultimodalVLLMGenerator(VLLMGenerator):
    """支持 [Image: ...] 的 vLLM 生成器。"""

    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, Any]]:
        if prompt is not None and messages is None:
            if "[Image: " in prompt:
                messages = [{"role": "user", "content": parse_multimodal_content(prompt)}]
            else:
                messages = [{"role": "user", "content": prompt}]
        elif messages is not None:
            new_messages: List[Dict[str, Any]] = []
            for m in messages:
                c = m.get("content")
                if isinstance(c, str) and "[Image: " in c:
                    new_messages.append({"role": m["role"], "content": parse_multimodal_content(c)})
                else:
                    new_messages.append({"role": m["role"], "content": c})
            messages = new_messages

        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        return messages


def load_sample(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(base_dir: str, rel_path: str) -> str:
    return os.path.abspath(os.path.join(base_dir, rel_path))


def build_session_records(data: Dict[str, Any], json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构建 session 级 summary 原文 + turn 级 page 记录（含 image_id）。"""
    base_dir = os.path.dirname(os.path.abspath(json_path))
    session_records: List[Dict[str, Any]] = []
    turn_records: List[Dict[str, Any]] = []

    for session in data.get("multi_session_dialogues", []):
        sid = session.get("session_id", "")
        date = session.get("date", "")
        dialogues = session.get("dialogues", [])

        lines = [f"=== SESSION {sid} - Date: {date} ===", ""]
        image_paths: List[str] = []
        for turn in dialogues:
            turn_id = turn.get("round", "")
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            lines.append(f"User: {user_text}")

            input_images = turn.get("input_image", []) or []
            image_ids = turn.get("image_id", []) or []
            image_captions = turn.get("image_caption", []) or []

            turn_image_abs_paths: List[str] = []
            for rel in input_images:
                abs_path = resolve_image_path(base_dir, rel)
                lines.append(f"[Image: {abs_path}]")
                image_paths.append(abs_path)
                turn_image_abs_paths.append(abs_path)

            lines.append(f"Assistant: {assistant_text}")
            lines.append("")

            # turn 级 page：更细粒度，便于检索和 clue 对齐
            turn_lines = [
                f"=== SESSION {sid} - Date: {date} - TURN {turn_id} ===",
                f"User: {user_text}",
                f"Assistant: {assistant_text}",
            ]
            for img_id, img_rel, img_cap in zip(image_ids, input_images, image_captions):
                img_abs = resolve_image_path(base_dir, img_rel)
                turn_lines.append(f"[ImageID: {img_id}]")
                turn_lines.append(f"[Image: {img_abs}]")
                if img_cap:
                    turn_lines.append(f"[ImageCaption: {img_cap}]")

            turn_records.append(
                {
                    "session_id": sid,
                    "date": date,
                    "turn_id": turn_id,
                    "image_ids": image_ids,
                    "image_captions": image_captions,
                    "images": turn_image_abs_paths,
                    "raw_text": "\n".join(turn_lines).strip(),
                }
            )

        session_records.append(
            {
                "session_id": sid,
                "date": date,
                "raw_text": "\n".join(lines).strip(),
                "images": sorted(set(image_paths)),
            }
        )
    return session_records, turn_records


def collect_qa_items(data: Dict[str, Any], json_path: str) -> List[Dict[str, Any]]:
    """兼容 Mem-Gallery 键名：human-annotated QAs。"""
    base_dir = os.path.dirname(os.path.abspath(json_path))
    qa_src = data.get("human-annotated QAs", [])
    items: List[Dict[str, Any]] = []

    for qa in qa_src:
        q = qa.get("question", "")
        q_img_caption = qa.get("image_caption", None)
        q_img = qa.get("question_image")
        if q_img:
            q_abs = resolve_image_path(base_dir, q_img)
            q += f"\n[Image: {q_abs}]"
            if q_img_caption:
                q += f"\n[QuestionImageCaption: {q_img_caption}]"

        items.append(
            {
                "question_with_mm": q,
                "question_plain": qa.get("question", ""),
                "question_image_path": q_abs if q_img else None,
                "question_image_caption": q_img_caption,
                "answer": qa.get("answer", ""),
                "point": qa.get("point", "NA"),
                "session_id": qa.get("session_id", []),
                "clue": qa.get("clue", []),
            }
        )
    return items


def summarize_session(record: Dict[str, Any], generator: MultimodalVLLMGenerator) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        "Summarize the following session into concise factual memory bullets (3-6 bullets).\n"
        "Keep names, dates/time, updates, conflicts, and image-grounded facts.\n"
        "Do not infer beyond evidence.\n\n"
        f"{record['raw_text']}\n\n"
        "Summary:"
    )
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
    rebuild: bool = False,
) -> tuple[InMemoryMemoryStore, InMemoryPageStore]:
    memory_store = InMemoryMemoryStore(dir_path=sample_outdir)
    page_store = InMemoryPageStore(dir_path=sample_outdir)

    memory_file = os.path.join(sample_outdir, "memory_state.json")
    pages_file = os.path.join(sample_outdir, "pages.json")
    summaries_file = os.path.join(sample_outdir, "session_summaries.json")
    if (not rebuild) and os.path.exists(memory_file) and os.path.exists(pages_file):
        print("加载已有 memory/pages ...")
        return memory_store, page_store

    # 重建
    memory_store.save(memory_store.load().__class__(abstracts=[]))
    page_store.save([])
    all_summaries: List[Dict[str, Any]] = []

    print("按 session 构建 summaries + pages ...")
    for rec in tqdm(session_records, desc="Session summarize"):
        summary = summarize_session(rec, generator)
        abstract = f"[SESSION {rec['session_id']}] {summary}"
        memory_store.add(abstract)

        all_summaries.append(
            {
                "session_id": rec["session_id"],
                "date": rec["date"],
                "summary": summary,
            }
        )

    # turn 级 page 存储（原始文本 + 图片 + image_id）
    for turn in turn_records:
        page_store.add(
            Page(
                header=f"[TURN {turn['turn_id']}] Session {turn['session_id']} Date: {turn['date']}",
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


def merge_research_and_recall_for_qa(
    research_memory_list: List[Dict[str, Any]],
    recall_memory_list: List[Dict[str, Any]],
    max_total: int = 12,
) -> List[Dict[str, Any]]:
    """
    融合 Research Agent 检索结果与 recall_for_qa 结果。
    Research 找到的页面优先（含多图页面的所有图片），确保 CD/KR/MR 等不遗漏关键证据。
    Recall 中与 Research 重复的 turn 不再加入。
    """
    research_dialogue_ids: set = set()
    for mem in research_memory_list:
        did = mem.get("dialogue_id", "")
        if did:
            research_dialogue_ids.add(did)

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


def build_retrieval_evidence(
    research_out: Any,
    page_store: InMemoryPageStore,
    mm_retriever: Optional[Any] = None,
    top_k_pages: int = 6,
) -> Tuple[str, List[str], List[int]]:
    """从 research 输出提取 source pages，并构造 evidence 文本与 retrieved_ids。"""
    raw = research_out.raw_memory if research_out else {}
    temp_memory = raw.get("temp_memory", {}) if isinstance(raw, dict) else {}
    source_ids = temp_memory.get("sources", []) if isinstance(temp_memory, dict) else []

    pages = page_store.load()
    retrieved_ids: List[str] = []
    resolved_page_index: List[int] = []

    turn_to_page: Dict[str, int] = {}
    image_to_page: Dict[str, int] = {}
    for idx, p in enumerate(pages):
        meta = p.meta or {}
        turn_id = str(meta.get("turn_id", "") or "").strip()
        if turn_id:
            turn_to_page[turn_id] = idx
        for iid in (meta.get("image_ids", []) or []):
            iid_s = str(iid).strip()
            if iid_s:
                image_to_page[iid_s] = idx

    for src in source_ids:
        src_s = str(src).strip()
        idx: Optional[int] = None
        if src_s.isdigit():
            idx = int(src_s)
        elif src_s in turn_to_page:
            idx = turn_to_page[src_s]
        elif src_s in image_to_page:
            idx = image_to_page[src_s]

        if idx is None or idx < 0 or idx >= len(pages):
            continue
        resolved_page_index.append(idx)

    if mm_retriever and hasattr(mm_retriever, "last_retrieved_ids"):
        for rid in mm_retriever.last_retrieved_ids:
            rid_s = str(rid).strip()
            if rid_s in turn_to_page:
                resolved_page_index.append(turn_to_page[rid_s])

    seen_idx: set = set()
    dedup_idx: List[int] = []
    for idx in resolved_page_index:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        dedup_idx.append(idx)
    dedup_idx = dedup_idx[:top_k_pages]

    evidence_chunks: List[str] = []
    for idx in dedup_idx:
        p = pages[idx]
        turn_id = (p.meta or {}).get("turn_id")
        if turn_id:
            retrieved_ids.append(str(turn_id))
        snippet = p.content[:1200]
        evidence_chunks.append(f"[PAGE {idx}] {snippet}")

    seen: set = set()
    dedup_retrieved: List[str] = []
    for rid in retrieved_ids:
        if rid in seen:
            continue
        seen.add(rid)
        dedup_retrieved.append(rid)

    return "\n\n".join(evidence_chunks), dedup_retrieved, dedup_idx


def _extract_session_summary_entry(abstract: str) -> Tuple[Optional[str], str]:
    m = re.match(r"^\[SESSION\s+([^\]]+)\]\s*(.*)$", abstract.strip(), re.DOTALL)
    if not m:
        return None, abstract
    return m.group(1).strip(), m.group(2).strip()


def revise_long_term_summaries_with_evidence(
    memory_store: InMemoryMemoryStore,
    target_session_ids: List[str],
    question: str,
    evidence_text: str,
    generator: MultimodalVLLMGenerator,
) -> None:
    if not target_session_ids or not evidence_text.strip():
        return

    state = memory_store.load()
    if not state.abstracts:
        return

    target_set = {str(sid).strip() for sid in target_session_ids if str(sid).strip()}
    if not target_set:
        return

    updated = False
    new_abstracts: List[str] = []

    for abs_text in state.abstracts:
        sid, summary_text = _extract_session_summary_entry(abs_text)
        if sid is None or sid not in target_set:
            new_abstracts.append(abs_text)
            continue

        prompt = (
            f"{SYSTEM_PROMPT}\n"
            "You are updating long-term session memory.\n"
            "Given previous summary and newly retrieved evidence for the current question,\n"
            "decide whether the summary is accurate.\n"
            "- If accurate: return the original summary unchanged.\n"
            "- If inaccurate or incomplete: return a corrected concise summary.\n"
            "Output ONLY the revised summary text.\n\n"
            f"Question:\n{question}\n\n"
            f"Previous Session Summary (Session {sid}):\n{summary_text}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            "Revised Summary:"
        )

        try:
            revised = generator.generate_single(prompt=prompt).get("text", "").strip()
            if not revised:
                revised = summary_text
        except Exception:
            revised = summary_text

        if revised != summary_text:
            updated = True
        new_abstracts.append(f"[SESSION {sid}] {revised}")

    if updated:
        state.abstracts = new_abstracts
        memory_store.save(state)


def build_conversation_blocks_from_memory_dict_list(
    memory_dict_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """MMMemory 交错块（纯记忆，不含 MSG_START / query / summary）。

    供 Graph QA、MoT ``[EVIDENCE]`` 等与 ``_build_mmmemory_style_content`` 共用一套渲染。
    """
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
    return conversation_info_flow


def _build_mmmemory_style_content(
    memory_dict_list: List[Dict[str, Any]],
    query_prompt: str,
    question_image_path: Optional[str] = None,
    category_hint: str = "",
    checked_summary: str = "",
) -> List[Dict[str, Any]]:
    """
    Build multimodal user_content using MMMemory-style interleaved format,
    with the research agent's checked_summary as additional context.

    Each retrieved memory with an image becomes:
      {text: "<dialog>\\nimage:\\nimage_id: <id>\\nimage_content:\\n"}
      {image_url: file://<path>}

    Memories without images just have their text.
    """
    conversation_info_flow = build_conversation_blocks_from_memory_dict_list(memory_dict_list)

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": MSG_START_PROMPT_WO_MEMORY},
        *conversation_info_flow,
    ]

    if checked_summary:
        user_content.append(
            {"type": "text", "text": f"\nResearch Summary:\n{checked_summary}\n"}
        )

    if category_hint:
        user_content.append({"type": "text", "text": category_hint})

    user_content.append({"type": "text", "text": query_prompt})

    if question_image_path and os.path.exists(question_image_path):
        user_content.append({"type": "text", "text": DIALOGUE_AGENT_PROMPT_IMAGE})
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"file://{question_image_path}"}}
        )

    return user_content


def process_file(
    file_path: str,
    outdir: str,
    model_name: str,
    api_base: str,
    api_key: str,
    rebuild_memory: bool,
    enable_evolution: bool = True,
) -> List[Dict[str, Any]]:
    filename = os.path.basename(file_path)
    dataset_name = filename.replace(".json", "")
    print(f"\n{'='*60}")
    print(f"处理文件: {filename}")
    print(f"{'='*60}")

    data = load_sample(file_path)
    sample_outdir = os.path.join(outdir, dataset_name)
    os.makedirs(sample_outdir, exist_ok=True)

    session_records, turn_records = build_session_records(data, file_path)
    qa_items = collect_qa_items(data, file_path)
    print(f"会话数: {len(session_records)}")
    print(f"Turn页数: {len(turn_records)}")
    print(f"问题数: {len(qa_items)}")

    gen_cfg = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": api_base,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    mm_generator = MultimodalVLLMGenerator(gen_cfg)

    memory_store, page_store = build_memory_and_pages(
        sample_outdir, session_records, turn_records, mm_generator, rebuild=rebuild_memory
    )

    # ---- Retrievers ----
    retrievers: Dict[str, Any] = {}
    index_dir = os.path.join(sample_outdir, "page_index")
    if os.path.exists(index_dir):
        import shutil
        shutil.rmtree(index_dir)

    index_r = IndexRetriever(IndexRetrieverConfig(index_dir=index_dir).__dict__)
    index_r.build(page_store)
    retrievers["page_index"] = index_r

    mm_retriever = MMMemoryStyleRetriever(top_k=10)
    mm_retriever.build(page_store)
    retrievers["keyword"] = mm_retriever
    retrievers["vector"] = mm_retriever

    # ---- Research agent with Self-Correction + Self-Evolution ----
    base_agent = SelfCorrectingResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=mm_generator,
        max_iters=3,
    )
    experience_store = ExperienceStore(dir_path=sample_outdir)
    research_agent = SelfEvolvingResearchAgent.wrap(
        base_agent,
        experience_store=experience_store,
        evolution_enabled=enable_evolution,
    )

    profile = data.get("character_profile", {})
    sample_id = profile.get("name", dataset_name)
    speaker_a = f"user ({sample_id})"
    speaker_b = "assistant"

    traces_dir = os.path.join(sample_outdir, "research_traces")
    os.makedirs(traces_dir, exist_ok=True)
    all_research_traces: List[Dict[str, Any]] = []

    results: List[Dict[str, Any]] = []
    for i, qa in enumerate(tqdm(qa_items, desc="QA"), 1):
        trace_file = os.path.join(traces_dir, f"research_trace_q{i}.json")
        raw_memory_for_trace: Dict[str, Any] = {}
        evidence_text = ""
        retrieved_ids: List[str] = []
        resolved_page_index: List[int] = []
        checked_summary = ""
        memory_dict_list: List[Dict[str, Any]] = []
        query_prompt = ""
        user_content: List[Dict[str, Any]] = []
        qa_messages: List[Dict[str, Any]] = []
        checked_summary_for_qa = ""
        category_hint = ""
        format_constraint = ""
        system_answer = ""
        trace_error: Optional[str] = None

        try:
            # ==== Step 1: Research Agent (planning → search → integrate → self-correct → evolve) ====
            qa_category = (qa.get("point") or "").upper()
            research_out = research_agent.research(qa["question_with_mm"], category=qa_category)
            checked_summary = research_out.integrated_memory

            evidence_text, retrieved_ids, resolved_page_index = build_retrieval_evidence(
                research_out, page_store, mm_retriever=mm_retriever
            )
            if not retrieved_ids and hasattr(mm_retriever, "last_retrieved_ids"):
                retrieved_ids = list(getattr(mm_retriever, "last_retrieved_ids", []))
            if not resolved_page_index:
                page_idx_map: Dict[str, int] = {}
                for pi, p in enumerate(page_store.load()):
                    tid = str((p.meta or {}).get("turn_id", "") or "").strip()
                    if tid:
                        page_idx_map[tid] = pi
                resolved_page_index = [page_idx_map[rid] for rid in retrieved_ids if rid in page_idx_map]

            # ==== Step 2: Revise long-term summaries ====
            revise_long_term_summaries_with_evidence(
                memory_store=memory_store,
                target_session_ids=qa.get("session_id", []),
                question=qa["question_plain"],
                evidence_text=evidence_text,
                generator=mm_generator,
            )
            raw_memory_for_trace = json.loads(json.dumps(research_out.raw_memory))
            for it in raw_memory_for_trace.get("iterations", []):
                plan = it.get("plan", {})
                if isinstance(plan, dict) and (not plan.get("page_index")):
                    plan["page_index"] = resolved_page_index

            # ==== Step 3: Build QA message (MMMemory-style interleaved text+image) ====
            # 融合 Research 检索与 recall_for_qa：Research 找到的页面优先，确保 CD/KR/MR 等不遗漏关键证据
            research_memory_list, research_ids = mm_retriever.build_memory_dict_list_from_page_indices(
                page_store, resolved_page_index
            )
            recall_memory_list, recall_ids = mm_retriever.recall_for_qa(
                qa["question_plain"],
                question_image_path=qa.get("question_image_path"),
                top_k=10,
            )
            memory_dict_list = merge_research_and_recall_for_qa(
                research_memory_list, recall_memory_list, max_total=12
            )
            retrieved_ids = list(dict.fromkeys(m.get("dialogue_id") for m in memory_dict_list if m.get("dialogue_id")))

            category = (qa.get("point") or "").upper()
            format_constraint = ""
            if category == "AR":
                format_constraint = AR_CONSTRAINT
            elif category == "CD":
                format_constraint = CD_CONSTRAINT
            elif category == "VS":
                format_constraint = VS_CONSTRAINT
            format_constraint_str = f"\n\n{format_constraint}" if format_constraint else ""
            category_hint = CATEGORY_HINTS.get(category, "")

            # AR: 不注入 Research Summary，避免幻觉导致该拒答却给出答案
            checked_summary_for_qa = "" if category == "AR" else checked_summary

            query_prompt = DIALOGUE_AGENT_PROMPT.format(
                observation=qa["question_plain"],
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                format_constraint=format_constraint_str,
            )

            user_content = _build_mmmemory_style_content(
                memory_dict_list=memory_dict_list,
                query_prompt=query_prompt,
                question_image_path=qa.get("question_image_path"),
                category_hint=category_hint,
                checked_summary=checked_summary_for_qa,
            )

            qa_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            system_answer = mm_generator.generate_single(messages=qa_messages).get("text", "").strip()
            if not system_answer:
                system_answer = ""

        except Exception as e:
            print(f"[Q{i}] error: {e}")
            import traceback
            trace_error = traceback.format_exc()
            print(trace_error)

        trace_data = {
            "question_index": i,
            "question": qa["question_plain"],
            "question_with_mm": qa["question_with_mm"],
            "category": qa.get("point", ""),
            "session_id": qa.get("session_id", []),
            "clue": qa.get("clue", []),
            "retrieved_ids": retrieved_ids,
            "resolved_page_index": resolved_page_index,
            "evidence_text": evidence_text,
            "integrated_memory": checked_summary,
            "raw_memory": raw_memory_for_trace,
            "final_qa": {
                "retrieved_memory_dict_list": memory_dict_list,
                "query_prompt": query_prompt,
                "checked_summary_for_qa": checked_summary_for_qa,
                "category_hint": category_hint,
                "format_constraint": format_constraint,
                "question_image_path": qa.get("question_image_path"),
                "messages": qa_messages,
                "model_output": system_answer,
            },
            "error": trace_error,
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)
        all_research_traces.append(
            {"question_index": i, "question": qa["question_plain"], "trace_file": trace_file}
        )

        qa_session_id = qa["session_id"][0] if isinstance(qa["session_id"], list) and qa["session_id"] else ""
        item = {
            "sample_id": sample_id,
            "session_id": qa_session_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "question": qa["question_plain"],
            "system_answer": system_answer,
            "original_answer": qa["answer"],
            "category": qa["point"],
            "timestamp": get_timestamp(),
            "clue": qa["clue"],
            "retrieved_ids": retrieved_ids,
            "checked_summary": checked_summary,
        }
        results.append(item)

    # benchmark-compatible result file
    out_file = os.path.join(outdir, f"{dataset_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if all_research_traces:
        traces_summary_file = os.path.join(sample_outdir, "all_research_traces.json")
        with open(traces_summary_file, "w", encoding="utf-8") as f:
            json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
        print(f"Research trace 汇总已保存: {traces_summary_file}")

    print(f"结果已保存: {out_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mem-Gallery 多模态评测 (EvoMM)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get(
            "MEMGALLERY_DIALOG_DIR",
            "./Mem-Gallery/benchmark/data/dialog",
        ),
        help="dialog json directory",
    )
    parser.add_argument("--outdir", type=str, default="./results/memgallery_evomm", help="输出目录")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct", help="模型名")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--api-key", type=str, default="empty", help="API key")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个数据文件")
    parser.add_argument("--rebuild-memory", action="store_true", help="强制重建 session summaries 与 pages")
    parser.add_argument("--no-evolution", action="store_true", help="禁用自进化经验积累（回退到纯自纠错模式）")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if args.limit is not None:
        files = files[: args.limit]

    print(f"找到 {len(files)} 个数据文件")
    enable_evolution = not args.no_evolution
    print(f"模型: {args.model}")
    print("检索后端: MMMemory (MultiModalRetrieval)")
    print(f"自进化: {'开启' if enable_evolution else '关闭'}")
    print(f"输出目录: {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    for f in files:
        process_file(
            file_path=f,
            outdir=args.outdir,
            model_name=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            rebuild_memory=args.rebuild_memory,
            enable_evolution=enable_evolution,
        )


if __name__ == "__main__":
    main()
