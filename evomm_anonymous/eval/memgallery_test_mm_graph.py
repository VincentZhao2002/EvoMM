#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMM + Mem-Gallery 评测（Graph-Enhanced 版本）

说明：
- 不修改原始 memgallery_test_mm.py
- 复用原始脚本的 memory/research/qa 主流程
- 新增图结构索引（turn-level graph）用于扩展检索候选
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import numpy as np

import memgallery_test_mm as base
from eval.mmmemory_retriever_adapter import MMMemoryStyleRetriever
from eval.self_correcting_research import SelfCorrectingResearchAgent

STOPWORDS = {
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "do", "did", "does", "to", "for", "from",
    "and", "or", "the", "a", "an", "in", "on", "at", "of", "with", "by",
    "this", "that", "these", "those", "it", "they", "he", "she", "we", "you",
    "i", "me", "my", "our", "your", "their", "his", "her", "as", "be", "been",
}

DEFAULT_LOCAL_MINILM_PATH = os.environ.get(
    "LOCAL_MINILM_PATH", "./models/all-MiniLM-L6-v2"
)


def default_memgallery_embedding_model(
    local_path: str = DEFAULT_LOCAL_MINILM_PATH,
) -> str:
    """优先环境变量 ``MEMGALLERY_EMBEDDING_MODEL``；否则若本地目录存在则用离线模型，否则 Hub 短名。"""
    env = os.environ.get("MEMGALLERY_EMBEDDING_MODEL")
    if env:
        return env
    return local_path if os.path.isdir(local_path) else "all-MiniLM-L6-v2"


def _clean_text_for_graph(raw_text: str) -> str:
    text = re.sub(r"^===\s.*===\s*$", "", raw_text, flags=re.MULTILINE)
    text = re.sub(r"^\[(Image|ImageID|ImageCaption|QuestionImageCaption):.*\]\s*$", "", text, flags=re.MULTILINE)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


def _extract_query_terms(question: str) -> List[str]:
    q = re.sub(r"\[Image: .*?\]", " ", question or "")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", q.lower())
    out: List[str] = []
    seen: Set[str] = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _extract_text_terms(text: str, max_terms: int = 64) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", (text or "").lower())
    out: List[str] = []
    seen: Set[str] = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _compute_text_embeddings(
    texts: List[str],
    backend: str = "sentence_transformers",
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[np.ndarray]:
    """
    Compute dense embeddings for page texts.
    Returns None if backend unavailable; caller should fallback to lexical similarity.
    """
    if not texts:
        return None
    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        except Exception as e:
            err = str(e).lower()
            net = any(
                x in err
                for x in ("network", "unreachable", "connection", "max retries", "timed out", "443")
            )
            hint = ""
            if net:
                hint = (
                    "\n  [离线/内网] embedding 需从 Hub 拉取。可选:\n"
                    "    • 镜像: export HF_ENDPOINT=https://hf-mirror.com（视环境是否可达）\n"
                    "    • 本地目录: 先 download 再 export MEMGALLERY_EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2\n"
                    "    • 或不用向量边: --semantic-builder lexical\n"
                )
            print(f"[Graph] sentence-transformers backend unavailable, fallback to lexical edges: {e}{hint}")
            return None
    print(f"[Graph] Unknown embedding backend: {backend}, fallback to lexical edges.")
    return None


def build_turn_graph(
    page_store: Any,
    outdir: str,
    rebuild: bool = False,
    semantic_topk: int = 6,
    semantic_min_sim: float = 0.12,
    semantic_builder: str = "embedding",
    embedding_backend: str = "sentence_transformers",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    graph_file = os.path.join(outdir, "graph_index.json")
    if (not rebuild) and os.path.exists(graph_file):
        with open(graph_file, "r", encoding="utf-8") as f:
            return json.load(f)

    pages = page_store.load()
    turn_to_page: Dict[str, int] = {}
    page_to_turn: Dict[str, str] = {}
    page_to_session: Dict[str, str] = {}
    session_to_pages: Dict[str, List[int]] = defaultdict(list)
    image_to_pages: Dict[str, List[int]] = defaultdict(list)
    keyword_to_pages: Dict[str, List[int]] = defaultdict(list)
    next_edges: Dict[str, List[int]] = defaultdict(list)
    prev_edges: Dict[str, List[int]] = defaultdict(list)
    page_terms: Dict[int, Set[str]] = {}
    semantic_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    clean_texts: List[str] = []

    for idx, p in enumerate(pages):
        meta = p.meta or {}
        turn_id = str(meta.get("turn_id", "") or "").strip()
        session_id = str(meta.get("session_id", "") or "").strip()
        key = str(idx)

        if turn_id:
            turn_to_page[turn_id] = idx
            page_to_turn[key] = turn_id
        if session_id:
            page_to_session[key] = session_id
            session_to_pages[session_id].append(idx)

        for iid in (meta.get("image_ids", []) or []):
            iid_s = str(iid).strip()
            if iid_s:
                image_to_pages[iid_s].append(idx)

        clean = _clean_text_for_graph(p.content)
        clean_texts.append(clean)
        terms = _extract_text_terms(clean)
        page_terms[idx] = set(terms)
        for kw in terms:
            keyword_to_pages[kw].append(idx)

    # Build temporal edges within each session by page index order.
    for sid, idxs in session_to_pages.items():
        sorted_idxs = sorted(set(idxs))
        for i in range(len(sorted_idxs) - 1):
            a, b = sorted_idxs[i], sorted_idxs[i + 1]
            next_edges[str(a)].append(b)
            prev_edges[str(b)].append(a)

    # Build semantic edges
    if semantic_builder == "embedding":
        embs = _compute_text_embeddings(
            clean_texts,
            backend=embedding_backend,
            model_name=embedding_model,
        )
    else:
        embs = None

    if embs is not None and len(embs) == len(pages):
        sim_mat = np.matmul(_l2_normalize(embs), _l2_normalize(embs).T)
        np.fill_diagonal(sim_mat, -1.0)
        for src_idx in range(len(pages)):
            row = sim_mat[src_idx]
            cand_idx = np.argpartition(-row, min(semantic_topk, len(row) - 1))[:semantic_topk + 1]
            scored: List[Tuple[int, float]] = []
            for j in cand_idx:
                sim = float(row[j])
                if j == src_idx or sim < semantic_min_sim:
                    continue
                scored.append((int(j), sim))
            scored.sort(key=lambda x: (-x[1], -x[0]))
            for j, sim in scored[:semantic_topk]:
                semantic_edges[str(src_idx)].append({"to": j, "sim": round(sim, 4)})
    else:
        # Fallback: lexical overlap (Jaccard) among keyword-sharing pages.
        for src_idx, src_terms in page_terms.items():
            if not src_terms:
                continue
            candidate_overlap: Dict[int, int] = defaultdict(int)
            for t in list(src_terms)[:40]:
                for cand in keyword_to_pages.get(t, [])[:80]:
                    if cand == src_idx:
                        continue
                    candidate_overlap[cand] += 1

            scored: List[Tuple[int, float]] = []
            for cand_idx, inter_hint in candidate_overlap.items():
                cand_terms = page_terms.get(cand_idx, set())
                if not cand_terms:
                    continue
                inter = inter_hint if inter_hint > 0 else len(src_terms & cand_terms)
                if inter == 0:
                    continue
                union = len(src_terms | cand_terms)
                sim = inter / union if union > 0 else 0.0
                if sim >= semantic_min_sim:
                    scored.append((cand_idx, sim))

            scored.sort(key=lambda x: (-x[1], -x[0]))
            for cand_idx, sim in scored[:semantic_topk]:
                semantic_edges[str(src_idx)].append({"to": cand_idx, "sim": round(sim, 4)})

    graph = {
        "turn_to_page": turn_to_page,
        "page_to_turn": page_to_turn,
        "page_to_session": page_to_session,
        "session_to_pages": dict(session_to_pages),
        "image_to_pages": dict(image_to_pages),
        "keyword_to_pages": dict(keyword_to_pages),
        "next_edges": dict(next_edges),
        "prev_edges": dict(prev_edges),
        "semantic_edges": dict(semantic_edges),
        "semantic_builder": semantic_builder if embs is not None else "lexical_fallback",
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "page_count": len(pages),
    }
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return graph


def expand_candidates_with_graph(
    graph: Dict[str, Any],
    question: str,
    category: str,
    seed_pages: List[int],
    max_extra: int = 8,
    allow_cross_session: bool = False,
) -> List[int]:
    cat = (category or "").upper()
    scores: Dict[int, float] = defaultdict(float)
    page_count = int(graph.get("page_count", 0))
    seed_set = [p for p in seed_pages if isinstance(p, int) and 0 <= p < page_count]
    page_to_session = graph.get("page_to_session", {})
    seed_sessions = {str(page_to_session.get(str(p), "")) for p in seed_set if page_to_session.get(str(p), "")}

    for p in seed_set:
        scores[p] += 3.0

    # 1-hop temporal expansion
    next_edges = graph.get("next_edges", {})
    prev_edges = graph.get("prev_edges", {})
    semantic_edges = graph.get("semantic_edges", {})
    hop_boost = 1.2 if cat in {"KR", "TR", "MR"} else 0.6
    for p in seed_set:
        for nb in next_edges.get(str(p), []):
            if 0 <= int(nb) < page_count:
                if (not allow_cross_session) and seed_sessions:
                    nb_sid = str(page_to_session.get(str(int(nb)), ""))
                    if nb_sid and nb_sid not in seed_sessions:
                        continue
                scores[int(nb)] += hop_boost
        for nb in prev_edges.get(str(p), []):
            if 0 <= int(nb) < page_count:
                if (not allow_cross_session) and seed_sessions:
                    nb_sid = str(page_to_session.get(str(int(nb)), ""))
                    if nb_sid and nb_sid not in seed_sessions:
                        continue
                scores[int(nb)] += hop_boost * 0.7

    # semantic expansion
    sem_boost = 1.0 if cat in {"MR", "FR", "CD", "KR"} else 0.5
    for p in seed_set:
        for edge in semantic_edges.get(str(p), []):
            try:
                nb = int(edge.get("to"))
                sim = float(edge.get("sim", 0.0))
            except Exception:
                continue
            if 0 <= nb < page_count and sim > 0:
                if (not allow_cross_session) and seed_sessions:
                    nb_sid = str(page_to_session.get(str(nb), ""))
                    if nb_sid and nb_sid not in seed_sessions:
                        continue
                scores[nb] += sem_boost * sim

    # keyword expansion
    keyword_to_pages = graph.get("keyword_to_pages", {})
    for kw in _extract_query_terms(question)[:12]:
        for p in keyword_to_pages.get(kw, [])[:20]:
            if 0 <= int(p) < page_count:
                if (not allow_cross_session) and seed_sessions:
                    p_sid = str(page_to_session.get(str(int(p)), ""))
                    if p_sid and p_sid not in seed_sessions:
                        continue
                scores[int(p)] += 0.4

    # image-id expansion for VS
    image_to_pages = graph.get("image_to_pages", {})
    image_ids = re.findall(r"D\d+:\s*IMG_\d+", question or "", flags=re.IGNORECASE)
    for iid in image_ids:
        key = re.sub(r"\s+", "", iid.upper())
        for p in image_to_pages.get(key, []):
            if 0 <= int(p) < page_count:
                if (not allow_cross_session) and seed_sessions:
                    p_sid = str(page_to_session.get(str(int(p)), ""))
                    if p_sid and p_sid not in seed_sessions:
                        continue
                scores[int(p)] += 1.5

    ordered = sorted(scores.items(), key=lambda x: (-x[1], -x[0]))
    out: List[int] = []
    seed_seen = set(seed_set)
    for p, _ in ordered:
        if p in seed_seen:
            continue
        out.append(p)
        if len(out) >= max_extra:
            break
    return out


def process_file_graph(
    file_path: str,
    outdir: str,
    model_name: str,
    api_base: str,
    api_key: str,
    rebuild_memory: bool,
    mm_retriever: Optional[Any] = None,
    rebuild_graph: bool = False,
    graph_extra_topk: int = 8,
    semantic_topk: int = 6,
    semantic_min_sim: float = 0.12,
    semantic_builder: str = "embedding",
    embedding_backend: str = "sentence_transformers",
    embedding_model: str = "all-MiniLM-L6-v2",
    enable_evolution: bool = True,
) -> List[Dict[str, Any]]:
    filename = os.path.basename(file_path)
    dataset_name = filename.replace(".json", "")
    print(f"\n{'='*60}")
    print(f"处理文件(Graph): {filename}")
    print(f"{'='*60}")

    data = base.load_sample(file_path)
    sample_outdir = os.path.join(outdir, dataset_name)
    os.makedirs(sample_outdir, exist_ok=True)

    session_records, turn_records = base.build_session_records(data, file_path)
    qa_items = base.collect_qa_items(data, file_path)

    gen_cfg = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": api_base,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    mm_generator = base.MultimodalVLLMGenerator(gen_cfg)

    memory_store, page_store = base.build_memory_and_pages(
        sample_outdir, session_records, turn_records, mm_generator, rebuild=rebuild_memory
    )
    graph = build_turn_graph(
        page_store,
        sample_outdir,
        rebuild=rebuild_graph,
        semantic_topk=semantic_topk,
        semantic_min_sim=semantic_min_sim,
        semantic_builder=semantic_builder,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    retrievers: Dict[str, Any] = {}
    index_dir = os.path.join(sample_outdir, "page_index")
    if os.path.exists(index_dir):
        import shutil
        shutil.rmtree(index_dir)
    index_r = base.IndexRetriever(base.IndexRetrieverConfig(index_dir=index_dir).__dict__)
    index_r.build(page_store)
    retrievers["page_index"] = index_r

    if mm_retriever is None:
        mm_retriever = MMMemoryStyleRetriever(top_k=10)
    else:
        mm_retriever.reset()
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
    experience_store = base.ExperienceStore(dir_path=sample_outdir)
    research_agent = base.SelfEvolvingResearchAgent.wrap(
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
    results: List[Dict[str, Any]] = []

    for i, qa in enumerate(tqdm(qa_items, desc="QA(Graph)"), 1):
        trace_file = os.path.join(traces_dir, f"research_trace_q{i}.json")
        trace_error: Optional[str] = None
        raw_memory_for_trace: Dict[str, Any] = {}
        evidence_text = ""
        retrieved_ids: List[str] = []
        resolved_page_index: List[int] = []
        graph_expanded_pages: List[int] = []
        memory_dict_list: List[Dict[str, Any]] = []
        checked_summary = ""
        system_answer = ""

        try:
            qa_category = (qa.get("point") or "").upper()
            research_out = research_agent.research(qa["question_with_mm"], category=qa_category)
            checked_summary = research_out.integrated_memory

            evidence_text, retrieved_ids, resolved_page_index = base.build_retrieval_evidence(
                research_out, page_store, mm_retriever=mm_retriever
            )

            # 根据证据修订长期记忆（session summaries）
            base.revise_long_term_summaries_with_evidence(
                memory_store=memory_store,
                target_session_ids=qa.get("session_id", []),
                question=qa["question_plain"],
                evidence_text=evidence_text,
                generator=mm_generator,
            )

            raw_memory_for_trace = json.loads(json.dumps(research_out.raw_memory))

            recall_memory_list, _ = mm_retriever.recall_for_qa(
                qa["question_plain"],
                question_image_path=qa.get("question_image_path"),
                top_k=10,
            )
            recall_ids = [m.get("dialogue_id") for m in recall_memory_list if m.get("dialogue_id")]
            seed_pages = list(resolved_page_index)
            for rid in recall_ids:
                pidx = graph.get("turn_to_page", {}).get(str(rid))
                if isinstance(pidx, int):
                    seed_pages.append(pidx)
            seed_pages = list(dict.fromkeys(seed_pages))

            # Enable graph expansion for all categories.
            # KR allows cross-session expansion; others keep session-local expansion.
            allow_cross_session = qa_category == "KR"
            graph_expanded_pages = expand_candidates_with_graph(
                graph=graph,
                question=qa["question_plain"],
                category=qa_category,
                seed_pages=seed_pages,
                max_extra=graph_extra_topk,
                allow_cross_session=allow_cross_session,
            )

            graph_pages = seed_pages + graph_expanded_pages
            graph_pages = list(dict.fromkeys(graph_pages))

            research_memory_list, _ = mm_retriever.build_memory_dict_list_from_page_indices(
                page_store, graph_pages
            )
            max_total = 10
            memory_dict_list = base.merge_research_and_recall_for_qa(
                research_memory_list, recall_memory_list, max_total=max_total
            )
            retrieved_ids = list(
                dict.fromkeys(m.get("dialogue_id") for m in memory_dict_list if m.get("dialogue_id"))
            )

            category = (qa.get("point") or "").upper()
            format_constraint = ""
            if category == "AR":
                format_constraint = base.AR_CONSTRAINT
            elif category == "CD":
                format_constraint = base.CD_CONSTRAINT
            elif category == "VS":
                format_constraint = base.VS_CONSTRAINT
            format_constraint_str = f"\n\n{format_constraint}" if format_constraint else ""
            category_hint = base.CATEGORY_HINTS.get(category, "")
            checked_summary_for_qa = "" if category == "AR" else checked_summary

            query_prompt = base.DIALOGUE_AGENT_PROMPT.format(
                observation=qa["question_plain"],
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                format_constraint=format_constraint_str,
            )
            user_content = base._build_mmmemory_style_content(
                memory_dict_list=memory_dict_list,
                query_prompt=query_prompt,
                question_image_path=qa.get("question_image_path"),
                category_hint=category_hint,
                checked_summary=checked_summary_for_qa,
            )
            qa_messages = [
                {"role": "system", "content": base.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            system_answer = mm_generator.generate_single(messages=qa_messages).get("text", "").strip() or ""

        except Exception as e:
            import traceback
            trace_error = traceback.format_exc()
            print(f"[Q{i}] error: {e}")
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
            "graph_expanded_pages": graph_expanded_pages,
            "evidence_text": evidence_text,
            "integrated_memory": checked_summary,
            "raw_memory": raw_memory_for_trace,
            "final_qa": {
                "retrieved_memory_dict_list": memory_dict_list,
                "model_output": system_answer,
            },
            "error": trace_error,
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)

        qa_session_id = qa["session_id"][0] if isinstance(qa["session_id"], list) and qa["session_id"] else ""
        results.append(
            {
                "sample_id": sample_id,
                "session_id": qa_session_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": qa["question_plain"],
                "system_answer": system_answer,
                "original_answer": qa["answer"],
                "category": qa["point"],
                "timestamp": base.get_timestamp(),
                "clue": qa["clue"],
                "retrieved_ids": retrieved_ids,
                "checked_summary": checked_summary,
            }
        )

    out_file = os.path.join(outdir, f"{dataset_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mem-Gallery 多模态评测 (EvoMM Graph)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get(
            "MEMGALLERY_DIALOG_DIR",
            "./Mem-Gallery/benchmark/data/dialog",
        ),
        help="dialog json directory",
    )
    parser.add_argument("--outdir", type=str, default="./results/memgallery_evomm_graph", help="输出目录")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct", help="模型名")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--api-key", type=str, default="empty", help="API key")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个数据文件")
    parser.add_argument("--rebuild-memory", action="store_true", help="强制重建 session summaries 与 pages")
    parser.add_argument("--rebuild-graph", action="store_true", help="强制重建 graph_index.json")
    parser.add_argument("--graph-extra-topk", type=int, default=4, help="图扩展额外候选数量")
    parser.add_argument("--semantic-topk", type=int, default=4, help="每个page保留的语义邻居数量")
    parser.add_argument("--semantic-min-sim", type=float, default=0.2, help="语义边最小相似度阈值")
    parser.add_argument("--semantic-builder", type=str, default="embedding", choices=["embedding", "lexical"], help="语义边构建方式")
    parser.add_argument("--embedding-backend", type=str, default="sentence_transformers", help="embedding 后端")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=default_memgallery_embedding_model(),
        help=f"embedding：本机默认 {DEFAULT_LOCAL_MINILM_PATH}（若存在）；否则 all-MiniLM-L6-v2；可 export MEMGALLERY_EMBEDDING_MODEL",
    )
    parser.add_argument("--no-evolution", action="store_true", help="禁用自进化经验积累")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if args.limit is not None:
        files = files[: args.limit]

    enable_evolution = not args.no_evolution
    os.makedirs(args.outdir, exist_ok=True)
    print(f"找到 {len(files)} 个数据文件")
    print(f"模型: {args.model}")
    print(f"图扩展额外候选: {args.graph_extra_topk}")
    print(f"语义边: topk={args.semantic_topk}, min_sim={args.semantic_min_sim}")
    print(f"语义构图: builder={args.semantic_builder}, backend={args.embedding_backend}, model={args.embedding_model}")
    print(f"输出目录: {args.outdir}")

    # 复用 MMMemoryStyleRetriever（含 GME 模型），避免每个 json 文件重复加载
    shared_mm_retriever = MMMemoryStyleRetriever(top_k=10) if files else None

    for f in files:
        process_file_graph(
            file_path=f,
            outdir=args.outdir,
            model_name=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            rebuild_memory=args.rebuild_memory,
            mm_retriever=shared_mm_retriever,
            rebuild_graph=args.rebuild_graph,
            graph_extra_topk=args.graph_extra_topk,
            semantic_topk=args.semantic_topk,
            semantic_min_sim=args.semantic_min_sim,
            semantic_builder=args.semantic_builder,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            enable_evolution=enable_evolution,
        )


if __name__ == "__main__":
    main()

