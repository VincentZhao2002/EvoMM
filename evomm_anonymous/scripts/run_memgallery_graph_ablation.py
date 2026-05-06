#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run single-factor Mem-Gallery ablations for the graph-enhanced evaluator.

This runner intentionally leaves eval/memgallery_test_mm_graph.py unchanged.
It calls process_file_graph() with different switches and applies a temporary
agent-class patch for the no_reflection variant.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVOMM_ROOT = os.path.dirname(SCRIPT_DIR)
ZT_ROOT = os.path.dirname(EVOMM_ROOT)
EVAL_DIR = os.path.join(EVOMM_ROOT, "eval")

for path in (EVOMM_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


DEFAULT_VARIANTS = "full,no_graph,no_reflection,no_evolution"
ALLOWED_VARIANTS = {"full", "no_graph", "no_reflection", "no_evolution"}
DEFAULT_LOCAL_MINILM_PATH = "./models/all-MiniLM-L6-v2"


def _load_graph_eval():
    return importlib.import_module("memgallery_test_mm_graph")


def _default_data_dir() -> str:
    return os.path.join(ZT_ROOT, "Mem-Gallery", "benchmark", "data", "dialog")


def _default_embedding_model(
    local_path: str = DEFAULT_LOCAL_MINILM_PATH,
) -> str:
    env = os.environ.get("MEMGALLERY_EMBEDDING_MODEL")
    if env:
        return env
    return local_path if os.path.isdir(local_path) else "all-MiniLM-L6-v2"


def _parse_variants(raw: str) -> List[str]:
    variants = [v.strip() for v in (raw or "").split(",") if v.strip()]
    if not variants:
        raise ValueError("No variants specified.")
    unknown = [v for v in variants if v not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_VARIANTS))}"
        )
    return variants


def _list_data_files(data_dir: str, limit: int | None) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if limit is not None:
        files = files[:limit]
    return files


def _variant_runtime_config(args: argparse.Namespace, variant: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "variant": variant,
        "graph_enabled": True,
        "reflection_enabled": True,
        "evolution_enabled": True,
        "graph_extra_topk": args.graph_extra_topk,
        "research_agent_patch": None,
    }

    if variant == "no_graph":
        config["graph_enabled"] = False
        config["graph_extra_topk"] = 0
    elif variant == "no_reflection":
        config["reflection_enabled"] = False
        config["research_agent_patch"] = (
            "memgallery_test_mm_graph.SelfCorrectingResearchAgent -> evomm.ResearchAgent"
        )
    elif variant == "no_evolution":
        config["evolution_enabled"] = False

    return config


def _write_ablation_config(
    variant_outdir: str,
    args: argparse.Namespace,
    variant_config: Dict[str, Any],
    files: List[str],
) -> None:
    os.makedirs(variant_outdir, exist_ok=True)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant_config["variant"],
        "switches": {
            "graph_enabled": variant_config["graph_enabled"],
            "reflection_enabled": variant_config["reflection_enabled"],
            "evolution_enabled": variant_config["evolution_enabled"],
            "research_agent_patch": variant_config["research_agent_patch"],
        },
        "model": {
            "model_name": args.model,
            "api_base": args.api_base,
            "api_key": args.api_key,
        },
        "data": {
            "data_dir": args.data_dir,
            "limit": args.limit,
            "num_files": len(files),
            "files": files,
        },
        "memory": {
            "rebuild_memory": args.rebuild_memory,
        },
        "graph": {
            "rebuild_graph": args.rebuild_graph,
            "graph_extra_topk": variant_config["graph_extra_topk"],
            "base_graph_extra_topk": args.graph_extra_topk,
            "semantic_topk": args.semantic_topk,
            "semantic_min_sim": args.semantic_min_sim,
            "semantic_builder": args.semantic_builder,
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
        },
    }
    path = os.path.join(variant_outdir, "ablation_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@contextmanager
def _maybe_patch_research_agent(graph_eval: Any, disable_reflection: bool) -> Iterator[None]:
    original_cls = graph_eval.SelfCorrectingResearchAgent
    if disable_reflection:
        from evomm import ResearchAgent

        graph_eval.SelfCorrectingResearchAgent = ResearchAgent
    try:
        yield
    finally:
        graph_eval.SelfCorrectingResearchAgent = original_cls


def _run_variant(
    args: argparse.Namespace,
    variant: str,
    files: List[str],
    graph_eval: Any,
) -> None:
    variant_config = _variant_runtime_config(args, variant)
    variant_outdir = os.path.join(args.outdir, variant)
    _write_ablation_config(variant_outdir, args, variant_config, files)

    print("\n" + "=" * 80)
    print(f"Running variant: {variant}")
    print(f"Output dir: {variant_outdir}")
    print(
        "Switches: "
        f"graph={variant_config['graph_enabled']}, "
        f"reflection={variant_config['reflection_enabled']}, "
        f"evolution={variant_config['evolution_enabled']}"
    )
    print("=" * 80)

    shared_mm_retriever = graph_eval.MMMemoryStyleRetriever(top_k=10) if files else None
    with _maybe_patch_research_agent(
        graph_eval=graph_eval,
        disable_reflection=not variant_config["reflection_enabled"]
    ):
        for file_path in files:
            graph_eval.process_file_graph(
                file_path=file_path,
                outdir=variant_outdir,
                model_name=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                rebuild_memory=args.rebuild_memory,
                mm_retriever=shared_mm_retriever,
                rebuild_graph=args.rebuild_graph,
                graph_extra_topk=variant_config["graph_extra_topk"],
                semantic_topk=args.semantic_topk,
                semantic_min_sim=args.semantic_min_sim,
                semantic_builder=args.semantic_builder,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                enable_evolution=variant_config["evolution_enabled"],
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mem-Gallery graph evaluator ablations."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=_default_data_dir(),
        help="Directory containing Mem-Gallery dialog json files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(EVOMM_ROOT, "results", "memgallery_ablation"),
        help="Root output directory. Each variant writes to a subdirectory.",
    )
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="empty")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        type=str,
        default=DEFAULT_VARIANTS,
        help=f"Comma-separated variants. Default: {DEFAULT_VARIANTS}",
    )
    parser.add_argument("--rebuild-memory", action="store_true")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument(
        "--graph-extra-topk",
        type=int,
        default=4,
        help="Extra graph-expanded candidates for the full setting.",
    )
    parser.add_argument("--semantic-topk", type=int, default=4)
    parser.add_argument("--semantic-min-sim", type=float, default=0.2)
    parser.add_argument(
        "--semantic-builder",
        type=str,
        default="embedding",
        choices=["embedding", "lexical"],
    )
    parser.add_argument("--embedding-backend", type=str, default="sentence_transformers")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=_default_embedding_model(),
        help="Embedding model name/path for graph construction.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        variants = _parse_variants(args.variants)
    except ValueError as exc:
        parser.error(str(exc))

    files = _list_data_files(args.data_dir, args.limit)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Found {len(files)} data file(s).")
    print(f"Variants: {', '.join(variants)}")
    print(f"Model: {args.model}")
    print(f"Output root: {args.outdir}")

    graph_eval = _load_graph_eval()
    for variant in variants:
        _run_variant(args, variant, files, graph_eval)


if __name__ == "__main__":
    main()
