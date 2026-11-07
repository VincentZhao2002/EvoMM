# EvoMM

Code for **EvoMM: Reinforced Self-Evolving Multimodal Agentic Memory**,
accepted at **NeurIPS 2026**.

A general memory system for agents, powered by deep-research.

## Project Layout

```
evomm/                          Core memory / research framework
├── agents/                     Memory, research, and evolving research agents
├── config/                     Generator and retriever configuration
├── generator/                  OpenAI / vLLM generator wrappers
├── prompts/                    Prompt templates (memory / research / experience)
├── retriever/                  BM25, dense, and index-based retrievers
└── schemas/                    Pydantic data models

eval/                           Benchmark runners and evaluation utilities
├── memgallery_test_mm.py       Mem-Gallery multimodal benchmark
├── memgallery_test_mm_graph.py Mem-Gallery + memory-graph variant
├── mmlongbench_test_mm.py      MMLongBench-Doc evaluation
├── mmlongbench_docqa_*.py      MMLongBench DocQA variants (graph / mem-gallery)
├── locomo_test_reflection.py   LoCoMo reflection benchmark
├── video_mme_eval.py           Video-MME evaluation
├── self_correcting_research.py Self-correcting research agent
└── mmmemory_retriever_adapter.py  Adapter exposing MMMemory as a retriever

scripts/                        Shell entry-points for the experiments above
docs/                           Design notes
```

## Installation

```bash
git clone https://github.com/VincentZhao2002/EvoMM.git
cd EvoMM
pip install -e .
```

Some optional features (vLLM serving, FAISS, BM25 via Pyserini) require
additional dependencies; see `requirements.txt`.

## Configuration

All filesystem locations are read from environment variables so that the code
can be run on any machine without modification. The most common variables are:

| Variable                       | Purpose                                     |
| ------------------------------ | ------------------------------------------- |
| `MEMGALLERY_BENCHMARK_ROOT`    | Root of the Mem-Gallery benchmark           |
| `MMLONGBENCH_DATA_ROOT`        | MMLongBench raw data directory              |
| `MMLONGBENCH_IMAGE_ROOT`       | MMLongBench page-image directory            |
| `MMLB_ROOT`                    | MMLongBench root (containing JSONL splits)  |
| `LOCAL_DENSE_MODEL_PATH`       | Local path to a dense embedding model       |
| `LOCAL_MINILM_PATH`            | Local path to all-MiniLM-L6-v2              |
| `LOCAL_BGE_PATH`               | Local path to bge-large-en                  |
| `EVOMM_HF_HOME`                | Optional override for `HF_HOME`             |
| `EVOMM_OUTDIR`                 | Default output directory                    |

## Quick Start

```bash
export MEMGALLERY_BENCHMARK_ROOT=/path/to/Mem-Gallery/benchmark
export EVOMM_OUTDIR=./results/memgallery_evomm

python eval/memgallery_test_mm.py \
  --data-dir "$MEMGALLERY_BENCHMARK_ROOT/data/dialog" \
  --outdir  "$EVOMM_OUTDIR" \
  --model qwen2.5-vl-7b-instruct \
  --api-base http://localhost:8000/v1 \
  --api-key empty
```

