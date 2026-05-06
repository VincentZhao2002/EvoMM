#!/bin/bash
# WorldMMMemory MMLongBench DocQA eval (longdocurl / mmlongdoc / slidevqa)
#
# 用法:
#   bash scripts/run_mmlongbench_docqa_worldmmmemory.sh
#
# 可选环境变量与 run_mmlongbench_docqa_graph.sh 基本一致:
#   MODEL=qwen3-vl-4b-instruct
#   API_BASE=http://localhost:8000/v1
#   API_KEY=empty
#   DATA_ROOT=./datasets/mmlongbench/mmlb_data
#   IMAGE_ROOT=./datasets/mmlongbench/mmlb_image
#   OUTDIR=<project>/results/mmlongbench_worldmmmemory_docqa
#   MMLB_ROOT=./MMLongBench/MMLongBench
#   NUM_WORKERS=1             # WorldMM/GME 构建较重，默认单线程
#   TOP_K=8                   # 每题送进 VLM 的最大页数
#   MAX_TOKENS=128
#   MAX_SAMPLES=100000
#   LIMIT=                    # 覆盖 MAX_SAMPLES，设了就只跑前 N 条
#   K_LIST="8 16 32 64 128"
#   SUBSETS="longdocurl mmlongdoc slidevqa"
#   NO_RETRIEVAL=0            # =1 时全量送页，不走 memory 检索
#   DOCQA_LLM_JUDGE=0
#   LLM_JUDGE_KEY=xxx
#   LLM_JUDGE_ENDPOINT=https://.../v1
#   LLM_JUDGE_MODEL=Doubao-Seed-1.8
#   FORCE_RERUN=0

set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-qwen3-vl-4b-instruct}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-empty}"
DATA_ROOT="${DATA_ROOT:-${MMLONGBENCH_DATA_ROOT:-./datasets/mmlongbench/mmlb_data}}"
IMAGE_ROOT="${IMAGE_ROOT:-${MMLONGBENCH_IMAGE_ROOT:-./datasets/mmlongbench/mmlb_image}}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/results/mmlongbench_worldmmmemory_docqa_${MODEL}}"
MMLB_ROOT="${MMLB_ROOT:-./MMLongBench/MMLongBench}"

NUM_WORKERS="${NUM_WORKERS:-4}"
TOP_K="${TOP_K:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
LIMIT="${LIMIT:-}"
K_LIST="${K_LIST:-16 32 64 128}"
SUBSETS="${SUBSETS:-longdocurl mmlongdoc slidevqa}"
NO_RETRIEVAL="${NO_RETRIEVAL:-0}"

DOCQA_LLM_JUDGE="${DOCQA_LLM_JUDGE:-0}"
LLM_JUDGE_KEY="${LLM_JUDGE_KEY:-}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Doubao-Seed-1.8}"

FORCE_RERUN="${FORCE_RERUN:-0}"

SCRIPT="${PROJECT_ROOT}/eval/mmlongbench_docqa_memgallery.py"
if [ ! -f "$SCRIPT" ]; then
  echo "[error] missing evaluator: $SCRIPT" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "[error] data root not found: $DATA_ROOT" >&2
  exit 1
fi
if [ ! -d "$IMAGE_ROOT" ]; then
  echo "[error] image root not found: $IMAGE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

echo "Memory       : WorldMMMemory"
echo "Model        : $MODEL"
echo "vLLM API     : $API_BASE"
echo "Data root    : $DATA_ROOT"
echo "Image root   : $IMAGE_ROOT"
echo "MMLB root    : $MMLB_ROOT"
echo "Out dir      : $OUTDIR"
echo "Workers      : $NUM_WORKERS (top_k=$TOP_K)"
echo "K list       : $K_LIST"
echo "Subsets      : $SUBSETS"
echo "No retrieval : $NO_RETRIEVAL"
echo ""

COMMON_ARGS=(
  --memory-name WorldMMMemory
  --image-root "$IMAGE_ROOT"
  --mmlongbench-root "$MMLB_ROOT"
  --model "$MODEL"
  --api-base "$API_BASE"
  --api-key "$API_KEY"
  --num-workers "$NUM_WORKERS"
  --top-k "$TOP_K"
  --max-tokens "$MAX_TOKENS"
  --max-samples "$MAX_SAMPLES"
)
if [ -n "$LIMIT" ]; then
  COMMON_ARGS+=(--limit "$LIMIT")
fi
if [ "$NO_RETRIEVAL" = "1" ]; then
  COMMON_ARGS+=(--no-retrieval)
fi
if [ "$DOCQA_LLM_JUDGE" = "1" ]; then
  if [ -z "$LLM_JUDGE_KEY" ] || [ -z "$LLM_JUDGE_ENDPOINT" ]; then
    echo "[error] DOCQA_LLM_JUDGE=1 requires LLM_JUDGE_KEY and LLM_JUDGE_ENDPOINT" >&2
    exit 1
  fi
  COMMON_ARGS+=(
    --docqa-llm-judge
    --llm-judge-key "$LLM_JUDGE_KEY"
    --llm-judge-endpoint "$LLM_JUDGE_ENDPOINT"
    --llm-judge-model "$LLM_JUDGE_MODEL"
  )
fi

for K in $K_LIST; do
  RUN_OUTDIR="${OUTDIR}/K${K}"
  mkdir -p "$RUN_OUTDIR"
  for sub in $SUBSETS; do
    DATA_FILE="${DATA_ROOT}/documentQA/${sub}_K${K}.jsonl"
    if [ ! -f "$DATA_FILE" ]; then
      echo "[skip] missing jsonl: $DATA_FILE"
      continue
    fi

    file_tag="$(basename "$DATA_FILE" .jsonl)"
    SCORE_FILE="${RUN_OUTDIR}/${file_tag}_worldmmmemory.json.score"
    CACHE_FILE="${RUN_OUTDIR}/${file_tag}_worldmmmemory.jsonl"
    if [ "$FORCE_RERUN" != "1" ] && [ -f "$SCORE_FILE" ]; then
      if [ -f "$CACHE_FILE" ]; then
        done_n=$(wc -l < "$CACHE_FILE" 2>/dev/null || echo 0)
      else
        done_n="?"
      fi
      echo "[skip] ${sub} K=${K}: already complete (done=${done_n}, score=${SCORE_FILE})"
      continue
    fi

    echo "========== DocQA[WorldMMMemory] ${sub} K=${K} =========="
    python "$SCRIPT" \
      --data-file "$DATA_FILE" \
      --outdir "$RUN_OUTDIR" \
      "${COMMON_ARGS[@]}"
  done
done

echo ""
echo "全部完成: 结果根目录 $OUTDIR"
