#!/bin/bash
# Mem-Gallery 多模态图评测；路径相对本仓库，可与 start_qwen3_vl_vllm.sh 默认服务对齐。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOMM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZT_ROOT="$(cd "$EVOMM_ROOT/.." && pwd)"

API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-4b-instruct-299}"

# 多卡且 vLLM 已占 0–3、评测要用 4–7 时取消注释下一行（与 Mem-Gallery DefaultGlobalConfig 旧约定一致）：
# export MEMGALLERY_USABLE_GPU=4,5,6,7
# GME 编码器目录（可选，默认见 Mem-Gallery DefaultGlobalConfig 的 GME_MODEL_PATH）：
# export GME_MODEL_PATH=./models/gme-qwen2-vl-2b
#
# 图构图 embedding：默认用本机已下载目录（不再访问 huggingface.co）。覆盖示例：
#   export MEMGALLERY_EMBEDDING_MODEL=/其它路径/all-MiniLM-L6-v2
MEMGALLERY_EMBEDDING_MODEL="${MEMGALLERY_EMBEDDING_MODEL:-./models/all-MiniLM-L6-v2}"
export MEMGALLERY_EMBEDDING_MODEL

GME_DEVICE="${GME_DEVICE:-cuda:0}" \
python "${EVOMM_ROOT}/eval/memgallery_test_mm_graph.py" \
  --data-dir "${ZT_ROOT}/Mem-Gallery/benchmark/data/dialog" \
  --outdir "${EVOMM_ROOT}/results/memgallery_evomm_graph_sem_qwen3_rl_299" \
  --model "${MODEL_NAME}" \
  --api-base "${API_BASE}" \
  --api-key "empty" \
  --semantic-builder embedding \
  --embedding-backend sentence_transformers \
  --embedding-model "${MEMGALLERY_EMBEDDING_MODEL}" \
  --graph-extra-topk 2 \
  --semantic-topk 2 \
  --semantic-min-sim 0.25
