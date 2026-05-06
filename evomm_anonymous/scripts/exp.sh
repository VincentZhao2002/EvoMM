
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOMM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZT_ROOT="$(cd "$EVOMM_ROOT/.." && pwd)"

API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-4b-instruct-299}"

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
