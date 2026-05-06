#!/bin/bash

set -euo pipefail

# 只跑 MMLongBench 的 vrag + documentQA(docvqa) 两类（仅 K32）
# 用法:
# bash scripts/run_mmlongbench_vrag_docqa.sh
# 可选环境变量:
#   MODEL=qwen2.5-vl-7b-instruct
#   API_BASE=http://localhost:8000/v1
#   API_KEY=empty
#   DATA_ROOT=/path/to/mmlb_data
#   IMAGE_ROOT=/path/to/mmlb_image
#   OUTDIR=/path/to/output
#   MAX_ROUNDS=3
#   MIN_ROUNDS=2
#   TOP_K=6
#   LIMIT=

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-qwen3-vl-4b-instruct-199}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-empty}"
DATA_ROOT="${DATA_ROOT:-${MMLONGBENCH_DATA_ROOT:-"./datasets/mmlongbench/mmlb_data"}}"
IMAGE_ROOT="${IMAGE_ROOT:-${MMLONGBENCH_IMAGE_ROOT:-"./datasets/mmlongbench/mmlb_image"}}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/results/mmlongbench_evomm_vrag_docqa_k32_199}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MIN_ROUNDS="${MIN_ROUNDS:-1}"
TOP_K="${TOP_K:-6}"
LIMIT="${LIMIT:-}"

SCRIPT="${PROJECT_ROOT}/eval/mmlongbench_test_mm.py"

if [ ! -f "$SCRIPT" ]; then
  echo "错误: 找不到评测脚本: $SCRIPT"
  exit 1
fi

# 自动检测数据目录：
# 1) 显式 DATA_ROOT（若用户传了）
# 2) mmlb_data
# 3) mmlb_data_example
if [ -z "$DATA_ROOT" ]; then
  for candidate in \
    "./datasets/mmlongbench/mmlb_data" \
    "./datasets/mmlongbench/mmlb_data_example"
  do
    if [ -d "$candidate" ]; then
      DATA_ROOT="$candidate"
      break
    fi
  done
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "错误: 找不到数据目录: $DATA_ROOT"
  exit 1
fi

if [ -z "$IMAGE_ROOT" ]; then
  parent_dir="$(dirname "$DATA_ROOT")"
  inferred_image_root="${parent_dir}/mmlb_image"
  if [ -d "$inferred_image_root" ]; then
    IMAGE_ROOT="$inferred_image_root"
  else
    for candidate in \
      "./datasets/mmlongbench/mmlb_image"
    do
      if [ -d "$candidate" ]; then
        IMAGE_ROOT="$candidate"
        break
      fi
    done
  fi
fi

if [ ! -d "$IMAGE_ROOT" ]; then
  echo "错误: 找不到图片目录: $IMAGE_ROOT"
  echo "请设置 IMAGE_ROOT 或 MMLONGBENCH_IMAGE_ROOT"
  exit 1
fi

mkdir -p "$OUTDIR"

echo "模型: $MODEL"
echo "API: $API_BASE"
echo "数据根目录: $DATA_ROOT"
echo "图片目录: $IMAGE_ROOT"
echo "输出目录: $OUTDIR"
echo "仅评测: vrag (K32)"
echo ""

# vrag (K32 only)
vrag_k32_found=0
for f in "$DATA_ROOT"/vrag/*_K32_*.jsonl "$DATA_ROOT"/vrag/*_K32.jsonl; do
  [ -e "$f" ] || continue
  vrag_k32_found=1
  echo "========== VRAG: $(basename "$f") =========="
  cmd=(
    python "$SCRIPT"
    --data-file "$f"
    --image-root "$IMAGE_ROOT"
    --outdir "$OUTDIR"
    --model "$MODEL"
    --api-base "$API_BASE"
    --api-key "$API_KEY"
    --max-rounds "$MAX_ROUNDS"
    --min-rounds "$MIN_ROUNDS"
    --top-k "$TOP_K"
  )
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  "${cmd[@]}"
done



if [ "$vrag_k32_found" -eq 0 ]; then
  echo "警告: 未找到 vrag 的 K32 文件（期望类似 *_K32_*.jsonl）"
fi

