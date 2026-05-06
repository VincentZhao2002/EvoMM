#!/bin/bash


MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-4b-instruct}"
PORT="${PORT:-8000}" 
# PORT=8000
HOST="${HOST:-0.0.0.0}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-200}"
DISABLE_MM_CACHE="${DISABLE_MM_CACHE:-1}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache}"


export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT"


python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port $PORT \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt "{\"image\": ${MM_IMAGE_LIMIT}}" \
    $( [ "$DISABLE_MM_CACHE" = "1" ] && echo "--mm-processor-cache-gb 0" )
