#!/bin/bash

# 启动 vLLM 服务器
# 使用方法：bash scripts/start_vllm_server.sh
#
# Qwen3-VL 与 vLLM 版本：若报错「limit_mm_per_prompt is only supported for multimodal models」，
# 不是下面缓存参数改错，而是当前 vLLM 过旧（如 0.9.x），registry 里没有 Qwen3VLForConditionalGeneration，
# 模型会退成 TransformersForCausalLM，被当成非多模态。请在 base 中升级 vLLM，例如: pip install -U 'vllm>=0.15'

# 设置模型路径 (多模态 VL 模型)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-4b-instruct}"
PORT="${PORT:-8000}" 
# PORT=8000
HOST="${HOST:-0.0.0.0}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-200}"
DISABLE_MM_CACHE="${DISABLE_MM_CACHE:-1}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache}"

# 关掉 HF tokenizers 的 fork-after-parallel 警告刷屏
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 统一缓存目录，避免写入默认的 ~/.cache
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT"

echo "正在启动 vLLM 服务器..."
echo "模型路径: $MODEL_PATH"
echo "模型别名: $SERVED_MODEL_NAME"
echo "服务地址: http://$HOST:$PORT/v1"
echo "每个请求最多图片数: $MM_IMAGE_LIMIT"
echo "禁用多模态预处理缓存: $DISABLE_MM_CACHE"
echo "允许本地媒体路径: $ALLOWED_LOCAL_MEDIA_PATH"
echo "缓存根目录: $CACHE_ROOT"
echo ""

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 启动 vLLM 服务器
# --served-model-name: 必须与评测脚本中的 --model 参数一致
# --allowed-local-media-path: 允许读取本地图片文件
# --limit-mm-per-prompt: vLLM 服务端图片上限（可通过 MM_IMAGE_LIMIT 环境变量调整）
# 禁用多模态预处理缓存：旧版 vLLM 无 --disable-mm-preprocessor-cache，用 --mm-processor-cache-gb 0 等价关闭
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
