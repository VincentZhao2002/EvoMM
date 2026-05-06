#!/bin/bash

# 启动 vLLM 服务器
# 使用方法：bash scripts/start_vllm_server.sh

# 设置模型路径 (多模态 VL 模型)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
    MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-200}"
    ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"

echo "正在启动 vLLM 服务器..."
echo "模型路径: $MODEL_PATH"
echo "模型别名: $SERVED_MODEL_NAME"
echo "服务地址: http://$HOST:$PORT/v1"
echo "每个请求最多图片数: $MM_IMAGE_LIMIT"
echo "允许本地媒体路径: $ALLOWED_LOCAL_MEDIA_PATH"
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
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port $PORT \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt "{\"image\": ${MM_IMAGE_LIMIT}}"
