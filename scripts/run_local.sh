#!/bin/bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MODEL_PATH="/mnt/blob-pretraining-hptraining/long_corpus/checkpoints/lcft_Meta-Llama-3-8B_ready_book-odl/checkpoint-1000"
# MODEL_NAME="THUDM/glm-4-9b-chat"
PORT=8000
API_KEY="token-abc123"

# 启动 vLLM 服务，后台运行
vllm serve $MODEL_PATH \
    --port $PORT \
    --api-key $API_KEY \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max_model_len 131072 \
    --trust-remote-code &

VLLM_PID=$!

# 等待 vLLM 服务启动完成
echo "Waiting for vLLM to be ready..."
until curl -s http://127.0.0.1:$PORT/v1/models >/dev/null; do
    sleep 1
done

echo "vLLM is ready! Starting prediction..."
python pred.py --model $MODEL_PATH

# 可选：推理结束后关闭 vLLM
kill $VLLM_PID

python result.py
