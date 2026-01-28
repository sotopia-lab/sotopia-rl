#!/bin/bash

# =============================================================================
# Serve a single DPO checkpoint for evaluation
# Usage: ./serve_dpo.sh <checkpoint_step> [gpu] [port]
# Example: ./serve_dpo.sh 1000 0 7020
# =============================================================================

CKPT=${1:-1000}
GPU=${2:-0}
PORT=${3:-7020}

# Configuration
REPO_FOLDER="/root/sotopia-rl"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CHAT_TEMPLATE="${REPO_FOLDER}/evals/qwen2.5-7b.jinja"
DPO_FOLDER="saves/dpo_checkpoint"

DPO_PATH="${REPO_FOLDER}/${DPO_FOLDER}/checkpoint-${CKPT}/"
MODEL_NAME="dpo-ckpt-${CKPT}"

echo "=============================================="
echo "Serving DPO Checkpoint"
echo "=============================================="
echo "Checkpoint: ${CKPT}"
echo "Path: ${DPO_PATH}"
echo "GPU: ${GPU}"
echo "Port: ${PORT}"
echo "Model name: ${MODEL_NAME}"
echo "=============================================="

if [ ! -d "$DPO_PATH" ]; then
    echo "ERROR: Checkpoint not found at ${DPO_PATH}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$PORT" \
    --max-lora-rank 32 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "${MODEL_NAME}=${DPO_PATH}"

