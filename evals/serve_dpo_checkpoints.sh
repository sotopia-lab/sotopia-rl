#!/bin/bash

# =============================================================================
# DPO Checkpoint Server Script
# Serves DPO and baseline models for evaluation
# Usage: ./serve_dpo_checkpoints.sh <dpo_checkpoint> [baseline|dpo_only]
# =============================================================================

set -e

# Configuration
export REPO_FOLDER_NAME="/root/sotopia-rl"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export CHAT_TEMPLATE="${REPO_FOLDER_NAME}/evals/qwen2.5-7b.jinja"

# DPO model configuration
export DPO_MODEL_FOLDER="saves/dpo_checkpoint"

# Baseline model (SFT)
export BASELINE_MODEL_FOLDER="saves/sft_0510_epoch_500"
export BASELINE_CKPT=200

# GPU and port configuration
export DPO_GPU=0
export BASELINE_GPU=1
export DPO_PORT=7020
export BASELINE_PORT=7010

# Get checkpoint from argument
DPO_CKPT=${1:-1000}
MODE=${2:-baseline}  # "baseline" or "dpo_only"

DPO_MODEL_NAME="dpo-ckpt-${DPO_CKPT}"
BASELINE_MODEL_NAME="sft-baseline"

DPO_PATH="${REPO_FOLDER_NAME}/${DPO_MODEL_FOLDER}/checkpoint-${DPO_CKPT}/"
BASELINE_PATH="${REPO_FOLDER_NAME}/${BASELINE_MODEL_FOLDER}/checkpoint-${BASELINE_CKPT}/"

echo "=============================================="
echo "DPO Checkpoint Server"
echo "=============================================="
echo "DPO Checkpoint: ${DPO_CKPT}"
echo "DPO Path: ${DPO_PATH}"
echo "DPO GPU: ${DPO_GPU}, Port: ${DPO_PORT}"
echo ""
echo "Baseline Path: ${BASELINE_PATH}"
echo "Baseline GPU: ${BASELINE_GPU}, Port: ${BASELINE_PORT}"
echo "Mode: ${MODE}"
echo "=============================================="

# Check if DPO checkpoint exists
if [ ! -d "$DPO_PATH" ]; then
    echo "ERROR: DPO checkpoint not found at ${DPO_PATH}"
    exit 1
fi

# Function to stop servers
stop_servers() {
    echo "Stopping existing vLLM servers..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3
}

# Stop any existing servers first
stop_servers

echo ""
echo "Starting DPO model server on GPU ${DPO_GPU}, port ${DPO_PORT}..."
CUDA_VISIBLE_DEVICES=$DPO_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$DPO_PORT" \
    --max-lora-rank 64 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "${DPO_MODEL_NAME}=${DPO_PATH}" &

DPO_PID=$!
echo "DPO server PID: ${DPO_PID}"

if [ "$MODE" == "baseline" ]; then
    if [ -d "$BASELINE_PATH" ]; then
        echo ""
        echo "Starting baseline model server on GPU ${BASELINE_GPU}, port ${BASELINE_PORT}..."
        CUDA_VISIBLE_DEVICES=$BASELINE_GPU python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --port "$BASELINE_PORT" \
            --max-lora-rank 64 \
            --chat-template $CHAT_TEMPLATE \
            --served-model-name qwen25-7b-instruct \
            --enable-lora \
            --lora-modules "${BASELINE_MODEL_NAME}=${BASELINE_PATH}" &
        
        BASELINE_PID=$!
        echo "Baseline server PID: ${BASELINE_PID}"
    else
        echo "WARNING: Baseline model not found at ${BASELINE_PATH}"
    fi
fi

echo ""
echo "=============================================="
echo "Servers Starting..."
echo "=============================================="
echo ""
echo "DPO Model Endpoint:"
echo "  custom/${DPO_MODEL_NAME}@http://localhost:${DPO_PORT}/v1"
echo ""
if [ "$MODE" == "baseline" ] && [ -d "$BASELINE_PATH" ]; then
    echo "Baseline Model Endpoint:"
    echo "  custom/${BASELINE_MODEL_NAME}@http://localhost:${BASELINE_PORT}/v1"
    echo ""
fi
echo "Press Ctrl+C to stop servers"
echo "=============================================="

# Wait for servers
wait

