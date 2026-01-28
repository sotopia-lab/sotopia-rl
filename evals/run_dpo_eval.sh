#!/bin/bash

# =============================================================================
# DPO Checkpoint Evaluation Script
# Runs sotopia evaluation against served models
# Usage: ./run_dpo_eval.sh <dpo_checkpoint>
# 
# Prerequisites: Run serve_dpo_checkpoints.sh first to start the servers
# =============================================================================

set -e

# Configuration
export REPO_FOLDER_NAME="/root/sotopia-rl"
export DPO_PORT=7020
export BASELINE_PORT=7010
export ENV_MODEL="gpt-4o"

# Environment IDs for evaluation (from existing eval scripts)
ENV_IDS='["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]'    

# Get checkpoint from argument
DPO_CKPT=${1:-1000}

DPO_MODEL_NAME="dpo-ckpt-${DPO_CKPT}"
BASELINE_MODEL_NAME="sft-baseline"

# Model endpoints
MODEL_A="custom/${DPO_MODEL_NAME}@http://localhost:${DPO_PORT}/v1"
MODEL_B="custom/${BASELINE_MODEL_NAME}@http://localhost:${BASELINE_PORT}/v1"

TAG="dpo_ckpt_${DPO_CKPT}_vs_sft_baseline"

echo "=============================================="
echo "DPO Checkpoint Evaluation"
echo "=============================================="
echo "DPO Checkpoint: ${DPO_CKPT}"
echo "Model A (DPO): ${MODEL_A}"
echo "Model B (Baseline): ${MODEL_B}"
echo "Tag: ${TAG}"
echo "Environment Model: ${ENV_MODEL}"
echo "=============================================="

# Check if servers are running
check_server() {
    local port=$1
    local name=$2
    if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo "✓ ${name} server is running on port ${port}"
        return 0
    else
        echo "✗ ${name} server is NOT running on port ${port}"
        return 1
    fi
}

echo ""
echo "Checking servers..."
DPO_OK=false
BASELINE_OK=false

if check_server $DPO_PORT "DPO"; then
    DPO_OK=true
fi

if check_server $BASELINE_PORT "Baseline"; then
    BASELINE_OK=true
fi

if [ "$DPO_OK" != "true" ]; then
    echo ""
    echo "ERROR: DPO server not running. Start it first with:"
    echo "  ./evals/serve_dpo_checkpoints.sh ${DPO_CKPT}"
    exit 1
fi

echo ""
echo "=============================================="
echo "Running Sotopia Evaluation"
echo "=============================================="

echo ""
echo "--- DPO as Agent1 ---"
python evals/experiment_eval.py \
    --gin_file evals/sotopia_conf/generation_utils_conf/generate.gin \
    --gin_file evals/sotopia_conf/server_conf/server.gin \
    --gin_file evals/sotopia_conf/run_async_server_in_batch.gin \
    --gin.BATCH_SIZE=20 \
    --gin.PUSH_TO_DB=True \
    "--gin.ENV_IDS=${ENV_IDS}" \
    "--gin.ENV_MODEL='${ENV_MODEL}'" \
    "--gin.AGENT1_MODEL='${MODEL_A}'" \
    "--gin.AGENT2_MODEL='${MODEL_B}'" \
    "--gin.TAG='${TAG}_dpo_agent1'"

echo ""
echo "--- DPO as Agent2 (reversed) ---"
python evals/experiment_eval.py \
    --gin_file evals/sotopia_conf/generation_utils_conf/generate.gin \
    --gin_file evals/sotopia_conf/server_conf/server.gin \
    --gin_file evals/sotopia_conf/run_async_server_in_batch.gin \
    --gin.BATCH_SIZE=20 \
    --gin.PUSH_TO_DB=True \
    "--gin.ENV_IDS=${ENV_IDS}" \
    "--gin.ENV_MODEL='${ENV_MODEL}'" \
    "--gin.AGENT1_MODEL='${MODEL_B}'" \
    "--gin.AGENT2_MODEL='${MODEL_A}'" \
    "--gin.TAG='${TAG}_dpo_agent2'"

echo ""
echo "=============================================="
echo "Evaluation Complete for DPO checkpoint ${DPO_CKPT}"
echo "=============================================="

