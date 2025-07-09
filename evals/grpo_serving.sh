export REPO_FOLDER_NAME="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export SFT_GPU=0
export PPO_GPU=1
export SFT_PORT=7010
export GRPO_PORT=7020
export SFT_MODEL_FOLDER_NAME="sft_checkpoints_qwen2.5-7b"
export GRPO_MODEL_FOLDER_NAME="grpo_checkpoints_qwen2.5-7b"
export SFT_MODEL_CKPT_STEP=1600
export GRPO_MODEL_CKPT_STEP=1600
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export PPO_MODEL_PATH="${REPO_FOLDER_NAME}/${PPO_MODEL_FOLDER_NAME}/checkpoint-${PPO_MODEL_CKPT_STEP}/"
export ENV_MODEL="gpt-4o"
export CHAT_TEMPLATE="${REPO_FOLDER_NAME}/evals/qwen2.5-7b.jinja"

export TAG="${GRPO_MODEL_FOLDER_NAME}_step_${GRPO_MODEL_CKPT_STEP}_vs_${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}"
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export GRPO_MODEL_NAME="${GRPO_MODEL_FOLDER_NAME}-gpu${GRPO_GPU}"
export MODEL_A=custom/${PPO_MODEL_NAME}@http://localhost:${PPO_PORT}/v1
export MODEL_B=custom/${SFT_MODEL_NAME}@http://localhost:${SFT_PORT}/v1

# Command 1: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$SFT_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$SFT_PORT" \
    --max-lora-rank 64 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$GRPO_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$GRPO_PORT" \
    --max-lora-rank 64 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$GRPO_MODEL_NAME=$GRPO_MODEL_PATH"

# Command 3: Run experiment evaluations.
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=20 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT1_MODEL='${MODEL_A}'" \
  "--gin.AGENT2_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'" \
&& \
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=20 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT2_MODEL='${MODEL_A}'" \
  "--gin.AGENT1_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'"
