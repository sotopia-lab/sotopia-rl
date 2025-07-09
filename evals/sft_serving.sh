export REPO_FOLDER_NAME="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export SFT_GPU=2
export ORI_GPU=3
export SFT_PORT=9050
export ORI_PORT=9080
export SFT_MODEL_FOLDER_NAME="new_sft_default_0506"
export SFT_MODEL_CKPT_STEP=200
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export ORI_MODEL_PATH="${MODEL_PATH}"
export ENV_MODEL="gpt-4o"
export CHAT_TEMPLATE="${REPO_FOLDER_NAME}/evals/qwen2.5-7b.jinja"



export TAG="Qwen2.5-7B-Instruct_vs_${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}"
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export ORI_MODEL_NAME="Qwen2.5-7B-Instruct-gpu${ORI_GPU}"
export MODEL_A=custom/${ORI_MODEL_NAME}@http://localhost:${ORI_PORT}/v1
export MODEL_B=custom/${SFT_MODEL_NAME}@http://localhost:${SFT_PORT}/v1

# Command 1: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$SFT_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$SFT_PORT" \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$ORI_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$ORI_PORT" \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name $ORI_MODEL_NAME  

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