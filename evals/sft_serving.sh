export SFT_GPU=2
export ORI_GPU=3
export SFT_PORT=9050
export ORI_PORT=9080
export SFT_MODEL_FOLDER_NAME="new_sft_default_0506"
export SFT_MODEL_CKPT_STEP=200
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export ORI_MODEL_PATH="/mnt/data_from_server1/models/Qwen2.5-7B-Instruct"
export ENV_MODEL="gpt-4o"


export TAG="Qwen2.5-7B-Instruct_vs_${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}-0504"
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export ORI_MODEL_NAME="Qwen2.5-7B-Instruct-gpu${ORI_GPU}"
export MODEL_A=custom/${ORI_MODEL_NAME}@http://localhost:${ORI_PORT}/v1
export MODEL_B=custom/${SFT_MODEL_NAME}@http://localhost:${SFT_PORT}/v1
export REDIS_OM_URL="redis://:QzmCUD3C3RdsR@35.232.108.130:6379"

# Command 1: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$SFT_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$SFT_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$ORI_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$ORI_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
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



# GPUs and ports
export SFT_GPU=7
export ORI_GPU=8
export SFT_PORT=8060
export ORI_PORT=8070

# Model folders and checkpoints
export SFT_MODEL_FOLDER_NAME="sft_0510_epoch_500"
export ORI_MODEL_FOLDER_NAME="Qwen2.5-7B-Instruct"
export SFT_MODEL_CKPT_STEP=200

# Paths
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export ORI_MODEL_PATH="/mnt/data_from_server1/models/${ORI_MODEL_FOLDER_NAME}"

# Experiment metadata
export ENV_MODEL="gpt-4o"
export TAG="${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}_vs_${ORI_MODEL_FOLDER_NAME}-0509_with_fixed_action_order"

# Names for served endpoints
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export ORI_MODEL_NAME="${ORI_MODEL_FOLDER_NAME}-gpu${ORI_GPU}"

export MODEL_A="custom/${SFT_MODEL_NAME}@http://localhost:${SFT_PORT}/v1"
export MODEL_B="custom/${ORI_MODEL_NAME}@http://localhost:${ORI_PORT}/v1"

export REDIS_OM_URL="redis://:QzmCUD3C3RdsR@35.232.108.130:6379"

# Command 1: Launch the SFT (fine-tuned) server
CUDA_VISIBLE_DEVICES=$SFT_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$SFT_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --served-model-name "qwen25-7b-instruct" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Launch the ORI (base/instruct) server
CUDA_VISIBLE_DEVICES=$ORI_GPU python -m vllm.entrypoints.openai.api_server \
    --model "$ORI_MODEL_PATH" \
    --port "$ORI_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --served-model-name "$ORI_MODEL_NAME"
