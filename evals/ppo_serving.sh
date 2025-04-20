export SFT_GPU=8
export PPO_GPU=9
export SFT_PORT=8090
export PPO_PORT=8095
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export PPO_MODEL_FOLDER_NAME="ppo_qwen25_7b_reward_only_response_gpt-4o"
export SFT_MODEL_CKPT_STEP=1000
export PPO_MODEL_CKPT_STEP=1500
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export PPO_MODEL_PATH="${REPO_FOLDER_NAME}/${PPO_MODEL_FOLDER_NAME}/checkpoint-${PPO_MODEL_CKPT_STEP}/"

export SFT_GPU=8
export PPO_GPU=9
export SFT_PORT=8070
export PPO_PORT=8075
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export PPO_MODEL_FOLDER_NAME="ppo_qwen25_7b_reward_utterance_quality_gpt-4o"
export SFT_MODEL_CKPT_STEP=1000
export PPO_MODEL_CKPT_STEP=1500
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export PPO_MODEL_PATH="${REPO_FOLDER_NAME}/${PPO_MODEL_FOLDER_NAME}/checkpoint-${PPO_MODEL_CKPT_STEP}/"


export SFT_GPU=2
export PPO_GPU=3
export SFT_PORT=8070
export PPO_PORT=8075
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b_sft_round_1_bc_data_top_2"
export SFT_MODEL_CKPT_STEP=1500
export PPO_MODEL_FOLDER_NAME="ppo_qwen25_7b_reward_utterance_quality_gpt-4o"
export PPO_MODEL_CKPT_STEP=2400
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export PPO_MODEL_PATH="${REPO_FOLDER_NAME}/${PPO_MODEL_FOLDER_NAME}/checkpoint-${PPO_MODEL_CKPT_STEP}/"
export ENV_MODEL="gpt-4o"


export SFT_GPU=0
export PPO_GPU=1
export SFT_PORT=8005
export PPO_PORT=8009
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b_sft_round_1_bc_data_top_2"
export PPO_MODEL_FOLDER_NAME="ppo_qwen25_7b_reward_direct_default_no_goal_gpt-4o_without_goal_leak_with_sft_self_play_data"
export SFT_MODEL_CKPT_STEP=1500
export PPO_MODEL_CKPT_STEP=15
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export PPO_MODEL_PATH="${REPO_FOLDER_NAME}/${PPO_MODEL_FOLDER_NAME}/checkpoint-${PPO_MODEL_CKPT_STEP}/"
export ENV_MODEL="gpt-4o"


export TAG="${PPO_MODEL_FOLDER_NAME}_step_${PPO_MODEL_CKPT_STEP}_vs_${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}-0327"
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export PPO_MODEL_NAME="${PPO_MODEL_FOLDER_NAME}-gpu${PPO_GPU}"
export MODEL_A=custom/${PPO_MODEL_NAME}@http://localhost:${PPO_PORT}/v1
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
CUDA_VISIBLE_DEVICES=$PPO_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$PPO_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$PPO_MODEL_NAME=$PPO_MODEL_PATH"

# Command 3: Run experiment evaluations.
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=1 \
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
  --gin.BATCH_SIZE=1 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT2_MODEL='${MODEL_A}'" \
  "--gin.AGENT1_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'"
