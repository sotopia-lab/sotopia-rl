#!/bin/bash
export VLLM_GPU=0
export DJANGO_GPU=1
export VLLM_PORT=8001
export DJANGO_PORT=8002
export REJ_SAMPLING_NUM=10
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export RM_FOLDER_NAME="rm_reward_direct_default_o3-mini"
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-1000/"
export RM_MODEL_PATH="${REPO_FOLDER_NAME}/${RM_FOLDER_NAME}/checkpoint-4000"


<<<<<<< HEAD
export VLLM_GPU=2
export DJANGO_GPU=3
export VLLM_PORT=8003
export DJANGO_PORT=8004
export REJ_SAMPLING_NUM=10
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export RM_FOLDER_NAME="rm_reward_goal_progress_gpt-4o"
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-1000/"
export RM_MODEL_PATH="${REPO_FOLDER_NAME}/${RM_FOLDER_NAME}/checkpoint-2240"
=======
CUDA_VISIBLE_DEVICES=1 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_reward_direct_default_o3-mini/checkpoint-3500" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --use_qlora \
    --port 8002
>>>>>>> 9d7e7487588bf98c1cc9a5769cb004dbe3b508c2


export VLLM_GPU=4
export DJANGO_GPU=5
export VLLM_PORT=8015
export DJANGO_PORT=8017
export REJ_SAMPLING_NUM=10
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export RM_FOLDER_NAME="rm_reward_discounting"
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-1000/"
export RM_MODEL_PATH="${REPO_FOLDER_NAME}/${RM_FOLDER_NAME}/checkpoint-2250"


export VLLM_GPU=6
export DJANGO_GPU=7
export VLLM_PORT=8025
export DJANGO_PORT=8037
export REJ_SAMPLING_NUM=10
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export RM_FOLDER_NAME="rm_reward_only_response_gpt-4o"
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-1000/"
export RM_MODEL_PATH="${REPO_FOLDER_NAME}/${RM_FOLDER_NAME}/checkpoint-3595"


export VLLM_GPU=8
export DJANGO_GPU=9
export VLLM_PORT=8045
export DJANGO_PORT=8057
export REJ_SAMPLING_NUM=10
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b"
export RM_FOLDER_NAME="rm_reward_utterance_quality_gpt-4o"
export REPO_FOLDER_NAME="/data/haofeiy2/sotopia-rl"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-1000/"
export RM_MODEL_PATH="${REPO_FOLDER_NAME}/${RM_FOLDER_NAME}/checkpoint-4200"


export TAG="${RM_FOLDER_NAME}_rej_sampling_num${REJ_SAMPLING_NUM}_vs_${SFT_MODEL_FOLDER_NAME}-0322"
export SFT_MODEL_NAME="qwen25-7b-instruct-sft-gpu${VLLM_GPU}"
export MODEL_A=custom/${RM_FOLDER_NAME}_rejsampling_num${REJ_SAMPLING_NUM}@http://localhost:${DJANGO_PORT}/sotopia
export MODEL_B=custom/${SFT_MODEL_NAME}@http://localhost:${VLLM_PORT}/v1
export REDIS_OM_URL="redis://:QzmCUD3C3RdsR@35.232.108.130:6379"
export SFT_MODEL_VLLM_API_URL="http://localhost:${VLLM_PORT}/v1/completions"


# Command 1: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$VLLM_PORT" \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Start the Django server with the specified configuration.
CUDA_VISIBLE_DEVICES=$DJANGO_GPU python /data/haofeiy2/sotopia-rl/serves/manage.py start_with_config \
    --sft_model_name "$SFT_MODEL_NAME" \
    --sft_model_vllm_api_url "$SFT_MODEL_VLLM_API_URL" \
    --reward_model_path "$RM_MODEL_PATH" \
    --reward_model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses "$REJ_SAMPLING_NUM" \
    --max_length 4096 \
    --port "$DJANGO_PORT" \
    --sft_batch_size 10 \
    --rm_batch_size 10

# Command 3: Run experiment evaluations.
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=1 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
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
  "--gin.AGENT2_MODEL='${MODEL_A}'" \
  "--gin.AGENT1_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'"