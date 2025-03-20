# ============================
# running rejection sampling with size 20
# ============================

# with sotopia-rl env, launch the rejection sampling model
CUDA_VISIBLE_DEVICES=0 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 1 \
    --max_length 4096 \
    --use_qlora \
    --port 8001

CUDA_VISIBLE_DEVICES=1 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --use_qlora \
    --port 8002

CUDA_VISIBLE_DEVICES=2 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 10 \
    --max_length 4096 \
    --use_qlora \
    --port 8003

CUDA_VISIBLE_DEVICES=3 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 20 \
    --max_length 4096 \
    --use_qlora \
    --port 8004

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port 8005 \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja  \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules qwen25-7b-instruct-sft-gpu4=/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/

CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port 8006 \
    --chat-template /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja  \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules qwen25-7b-instruct-sft-gpu5=/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/


# with sotopia env, launch the evaluation


# ============================
# switch agent1 and agent2
# ============================

# with sotopia-rl env, launch the rejection sampling model
CUDA_VISIBLE_DEVICES=4,5 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 40 \
    --max_length 4096 \
    --sft_batch_size 10 \
    --rm_batch_size 3 \
    --port 8002

# with sotopia env, launch the SFT model
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
--model /data/models/gemma-2-2b-it \
--port 8006 \
--chat-template /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja  \
--served-model-name gemma-2-2b-it \
--enable-lora \
--lora-modules sotopia_gemma-2-2b-it-sft=/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270


# with sotopia env, launch the evaluation
poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_rejection_sampling@http://localhost:8002/sotopia"' \
'--gin.AGENT2_MODEL="custom/sotopia_gemma-2-2b-it-sft@http://localhost:8006/v1"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1204_sample_40"'
