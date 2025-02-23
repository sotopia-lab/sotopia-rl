# ============================
# running rejection sampling with size 20
# ============================

# with sotopia-rl env, launch the rejection sampling model
CUDA_VISIBLE_DEVICES=2,3,4 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 20 \
    --max_length 4096 \
    --sft_batch_size 10 \
    --rm_batch_size 3 \
    --port 8001

# with sotopia env, launch the SFT model
CUDA_VISIBLE_DEVICES=8 python -m vllm.entrypoints.openai.api_server \
--model /data/models/gemma-2-2b-it \
--port 8005 \
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
'--gin.AGENT1_MODEL="custom/sotopia_rejection_sampling@http://localhost:8001/sotopia"' \
'--gin.AGENT2_MODEL="custom/sotopia_gemma-2-2b-it-sft@http://localhost:8005/v1"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
<<<<<<< HEAD
'--gin.TAG="sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1204_v2_sample_20"'
=======
'--gin.TAG="sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1203_sample_20"'

>>>>>>> 4cd3d6a03b7d23ef240b8bb0838e90de02f38fc8







<<<<<<< HEAD
# ============================
# switch agent1 and agent2
# ============================
=======
# switch agent1 and agent2
>>>>>>> 4cd3d6a03b7d23ef240b8bb0838e90de02f38fc8

# with sotopia-rl env, launch the rejection sampling model
CUDA_VISIBLE_DEVICES=4,5 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
<<<<<<< HEAD
    --max_responses 40 \
=======
    --max_responses 1 \
>>>>>>> 4cd3d6a03b7d23ef240b8bb0838e90de02f38fc8
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
<<<<<<< HEAD
'--gin.AGENT1_MODEL="custom/sotopia_rejection_sampling@http://localhost:8002/sotopia"' \
'--gin.AGENT2_MODEL="custom/sotopia_gemma-2-2b-it-sft@http://localhost:8006/v1"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1204_sample_40"'
=======
'--gin.AGENT1_MODEL="custom/sotopia_gemma-2-2b-it-sft@http://localhost:8006/v1"' \
'--gin.AGENT2_MODEL="custom/sotopia_rejection_sampling@http://localhost:8002/sotopia"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_rejection-sampling-rm-direct-prompt-and-sft_vs_sotopia_gemma-2-2b-it-sft-1203_sample_1"'
>>>>>>> 4cd3d6a03b7d23ef240b8bb0838e90de02f38fc8
