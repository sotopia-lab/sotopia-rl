export PYTHONPATH="/workspace/sotopia-rl:$PYTHONPATH"

# rejection sampling
python manage.py start_with_config \
    --sft_model_path "/workspace/sotopia-rl/data/models/sft_checkpoint-270" \
    --reward_model_path "/workspace/sotopia-rl/data/models/rm_baseline_checkpoint-14000" \
    --model_name "google/gemma-2-2b-it" \
    --template_path "/workspace/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 10 \
    --sft_batch_size 2 \
    --max_length 4096 \
    --port 8001

# ppo
export PYTHONPATH="/workspace/sotopia-rl:$PYTHONPATH"

python manage.py start_with_config \
    --sft_model_path "/workspace/sotopia-rl/data/models/sft_checkpoint-270" \
    --reward_model_path "/workspace/sotopia-rl/data/models/rm_baseline_checkpoint-14000" \
    --model_name "google/gemma-2-2b-it" \
    --template_path "/workspace/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 10 \
    --sft_batch_size 2 \
    --max_length 4096 \
    --port 8002

curl http://localhost:8002/sotopia_server/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-4o-mini",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'

curl -X POST http://localhost:8002/sotopia_server/train/tag \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{"tag": "example_tag"}'

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model google/gemma-2-2b-it --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja --port 8005 --enable-lora --lora-modules sotopia_gemma-2-2b-it-sft=/workspace/sotopia-rl/data/models/sft_checkpoint-270

poetry run python /workspace/sotopia-rl/evals/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_rejection_sampling@http://localhost:8002/sotopia"' \
'--gin.AGENT2_MODEL="custom/sotopia_gemma-2-2b-it-sft@http://localhost:8005/v1"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=False' \
'--gin.TAG="sotopia_rejection-sampling-rm-key-utterance-and-sft_vs_sotopia_gemma-2-2b-it-sft-1109_sample_10"'
