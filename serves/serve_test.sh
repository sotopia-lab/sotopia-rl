CUDA_VISIBLE_DEVICES=0,1 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4000/" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/rm_reward_direct_default_o3-mini/checkpoint-3500" \
    --model_name "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --use_qlora \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_baseline/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_baseline/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 10 \
    --max_length 4096 \
    --port 8001

CUDA_VISIBLE_DEVICES=1 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_baseline/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 20 \
    --max_length 4096 \
    --port 8001

CUDA_VISIBLE_DEVICES=1 python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 20 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_direct_prompt/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 10 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_key_utterance/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 1 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_key_utterance/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 5 \
    --max_length 4096 \
    --port 8001

python manage.py start_with_config \
    --sft_model_path "/data/haofeiy2/sotopia-rl/saves/llama_factory_sft/checkpoint-270" \
    --reward_model_path "/data/haofeiy2/sotopia-rl/saves/rm_key_utterance/checkpoint-14000" \
    --model_name "/data/models/gemma-2-2b-it/" \
    --template_path "/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja" \
    --max_responses 10 \
    --max_length 4096 \
    --port 8001

curl http://localhost:8001/sotopia/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-4o-mini",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
