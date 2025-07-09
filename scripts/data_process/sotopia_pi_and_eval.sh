# generate self-play script
python generate_conversations.py \
    --eval-script sotopia_pi_self_play_script.sh \
    --env-file used_env.json \
    --experiment-name selftrain-round-2 \
    --tag qwen-sft-qwen-sft-3-26-v1 \
    --batch-size 20 \
    --agent1-model custom/sotopia-sft-1@http://localhost:8005/v1 \
    --agent2-model custom/sotopia-sft-2@http://localhost:8006/v1 \
    --push-to-db True

# generate sotopia experiment script
python generate_conversations.py \
    --eval-script sotopia_all_eval_script.sh \
    --env-file used_env.json \
    --experiment-name sotopia_env \
    --tag xx \
    --batch-size 20 \
    --agent1-model agent1-model \
    --agent2-model agent2-model \
    --push-to-db True

# host models
# sotopia-sft-1
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8005 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--enable-lora \
--lora-modules sotopia-sft-1=/root/sotopia-rl/Untitled/.cache/final_sft/checkpoint-1000/ \
--served-model-name xx

# sotopia-sft-2
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8006 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--enable-lora \
--lora-modules sotopia-sft-2=/root/sotopia-rl/Untitled/.cache/final_sft/checkpoint-1000/ \
--served-model-name xx

## generate sft for ppo
python generate_sft_from_episodes.py \
--data_dir ../../data \
--utterances_output_subdir sotopia_pi_round1_qwen_utterances \
--episodes_file sotopia_pi_round1_qwen_episodes.jsonl \
--sft_output_file sotopia_pi_round1_qwen_sft_all.json

## generate sft for sotopia-pi
python generate_sft_from_episodes.py \
--data_dir ../../data \
--utterances_output_subdir sotopia_pi_round1_qwen_utterances_filtered \
--episodes_file sotopia_pi_round1_qwen_episodes_filtered.jsonl \
--sft_output_file sotopia_pi_round1_qwen_sft_pi.json
