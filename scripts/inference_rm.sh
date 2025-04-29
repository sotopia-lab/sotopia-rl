#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=9 python inference_rm.py \
  --model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/haofeiy2/sotopia-rl/rm_reward_mixed/checkpoint-4000" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json"


CUDA_VISIBLE_DEVICES=5 python inference_rm.py \
  --model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/haofeiy2/sotopia-rl/rm_token_length/checkpoint-800" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json"
