#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=9 python inference_rm.py \
  --model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/haofeiy2/sotopia-rl/rm_direct/checkpoint-5397" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_example.json" \
  --use_qlora


CUDA_VISIBLE_DEVICES=6 python inference_rm.py \
  --model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/haofeiy2/sotopia-rl/rm_direct_overfit/checkpoint-500" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json" \
  --use_qlora
