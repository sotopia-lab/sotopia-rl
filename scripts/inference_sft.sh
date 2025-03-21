#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=9 python inference_sft.py \
  --model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-2500" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_sft_overfit.json" \
  --max_length 4096 \
  --use_qlora
