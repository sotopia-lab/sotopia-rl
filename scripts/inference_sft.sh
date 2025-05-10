#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=1 python inference_sft.py \
  --model_path "/data/disk0/models/Qwen2.5-7B-Instruct" \
  --adapter_path "/data/disk0/sotopia-rl/overfit_sft_test/checkpoint-500" \
  --template_path "/data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --example_path "/data/disk0/sotopia-rl/data/sotopia_pi_gpt4_sft_overfit.json" \
  --max_length 4096 \
  --use_qlora
