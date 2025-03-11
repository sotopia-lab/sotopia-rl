#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=8 python inference_sft.py \
  --base_model "/data/models/Qwen2.5-7B-Instruct" \
  --lora_checkpoint "/data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-2000" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --use_4bit \
