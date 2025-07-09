#!/bin/bash

# Run the single evaluation script with your model checkpoint
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 python inference_grpo.py \
  --model_path $MODEL_PATH \
  --adapter_path "../grpo_checkpoint/grpo_checkpoints_qwen2.5-7b/best-checkpoint" \
  --template_path "../evals/qwen2.5-7b.jinja" \
  --example_path "../data/sotopia_pi_gpt4_ppo_overfit.json" \
  --max_length 4096 \
  --use_qlora
