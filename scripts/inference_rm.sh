#!/bin/bash

# Run the single evaluation script with your model checkpoint
CUDA_VISIBLE_DEVICES=9 python inference_rm.py \
  --base_model "/data/models/Qwen2.5-7B-Instruct" \
  --checkpoint_path "/data/haofeiy2/sotopia-rl/rm_direct/checkpoint-5397" \
  --template_path "/data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja" \
  --use_4bit