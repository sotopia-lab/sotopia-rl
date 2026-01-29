#!/bin/bash

# ========================================
# Step 1: Generate DPO pairs using vLLM
# ========================================
# vLLM uses tensor parallelism for multi-GPU inference
# --tensor_parallel_size: Number of GPUs for tensor parallelism (model sharding)

python dpo/generate_dpo_pairs.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --input_path data/sotopia_grpo.json \
    --output_path data/dpo_pairs_generated.json \
    --batch_size 64 \
    --tensor_parallel_size 4

# ========================================
# Step 2: Score DPO pairs using reward model
# ========================================
# Uses HuggingFace transformers with data parallelism via accelerate
# For multi-GPU data parallel scoring:

# Option A: Using accelerate for data parallelism
accelerate launch \
    --num_processes 4 \
    --multi_gpu \
    dpo/score_dpo_pairs.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path saves/rm_goal_0503_w_relationship_knowledge_0507/checkpoint-6800 \
    --input_path data/dpo_pairs_generated.json \
    --output_path data/dpo_pairs_scored.json \
    --batch_size 16 \
    --num_workers 8

# ========================================
# Step 3: Train DPO model
# ========================================
# Uses TRL DPOTrainer with LoRA for efficient fine-tuning
# Trains policy model to prefer chosen responses over rejected ones
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file scripts/accelerate_config_sft.yaml \
    dpo/train_dpo.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dpo_data_path data/dpo_pairs_scored.json \
    --template_path evals/qwen2.5-7b.jinja \
    --output_dir saves/dpo_checkpoint \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length 4096 \
    --max_prompt_length 2048 \
    --save_steps 100 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules q_proj,v_proj \
    --wandb_project sotopia-dpo \
    --wandb_run_name dpo-qwen2.5-7b
