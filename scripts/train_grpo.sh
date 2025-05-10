CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
  --config_file /mnt/data/sotopia-rl/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /mnt/data/sotopia-rl/scripts/train_grpo.py \
  --model_name /mnt/data/models/Qwen2.5-7B-Instruct \
  --reward_adapter_path /mnt/data/sotopia-rl/rm_goal_direct_0507/checkpoint-6800 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --grpo_data_path /mnt/data/sotopia-rl/data/sft_self_play_0507.json \
  --template_path /mnt/data/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --use_lora_train_grpo \
  --num_generations 16 \
  --output_dir /mnt/data/sotopia-rl/grpo_rm_goal_direct_0507_with_beta_004
