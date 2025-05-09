CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /data/disk0/sotopia-rl/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /data/disk0/sotopia-rl/scripts/train_grpo.py \
  --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/disk0/sotopia-rl/new_sft_default_0506_checkpoint-1000 \
  --reward_adapter_path /data/disk0/sotopia-rl/rm_goal_0503_w_relationship_knowledge_0507/checkpoint-18300 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --grpo_data_path /data/disk0/sotopia-rl/data/sft_self_play_0507.json \
  --template_path /data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --use_lora_train_grpo \
  --num_generations 16 \
  --output_dir /data/disk0/sotopia-rl/grpo_rm_goal_relationship_knowledge
