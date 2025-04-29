CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /data/disk0/sotopia-rl/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /data/disk0/sotopia-rl/scripts/train_grpo.py \
  --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/disk0/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /data/disk0/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --grpo_data_path /data/disk0/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --use_lora_train_grpo \
  --num_generations 16 \
  --beta 0.04 \
  --output_dir /data/disk0/sotopia-rl/grpo_rm_reward_direct_default_beta_004