CUDA_VISIBLE_DEVICES=1,6,7 accelerate launch \
  --config_file /data/haofeiy2/sotopia-rl/scripts/accelerate_config.yaml \
  --main_process_port 29511 \
  /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --learning_rate 5e-6 \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --local_rollout_forward_batch_size 1 \
  --num_mini_batches 1 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_ppo_epochs 2 \
  --num_train_epochs 5 \
  --gamma 0.99 \
  --lam 0.95 \
  --output_dir /data/haofeiy2/sotopia-rl/ppo_origin_qwen25_7b_reward_direct_default_no_goal_gpt-4o_without_goal_leak_with_sft_self_play_data_use_sotopia_pi_full_data_0408 \
  --use_lora_train_ppo