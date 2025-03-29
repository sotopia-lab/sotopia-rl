# final direct
CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch \
  --config_file /workspace/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29511 \
  /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_round_1_bc_checkpoint-1500 \
  --ref_adapter_path /workspace/sotopia-rl/sft_round_1_bc_checkpoint-1500 \
  --reward_adapter_path /workspace/sotopia-rl/rm_direct_default_checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_direct_default_checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --local_rollout_forward_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 2 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja \
  --ppo_epochs 4 \
  --num_epochs 20 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_direct_default_no_goal_gpt-4o_without_goal_leak_with_sft_self_play_data_use_sotopia_pi_full_data

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file /workspace/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29530 \
  /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-1000 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-1000 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_direct_default_o3-mini/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_direct_default_o3-mini/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja \
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_direct_default_o3-mini

CUDA_VISIBLE_DEVICES=4 python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-1000 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-1000 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_mixed/checkpoint-5000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_mixed/checkpoint-5000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_mini_batches 1 \
  --gradient_accumulation_steps 1 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja \
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_only_response_gpt-4o

CUDA_VISIBLE_DEVICES=2,6,9 poetry run python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /workspace/sotopia-rl/rm_direct_o3_mini/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_direct_o3_mini/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b


  CUDA_VISIBLE_DEVICES=9 poetry run python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_only_response_gpt_4o/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_only_response_gpt_4o/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_only_response_gpt_4o

  CUDA_VISIBLE_DEVICES=8 poetry run python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_direct_average/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_direct_average/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_direct_average


  CUDA_VISIBLE_DEVICES=7 poetry run python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_goal_progress_gpt_4o/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_goal_progress_gpt_4o/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_goal_progress_gpt_4o

  CUDA_VISIBLE_DEVICES=6 poetry run python /workspace/sotopia-rl/scripts/train_ppo.py \
  --model_name /workspace/Qwen2.5-7B-Instruct \
  --value_model_name /workspace/Qwen2.5-7B-Instruct \
  --reward_model_name /workspace/Qwen2.5-7B-Instruct \
  --policy_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /workspace/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /workspace/sotopia-rl/rm_reward_discounting/checkpoint-4000 \
  --value_adapter_path /workspace/sotopia-rl/rm_reward_discounting/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /workspace/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /workspace/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /workspace/sotopia-rl/ppo_qwen25_7b_reward_discounting
