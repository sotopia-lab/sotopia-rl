CUDA_VISIBLE_DEVICES=7,8,9 accelerate launch \
  --config_file /home/haofeiy2/.cache/huggingface/accelerate/default_config.yaml \
  /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /data/models/Qwen2.5-7B-Instruct \
  --value_model_name /data/models/Qwen2.5-7B-Instruct \
  --reward_model_name /data/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-5397 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-5397 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_direct_4o/checkpoint-5391 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_direct_4o/checkpoint-5391 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b

CUDA_VISIBLE_DEVICES=2,6,9 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --value_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --reward_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_direct_o3_mini/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b


  CUDA_VISIBLE_DEVICES=9 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --value_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --reward_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_only_response_gpt_4o/checkpoint-4000 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_only_response_gpt_4o/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b_reward_only_response_gpt_4o

  CUDA_VISIBLE_DEVICES=8 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --value_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --reward_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_average/checkpoint-4000 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_average/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b_reward_direct_average


  CUDA_VISIBLE_DEVICES=7 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --value_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --reward_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_goal_progress_gpt_4o/checkpoint-4000 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_goal_progress_gpt_4o/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b_reward_goal_progress_gpt_4o

  CUDA_VISIBLE_DEVICES=6 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --value_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --reward_model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b/checkpoint-4495 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_discounting/checkpoint-4000 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_discounting/checkpoint-4000 \
  --policy_use_qlora \
  --reward_use_qlora \
  --value_use_qlora \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --mini_batch_size 4 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --use_lora \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/ppo_qwen25_7b_reward_discounting
