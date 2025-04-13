CUDA_VISIBLE_DEVICES=1,4 accelerate launch \
  --config_file /data/haofeiy2/sotopia-rl-test/sotopia-rl/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /data/haofeiy2/sotopia-rl-test/sotopia-rl/scripts/train_grpo.py \
  --model_name /data/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --value_adapter_path /data/haofeiy2/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --grpo_data_path /mnt/data_from_server2/haofeiy2/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /data/haofeiy2/sotopia-rl-test/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --num_train_epochs 5 \
  --output_dir /data/haofeiy2/sotopia-rl-test/sotopia-rl/grpo__test