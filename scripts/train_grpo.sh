CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file /root/sotopia-rl/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /root/sotopia-rl/scripts/train_grpo.py \
  --model_name /root/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /root/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /root/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /root/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --value_adapter_path /root/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --grpo_data_path /root/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /root/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 3 \
  --num_train_epochs 2 \
  --output_dir /root/sotopia-rl/grpo__test