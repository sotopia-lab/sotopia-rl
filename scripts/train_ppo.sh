# parameter I used for final PPO checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29439 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_reward_direct_default_without_that_n_error_as_the_end/checkpoint-4480 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/ppo_top_2_sft_step1500_for_pretrained_value_model_direct_rm/checkpoint-90 \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 30 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 0.99 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/ppo_top_2_sft_step1500_with_pretrained_value_model_gamma_099_direct_rm

# param for pretrain value adapter
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29439 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_goal_w_conversation_behavior_4_23/checkpoint-9400 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_goal_w_conversation_behavior_4_23/checkpoint-9400 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 30 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/ppo_top_2_sft_step1500_for_pretrained_value_model_direct_rm



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29439 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_checkpoint-800 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/ppo_top_2_sft_1_epoch_step160_default_kl_token_length_pretrained_value_model \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/ppo_top_2_sft_1_epoch_step160_default_kl_token_length_with_pretrained_value_model

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29449 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-30 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-30 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/grpo_top_2_sft_step30_default_kl


CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29469 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-50 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-50 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/grpo_top_2_sft_step50_default_kl


CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29499 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-70 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-70 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/grpo_top_2_sft_step70_default_kl


CUDA_VISIBLE_DEVICES=5 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29549 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-30 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_round_1_bc_data_top_2_ckpt/checkpoint-30 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/

CUDA_VISIBLE_DEVICES=6 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29559 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_bc/checkpoint-500 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_bc/checkpoint-500 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/ppo_token_length_normalized_with_sft_testing_ckpt500

CUDA_VISIBLE_DEVICES=7 accelerate launch \
  --config_file /projects/bdpw/haofeiy/sotopia-rl/scripts/accelerate_config_ppo.yaml \
  --main_process_port 29569 \
 /projects/bdpw/haofeiy/sotopia-rl/scripts/train_ppo.py \
  --model_name /projects/bdpw/haofeiy/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_bc/checkpoint-700 \
  --ref_adapter_path /projects/bdpw/haofeiy/sotopia-rl/sft_qwen25_7b_bc/checkpoint-700 \
  --reward_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --value_adapter_path /projects/bdpw/haofeiy/sotopia-rl/rm_token_length_normalized/checkpoint-500 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --num_mini_batches 1 \
  --ppo_data_path /projects/bdpw/haofeiy/sotopia-rl/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /projects/bdpw/haofeiy/sotopia-rl/evals/qwen2.5-7b.jinja \
  --num_train_epochs 5 \
  --max_length 4096 \
  --num_ppo_epochs 2 \
  --gamma 1.00 \
  --use_lora_train_ppo \
  --output_dir /projects/bdpw/haofeiy/sotopia-rl/ppo_token_length_normalized_with_sft_testing_ckpt700
