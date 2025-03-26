CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --nproc_per_node=1 --master_port=29504 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 10 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_all_the_same.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_all_the_same \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run --nproc_per_node=1 --master_port=29506 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 10 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_10-scale_no_goal_gpt-4o_without_goal_leak.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_10-scale_no_goal_gpt-4o_without_goal_leak \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port=29508 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 10 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_default_no_goal_gpt-4o_without_goal_leak.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_default_no_goal_gpt-4o_without_goal_leak \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=8 python -m torch.distributed.run --nproc_per_node=1 --master_port=29510 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 10 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_default_normalized_no_goal_gpt-4o_without_goal_leak.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_default_normalized_no_goal_gpt-4o_without_goal_leak \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.run --nproc_per_node=1 --master_port=29512 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 10 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_key_utterance_no_goal_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_key_utterance_no_goal_gpt-4o \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json


CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python -m torch.distributed.run --nproc_per_node=6 --master_port=29512 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 1 \
--num_epochs 20 \
--evaluation_steps 400 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_default_no_goal_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_default_no_goal_gpt-4o \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json


CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 --master_port=29511 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 20 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_default_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_default_gpt-4o \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 20 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_5-scale_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_5-scale_gpt-4o \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 8e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 20 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_mixed_direct_o3_only_response.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_mixed_direct_o3_only_response \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 5e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_goal_progress_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_goal_progress_gpt-4o \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port=29502 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 5e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 1 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_discounting.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_discounting \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json


CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
/data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 5e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 1000 \
--evaluation_steps 50 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_overfit \
--deepspeed \
--deepspeed_config /data/haofeiy2/sotopia-rl/scripts/ds_config_rm.json


CUDA_VISIBLE_DEVICES=9 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 5e-4 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 1 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_goal_progress_gpt-4o_cleaned.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_goal_progress_gpt-4o_cleaned

CUDA_VISIBLE_DEVICES=5 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_o3-mini_attribution_direct.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_o3_mini

CUDA_VISIBLE_DEVICES=9 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_discounting.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_discounting

CUDA_VISIBLE_DEVICES=8 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 100 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_direct_average.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_direct_average

CUDA_VISIBLE_DEVICES=8 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-4 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_goal_progress_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_goal_progress_gpt_4o

CUDA_VISIBLE_DEVICES=9 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_only_response_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_reward_only_response_gpt_4o

CUDA_VISIBLE_DEVICES=1 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 2 \
--val_batch_size 2 \
--accumulation_steps 4 \
--num_epochs 5 \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_5-scale_gpt-4o.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_gpt4o


CUDA_VISIBLE_DEVICES=9 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 5e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 1 \
--accumulation_steps 8 \
--num_epochs 1000 \
--evaluation_steps 100 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_overfit
