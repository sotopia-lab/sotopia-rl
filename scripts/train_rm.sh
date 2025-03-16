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
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_o3_mini \
--use_qlora

CUDA_VISIBLE_DEVICES=0 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 1 \
--accumulation_steps 4 \
--num_epochs 1000 \
--evaluation_steps 100 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_rm_overfit.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/rm_direct_overfit \
--use_qlora
