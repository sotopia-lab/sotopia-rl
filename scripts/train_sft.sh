CUDA_VISIBLE_DEVICES=6 python /data/haofeiy2/sotopia-rl/scripts/train_sft.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--use_lora \
--evaluation_steps 500 \
--sft_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_sft.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir sft_qwen25_7b \
--use_qlora


CUDA_VISIBLE_DEVICES=6 python /data/haofeiy2/sotopia-rl/scripts/train_sft.py \
--model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 1 \
--accumulation_steps 1 \
--num_epochs 300 \
--use_lora \
--evaluation_steps 100 \
--sft_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_gpt4_sft_overfit.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir sft_qwen25_7b_overfit \
--use_qlora
