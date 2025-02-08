CUDA_VISIBLE_DEVICES=2 python /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
--model_name /data/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 1 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_annotated_reward.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves_qwen-7b/rm_baseline\
--use_qlora


CUDA_VISIBLE_DEVICES=2 python /data/haofeiy2/sotopia-rl/sotopia_rl/rm_trainer.py \
--model_name /data/models/Qwen2.5-1.5B-Instruct\
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_annotated_reward.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves_qwen/rm_direct_prompt


CUDA_VISIBLE_DEVICES=3 python /data/haofeiy2/sotopia-rl/sotopia_rl/rm_trainer.py \
--model_name /data/models/Qwen/Qwen2.5-1.5B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_annotated_reward.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves_qwen/rm_key_utterance