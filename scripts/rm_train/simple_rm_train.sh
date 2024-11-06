CUDA_VISIBLE_DEVICES=0 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_baseline.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves/rm_baseline


CUDA_VISIBLE_DEVICES=2 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_baseline.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves/rm_direct_prompt


CUDA_VISIBLE_DEVICES=3 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 2000 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_baseline.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir /data/haofeiy2/sotopia-rl/saves/rm_key_utterance
