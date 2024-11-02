CUDA_VISIBLE_DEVICES=0 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_baseline.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir rm_baseline


CUDA_VISIBLE_DEVICES=1 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_direct_prompt.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir rm_direct_prompt


CUDA_VISIBLE_DEVICES=2 python /data/haofeiy2/sotopia-rl/sotopia-rl/train_rm.py \
--model_name /data/models/gemma-2-2b-it \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--evaluation_steps 200 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_reward_key_utterance.json \
--template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
--checkpoint_dir rm_key_utterance

