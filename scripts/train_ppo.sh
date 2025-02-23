# CUDA_VISIBLE_DEVICES=1 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
#   --model_name "/data/models/gemma-2-2b-it" \
#   --batch_size 1 \
#   --num_epochs 3 \
#   --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_utterance_ppo.json \
#   --template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
#   --ppo_epochs 4 \
#   --gamma 0.99 \
#   --lam 0.95

CUDA_VISIBLE_DEVICES=3 poetry run python /data/haofeiy2/sotopia-rl/scripts/train_ppo.py \
  --model_name /data/haofeiy2/sotopia-rl/saves_qwen/sft/checkpoint-500 \
  --reward_model_name /data/haofeiy2/sotopia-rl/saves_qwen/rm_baseline/checkpoint-5391 \
  --batch_size 1 \
  --num_epochs 1 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_annotated_ppo.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja\
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95 \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/saves_qwen/ppo