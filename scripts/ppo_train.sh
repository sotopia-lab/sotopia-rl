CUDA_VISIBLE_DEVICES=1 python /data/haofeiy2/sotopia-rl/sotopia_rl/ppo_trainer.py \
  --model_name "/data/models/gemma-2-2b-it" \
  --batch_size 1 \
  --num_epochs 3 \
  --ppo_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_utterance_ppo.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
  --ppo_epochs 4 \
  --gamma 0.99 \
  --lam 0.95
