CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file /data/haofeiy2/sotopia-rl/scripts/accelerate_config_rm.yaml \
  --main_process_port 29500 \
  /data/haofeiy2/sotopia-rl/scripts/train_rm.py \
  --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
  --learning_rate 1e-5 \
  --max_length 4096 \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --accumulation_steps 8 \
  --num_epochs 30 \
  --evaluation_steps 100 \
  --reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_reward_token_length.json \
  --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
  --checkpoint_dir /data/haofeiy2/sotopia-rl/rm_token_length
