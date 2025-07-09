export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=5,6,7,8,9 accelerate launch \
  --config_file ./accelerate_config_rm.yaml \
  --main_process_port 29500 \
  ./scripts/train_rm.py \
  --model_name $MODEL_PATH \
  --learning_rate 1e-5 \
  --max_length 4096 \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --accumulation_steps 8 \
  --num_epochs 30 \
  --evaluation_steps 50 \
  --reward_data_path ../data/sotopia_pi_bc_episodes_rm.json \
  --template_path ../evals/qwen2.5-7b.jinja \
  --checkpoint_dir ../rm_checkpoints_qwen2.5-7b
