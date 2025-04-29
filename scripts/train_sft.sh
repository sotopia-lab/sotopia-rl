# parameter I used for final SFT checkpoint, I use ckpt 160
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /data/disk0/sotopia-rl/scripts/accelerate_config_sft.yaml \
  --main_process_port 29112 \
    /data/disk0/sotopia-rl/scripts/train_sft.py \
    --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
    --learning_rate 1e-5 \
    --max_length 4096 \
    --train_batch_size 1 \
    --val_batch_size 4 \
    --accumulation_steps 1 \
    --num_epochs 1 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path /data/disk0/sotopia-rl/data/sft_round_1_bc_data_top_2.json \
    --template_path /data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja \
    --checkpoint_dir /data/disk0/sotopia-rl/sft_round_1_bc_data_top_2_only_1_epoch


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /data/disk0/sotopia-rl/scripts/accelerate_config_sft.yaml \
  --main_process_port 29112 \
    /data/disk0/sotopia-rl/scripts/train_sft.py \
    --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 1 \
    --val_batch_size 4 \
    --accumulation_steps 6 \
    --num_epochs 20 \
    --use_lora \
    --evaluation_steps 10 \
    --sft_data_path /data/disk0/sotopia-rl/data/sft_round_1_bc_data_top_2.json \
    --template_path /data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja \
    --checkpoint_dir /data/disk0/sotopia-rl/sft_round_1_bc_data_top_2_ckpt