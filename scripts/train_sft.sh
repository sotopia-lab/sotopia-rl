CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file /data/disk0/sotopia-rl/scripts/accelerate_config_sft.yaml \
  --main_process_port 29512 \
    /data/disk0/sotopia-rl/scripts/train_sft.py \
    --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 1 \
    --val_batch_size 4 \
    --accumulation_steps 6 \
    --num_epochs 20 \
    --use_lora \
    --evaluation_steps 50 \
    --sft_data_path /data/disk0/sotopia-rl/data/sotopia_pi_round1_qwen_sft_pi_with_instruct_string.json \
    --template_path /data/disk0/sotopia-rl/evals/qwen2.5-7b.jinja \
    --checkpoint_dir /data/disk0/sotopia-rl/sft_qwen25_7b_pi_round1_qwen_sft_pi \
    --use_qlora 