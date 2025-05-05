CUDA_VISIBLE_DEVICES=0,2,3,7 accelerate launch \
  --config_file /data/haofeiy2/sotopia-rl/scripts/accelerate_config_sft.yaml \
  --main_process_port 29512 \
    /data/haofeiy2/sotopia-rl/scripts/train_sft.py \
    --model_name /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --accumulation_steps 1 \
    --num_epochs 20 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path /data/haofeiy2/sotopia-rl/data/sft_round_1_bc_data_top_2.json \
    --template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
    --checkpoint_dir /data/haofeiy2/sotopia-rl/new_sft_round_1_bc_data_top_2 \
    --use_qlora 