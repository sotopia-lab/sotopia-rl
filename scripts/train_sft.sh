CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
  --config_file /mnt/data/sotopia-rl/scripts/accelerate_config_sft.yaml \
  --main_process_port 29512 \
    /mnt/data/sotopia-rl/scripts/train_sft.py \
    --model_name /mnt/data/models/Qwen2.5-7B-Instruct \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --val_batch_size 1 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path /mnt/data/sotopia-rl/data/sft_round_1_bc_data_top_2_with_aligned_format_instruction_prompt_0509.json \
    --template_path /mnt/data/sotopia-rl/evals/qwen2.5-7b.jinja \
    --checkpoint_dir /mnt/data/sotopia-rl/sft_round_1_bc_data_top_2_with_aligned_format_instruction_prompt_weight_decay_0_0510
