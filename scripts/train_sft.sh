export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
  --config_file ./accelerate_config_sft.yaml \
  --main_process_port 29512 \
    ./train_sft.py \
    --model_name $MODEL_PATH \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --val_batch_size 1 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path ../data/sft_round_1_bc_data_top_2_with_aligned_format_instruction_prompt_0509.json \
    --template_path ../evals/qwen2.5-7b.jinja \
    --checkpoint_dir ../sft_checkpoints_qwen2.5-7b \
