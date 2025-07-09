export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file ./accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  ./train_grpo.py \
  --model_name $MODEL_PATH \
  --policy_adapter_path ../sft_checkpoints_qwen2.5-7b/best-checkpoint \
  --reward_adapter_path ../rm_checkpoints_qwen2.5-7b/best-checkpoint \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --grpo_data_path ../data/sotopia_grpo_data.json \
  --template_path ../evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --use_lora_train_grpo \
  --num_generations 16 \
  --output_dir ../grpo_checkpoints_qwen2.5-7b
