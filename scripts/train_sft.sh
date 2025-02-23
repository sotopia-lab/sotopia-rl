# CUDA_VISIBLE_DEVICES=6 python /data/haofeiy2/sotopia-rl/scripts/train_sft.py \
# --model_name /data/models/gemma-2-2b-it \
# --learning_rate 1e-5 \
# --max_length 4096 \
# --train_batch_size 1 \
# --val_batch_size 4 \
# --accumulation_steps 4 \
# --num_epochs 3 \
# --use_lora \
# --evaluation_steps 500 \
# --reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_utterance_ppo.json \
# --template_path /data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja \
# --checkpoint_dir sft

CUDA_VISIBLE_DEVICES=2 python /data/haofeiy2/sotopia-rl/scripts/train_sft.py \
--model_name /data/models/Qwen2.5-7B-Instruct \
--learning_rate 1e-5 \
--max_length 4096 \
--train_batch_size 1 \
--val_batch_size 4 \
--accumulation_steps 4 \
--num_epochs 3 \
--use_lora \
--evaluation_steps 500 \
--reward_data_path /data/haofeiy2/sotopia-rl/data/sotopia_pi_bc_episodes_annotated_ppo.json \
--template_path /data/haofeiy2/sotopia-rl/evals/qwen2.5-7b.jinja \
--checkpoint_dir sft\
--use_qlora
