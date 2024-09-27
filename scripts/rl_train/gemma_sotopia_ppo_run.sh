CUDA_VISIBLE_DEVICES=0 python -m src.llmtuner.cli train examples/lora_sotopia/gemma_lora_sotopia_ppo_rm_baseline.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.llmtuner.cli train examples/lora_sotopia/gemma_lora_sotopia_ppo_rm_direct.yaml
CUDA_VISIBLE_DEVICES=2 python -m src.llmtuner.cli train examples/lora_sotopia/gemma_lora_sotopia_ppo_rm_key.yaml
# jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./gemma_sotopia_ppo_run.log -err ./gemma_sotopia_ppo_run.err bash ./scripts/rl_train/gemma_sotopia_ppo_run.sh
