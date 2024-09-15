python -m src.llmtuner.cli train examples/lora_sotopia/gemma_lora_sotopia_rloo.yaml
# jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./gemma_lora_sotopia_rloo.log -err ./gemma_lora_sotopia_rloo.err bash ./scripts/rl_train/gemma_lora_sotopia_rloo.sh
