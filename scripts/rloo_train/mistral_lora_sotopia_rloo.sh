python -m src.llmtuner.cli train examples/lora_sotopia/mistral_lora_sotopia_rloo.yaml
# jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mistral_lora_sotopia_rloo.log -err ./mistral_lora_sotopia_rloo.err bash ./scripts/rloo_train/mistral_lora_sotopia_rloo.sh
