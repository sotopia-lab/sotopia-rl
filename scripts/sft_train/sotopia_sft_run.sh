python -m src.llmtuner.cli train examples/lora_sotopia/mistral_lora_sotopia_sft.yaml
# jbsub -mem 80g -cores 20+4 -q alt_24h -require h100 -out ./mistral_lora_sotopia_sft.log -err ./mistral_lora_sotopia_sft.err bash ./scripts/sft_train/mistral_lora_sotopia_sft.sh
