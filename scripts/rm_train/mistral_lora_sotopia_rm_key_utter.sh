python -m src.llmtuner.cli train examples/lora_sotopia/mistral_lora_sotopia_rm_key_utter.yaml
# jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mistral_lora_sotopia_rm_key_utter.log -err ./mistral_lora_sotopia_rm_key_utter.err bash ./scripts/rm_train/mistral_lora_sotopia_rm_key_utter.sh
