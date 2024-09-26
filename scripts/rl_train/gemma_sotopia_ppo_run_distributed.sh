CUDA_VISIBLE_DEVICE=0,1,2,3 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    -m src.llmtuner.cli train examples/lora_sotopia/gemma_lora_sotopia_ppo.yaml
# jbsub -mem 80g -cores 20+4 -q alt_24h -require h100 -out ./gemma_sotopia_ppo_run_distributed.log -err ./gemma_sotopia_ppo_run_distributed.err bash ./scripts/rl_train/gemma_sotopia_ppo_run_distributed.sh
