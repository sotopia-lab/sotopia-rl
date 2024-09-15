CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    -m src.llmtuner.cli train examples/lora_sotopia/llama3_lora_sotopia_rm_predict.yaml