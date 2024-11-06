CUDA_VISIBLE_DEVICES=9 python rejection_sampler.py --sft_model_path "/data/models/gemma-2-2b-it" \
                             --rm_model_path "/data/models/gemma-2-2b-it" \
                             --model_name "/data/models/gemma-2-2b-it" \
                             --rejection_threshold 0.6 \
                             --max_samples 10 \
                             --prompt "Your prompt here"
