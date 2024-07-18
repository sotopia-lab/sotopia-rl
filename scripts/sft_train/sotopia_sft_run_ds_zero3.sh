#!/bin/bash

NPROC_PER_NODE=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_sotopia/mistral_lora_sotopia_sft_ds.yaml
# jbsub -mem 80g -cores 20+4 -q alt_24h -require h100 -out ./mistral_lora_sotopia_sft_ds.log -err ./mistral_lora_sotopia_sft_ds.err bash ./scripts/sft_train/sotopia_sft_run_ds_zero3.sh