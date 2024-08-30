#!/bin/bash

NPROC_PER_NODE=1

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_sotopia/gemma_lora_sotopia_rloo_ds.yaml
# jbsub -mem 80g -cores 20+4 -q alt_24h -require h100 -out ./gemma_lora_sotopia_rloo_ds.log -err ./gemma_lora_sotopia_rloo_ds.err bash ./scripts/rloo_train/sotopia_rloo_run_ds_zero3.sh
